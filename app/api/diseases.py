from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_llm_service, get_retrieval_service
from app.knowledge_engine.llm import LLMService
from app.knowledge_engine.retrieval import RetrievalService
from app.ml.registry import feature_names, get_disease_config
from app.ml.service import disease_prediction_service


router = APIRouter(prefix="/diseases", tags=["Disease Prediction"])


class DiseasePredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature values keyed by disease feature name.")


class DiseasePredictionResponse(BaseModel):
    disease_id: str
    prediction: int
    risk_label: str
    risk_probability: float
    confidence_level: str
    threshold: float
    selected_model: str
    top_contributors: list[dict[str, Any]]
    artifact_version: str
    advice: str
    retrieved_context_count: int
    retrieval_validation: dict[str, Any]
    disclaimer: str


@dataclass(frozen=True)
class DiseaseRetrievalPolicy:
    disease_id: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    minimum_score: float = 0.75


DISEASE_RETRIEVAL_POLICIES: dict[str, DiseaseRetrievalPolicy] = {
    "breast_cancer": DiseaseRetrievalPolicy(
        disease_id="breast_cancer",
        required_terms=("breast", "cancer", "malignant", "mammogram", "biopsy", "tumor"),
        forbidden_terms=("abdominal ultrasound", "fasting", "gallbladder", "kidney ultrasound"),
    ),
    "heart_disease": DiseaseRetrievalPolicy(
        disease_id="heart_disease",
        required_terms=("heart", "cardiac", "coronary", "angina", "cholesterol", "blood pressure"),
        forbidden_terms=("ultrasound procedure", "mammogram", "breast cancer"),
    ),
    "diabetes": DiseaseRetrievalPolicy(
        disease_id="diabetes",
        required_terms=("diabetes", "glucose", "blood sugar", "a1c", "insulin"),
        forbidden_terms=("ultrasound procedure", "mammogram", "biopsy"),
    ),
    "parkinsons": DiseaseRetrievalPolicy(
        disease_id="parkinsons",
        required_terms=("parkinson", "tremor", "movement", "neurologic", "dopamine"),
        forbidden_terms=("ultrasound procedure", "mammogram", "blood sugar"),
    ),
}


DISEASE_QUERY_TERMS: dict[str, str] = {
    "breast_cancer": "breast cancer malignant tissue morphology mammogram biopsy clinical follow up",
    "heart_disease": "heart disease coronary risk angina blood pressure cholesterol clinical follow up",
    "diabetes": "diabetes glucose blood sugar a1c insulin monitoring clinical follow up",
    "parkinsons": "parkinson disease tremor movement neurologic evaluation clinical follow up",
}


def confidence_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability <= 0.70:
        return "Moderate"
    return "High"


def _text_matches_policy(text: str, policy: DiseaseRetrievalPolicy) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in policy.forbidden_terms):
        return False
    return any(term in lowered for term in policy.required_terms)


def validate_disease_context(
    disease_id: str,
    retrieved_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = DISEASE_RETRIEVAL_POLICIES.get(disease_id)
    if not policy:
        return retrieved_results, {
            "disease_id": disease_id,
            "minimum_score": None,
            "accepted": len(retrieved_results),
            "rejected": 0,
            "status": "no_policy_available",
        }

    accepted: list[dict[str, Any]] = []
    rejected = 0
    for item in retrieved_results:
        content = str(item.get("content", ""))
        metadata = item.get("metadata") or {}
        topic_text = " ".join([content, *(str(value) for value in metadata.values())])
        score = float(item.get("embedding_score", item.get("score", 0.0)) or 0.0)
        if score >= policy.minimum_score and _text_matches_policy(topic_text, policy):
            accepted.append(item)
        else:
            rejected += 1

    return accepted, {
        "disease_id": disease_id,
        "minimum_score": policy.minimum_score,
        "accepted": len(accepted),
        "rejected": rejected,
        "status": "passed" if accepted else "context_rejected",
    }


def format_probability(probability: float) -> str:
    percentage = min(max(probability * 100, 0.0), 99.9)
    return f"{percentage:.1f}%"


def disease_advice_template(
    disease_id: str,
    prediction_result: dict[str, Any],
    context_count: int,
) -> str:
    config = get_disease_config(disease_id)
    contributors = prediction_result.get("top_contributors", [])[:4]
    contributor_lines = "\n".join(
        f"- {item.get('feature')}: SHAP impact {item.get('contribution')}"
        for item in contributors
    ) or "- Local SHAP values were unavailable for this prediction."

    probability = format_probability(float(prediction_result["risk_probability"]))
    confidence = prediction_result.get("confidence_level", confidence_level(float(prediction_result["risk_probability"])))
    validation_note = (
        f"Disease-specific retrieval validation accepted {context_count} relevant context passage(s)."
        if context_count
        else "No disease-specific retrieved context passed validation, so this response uses the model output and fixed safety template only."
    )

    if disease_id == "breast_cancer":
        summary = (
            "The machine learning system estimated breast cancer risk from tissue morphology "
            "features in the Wisconsin Diagnostic Breast Cancer-style dataset."
        )
        interpretation = (
            "An elevated result indicates a higher likelihood of abnormal tissue characteristics "
            "associated with malignant patterns in the dataset. A low-risk result indicates the "
            "submitted morphology values were closer to benign patterns learned by the model."
        )
        next_steps = (
            "- Consult a qualified healthcare professional.\n"
            "- Discuss whether additional imaging, biopsy, or diagnostic evaluation is appropriate.\n"
            "- Continue routine clinical screening practices."
        )
    else:
        summary = f"The machine learning system estimated {config.display_name.lower()} from the submitted dataset features."
        interpretation = (
            "The result reflects statistical patterns learned from the training dataset and should be treated "
            "as decision-support information, not as a diagnosis."
        )
        next_steps = (
            "- Consult a qualified healthcare professional.\n"
            "- Discuss whether confirmatory testing or clinical evaluation is appropriate.\n"
            "- Continue routine monitoring and screening practices recommended by a clinician."
        )

    return f"""
### Prediction Summary
{summary}

- Risk label: {prediction_result["risk_label"]}
- Calibrated probability: {probability} ({float(prediction_result["risk_probability"]):.4f})
- Confidence level: {confidence}

### Key Influential Features
{contributor_lines}

### Interpretation
{interpretation}

### Important
- This result is generated for research and decision-support purposes only.
- The model does not provide a medical diagnosis.
- Clinical evaluation by a qualified healthcare professional is required for confirmation.
- {validation_note}

### Recommended Next Steps
{next_steps}
""".strip()


def generate_disease_advice(
    disease_id: str,
    features: dict[str, float],
    prediction_result: dict[str, Any],
    retrieval_service: RetrievalService,
    llm_service: LLMService,
) -> tuple[str, int]:
    config = get_disease_config(disease_id)
    rag_query = DISEASE_QUERY_TERMS.get(
        disease_id,
        f"{config.display_name} clinical follow up monitoring decision support",
    )
    retrieved_results = retrieval_service.search(rag_query, top_k=10)
    validated_results, validation = validate_disease_context(disease_id, retrieved_results)
    context = [item["content"] for item in validated_results[:5]]

    if disease_id in DISEASE_RETRIEVAL_POLICIES:
        prediction_result["retrieval_validation"] = validation
        return disease_advice_template(disease_id, prediction_result, len(context)), len(context)

    input_summary = "\n".join(f"- {name}: {features[name]}" for name in feature_names(config))
    contributor_summary = "\n".join(
        f"- {item.get('feature')}: value={item.get('value')}, contribution={item.get('contribution')}"
        for item in prediction_result.get("top_contributors", [])
    ) or "No local explanation values were available."

    context_summary = "\n\n".join(context) if context else "No retrieved medical context was available."

    prompt = f"""
You are a cautious health education assistant. A machine learning model estimated risk for {config.display_name}.

Model output:
- Risk label: {prediction_result["risk_label"]}
- Risk probability: {prediction_result["risk_probability"]}
- Decision threshold: {prediction_result["threshold"]}
- Selected model: {prediction_result["selected_model"]}

Patient inputs:
{input_summary}

Top model contributors:
{contributor_summary}

Retrieved knowledge context:
{context_summary}

Write practical, concise guidance with these sections:
1. What the result means
2. Precautions
3. What to do next
4. When to contact a doctor urgently

Important safety rules:
- Do not diagnose disease.
- Do not prescribe medication.
- Recommend confirmation with a qualified clinician and appropriate testing.
- Keep the advice understandable for a patient.
""".strip()

    stream = getattr(llm_service, "stream", None)
    if callable(stream):
        advice = "".join(stream(prompt)).strip()
    else:
        generate = getattr(llm_service, "generate", None)
        advice = generate(prompt).strip() if callable(generate) else ""

    if not advice:
        advice = (
            "The model result should be reviewed with a qualified clinician. "
            "Consider confirmatory testing, healthy lifestyle changes, monitoring if advised, "
            "and urgent medical care for severe or rapidly worsening symptoms."
        )

    prediction_result["retrieval_validation"] = validation
    return advice, len(context)


@router.get("")
def list_diseases():
    return {"diseases": disease_prediction_service.list_diseases()}


@router.get("/{disease_id}/metrics")
def disease_metrics(disease_id: str):
    try:
        return disease_prediction_service.metrics(disease_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{disease_id}/predict", response_model=DiseasePredictionResponse)
def predict_disease(
    disease_id: str,
    request: DiseasePredictionRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    try:
        config = get_disease_config(disease_id)
        expected = set(feature_names(config))
        missing = sorted(expected - set(request.features))
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required features: {', '.join(missing)}")

        invalid_choices = []
        for feature in config.feature_specs:
            if feature.choices and feature.name in request.features:
                value = request.features[feature.name]
                normalized_choices = {float(choice) for choice in feature.choices}
                if float(value) not in normalized_choices:
                    invalid_choices.append(
                        f"{feature.name}={value} must be one of {sorted(normalized_choices)}"
                    )
        if invalid_choices:
            raise HTTPException(status_code=422, detail="; ".join(invalid_choices))

        features = {name: float(request.features[name]) for name in feature_names(config)}
        prediction_result = disease_prediction_service.predict(disease_id, features)
        advice, context_count = generate_disease_advice(
            disease_id,
            features,
            prediction_result,
            retrieval_service,
            llm_service,
        )
        response_payload = dict(prediction_result)
        retrieval_validation = response_payload.pop(
            "retrieval_validation",
            {"status": "not_run", "accepted": 0, "rejected": 0},
        )
        return DiseasePredictionResponse(
            **response_payload,
            advice=advice,
            retrieved_context_count=context_count,
            retrieval_validation=retrieval_validation,
        )
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
