from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_llm_service, get_retrieval_service
from app.knowledge_engine.llm import LLMService
from app.knowledge_engine.retrieval import RetrievalService
from app.services.diabetes_prediction import FEATURE_NAMES, diabetes_prediction_service


router = APIRouter(prefix="/diabetes", tags=["Diabetes Prediction"])


class DiabetesPredictionRequest(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)


class DiabetesPredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    risk_probability: float
    threshold: float
    advice: str
    retrieved_context_count: int
    disclaimer: str


def _build_advice_prompt(
    request: DiabetesPredictionRequest,
    prediction_result: dict,
    context: list[str],
) -> str:
    input_summary = "\n".join(
        f"- {name}: {getattr(request, name)}" for name in FEATURE_NAMES
    )
    context_summary = "\n\n".join(context) if context else "No retrieved medical context was available."

    return f"""
You are a cautious health education assistant. A machine learning model predicted diabetes risk from the following clinical-style inputs.

Model output:
- Risk label: {prediction_result["risk_label"]}
- Diabetes risk probability: {prediction_result["risk_probability"]}
- Decision threshold: {prediction_result["threshold"]}
- Predicted class: {prediction_result["prediction"]} where 1 means diabetes risk detected and 0 means low diabetes risk

Patient inputs:
{input_summary}

Retrieved knowledge context:
{context_summary}

Write practical, concise guidance with these sections:
1. What the result means
2. Precautions
3. What to do next
4. When to contact a doctor urgently

Important safety rules:
- Do not diagnose diabetes.
- Do not prescribe medication.
- Recommend confirmation with a qualified clinician and appropriate lab testing.
- Keep the advice understandable for a patient.
""".strip()


@router.post("/predict", response_model=DiabetesPredictionResponse)
def predict_diabetes(
    request: DiabetesPredictionRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    features = request.model_dump()
    prediction_result = diabetes_prediction_service.predict(features)

    rag_query = (
        "diabetes risk precautions lifestyle diet exercise monitoring glucose "
        "when to consult doctor"
    )
    retrieved_results = retrieval_service.search(rag_query, top_k=5)
    context = [item["content"] for item in retrieved_results]

    prompt = _build_advice_prompt(request, prediction_result, context)
    advice = "".join(llm_service.stream(prompt)).strip()

    if not advice:
        advice = (
            "The model result should be reviewed with a qualified clinician. "
            "Consider confirmatory lab testing, healthy diet changes, regular physical activity, "
            "weight management if relevant, and glucose monitoring if advised by a doctor."
        )

    return DiabetesPredictionResponse(
        **prediction_result,
        advice=advice,
        retrieved_context_count=len(context),
        disclaimer=(
            "This is educational support from an ML model and RAG assistant, not a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        ),
    )
