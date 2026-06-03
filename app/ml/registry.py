from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    minimum: float = 0.0
    default: float = 0.0
    step: float = 1.0
    unit: str = ""
    help: str = ""
    choices: tuple[float | str, ...] = ()
    choice_labels: dict[float | str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DiseaseConfig:
    disease_id: str
    display_name: str
    dataset_path: Path
    target_column: str
    positive_labels: tuple[int | str, ...]
    feature_specs: tuple[FeatureSpec, ...]
    provenance: dict[str, Any]
    invalid_zero_columns: tuple[str, ...] = ()
    artifact_dir: Path = field(default_factory=Path)


DIABETES_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("Pregnancies", "Pregnancies", default=6.0, step=1.0, help="Number of times pregnant."),
    FeatureSpec("Glucose", "Glucose", default=148.0, step=1.0, unit="mg/dL", help="Plasma glucose concentration."),
    FeatureSpec("BloodPressure", "Blood pressure", default=72.0, step=1.0, unit="mm Hg", help="Diastolic blood pressure."),
    FeatureSpec("SkinThickness", "Skin thickness", default=35.0, step=1.0, unit="mm", help="Triceps skinfold thickness."),
    FeatureSpec("Insulin", "Insulin", default=0.0, step=1.0, help="2-hour serum insulin value."),
    FeatureSpec("BMI", "BMI", default=33.6, step=0.1, help="Body mass index."),
    FeatureSpec(
        "DiabetesPedigreeFunction",
        "Diabetes pedigree function",
        default=0.627,
        step=0.001,
        help="Family-history diabetes risk score from the dataset.",
    ),
    FeatureSpec("Age", "Age", default=50.0, step=1.0, unit="years", help="Age in years."),
)


HEART_DISEASE_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("age", "Age", default=54.0, step=1.0, unit="years", help="Age in years."),
    FeatureSpec(
        "sex",
        "Sex",
        default=1.0,
        step=1.0,
        help="Biological sex encoded by the UCI Cleveland dataset.",
        choices=(0.0, 1.0),
        choice_labels={0.0: "female", 1.0: "male"},
    ),
    FeatureSpec(
        "cp",
        "Chest pain type",
        default=4.0,
        step=1.0,
        help="Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic.",
        choices=(1.0, 2.0, 3.0, 4.0),
        choice_labels={1.0: "typical angina", 2.0: "atypical angina", 3.0: "non-anginal pain", 4.0: "asymptomatic"},
    ),
    FeatureSpec("trestbps", "Resting blood pressure", default=130.0, step=1.0, unit="mm Hg", help="Resting blood pressure in mm Hg at hospital admission."),
    FeatureSpec("chol", "Serum cholesterol", default=246.0, step=1.0, unit="mg/dL", help="Serum cholesterol in mg/dL."),
    FeatureSpec(
        "fbs",
        "Fasting blood sugar > 120",
        default=0.0,
        step=1.0,
        help="Whether fasting blood sugar is greater than 120 mg/dL.",
        choices=(0.0, 1.0),
        choice_labels={0.0: "false", 1.0: "true"},
    ),
    FeatureSpec(
        "restecg",
        "Resting ECG",
        default=0.0,
        step=1.0,
        help="Resting electrocardiographic result: 0 normal, 1 ST-T abnormality, 2 probable/definite left ventricular hypertrophy.",
        choices=(0.0, 1.0, 2.0),
        choice_labels={0.0: "normal", 1.0: "ST-T abnormality", 2.0: "left ventricular hypertrophy"},
    ),
    FeatureSpec("thalach", "Maximum heart rate", default=150.0, step=1.0, help="Maximum heart rate achieved during exercise testing."),
    FeatureSpec(
        "exang",
        "Exercise induced angina",
        default=0.0,
        step=1.0,
        help="Whether exercise induced angina was present.",
        choices=(0.0, 1.0),
        choice_labels={0.0: "no", 1.0: "yes"},
    ),
    FeatureSpec("oldpeak", "ST depression", default=1.0, step=0.1, help="ST depression induced by exercise."),
    FeatureSpec(
        "slope",
        "ST slope",
        default=2.0,
        step=1.0,
        help="Slope of peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping.",
        choices=(1.0, 2.0, 3.0),
        choice_labels={1.0: "upsloping", 2.0: "flat", 3.0: "downsloping"},
    ),
    FeatureSpec(
        "ca",
        "Major vessels",
        default=0.0,
        step=1.0,
        help="Number of major vessels colored by fluoroscopy.",
        choices=(0.0, 1.0, 2.0, 3.0),
    ),
    FeatureSpec(
        "thal",
        "Thalassemia",
        default=3.0,
        step=1.0,
        help="Thalassemia result in the UCI Cleveland encoding: 3 normal, 6 fixed defect, 7 reversible defect.",
        choices=(3.0, 6.0, 7.0),
        choice_labels={3.0: "normal", 6.0: "fixed defect", 7.0: "reversible defect"},
    ),
)


BREAST_CANCER_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("mean_radius", "Mean radius", default=14.0, step=0.1),
    FeatureSpec("mean_texture", "Mean texture", default=19.0, step=0.1),
    FeatureSpec("mean_perimeter", "Mean perimeter", default=92.0, step=0.1),
    FeatureSpec("mean_area", "Mean area", default=655.0, step=1.0),
    FeatureSpec("mean_smoothness", "Mean smoothness", default=0.1, step=0.001),
    FeatureSpec("mean_compactness", "Mean compactness", default=0.1, step=0.001),
    FeatureSpec("mean_concavity", "Mean concavity", default=0.09, step=0.001),
    FeatureSpec("mean_concave_points", "Mean concave points", default=0.05, step=0.001),
    FeatureSpec("mean_symmetry", "Mean symmetry", default=0.18, step=0.001),
    FeatureSpec("mean_fractal_dimension", "Mean fractal dimension", default=0.06, step=0.001),
    FeatureSpec("radius_error", "Radius error", default=0.4, step=0.01),
    FeatureSpec("texture_error", "Texture error", default=1.2, step=0.01),
    FeatureSpec("perimeter_error", "Perimeter error", default=2.9, step=0.01),
    FeatureSpec("area_error", "Area error", default=40.0, step=0.1),
    FeatureSpec("smoothness_error", "Smoothness error", default=0.007, step=0.001),
    FeatureSpec("compactness_error", "Compactness error", default=0.025, step=0.001),
    FeatureSpec("concavity_error", "Concavity error", default=0.03, step=0.001),
    FeatureSpec("concave_points_error", "Concave points error", default=0.012, step=0.001),
    FeatureSpec("symmetry_error", "Symmetry error", default=0.02, step=0.001),
    FeatureSpec("fractal_dimension_error", "Fractal dimension error", default=0.004, step=0.001),
    FeatureSpec("worst_radius", "Worst radius", default=16.0, step=0.1),
    FeatureSpec("worst_texture", "Worst texture", default=25.0, step=0.1),
    FeatureSpec("worst_perimeter", "Worst perimeter", default=107.0, step=0.1),
    FeatureSpec("worst_area", "Worst area", default=880.0, step=1.0),
    FeatureSpec("worst_smoothness", "Worst smoothness", default=0.13, step=0.001),
    FeatureSpec("worst_compactness", "Worst compactness", default=0.25, step=0.001),
    FeatureSpec("worst_concavity", "Worst concavity", default=0.27, step=0.001),
    FeatureSpec("worst_concave_points", "Worst concave points", default=0.11, step=0.001),
    FeatureSpec("worst_symmetry", "Worst symmetry", default=0.29, step=0.001),
    FeatureSpec("worst_fractal_dimension", "Worst fractal dimension", default=0.08, step=0.001),
)


PARKINSONS_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("MDVP_Fo_Hz", "Average vocal frequency", default=154.0, step=0.1, unit="Hz"),
    FeatureSpec("MDVP_Fhi_Hz", "Maximum vocal frequency", default=197.0, step=0.1, unit="Hz"),
    FeatureSpec("MDVP_Flo_Hz", "Minimum vocal frequency", default=116.0, step=0.1, unit="Hz"),
    FeatureSpec("MDVP_Jitter_percent", "Jitter percent", default=0.006, step=0.001),
    FeatureSpec("MDVP_Jitter_Abs", "Absolute jitter", default=0.00004, step=0.00001),
    FeatureSpec("MDVP_RAP", "RAP jitter", default=0.003, step=0.001),
    FeatureSpec("MDVP_PPQ", "PPQ jitter", default=0.003, step=0.001),
    FeatureSpec("Jitter_DDP", "DDP jitter", default=0.009, step=0.001),
    FeatureSpec("MDVP_Shimmer", "Shimmer", default=0.03, step=0.001),
    FeatureSpec("MDVP_Shimmer_dB", "Shimmer dB", default=0.28, step=0.01),
    FeatureSpec("Shimmer_APQ3", "APQ3 shimmer", default=0.016, step=0.001),
    FeatureSpec("Shimmer_APQ5", "APQ5 shimmer", default=0.017, step=0.001),
    FeatureSpec("MDVP_APQ", "APQ shimmer", default=0.024, step=0.001),
    FeatureSpec("Shimmer_DDA", "DDA shimmer", default=0.048, step=0.001),
    FeatureSpec("NHR", "Noise-to-harmonics ratio", default=0.025, step=0.001),
    FeatureSpec("HNR", "Harmonics-to-noise ratio", default=21.0, step=0.1),
    FeatureSpec("RPDE", "RPDE", default=0.5, step=0.001),
    FeatureSpec("DFA", "DFA", default=0.72, step=0.001),
    FeatureSpec("spread1", "Spread 1", default=-5.7, minimum=-10.0, step=0.01),
    FeatureSpec("spread2", "Spread 2", default=0.22, step=0.001),
    FeatureSpec("D2", "D2", default=2.4, step=0.01),
    FeatureSpec("PPE", "PPE", default=0.2, step=0.001),
)


DISEASE_REGISTRY: dict[str, DiseaseConfig] = {
    "diabetes": DiseaseConfig(
        disease_id="diabetes",
        display_name="Diabetes Risk",
        dataset_path=Path("ML-MODEL/Data/diabetes.csv"),
        target_column="Outcome",
        positive_labels=(1,),
        feature_specs=DIABETES_FEATURES,
        invalid_zero_columns=("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"),
        artifact_dir=Path("ML-MODEL/artifacts/diabetes"),
        provenance={
            "source": "Local approved diabetes CSV in ML-MODEL/Data/diabetes.csv",
            "notes": (
                "Common Pima-style diabetes screening dataset. Confirm final citation, "
                "license, and inclusion criteria before IEEE submission."
            ),
        },
    ),
    "heart_disease": DiseaseConfig(
        disease_id="heart_disease",
        display_name="Heart Disease Risk",
        dataset_path=Path("ML-MODEL/Data/heart_disease_uci.csv"),
        target_column="target",
        positive_labels=(1, 2, 3, 4),
        feature_specs=HEART_DISEASE_FEATURES,
        artifact_dir=Path("ML-MODEL/artifacts/heart_disease"),
        provenance={
            "source": "UCI Heart Disease processed Cleveland dataset",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "notes": "Target values 1-4 are treated as disease-present; 0 is absence.",
        },
    ),
    "breast_cancer": DiseaseConfig(
        disease_id="breast_cancer",
        display_name="Breast Cancer Risk",
        dataset_path=Path("ML-MODEL/Data/breast_cancer_wdbc.csv"),
        target_column="target",
        positive_labels=(0,),
        feature_specs=BREAST_CANCER_FEATURES,
        artifact_dir=Path("ML-MODEL/artifacts/breast_cancer"),
        provenance={
            "source": "Wisconsin Diagnostic Breast Cancer dataset packaged with scikit-learn",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
            "notes": "scikit-learn target 0 is malignant and is treated as the positive/risk class.",
        },
    ),
    "parkinsons": DiseaseConfig(
        disease_id="parkinsons",
        display_name="Parkinson's Disease Risk",
        dataset_path=Path("ML-MODEL/Data/parkinsons_uci.csv"),
        target_column="status",
        positive_labels=(1,),
        feature_specs=PARKINSONS_FEATURES,
        artifact_dir=Path("ML-MODEL/artifacts/parkinsons"),
        provenance={
            "source": "UCI Parkinsons dataset",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            "notes": "status=1 is treated as Parkinson's disease present; status=0 is healthy.",
        },
    ),
}


def get_disease_config(disease_id: str) -> DiseaseConfig:
    try:
        return DISEASE_REGISTRY[disease_id]
    except KeyError as exc:
        known = ", ".join(sorted(DISEASE_REGISTRY))
        raise ValueError(f"Unknown disease_id '{disease_id}'. Known diseases: {known}") from exc


def feature_names(config: DiseaseConfig) -> list[str]:
    return [feature.name for feature in config.feature_specs]
