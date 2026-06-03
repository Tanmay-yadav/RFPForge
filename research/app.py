from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.src.inference import predict_sample
from research.src.utils import DISEASES, REPORTS_DIR, dataset_path, model_path


st.set_page_config(page_title="Research Disease Prediction", page_icon="ML", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f8fafc; color: #111827; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #f9fafb; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    .research-note {
        border: 1px solid #d1d5db;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_defaults(disease_id: str) -> dict:
    spec = DISEASES[disease_id]
    frame = pd.read_csv(dataset_path(disease_id)).drop(columns=[spec.target])
    row = frame.median(numeric_only=True).to_dict()
    for column in frame.columns:
        if column not in row:
            mode = frame[column].dropna().mode()
            row[column] = mode.iloc[0] if not mode.empty else ""
    return row


st.sidebar.title("Research Workspace")
view = st.sidebar.radio("Navigation", ["Prediction", "Results", "IEEE Notes"])

if view == "Prediction":
    st.title("Multi-Disease Prediction Research UI")
    st.markdown(
        '<div class="research-note">Educational screening support only. This system is for research demonstration and is not a clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )
    disease_id = st.selectbox("Disease", list(DISEASES), format_func=lambda key: DISEASES[key].display_name)
    spec = DISEASES[disease_id]

    if not model_path(disease_id).exists():
        st.warning("Model artifact not found. Train this disease before inference.")
        st.stop()

    defaults = load_defaults(disease_id)
    frame = pd.read_csv(dataset_path(disease_id)).drop(columns=[spec.target])
    payload = {}
    with st.form("prediction_form"):
        columns = st.columns(4)
        for index, column in enumerate(frame.columns):
            with columns[index % 4]:
                help_text = spec.feature_help.get(column, "")
                numeric = pd.to_numeric(frame[column], errors="coerce")
                if numeric.notna().mean() > 0.8:
                    payload[column] = st.number_input(
                        column,
                        value=float(defaults.get(column, 0.0)),
                        step=0.1,
                        help=help_text,
                    )
                else:
                    options = sorted(str(value) for value in frame[column].dropna().unique())
                    default_value = str(defaults.get(column, options[0] if options else ""))
                    index_value = options.index(default_value) if default_value in options else 0
                    payload[column] = st.selectbox(column, options, index=index_value, help=help_text)
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        result = predict_sample(disease_id, payload)
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Risk detected" if result["prediction"] else "Low risk")
        col2.metric("Probability", f"{result['risk_probability'] * 100:.2f}%")
        col3.metric("Model", result["model"])
        if result["top_contributors"]:
            st.subheader("Top SHAP Contributors")
            st.dataframe(result["top_contributors"], use_container_width=True, hide_index=True)

elif view == "Results":
    st.title("Research Results")
    for filename in ["dataset_summary.csv", "results_table.csv", "metrics_summary.csv"]:
        path = REPORTS_DIR / filename
        st.subheader(filename)
        if path.exists():
            st.dataframe(pd.read_csv(path), use_container_width=True)
        else:
            st.info("Not generated yet.")

else:
    st.title("IEEE Notes")
    path = REPORTS_DIR / "ieee_notes.md"
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.info("Generate IEEE notes after training.")

