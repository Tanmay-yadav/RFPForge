import ast
import uuid

import requests
import streamlit as st


BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RFPForge", page_icon="RF", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f7f8;
        }

        [data-testid="stSidebar"] {
            background: #171717;
            color: #f4f4f5;
        }

        [data-testid="stSidebar"] * {
            color: #f4f4f5;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: 1px solid #3f3f46;
            background: #242424;
            color: #fafafa;
            border-radius: 8px;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #737373;
            background: #303030;
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 1.25rem;
            padding-bottom: 7rem;
        }

        .rfp-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.9rem;
            margin-bottom: 1.5rem;
        }

        .rfp-brand {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }

        .rfp-title {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .rfp-subtitle {
            color: #6b7280;
            font-size: 0.84rem;
        }

        .rfp-session {
            color: #52525b;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            font-size: 0.75rem;
            padding: 0.35rem 0.7rem;
            white-space: nowrap;
        }

        .rfp-empty {
            min-height: 52vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #111827;
        }

        .rfp-empty h1 {
            font-size: clamp(2rem, 5vw, 3.3rem);
            font-weight: 650;
            letter-spacing: 0;
            margin-bottom: 0.75rem;
        }

        .rfp-empty p {
            color: #6b7280;
            max-width: 560px;
            font-size: 1rem;
            line-height: 1.55;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            padding: 0.55rem 0;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: #1f2937;
            font-size: 0.97rem;
            line-height: 1.65;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            flex-direction: row-reverse;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        [data-testid="stMarkdownContainer"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 0.8rem 1rem;
            max-width: min(720px, 88%);
            margin-left: auto;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        [data-testid="stChatInput"] {
            max-width: 900px;
            margin: 0 auto;
        }

        [data-testid="stChatInput"] > div {
            border-color: #d4d4d8 !important;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: #9ca3af !important;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 18px;
            border-color: #d4d4d8;
            background: #ffffff;
            color: #111827 !important;
            caret-color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #737373 !important;
            -webkit-text-fill-color: #737373 !important;
            opacity: 1;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: #9ca3af;
            box-shadow: none;
        }

        .stAlert {
            border-radius: 8px;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.75rem;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            color: #111827;
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .risk-high {
            color: #b91c1c;
        }

        .risk-low {
            color: #047857;
        }

        .clinical-note {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            color: #4b5563;
            font-size: 0.88rem;
            line-height: 1.55;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
        }

        .input-guide {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem 1rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            color: #374151;
            font-size: 0.86rem;
            line-height: 1.45;
            padding: 0.95rem 1rem;
            margin: 0.75rem 0 1.1rem;
        }

        .input-guide strong {
            color: #111827;
            font-weight: 700;
        }

        .ordered-input-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.95rem 1rem 1rem;
            margin: 0 0 1rem;
        }

        .ordered-input-title {
            color: #111827;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .ordered-input-copy {
            color: #6b7280;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }

        [data-testid="stNumberInput"] label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
            color: #111827 !important;
            font-weight: 650 !important;
            opacity: 1 !important;
        }

        [data-testid="stNumberInput"] input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            background: #ffffff !important;
            border-color: #d4d4d8 !important;
            caret-color: #111827 !important;
        }

        [data-testid="stNumberInput"] input:focus {
            border-color: #9ca3af !important;
            box-shadow: 0 0 0 1px #9ca3af !important;
        }

        [data-testid="stNumberInput"] div[data-baseweb="input"],
        [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
            background: #ffffff !important;
            border-color: #d4d4d8 !important;
        }

        [data-testid="stNumberInput"] button {
            background: #ffffff !important;
            color: #111827 !important;
            border-color: #d4d4d8 !important;
        }

        [data-testid="stNumberInput"] button svg {
            fill: #111827 !important;
            color: #111827 !important;
        }

        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .input-guide {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = "idle"
    if "upload_detail" not in st.session_state:
        st.session_state.upload_detail = ""
    if "active_view" not in st.session_state:
        st.session_state.active_view = "RFP chat"
    if "diabetes_messages" not in st.session_state:
        st.session_state.diabetes_messages = []
    if "diabetes_result" not in st.session_state:
        st.session_state.diabetes_result = None
    if "selected_disease_id" not in st.session_state:
        st.session_state.selected_disease_id = "diabetes"


def disease_input_key(disease_id: str, feature_name: str) -> str:
    return f"disease_input_{disease_id}_{feature_name}"


def ordered_input_key(disease_id: str) -> str:
    return f"ordered_input_{disease_id}"


def strip_python_comments(raw_text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in raw_text.splitlines())


def parse_ordered_feature_values(raw_text: str, features: list[dict]) -> list[float]:
    text = strip_python_comments(raw_text).strip()
    if not text:
        raise ValueError("Paste an ordered feature list first.")

    if "=" in text:
        text = text.split("=", 1)[1].strip()

    values = ast.literal_eval(text)
    while isinstance(values, (list, tuple)) and len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]

    if not isinstance(values, (list, tuple)):
        raise ValueError("Input must be a list of numeric values.")

    if len(values) != len(features):
        raise ValueError(f"Expected {len(features)} values, received {len(values)}.")

    return [float(value) for value in values]


def apply_ordered_feature_values(disease_id: str, features: list[dict], values: list[float]) -> None:
    for feature, value in zip(features, values):
        st.session_state[disease_input_key(disease_id, feature["name"])] = value


def breast_cancer_high_risk_values() -> list[float]:
    return [
        20.5,
        25.2,
        135.0,
        1300.0,
        0.13,
        0.22,
        0.30,
        0.15,
        0.25,
        0.09,
        1.2,
        1.8,
        8.5,
        150.0,
        0.009,
        0.05,
        0.06,
        0.02,
        0.03,
        0.006,
        28.0,
        35.0,
        190.0,
        2400.0,
        0.18,
        0.50,
        0.70,
        0.30,
        0.45,
        0.12,
    ]


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def new_disease_chat() -> None:
    st.session_state.diabetes_messages = []
    st.session_state.diabetes_result = None


def post_file(uploaded_file) -> None:
    st.session_state.upload_status = "uploading"
    st.session_state.upload_detail = "Uploading and indexing document..."

    try:
        response = requests.post(
            f"{BASE_URL}/knowledge/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state.upload_status = "done"
        st.session_state.upload_detail = (
            f"Processed {payload.get('chunks', 0)} chunks from {uploaded_file.name}."
        )
    except requests.RequestException as exc:
        st.session_state.upload_status = "error"
        st.session_state.upload_detail = f"Upload failed: {exc}"


def ingest_knowledge() -> None:
    with st.sidebar.status("Ingesting knowledge...", expanded=False) as status:
        try:
            response = requests.post(f"{BASE_URL}/knowledge/ingest", timeout=180)
            response.raise_for_status()
            payload = response.json()
            status.update(
                label=f"Ingested {payload.get('chunks_created', 0)} chunks.",
                state="complete",
            )
        except requests.RequestException as exc:
            status.update(label=f"Ingestion failed: {exc}", state="error")


def stream_reply(prompt: str) -> str:
    full_response = ""

    with requests.post(
        f"{BASE_URL}/chat/stream",
        json={"session_id": st.session_state.session_id, "message": prompt},
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
            if chunk:
                full_response += chunk
                yield full_response


def get_diseases() -> list[dict]:
    response = requests.get(f"{BASE_URL}/diseases", timeout=60)
    response.raise_for_status()
    return response.json().get("diseases", [])


def predict_disease(disease_id: str, payload: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}/diseases/{disease_id}/predict",
        json={"features": payload},
        timeout=240,
    )
    response.raise_for_status()
    return response.json()


def render_topbar(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f"""
        <div class="rfp-topbar">
            <div class="rfp-brand">
                <div class="rfp-title">{title}</div>
                <div class="rfp-subtitle">{subtitle}</div>
            </div>
            <div class="rfp-session">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disease_result(result: dict) -> None:
    prediction_text = "Risk detected" if result["prediction"] == 1 else "Low risk"
    risk_class = "risk-high" if result["prediction"] == 1 else "risk-low"
    probability = f'{min(result["risk_probability"] * 100, 99.9):.1f}%'
    threshold = f'{result["threshold"] * 100:.2f}%'
    confidence = result.get("confidence_level", "Unknown")

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {risk_class}">{prediction_text}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk probability</div>
                <div class="metric-value">{probability}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">{threshold}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    contributors = result.get("top_contributors", [])
    if contributors:
        st.markdown("#### Top model contributors")
        st.dataframe(
            [
                {
                    "Feature": item.get("feature"),
                    "Value": item.get("value"),
                    "SHAP Impact": item.get("contribution"),
                }
                for item in contributors
            ],
            use_container_width=True,
            hide_index=True,
        )

    validation = result.get("retrieval_validation", {})
    st.caption(
        f'RAG context passages: {result.get("retrieved_context_count", 0)} '
        f'| Retrieval validation: {validation.get("status", "unknown")}'
    )
    st.caption(f'Selected model: {result.get("selected_model", "model")}')
    st.caption(result["disclaimer"])


def render_rfp_chat() -> None:
    render_topbar(
        "RFPForge Chat",
        "Ask questions against your uploaded RFP knowledge base.",
        f"Session {st.session_state.session_id[:8]}",
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="rfp-empty">
                <h1>How can I help with this RFP?</h1>
                <p>Upload a document from the sidebar, then ask for summaries, compliance answers, proposal language, or evidence-backed responses.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Message RFPForge")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                assistant_response = ""
                for partial_response in stream_reply(prompt):
                    assistant_response = partial_response
                    response_placeholder.markdown(assistant_response)

                if not assistant_response:
                    assistant_response = "I did not receive a response from the backend."
                    response_placeholder.markdown(assistant_response)
            except requests.RequestException as exc:
                assistant_response = f"Backend request failed: {exc}"
                response_placeholder.error(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )


def render_diabetes_chat() -> None:
    render_topbar(
        "Multi-Disease Prediction RAG Chat",
        "Run trained disease screening models and receive RAG-generated precautions and next steps.",
        "ML Pipeline + RAG",
    )

    st.markdown(
        """
        <div class="clinical-note">
            This tool is for educational screening support only. It does not diagnose disease or replace medical testing by a qualified clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        diseases = get_diseases()
    except requests.RequestException as exc:
        st.error(f"Could not load disease registry: {exc}")
        return

    if not diseases:
        st.warning("No disease configurations are available.")
        return

    disease_ids = [item["disease_id"] for item in diseases]
    if st.session_state.selected_disease_id not in disease_ids:
        st.session_state.selected_disease_id = disease_ids[0]

    selected_index = disease_ids.index(st.session_state.selected_disease_id)
    selected_name = st.selectbox(
        "Disease model",
        disease_ids,
        index=selected_index,
        format_func=lambda disease_id: next(
            item["display_name"] for item in diseases if item["disease_id"] == disease_id
        ),
    )
    st.session_state.selected_disease_id = selected_name
    selected_disease = next(item for item in diseases if item["disease_id"] == selected_name)

    if selected_disease.get("artifact_type") == "legacy":
        st.info("Using legacy diabetes artifacts. Run the training script to generate publication-grade pipeline artifacts.")
    elif not selected_disease.get("trained"):
        st.warning("This disease has no trained model artifact yet.")

    feature_help = "".join(
        f"<div><strong>{feature['label']}:</strong> {feature.get('help') or 'Clinical measurement from the source dataset.'}</div>"
        for feature in selected_disease["features"]
    )
    st.markdown(f'<div class="input-guide">{feature_help}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ordered-input-panel">
            <div class="ordered-input-title">Ordered feature input</div>
            <div class="ordered-input-copy">
                Paste a Python-style assignment like <code>breast_cancer_high_risk = [[...]]</code> or a plain list like <code>[[1, 2, 3]]</code>.
                Click <strong>Apply ordered input</strong> to populate the manual form fields below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        ordered_text = st.text_area(
            "Python or JSON list",
            key=ordered_input_key(selected_name),
            height=150,
            placeholder=f"{selected_name}_sample = [[1.0, 2.0, 3.0]]",
        )
        button_columns = st.columns(3)
        with button_columns[0]:
            if st.button("Apply ordered input", use_container_width=True):
                try:
                    values = parse_ordered_feature_values(ordered_text, selected_disease["features"])
                    apply_ordered_feature_values(selected_name, selected_disease["features"], values)
                    st.success("Applied ordered values to the form.")
                except (SyntaxError, ValueError, TypeError) as exc:
                    st.error(f"Could not apply ordered input: {exc}")
        with button_columns[1]:
            if selected_name == "breast_cancer" and st.button(
                "Load breast cancer high-risk sample",
                use_container_width=True,
            ):
                apply_ordered_feature_values(
                    selected_name,
                    selected_disease["features"],
                    breast_cancer_high_risk_values(),
                )
                st.success("Loaded breast cancer high-risk sample.")
        with button_columns[2]:
            if st.button("Clear ordered input", use_container_width=True):
                st.session_state[ordered_input_key(selected_name)] = ""
                st.success("Cleared ordered input text.")

    with st.form("disease_prediction_form"):
        payload = {}
        columns = st.columns(4)
        for index, feature in enumerate(selected_disease["features"]):
            with columns[index % 4]:
                widget_key = disease_input_key(selected_name, feature["name"])
                choices = feature.get("choices") or []
                choice_labels = feature.get("choice_labels") or {}
                if choices:
                    def format_choice(value, labels=choice_labels):
                        label = labels.get(str(value), labels.get(value, ""))
                        return f"{value:g} - {label}" if label else f"{value:g}"

                    default_value = float(feature.get("default", choices[0]))
                    numeric_choices = [float(value) for value in choices]
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = default_value
                    selected_index = (
                        numeric_choices.index(float(st.session_state[widget_key]))
                        if float(st.session_state[widget_key]) in numeric_choices
                        else 0
                    )
                    payload[feature["name"]] = st.selectbox(
                        feature["label"],
                        numeric_choices,
                        index=selected_index,
                        format_func=format_choice,
                        help=feature.get("help", ""),
                        key=widget_key,
                    )
                else:
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = float(feature.get("default", 0.0))
                    input_kwargs = {
                        "label": feature["label"],
                        "min_value": float(feature.get("minimum", 0.0)),
                        "value": float(st.session_state[widget_key]),
                        "step": float(feature.get("step", 1.0)),
                        "help": feature.get("help", ""),
                        "key": widget_key,
                    }
                    if float(feature.get("step", 1.0)) < 0.01:
                        input_kwargs["format"] = "%.3f"
                    payload[feature["name"]] = st.number_input(**input_kwargs)

        submitted = st.form_submit_button("Predict and generate advice", use_container_width=True)

    if submitted:
        user_summary = "\n".join(f"- {key}: {value}" for key, value in payload.items())
        st.session_state.diabetes_messages.append(
            {
                "role": "user",
                "content": (
                    f"Predict {selected_disease['display_name']} risk using these values:\n\n"
                    f"{user_summary}"
                ),
            }
        )

        with st.spinner("Running disease prediction and generating RAG advice..."):
            try:
                result = predict_disease(selected_name, payload)
                st.session_state.diabetes_result = result
                st.session_state.diabetes_messages.append(
                    {"role": "assistant", "content": result["advice"]}
                )
            except requests.RequestException as exc:
                st.session_state.diabetes_result = None
                st.error(f"Prediction failed: {exc}")

    for message in st.session_state.diabetes_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.diabetes_result:
        render_disease_result(st.session_state.diabetes_result)


init_state()

with st.sidebar:
    st.markdown("### RFPForge")
    st.session_state.active_view = st.radio(
        "Workspace",
        ["RFP chat", "Disease prediction"],
        index=["RFP chat", "Disease prediction"].index(st.session_state.active_view),
    )

    if st.session_state.active_view == "RFP chat":
        st.button("New chat", on_click=new_chat, use_container_width=True)
    else:
        st.button("Clear disease chat", on_click=new_disease_chat, use_container_width=True)

    st.divider()
    st.markdown("#### Knowledge")
    uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx", "doc", "txt"])

    if uploaded_file and st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        post_file(uploaded_file)

    if st.session_state.upload_status == "done":
        st.success(st.session_state.upload_detail)
    elif st.session_state.upload_status == "error":
        st.error(st.session_state.upload_detail)
    elif st.session_state.upload_status == "uploading":
        st.info(st.session_state.upload_detail)

    if st.button("Ingest folder", use_container_width=True):
        ingest_knowledge()

    st.divider()
    st.caption(f"Backend: {BASE_URL}")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

if st.session_state.active_view == "RFP chat":
    render_rfp_chat()
else:
    render_diabetes_chat()
