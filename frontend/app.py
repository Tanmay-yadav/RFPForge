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


def predict_diabetes(payload: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}/diabetes/predict",
        json=payload,
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


def render_diabetes_result(result: dict) -> None:
    prediction_text = "Risk detected" if result["prediction"] == 1 else "Low risk"
    risk_class = "risk-high" if result["prediction"] == 1 else "risk-low"
    probability = f'{result["risk_probability"] * 100:.2f}%'
    threshold = f'{result["threshold"] * 100:.2f}%'

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
                <div class="metric-label">Model threshold</div>
                <div class="metric-value">{threshold}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RAG context</div>
                <div class="metric-value">{result["retrieved_context_count"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        "Disease Prediction RAG Chat",
        "Run the diabetes XGBoost model and receive RAG-generated precautions and next steps.",
        "XGBoost + RAG",
    )

    st.markdown(
        """
        <div class="clinical-note">
            This tool is for educational screening support only. It does not diagnose diabetes or replace medical testing by a qualified clinician.
        </div>
        <div class="input-guide">
            <div><strong>Pregnancies:</strong> number of times pregnant.</div>
            <div><strong>Glucose:</strong> plasma glucose concentration value.</div>
            <div><strong>BloodPressure:</strong> diastolic blood pressure in mm Hg.</div>
            <div><strong>SkinThickness:</strong> triceps skinfold thickness in mm.</div>
            <div><strong>Insulin:</strong> 2-hour serum insulin value.</div>
            <div><strong>BMI:</strong> body mass index, weight-to-height ratio.</div>
            <div><strong>DiabetesPedigreeFunction:</strong> family-history diabetes risk score.</div>
            <div><strong>Age:</strong> age in years.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("diabetes_prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0.0,
                value=6.0,
                step=1.0,
                help="Number of times pregnant.",
            )
            skin_thickness = st.number_input(
                "SkinThickness",
                min_value=0.0,
                value=35.0,
                step=1.0,
                help="Triceps skinfold thickness measured in millimeters.",
            )
        with col2:
            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                value=148.0,
                step=1.0,
                help="Plasma glucose concentration value.",
            )
            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="2-hour serum insulin value.",
            )
        with col3:
            blood_pressure = st.number_input(
                "BloodPressure",
                min_value=0.0,
                value=72.0,
                step=1.0,
                help="Diastolic blood pressure measured in mm Hg.",
            )
            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                value=33.6,
                step=0.1,
                help="Body mass index.",
            )
        with col4:
            pedigree = st.number_input(
                "DiabetesPedigreeFunction",
                min_value=0.0,
                value=0.627,
                step=0.001,
                format="%.3f",
                help="A diabetes family-history risk score from the dataset.",
            )
            age = st.number_input(
                "Age",
                min_value=0.0,
                value=50.0,
                step=1.0,
                help="Age in years.",
            )

        submitted = st.form_submit_button("Predict and generate advice", use_container_width=True)

    if submitted:
        payload = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }

        user_summary = "\n".join(f"- {key}: {value}" for key, value in payload.items())
        st.session_state.diabetes_messages.append(
            {"role": "user", "content": f"Predict diabetes risk using these values:\n\n{user_summary}"}
        )

        with st.spinner("Running XGBoost prediction and generating RAG advice..."):
            try:
                result = predict_diabetes(payload)
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
        render_diabetes_result(st.session_state.diabetes_result)


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
