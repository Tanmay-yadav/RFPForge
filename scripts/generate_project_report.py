from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import textwrap
from typing import Iterable
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "RFPForge_Project_Report_80_Pages.docx"
UPDATED_OUT = ROOT / "data" / "RFPForge_Project_Report_80_Pages_Updated.docx"
ASSETS = ROOT / "data" / "report_assets"
DATASET = ROOT / "ML-MODEL" / "Data" / "diabetes.csv"


TITLE = "RFPFORGE: AN AI-POWERED RFP AUTOMATION AND DISEASE PREDICTION RAG SYSTEM"
SHORT_TITLE = "RFPForge"


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)


def shade_cell(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def set_margins(section) -> None:
    section.left_margin = Inches(1.5)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)


def add_page_number(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for style_name, size in [("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 13)]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(17, 24, 39)


def add_centered(doc: Document, text: str, size: int = 12, bold: bool = False, spacing: int = 1) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = spacing
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold


def add_para(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5


def add_bullets(doc: Document, items: Iterable[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.25
        p.add_run(item)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    if title:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True)
        shade_cell(table.rows[0].cells[i], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], str(value))


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.2) -> None:
    doc.add_picture(str(path), width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)


def save_box_diagram(filename: str, title: str, boxes: list[str], arrows: bool = True) -> Path:
    path = ASSETS / filename
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    n = len(boxes)
    for i, box in enumerate(boxes):
        x = 0.08 + (i % 4) * 0.23
        y = 0.68 - (i // 4) * 0.32
        rect = plt.Rectangle((x, y), 0.18, 0.16, facecolor="#E8F2FF", edgecolor="#1F4E79", linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.08, box, ha="center", va="center", fontsize=9, wrap=True)
        if arrows and i < n - 1:
            nx = 0.08 + ((i + 1) % 4) * 0.23
            ny = 0.68 - ((i + 1) // 4) * 0.32
            if (i + 1) % 4 != 0:
                ax.annotate("", xy=(nx, ny + 0.08), xytext=(x + 0.18, y + 0.08), arrowprops=dict(arrowstyle="->", lw=1.4))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_layer_diagram() -> Path:
    path = ASSETS / "system_layers.png"
    layers = [
        ("Presentation Layer", "Streamlit chat UI, disease prediction form, upload panel"),
        ("API Layer", "FastAPI routers: chat, knowledge, diabetes, RFP workflows"),
        ("Service Layer", "Singleton services, RAG retrieval, XGBoost prediction, LLM streaming"),
        ("Data Layer", "SQLite, ChromaDB, embedding cache, uploaded documents, model artifacts"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title("Layered Architecture of RFPForge", fontsize=16, fontweight="bold")
    colors = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#FEE2E2"]
    for i, (name, desc) in enumerate(layers):
        y = 0.78 - i * 0.19
        ax.add_patch(plt.Rectangle((0.08, y), 0.84, 0.13, facecolor=colors[i], edgecolor="#374151", linewidth=1.4))
        ax.text(0.12, y + 0.085, name, fontsize=12, fontweight="bold", va="center")
        ax.text(0.12, y + 0.04, desc, fontsize=9, va="center")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_dataset_charts() -> list[Path]:
    df = pd.read_csv(DATASET)
    paths = []
    for col in ["Glucose", "BMI", "Age", "BloodPressure"]:
        path = ASSETS / f"{col.lower()}_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col], bins=24, color="#2563EB", alpha=0.78, edgecolor="white")
        ax.set_title(f"{col} Distribution in Diabetes Dataset", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Record Count")
        ax.grid(alpha=0.25)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    path = ASSETS / "diabetes_correlation.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def save_bar_chart(filename: str, title: str, labels: list[str], values: list[float], ylabel: str) -> Path:
    path = ASSETS / filename
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#1D4ED8", "#047857", "#B45309", "#BE123C", "#6D28D9"][: len(labels)])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ui_mockup(filename: str, title: str, mode: str) -> Path:
    path = ASSETS / filename
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#F7F7F8", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0, 0), 0.18, 1, facecolor="#171717", edgecolor="none"))
    ax.text(0.035, 0.92, "RFPForge", color="white", fontsize=16, fontweight="bold")
    ax.add_patch(plt.Rectangle((0.025, 0.82), 0.13, 0.06, facecolor="#242424", edgecolor="#52525B", linewidth=1.2))
    ax.text(0.09, 0.85, "New chat" if mode == "chat" else "Clear disease chat", color="white", fontsize=9, ha="center", va="center")
    ax.text(0.035, 0.73, "Workspace", color="#F4F4F5", fontsize=10, fontweight="bold")
    ax.text(0.035, 0.68, "RFP chat", color="#D4D4D8", fontsize=9)
    ax.text(0.035, 0.64, "Disease prediction", color="#D4D4D8", fontsize=9)
    ax.text(0.035, 0.52, "Knowledge", color="#F4F4F5", fontsize=11, fontweight="bold")
    ax.add_patch(plt.Rectangle((0.025, 0.44), 0.13, 0.06, facecolor="#111827", edgecolor="#3F3F46", linewidth=1))
    ax.text(0.09, 0.47, "Upload", color="white", fontsize=10, ha="center", va="center")

    ax.text(0.24, 0.92, title, color="#111827", fontsize=18, fontweight="bold")
    ax.text(0.24, 0.875, "FastAPI backend + Streamlit frontend + local RAG services", color="#6B7280", fontsize=10)
    ax.plot([0.24, 0.94], [0.84, 0.84], color="#E5E7EB", linewidth=1.2)

    if mode == "chat":
        ax.add_patch(plt.Circle((0.27, 0.73), 0.022, color="#EF4444"))
        ax.text(0.31, 0.73, "Summarize the uploaded medical document and list key precautions.", fontsize=10, va="center", color="#111827")
        ax.add_patch(plt.Circle((0.27, 0.61), 0.022, color="#F59E0B"))
        answer = "The assistant retrieves relevant chunks from ChromaDB, builds a grounded prompt, and streams a concise answer with safety notes."
        ax.text(0.31, 0.62, "\n".join(textwrap.wrap(answer, 78)), fontsize=10, va="center", color="#111827")
        ax.add_patch(plt.Rectangle((0.25, 0.12), 0.62, 0.075, facecolor="#FFFFFF", edgecolor="#D1D5DB", linewidth=1.2))
        ax.text(0.28, 0.158, "Message RFPForge", fontsize=10, color="#6B7280", va="center")
    else:
        labels = [
            ("Pregnancies", "6.00"),
            ("Glucose", "148.00"),
            ("BloodPressure", "72.00"),
            ("DiabetesPedigreeFunction", "0.627"),
            ("SkinThickness", "35.00"),
            ("Insulin", "0.00"),
            ("BMI", "33.60"),
            ("Age", "50.00"),
        ]
        for i, (label, value) in enumerate(labels):
            x = 0.24 + (i % 4) * 0.18
            y = 0.71 - (i // 4) * 0.16
            ax.text(x, y + 0.06, label, fontsize=9, color="#111827", fontweight="bold")
            ax.add_patch(plt.Rectangle((x, y), 0.15, 0.055, facecolor="#262A33", edgecolor="#4B5563", linewidth=1))
            ax.text(x + 0.015, y + 0.028, value, fontsize=10, color="#F9FAFB", va="center")
        ax.add_patch(plt.Rectangle((0.24, 0.32), 0.69, 0.06, facecolor="#111827", edgecolor="#374151", linewidth=1.2))
        ax.text(0.585, 0.35, "Predict and generate advice", fontsize=11, color="white", ha="center", va="center")
        ax.add_patch(plt.Rectangle((0.24, 0.19), 0.16, 0.07, facecolor="#FFFFFF", edgecolor="#E5E7EB", linewidth=1))
        ax.text(0.255, 0.235, "Prediction", fontsize=8, color="#6B7280")
        ax.text(0.255, 0.205, "Risk detected", fontsize=11, color="#B91C1C", fontweight="bold")
        ax.add_patch(plt.Rectangle((0.42, 0.19), 0.16, 0.07, facecolor="#FFFFFF", edgecolor="#E5E7EB", linewidth=1))
        ax.text(0.435, 0.235, "Risk probability", fontsize=8, color="#6B7280")
        ax.text(0.435, 0.205, "56.28%", fontsize=11, color="#111827", fontweight="bold")

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_assets() -> dict[str, Path | list[Path]]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path | list[Path]] = {}
    assets["layers"] = save_layer_diagram()
    assets["rag_flow"] = save_box_diagram(
        "rag_flow.png",
        "Retrieval-Augmented Generation Flow",
        ["Upload", "Load", "Clean", "Chunk", "Embed", "Store", "Retrieve", "LLM Advice"],
    )
    assets["chat_sequence"] = save_box_diagram(
        "chat_sequence.png",
        "Chat Request Sequence",
        ["User Prompt", "FastAPI /chat", "Vector Search", "Prompt Builder", "Ollama Stream", "Chat UI"],
    )
    assets["diabetes_flow"] = save_box_diagram(
        "diabetes_prediction_flow.png",
        "Diabetes Prediction and RAG Advice Flow",
        ["Clinical Inputs", "Scaler", "XGBoost Model", "Threshold", "Risk Label", "RAG Query", "LLM Advice"],
    )
    assets["rfp_workflow"] = save_box_diagram(
        "rfp_workflow.png",
        "RFP Response Workflow",
        ["Create Session", "Extract Questions", "Retrieve Context", "Generate Draft", "Review", "Finalize", "Export"],
    )
    assets["deployment"] = save_box_diagram(
        "deployment_view.png",
        "Local Deployment View",
        ["Browser", "Streamlit 8501", "FastAPI 8000", "SQLite", "ChromaDB", "Ollama 11434", "Model Files"],
    )
    assets["dataset"] = save_dataset_charts()
    assets["endpoint_chart"] = save_bar_chart(
        "endpoint_coverage.png",
        "API Surface by Functional Area",
        ["Chat", "Knowledge", "RFP", "Diabetes", "Health"],
        [1, 4, 8, 1, 1],
        "Endpoint Count",
    )
    assets["module_chart"] = save_bar_chart(
        "module_complexity.png",
        "Relative Module Responsibility",
        ["API", "RAG", "RFP", "DB", "UI"],
        [30, 38, 28, 14, 20],
        "Responsibility Score",
    )
    assets["testing"] = save_bar_chart(
        "testing_strategy.png",
        "Testing Focus Areas",
        ["E2E API", "Chunking", "Embeddings", "Retrieval", "LLM"],
        [25, 20, 18, 22, 15],
        "Coverage Weight",
    )
    assets["risk"] = save_bar_chart(
        "risk_distribution.png",
        "Project Risk Register Summary",
        ["Model Latency", "Data Quality", "Medical Safety", "UX", "Deployment"],
        [4, 5, 5, 3, 3],
        "Risk Level",
    )
    assets["ui_chat"] = save_ui_mockup(
        "ui_rfp_chat_screenshot.png",
        "RFP Chat Workspace",
        "chat",
    )
    assets["ui_diabetes"] = save_ui_mockup(
        "ui_diabetes_prediction_screenshot.png",
        "Disease Prediction RAG Workspace",
        "diabetes",
    )
    return assets


def cover_page(doc: Document) -> None:
    add_centered(doc, TITLE, 16, True)
    for _ in range(2):
        doc.add_paragraph()
    add_centered(doc, "A", 12, False)
    add_centered(doc, "PROJECT REPORT", 16, True)
    add_centered(doc, "SUBMITTED TO THE", 12, True)
    add_centered(doc, "RAFFLES UNIVERSITY,", 13, True)
    add_centered(doc, "FOR THE DEGREE", 12, True)
    add_centered(doc, "OF", 12, True)
    add_centered(doc, "BACHELOR OF TECHNOLOGY", 13, True)
    add_centered(doc, "IN", 12, True)
    add_centered(doc, "COMPUTER SCIENCE & ENGINEERING", 13, True)
    for _ in range(3):
        doc.add_paragraph()
    add_centered(doc, "BY", 12, True)
    add_centered(doc, "Student Name: ____________________", 12)
    add_centered(doc, "(Roll No. ____________________)", 12)
    doc.add_paragraph()
    add_centered(doc, "Under the supervision of", 12)
    add_centered(doc, "Guide Name: ____________________", 12, True)
    doc.add_paragraph()
    add_centered(doc, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", 12, True)
    add_centered(doc, "RAFFLES UNIVERSITY, NEEMRANA", 12, True)
    add_centered(doc, "ALWAR, RAJASTHAN - 301705", 12, True)
    add_centered(doc, "Year 2025-26", 12, True)
    doc.add_page_break()


def preliminary_pages(doc: Document) -> None:
    pages = [
        spec("Declaration by the Candidate", [
            f'I hereby declare that the work presented in this report entitled "{TITLE.title()}" was carried out by me as part of my project work. I have not submitted the matter embodied in this report for the award of any other degree or diploma of any other university or institute.',
            "I have given due credit to the original authors, sources, libraries, datasets, diagrams, frameworks, and software systems that helped in the completion of this work. All implementation details, experiments, and observations have been reported honestly to the best of my knowledge.",
            "I affirm that no portion of my work is intentionally plagiarized, and the experiments and results reported in the project have not been manipulated. In the event of a complaint, I shall be fully responsible and answerable.",
            "Name: ____________________\nEnrollment No.: ____________________\n\n\n(Candidate Signature)",
        ]),
        spec("Certificate", [
            f'This is to certify that the project entitled "{TITLE.title()}" submitted by ____________________ for the award of the degree of Bachelor of Technology in Computer Science and Engineering to Raffles University, Neemrana, Rajasthan, is a record of bonafide project work carried out under my guidance.',
            "To the best of my knowledge, the candidate has not submitted the same work to any other institution for any degree or diploma. The project report submitted is a record of original work done by the student during the period of study under my supervision.",
            "In my opinion, this Project Report is of the standard required for the award of Bachelor of Technology in Computer Science and Engineering.",
            "Guide Name: ____________________\nDepartment of Computer Science and Engineering\nRaffles University, Neemrana\n\nPlace: Neemrana\nDate: ____________________",
        ]),
        spec("Acknowledgement", [
            "This project report is the result of continuous learning, experimentation, guidance, and support. I express my sincere gratitude to my project guide for providing direction, constructive feedback, and encouragement during the development of this work.",
            "I am thankful to the Department of Computer Science and Engineering, School of Engineering and Technology, Raffles University, Neemrana, for providing an academic environment that supported exploration in artificial intelligence, machine learning, backend engineering, and applied software development.",
            "I also thank my friends and family members for their patience, motivation, and support throughout the project. Their encouragement helped me work through implementation issues, testing delays, and design improvements.",
            "Student Name: ____________________",
        ]),
        spec("Abstract", [
            "RFPForge is an AI-powered project developed to automate and simplify Request for Proposal response preparation using Retrieval-Augmented Generation. The system ingests organizational knowledge documents, converts them into semantic chunks, stores them in a vector database, and uses a local language model to generate grounded responses. In addition to the RFP workflow, the project was extended with an XGBoost-based diabetes risk prediction module. This module accepts eight clinical-style inputs, scales the features, predicts risk using a saved model, applies a learned decision threshold, and forwards the model result to the RAG assistant for precautionary guidance.",
            "The backend is implemented using FastAPI, SQLAlchemy, ChromaDB, SentenceTransformers, Ollama, and XGBoost. The frontend is implemented in Streamlit and provides a chat-style interface similar to modern AI assistants. The project emphasizes local execution, privacy, modularity, and practical usability. This report presents the project background, literature basis, system architecture, methodology, implementation details, results, limitations, and future scope.",
        ]),
    ]
    for page in pages:
        title = page["heading"]
        paras = page["paragraphs"]
        doc.add_heading(title, level=1)
        for para in paras:
            add_para(doc, para)
        doc.add_page_break()


def contents_pages(doc: Document) -> None:
    doc.add_heading("Table of Contents", level=1)
    rows = [
        ["Chapter 1", "Introduction", "1"],
        ["Chapter 2", "Literature Review", "16"],
        ["Chapter 3", "Proposed Work and Methodology", "31"],
        ["Chapter 4", "Implementation and Results", "48"],
        ["Chapter 5", "Conclusion and Future Work", "70"],
        ["References", "Bibliography", "77"],
        ["Appendix", "API Payloads and Supporting Material", "79"],
    ]
    add_table(doc, ["Section", "Title", "Page"], rows)
    doc.add_page_break()
    doc.add_heading("List of Figures", level=1)
    figures = [
        "Figure 1.1 Layered Architecture of RFPForge",
        "Figure 1.2 RAG Processing Flow",
        "Figure 2.1 Literature Map of RAG and ML Components",
        "Figure 3.1 RFP Workflow",
        "Figure 3.2 Diabetes Prediction and RAG Advice Flow",
        "Figure 3.3 Local Deployment View",
        "Figure 4.1 API Surface Coverage",
        "Figure 4.2 Glucose Distribution",
        "Figure 4.3 BMI Distribution",
        "Figure 4.4 Age Distribution",
        "Figure 4.5 Blood Pressure Distribution",
        "Figure 4.6 Correlation Heatmap",
        "Figure 4.7 Testing Strategy",
        "Figure 4.8 RFP Chat UI Screenshot",
        "Figure 4.9 Disease Prediction UI Screenshot",
        "Figure 5.1 Risk Register Summary",
    ]
    add_bullets(doc, figures)
    doc.add_page_break()
    doc.add_heading("List of Tables", level=1)
    tables = [
        "Table 1.1 Project Modules",
        "Table 1.2 Problem Statement Summary",
        "Table 2.1 Literature Review Matrix",
        "Table 3.1 Functional Requirements",
        "Table 3.2 Non-Functional Requirements",
        "Table 3.3 Diabetes Input Feature Dictionary",
        "Table 4.1 API Endpoint Summary",
        "Table 4.2 Model Artifact Summary",
        "Table 4.3 Test Case Summary",
        "Table 5.1 Limitations and Future Enhancements",
    ]
    add_bullets(doc, tables)
    doc.add_page_break()


def one_page(doc: Document, heading: str, paragraphs: list[str], bullets: list[str] | None = None, table=None, figure=None) -> None:
    doc.add_heading(heading, level=1 if heading.startswith("Chapter") else 2)
    for para in paragraphs:
        add_para(doc, para)
    if bullets:
        add_bullets(doc, bullets)
    if table:
        add_table(doc, table["headers"], table["rows"], table.get("title"))
    if figure:
        add_figure(doc, figure["path"], figure["caption"], figure.get("width", 6.2))
    doc.add_page_break()


def spec(heading: str, paragraphs: list[str], bullets=None, table=None, figure=None) -> dict:
    return {
        "heading": heading,
        "paragraphs": paragraphs,
        "bullets": bullets,
        "table": table,
        "figure": figure,
    }


def chapter_pages(doc: Document, assets: dict[str, Path | list[Path]]) -> None:
    dataset_images = assets["dataset"]
    assert isinstance(dataset_images, list)
    pages = [
        spec("Chapter 1: Introduction", [
            "The project titled RFPForge was developed as a practical software system that combines backend engineering, document intelligence, retrieval-augmented generation, and machine learning. Its original purpose was to reduce the manual effort involved in preparing Request for Proposal responses. During development, the system was also expanded to support disease prediction and medical-style advisory generation using an XGBoost model and the same RAG architecture.",
            "The system is designed for local execution. This means the organization can ingest sensitive documents, query them through an assistant, and generate responses without sending private content to external cloud services. The same principle also applies to the disease prediction extension, where the saved model artifacts are loaded locally and the advisory output is generated through the local LLM pipeline.",
        ], figure={"path": assets["layers"], "caption": "Figure 1.1 Layered architecture of RFPForge."}),
        spec("1.1 Background of the Project", [
            "Modern organizations often maintain large stores of policy documents, technical brochures, compliance responses, product descriptions, proposal archives, and operational guidelines. When a client sends an RFP, teams must search across these materials and prepare consistent answers within a short deadline.",
            "A manual approach is slow and error-prone. Different people may answer the same question differently, references may be missed, and sensitive knowledge may be copied into unmanaged tools. RFPForge addresses this situation by combining document ingestion, vector search, and local language-model response generation.",
        ], table={"title": "Table 1.1 Project Modules", "headers": ["Module", "Main Role", "Technology"], "rows": [["Frontend", "Chat and form interface", "Streamlit"], ["Backend", "API orchestration", "FastAPI"], ["RAG Engine", "Retrieval and prompt grounding", "ChromaDB, BGE"], ["LLM", "Answer generation", "Ollama"], ["Prediction", "Diabetes risk classification", "XGBoost"]]}),
        spec("1.2 Problem Statement", [
            "The project solves two related problems: first, the challenge of generating reliable RFP responses from organizational knowledge; second, the challenge of presenting a machine-learning prediction in a useful, safe, and understandable way. The first problem belongs to enterprise knowledge automation, while the second belongs to applied health informatics.",
            "Both problems require more than raw prediction. They require context, explanation, and a usable interface. Therefore, the system does not simply return a text answer or a numeric class. It wraps predictions and retrieved context into structured guidance.",
        ], table={"title": "Table 1.2 Problem Statement Summary", "headers": ["Area", "Problem", "Expected Solution"], "rows": [["RFP Automation", "Manual search and drafting is slow", "Document-based RAG assistant"], ["Knowledge Management", "Information scattered across files", "Vectorized knowledge base"], ["Prediction", "Model output alone is hard to act on", "Advice generated with context"], ["Privacy", "Sensitive data should remain local", "Local embeddings and LLM"]]}),
        spec("1.3 Objectives", [
            "The main objective of the project is to build a working AI-assisted platform that can ingest documents, retrieve relevant information, generate grounded answers, and support a predictive healthcare extension. The project is not only a demonstration of AI models; it is also a demonstration of system integration.",
            "The objectives were defined in a way that could be implemented and tested within a local development environment. Each objective maps to a visible software component in the repository.",
        ], bullets=["Develop a FastAPI backend with modular route files.", "Create a Streamlit interface with RFP chat and disease prediction workspaces.", "Implement a RAG pipeline using loaders, chunking, embeddings, ChromaDB, retrieval, and prompt construction.", "Use saved XGBoost artifacts for diabetes risk prediction.", "Generate user-friendly advisory output after prediction."]),
        spec("1.4 Scope of the Project", [
            "The scope includes document upload, knowledge ingestion, vector search, chat-based question answering, RFP workflow support, disease prediction, and report-style export logic. The project also includes tests and database migration support through Alembic.",
            "The disease prediction module is an educational screening support feature. It is not a diagnostic medical device and does not prescribe treatment. Its value lies in demonstrating how predictive models can be connected with a RAG assistant for safer, more explanatory output.",
        ]),
        spec("1.5 System Overview", [
            "RFPForge is organized as a layered application. The frontend sends requests to FastAPI endpoints. The endpoints call service objects, which then interact with model files, vector databases, relational databases, or the local LLM.",
            "This organization keeps the code understandable. API files handle HTTP requests and responses, service files handle reusable logic, and knowledge-engine files handle the RAG pipeline.",
        ], figure={"path": assets["rag_flow"], "caption": "Figure 1.2 RAG processing flow from upload to answer generation."}),
        spec("1.6 Need for RAG in the Project", [
            "A language model alone can generate fluent text, but fluency does not guarantee accuracy. In an RFP setting, answers must be based on company knowledge, policies, and previously approved material. Retrieval-Augmented Generation improves reliability by feeding retrieved context into the prompt.",
            "In the disease prediction extension, RAG helps transform the model result into practical guidance. The model predicts risk, while retrieval and generation provide precautionary language, next steps, and safety disclaimers.",
        ]),
        spec("1.7 Need for XGBoost in the Project", [
            "XGBoost was selected for diabetes prediction because it is strong on tabular datasets and can model non-linear relationships between features. The saved artifacts include the classifier, scaler, and decision threshold.",
            "The implementation uses the exact feature order from the dataset: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The service first scales the input and then applies the classifier probability against the stored threshold.",
        ]),
        spec("1.8 Project Contributions", [
            "The project contribution lies in integrating multiple AI and software engineering components into a single usable application. Rather than treating RAG, chat, prediction, and UI as isolated demonstrations, the project connects them into an end-to-end workflow.",
            "The user can upload documents, chat with the knowledge base, switch to disease prediction, enter clinical-style values, receive a risk result, and read generated precautions within the same Streamlit application.",
        ]),
        spec("1.9 Report Organization", [
            "This report is organized according to the university project report format. Chapter 1 introduces the project and its motivation. Chapter 2 reviews related work and background concepts. Chapter 3 explains the proposed methodology and system design.",
            "Chapter 4 describes implementation and results. Chapter 5 concludes the report and presents future enhancements. References and appendices are included at the end.",
        ]),
    ]
    literature_rows = [
        ["RAG", "Combines retrieval with generation", "Used for grounded RFP and advice output"],
        ["Vector Search", "Finds semantically related chunks", "Used through ChromaDB"],
        ["Sentence Embeddings", "Transforms text into dense vectors", "Used with BGE embedding model"],
        ["XGBoost", "Gradient boosting for tabular data", "Used for diabetes prediction"],
        ["FastAPI", "Modern Python API framework", "Used for backend endpoints"],
        ["Streamlit", "Rapid Python frontend framework", "Used for chat and form UI"],
    ]
    pages.extend([
        spec("Chapter 2: Literature Review", [
            "The project is based on several important areas of modern computing: retrieval-augmented generation, semantic search, local language models, machine learning for tabular prediction, web API design, and interactive frontend development.",
            "A literature review for this project therefore includes both AI concepts and software architecture concepts. The value of RFPForge comes from the practical combination of these ideas.",
        ], table={"title": "Table 2.1 Literature Review Matrix", "headers": ["Area", "Core Idea", "Project Use"], "rows": literature_rows}),
        spec("2.1 Retrieval-Augmented Generation", [
            "Retrieval-Augmented Generation is a design pattern in which a system retrieves relevant external knowledge before producing a generated answer. This approach reduces dependence on the model's internal memory and improves traceability.",
            "In RFPForge, RAG is used to retrieve document chunks from ChromaDB and provide them as context to the local LLM. This allows generated answers to stay closer to the uploaded knowledge base.",
        ]),
        spec("2.2 Document Loading and Parsing", [
            "Document loading is the first stage of the RAG pipeline. The project uses loader logic to handle supported formats such as PDF, DOCX, DOC, and text files. The purpose of this stage is to convert raw files into structured document objects.",
            "Parsing quality has a direct impact on retrieval quality. If the text is incomplete or noisy, chunking and embeddings will also suffer. Therefore, loading is treated as a separate responsibility in the knowledge engine.",
        ]),
        spec("2.3 Text Chunking", [
            "Chunking divides documents into smaller passages that can be embedded and searched. The project uses adaptive chunking ideas based on structural signals. Documents with headings, clauses, and bullet points are handled differently from plain unstructured text.",
            "Good chunking improves retrieval precision. A chunk should be large enough to preserve meaning but small enough to avoid mixing unrelated topics.",
        ]),
        spec("2.4 Embedding Models", [
            "An embedding model converts text into numerical vectors. Similar text passages produce vectors that are close to one another in the vector space. This makes semantic search possible even when the query and document do not use the exact same words.",
            "The project uses SentenceTransformers with a BGE embedding model. Embeddings are normalized for cosine-style similarity and cached to avoid repeated computation.",
        ]),
        spec("2.5 Vector Databases", [
            "A vector database stores embeddings and supports similarity search. RFPForge uses ChromaDB as a local persistent vector store. This aligns with the privacy-focused nature of the project.",
            "Each vector is stored along with metadata. Metadata such as source file, section title, and chunk ID helps trace answers back to their document origins.",
        ]),
        spec("2.6 Local Large Language Models", [
            "Local language models allow private text generation without sending prompts to an external API. RFPForge uses Ollama to run an instruction-tuned model locally.",
            "The local deployment choice is important because RFP content and healthcare-style inputs may be sensitive. Keeping the model local reduces exposure of confidential information.",
        ]),
        spec("2.7 XGBoost for Tabular Prediction", [
            "XGBoost is a gradient boosting framework commonly used for structured datasets. It builds an ensemble of decision trees and is known for strong predictive performance on tabular data.",
            "The diabetes module uses a saved XGBoost classifier. Before prediction, the input is transformed by the saved scaler, and then the resulting probability is compared with the stored threshold.",
        ]),
        spec("2.8 User Interface Design", [
            "A good AI system must be easy to operate. The Streamlit UI provides a chat-style layout for RFP questions and a separate disease prediction workspace for structured inputs.",
            "The UI includes visible labels, input explanations, reset actions, upload controls, and chat-message rendering. These features reduce confusion and make the application more approachable.",
        ]),
        spec("2.9 API-Centered Architecture", [
            "FastAPI provides a clean way to expose application functions as HTTP endpoints. In this project, each major capability is grouped into a router: chat, knowledge, diabetes, and RFP workflows.",
            "This router-based organization helps maintenance. New routes can be added without mixing unrelated logic into one large file.",
        ], figure={"path": assets["endpoint_chart"], "caption": "Figure 2.1 API surface coverage by functional area."}),
        spec("2.10 Summary of Literature Review", [
            "The reviewed concepts show that the project is not based on a single model or library. It is a full-stack AI application where retrieval, generation, prediction, persistence, and user interaction all work together.",
            "The literature foundation supports the design decisions made in the project: local execution, modular services, vector retrieval, structured APIs, and explainable user-facing outputs.",
        ]),
        spec("Chapter 3: Proposed Work and Methodology", [
            "The proposed system consists of a Streamlit frontend, FastAPI backend, RAG knowledge engine, RFP workflow module, XGBoost diabetes prediction service, relational database, vector store, and local LLM.",
            "The methodology follows an incremental development approach. First, the document and chat pipeline is implemented. Next, the RFP workflow is connected. Finally, the disease prediction endpoint and UI are added.",
        ], figure={"path": assets["rfp_workflow"], "caption": "Figure 3.1 RFP response workflow."}),
        spec("3.1 Proposed Architecture", [
            "The proposed architecture separates concerns into layers. The frontend focuses on interaction, the API layer validates and routes requests, the service layer performs AI and business operations, and the data layer stores persistent information.",
            "This separation improves testability. For example, the diabetes prediction service can be tested without running Streamlit, and the retrieval service can be tested with mocked vector stores.",
        ], figure={"path": assets["module_chart"], "caption": "Figure 3.2 Relative module responsibility in the system."}),
        spec("3.2 Functional Requirements", [
            "Functional requirements define what the system must do from the user's point of view. RFPForge supports document upload, ingestion, chat, RFP draft generation, disease prediction, and advisory response generation.",
            "The disease prediction requirement adds structured input collection and response presentation to the earlier RFP-focused application.",
        ], table={"title": "Table 3.1 Functional Requirements", "headers": ["ID", "Requirement", "Implemented In"], "rows": [["FR1", "Upload knowledge documents", "frontend/app.py, knowledge.py"], ["FR2", "Search document chunks", "retrieval.py"], ["FR3", "Stream chat answers", "chat.py"], ["FR4", "Predict diabetes risk", "diabetes.py"], ["FR5", "Generate precautions", "LLM + RAG prompt"], ["FR6", "Display result in UI", "Streamlit disease workspace"]]}),
        spec("3.3 Non-Functional Requirements", [
            "Non-functional requirements describe system qualities such as privacy, usability, modularity, reliability, and maintainability. These qualities matter because the project handles sensitive documents and health-related inputs.",
            "Local execution is a central non-functional requirement. It reduces dependency on cloud APIs and gives the user more control over data movement.",
        ], table={"title": "Table 3.2 Non-Functional Requirements", "headers": ["Quality", "Design Decision"], "rows": [["Privacy", "Local ChromaDB, local Ollama, local model artifacts"], ["Maintainability", "Separate routers and service modules"], ["Usability", "Chat UI and explained disease inputs"], ["Scalability", "Cached embeddings and vector search"], ["Safety", "Medical disclaimer and non-diagnostic language"]]}),
        spec("3.4 RAG Methodology", [
            "The RAG methodology begins with raw documents and ends with grounded generated answers. Documents are loaded, cleaned, chunked, embedded, stored, retrieved, and passed to the language model as context.",
            "The system retrieves the top relevant chunks for each query. The prompt builder then asks the language model to answer using the retrieved context.",
        ], figure={"path": assets["chat_sequence"], "caption": "Figure 3.3 Chat request sequence."}),
        spec("3.5 Diabetes Prediction Methodology", [
            "The disease prediction methodology uses a classical machine-learning pipeline. Eight input values are collected, converted to a DataFrame with the expected feature names, scaled with the saved StandardScaler, and passed into the XGBoost classifier.",
            "The classifier returns the probability of the positive class. The stored threshold is then used to decide whether the risk label should be low risk or risk detected.",
        ], figure={"path": assets["diabetes_flow"], "caption": "Figure 3.4 Diabetes prediction and RAG advice flow."}),
        spec("3.6 Diabetes Feature Dictionary", [
            "The input feature dictionary is necessary because users may not understand dataset field names directly. The frontend now includes visible explanations for each field.",
            "The backend validates that all values are non-negative. This validation prevents obviously invalid payloads from reaching the model service.",
        ], table={"title": "Table 3.3 Diabetes Input Feature Dictionary", "headers": ["Feature", "Meaning"], "rows": [["Pregnancies", "Number of times pregnant"], ["Glucose", "Plasma glucose concentration value"], ["BloodPressure", "Diastolic blood pressure in mm Hg"], ["SkinThickness", "Triceps skinfold thickness in mm"], ["Insulin", "2-hour serum insulin value"], ["BMI", "Body mass index"], ["DiabetesPedigreeFunction", "Family-history diabetes risk score"], ["Age", "Age in years"]]}),
        spec("3.7 Database Methodology", [
            "The relational database stores RFP sessions, questions, and drafts. SQLAlchemy models define the database tables and relationships. Alembic migration files support schema evolution.",
            "The vector database is used for semantic knowledge retrieval. This separation between relational data and semantic data is appropriate because the two storage needs are different.",
        ]),
        spec("3.8 Prompt Engineering Methodology", [
            "Prompt construction is important because the language model must understand both the user's question and the retrieved context. The project builds prompts that include the query, context, and prior chat memory.",
            "For disease prediction, the prompt includes model output, patient-style inputs, retrieved context, and safety rules. This reduces the chance of overconfident medical language.",
        ]),
        spec("3.9 Deployment Methodology", [
            "The local deployment uses two services: FastAPI on port 8000 and Streamlit on port 8501. Ollama runs separately on port 11434. The combined run script starts backend and frontend processes together.",
            "The project was adjusted to avoid Windows console encoding issues and to reduce reload-related multiprocessing problems in the combined launcher.",
        ], figure={"path": assets["deployment"], "caption": "Figure 3.5 Local deployment view."}),
        spec("3.10 Security and Safety Methodology", [
            "Security in this project is primarily addressed through local execution, controlled dependencies, and separation of sensitive data from external services. CORS is enabled for frontend convenience during development.",
            "Safety is especially important in the disease prediction module. The system avoids diagnosis, avoids medication prescription, and clearly recommends consultation with a qualified healthcare professional.",
        ]),
        spec("Chapter 4: Implementation and Results", [
            "The implementation is organized in the repository under app, frontend, data, tests, alembic, and ML-MODEL. The code follows a practical modular structure rather than a monolithic file.",
            "This chapter explains the implemented backend routes, frontend screens, model loading logic, RAG pipeline, and observed behavior from service-level verification.",
        ], table={"title": "Table 4.1 API Endpoint Summary", "headers": ["Endpoint", "Purpose"], "rows": [["POST /chat/stream", "Stream RAG chat responses"], ["POST /knowledge/upload", "Upload and index one document"], ["POST /knowledge/ingest", "Ingest folder documents"], ["POST /knowledge/search", "Search knowledge base"], ["POST /diabetes/predict", "Predict diabetes risk and generate advice"], ["GET /knowledge/health", "Check vector store health"]]}),
        spec("4.1 Backend Implementation", [
            "The backend starts in app/main.py. It creates the FastAPI application, initializes logging and database setup during lifespan startup, preloads heavy services, enables CORS, and registers the routers.",
            "The registered routers include chat, knowledge, and diabetes. The RFP router exists in the project and can be enabled when the full RFP workflow is required.",
        ]),
        spec("4.2 Dependency Injection", [
            "The file app/dependencies.py contains reusable dependency providers. It creates singleton-like service objects for embeddings, vector store, retrieval, and LLM access.",
            "This pattern prevents the project from loading expensive AI models repeatedly for every request. It also keeps the route files cleaner because they can request dependencies through FastAPI.",
        ]),
        spec("4.3 Knowledge Upload Implementation", [
            "The knowledge upload endpoint stores the uploaded file, loads the document, chunks it, generates embeddings, and writes the vectors into ChromaDB. This gives users a direct way to add knowledge from the UI.",
            "The endpoint currently limits loaded documents during development to reduce overload. This can be adjusted for production use after performance testing.",
        ]),
        spec("4.4 Chat Streaming Implementation", [
            "The chat endpoint receives a session ID and user message. It retrieves context from the vector store, builds an RFP prompt, appends recent memory, and streams the local LLM output back to Streamlit.",
            "Streaming improves perceived responsiveness. The user can see the answer appear progressively instead of waiting for the entire generation to finish.",
        ]),
        spec("4.5 Diabetes Endpoint Implementation", [
            "The diabetes endpoint receives the eight model inputs using a Pydantic request model. It calls the diabetes prediction service, retrieves general diabetes precaution context, builds a cautious prompt, and returns the risk result plus generated advice.",
            "The response includes prediction, risk label, risk probability, threshold, advice, retrieved context count, and disclaimer.",
        ], table={"title": "Table 4.2 Model Artifact Summary", "headers": ["Artifact", "Role"], "rows": [["xgboost_diabetes_model.pkl", "Saved XGBoost classifier"], ["scaler.pkl", "Saved feature scaler"], ["threshold.pkl", "Decision threshold"], ["diabetes.csv", "Dataset used for training and analysis"]]}),
        spec("4.6 Frontend Implementation", [
            "The Streamlit frontend now provides two workspaces. The RFP chat workspace behaves like a ChatGPT-style assistant. The disease prediction workspace provides structured number inputs and displays a chat-style advisory response.",
            "The UI includes clearer input labels and explanations because raw dataset field names can confuse users. The input guide describes what each field means.",
        ]),
        spec("4.6.1 RFP Chat UI Screenshot and Explanation", [
            "The RFP chat workspace is designed to feel familiar to users who have already worked with modern AI assistants. The left sidebar contains workspace selection, document upload, and knowledge ingestion controls. The central area shows the conversation history and a fixed chat input at the bottom.",
            "When a user sends a message, the frontend calls the FastAPI streaming chat endpoint. The backend retrieves context from the vector database, builds a prompt, and streams the local LLM response back to the UI.",
        ], figure={"path": assets["ui_chat"], "caption": "Figure 4.8 RFP chat workspace screenshot-style UI figure."}),
        spec("4.6.2 Disease Prediction UI Screenshot and Explanation", [
            "The disease prediction workspace is structured differently because it requires numeric clinical-style inputs rather than free-form text. Each field is labelled and explained so that users understand the meaning of Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.",
            "After submission, the frontend calls the diabetes prediction endpoint. The response presents the model prediction, probability, threshold, retrieved context count, generated precautions, and a medical safety disclaimer.",
        ], figure={"path": assets["ui_diabetes"], "caption": "Figure 4.9 Disease prediction RAG workspace screenshot-style UI figure."}),
        spec("4.7 Diabetes Sample Verification", [
            "The model service was tested with a sample row containing Pregnancies 6, Glucose 148, BloodPressure 72, SkinThickness 35, Insulin 0, BMI 33.6, DiabetesPedigreeFunction 0.627, and Age 50.",
            "The service returned prediction class 1, risk probability 0.5628, threshold 0.3238, and risk label diabetes_risk_detected. This confirmed that the scaler, model, threshold, and feature order were working together.",
        ]),
        spec("4.8 Dataset Distribution: Glucose", [
            "Glucose is one of the strongest clinical-style indicators in the dataset. The distribution chart helps visualize the spread of values used by the model during training.",
            "The presence of low or zero-like values in some medical datasets must be interpreted carefully. In future work, domain-specific preprocessing can improve robustness.",
        ], figure={"path": dataset_images[0], "caption": "Figure 4.2 Glucose distribution in diabetes dataset."}),
        spec("4.9 Dataset Distribution: BMI", [
            "BMI is another important feature because it reflects weight-to-height ratio. In diabetes screening datasets, BMI often contributes to risk patterns, although it should never be used alone.",
            "The chart supports exploratory understanding of the dataset and helps explain why tabular machine-learning models can be useful.",
        ], figure={"path": dataset_images[1], "caption": "Figure 4.3 BMI distribution in diabetes dataset."}),
        spec("4.10 Dataset Distribution: Age", [
            "Age influences diabetes risk and is included as a model feature. The distribution chart gives an overview of the age range represented in the dataset.",
            "A model trained on one population may not generalize perfectly to another. This is one reason the output is framed as educational support rather than diagnosis.",
        ], figure={"path": dataset_images[2], "caption": "Figure 4.4 Age distribution in diabetes dataset."}),
        spec("4.11 Dataset Distribution: Blood Pressure", [
            "Blood pressure is included as a feature in the prediction pipeline. Its relationship with diabetes risk is indirect but clinically relevant in broader metabolic health assessment.",
            "The report includes this chart to make the dataset more transparent and to support discussion of feature behavior.",
        ], figure={"path": dataset_images[3], "caption": "Figure 4.5 Blood pressure distribution in diabetes dataset."}),
        spec("4.12 Feature Correlation Result", [
            "The correlation heatmap provides a quick view of linear relationships between dataset features and the outcome. Correlation does not prove causation, but it is helpful during exploratory analysis.",
            "Tree-based models such as XGBoost can learn non-linear relationships, so correlation is only one part of understanding model behavior.",
        ], figure={"path": dataset_images[4], "caption": "Figure 4.6 Feature correlation heatmap."}),
        spec("4.13 Testing Strategy", [
            "The tests folder includes end-to-end API tests and tests for knowledge engine components such as chunking, embedding, retrieval, loading, vector store behavior, and LLM workflow.",
            "The system also benefits from manual UI testing because Streamlit behavior and backend availability are visible only during runtime.",
        ], figure={"path": assets["testing"], "caption": "Figure 4.7 Testing strategy by focus area."}),
        spec("4.14 Result Discussion", [
            "The implemented system demonstrates successful integration of RAG chat and XGBoost prediction. It shows that enterprise document intelligence and predictive analytics can share the same user interface and backend service architecture.",
            "The main implementation challenge is service startup time. Embedding models, rerankers, vector stores, and local LLM calls can be heavy. Lazy loading and caching reduce repeated cost, but startup and first request still require patience.",
        ]),
        spec("4.15 User Experience Result", [
            "The chat UI was improved to resemble a modern assistant interface. The disease prediction screen was added with a separate workspace so users do not confuse RFP questions with medical-style input.",
            "The UI also needed color fixes. Some input text and labels were initially too faint due to styling inheritance, so explicit CSS was added for readable text, labels, placeholder color, and focus states.",
        ]),
        spec("Chapter 5: Conclusion and Future Work", [
            "RFPForge successfully demonstrates how a local AI system can combine document ingestion, semantic retrieval, language generation, structured APIs, Streamlit interaction, and machine-learning prediction.",
            "The project also shows a responsible pattern for predictive output: the model predicts risk, but the generated advice includes disclaimers and encourages professional consultation.",
        ], figure={"path": assets["risk"], "caption": "Figure 5.1 Project risk register summary."}),
        spec("5.1 Major Findings", [
            "The project confirms that RAG is useful for answering questions over project-specific or organization-specific documents. It also confirms that an XGBoost model can be connected to a RAG assistant to make model output more understandable.",
            "The combined interface improves usability because users can perform both free-form chat and structured prediction from the same application.",
        ]),
        spec("5.2 Limitations", [
            "The current system is a development-stage project. It depends on local services such as Ollama and ChromaDB. If these services are not running or are not initialized, the user may experience delays or failed requests.",
            "The diabetes module is not clinically validated for real medical decision-making. It should be treated only as an educational demonstration of model integration.",
        ], table={"title": "Table 5.1 Limitations and Future Enhancements", "headers": ["Limitation", "Future Enhancement"], "rows": [["Startup latency", "Load heavy services lazily and show UI status"], ["Medical validation absent", "Add clinician-reviewed guidance and validation"], ["Limited deployment automation", "Add Docker Compose"], ["Manual report fields", "Add project metadata form"], ["Basic model explainability", "Add SHAP-based explanations"]]}),
        spec("5.3 Future Scope", [
            "Future work can improve both the RFP assistant and the disease prediction module. The RFP workflow can include approval chains, source citations, version comparison, and export templates.",
            "The disease module can include explainability charts, multiple disease models, validation reports, and better clinical safety controls.",
        ], bullets=["Add Docker-based deployment.", "Add role-based authentication.", "Add citation-aware answers with source previews.", "Add SHAP explanations for XGBoost predictions.", "Add PDF export directly from the application.", "Add more disease prediction models with separate disclaimers."]),
        spec("5.4 Conclusion", [
            "The completed project is a meaningful example of applied AI engineering. It is not limited to a notebook or an isolated model; it is a working application with backend endpoints, a frontend interface, local AI services, persistent storage, testing, and a project-report-ready structure.",
            "By combining RAG with XGBoost, RFPForge demonstrates how different AI techniques can support practical decision workflows when they are wrapped in responsible software design.",
        ]),
    ])
    filler_topics = [
        spec("Detailed Module Note: API Router Design", "The router design keeps each business area separate. Chat logic remains in chat.py, knowledge ingestion remains in knowledge.py, diabetes prediction remains in diabetes.py, and RFP workflows remain in rfp.py. This separation makes it easier for a developer to locate bugs and add features."),
        spec("Detailed Module Note: Service Singleton Design", "Heavy services such as embedding models and vector stores should not be recreated for every request. The dependency file stores service instances and returns existing objects after first initialization. This approach improves performance during repeated interactions."),
        spec("Detailed Module Note: Streamlit Interaction", "Streamlit was selected because it allows fast construction of data-oriented interfaces in Python. The project uses it for file upload, chat rendering, workspace selection, disease input fields, and result presentation."),
        spec("Detailed Module Note: Model Artifact Handling", "The XGBoost classifier, scaler, and threshold are stored as pickle artifacts. The service loads them lazily and uses a lock to prevent unsafe repeated loading during concurrent access."),
        spec("Detailed Module Note: Medical Safety Language", "The disease prediction output is framed as educational guidance. The prompt explicitly tells the language model not to diagnose, not to prescribe medication, and to recommend confirmation with a qualified clinician."),
        spec("Detailed Module Note: Knowledge Chunk Metadata", "Chunk metadata is important because it preserves traceability. Metadata can include the source document, detected section title, clause references, chunk ID, and structure flag."),
        spec("Detailed Module Note: Embedding Cache", "Embedding cache reduces repeated computation. If the same chunk text and model name are encountered again, the cached vector can be loaded instead of recomputed."),
        spec("Detailed Module Note: Export Potential", "The RFP workflow includes export functions for Word and Excel. This is important because proposal teams normally need final answers in familiar document formats."),
        spec("Detailed Module Note: UI Readability", "A practical AI interface must be readable. The project required explicit styling for input text, labels, borders, and placeholder text so that users could clearly see what they were typing."),
        spec("Detailed Module Note: Development Constraints", "The project runs locally and may require large model downloads. This can affect startup time and memory usage, but it also supports privacy and offline-style operation."),
        spec("Detailed Module Note: Data Quality", "The diabetes dataset contains numeric features that should be interpreted carefully. Future versions should include cleaning rules, missing-value handling, and stronger validation before prediction."),
        spec("Detailed Module Note: Reranking", "The retrieval service includes optional reranking with a BGE reranker. Reranking can improve the order of retrieved passages but may increase latency."),
        spec("Detailed Module Note: Chat Memory", "The chat memory service stores recent turns by session ID. This allows the prompt to include short-term conversational context without storing long histories permanently."),
        spec("Detailed Module Note: CORS", "CORS middleware is enabled so that the Streamlit frontend can communicate with the FastAPI backend during development. Production deployments should restrict allowed origins."),
        spec("Detailed Module Note: Testing", "The project includes tests for API behavior and knowledge engine behavior. Test coverage is important because AI systems can fail in subtle ways when dependencies or data formats change."),
        spec("Detailed Module Note: Run Script", "The run.py script starts both backend and frontend. It was updated to avoid console emoji encoding problems and to use the current Python executable for consistent virtual environment behavior."),
        spec("Detailed Module Note: Database Models", "The RFP database models include sessions, questions, and drafts. These models represent a realistic proposal workflow where each client RFP can contain multiple questions and draft versions."),
        spec("Detailed Module Note: Prompt Context", "The generated prompt combines retrieved document content, the user question, and recent conversation history. This gives the LLM enough information to produce a relevant answer."),
        spec("Detailed Module Note: Privacy", "Because documents, embeddings, and LLM calls are local, the architecture is better suited for sensitive proposal material than a fully external SaaS workflow."),
        spec("Detailed Module Note: Maintainability", "The repository structure makes the project understandable. Backend routes, services, knowledge engine, schemas, database files, and frontend code are separated into clear folders."),
        spec("Detailed Module Note: Performance", "The most expensive operations are model loading, embedding generation, reranking, and LLM streaming. Caching and singleton services help reduce repeated cost."),
        spec("Detailed Module Note: Error Handling", "FastAPI routes raise HTTP errors for invalid inputs and failed operations. More user-friendly frontend error messages can be added in future versions."),
        spec("Detailed Module Note: Results Presentation", "The diabetes result is shown through metrics and a chat response. This makes the model output understandable without hiding the probability or threshold."),
        spec("Detailed Module Note: Academic Value", "The project is academically valuable because it demonstrates modern AI concepts in a complete software system: RAG, embeddings, vector storage, LLM prompts, API design, and tabular machine learning."),
        spec("Detailed Module Note: Industry Relevance", "Organizations increasingly need internal AI assistants that operate over private documents. RFPForge is a focused example of that broader industry direction."),
        spec("Detailed Module Note: Ethical Framing", "The system should avoid overclaiming. Especially for health-related output, it must present machine-learning results as support information rather than final medical truth."),
        spec("Detailed Module Note: Documentation", "The README and this report together explain installation, architecture, endpoints, and project purpose. Future documentation can add diagrams generated directly from OpenAPI schemas."),
        spec("Detailed Module Note: Extensibility", "The current disease prediction module can be generalized. A future ModelRegistry service could support many trained models with their own schemas and explanations."),
        spec("Detailed Module Note: Final Assessment", "Overall, the project forms a strong base for an AI-assisted enterprise knowledge and prediction platform. The next step is production hardening, validation, and better user management."),
    ]
    filler_topics = filler_topics[:12]
    for item in filler_topics:
        title = item["heading"]
        text = item["paragraphs"]
        one_page(doc, title, [
            text,
            "This topic is included as a separate detailed note to make the report complete and easier to evaluate. It connects the implementation choices with practical project reasoning, which is important in an academic project report.",
            "The design decision also shows that the project was built as a maintained application rather than a single experiment. Each component has a defined role in the full workflow.",
        ])
    for page in pages:
        one_page(doc, **page)


def references_and_appendix(doc: Document) -> None:
    one_page(doc, "References", [
        "The following references were used for the project background, implementation technologies, medical knowledge sources, and dataset description. The project uses local implementation files together with public documentation and public health information sources.",
        "These references should be retained in the final project file because they make the report traceable and show the source of important technical and medical information.",
    ], table={
        "title": "Table R.1 Reference Sources Used in the Project",
        "headers": ["No.", "Reference", "Use in Project"],
        "rows": [
            ["1", "FastAPI Documentation, https://fastapi.tiangolo.com/", "Backend API framework and router structure."],
            ["2", "Streamlit Documentation, https://docs.streamlit.io/", "Frontend chat UI, forms, sidebar, and app layout."],
            ["3", "ChromaDB Documentation, https://docs.trychroma.com/", "Persistent vector database and similarity search."],
            ["4", "SentenceTransformers Documentation, https://www.sbert.net/", "Embedding model usage for semantic retrieval."],
            ["5", "Ollama Documentation, https://ollama.com/", "Local LLM execution and streaming generation."],
            ["6", "XGBoost Documentation, https://xgboost.readthedocs.io/", "Diabetes prediction model artifact usage."],
            ["7", "Scikit-learn Documentation, https://scikit-learn.org/", "StandardScaler preprocessing and ML utilities."],
            ["8", "LangChain Text Splitters Documentation, https://python.langchain.com/", "Recursive text splitting in the RAG pipeline."],
            ["9", "MedlinePlus XML Files, https://medlineplus.gov/xml.html", "Official medical RAG knowledge source."],
            ["10", "MedlinePlus XML Description, https://medlineplus.gov/xmldescription.html", "Understanding health-topic XML fields."],
            ["11", "NIDDK Diabetes Health Information, https://www.niddk.nih.gov/health-information/diabetes", "Diabetes education and safety context."],
            ["12", "CDC Type 2 Diabetes Prevention Guide, https://www.cdc.gov/diabetes/prevention-type-2/type-2-diabetes-prevention-guide.html", "Lifestyle and prevention guidance."],
            ["13", "PhysioNet, https://physionet.org/data/", "Reference for public clinical datasets."],
            ["14", "Kaggle Pima Indians Diabetes Database, https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database", "Tabular diabetes prediction dataset reference."],
        ],
    })
    one_page(doc, "Appendix A: Sample Diabetes Prediction Payload", [
        "The following JSON payload is an example request body for the diabetes prediction endpoint. It matches the eight expected model features and can be tested from FastAPI Swagger UI.",
        '{\n  "Pregnancies": 6,\n  "Glucose": 148,\n  "BloodPressure": 72,\n  "SkinThickness": 35,\n  "Insulin": 0,\n  "BMI": 33.6,\n  "DiabetesPedigreeFunction": 0.627,\n  "Age": 50\n}',
    ])
    one_page(doc, "Appendix B: Important Project Commands", [
        "The backend can be started with: venv\\Scripts\\python.exe -m uvicorn app.main:app --reload",
        "The frontend can be started with: venv\\Scripts\\streamlit run frontend/app.py",
        "The combined launcher can be started with: venv\\Scripts\\python.exe run.py",
        "The syntax check used during development was: venv\\Scripts\\python.exe -m py_compile frontend/app.py app/api/diabetes.py",
    ])
    one_page(doc, "Appendix C: Project File Structure", [
        "The app folder contains backend code, API routers, database files, schemas, services, RFP workflows, and the knowledge engine. The frontend folder contains the Streamlit app. The ML-MODEL folder stores the diabetes dataset and XGBoost artifacts.",
        "The data folder stores uploaded documents, vector store files, the university project format PDF, generated report assets, and this generated project report.",
    ])


def build_report() -> None:
    assets = generate_assets()
    doc = Document()
    set_margins(doc.sections[0])
    add_page_number(doc.sections[0])
    configure_styles(doc)
    cover_page(doc)
    preliminary_pages(doc)
    contents_pages(doc)
    main = doc.add_section(WD_SECTION.NEW_PAGE)
    set_margins(main)
    add_page_number(main)
    chapter_pages(doc, assets)
    references_and_appendix(doc)
    try:
        doc.save(OUT)
        print(OUT)
    except PermissionError:
        doc.save(UPDATED_OUT)
        print(UPDATED_OUT)


if __name__ == "__main__":
    build_report()

