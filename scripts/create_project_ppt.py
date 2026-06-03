from __future__ import annotations

import html
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "RFPForge_Project_Presentation.pptx"
ASSETS = ROOT / "data" / "ppt_assets"
REPORT_ASSETS = ROOT / "data" / "report_assets"
ML_ARTIFACTS = ROOT / "ML-MODEL" / "artifacts"

SLIDE_W = 12192000
SLIDE_H = 6858000
MEDIA_DIR = "ppt/media"


def emu(inches: float) -> int:
    return int(inches * 914400)


def safe(text: str) -> str:
    return html.escape(str(text), quote=True)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def round_rect(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            test = f"{current} {word}"
            if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(draw, text: str, xy, fnt, fill, max_width: int, line_gap: int = 6) -> int:
    x, y = xy
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def save_breast_ordered_input_mock() -> Path:
    path = ASSETS / "ui_breast_ordered_input.png"
    img = Image.new("RGB", (1600, 900), "#f7f7f8")
    d = ImageDraw.Draw(img)
    title = font(34, True)
    body = font(22)
    small = font(18)
    mono = font(18)

    d.rectangle((0, 0, 250, 900), fill="#171717")
    d.text((48, 44), "RFPForge", font=font(30, True), fill="#ffffff")
    round_rect(d, (35, 135, 215, 185), 12, "#242424", "#52525b", 2)
    d.text((66, 149), "Disease prediction", font=small, fill="#f4f4f5")
    d.text((48, 270), "Knowledge", font=font(20, True), fill="#f4f4f5")

    d.text((320, 58), "Multi-Disease Prediction RAG Chat", font=title, fill="#111827")
    d.text((320, 105), "Run trained disease screening models and receive RAG-generated precautions and next steps.", font=small, fill="#6b7280")
    d.line((320, 145, 1500, 145), fill="#e5e7eb", width=2)

    round_rect(d, (320, 178, 1500, 238), 8, "#ffffff", "#e5e7eb", 2)
    d.text((346, 196), "Disease model", font=small, fill="#111827")
    round_rect(d, (530, 188, 820, 226), 6, "#ffffff", "#d1d5db", 2)
    d.text((548, 197), "Breast Cancer Risk", font=small, fill="#111827")

    round_rect(d, (320, 270, 1500, 615), 10, "#ffffff", "#d1d5db", 2)
    d.text((348, 292), "Ordered feature input", font=font(25, True), fill="#111827")
    d.text((348, 326), "Paste a Python or JSON list in registry feature order, then apply it to the form.", font=small, fill="#6b7280")

    code = """breast_cancer_high_risk = [[
    20.5, 25.2, 135.0, 1300.0, 0.13,
    0.22, 0.30, 0.15, 0.25, 0.09,
    1.2, 1.8, 8.5, 150.0, 0.009,
    0.05, 0.06, 0.02, 0.03, 0.006,
    28.0, 35.0, 190.0, 2400.0, 0.18,
    0.50, 0.70, 0.30, 0.45, 0.12
]]"""
    round_rect(d, (348, 365, 1105, 568), 8, "#111827", "#374151", 2)
    y = 386
    for line in code.splitlines():
        d.text((372, y), line, font=mono, fill="#e5e7eb")
        y += 24

    round_rect(d, (1130, 390, 1455, 448), 8, "#111827", "#111827", 1)
    d.text((1191, 407), "Apply ordered input", font=body, fill="#ffffff")
    round_rect(d, (1130, 468, 1455, 526), 8, "#ffffff", "#d1d5db", 2)
    d.text((1163, 485), "Load high-risk sample", font=body, fill="#111827")

    d.text((320, 655), "Existing manual form remains available after applying ordered input", font=font(24, True), fill="#111827")
    labels = ["Mean radius", "Mean texture", "Mean perimeter", "Mean area"]
    vals = ["20.500", "25.200", "135.000", "1300.000"]
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 320 + i * 295
        d.text((x, 700), label, font=small, fill="#111827")
        round_rect(d, (x, 730, x + 250, 785), 8, "#ffffff", "#d1d5db", 2)
        d.text((x + 18, 746), val, font=body, fill="#111827")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def save_breast_output_mock() -> Path:
    path = ASSETS / "ui_breast_output_example.png"
    img = Image.new("RGB", (1600, 900), "#f7f7f8")
    d = ImageDraw.Draw(img)
    title = font(32, True)
    body = font(22)
    small = font(18)
    tiny = font(16)

    d.rectangle((0, 0, 250, 900), fill="#171717")
    d.text((48, 44), "RFPForge", font=font(30, True), fill="#ffffff")
    round_rect(d, (35, 135, 215, 185), 12, "#242424", "#52525b", 2)
    d.text((66, 149), "Disease prediction", font=small, fill="#f4f4f5")

    d.text((320, 58), "Breast Cancer Risk Example Run", font=title, fill="#111827")
    d.text((320, 105), "High-risk ordered input produces a disease-aware, validated explanation.", font=small, fill="#6b7280")
    d.line((320, 145, 1500, 145), fill="#e5e7eb", width=2)

    cards = [
        ("Prediction", "Risk detected", "#b91c1c"),
        ("Risk probability", "99.9%", "#111827"),
        ("Confidence", "High", "#111827"),
        ("Threshold", "44.00%", "#111827"),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 320 + i * 295
        round_rect(d, (x, 178, x + 260, 275), 8, "#ffffff", "#e5e7eb", 2)
        d.text((x + 22, 198), label, font=tiny, fill="#6b7280")
        d.text((x + 22, 228), value, font=font(25, True), fill=color)

    round_rect(d, (320, 315, 880, 615), 8, "#ffffff", "#e5e7eb", 2)
    d.text((348, 340), "Top model contributors", font=font(24, True), fill="#111827")
    headers = ["Feature", "Value", "SHAP Impact"]
    xs = [348, 570, 715]
    for x, header in zip(xs, headers):
        d.text((x, 388), header, font=tiny, fill="#6b7280")
    rows = [
        ("radius_error", "3.3103", "+4.4071"),
        ("area_error", "3.1859", "+3.5899"),
        ("worst_concave_points", "2.7857", "+2.8019"),
        ("worst_symmetry", "2.5351", "+2.2872"),
    ]
    y = 425
    for row in rows:
        d.line((348, y - 10, 835, y - 10), fill="#e5e7eb", width=1)
        for x, value in zip(xs, row):
            d.text((x, y), value, font=small, fill="#111827")
        y += 43

    round_rect(d, (920, 315, 1500, 720), 8, "#ffffff", "#e5e7eb", 2)
    d.text((948, 340), "Generated guidance", font=font(24, True), fill="#111827")
    guidance = (
        "Prediction Summary\n"
        "The system estimated elevated breast cancer risk from tissue morphology features.\n\n"
        "Interpretation\n"
        "The result indicates abnormal tissue characteristics associated with malignant patterns in the dataset.\n\n"
        "Important\n"
        "This is research decision-support, not a diagnosis. Clinical imaging, biopsy, and physician evaluation are required.\n\n"
        "Retrieval validation: passed"
    )
    draw_wrapped(d, guidance, (948, 385), small, "#374151", 500, 7)

    d.text((320, 650), "Disease-specific retrieval validation prevents unrelated ultrasound or fasting instructions from appearing.", font=body, fill="#111827")
    round_rect(d, (320, 700, 880, 760), 8, "#dcfce7", "#86efac", 2)
    d.text((345, 718), "Accepted context: breast cancer / mammogram / biopsy", font=small, fill="#166534")
    round_rect(d, (320, 778, 880, 838), 8, "#fee2e2", "#fecaca", 2)
    d.text((345, 796), "Rejected context: abdominal ultrasound procedure", font=small, fill="#991b1b")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def save_retrieval_guardrail_diagram() -> Path:
    path = ASSETS / "retrieval_guardrails.png"
    img = Image.new("RGB", (1600, 900), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((80, 60), "Disease-Specific Retrieval Validation", font=font(40, True), fill="#111827")
    boxes = [
        ("Prediction result", "breast_cancer"),
        ("Disease-aware query", "breast cancer malignant biopsy"),
        ("Vector retrieval", "candidate passages"),
        ("Validation", "score >= 0.75 + topic match"),
        ("Generation", "template + accepted context only"),
    ]
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2", "#ede9fe"]
    for i, ((head, sub), color) in enumerate(zip(boxes, colors)):
        x = 90 + i * 295
        round_rect(d, (x, 270, x + 240, 430), 14, color, "#374151", 2)
        d.text((x + 22, 300), head, font=font(24, True), fill="#111827")
        draw_wrapped(d, sub, (x + 22, 340), font(20), "#374151", 195)
        if i < len(boxes) - 1:
            d.line((x + 245, 350, x + 285, 350), fill="#374151", width=4)
            d.polygon([(x + 285, 350), (x + 270, 340), (x + 270, 360)], fill="#374151")
    round_rect(d, (220, 570, 690, 720), 12, "#f0fdf4", "#86efac", 2)
    d.text((250, 600), "Accepted", font=font(28, True), fill="#166534")
    d.text((250, 648), "Breast cancer screening, mammogram,\nbiopsy, malignant tissue morphology", font=font(21), fill="#166534")
    round_rect(d, (900, 570, 1370, 720), 12, "#fef2f2", "#fca5a5", 2)
    d.text((930, 600), "Rejected", font=font(28, True), fill="#991b1b")
    d.text((930, 648), "Abdominal ultrasound procedure,\nfasting instructions, unrelated symptoms", font=font(21), fill="#991b1b")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def ensure_assets() -> dict[str, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    assets = {
        "ordered_input": save_breast_ordered_input_mock(),
        "breast_output": save_breast_output_mock(),
        "guardrails": save_retrieval_guardrail_diagram(),
    }
    for name in [
        "system_layers.png",
        "rag_flow.png",
        "deployment_view.png",
        "endpoint_coverage.png",
        "testing_strategy.png",
        "ui_rfp_chat_screenshot.png",
        "ui_diabetes_prediction_screenshot.png",
    ]:
        p = REPORT_ASSETS / name
        if p.exists():
            assets[name.removesuffix(".png")] = p
    for disease in ["breast_cancer", "heart_disease", "diabetes", "parkinsons"]:
        for fig in ["roc_curve", "confusion_matrix", "feature_importance", "shap_summary"]:
            p = ML_ARTIFACTS / disease / "figures" / f"{fig}.png"
            if p.exists():
                assets[f"{disease}_{fig}"] = p
    return assets


class Slide:
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.shapes: list[str] = []
        self.rels: list[tuple[str, str, str]] = []

    def add_text(self, text: str, x: float, y: float, w: float, h: float, size: int = 20, color: str = "374151", bold: bool = False):
        body = "".join(
            f'<a:p><a:r><a:rPr lang="en-US" sz="{size * 100}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{safe(line)}</a:t></a:r></a:p>'
            for line in text.split("\n")
        )
        self.shapes.append(
            f"""
            <p:sp><p:nvSpPr><p:cNvPr id="{100 + len(self.shapes)}" name="Text"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
            <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
            <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{body}</p:txBody></p:sp>
            """
        )

    def add_bullets(self, bullets: Iterable[str], x: float, y: float, w: float, h: float, size: int = 20):
        body = ""
        for bullet in bullets:
            body += (
                f'<a:p><a:pPr marL="342900" indent="-171450"><a:buChar char="•"/></a:pPr>'
                f'<a:r><a:rPr lang="en-US" sz="{size * 100}"><a:solidFill><a:srgbClr val="374151"/></a:solidFill></a:rPr>'
                f'<a:t>{safe(bullet)}</a:t></a:r></a:p>'
            )
        self.shapes.append(
            f"""
            <p:sp><p:nvSpPr><p:cNvPr id="{100 + len(self.shapes)}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
            <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
            <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{body}</p:txBody></p:sp>
            """
        )

    def add_image(self, path: Path, x: float, y: float, w: float, h: float | None = None):
        rid = f"rId{len(self.rels) + 2}"
        with Image.open(path) as im:
            iw, ih = im.size
        if h is None:
            h = w * ih / iw
        media_name = f"image_{abs(hash(str(path))) % 10_000_000}_{len(self.rels)}{path.suffix.lower()}"
        self.rels.append((rid, str(path), media_name))
        self.shapes.append(
            f"""
            <p:pic><p:nvPicPr><p:cNvPr id="{200 + len(self.shapes)}" name="{safe(path.name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
            <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
            <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>
            """
        )

    def add_card(self, text: str, x: float, y: float, w: float, h: float, fill: str = "FFFFFF"):
        self.shapes.append(
            f"""
            <p:sp><p:nvSpPr><p:cNvPr id="{300 + len(self.shapes)}" name="Card"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
            <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="12700"><a:solidFill><a:srgbClr val="E5E7EB"/></a:solidFill></a:ln></p:spPr>
            <p:txBody><a:bodyPr wrap="square" lIns="171450" tIns="114300" rIns="171450" bIns="114300"/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"><a:solidFill><a:srgbClr val="111827"/></a:solidFill></a:rPr><a:t>{safe(text)}</a:t></a:r></a:p></p:txBody></p:sp>
            """
        )

    def xml(self) -> str:
        title = f"""
        <p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="{emu(0.55)}" y="{emu(0.28)}"/><a:ext cx="{emu(12.2)}" cy="{emu(0.55)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
        <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="3000" b="1"><a:solidFill><a:srgbClr val="111827"/></a:solidFill></a:rPr><a:t>{safe(self.title)}</a:t></a:r></a:p></p:txBody></p:sp>
        """
        subtitle = ""
        if self.subtitle:
            subtitle = f"""
            <p:sp><p:nvSpPr><p:cNvPr id="3" name="Subtitle"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
            <p:spPr><a:xfrm><a:off x="{emu(0.58)}" y="{emu(0.82)}"/><a:ext cx="{emu(11.8)}" cy="{emu(0.35)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
            <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1500"><a:solidFill><a:srgbClr val="6B7280"/></a:solidFill></a:rPr><a:t>{safe(self.subtitle)}</a:t></a:r></a:p></p:txBody></p:sp>
            """
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
        <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>
        <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
        {title}{subtitle}{''.join(self.shapes)}
        </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""

    def rels_xml(self) -> str:
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        ]
        for rid, _src, media_name in self.rels:
            rels.append(
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_name}"/>'
            )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def metrics_text(disease: str) -> str:
    path = ML_ARTIFACTS / disease / "metrics.json"
    if not path.exists():
        return "Metrics artifact unavailable"
    data = json.loads(path.read_text(encoding="utf-8"))
    tm = data.get("test_metrics", {})
    return (
        f"Selected: {data.get('selected_model', 'model')}\n"
        f"Accuracy: {tm.get('accuracy', 0):.4f}\n"
        f"F1: {tm.get('f1', 0):.4f}\n"
        f"ROC-AUC: {tm.get('roc_auc', 0):.4f}"
    )


def build_slides(assets: dict[str, Path]) -> list[Slide]:
    slides: list[Slide] = []

    s = Slide("RFPForge", "AI-powered RFP automation with multi-disease prediction, RAG explanations, SHAP, and calibrated probabilities")
    s.add_text("Professional Project Presentation", 0.8, 1.55, 6.1, 0.5, 24, "111827", True)
    s.add_bullets([
        "FastAPI backend, Streamlit frontend, ChromaDB vector store, local LLM workflow",
        "Disease prediction models for diabetes, heart disease, breast cancer, and Parkinson's disease",
        "Disease-specific retrieval validation prevents irrelevant medical context generation",
    ], 0.8, 2.15, 6.0, 2.2, 21)
    if "system_layers" in assets:
        s.add_image(assets["system_layers"], 7.05, 1.35, 5.4)
    slides.append(s)

    s = Slide("Project Problem", "Manual proposal drafting and raw ML outputs both need grounded, explainable assistance")
    s.add_card("RFP Problem\nTeams spend time searching scattered documents, copying clauses, and drafting answers manually.", 0.8, 1.55, 3.8, 1.45, "DBEAFE")
    s.add_card("Healthcare AI Problem\nPrediction numbers alone are hard to interpret and can become unsafe if generic medical text is retrieved.", 4.85, 1.55, 3.8, 1.45, "DCFCE7")
    s.add_card("Solution\nA single app combines document RAG, structured ML prediction, disease-aware templates, and retrieval validation.", 8.9, 1.55, 3.6, 1.45, "FEF3C7")
    s.add_bullets(["Goal: make responses faster, more traceable, and easier to understand.", "Scope: local educational research system, not a clinical diagnostic device.", "Output: usable Streamlit workflows plus reproducible backend services."], 1.0, 3.55, 11.2, 2.2, 23)
    slides.append(s)

    s = Slide("System Architecture", "Layered architecture keeps UI, API, RAG, ML, and storage responsibilities clear")
    if "system_layers" in assets:
        s.add_image(assets["system_layers"], 0.8, 1.35, 5.8)
    if "deployment_view" in assets:
        s.add_image(assets["deployment_view"], 7.0, 1.35, 5.4)
    slides.append(s)

    s = Slide("RAG Workflow", "Documents are loaded, chunked, embedded, retrieved, and converted into grounded responses")
    if "rag_flow" in assets:
        s.add_image(assets["rag_flow"], 0.75, 1.25, 5.9)
    if "ui_rfp_chat_screenshot" in assets:
        s.add_image(assets["ui_rfp_chat_screenshot"], 6.9, 1.35, 5.45)
    slides.append(s)

    s = Slide("Disease Prediction Workflow", "Users can use manual feature fields or ordered-list input for exact research samples")
    if "ui_diabetes_prediction_screenshot" in assets:
        s.add_image(assets["ui_diabetes_prediction_screenshot"], 0.75, 1.35, 5.7)
    s.add_bullets([
        "Registry-driven disease selector and feature form",
        "Ordered list parser supports Python-style inputs with comments",
        "Prediction response includes probability, confidence, SHAP impacts, and validated guidance",
        "Known diseases use disease-aware templates to avoid hallucinated clinical procedures",
    ], 6.8, 1.55, 5.65, 3.8, 22)
    slides.append(s)

    s = Slide("Ordered Input Example", "Breast cancer high-risk sample pasted in the exact feature order")
    s.add_image(assets["ordered_input"], 0.75, 1.2, 11.8)
    slides.append(s)

    s = Slide("Example Prediction Output", "The output is what a user would see after applying the sample and running prediction")
    s.add_image(assets["breast_output"], 0.75, 1.2, 11.8)
    slides.append(s)

    s = Slide("Retrieval Guardrails", "Unrelated ultrasound, fasting, and symptom blocks are rejected before generation")
    s.add_image(assets["guardrails"], 0.8, 1.25, 11.7)
    slides.append(s)

    s = Slide("Model Coverage", "Current disease models and artifact-backed metrics")
    diseases = [("diabetes", "Diabetes"), ("heart_disease", "Heart Disease"), ("breast_cancer", "Breast Cancer"), ("parkinsons", "Parkinson's")]
    for i, (disease_id, label) in enumerate(diseases):
        x = 0.8 + (i % 2) * 6.1
        y = 1.45 + (i // 2) * 2.25
        s.add_card(f"{label}\n{metrics_text(disease_id)}", x, y, 5.5, 1.75, ["DBEAFE", "DCFCE7", "FEF3C7", "EDE9FE"][i])
    slides.append(s)

    s = Slide("Breast Cancer Explainability", "SHAP, ROC curve, and confusion matrix support academic explainability")
    if "breast_cancer_shap_summary" in assets:
        s.add_image(assets["breast_cancer_shap_summary"], 0.7, 1.25, 4.2)
    if "breast_cancer_roc_curve" in assets:
        s.add_image(assets["breast_cancer_roc_curve"], 4.85, 1.25, 3.9)
    if "breast_cancer_confusion_matrix" in assets:
        s.add_image(assets["breast_cancer_confusion_matrix"], 8.8, 1.25, 3.8)
    slides.append(s)

    s = Slide("Heart Disease Explainability", "Model behavior is reviewed with ROC, confusion matrix, and feature importance")
    if "heart_disease_feature_importance" in assets:
        s.add_image(assets["heart_disease_feature_importance"], 0.75, 1.3, 4.3)
    if "heart_disease_roc_curve" in assets:
        s.add_image(assets["heart_disease_roc_curve"], 4.75, 1.3, 3.9)
    if "heart_disease_confusion_matrix" in assets:
        s.add_image(assets["heart_disease_confusion_matrix"], 8.75, 1.3, 3.9)
    slides.append(s)

    s = Slide("Diabetes Explainability", "Legacy and publication-grade diabetes artifacts support comparison and reporting")
    if "diabetes_shap_summary" in assets:
        s.add_image(assets["diabetes_shap_summary"], 0.75, 1.25, 4.3)
    if "diabetes_roc_curve" in assets:
        s.add_image(assets["diabetes_roc_curve"], 4.8, 1.25, 3.85)
    if "diabetes_confusion_matrix" in assets:
        s.add_image(assets["diabetes_confusion_matrix"], 8.75, 1.25, 3.85)
    slides.append(s)

    s = Slide("Testing and API Surface", "Focused automated tests protect retrieval, ML contracts, and core API behavior")
    if "endpoint_coverage" in assets:
        s.add_image(assets["endpoint_coverage"], 0.9, 1.45, 5.2)
    if "testing_strategy" in assets:
        s.add_image(assets["testing_strategy"], 6.8, 1.45, 5.2)
    slides.append(s)

    s = Slide("Key Improvements Implemented", "The publication-strength changes focus on safety, explainability, and credibility")
    s.add_bullets([
        "Semantic retrieval validation with disease-specific constraints",
        "Disease-aware response templates for breast cancer and other disease models",
        "SHAP impact table shown in the UI instead of generic medical paragraphs",
        "CalibratedClassifierCV added to the training pipeline for future artifacts",
        "Probability display capped at 99.9% and confidence labels added",
        "Ordered input button supports exact paper/test samples and manual field editing",
    ], 1.0, 1.55, 11.4, 4.5, 24)
    slides.append(s)

    s = Slide("Conclusion", "RFPForge is a complete applied AI system, not only a model or a notebook")
    s.add_card("What works\nRAG chat, document ingestion, vector search, disease prediction, SHAP display, ordered input, validated guidance.", 0.9, 1.55, 5.6, 2.0, "DCFCE7")
    s.add_card("Future work\nExternal clinical validation, citation previews, authentication, Docker deployment, richer calibration plots, clinician-reviewed content.", 6.9, 1.55, 5.6, 2.0, "DBEAFE")
    s.add_text("Final message: explainability, retrieval filtering, calibrated probabilities, and disease-aware generation improve the paper more than chasing another 1-2% accuracy.", 1.0, 4.25, 11.3, 1.2, 25, "111827", True)
    slides.append(s)

    return slides


def content_types(slide_count: int, media_exts: Iterable[str]) -> str:
    media_defaults = ""
    for ext in sorted(set(e.lstrip(".") for e in media_exts)):
        ctype = "image/png" if ext == "png" else "image/jpeg"
        media_defaults += f'<Default Extension="{ext}" ContentType="{ctype}"/>'
    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>{media_defaults}
    <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
    <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
    <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
    <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
    <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
    <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
    {slide_overrides}</Types>"""


def write_pptx(slides: list[Slide], out: Path) -> None:
    media: dict[str, Path] = {}
    for slide in slides:
        for _rid, src, media_name in slide.rels:
            media[media_name] = Path(src)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides), [p.suffix for p in media.values()]))
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""")
        now = datetime.now(timezone.utc).isoformat()
        z.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>RFPForge Project Presentation</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>""")
        z.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>RFPForge PPT Generator</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""")
        slide_ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1))
        z.writestr("ppt/presentation.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>""")
        pres_rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(slides) + 1))
        pres_rels += f'<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId{len(slides)+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
        z.writestr("ppt/_rels/presentation.xml.rels", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{pres_rels}</Relationships>""")
        z.writestr("ppt/slideMasters/slideMaster1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>""")
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""")
        z.writestr("ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="RFPForge"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="374151"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="047857"/></a:accent2><a:accent3><a:srgbClr val="B45309"/></a:accent3><a:accent4><a:srgbClr val="BE123C"/></a:accent4><a:accent5><a:srgbClr val="6D28D9"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="6D28D9"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>""")
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide.rels_xml())
        for media_name, src in media.items():
            z.write(src, f"{MEDIA_DIR}/{media_name}")


def main() -> None:
    assets = ensure_assets()
    slides = build_slides(assets)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_pptx(slides, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
