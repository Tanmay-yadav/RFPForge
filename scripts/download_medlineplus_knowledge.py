from __future__ import annotations

import argparse
import html
import re
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"
OUTPUT_DIR = Path("data/knowledge_docs/medlineplus")
ARCHIVE_DIR = Path("data/knowledge_docs/_source_archives")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:90] or "topic"


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</p>|</li>|</ul>|</ol>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return textwrap.fill(value.strip(), width=100)


def latest_compressed_xml_url() -> str:
    with urllib.request.urlopen(MEDLINEPLUS_XML_PAGE, timeout=60) as response:
        page = response.read().decode("utf-8", errors="ignore")

    match = re.search(r'href="([^"]*mplus_topics[^"]*\.zip)"', page)
    if not match:
        raise RuntimeError("Could not find the MedlinePlus compressed topic XML link.")

    href = html.unescape(match.group(1))
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://medlineplus.gov" + href
    return "https://medlineplus.gov/" + href


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def extract_xml(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError(f"No XML file found in {zip_path}")
        xml_name = xml_names[0]
        target = ARCHIVE_DIR / Path(xml_name).name
        target.write_bytes(archive.read(xml_name))
        return target


def topic_text(topic) -> str:
    title = topic.attrib.get("title", "Untitled Topic")
    url = topic.attrib.get("url", "")
    language = topic.attrib.get("language", "")
    summary = strip_html(topic.findtext("full-summary"))
    also_called = [item.text.strip() for item in topic.findall("also-called") if item.text]
    groups = [item.text.strip() for item in topic.findall("group") if item.text]

    sections = [
        f"Title: {title}",
        f"Source: MedlinePlus",
        f"URL: {url}",
        f"Language: {language}",
    ]

    if also_called:
        sections.append("Also called: " + ", ".join(also_called))
    if groups:
        sections.append("Health topic groups: " + ", ".join(groups))
    if summary:
        sections.append("\nSummary:\n" + summary)

    sections.append(
        "\nSafety note: This file is for educational retrieval only and does not replace medical advice, diagnosis, or treatment from a qualified clinician."
    )
    return "\n".join(sections).strip() + "\n"


def convert_topics(xml_path: Path, output_dir: Path, limit: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    count = 0

    for topic in root.findall("health-topic"):
        if topic.attrib.get("language") != "English":
            continue

        title = topic.attrib.get("title", "Untitled Topic")
        topic_id = topic.attrib.get("id", str(count))
        filename = f"{slugify(title)}-{topic_id}.txt"
        (output_dir / filename).write_text(topic_text(topic), encoding="utf-8")
        count += 1

        if limit and count >= limit:
            break

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MedlinePlus health topics and convert them into RAG-ready .txt files."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of English topics to write.")
    args = parser.parse_args()

    url = latest_compressed_xml_url()
    zip_path = ARCHIVE_DIR / Path(url).name
    download_file(url, zip_path)
    xml_path = extract_xml(zip_path)
    count = convert_topics(xml_path, OUTPUT_DIR, args.limit)

    print(f"Downloaded: {url}")
    print(f"Archive: {zip_path}")
    print(f"XML: {xml_path}")
    print(f"Wrote {count} text files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
