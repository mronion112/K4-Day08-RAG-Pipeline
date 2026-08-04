"""Task 3 - Standardize landing documents as UTF-8 Markdown.

Legal PDF/DOCX files are converted with Microsoft MarkItDown. Crawled news JSON
is normalized directly so its Markdown body and retrieval metadata are kept.
The landing directory is read-only input; generated files mirror its legal/news
layout under ``data/standardized``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from markitdown import MarkItDown


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}
MIN_CONTENT_LENGTH = 200


def _yaml_scalar(value: Any) -> str:
    """Encode a value as a YAML-compatible JSON scalar without extra packages."""
    return json.dumps(value, ensure_ascii=False)


def _front_matter(metadata: dict[str, Any]) -> str:
    """Create stable YAML front matter from simple scalar metadata."""
    lines = ["---"]
    for key, value in metadata.items():
        if value is not None and value != "":
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def _write_markdown(path: Path, metadata: dict[str, Any], content: str) -> None:
    """Validate and atomically write one standardized Markdown document."""
    content = content.strip()
    if len(content) < MIN_CONTENT_LENGTH:
        raise ValueError(f"Converted content for {path.name} is too short: {len(content)} chars")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(_front_matter(metadata) + content + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_legal_manifest() -> dict[str, dict]:
    """Index optional Task 1 metadata by its landing filename."""
    manifest_path = LANDING_DIR / "legal" / "sources.json"
    if not manifest_path.is_file():
        return {}
    records = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {manifest_path}")
    return {
        str(record.get("filename")): record
        for record in records
        if isinstance(record, dict) and record.get("filename")
    }


def convert_legal_docs() -> list[Path]:
    """Convert all legal PDF/DOC/DOCX inputs using MarkItDown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_legal_manifest()
    converter = MarkItDown()
    outputs: list[Path] = []

    if not legal_dir.is_dir():
        return outputs

    for source_path in sorted(legal_dir.iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in LEGAL_EXTENSIONS:
            continue
        print(f"Converting legal: {source_path.name}")
        result = converter.convert(str(source_path))
        content = result.text_content.strip()
        source_metadata = manifest.get(source_path.name, {})
        metadata = {
            "doc_id": source_metadata.get("doc_id", source_path.stem),
            "title": source_metadata.get("title", source_path.stem.replace("-", " ").title()),
            "source_url": source_metadata.get("url", ""),
            "source_file": str(source_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "customer_role": source_metadata.get("customer_role", "both"),
            "category": source_metadata.get("category", "legal_policy"),
            "type": "legal",
            "language": "vi",
        }
        output_path = output_dir / f"{source_path.stem}.md"
        _write_markdown(output_path, metadata, content)
        outputs.append(output_path)
        print(f"  Saved: {output_path.name} ({len(content)} chars)")
    return outputs


def convert_news_articles() -> list[Path]:
    """Convert crawled news JSON documents to Markdown with front matter."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if not news_dir.is_dir():
        return outputs

    for source_path in sorted(news_dir.glob("*.json")):
        print(f"Converting news: {source_path.name}")
        data = json.loads(source_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {source_path}")
        content = str(data.get("content_markdown") or data.get("markdown_content") or "").strip()
        if not content:
            raise ValueError(f"Missing content_markdown in {source_path}")
        metadata = {
            "doc_id": data.get("doc_id", source_path.stem),
            "title": data.get("title", source_path.stem.replace("-", " ").title()),
            "source_url": data.get("url", ""),
            "date_crawled": data.get("date_crawled") or data.get("crawl_date", ""),
            "source_file": str(source_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "customer_role": data.get("customer_role", "both"),
            "category": data.get("category", "customer_guide"),
            "type": data.get("type", "faq_guide"),
            "language": data.get("language", "vi"),
        }
        output_path = output_dir / f"{source_path.stem}.md"
        _write_markdown(output_path, metadata, content)
        outputs.append(output_path)
        print(f"  Saved: {output_path.name} ({len(content)} chars)")
    return outputs


def convert_all() -> list[Path]:
    """Run the complete Task 3 standardization pipeline."""
    print("=" * 60)
    print("Task 3: landing -> standardized Markdown")
    print("=" * 60)
    legal_outputs = convert_legal_docs()
    news_outputs = convert_news_articles()
    outputs = legal_outputs + news_outputs
    print(
        f"Completed: {len(legal_outputs)} legal + {len(news_outputs)} news "
        f"= {len(outputs)} Markdown files"
    )
    return outputs


if __name__ == "__main__":
    convert_all()
