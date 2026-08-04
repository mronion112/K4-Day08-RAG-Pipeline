"""Task 1 - Prepare Shopee policy documents in ``data/landing/legal``.

The supplied corpus contains Markdown snapshots and a ``sources.csv`` catalog.
This script selects the three policies used by the project, converts them to
Unicode PDF, and writes a metadata manifest containing ``customer_role``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_ROOT = PROJECT_DIR / "shopee_policy"
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "sources.json"
SELECTED_DOCUMENTS = {
    "chinh-sach-bao-mat": "both",
    "chinh-sach-tra-hang-va-hoan-tien": "buyer",
    "chinh-sach-van-chuyen-shopee": "buyer",
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_source_catalog() -> Path:
    catalogs = list(EXTRACTED_ROOT.rglob("legal/sources.csv"))
    if not catalogs:
        raise FileNotFoundError(
            f"Không tìm thấy legal/sources.csv trong {EXTRACTED_ROOT}. "
            "Hãy giải nén shopee_policy.zip trước."
        )
    return catalogs[0]


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"[*_`]{1,3}", "", text).strip()


def markdown_to_pdf(source: Path, target: Path, title: str) -> None:
    from fpdf import FPDF

    regular_font = Path("C:/Windows/Fonts/arial.ttf")
    bold_font = Path("C:/Windows/Fonts/arialbd.ttf")
    if not regular_font.is_file():
        raise FileNotFoundError("Không tìm thấy font C:/Windows/Fonts/arial.ttf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("ArialUnicode", fname=str(regular_font))
    pdf.add_font("ArialUnicode", style="B", fname=str(bold_font))
    pdf.add_page()
    pdf.set_font("ArialUnicode", style="B", size=15)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)
    pdf.set_font("ArialUnicode", size=10)
    pdf.multi_cell(0, 5.5, _clean_markdown(source.read_text(encoding="utf-8-sig")))
    pdf.output(str(target))


def collect_documents() -> list[dict]:
    setup_directory()
    catalog_path = find_source_catalog()
    records: list[dict] = []

    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        catalog = {row["doc_id"]: row for row in csv.DictReader(stream)}

    for doc_id, customer_role in SELECTED_DOCUMENTS.items():
        row = catalog.get(doc_id)
        if row is None:
            raise KeyError(f"Thiếu {doc_id} trong {catalog_path}")
        source = catalog_path.parent / Path(row["file_path"]).name
        target = DATA_DIR / f"{doc_id}.pdf"
        markdown_to_pdf(source, target, row["title"])
        records.append(
            {
                "doc_id": doc_id,
                "filename": target.name,
                "title": row["title"],
                "url": row["source_url"],
                "retrieved_at": row.get("retrieved_at", ""),
                "category": row.get("category", "legal_policy"),
                "customer_role": customer_role,
                "original_snapshot": str(source.relative_to(PROJECT_DIR)).replace("\\", "/"),
            }
        )

    MANIFEST_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    outputs = collect_documents()
    print(f"Đã chuẩn bị {len(outputs)} tài liệu tại {DATA_DIR}")
