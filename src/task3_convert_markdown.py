"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import shutil
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"  ↷ Kept existing standardized file: {output_path}")
                continue
            try:
                result = md.convert(str(filepath))
                output_path.write_text(result.text_content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path}")
            except Exception as e:
                print(f"  ✗ Error: {e}")


def convert_news_articles():
    """Chuẩn hóa PDF/DOCX/JSON/Markdown trong ``landing/news``."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(news_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc", ".json", ".md"):
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        print(f"Converting: {filepath.name}")
        try:
            if suffix == ".md":
                # Dữ liệu nhóm nhận được đã là Markdown có YAML front matter.
                # Copy nguyên văn để không làm mất metadata citation/customer_role.
                if filepath.resolve() != output_path.resolve():
                    shutil.copyfile(filepath, output_path)
            elif suffix == ".json":
                data = json.loads(filepath.read_text(encoding="utf-8"))
                header = (
                    f"# {data.get('title', 'Unknown')}\n\n"
                    f"**Source:** {data.get('url', 'N/A')}\n"
                    f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
                )
                content = data.get("content_markdown") or data.get("content") or ""
                output_path.write_text(header + str(content), encoding="utf-8")
            else:
                result = md.convert(str(filepath))
                output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
