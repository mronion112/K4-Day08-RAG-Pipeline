"""Task 2 - Crawl Shopee customer-support articles with Crawl4AI.

Each article is stored as one UTF-8 JSON document in ``data/landing/news``.
The output schema is also the input contract expected by Task 3.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"
STANDARDIZED_NEWS_DIR = PROJECT_DIR / "data" / "standardized" / "news"
CACHE_DIR = PROJECT_DIR / ".cache" / "crawl4ai"

# Crawl4AI reads this variable while it is imported. Keeping its database and
# cache in the project avoids Windows permission errors under the user profile.
os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(CACHE_DIR))


ARTICLES = [
    {
        "doc_id": "evoucher-nhan-sau-khi-mua",
        "url": "https://help.shopee.vn/portal/4/article/79583",
        "customer_role": "buyer",
        "category": "guide_voucher",
    },
    {
        "doc_id": "shopeefood-dat-mon",
        "url": "https://help.shopee.vn/portal/4/article/79563",
        "customer_role": "buyer",
        "category": "guide_ordering",
    },
    {
        "doc_id": "shopeefood-lien-ket-tai-khoan",
        "url": "https://help.shopee.vn/portal/4/article/79521",
        "customer_role": "both",
        "category": "guide_account_link",
    },
    {
        "doc_id": "tai-khoan-ngan-hang-cap-nhat-thong-tin",
        "url": "https://help.shopee.vn/portal/4/article/79076",
        "customer_role": "both",
        "category": "guide_bank_account",
    },
    {
        "doc_id": "spaylater-thanh-toan-shopeefood",
        "url": "https://help.shopee.vn/portal/4/article/153679",
        "customer_role": "buyer",
        "category": "guide_spaylater",
    },
]

# Backwards-compatible name used by the lab starter and other team modules.
ARTICLE_URLS = [article["url"] for article in ARTICLES]
MIN_CONTENT_LENGTH = 500
MIN_SNAPSHOT_CONTENT_LENGTH = 200
MAX_ATTEMPTS = 3


def setup_directory() -> None:
    """Create writable landing and Crawl4AI cache directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_standardized_markdown(path: Path) -> tuple[dict, str]:
    """Read YAML front matter and body from one standardized news document."""
    import yaml

    text = path.read_text(encoding="utf-8-sig")
    metadata: dict = {}
    content = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            metadata = yaml.safe_load(parts[1]) or {}
            content = parts[2].strip()
    return metadata, content


def import_standardized_news() -> list[Path]:
    """Rebuild landing JSON from the five supplied standardized snapshots.

    This offline recovery mode is useful when live browser crawling is blocked.
    It deliberately reads ``standardized/news`` rather than ``standardized/legal``
    so customer guides are not mixed with policy documents.
    """
    setup_directory()
    article_by_id = {article["doc_id"]: article for article in ARTICLES}
    markdown_files = sorted(STANDARDIZED_NEWS_DIR.glob("*.md"))
    if len(markdown_files) < 5:
        raise RuntimeError(
            f"Expected at least 5 Markdown news files in {STANDARDIZED_NEWS_DIR}; "
            f"found {len(markdown_files)}"
        )

    outputs: list[Path] = []
    for source_path in markdown_files:
        metadata, content = _read_standardized_markdown(source_path)
        doc_id = str(metadata.get("doc_id") or source_path.stem)
        configured = article_by_id.get(doc_id, {})
        url = str(metadata.get("source_url") or "").strip()
        if not url.startswith("http"):
            url = configured.get("url", "")
        if not url.startswith("https://help.shopee.vn/"):
            raise ValueError(f"Missing official Shopee URL for {source_path.name}")
        if len(content) < MIN_SNAPSHOT_CONTENT_LENGTH:
            raise ValueError(
                f"Content in {source_path.name} is too short ({len(content)} chars)"
            )

        data = {
            "doc_id": doc_id,
            "url": url,
            "title": str(metadata.get("title") or doc_id.replace("-", " ").title()),
            "date_crawled": str(
                metadata.get("retrieved_at")
                or datetime.now().astimezone().isoformat(timespec="seconds")
            ),
            "content_markdown": content,
            "customer_role": str(
                metadata.get("customer_role") or configured.get("customer_role", "both")
            ),
            "category": str(
                metadata.get("category") or configured.get("category", "customer_guide")
            ),
            "language": str(metadata.get("language") or "vi"),
            "type": str(metadata.get("type") or "faq_guide"),
            "recovered_from": str(source_path.relative_to(PROJECT_DIR)),
        }
        output_path = DATA_DIR / f"{doc_id}.json"
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs.append(output_path)
        print(f"Recovered {output_path.name} ({len(content)} chars)")
    return outputs


def _markdown_text(value: Any) -> str:
    """Normalize Crawl4AI markdown across old and new library versions."""
    if isinstance(value, str):
        return value.strip()
    for attribute in ("raw_markdown", "fit_markdown"):
        text = getattr(value, attribute, None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return str(value or "").strip()


async def crawl_article(crawler: Any, article: dict) -> dict:
    """Crawl one article, validate it, and return normalized metadata."""
    last_error = "unknown crawl error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await crawler.arun(url=article["url"])
            if not getattr(result, "success", False):
                last_error = getattr(result, "error_message", "crawl unsuccessful")
            else:
                content = _markdown_text(result.markdown)
                if len(content) < MIN_CONTENT_LENGTH:
                    last_error = f"content too short ({len(content)} characters)"
                else:
                    metadata = getattr(result, "metadata", None) or {}
                    title = metadata.get("title") or article["doc_id"].replace("-", " ").title()
                    return {
                        "doc_id": article["doc_id"],
                        "url": article["url"],
                        "title": title.strip(),
                        "date_crawled": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "content_markdown": content,
                        "customer_role": article["customer_role"],
                        "category": article["category"],
                        "language": "vi",
                        "type": "faq_guide",
                    }
        except Exception as exc:  # retry transient browser/network failures
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(attempt * 2)

    raise RuntimeError(
        f"Failed to crawl {article['url']} after {MAX_ATTEMPTS} attempts: {last_error}"
    )


async def crawl_all() -> list[Path]:
    """Crawl all configured articles using a single browser session."""
    setup_directory()
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    browser_config = BrowserConfig(headless=True, verbose=False)
    saved_files: list[Path] = []
    failures: list[str] = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, article in enumerate(ARTICLES, start=1):
            print(f"[{index}/{len(ARTICLES)}] Crawling: {article['url']}")
            try:
                data = await crawl_article(crawler, article)
                output_path = DATA_DIR / f"{article['doc_id']}.json"
                output_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                saved_files.append(output_path)
                print(f"  Saved {output_path.name} ({len(data['content_markdown'])} chars)")
            except RuntimeError as exc:
                failures.append(str(exc))
                print(f"  ERROR: {exc}")

    if failures:
        raise RuntimeError(
            f"Crawled {len(saved_files)}/{len(ARTICLES)} articles.\n- "
            + "\n- ".join(failures)
        )
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-standardized",
        action="store_true",
        help="rebuild landing JSON from data/standardized/news without a browser",
    )
    args = parser.parse_args()
    outputs = (
        import_standardized_news()
        if args.from_standardized
        else asyncio.run(crawl_all())
    )
    print(f"Completed: {len(outputs)} articles saved to {DATA_DIR}")
