"""Task 6 - In-memory BM25 lexical retrieval over Task 4 chunks."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:  # allow: python src/task6_lexical_search.py
    from task4_chunking_indexing import chunk_documents, load_documents


CORPUS: list[dict] = []
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
_QUERY_ALIASES = {
    "payment": ["thanh", "toán"],
    "methods": ["phương", "thức"],
    "return": ["trả", "hàng"],
    "refund": ["hoàn", "tiền"],
    "seller": ["người", "bán"],
    "order": ["đơn", "hàng"],
    "tracking": ["theo", "dõi"],
}


def tokenize(text: str) -> list[str]:
    """Unicode-aware lowercase tokenizer that preserves codes and numbers."""
    return _TOKEN_PATTERN.findall(text.casefold())


def _expanded_query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_ALIASES.get(token, []))
    return expanded


def _searchable_text(document: dict) -> str:
    """Include scalar metadata so filenames/categories are exact-searchable."""
    metadata = document.get("metadata", {})
    fields = [document.get("content", "")]
    fields.extend(str(value) for value in metadata.values() if value is not None)
    return " ".join(fields)


def build_bm25_index(corpus: list[dict]):
    """Build BM25Okapi (k1=1.5, b=0.75) from retrieval chunks."""
    if not corpus:
        raise ValueError("corpus không được rỗng")
    tokenized_corpus = [tokenize(_searchable_text(document)) for document in corpus]
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


@lru_cache(maxsize=1)
def _default_index() -> tuple[list[dict], BM25Okapi]:
    """Load/chunk the corpus once per process and cache its BM25 index."""
    global CORPUS
    CORPUS = chunk_documents(load_documents())
    if not CORPUS:
        raise RuntimeError("Không có chunks; hãy hoàn thành Task 3/4 trước")
    return CORPUS, build_bm25_index(CORPUS)


def _matches_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        actual = metadata.get(key)
        if key == "customer_role" and expected in {"buyer", "seller"}:
            if actual not in {expected, "both"}:
                return False
        elif actual != expected:
            return False
    return True


def lexical_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Return positive-score BM25 chunks sorted by descending score."""
    query = query.strip()
    if not query:
        raise ValueError("query không được để trống")
    if top_k <= 0:
        return []

    corpus, bm25 = _default_index()
    scores = bm25.get_scores(_expanded_query_tokens(query))
    ranked_indices = np.argsort(scores)[::-1]
    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            break
        document = corpus[int(index)]
        if not _matches_filter(document["metadata"], metadata_filter):
            continue
        results.append(
            {
                "content": document["content"],
                "score": round(score, 6),
                "metadata": document["metadata"],
                "source": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for item in lexical_search("phương thức thanh toán Shopee", top_k=5):
        print(f"[{item['score']:.3f}] {item['metadata'].get('source')}: {item['content'][:100]}...")
