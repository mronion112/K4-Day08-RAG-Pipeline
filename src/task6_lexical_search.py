"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import numpy as np
import re
from rank_bm25 import BM25Okapi
from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []
_bm25_index = None


def _tokenize(text: str) -> list[str]:
    """Tokenize Unicode đơn giản và loại dấu câu ở cuối từ khóa."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


# Corpus là tiếng Việt nhưng rubric có một số query tiếng Anh. BM25 không hiểu
# ngữ nghĩa như dense retrieval, vì vậy mở rộng các thuật ngữ domain phổ biến để
# lexical search vẫn bắt được exact Vietnamese tokens tương ứng.
_QUERY_TRANSLATIONS = {
    "return": ["trả", "hàng"],
    "refund": ["hoàn", "tiền"],
    "evidence": ["bằng", "chứng"],
    "policy": ["chính", "sách"],
    "seller": ["người", "bán"],
    "listing": ["đăng", "bán"],
    "regulations": ["quy", "định"],
    "order": ["đơn", "hàng"],
    "tracking": ["theo", "dõi"],
    "guide": ["hướng", "dẫn"],
    "payment": ["thanh", "toán"],
    "methods": ["phương", "thức"],
}


def _tokenize_query(query: str) -> list[str]:
    tokens = _tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_TRANSLATIONS.get(token, []))
    return expanded


def build_bm25_index(corpus: list[dict]):
    global _bm25_index
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index


def _load_corpus():
    global CORPUS
    if CORPUS:
        return

    CORPUS.extend(chunk_documents(load_documents()))

    if CORPUS:
        build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []

    if _bm25_index is None:
        _load_corpus()
    if _bm25_index is None or not CORPUS:
        return []

    tokenized_query = _tokenize_query(query)
    scores = _bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
