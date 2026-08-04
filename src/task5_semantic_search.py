"""Task 5 - Dense semantic retrieval and HyDE over the Task 4 Chroma index."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1024
HYDE_MODEL = os.getenv("HYDE_MODEL", "gpt-4o-mini")


def _openai_client():
    """Create the API client only when retrieval actually needs an embedding."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong file .env")
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def embed_query(text: str) -> list[float]:
    """Embed one query with the same model and dimension used by Task 4."""
    response = _openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIM,
    )
    vector = response.data[0].embedding
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} dimensions, got {len(vector)}")
    return vector


def get_collection():
    """Open the existing Task 4 collection and fail clearly if it is absent."""
    import chromadb

    if not CHROMA_DIR.is_dir():
        raise RuntimeError("Chưa có chroma_db; hãy chạy Task 4 trước")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Không tìm thấy collection {COLLECTION_NAME!r}; hãy chạy Task 4 trước"
        ) from exc
    if collection.count() == 0:
        raise RuntimeError(f"Collection {COLLECTION_NAME!r} đang rỗng")
    return collection


def _chroma_where(metadata_filter: dict | None) -> dict | None:
    """Convert simple role filters to Chroma syntax, including shared docs."""
    if not metadata_filter:
        return None
    filters = dict(metadata_filter)
    role = filters.pop("customer_role", None)
    conditions: list[dict] = [{key: value} for key, value in filters.items()]
    if role in {"buyer", "seller"}:
        conditions.append(
            {"$or": [{"customer_role": role}, {"customer_role": "both"}]}
        )
    elif role is not None:
        conditions.append({"customer_role": role})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def semantic_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
    *,
    query_text_for_embedding: str | None = None,
) -> list[dict]:
    """Return cosine-similar chunks sorted by descending similarity."""
    query = query.strip()
    if not query:
        raise ValueError("query không được để trống")
    if top_k <= 0:
        return []

    collection = get_collection()
    n_results = min(top_k, collection.count())
    kwargs: dict[str, Any] = {
        "query_embeddings": [embed_query(query_text_for_embedding or query)],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    where = _chroma_where(metadata_filter)
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    output: list[dict] = []
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]
    for item_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        output.append(
            {
                "id": item_id,
                "content": document,
                "score": round(1.0 - float(distance), 6),
                "metadata": metadata or {},
                "source": "semantic",
            }
        )
    return sorted(output, key=lambda item: item["score"], reverse=True)[:top_k]


def generate_hypothetical_document(query: str) -> str:
    """Generate a short answer-like document for Hypothetical Document Embedding."""
    query = query.strip()
    if not query:
        raise ValueError("query không được để trống")
    response = _openai_client().chat.completions.create(
        model=HYDE_MODEL,
        temperature=0,
        max_tokens=250,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn viết một đoạn tài liệu Shopee giả định để hỗ trợ tìm kiếm. "
                    "Viết bằng tiếng Việt, ngắn gọn, chứa thuật ngữ có khả năng xuất "
                    "hiện trong chính sách. Không bịa mã, số tiền hoặc thời hạn cụ thể."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    document = (response.choices[0].message.content or "").strip()
    if not document:
        raise RuntimeError("HyDE không tạo được hypothetical document")
    return document


def hyde_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Search using an answer-like hypothetical document as the dense query."""
    hypothetical_document = generate_hypothetical_document(query)
    results = semantic_search(
        query,
        top_k=top_k,
        metadata_filter=metadata_filter,
        query_text_for_embedding=hypothetical_document,
    )
    for result in results:
        result["source"] = "hyde"
        result["metadata"] = {
            **result["metadata"],
            "retrieval_method": "hyde",
        }
    return results


if __name__ == "__main__":
    for item in semantic_search("quy định trả hàng hoàn tiền Shopee", top_k=5):
        print(f"[{item['score']:.3f}] {item['metadata'].get('source')}: {item['content'][:100]}...")
