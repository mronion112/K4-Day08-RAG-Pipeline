"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import os
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# HELPERS
# =============================================================================

_openai_client = None
_chroma_collection = None


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Tách YAML front matter đơn giản từ corpus Markdown của nhóm."""
    if not text.startswith("---\n"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2].strip()


def get_embedding_model():
    """Trả về callable embed_texts(texts: list[str]) -> list[list[float]]."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        global _openai_client
        if _openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Thiếu OPENAI_API_KEY để tạo embeddings.")
            _openai_client = OpenAI(api_key=api_key)

        def embed_texts(texts):
            resp = _openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [d.embedding for d in resp.data]

        return embed_texts

    if provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")

        def embed_texts(texts):
            return model.encode(texts, show_progress_bar=False).tolist()

        return embed_texts

    if provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        def embed_texts(texts):
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
            )
            return result["embedding"]

        return embed_texts

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# =============================================================================
# IMPLEMENTATION
# =============================================================================

ROLE_MAP = {
    "chinh-sach-tra-hang-va-hoan-tien.md": "buyer",
    "chinh-sach-van-chuyen-shopee.md": "buyer",
    "chinh-sach-bao-mat.md": "both",
    "evoucher-nhan-sau-khi-mua.md": "buyer",
    "shopeefood-dat-mon.md": "buyer",
    "shopeefood-lien-ket-tai-khoan.md": "buyer",
    "spaylater-thanh-toan-shopeefood.md": "buyer",
    "tai-khoan-ngan-hang-cap-nhat-thong-tin.md": "buyer",
}


def load_documents() -> list[dict]:
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        front_matter, content = _parse_front_matter(raw_text)
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        customer_role = front_matter.get(
            "customer_role", ROLE_MAP.get(md_file.name, "both")
        )
        metadata = {
            "source": md_file.name,
            "type": front_matter.get("type", doc_type),
            "customer_role": customer_role,
        }
        # Chroma metadata chỉ nhận scalar; front matter của corpus hiện đều là
        # chuỗi nên có thể giữ lại các trường dùng cho citation/filter UI.
        for key in (
            "doc_id", "title", "source_url", "retrieved_at",
            "document_version", "category", "language",
        ):
            value = front_matter.get(key)
            if value:
                metadata[key] = value
        documents.append({
            "content": content,
            "metadata": metadata,
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embed_fn = get_embedding_model()
    texts = [c["content"] for c in chunks]

    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(embed_fn(batch))

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    # Xoá collection cũ để reindex
    global _chroma_collection
    _chroma_collection = None
    if CHROMA_DIR.exists():
        shutil.rmtree(str(CHROMA_DIR))

    collection = get_collection()

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
