"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

from pageindex.client import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def upload_documents():
    client = _get_client()

    pdf_files = list((LANDING_DIR / "legal").glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/landing/legal/")
        return []

    doc_ids = []
    for pdf_path in pdf_files:
        print(f"Uploading: {pdf_path.name}...")
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            print(f"  ✓ doc_id: {doc_id}")

            # Poll until retrieval ready
            for _ in range(30):
                if client.is_retrieval_ready(doc_id):
                    print(f"  ✓ Retrieval ready")
                    break
                time.sleep(2)

            doc_ids.append(doc_id)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    client = _get_client()

    # Get all existing documents
    docs_resp = client.list_documents(limit=50)
    doc_ids = [d["id"] for d in docs_resp.get("documents", [])]

    if not doc_ids:
        print("No documents in PageIndex. Run upload_documents() first.")
        return []

    all_results = []
    for doc_id in doc_ids:
        try:
            # Submit retrieval query
            submit_resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submit_resp.get("retrieval_id") or submit_resp.get("id")

            # Poll for results
            for _ in range(30):
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "")
                if status == "completed":
                    break
                time.sleep(2)

            retrieved_nodes = retrieval.get("retrieved_nodes", [])
            for node in retrieved_nodes:
                for group in node.get("relevant_contents", []):
                    for item in group:
                        all_results.append({
                            "content": item.get("relevant_content", ""),
                            "score": 1.0,
                            "metadata": {
                                "section": item.get("section_title", ""),
                                "doc_id": doc_id,
                            },
                            "source": "pageindex",
                        })
        except Exception as e:
            print(f"  ✗ Error querying doc {doc_id}: {e}")

    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("quy định trả hàng hoàn tiền", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] source={r['source']} | {r['content'][:100]}...")
