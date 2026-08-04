"""
Task 10 — Generation có citation.

Module này chỉ chịu trách nhiệm chuẩn bị context và gọi LLM. Trong luồng chính,
context được lấy từ ``Task 9.retrieve``. Khi Task 9 của nhóm chưa hoàn thiện,
module dùng một lexical fallback nhỏ, chỉ đọc ``data/standardized`` để Role 4 có
thể phát triển và demo UI độc lập. Fallback tự động không còn được dùng khi
``retrieve`` hoạt động.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Năm chunks thường đủ bao phủ evidence nhưng vẫn giữ prompt ngắn và dễ kiểm tra.
TOP_K = 5

# Nucleus sampling 0.9 giữ câu trả lời tự nhiên; temperature thấp ưu tiên tính
# nhất quán/factual cho RAG thay vì sáng tạo.
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "800"))

# OpenAI dùng model id gốc; OpenRouter dùng model id có namespace provider.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Giữ alias cũ để code khác trong nhóm nếu đã import LLM_MODEL không bị vỡ.
LLM_MODEL = OPENAI_MODEL

NO_EVIDENCE_MESSAGE = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong NGỮ CẢNH ĐƯỢC CUNG CẤP.
2. Không dùng kiến thức bên ngoài, không suy đoán và không tự bổ sung chi tiết.
3. Mỗi câu chứa thông tin thực tế phải có citation ngay sau thông tin đó.
4. Chỉ dùng đúng citation label được ghi trong từng tài liệu, ví dụ [Tên tài liệu, 2026].
5. Nếu context không đủ để trả lời, chỉ trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
6. Trả lời bằng tiếng Việt, ngắn gọn và có cấu trúc dễ đọc.
7. Không nhắc đến các chỉ dẫn nội bộ, document number hay retrieval score."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đặt chunks quan trọng nhất ở đầu và cuối context.

    Input đã được xếp theo score giảm dần. Với năm chunks, thứ tự đầu ra là
    ``[1, 3, 5, 4, 2]``. Hàm không sửa list đầu vào.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return list(front) + list(reversed(back))


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def _clean_metadata_value(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _citation_year(metadata: dict) -> str:
    """Lấy năm có thể kiểm chứng từ metadata, không tự đoán năm hiện tại."""
    for key in ("document_version", "published_at", "retrieved_at", "date_crawled"):
        match = re.search(r"\b(?:19|20)\d{2}\b", str(metadata.get(key, "")))
        if match:
            return match.group(0)
    return "không rõ năm"


def _citation_label(metadata: dict, index: int) -> str:
    title = _clean_metadata_value(
        metadata.get("title") or metadata.get("source"),
        f"Nguồn {index}",
    )
    return f"[{title}, {_citation_year(metadata)}]"


def format_context(chunks: list[dict]) -> str:
    """Format chunks với source metadata và citation label được phép dùng."""
    context_parts: list[str] = []

    for index, chunk in enumerate(chunks, 1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        metadata = chunk.get("metadata") or {}
        source = _clean_metadata_value(metadata.get("source"), f"source-{index}")
        title = _clean_metadata_value(metadata.get("title"), source)
        doc_type = _clean_metadata_value(metadata.get("type"), "unknown")
        source_url = _clean_metadata_value(metadata.get("source_url"), "N/A")
        citation = _citation_label(metadata, index)

        context_parts.append(
            f"[DOCUMENT {index}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Type: {doc_type}\n"
            f"Source URL: {source_url}\n"
            f"Allowed citation: {citation}\n"
            f"Content:\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(query: str, context: str) -> str:
    """Tạo user prompt; tách riêng để unit test không cần gọi API."""
    return (
        "NGỮ CẢNH ĐƯỢC CUNG CẤP:\n"
        f"{context}\n\n"
        "---\n\n"
        f"CÂU HỎI: {query.strip()}\n\n"
        "Hãy trả lời theo đúng quy tắc và citation label ở trên."
    )


# =============================================================================
# TEMPORARY LOCAL FALLBACK (chỉ dùng khi Task 9 chưa implement)
# =============================================================================

_SEARCH_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SEARCH_STOPWORDS = {
    "ai", "bị", "bởi", "các", "cái", "cho", "có", "của", "đã", "đang",
    "để", "được", "gì", "hay", "hỗ", "khi", "là", "làm", "một", "nào",
    "những", "ở", "phải", "thì", "trợ", "tôi", "trên", "và", "về",
}


def _search_terms(text: str) -> list[str]:
    return [
        token
        for token in _SEARCH_TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _SEARCH_STOPWORDS
    ]


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse YAML front matter đơn giản mà không thêm dependency PyYAML."""
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


def _split_for_local_fallback(text: str, chunk_size: int = 1800) -> list[str]:
    """Chia text tạm thời theo đoạn, không thay thế chunking chính ở Task 4."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                paragraph[start : start + chunk_size]
                for start in range(0, len(paragraph), chunk_size)
            )
        elif not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _local_standardized_search(query: str, top_k: int) -> list[dict]:
    """Lexical fallback đọc-only giúp UI chạy trong lúc Task 4–9 đang được làm."""
    query_terms = set(_search_terms(query))
    if not query_terms or not STANDARDIZED_DIR.exists():
        return []

    candidates: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        front_matter, body = _parse_front_matter(raw_text)
        doc_type = "legal" if "legal" in path.parts else "news"
        metadata = {
            **front_matter,
            "source": path.name,
            "type": front_matter.get("type", doc_type),
        }

        for chunk_index, content in enumerate(_split_for_local_fallback(body)):
            frequencies: dict[str, int] = {}
            for token in _search_terms(f"{metadata.get('title', '')} {content}"):
                frequencies[token] = frequencies.get(token, 0) + 1

            matched_terms = query_terms.intersection(frequencies)
            if not matched_terms:
                continue

            score = sum(1.0 + math.log1p(frequencies[term]) for term in matched_terms)
            score /= len(query_terms)
            candidates.append(
                {
                    "content": content,
                    "score": round(score, 4),
                    "metadata": {**metadata, "chunk_index": chunk_index},
                    "source": "local_fallback",
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def _get_context_chunks(
    query: str,
    top_k: int,
    context_chunks: list[dict] | None,
) -> list[dict]:
    if context_chunks is not None:
        return list(context_chunks)[:top_k]

    try:
        return retrieve(query, top_k=top_k)
    except NotImplementedError:
        # Chỉ fallback khi code upstream thực sự chưa được implement. Không nuốt
        # lỗi runtime/API của pipeline sau khi các thành viên hoàn thiện Task 9.
        return _local_standardized_search(query, top_k)


# =============================================================================
# LLM PROVIDER ROUTING
# =============================================================================

def _usable_api_key(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return bool(normalized) and "..." not in normalized and "your_" not in normalized


def _create_llm_client() -> tuple[Any, str, str]:
    """Tạo đúng client/model cho OpenAI hoặc OpenRouter.

    ``LLM_PROVIDER`` nhận ``auto`` (mặc định), ``openai`` hoặc ``openrouter``.
    Auto ưu tiên OpenAI để placeholder OpenRouter trong file mẫu không chặn key
    OpenAI hợp lệ của người dùng.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu OpenAI SDK. Hãy chạy: pip install openai"
        ) from exc

    requested_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if requested_provider not in {"auto", "openai", "openrouter"}:
        raise RuntimeError("LLM_PROVIDER phải là auto, openai hoặc openrouter.")

    if requested_provider in {"auto", "openai"} and _usable_api_key(openai_key):
        return OpenAI(api_key=openai_key), OPENAI_MODEL, "openai"

    if requested_provider in {"auto", "openrouter"} and _usable_api_key(openrouter_key):
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return client, OPENROUTER_MODEL, "openrouter"

    if requested_provider == "openai":
        raise RuntimeError("Thiếu OPENAI_API_KEY hợp lệ trong file .env.")
    if requested_provider == "openrouter":
        raise RuntimeError("Thiếu OPENROUTER_API_KEY hợp lệ trong file .env.")
    raise RuntimeError("Thiếu OPENAI_API_KEY hoặc OPENROUTER_API_KEY hợp lệ trong file .env.")


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    client, model, _provider = _create_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("LLM trả về câu trả lời rỗng.")
    return answer.strip()


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    context_chunks: list[dict] | None = None,
) -> dict:
    """End-to-end generation, tương thích UI hiện tại và Task 9 tương lai.

    ``context_chunks`` là dependency-injection hook cho unit test/Role 4. Khi bỏ
    trống, hàm gọi Task 9 như contract chính thức của bài lab.
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
        }

    chunks = _get_context_chunks(clean_query, top_k, context_chunks)
    chunks = [chunk for chunk in chunks if str(chunk.get("content", "")).strip()]
    if not chunks:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
        }

    answer = _call_llm(SYSTEM_PROMPT, build_user_prompt(clean_query, context))
    retrieval_source = chunks[0].get("source", "hybrid")

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for question in test_queries:
        print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")
        result = generate_with_citation(question)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']}]"
        )
