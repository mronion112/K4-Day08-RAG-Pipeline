"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Cross-encoder Reranker (BGE Reranker — local, multilingual)
# =============================================================================

_bge_model = None
_bge_tokenizer = None


def _load_bge_reranker():
    global _bge_model, _bge_tokenizer
    if _bge_model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "BGE Reranker requires transformers + torch. "
                "Install: pip install transformers torch"
            )
        model_name = "BAAI/bge-reranker-v2-m3"
        _bge_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bge_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _bge_model.eval()
    return _bge_model, _bge_tokenizer


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    model, tokenizer = _load_bge_reranker()

    pairs = [[query, c["content"]] for c in candidates]
    inputs = tokenizer(
        pairs, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

    results = []
    for c, score in zip(candidates, scores.tolist()):
        results.append({**c, "score": round(score, 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# Maximal Marginal Relevance
# =============================================================================

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_similarity(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_similarity(
                    candidates[idx]["embedding"], candidates[sel_idx]["embedding"]
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    results = []
    for i in selected:
        item = candidates[i].copy()
        item["score"] = round(float(item.get("score", 0.0)), 4)
        results.append(item)

    return results


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 4)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("MMR requires query_embedding — call rerank_mmr directly")
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")

    print("\n--- RRF with 2 ranked lists ---")
    sem_results = [
        {"content": "Trả hàng hoàn tiền trong 7 ngày", "score": 0.9, "metadata": {}},
        {"content": "Thanh toán COD và chuyển khoản", "score": 0.7, "metadata": {}},
    ]
    lex_results = [
        {"content": "Chính sách bảo mật thông tin", "score": 8.5, "metadata": {}},
        {"content": "Trả hàng hoàn tiền trong 7 ngày", "score": 5.0, "metadata": {}},
    ]
    fused = rerank_rrf([sem_results, lex_results], top_k=3)
    for r in fused:
        print(f"[{r['score']:.4f}] {r['content']}")
