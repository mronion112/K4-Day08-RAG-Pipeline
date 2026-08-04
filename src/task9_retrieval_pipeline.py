"""
Task 9 — Retrieval Pipeline không phụ thuộc Task 7/8.

Kết hợp semantic search + lexical search thành một pipeline thống nhất. Nhóm đã
chủ động bỏ Task 7 (reranker riêng) và Task 8 (PageIndex API), vì vậy RRF tối
thiểu được đặt ngay trong module này và câu hỏi evidence yếu trả về danh sách
rỗng để Task 10 từ chối trả lời an toàn.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF)
    3. Nếu evidence yếu và BM25 cũng không match → trả về []
    4. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrated from actual query scores:
#   Relevant:  0.38-0.57
#   Irrelevant: 0.27-0.35
# Threshold 0.30 safely separates those ranges.
SCORE_THRESHOLD = 0.30
DEFAULT_TOP_K = 5
RRF_K = 60


def _result_key(item: dict) -> tuple:
    metadata = item.get("metadata") or {}
    return (
        metadata.get("source"),
        metadata.get("chunk_index"),
        item.get("content", ""),
    )


def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """RRF cục bộ để Task 9 chạy dù Task 7 được bỏ khỏi phạm vi nhóm."""
    scores: dict[tuple, float] = {}
    items: dict[tuple, dict] = {}

    for result_type, ranked_list in (
        ("dense", dense_results),
        ("sparse", sparse_results),
    ):
        for rank, item in enumerate(ranked_list, 1):
            key = _result_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            stored = items.setdefault(key, dict(item))
            stored[f"{result_type}_score"] = item.get("score", 0.0)

    ordered_keys = sorted(scores, key=scores.get, reverse=True)
    fused: list[dict] = []
    for key in ordered_keys[:top_k]:
        item = items[key]
        item["score"] = round(scores[key], 6)
        item["rrf_score"] = item["score"]
        item["source"] = "hybrid"
        fused.append(item)
    return fused


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    # Step 1: Run semantic + lexical in parallel
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Task 8/PageIndex được bỏ: evidence semantic yếu và không có lexical match
    # phải trả [] để Task 10 dùng NO_EVIDENCE_MESSAGE thay vì sinh câu trả lời rác.
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    if best_dense_score < score_threshold and not sparse_results:
        return []

    if use_reranking:
        return _rrf_fuse(dense_results, sparse_results, top_k=top_k)

    # Chế độ dense-only phục vụ so sánh/debug, vẫn giữ output contract Task 10.
    output = []
    for item in dense_results[:top_k]:
        copied = dict(item)
        copied["dense_score"] = copied.get("score", 0.0)
        copied["source"] = "hybrid"
        output.append(copied)
    return output


if __name__ == "__main__":
    test_queries = [
        "trả hàng hoàn tiền shopee",
        "vận chuyển giao hàng",
        "thanh toán spaylater",
        "bóng đá world cup 2026",   # Lạc đề → test fallback
        "xyzabc123nonsense",        # Rác → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
