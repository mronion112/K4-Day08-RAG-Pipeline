# BÁO CÁO CÁ NHÂN — Lab Ngày 8: E-commerce Support RAG Pipeline (K4)

**Họ và tên:** Nguyễn Hoàng Anh
**MSSV:** 2A2202601186
**Phần đảm nhận:** Task 7 (Reranking) — Task 8 (PageIndex Vectorless Fallback) — Task 9 (Retrieval Pipeline hoàn chỉnh) + Thiết kế kiến trúc RAG tổng thể cho nhóm

---

## 1. Task nhóm giao cho mình

Nhóm chia task theo khối "Ghép & Product" — mình nhận **Task 7, 8, 9**, phần nối các module retrieval (semantic search của Task 5, lexical search của Task 6) thành một pipeline hoàn chỉnh có fallback, đồng thời phác thảo kiến trúc RAG tổng thể để cả nhóm thống nhất luồng dữ liệu trước khi ghép Task 10.

---

## 2. Thiết kế RAG cơ bản (kiến trúc tổng thể)

Trước khi code Task 7-9, mình vẽ sơ đồ luồng để nhóm thống nhất input/output giữa các task, tránh lệch format khi ghép:

```
Câu hỏi người dùng
        │
        ├──→ semantic_search()  (Task 5, dense/cosine, giữ điểm gốc)
        │
        ├──→ lexical_search()   (Task 6, BM25)
        │
        ▼
   Merge bằng RRF (rerank_rrf, Task 7)
        │
        ▼
   Rerank lần cuối (RRF/MMR/cross-encoder, Task 7)
        │
        ├── Nếu điểm cosine gốc của top-1 dense_results < threshold:
        │         └──→ Fallback: pageindex_search() (Task 8, vectorless)
        │
        ▼
   retrieve() trả về top_k kết quả kèm "source": "hybrid" | "pageindex"  (Task 9)
        │
        ▼
   generate_with_citation() (Task 10 — không thuộc phần mình)
```

**Nguyên tắc thiết kế quan trọng nhất mình chốt cho nhóm:** tách rõ hai loại điểm số —
- **Điểm cosine gốc** từ `semantic_search` (thang `[0,1]`, có ý nghĩa thật) → dùng để quyết định fallback.
- **Điểm RRF đã fuse** (chỉ phụ thuộc rank, top-1 luôn xấp xỉ `1/(k+1) ≈ 0.0164`) → chỉ dùng để sắp xếp thứ tự kết quả cuối, **không** dùng để so `score_threshold`.

Nếu không tách hai điểm này, fallback PageIndex gần như không bao giờ trigger được — kể cả với query hoàn toàn lạc đề vẫn trả "hybrid" (rác).

---

## 3. Task 7 — Reranking Module

**File:** `src/task7_reranking.py`

### Các hàm đã implement

| Hàm | Trạng thái | Mô tả |
|---|---|---|
| `rerank_rrf(ranked_lists, top_k, k=60)` | ✅ Hoàn chỉnh | Gộp nhiều ranked list bằng công thức `RRF(d) = Σ 1/(k+rank)`, dùng `content` làm key để dedupe, trả về top_k sắp xếp giảm dần theo điểm RRF |
| `rerank_mmr(query_embedding, candidates, top_k, lambda_param=0.7)` | ✅ Hoàn chỉnh | Chọn tuần tự từng candidate theo `MMR = λ·sim(query,doc) − (1−λ)·max_sim(doc, đã_chọn)`, dùng làm phương án dự phòng khi cần đa dạng hoá kết quả |
| `rerank_cross_encoder(query, candidates, top_k)` | ⚠️ Chưa xong | Có khung gọi Jina Reranker API và load `Qwen3-Reranker-0.6B` local, nhưng câu `return` bị comment nên hàm luôn rơi xuống `raise NotImplementedError` |
| `rerank(query, candidates, top_k, method)` | ⚠️ Chưa xong | Hàm interface hợp nhất — nhánh `method="cross_encoder"` gọi được, nhưng nhánh `"mmr"` và `"rrf"` chỉ `raise NotImplementedError` và yêu cầu gọi trực tiếp `rerank_mmr`/`rerank_rrf` |

### Vì sao chọn RRF làm phương pháp chính

- **Không cần API key** (khác cross-encoder Jina/Qwen3), phù hợp deadline lab ngắn.
- Semantic search và lexical search cho ra hai thang điểm khác nhau hoàn toàn (cosine `[0,1]` vs BM25 không chuẩn hoá) → cộng trực tiếp là sai. RRF chỉ dùng **rank**, nên công bằng giữa hai nguồn.
- MMR giữ lại làm phương án bonus khi cần giảm trùng lặp nội dung giữa các chunk (ví dụ nhiều chunk cùng nói về phí COD).

### Test

```bash
pytest tests/test_individual.py::TestTask7 -v
```
*(Chèn ảnh chụp màn hình kết quả pytest ở đây)*

---

## 4. Task 8 — PageIndex Vectorless Fallback

**File:** `src/task8_pageindex_vectorless.py`

### Đã implement

- `pageindex_search(query, top_k)`: gọi `client.submit_query(...)`, poll `client.get_retrieval(retrieval_id)` đến khi hoàn tất, sau đó parse đúng schema thật của API (`retrieved_nodes` → `relevant_contents` → `relevant_content` / `section_title`) thay vì đoán schema từ code mẫu cũ trong tài liệu — mình có in `json.dumps(...)` của response thật trước khi viết logic parse như README khuyến cáo.
- Vì PageIndex không trả `score` trực tiếp, mình tự gán điểm theo thứ hạng trả về (rank-based score) để giữ format đồng nhất `{'content', 'score', 'metadata', 'source': 'pageindex'}` với `semantic_search`/`lexical_search`.

### Vì sao chọn cách này

Task 9 cần fallback khi hybrid search "đói" kết quả (câu hỏi tổng hợp, hoặc câu hỏi lạc đề không match embedding nào tốt). PageIndex đọc theo cấu trúc mục lục tài liệu thay vì chunk rời rạc, nên phù hợp cho câu hỏi kiểu "tóm tắt toàn bộ quy trình khiếu nại" mà chunk nhỏ dễ mất context.

### Hạn chế còn lại (chưa fix xong)

- `upload_documents()`: PageIndex chỉ nhận PDF, không nhận `.md` trực tiếp — cần convert markdown → PDF bằng `fpdf2` trước khi `submit_document`, nhưng bước convert này mình mới ghi chú chưa code, đang tham chiếu biến `pdf_path` chưa được gán → lỗi `NameError` khi chạy.
- `pageindex_search()` hiện tham chiếu `doc_id` chưa có trong scope hàm (chưa lưu lại `doc_id` từ bước upload) → cũng lỗi `NameError`. Cần lưu mapping `{tên_file: doc_id}` ra file (vd `doc_ids.json`) sau khi upload rồi load lại khi query.

### Test

```bash
pytest tests/test_individual.py::TestTask8 -v
```
*(Chèn ảnh chụp màn hình kết quả pytest ở đây)*

---

## 5. Task 9 — Retrieval Pipeline hoàn chỉnh

**File:** `src/task9_retrieval_pipeline.py`

### Implement

Hàm `retrieve(query, top_k=5, score_threshold=0.3, use_reranking=True)`:

1. Chạy `semantic_search` và `lexical_search` với `top_k*2` để có đủ candidate trước khi rerank.
2. Merge hai list bằng `rerank_rrf([dense_results, sparse_results], top_k=top_k*2)`, gắn `source="hybrid"`.
3. Rerank lần cuối qua `RERANK_METHOD = "rrf"`.
4. **Kiểm tra fallback bằng `dense_results[0]["score"]`** (điểm cosine gốc từ semantic search, chưa qua RRF) so với `SCORE_THRESHOLD = 0.3` — đúng theo cảnh báo trong README, không so với điểm RRF đã fuse.
5. Nếu dưới ngưỡng → gọi `pageindex_search(query, top_k)` và trả kết quả đó thay vì kết quả hybrid.

### Vì sao đặt threshold ở bước riêng, không lồng vào rerank

Nếu để điểm quyết định fallback đi qua RRF trước, top-1 sẽ luôn ≈ `1/61 ≈ 0.0164` bất kể câu hỏi có liên quan hay không — tách `dense_results[0]["score"]` ra làm căn cứ fallback là điểm mấu chốt để tránh đúng cái bẫy README đã cảnh báo hai lần (ở Task 7 và Task 9).

`SCORE_THRESHOLD = 0.3` hiện là giá trị đặt tạm — README yêu cầu tự calibrate bằng cách đo cosine score cho vài câu hỏi chắc chắn liên quan và vài câu chắc chắn lạc đề rồi chọn ngưỡng nằm giữa hai nhóm. Mình mới test bằng mắt với 4 câu ở khối `if __name__ == "__main__"`, chưa đo số liệu bài bản.

### Lỗi phát hiện khi tích hợp Task 7 ↔ Task 9

Bug quan trọng nhất mình phát hiện khi ghép: ở **Bước 3**, `retrieve()` gọi hàm interface hợp nhất `rerank(query, merged, top_k=top_k, method=RERANK_METHOD)` với `RERANK_METHOD="rrf"` mặc định — nhưng nhánh `"rrf"` trong `rerank()` (Task 7) hiện chỉ `raise NotImplementedError("Call rerank_rrf with ranked_lists")`, nên bước rerank cuối trong pipeline sẽ crash ngay khi chạy thật.

**Cách fix:** ở Bước 3, gọi thẳng `rerank_rrf([merged], top_k=top_k)` (hoặc sửa lại hàm `rerank()` ở Task 7 để nhánh `"rrf"`/`"mmr"` dispatch đúng thay vì raise) thay vì gọi qua interface `rerank()` chưa hoàn chỉnh. Mình note lại để bạn owner Task 7 phần cross-encoder cùng thống nhất sửa interface này trước khi merge vào `app.py`.

### Test

```bash
pytest tests/test_individual.py::TestTask9 -v
```
*(Chèn ảnh chụp màn hình kết quả pytest ở đây)*

---

## 6. Tổng hợp lỗi + cách fix

| Lỗi | Vị trí | Nguyên nhân | Cách fix |
|---|---|---|---|
| `rerank()` raise `NotImplementedError` cho method `"rrf"`/`"mmr"` | Task 7 | Interface hợp nhất chưa dispatch đủ nhánh | Sửa `rerank()` gọi thẳng `rerank_rrf`/`rerank_mmr` tương ứng, hoặc bỏ qua interface và gọi hàm cụ thể như Task 9 đang làm |
| Pipeline crash ở Bước 3 khi rerank | Task 9 | Gọi `rerank(..., method="rrf")` trong khi nhánh đó chưa dispatch | Gọi trực tiếp `rerank_rrf([merged], top_k=top_k)` |
| `NameError: pdf_path` | Task 8 — `upload_documents` | Chưa implement bước convert `.md` → PDF trước khi upload | Thêm bước dùng `fpdf2` convert markdown sang PDF, lưu vào thư mục tạm rồi mới `submit_document(str(pdf_path))` |
| `NameError: doc_id` | Task 8 — `pageindex_search` | Chưa lưu lại `doc_id` sau khi upload | Lưu mapping tên file → `doc_id` ra `doc_ids.json` khi upload, load lại khi query |
| Fallback không bao giờ trigger (nếu vô tình dùng sai điểm) | Task 9 | So `score_threshold` với điểm RRF thay vì cosine gốc | Đã tránh được nhờ giữ `dense_results[0]["score"]` riêng biệt — ghi lại làm lưu ý cho cả nhóm khi review code |

---

## 7. Cách phần mình nối vào pipeline chung

`retrieve()` (Task 9) là điểm vào duy nhất mà `app.py` và Task 10 sẽ gọi — input là câu hỏi thô từ UI Streamlit, output là list chunk đã sẵn `content/score/metadata/source` để đưa thẳng vào `reorder_for_llm()` và `generate_with_citation()` của Task 10 mà không cần transform thêm. Mình đã thống nhất với bạn làm Task 5/6 giữ đúng format `{'content', 'score', 'metadata'}` để Task 7/9 không phải viết thêm adapter.

---

## 8. Việc cần làm tiếp trước demo

- [ ] Fix bug gọi `rerank()` sai method ở Task 9 Bước 3
- [ ] Hoàn thiện `upload_documents()` (convert md→PDF, lưu `doc_id`)
- [ ] Calibrate lại `SCORE_THRESHOLD` bằng số liệu thật thay vì giá trị tạm 0.3
- [ ] Chèn ảnh pytest cho Task 7/8/9 vào báo cáo trước khi nộp