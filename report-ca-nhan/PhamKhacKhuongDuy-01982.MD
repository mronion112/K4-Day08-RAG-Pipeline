# BÁO CÁO CÁ NHÂN — GOLDEN DATASET VÀ RAGAS EVALUATION

## Thông tin cá nhân

- **Họ và tên:** Phạm Khắc Khương Duy
- **MSSV:**2A202601982
- **Phần việc được giao:** Bộ câu hỏi Golden Dataset và Đánh giá RAGAS & báo cáo so sánh A/B
- **Các file phụ trách:** `eval/golden_dataset.json`, `eval/run_eval.py`, `eval/results.md`

## 1. Task được nhóm giao

Phần việc cá nhân gồm hai nội dung, theo yêu cầu "Yêu cầu 2: RAG Evaluation Pipeline" trong README:

1. Xây dựng **Golden Dataset**:
   - tối thiểu 15 cặp Q&A (`question`, `expected_answer`, `expected_context`);
   - phủ đều nội dung của cả 3 văn bản chính sách nhóm đã crawl ở Task 1.
2. Xây dựng **RAGAS Evaluation Pipeline**:
   - chạy RAG pipeline của nhóm trên toàn bộ golden dataset;
   - chấm điểm 4 metric: Faithfulness, Answer Relevancy, Context Recall, Context Precision;
   - so sánh A/B giữa ít nhất 2 config retrieval khác nhau;
   - xuất báo cáo `results.md` gồm bảng điểm, phân tích worst performers và đề xuất cải tiến.

### Cập nhật phạm vi so với kế hoạch ban đầu

Bản đánh giá ban đầu được thiết kế để so sánh A/B giữa **Config A (hybrid + rerank)** và **Config B (dense-only)**. Tuy nhiên do Task 7 (Reranking) và Task 8 (PageIndex fallback) không kịp hoàn thiện trong thời gian trên lớp, pipeline `retrieve()` của Task 9 chỉ có **semantic search + BM25 → RRF (k=60)**, không có bước rerank. Vì vậy phần đánh giá thực tế đã chạy (`run_eval.py`) là **single-config** trên pipeline hybrid không rerank, chưa thực hiện được so sánh A/B đúng như thiết kế ban đầu. Đây là hạn chế được ghi nhận công khai, không trình bày như đã hoàn thành đầy đủ yêu cầu A/B.

## 2. Nội dung đã implement

### 2.1. Golden Dataset — `golden_dataset.json`

20 cặp Q&A, vượt yêu cầu tối thiểu 15 câu:

| Tài liệu nguồn | Số câu hỏi |
|---|---:|
| `chinh-sach-tra-hang-va-hoan-tien.md` | 8 |
| `chinh-sach-van-chuyen-shopee.md` | 8 |
| `chinh-sach-bao-mat.md` | 4 |
| **Tổng** | **20** |

Cấu trúc mỗi bản ghi gồm `id`, `question`, `expected_answer`, `expected_context` (kèm `doc_id`/`section`/`snippet` để đối chiếu ngược với chunk thật khi phân tích lỗi retrieval).

### 2.2. Evaluation Pipeline — `run_eval.py`

Pipeline chạy thật đã dùng:

- **Embedding:** `text-embedding-3-small` (OpenAI, 1536-dim)
- **Chunking:** `RecursiveCharacterTextSplitter` (800/100)
- **Retrieval:** Semantic search + BM25 → hợp nhất bằng RRF (k=60), gọi qua `src.task9_retrieval_pipeline.retrieve()`
- **Generation:** `gpt-4o-mini`, temperature=0, có system prompt yêu cầu trả lời dựa trên tài liệu và từ chối khi thiếu evidence
- **Evaluation:** RAGAS với 4 metric `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`

Luồng: `load_golden_dataset()` → với mỗi câu hỏi: `retrieve()` lấy context → `generate_answer()` sinh câu trả lời → gom thành `datasets.Dataset` → `ragas.evaluate()` → `export_results()` ghi `results.md`.

## 3. Kết quả đánh giá thật (RAGAS, 20 câu, single-config)

### Overall Scores

| Metric | Score |
|---|---:|
| Faithfulness | 0.8667 |
| Answer Relevancy | 0.8187 |
| Context Recall | 0.9833 |
| Context Precision | 0.9975 |

**Nhận xét:** Context Recall và Context Precision gần như tuyệt đối (≈0.98–1.00) — pipeline retrieval (semantic + BM25 + RRF) tìm đúng và đủ context cho hầu hết câu hỏi, dù chưa có bước rerank. Faithfulness (0.87) và Answer Relevancy (0.82) thấp hơn đáng kể — điểm yếu nằm ở **bước generation**, không phải retrieval.

### Worst Performers (Bottom 3, theo Faithfulness)

| # | Question | Faithfulness | Answer Relevancy | Context Recall | Failure Stage (phân tích) | Root Cause (phân tích) |
|---|---|---:|---:|---:|---|---|
| 1 | gd-15 — Mức bồi thường tối đa cho người bán khi hàng hóa bị thất lạc/hư hỏng... | 0.000 | 0.000 | 1.000 | **Generation** | Context Recall = 1.0 chứng tỏ retrieval lấy đúng đoạn D.4.b, nhưng cả Faithfulness và Answer Relevancy đều bằng 0 — dấu hiệu model đã từ chối trả lời (rơi vào nhánh "Tôi không tìm thấy thông tin này") dù context đủ, hoặc sinh câu trả lời không liên quan tới context được cấp |
| 2 | gd-10 — Đơn hàng giá trị hàng hóa lớn hơn bao nhiêu thì không được hỗ trợ vận chuyển | 0.500 | 0.866 | 1.000 | **Generation** | Recall hoàn hảo nhưng Faithfulness chỉ 0.5 — câu trả lời có khả năng thêm chi tiết không được context xác nhận trực tiếp, hoặc diễn giải sai một phần điều kiện (vd nhầm điều kiện tính giá trị hàng hóa) |
| 3 | gd-11 — Người bán nên chọn kênh vận chuyển nào cho sản phẩm cây cảnh | 0.500 | 1.000 | 1.000 | **Generation** | Answer Relevancy tuyệt đối (1.0) cho thấy câu trả lời đúng trọng tâm câu hỏi, nhưng Faithfulness thấp — có thể model đã suy diễn thêm ngoài phạm vi B.2.a (vd tự thêm lý do/khuyến nghị không có trong điều khoản gốc) |

### Nhận xét chung

Cả 3 câu worst performer đều có **Context Recall = 1.000** — nghĩa là retrieval không phải nguyên nhân chính gây điểm thấp trong lần chạy này. Vấn đề tập trung ở **generation**: model hoặc từ chối trả lời không cần thiết (gd-15), hoặc thêm thông tin/diễn giải vượt ra ngoài phạm vi context được cấp (gd-10, gd-11).

## 4. Lý do lựa chọn giải pháp (thiết kế ban đầu)

### Chọn RAGAS thay vì DeepEval/TruLens

RAGAS có sẵn đủ 4/4 metric yêu cầu của rubric, tích hợp gọn với `datasets.Dataset`, khớp trực tiếp format `contexts`/`ground_truth` mà pipeline Task 9-10 trả về.

### `expected_context` giữ metadata thay vì string thuần

Giữ `doc_id` + `section` giúp truy ngược điều khoản gốc nhanh khi cần phân tích lỗi retrieval — dù trong lần chạy này retrieval không phải điểm nghẽn.

### A/B Hybrid+Rerank vs Dense-only (chưa thực hiện được)

Thiết kế ban đầu nhằm đo giá trị gia tăng của reranking (Task 7). Do Task 7-8 bị skip, câu hỏi này còn bỏ ngỏ và nên được bổ sung khi có thời gian, vì retrieval hiện đã gần như hoàn hảo (Recall/Precision ≈0.98–1.00) — reranking khó cải thiện thêm nhiều ở retrieval, nhưng đáng thử nếu nhóm mở rộng dataset với câu hỏi khó/ nhập nhằng hơn.

## 5. Lỗi/rủi ro gặp phải và cách khắc phục

### Không kịp so sánh A/B như kế hoạch

- **Hiện tượng:** Task 7 (rerank) và Task 8 (PageIndex fallback) không hoàn thiện trong thời gian trên lớp, nên chỉ có 1 pipeline để chạy eval, không có Config B để so sánh.
- **Cách khắc phục:** báo cáo minh bạch đây là single-config, không gán nhãn "đã hoàn thành A/B". Đề xuất bổ sung sau khi Task 7 xong (xem mục Recommendations).

### Quota giới hạn khi dùng model free

- **Hiện tượng:** RAGAS gọi LLM giám khảo nhiều lần/metric/câu hỏi; với 20 câu × 4 metric đã là dùng đáng kể quota.
- **Cách khắc phục:** dùng OpenAI trực tiếp (`OPENAI_API_KEY`) thay vì model free OpenRouter cho lần chạy chính thức, tránh gián đoạn giữa chừng.

### `faithfulness = nan` ở gd-06 (câu hỏi thu thập dữ liệu cá nhân)

- **Hiện tượng:** một dòng kết quả trong bảng per-question có giá trị `nan` cho Faithfulness dù Context Recall = 0.667.
- **Cách khắc phục (đề xuất):** kiểm tra log RAGAS cho câu này — `nan` thường do câu trả lời rỗng hoặc lỗi parse khi RAGAS tách các "statement" từ answer để chấm faithfulness; cần xem answer thật đã sinh ra là gì trước khi kết luận.

## 6. Cách phần việc nối vào pipeline chung

```text
Task 5: Semantic Search  ─┐
Task 6: BM25 Lexical      ├─→ Task 9: retrieve() = Semantic + BM25 → RRF (k=60)
Task 7: Reranking (skip)  ─┘      (chưa có bước rerank)
Task 8: PageIndex (skip)
        │
        ▼
Task 10 / generate_answer(): gpt-4o-mini, temperature=0
        │
        ▼
run_eval.py: build eval_data → ragas.evaluate() → export_results() → results.md
```

Điểm nối chính: `retrieve()` trả về `list[{"content": str, "score": float, "metadata": dict}]`, dùng trực tiếp `metadata["source"]` để log nguồn context, và `contexts = [r["content"] for r in retrieval_results]` đưa thẳng vào cả bước generate và bước eval.

## 7. Recommendations

### Cải tiến 1
**Action:** Hoàn thiện Task 7 (Reranking) và chạy lại eval với Config B = dense-only để có so sánh A/B thật như yêu cầu ban đầu.
**Expected impact:** Xác nhận bằng số liệu liệu rerank có cải thiện Faithfulness/Answer Relevancy hay không, vì Recall/Precision hiện đã gần trần (~1.0) nên khó đo thêm giá trị của rerank ở đó.

### Cải tiến 2
**Action:** Điều chỉnh system prompt ở `generate_answer()` để giảm tình trạng từ chối trả lời không cần thiết (trường hợp gd-15) — có thể do ngưỡng "không đủ evidence" bị set quá chặt.
**Expected impact:** Kỳ vọng nâng Faithfulness trung bình từ 0.87 lên gần 0.95+, vì 1/3 worst performer hiện tại có điểm 0.000 do lỗi này.

### Cải tiến 3
**Action:** Thêm bước kiểm tra câu trả lời sinh ra so với context trước khi generation kết thúc (self-check hoặc citation bắt buộc theo câu), để hạn chế model diễn giải/suy đoán vượt phạm vi context như ở gd-10, gd-11.
**Expected impact:** Giảm khoảng cách giữa Context Recall gần tuyệt đối và Faithfulness còn thấp hơn — thu hẹp GAP hiện tại (~0.12) giữa 2 metric này.