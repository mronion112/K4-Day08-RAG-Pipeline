# Report Cá Nhân
## Trần Quang Minh — 01210
### Vai trò: Data Engineer & Team Leader (RAG Architect)

---

## Phân Công Trong Nhóm

| Thành viên | Task | Vai trò |
|-----------|------|---------|
| **Trần Quang Minh (01210)** | **1, 2** + Ghép pipeline | Data Engineer + Team Leader |
| Ngô Văn Nam (01340) | 3, 4 | Convert & Chunking/Indexing |
| Nguyễn Hoàng Anh | 5, 6 | Semantic Search & Lexical Search |
| Thành viên khác | 7, 8, 9 | Reranking, PageIndex, Pipeline |

---

## Phần 1: Task Tôi Phụ Trách (Data Engineer)

### Task 1 — Thu Thập Tài Liệu Pháp Luật (≥3 file PDF)

**Cách làm:**
1. Lấy 3 file chính sách Shopee từ `shopee_policy.zip` (có sẵn file markdown gốc)
2. Convert markdown → PDF bằng Python (`markdown` + `PyMuPDF/fitz`)
3. Lưu vào `data/landing/legal/`

**Kết quả:**

| File | Dung lượng | `customer_role` | Lý do chọn role |
|------|-----------|-----------------|-----------------|
| `chinh-sach-tra-hang-va-hoan-tien.pdf` | 427KB | `buyer` | Chỉ người mua trả hàng |
| `chinh-sach-van-chuyen-shopee.pdf` | 456KB | `buyer` | Chính sách giao hàng cho buyer |
| `chinh-sach-bao-mat.pdf` | 504KB | `both` | Áp dụng buyer + seller |

**Tại sao chọn 3 chính sách này:**
- 3 chủ đề khách hàng hỏi nhiều nhất: trả hàng, vận chuyển, bảo mật
- Đa dạng độ dài (13K → 29K → 56K markdown) để test RAG ở nhiều mức độ
- Tên file giữ nguyên tiếng Việt từ Shopee → dễ trace nguồn

**Tại sao convert sang PDF:**
- PDF là format phổ biến nhất của tài liệu chính sách thực tế
- PyMuPDF (`fitz`) render HTML → PDF với CSS màu cam Shopee (`#ee4d2d`)
- Không dùng DOCX vì phức tạp hơn, không cần thiết

**Pytest:**
```
TestTask1::test_files_not_empty          ✅ PASSED
TestTask1::test_landing_legal_dir_exists ✅ PASSED
TestTask1::test_minimum_3_legal_files    ✅ PASSED
```

---

### Task 2 — Crawl Bài Hướng Dẫn (≥5 file)

**Cách làm:**
- Lấy 5 file markdown từ `shopee_policy.zip` copy vào `data/landing/news/`

**5 chủ đề:**

| File | Chủ đề |
|------|--------|
| `evoucher-nhan-sau-khi-mua.md` | Nhận e-voucher sau mua hàng |
| `shopeefood-dat-mon.md` | Hướng dẫn đặt món ShopeeFood |
| `shopeefood-lien-ket-tai-khoan.md` | Liên kết tài khoản ShopeeFood |
| `spaylater-thanh-toan-shopeefood.md` | Thanh toán SPayLater trên ShopeeFood |
| `tai-khoan-ngan-hang-cap-nhat-thong-tin.md` | Cập nhật tài khoản ngân hàng |

**Tại sao dùng file .md thay vì crawl:**
- Dữ liệu có sẵn từ zip, nội dung đầy đủ, không bị thiếu
- Hướng dẫn Lab cảnh báo: crawl Shopee gặp SPA (JavaScript render) → có thể chỉ lấy được tiêu đề, mất nội dung
- Định dạng `.md` được test chấp nhận (`.json/.html/.md/.txt`) → đơn giản nhất

**Pytest:**
```
TestTask2::test_landing_news_dir_exists  ✅ PASSED
TestTask2::test_minimum_5_news_files     ✅ PASSED
TestTask2::test_news_files_have_content  ✅ PASSED
TestTask2::test_json_files_have_metadata ⏭ SKIPPED (không dùng JSON — test cho phép)
```

---

## Phần 2: Ghép Pipeline Hoàn Chỉnh (Team Leader / RAG Architect)

Là Team Leader, tôi chịu trách nhiệm **ghép code từ tất cả thành viên** thành 1 pipeline RAG hoàn chỉnh chạy end-to-end.

### Kiến trúc pipeline sau khi ghép

```
User Query
    │
    ├─→ Semantic Search (Task 5) ─→ kết quả theo ngữ nghĩa
    ├─→ Lexical BM25 (Task 6)    ─→ kết quả theo từ khóa
    │
    └─→ RRF Merge (Task 7) ─→ gộp thứ hạng 2 ranker
            │
            ├─→ Score < threshold? ──YES──→ PageIndex Fallback (Task 8)
            │        │
            │       NO
            │        │
            └─→ Trả về top_k kết quả (source: "hybrid" hoặc "pageindex")
```

### Các điểm quan trọng khi ghép pipeline

**1. Đảm bảo tương thích giữa các task:**

| Mối nối | Kiểm tra |
|---------|----------|
| Task 4 → Task 5 | Cùng `get_embedding_model()` và `get_collection()` — tránh sai dimension |
| Task 5 + 6 → Task 7 | `rerank_rrf()` nhận `list[list[dict]]` — 2 ranked list từ 2 ranker |
| Task 9 → Task 8 | Lazy import `pageindex_search` trong `try/except` — không crash nếu API lỗi |

**2. Bẫy threshold fallback (Task 9):**
- **Sai**: Dùng điểm RRF (max ~0.016) để so threshold → không câu hỏi nào đủ thấp để trigger fallback
- **Đúng**: Dùng **cosine score gốc** từ `semantic_search` — threshold = 0.30
- Calibrate từ thực nghiệm: relevant query ~0.38-0.57, irrelevant ~0.27-0.35

**3. `customer_role` xuyên suốt pipeline:**
- Task 1 → gán role cho từng file trong `ROLE_MAP`
- Task 4 → embed `customer_role` vào metadata mỗi chunk trong ChromaDB
- Đảm bảo không mất metadata khi chunk → RRF merge → output

**4. Xử lý lỗi & edge cases:**
- PageIndex API lỗi → `try/except` trả về hybrid results thay vì crash
- ChromaDB collection cũ → tự động xóa + reindex khi chạy lại
- `get_embedding_model()` dispatch theo `EMBEDDING_PROVIDER` trong `.env` — linh hoạt đổi provider không cần sửa code

### Code ghép chính: `src/task9_retrieval_pipeline.py`

```python
def retrieve(query, top_k=5, score_threshold=0.30):
    # Bước 1: Chạy song song semantic + lexical
    dense = semantic_search(query, top_k * 2)
    sparse = lexical_search(query, top_k * 2)

    # Bước 2: Merge bằng RRF (Task 7)
    merged = rerank_rrf([dense, sparse], top_k * 2)

    # Bước 3: Kiểm tra threshold — dùng COSINE GỐC
    if dense[0]["score"] < score_threshold:
        return pageindex_search(query, top_k)  # Fallback (Task 8)

    return merged[:top_k]
```

---

## Tổng Kết — Toàn Bộ Pipeline

```
$ pytest tests/test_individual.py -v
======================== 28 passed, 7 skipped ========================
```

| Task | Mô tả | Người làm | Pytest |
|------|-------|-----------|--------|
| 1 | Thu thập legal docs | **Tôi** | ✅ 3/3 |
| 2 | Crawl news articles | **Tôi** | ✅ 3/4 |
| 3 | Convert markdown | Nam | ✅ 4/4 |
| 4 | Chunking & Indexing | Nam | ✅ 4/4 |
| 5 | Semantic Search | Hoàng Anh | ✅ 4/4 |
| 6 | Lexical Search (BM25) | Hoàng Anh | 1/4 |
| 7 | RRF Reranking | Khác | ✅ 3/3 |
| 8 | PageIndex Fallback | Khác | ✅ 2/2 |
| 9 | Retrieval Pipeline | **Tôi ghép** | ✅ 4/4 |

---

## Khó Khăn & Cách Khắc Phục Khi Ghép Pipeline

| Lỗi | Nguyên nhân | Cách fix |
|-----|-------------|----------|
| Task 5 import Task 4 bị sai module path | Mỗi thành viên code trong file riêng, import path khác nhau | Chuẩn hóa `from .task4_chunking_indexing import get_embedding_model` |
| `KeyError: 'customer_role'` | Task 4 không gán metadata khi chunk | Thêm `ROLE_MAP` mapping tên file → role |
| PageIndex timeout khi chạy full test | API chậm, SDK không có timeout | Lazy import + try/except trong Task 9 |
| ChromaDB dimension mismatch | Đổi embedding model không xóa DB cũ | Tự động `shutil.rmtree(chroma_db)` trước khi reindex |
| RRF score không phản ánh relevance thật | RRF chỉ dựa trên thứ hạng, không phải similarity | Giữ cosine score gốc riêng để quyết định fallback |
