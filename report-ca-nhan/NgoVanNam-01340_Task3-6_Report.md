# BÁO CÁO CÁ NHÂN — K4 DAY 08 RAG PIPELINE

## 1. Thông tin cá nhân

| Nội dung | Thông tin |
|---|---|
| Họ và tên | Ngô Văn Nam |
| Mã số | 01340 |
| Branch cá nhân | `NgoVanNam-01340` |
| Vai trò | Data Standardization & Retrieval Developer |
| Task phụ trách | Task 3, Task 4, Task 5, Task 6 |
| Chủ đề | E-commerce Support RAG — Shopee Vietnam |

## 2. Phạm vi công việc

Phần việc của tôi bắt đầu từ dữ liệu landing do Task 1–2 cung cấp và kết thúc ở hai bộ kết quả retrieval để bàn giao cho Task 7/9:

```text
data/landing/legal/*.pdf ─┐
                         ├─ Task 3 ─> data/standardized/**/*.md
data/landing/news/*.json ─┘
                                      │
                                      ▼
                              Task 4: chunk 800/100
                              + customer_role
                              + OpenAI embedding 1024D
                                      │
                                      ▼
                         ChromaDB: ecommerce_support_docs
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
              Task 5: Semantic/HyDE        Task 6: BM25
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                         List[Dict] cho Task 7/9
```

Task 1–2 không thuộc phần owner của tôi; đây là nguồn đầu vào gồm 3 PDF chính sách và 5 JSON bài hướng dẫn.

## 3. Task 3 — Chuẩn hóa dữ liệu sang Markdown

### 3.1 Mục tiêu

Chuyển các định dạng không đồng nhất trong `data/landing/` về Markdown UTF-8, đồng thời giữ metadata cần cho retrieval.

### 3.2 File và hàm đã triển khai

File: `src/task3_convert_markdown.py`

| Hàm | Chức năng |
|---|---|
| `convert_legal_docs()` | Dùng Microsoft MarkItDown chuyển PDF/DOC/DOCX sang Markdown |
| `convert_news_articles()` | Chuyển JSON bài hướng dẫn sang Markdown |
| `_load_legal_manifest()` | Đọc metadata Task 1 từ `sources.json` |
| `_front_matter()` | Sinh YAML front matter cho tài liệu chuẩn hóa |
| `_write_markdown()` | Kiểm tra độ dài và ghi file an toàn qua file tạm |
| `convert_all()` | Chạy toàn bộ pipeline Task 3 |

### 3.3 Metadata đầu ra

Mỗi Markdown chứa YAML front matter:

```yaml
---
doc_id: "chinh-sach-tra-hang-va-hoan-tien"
title: "Chính sách trả hàng và hoàn tiền"
source_url: "https://help.shopee.vn/..."
source_file: "data/landing/legal/chinh-sach-tra-hang-va-hoan-tien.pdf"
customer_role: "buyer"
category: "returns_refunds"
type: "legal"
language: "vi"
---
```

### 3.4 Kết quả

| Nhóm dữ liệu | Input | Output Markdown |
|---|---:|---:|
| Legal | 3 PDF | 3 file |
| News/FAQ | 5 JSON | 5 file |
| Tổng | 8 file | 8 file |

### 3.5 Lý do lựa chọn

- MarkItDown phù hợp với yêu cầu lab và trích được nội dung PDF mà không phải tự viết parser.
- Markdown là định dạng văn bản nhẹ, dễ chunk và giữ được heading.
- YAML front matter tách metadata khỏi nội dung nhưng vẫn dễ đọc và dễ parse.
- Ghi qua file `.tmp` rồi replace giúp hạn chế để lại file hỏng nếu conversion dừng giữa chừng.

## 4. Task 4 — Chunking, Embedding và ChromaDB

### 4.1 Cấu hình

| Thành phần | Giá trị |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter` |
| `CHUNK_SIZE` | 800 ký tự |
| `CHUNK_OVERLAP` | 100 ký tự |
| Embedding provider | OpenAI API |
| Embedding model thực tế | `text-embedding-3-small` |
| Số chiều | 1024 |
| Vector store | ChromaDB local |
| Collection | `ecommerce_support_docs` |
| Distance | Cosine |

Ban đầu lab gợi ý local model `BAAI/bge-m3`. Tôi chuyển sang OpenAI API vì máy không cần tải/chạy model local dung lượng lớn. API `text-embedding-3-small` hỗ trợ tham số `dimensions=1024`, nên vẫn giữ đúng kích thước vector thiết kế.

API key được đọc từ `.env`, không hard-code và `.env` được loại khỏi Git.

### 4.2 File và hàm đã triển khai

File: `src/task4_chunking_indexing.py`

| Hàm | Chức năng |
|---|---|
| `load_documents()` | Đọc 8 Markdown, parse front matter và kiểm tra `customer_role` |
| `chunk_documents()` | Cắt recursive 800/100 và truyền metadata xuống từng chunk |
| `embed_chunks()` | Gọi OpenAI Embeddings API, tạo vector 1024 chiều |
| `index_to_vectorstore()` | Upsert content, embedding và metadata vào ChromaDB |
| `run_pipeline()` | Chạy load → chunk → embed → index |

### 4.3 Kết quả indexing

```text
Loaded documents : 8
Created chunks   : 116
Max chunk length : 798
Embedded chunks  : 116
Chroma records   : 116
```

### 4.4 Bảng `customer_role`

#### Theo tài liệu đầu vào

| Tài liệu | Loại | `customer_role` |
|---|---|---|
| `chinh-sach-bao-mat.md` | legal | `both` |
| `chinh-sach-tra-hang-va-hoan-tien.md` | legal | `buyer` |
| `chinh-sach-van-chuyen-shopee.md` | legal | `buyer` |
| `evoucher-nhan-sau-khi-mua.md` | news | `buyer` |
| `shopeefood-dat-mon.md` | news | `buyer` |
| `shopeefood-lien-ket-tai-khoan.md` | news | `both` |
| `spaylater-thanh-toan-shopeefood.md` | news | `buyer` |
| `tai-khoan-ngan-hang-cap-nhat-thong-tin.md` | news | `both` |

#### Theo chunk

| Role | Số chunk |
|---|---:|
| `buyer` | 53 |
| `seller` | 0 |
| `both` | 63 |
| **Tổng** | **116** |

Corpus hiện tại chưa có tài liệu dành riêng cho seller nên số chunk `seller` bằng 0. Code vẫn kiểm tra và hỗ trợ đủ ba giá trị `buyer`/`seller`/`both`. Khi Role Data bổ sung tài liệu seller, Task 4 chỉ cần chạy lại để index.

### 4.5 Lý do lựa chọn

- 800 ký tự cung cấp đủ ngữ cảnh cho một quy định nhưng không làm kết quả retrieval quá dài.
- Overlap 100 bảo vệ câu hoặc ý nằm ở ranh giới hai chunk.
- Cosine phù hợp với normalized semantic embedding.
- Metadata role giúp tránh trả quy định người mua cho người bán và ngược lại.
- ChromaDB chạy local, persistent và hỗ trợ filter metadata.

## 5. Task 5 — Semantic Search và HyDE

### 5.1 File và API

File: `src/task5_semantic_search.py`

```python
semantic_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]

hyde_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]
```

### 5.2 Cách hoạt động

Semantic Search:

1. Embed query bằng đúng `text-embedding-3-small`, 1024 chiều.
2. Query collection `ecommerce_support_docs`.
3. Chuyển cosine distance thành similarity: `score = 1 - distance`.
4. Sắp xếp score giảm dần và trả tối đa `top_k`.

HyDE:

1. Dùng `gpt-4o-mini` viết một hypothetical document ngắn giống văn phong chính sách.
2. Embed hypothetical document thay vì câu hỏi ngắn.
3. Tìm kiếm vector trên ChromaDB.

HyDE thử nghiệm với câu hỏi:

```text
Tôi muốn trả hàng và được hoàn tiền như thế nào?
```

Kết quả top 3 đều đến từ `chinh-sach-tra-hang-va-hoan-tien.md`, score cosine khoảng `0.77–0.79`.

### 5.3 Filter role

Khi query cho buyer, tài liệu `both` cũng phải được phép xuất hiện:

```python
{"$or": [
    {"customer_role": "buyer"},
    {"customer_role": "both"}
]}
```

Điều này được đóng gói trong `_chroma_where()`.

### 5.4 Output contract

```python
{
    "id": "chunk-id",
    "content": "Nội dung chunk...",
    "score": 0.790789,
    "metadata": {
        "source": "chinh-sach-tra-hang-va-hoan-tien.md",
        "type": "legal",
        "customer_role": "buyer",
        "chunk_index": 4
    },
    "source": "semantic"  # hoặc "hyde"
}
```

## 6. Task 6 — Lexical Search bằng BM25

### 6.1 File và API

File: `src/task6_lexical_search.py`

```python
lexical_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
) -> list[dict]
```

### 6.2 Cách hoạt động

- Dùng `BM25Okapi` với `k1=1.5`, `b=0.75`.
- Tokenizer Unicode hỗ trợ tiếng Việt, số, mã voucher và dấu gạch nối.
- Index cả nội dung và metadata scalar như filename, title, category.
- Cache corpus/index trong process để không build lại mỗi query.
- Chỉ trả tài liệu có score dương, sorted giảm dần.
- Hỗ trợ cùng quy tắc filter `buyer`/`seller`/`both` như Task 5.

Tôi bổ sung query expansion Anh–Việt nhỏ cho các từ phổ biến trong benchmark, ví dụ:

| English | Token tiếng Việt |
|---|---|
| payment methods | thanh toán, phương thức |
| return/refund | trả hàng, hoàn tiền |
| seller | người bán |
| order tracking | đơn hàng, theo dõi |

BM25 vẫn giữ ưu thế exact match với mã voucher hoặc mã đơn hàng; query expansion chỉ hỗ trợ corpus tiếng Việt khi benchmark dùng tiếng Anh.

## 7. Vì sao cần kết hợp Task 5 và Task 6?

| Tình huống | Semantic/HyDE | BM25 |
|---|---|---|
| Người dùng diễn đạt khác tài liệu | Tốt | Yếu |
| Mã voucher/mã đơn chính xác | Có thể bỏ sót | Tốt |
| Từ đồng nghĩa “lấy lại tiền” ↔ “hoàn tiền” | Tốt | Có thể không match |
| Keyword hiếm | Có thể bị làm mờ | Tốt |

Hai danh sách sẽ được Task 7 hợp nhất bằng RRF. Không cộng trực tiếp cosine score và BM25 score vì hai thang điểm khác nhau.

## 8. Kiểm thử

### 8.1 Lệnh test

```bash
python -m pytest tests/test_individual.py -k TestTask3 -v -p no:cacheprovider
python -m pytest tests/test_individual.py -k TestTask4 -v -p no:cacheprovider
python -m pytest tests/test_individual.py -k TestTask5 -v -p no:cacheprovider
python -m pytest tests/test_individual.py -k TestTask6 -v -p no:cacheprovider
```

Hoặc chạy chung:

```bash
python -m pytest tests/test_individual.py \
  -k "TestTask3 or TestTask4 or TestTask5 or TestTask6" \
  -v -p no:cacheprovider
```

### 8.2 Kết quả

| Task | Kết quả |
|---|---:|
| Task 3 | 4/4 passed |
| Task 4 | 4/4 passed |
| Task 5 | 4/4 passed |
| Task 6 | 4/4 passed |
| **Tổng phần phụ trách** | **16/16 passed** |

> Khi nộp báo cáo PDF, chèn ảnh chụp terminal chạy lệnh test chung vào vị trí này để đáp ứng yêu cầu “Screenshot pytest task của mình”.

## 9. Lỗi gặp phải và cách khắc phục

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `MissingDependencyException` khi đọc PDF | Thiếu extra PDF của MarkItDown | Cài `markitdown[pdf]` |
| Task 4 bị pytest skip | `.venv` thiếu `python-dotenv` nên module không import được | Cài dependency vào đúng `.venv` |
| Task 4 fail khi chưa có key | `OpenAI()` được tạo ngay lúc import | Chuyển client sang lazy initialization trong `embed_chunks()` |
| `Thiếu OPENAI_API_KEY` | Chưa tạo `.env` | Đọc key bằng `python-dotenv`; giữ `.env` ngoài Git |
| Local BGE-M3 tải lâu | Model local lớn và phụ thuộc tài nguyên máy | Chuyển sang OpenAI embedding 1024 chiều |
| Dễ trộn buyer/seller | Metadata role không được truyền xuống chunk/filter | Parse YAML front matter, validate role và filter cả `both` |
| BM25 query tiếng Anh không match corpus Việt | Lexical search yêu cầu từ khóa giống nhau | Thêm query expansion Anh–Việt giới hạn |

## 10. Cách phần việc nối vào pipeline chung

Task 3–4 cung cấp kho tri thức chuẩn hóa và indexed collection. Task 5–6 trả cùng cấu trúc `List[Dict]`, vì vậy Task 7 có thể nhận hai danh sách mà không cần chuyển đổi schema:

```python
dense_results = semantic_search(query, top_k=10, metadata_filter=role_filter)
sparse_results = lexical_search(query, top_k=10, metadata_filter=role_filter)

# Task 7
final_results = rrf_rerank(dense_results, sparse_results)
```

Sau RRF, Task 9 quyết định fallback PageIndex và Task 10 dùng chunks cuối để sinh câu trả lời có citation.

## 11. Commit cá nhân

| Commit | Nội dung |
|---|---|
| `948335e` | Hoàn thiện data pipeline Task 1–4; phần cá nhân liên quan trực tiếp là Task 3–4 |
| `5f4b49a` | Hoàn thiện Task 5 Semantic/HyDE và Task 6 BM25 |

Repository/branch:

```text
https://github.com/mronion112/K4-Day08-RAG-Pipeline
branch: NgoVanNam-01340
```

## 12. Hạn chế và hướng cải tiến

1. Corpus hiện chỉ có 8 tài liệu và chưa có tài liệu `seller`; cần bổ sung chính sách đăng bán/phí sàn để đánh giá role filter đầy đủ.
2. HyDE phát sinh thêm một lần gọi generation API và một lần embedding, nên cần cache hypothetical document khi triển khai production.
3. BM25 đang chạy in-memory; corpus lớn nên chuyển sang Elasticsearch/OpenSearch hoặc hybrid database.
4. Cần đánh giá A/B: semantic-only, BM25-only, hybrid và hybrid+HyDE trên golden dataset.
5. ChromaDB local không được commit vì là artefact sinh lại được; khi clone repo cần chạy Task 4 để reindex.

## 13. Kết luận

Tôi đã hoàn thành pipeline từ chuẩn hóa dữ liệu đến retrieval cho Task 3–6. Kết quả gồm 8 tài liệu Markdown, 116 chunks có metadata `customer_role`, 116 vectors 1024 chiều trong ChromaDB, Semantic Search + HyDE và BM25 với cùng output contract. Toàn bộ 16 test thuộc phạm vi Task 3–6 đều pass và đầu ra sẵn sàng để nhóm tích hợp RRF, fallback và generation có citation.
