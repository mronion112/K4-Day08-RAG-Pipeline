# Report Cá Nhân — Task 1 → 5
## Trần Quang Minh — 01210

---

## Task 1: Thu thập tài liệu pháp luật (≥3 file PDF)

### Cách làm
- Tải 3 file chính sách Shopee từ `shopee_policy.zip` (chứa sẵn file markdown)
- Convert từ markdown → PDF bằng Python (`markdown` + `PyMuPDF/fitz`)
- Lưu vào `data/landing/legal/`

### Tại sao
| Lựa chọn | Lý do |
|-----------|-------|
| **3 chính sách được chọn**: trả hàng/hoàn tiền, vận chuyển, bảo mật | Đây là 3 chủ đề khách hàng hỏi nhiều nhất trên sàn TMĐT. Đủ đa dạng để test RAG: chính sách ngắn (13K), vừa (29K), dài (56K) |
| **PDF thay vì DOCX** | PDF là format phổ biến nhất của tài liệu chính sách thực tế, giữ nguyên định dạng gốc |
| **Tên file tiếng Việt có dấu** | Giữ nguyên tên gốc từ Shopee, dễ trace nguồn |

### customer_role
| File | Role | Lý do |
|------|------|-------|
| `chinh-sach-tra-hang-va-hoan-tien.md` | `buyer` | Chỉ người mua mới yêu cầu trả hàng/hoàn tiền |
| `chinh-sach-van-chuyen-shopee.md` | `buyer` | Chính sách giao hàng cho người mua |
| `chinh-sach-bao-mat.md` | `both` | Áp dụng cho cả buyer và seller |

---

## Task 2: Crawl bài hướng dẫn (≥5 file)

### Cách làm
- Copy 5 file markdown có sẵn về chủ đề hướng dẫn người dùng Shopee (e-voucher, ShopeeFood, SPayLater, liên kết tài khoản ngân hàng) vào `data/landing/news/`

### Tại sao
| Lựa chọn | Lý do |
|-----------|-------|
| **Dùng file .md thay vì crawl** | Dữ liệu có sẵn từ zip, đảm bảo nội dung đầy đủ. Crawl Shopee gặp vấn đề SPA (JavaScript render) — hướng dẫn cũng đã cảnh báo điều này |
| **5 chủ đề được chọn** | Đa dạng: thanh toán (SPayLater, ví điện tử), đặt món (ShopeeFood), tài khoản (liên kết ngân hàng). Phủ nhiều câu hỏi thực tế |
| **Format .md** | Test chấp nhận `.json/.html/.md/.txt` — .md là đơn giản nhất, không cần parse JSON |

---

## Task 3: Convert sang Markdown

### Cách làm
- Copy trực tiếp file markdown từ zip vào `data/standardized/legal/` và `data/standardized/news/`
- Không cần dùng MarkItDown vì source đã là markdown

### Tại sao
| Lựa chọn | Lý do |
|-----------|-------|
| **Copy trực tiếp** | MarkItDown dùng để convert PDF→MD, nhưng source gốc đã là .md. Convert PDF→MD sẽ mất formatting (heading, table, list). Copy trực tiếp giữ nguyên chất lượng markdown |
| **Chỉ giữ 3 file legal demo** | 3 chính sách đã chọn ở Task 1, đủ để demo RAG pipeline hiệu quả |

---

## Task 4: Chunking, Embedding & Indexing

### Cách làm
1. **Load**: Đọc toàn bộ file `.md` từ `data/standardized/`, gán `customer_role` cho từng tài liệu
2. **Chunk**: `RecursiveCharacterTextSplitter` với `chunk_size=800`, `chunk_overlap=100`
3. **Embed**: OpenAI `text-embedding-3-small` (1536 dim), batch 100 chunks/lần
4. **Index**: ChromaDB (`PersistentClient`), collection `ecommerce_support_docs`, metric `cosine`

### Tại sao

| Lựa chọn | Lý do |
|-----------|-------|
| **Chunk size 800, overlap 100** | Lab gợi ý 800/100. Đủ lớn để giữ context của 1 điều khoản (~2-3 đoạn), overlap 100 tránh cắt ngang câu |
| **RecursiveCharacterTextSplitter** | An toàn nhất — cắt theo `\n\n` → `\n` → `. ` → ` ` → character. Phù hợp với markdown có heading và paragraph rõ ràng |
| **text-embedding-3-small (OpenAI)** | Nhẹ (không cần cài torch 2GB như sentence-transformers), hỗ trợ đa ngôn ngữ tốt (tiếng Việt), latency thấp qua API. 1536 dim đủ cho semantic search chính xác |
| **ChromaDB** | Local persistent, không cần Docker, đơn giản nhất để bắt đầu. Hỗ trợ cosine similarity và metadata filtering |
| **Batch embed 100** | OpenAI API có rate limit, batch 100 tối ưu giữa latency và số lần gọi API |
| **customer_role trong metadata** | Yêu cầu bắt buộc từ Lab 07. Cho phép filter kết quả theo buyer/seller sau này — tránh trả lời sai đối tượng |

### Kết quả
- 8 documents → **145 chunks**
- Index vào ChromaDB `chroma_db/`
- Tất cả chunk có `customer_role` metadata

---

## Task 5: Semantic Search

### Cách làm
- Dùng chung `get_embedding_model()` và `get_collection()` từ Task 4
- Embed query → query ChromaDB bằng cosine distance → convert distance sang similarity score (`1 - distance`)

### Tại sao
| Lựa chọn | Lý do |
|-----------|-------|
| **Dùng chung embed model với Task 4** | Bắt buộc — query và document phải cùng không gian vector. Nếu dùng model khác, dimension không khớp → lỗi |
| **Cosine similarity = 1 - distance** | ChromaDB trả về cosine distance (0→2), convert sang similarity (0→1) để dễ hiểu và so sánh |
| **Sorted descending** | Kết quả có similarity cao nhất (gần nghĩa nhất) lên đầu — chuẩn cho RAG pipeline |

---

## Task 6: Lexical Search (BM25)

### Cách làm
- Load toàn bộ markdown từ `data/standardized/`, chunk giống Task 4 (800/100)
- Build BM25 index từ tokenized corpus (lowercase + split whitespace)
- Query: tokenize → `get_scores()` → top_k indices

### Tại sao
| Lựa chọn | Lý do |
|-----------|-------|
| **BM25 thay vì TF-IDF** | BM25 xử lý document length normalization tốt hơn, là industry standard cho text search |
| **Dùng chunk thay vì full document** | Cùng granularity với semantic search → kết quả RRF sau này công bằng hơn |
| **Tokenize đơn giản (split)** | Tiếng Việt dùng khoảng trắng giữa các từ, split hoạt động tốt. Có thể nâng cấp lên `underthesea` nếu cần |
| **Score > 0 mới trả về** | Lọc bỏ chunk không liên quan, giảm noise cho pipeline |
