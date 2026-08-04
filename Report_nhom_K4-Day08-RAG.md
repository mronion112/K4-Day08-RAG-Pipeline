# BÁO CÁO NHÓM - K4 DAY 08: RAG PIPELINE HỖ TRỢ E-COMMERCE

## 1. Giới thiệu

Trong buổi thực hành ngày 08, nhóm thực hiện xây dựng một hệ thống Retrieval-Augmented Generation (RAG) nhằm hỗ trợ người dùng tra cứu thông tin liên quan đến chính sách và hướng dẫn của Shopee Việt Nam. Hệ thống cho phép người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên, truy xuất các đoạn văn bản phù hợp từ kho dữ liệu chuẩn hóa và trả lời dựa trên tài liệu thật, có kèm citation và nguồn tham khảo.

Mục tiêu của nhóm không chỉ là hoàn thành một demo chạy được, mà còn rèn luyện cách làm việc theo pipeline chuyên nghiệp: thu thập dữ liệu, chuẩn hóa dữ liệu, tạo chỉ mục, truy xuất, rerank, sinh câu trả lời và xây dựng giao diện demo.

## 2. Thông tin nhóm

- Tên đề tài: Xây dựng hệ thống RAG hỗ trợ khách hàng Shopee Vietnam
- Mục tiêu chính: Xây dựng pipeline truy xuất tri thức từ các tài liệu pháp lý và bài hướng dẫn, hỗ trợ trả lời câu hỏi bằng câu trả lời có căn cứ và có nguồn.
- Phương thức làm việc: Chia nhỏ các nhiệm vụ thành các task riêng biệt, sau đó tích hợp lại thành một hệ thống end-to-end.

## 3. Phân công công việc

| Thành viên | Vai trò | Nội dung chính |
|---|---|---|
| Trần Quang Minh | Team Leader / Data Engineer | Thu thập dữ liệu đầu vào, ghép pipeline, kiểm thử toàn hệ thống |
| Ngô Văn Nam | Data Standardization & Retrieval Developer | Task 3, 4, 5, 6: chuẩn hóa dữ liệu, chunking, indexing, semantic/BM25 search |
| Nguyễn Hoàng Anh | Retrieval & Reranking Engineer | Task 7, 8, 9: RRF, PageIndex fallback, pipeline retrieval |
| Đào Kiều Thịnh Quang | Generation & UI Engineer | Task 10: generation có citation, giao diện Streamlit |
| Phạm Khắc Khương Duy | Thành viên hỗ trợ | Hỗ trợ tổng hợp nội dung báo cáo và đóng góp ý tưởng cho báo cáo nhóm |

## 4. Mục tiêu chung của nhóm

Nhóm đặt ra bốn mục tiêu chính như sau:

1. Xây dựng kho tri thức từ các tài liệu đầu vào có chất lượng tốt và dễ truy xuất.
2. Triển khai hệ thống retrieval có thể tìm đúng tài liệu liên quan tới câu hỏi của người dùng.
3. Tạo quá trình generation có kiểm soát, tránh hallucination bằng cách chỉ dùng evidence từ tài liệu.
4. Tích hợp hệ thống vào giao diện demo thân thiện để trình bày và kiểm thử trực quan.

## 5. Kiến trúc hệ thống

Pipeline được triển khai theo các bước sau:

```text
Người dùng nhập câu hỏi
   ↓
Task 1–2: thu thập dữ liệu đầu vào
   ↓
Task 3: chuẩn hóa tài liệu sang Markdown
   ↓
Task 4: chunking + embedding + indexing vào ChromaDB
   ↓
Task 5: semantic search
   ↓
Task 6: lexical search (BM25)
   ↓
Task 7: reranking bằng RRF
   ↓
Task 8: fallback PageIndex vectorless
   ↓
Task 9: retrieval pipeline hoàn chỉnh
   ↓
Task 10: generation có citation
   ↓
Streamlit UI: hiển thị câu trả lời và nguồn tham khảo
```

### 5.1 Vai trò từng bước trong pipeline

- Thu thập dữ liệu: tạo nguồn tri thức ban đầu từ tài liệu pháp lý và bài hướng dẫn.
- Chuẩn hóa dữ liệu: chuyển tài liệu về định dạng đồng nhất, hỗ trợ parse metadata và chunking.
- Chunking và indexing: chia tài liệu thành các đoạn nhỏ, tạo embedding và lưu vào vector store.
- Retrieval: tìm kiếm các đoạn liên quan bằng nhiều chiến lược khác nhau.
- Reranking: sắp xếp lại kết quả để ưu tiên các đoạn có liên quan nhất.
- Generation: dùng các đoạn tìm được để sinh câu trả lời có citation.
- UI: hiển thị câu trả lời, source và các thông tin hỗ trợ cho người dùng.

## 6. Các công việc đã thực hiện

### 6.1 Thu thập và chuẩn hóa dữ liệu

Nhóm đã thu thập các tài liệu chính sách và tài liệu hướng dẫn làm nguồn dữ liệu nền cho hệ thống. Các tài liệu này được chuyển về định dạng Markdown chuẩn hóa để dễ dàng xử lý, phân loại và lưu metadata. Các metadata quan trọng như doc_id, title, source_url, customer_role, category và type được giữ lại nhằm hỗ trợ retrieval sau này.

### 6.2 Chunking, embedding và indexing

Sau khi chuẩn hóa dữ liệu, nhóm chia mỗi tài liệu thành các chunk nhỏ để giảm độ dài và tăng khả năng tối ưu cho tìm kiếm. Mỗi chunk được tạo embedding và đưa vào ChromaDB để phục vụ việc truy xuất theo ngữ nghĩa. Quá trình này giúp hệ thống có thể tìm các đoạn văn bản phù hợp với câu hỏi của người dùng thay vì chỉ dựa vào từ khóa đơn giản.

### 6.3 Retrieval đa chiến lược

Hệ thống sử dụng nhiều cách tiếp cận khác nhau để tăng độ tin cậy của retrieval:

- Semantic search: dùng ngữ nghĩa để tìm tài liệu phù hợp với câu hỏi.
- BM25: dùng từ khóa và khớp chính xác để tìm các đoạn có chứa các từ quan trọng.
- RRF: gộp các kết quả từ nhiều ranker để ưu tiên tài liệu phù hợp nhất.
- PageIndex fallback: khi kết quả retrieval không đủ tốt, hệ thống dùng phương án dự phòng để tiếp tục tìm kiếm.

### 6.4 Generation có citation

Sau khi lấy được các chunk phù hợp, hệ thống chuyển chúng vào module generation. Mỗi câu trả lời được xây dựng dựa trên evidence từ các chunk đã được retrieval. Ngoài ra, hệ thống còn gắn citation và nguồn tham khảo để người dùng có thể kiểm chứng. Khi không đủ evidence, hệ thống có thể từ chối trả lời thay vì suy đoán.

### 6.5 Giao diện demo

Nhóm cũng tích hợp giao diện bằng Streamlit để người dùng có thể nhập câu hỏi và xem kết quả trực tiếp. Giao diện hiển thị câu trả lời, danh sách nguồn và các thông tin hỗ trợ như score retrieval và metadata của tài liệu.

## 7. Kết quả đạt được

### 7.1 Về dữ liệu

- Đã chuẩn hóa đủ số lượng tài liệu đầu vào phục vụ demo.
- Tạo được bộ dữ liệu chunked và indexed để dùng cho retrieval.
- Giữ được metadata quan trọng để phục vụ lọc và truy xuất đúng tài liệu hơn.

### 7.2 Về retrieval

- Hệ thống có khả năng kết hợp semantic search và BM25 để tăng độ phủ sóng khi tìm câu trả lời.
- RRF giúp gộp kết quả từ nhiều phương thức một cách hợp lý.
- Fallback giúp hệ thống vẫn hoạt động tốt hơn khi một phương thức tìm kiếm không hiệu quả.

### 7.3 Về generation và UI

- Hệ thống trả lời được câu hỏi dựa trên tài liệu thật.
- Câu trả lời có citation và nguồn tham khảo rõ ràng.
- Giao diện demo dễ dùng và phù hợp cho mục đích trình bày bài tập.

## 8. Kết quả kiểm thử

Các báo cáo cá nhân của nhóm ghi nhận các kết quả kiểm thử quan trọng như sau:

- Task 3: hoàn thành và pass các test liên quan chuẩn hóa dữ liệu.
- Task 4: pass các test liên quan chunking, embedding và indexing.
- Task 5 và Task 6: pass các test retrieval theo semantic và lexical.
- Task 7, 8 và 9: có triển khai pipeline retrieval và kiểm thử ở mức cơ bản.
- Task 10: pass các test generation có citation và tích hợp UI.

Tổng thể, nhóm đã xây dựng được một nền tảng RAG end-to-end, đủ điều kiện để demo trong môi trường lab.

## 9. Những vấn đề phát sinh và cách xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| Chênh lệch format giữa các task | Mỗi thành viên dùng cấu trúc đầu ra khác nhau | Thống nhất schema đầu vào và đầu ra trước khi ghép pipeline |
| Fallback PageIndex chưa thật sự ổn định | Phụ thuộc vào môi trường API và dữ liệu đầu vào | Giữ fallback ở chế độ lazy để không làm crash toàn bộ hệ thống |
| Hallucination khi không đủ evidence | LLM có thể suy đoán từ kiến thức nền | Thêm kiểm tra evidence và chặn generation khi không đủ dữ liệu |
| Khác biệt thang điểm giữa semantic và BM25 | Semantic dùng cosine, BM25 dùng lexical score | Chỉ dùng RRF cho việc merge thứ hạng, giữ cosine score riêng cho fallback |
| Một số báo cáo cá nhân chưa được viết đầy đủ | Nội dung nguồn chưa được cập nhật đồng bộ | Nhóm tổng hợp nội dung từ workspace và ghi nhận các điểm cần bổ sung |

## 10. Đánh giá chung

Dự án đã đạt được những kết quả cơ bản và có ý nghĩa về mặt học thuật và thực hành. Nhóm đã hiểu rõ quy trình xây dựng một hệ thống RAG từ đầu đến cuối, từ khâu thu thập dữ liệu cho tới việc sinh câu trả lời có căn cứ. Đây là nền tảng tốt để tiếp tục cải thiện về độ chính xác, tốc độ xử lý và tính ổn định trong các dự án lớn hơn.

## 11. Hướng phát triển tiếp theo

1. Hoàn thiện các module còn chưa ổn định như PageIndex và reranking cross-encoder.
2. Tăng cường đánh giá bằng golden dataset và các metric chất lượng retrieval/generation.
3. Cải thiện prompt generation để câu trả lời ngắn gọn, chính xác và có citation rõ hơn.
4. Tối ưu hiệu năng pipeline cho môi trường triển khai thực tế.
5. Bổ sung nội dung báo cáo cá nhân đầy đủ hơn cho các thành viên để báo cáo nhóm có tính toàn diện hơn.
