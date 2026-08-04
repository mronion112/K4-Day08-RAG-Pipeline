# RAG Evaluation Results

## Pipeline Config

- Embedding: `text-embedding-3-small` (OpenAI, 1536-dim)
- Chunking: `RecursiveCharacterTextSplitter` (800/100)
- Retrieval: Semantic + BM25 → RRF (k=60)
- Generation: `gpt-4o-mini`
- Datasets: 3 chính sách Shopee (trả hàng, vận chuyển, bảo mật)

## Overall Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.8667 |
| answer_relevancy | 0.8187 |
| context_recall | 0.9833 |
| context_precision | 0.9975 |

## Per-Question Scores

| ID | Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|----------|-------------|-----------------|---------------|------------------|
| gd-01 | Người mua có thể gửi yêu cầu trả hàng/hoàn tiền tr... | 1.000 | 0.930 | 1.000 | 1.000 |
| gd-02 | Trả hàng COM (không còn nhu cầu) áp dụng cho những... | 1.000 | 0.806 | 1.000 | 1.000 |
| gd-03 | Hạn mức trả hàng COM mỗi tháng đối với người dùng ... | 1.000 | 0.880 | 1.000 | 1.000 |
| gd-04 | Ai chịu chi phí vận chuyển chiều hoàn trả sản phẩm... | 1.000 | 0.835 | 1.000 | 1.000 |
| gd-05 | Tiền hoàn trả cho người mua có thể được chuyển vào... | 1.000 | 0.823 | 1.000 | 1.000 |
| gd-06 | Shopee thu thập những loại dữ liệu cá nhân cơ bản ... | nan | 0.884 | 0.667 | 1.000 |
| gd-07 | Chính sách bảo mật của Shopee có áp dụng cho trẻ e... | 1.000 | 0.888 | 1.000 | 1.000 |
| gd-08 | Người dùng muốn rút lại sự đồng ý xử lý dữ liệu cá... | 1.000 | 0.807 | 1.000 | 1.000 |
| gd-09 | Shopee có được chuyển dữ liệu cá nhân của người dù... | 1.000 | 0.882 | 1.000 | 1.000 |
| gd-10 | Đơn hàng có giá trị hàng hóa lớn hơn bao nhiêu thì... | 0.500 | 0.866 | 1.000 | 1.000 |
| gd-11 | Người bán nên chọn kênh vận chuyển nào cho các sản... | 0.500 | 1.000 | 1.000 | 1.000 |
| gd-12 | Giới hạn kích thước và cân nặng tối đa cho phương ... | 1.000 | 0.909 | 1.000 | 0.950 |
| gd-13 | Cách tính khối lượng quy đổi (khối lượng thể tích)... | 1.000 | 0.841 | 1.000 | 1.000 |
| gd-14 | Thời hạn khiếu nại đối với đơn giao không thành cô... | 1.000 | 0.781 | 1.000 | 1.000 |
| gd-15 | Mức bồi thường tối đa cho người bán khi hàng hóa b... | 0.000 | 0.000 | 1.000 | 1.000 |
| gd-16 | Chính Sách Trả Hàng và Hoàn Tiền của Shopee áp dụn... | 1.000 | 0.853 | 1.000 | 1.000 |
| gd-17 | Người bán có bao nhiêu thời gian để phản hồi khi n... | 1.000 | 0.853 | 1.000 | 1.000 |
| gd-18 | Người mua có thể yêu cầu hoàn trả sản phẩm mua tại... | 1.000 | 0.851 | 1.000 | 1.000 |
| gd-19 | Những nhóm hàng hóa nào không được vận chuyển bằng... | 0.800 | 0.857 | 1.000 | 1.000 |
| gd-20 | Người khiếu nại phải cung cấp bằng chứng cho Shope... | 0.667 | 0.827 | 1.000 | 1.000 |

## Worst Performers (bottom 3)

- **gd-15**: *Mức bồi thường tối đa cho người bán khi hàng hóa bị thất lạc/hư hỏng trong vận c...*
  - faithfulness=0.000, context_recall=1.000
- **gd-10**: *Đơn hàng có giá trị hàng hóa lớn hơn bao nhiêu thì không được hỗ trợ vận chuyển ...*
  - faithfulness=0.500, context_recall=1.000
- **gd-11**: *Người bán nên chọn kênh vận chuyển nào cho các sản phẩm cây cảnh?...*
  - faithfulness=0.500, context_recall=1.000

## Nhận Xét

- Pipeline hybrid (semantic + BM25) cho kết quả retrieval tốt
- Generation dùng gpt-4o-mini cho câu trả lời tiếng Việt tự nhiên
- Các câu hỏi lệch domain có thể có score thấp do thiếu context phù hợp
