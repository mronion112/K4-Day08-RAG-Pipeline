# BÁO CÁO CÁ NHÂN — TASK 10 VÀ STREAMLIT UI

## Thông tin cá nhân

- **Họ và tên:** Đào Kiều Thịnh Quang
- **MSSV:** 2A202601014
- **Phần việc được giao:** Task 10 — Generation có Citation và giao diện Streamlit Chatbot
- **Các file phụ trách:** `src/task10_generation.py` và tích hợp/kiểm tra `app.py`

## 1. Task được nhóm giao

Phần việc cá nhân gồm hai nội dung:

1. Hoàn thiện **Task 10 — Generation có Citation**:
   - nhận các chunks từ retrieval pipeline;
   - sắp xếp lại chunks để hạn chế hiện tượng *lost in the middle*;
   - tạo context chứa metadata nguồn và citation label;
   - gọi LLM để sinh câu trả lời dựa hoàn toàn trên evidence;
   - từ chối trả lời khi không có đủ evidence.
2. Tích hợp Task 10 vào **Streamlit UI** để người dùng nhập câu hỏi, xem câu trả lời, nguồn tham khảo và retrieval score.

## 2. Nội dung đã implement

### 2.1. Task 10 — `src/task10_generation.py`

#### `reorder_for_llm(chunks)`

Hàm nhận danh sách chunks đã xếp theo score giảm dần và đổi thứ tự theo mẫu:

```text
[1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
```

Chunk quan trọng nhất được giữ ở đầu, chunk quan trọng thứ hai được đưa về cuối. Hàm tạo list mới và không sửa dữ liệu đầu vào.

#### `format_context(chunks)`

Mỗi chunk được format với các trường:

- `title`;
- `source`;
- loại tài liệu;
- URL nguồn;
- citation label được phép sử dụng;
- nội dung evidence.

Năm của citation chỉ được lấy từ metadata như `document_version`, `published_at`, `retrieved_at` hoặc `date_crawled`. Nếu không có năm kiểm chứng được, hệ thống sử dụng `không rõ năm` thay vì tự đoán.

#### `generate_with_citation(query, top_k, context_chunks)`

Luồng xử lý chính:

```text
Query
  -> Task 9 retrieve(query, top_k)
  -> lọc chunks rỗng
  -> reorder_for_llm
  -> format_context
  -> tạo prompt
  -> gọi OpenAI/OpenRouter
  -> trả về answer + sources + retrieval_source
```

Contract trả về cho UI:

```python
{
    "answer": "Câu trả lời có citation",
    "sources": [...],
    "retrieval_source": "hybrid"
}
```

Tham số generation:

| Tham số | Giá trị mặc định | Mục đích |
|---|---:|---|
| `top_k` | 5 | Đủ evidence nhưng không làm prompt quá dài |
| `temperature` | 0.3 | Ưu tiên câu trả lời ổn định, ít sáng tạo ngoài nguồn |
| `top_p` | 0.9 | Giữ câu trả lời tự nhiên trong phạm vi xác suất hợp lý |
| `max_tokens` | 800 | Giới hạn chi phí và độ dài output |
| OpenAI model | `gpt-4o-mini` | Phù hợp chatbot RAG, tốc độ nhanh và chi phí thấp |

Provider được chọn bằng `LLM_PROVIDER=auto|openai|openrouter`. Chế độ `auto` ưu tiên `OPENAI_API_KEY`; nếu không có mới sử dụng `OPENROUTER_API_KEY`.

#### Kiểm soát hallucination

System prompt yêu cầu:

- chỉ sử dụng context được cung cấp;
- mỗi phát biểu thực tế phải có citation;
- không dùng citation ngoài danh sách được cung cấp;
- không tiết lộ retrieval score trong câu trả lời;
- không đủ evidence thì trả lời: `Tôi không thể xác minh thông tin này từ nguồn hiện có.`

#### Fallback phục vụ phát triển độc lập

Trong thời gian Task 4–9 chưa hoàn thiện, Task 10 có lexical fallback chỉ đọc các file Markdown trong `data/standardized`. Fallback chỉ chạy khi `Task 9.retrieve()` ném `NotImplementedError`; các lỗi runtime của pipeline thật không bị che giấu. Khi Task 9 hoàn thiện, Task 10 tự động sử dụng pipeline chính mà không cần thay đổi UI.

### 2.2. Streamlit UI — `app.py`

UI đã có:

- khung hội thoại bằng `st.chat_message` và `st.chat_input`;
- lưu và hiển thị lịch sử tin nhắn bằng `st.session_state`;
- năm câu hỏi gợi ý trong sidebar;
- slider chọn `top_k` từ 3 đến 10;
- gọi trực tiếp `generate_with_citation(query, top_k=top_k)`;
- spinner trong thời gian retrieval và generation;
- hiển thị câu trả lời Markdown;
- expander hiển thị source document, loại tài liệu, score và trích đoạn evidence;
- xử lý riêng `NotImplementedError` và lỗi runtime để UI không crash.

Lịch sử hiện được giữ trong phiên Streamlit để người dùng xem lại hội thoại. Việc dùng toàn bộ lịch sử làm context cho câu hỏi follow-up sẽ được nối thêm sau khi nhóm thống nhất contract hội thoại của pipeline chung.

## 3. Lý do lựa chọn giải pháp

### Document reordering

LLM thường chú ý tốt hơn tới đầu và cuối prompt, trong khi thông tin ở giữa dễ bị bỏ qua. Đặt hai chunks quan trọng nhất ở hai biên giúp evidence có score cao được ưu tiên mà không phải loại bỏ các chunks còn lại.

### Citation label lấy từ metadata

Việc tạo sẵn citation label trong context giúp LLM dùng đúng tên tài liệu và năm có thể kiểm chứng. Cách này an toàn hơn việc chỉ yêu cầu LLM tự tạo citation từ nội dung tự do.

### Temperature thấp

Chatbot chính sách cần độ chính xác hơn tính sáng tạo. `temperature=0.3` giảm biến động giữa các lần gọi và hạn chế việc bổ sung thông tin không có trong tài liệu.

### Dependency injection qua `context_chunks`

Tham số này cho phép unit test Task 10 bằng dữ liệu giả, không cần API embedding, ChromaDB hay Task 9. Trong production, bỏ trống tham số để dùng retrieval pipeline thật.

### Provider routing

Hỗ trợ cả OpenAI và OpenRouter giúp code dùng được với API key khác nhau của từng thành viên. Chế độ `auto` ưu tiên OpenAI để một placeholder OpenRouter không chặn API key OpenAI hợp lệ.

## 4. Kết quả pytest Task 10

Lệnh kiểm tra chính thức theo rubric:

```bash
.venv/bin/pytest tests/test_individual.py::TestTask10 -v -rs
```

Kết quả:

```text
tests/test_individual.py::TestTask10::test_format_context_includes_source PASSED
tests/test_individual.py::TestTask10::test_generate_returns_dict_with_answer PASSED
tests/test_individual.py::TestTask10::test_reorder_function_exists PASSED

============================== 3 passed in 3.45s ===============================
```

## 5. Lỗi gặp phải và cách khắc phục

### Task 9 chưa được implement

- **Hiện tượng:** UI chỉ hiện cảnh báo Task 10 chưa chạy hoặc không có context đầu vào.
- **Cách khắc phục:** thêm lexical fallback đọc `data/standardized` để Task 10 và UI có thể phát triển độc lập. Fallback tự ngừng sử dụng khi Task 9 hoạt động.

### Có `OPENAI_API_KEY` nhưng code ưu tiên OpenRouter

- **Hiện tượng:** người dùng chỉ có OpenAI key nhưng provider mặc định có thể chọn sai OpenRouter.
- **Cách khắc phục:** thêm provider routing; chế độ `auto` ưu tiên OpenAI key hợp lệ và bỏ qua placeholder API key.

### Không đủ evidence nhưng LLM vẫn có khả năng suy đoán

- **Hiện tượng:** nếu vẫn gọi LLM với context rỗng, model có thể trả lời bằng kiến thức nền.
- **Cách khắc phục:** kiểm tra evidence trước API call; nếu không có chunk hợp lệ thì trả về thông báo không thể xác minh và không gọi LLM.

### Citation có thể chứa năm không có trong tài liệu

- **Hiện tượng:** tự dùng năm hiện tại sẽ tạo citation không được nguồn chứng minh.
- **Cách khắc phục:** chỉ lấy năm từ metadata; thiếu năm thì ghi `không rõ năm`.

### Dependency giữa RAGAS, LangChain và OpenAI

- **Hiện tượng:** cài các phiên bản mới nhất gây pip backtracking và xung đột.
- **Cách khắc phục:** dùng bộ tương thích gồm RAGAS 0.1.21, LangChain 0.2.x và OpenAI SDK 1.109.1; xác nhận bằng `pip check` và import smoke test.

## 6. Cách phần việc nối vào pipeline chung

Task 10 là bước cuối sau retrieval:

```text
Task 4: Chunk + ChromaDB
  -> Task 5: Semantic Search
  -> Task 6: BM25
  -> Task 7: RRF Reranking
  -> Task 8: PageIndex Fallback
  -> Task 9: Retrieval Pipeline
  -> Task 10: Reorder + Context + LLM + Citation
  -> app.py: Chat UI + Sources
```

Điểm nối chính là:

```python
chunks = retrieve(query, top_k=top_k)
```

Task 10 không phụ thuộc cách Task 9 thực hiện semantic search, BM25 hay fallback, miễn Task 9 trả về danh sách theo contract:

```python
{
    "content": str,
    "score": float,
    "metadata": dict,
    "source": str
}
```

Sau khi nhận kết quả, Task 10 tạo câu trả lời và trả lại nguyên danh sách sources cho `app.py`. UI dùng `answer` để hiển thị phản hồi và dùng `sources` để hiển thị evidence, metadata và score. Thiết kế này tách biệt retrieval, generation và presentation nên việc merge Task 1–9 chỉ cần giữ đúng contract dữ liệu.

## Kết luận

Phần Task 10 đã đáp ứng document reordering, context formatting, citation, no-evidence handling, provider routing và output contract cho Streamlit. Pytest chính thức của Task 10 đạt **3/3**, và Streamlit AppTest khởi chạy không có exception.
