# Đặc tả metadata và giao diện dữ liệu

## 1. Cấu trúc metadata

Metadata sử dụng JSONL, mã hóa UTF-8. Mỗi dòng mô tả một ảnh và toàn bộ caption của ảnh đó.

| Trường | Kiểu | Bắt buộc | Mặc định | Ràng buộc |
|---|---|---|---|---|
| `schema_version` | string | Không | 1.0.0 | Giá trị 1.0.0 |
| id | string | Có | — | Bằng `dataset:source_id`; duy nhất trong tập dữ liệu; tối đa 160 ký tự |
| dataset | string | Có | — | Khớp biểu thức `^[a-z0-9][a-z0-9_-]*$` |
| `source_id` | string | Có | — | 1–128 ký tự; giữ nguyên số 0 ở đầu |
| split | string | Có | — | train, val, test hoặc unspecified |
| `image_path` | string | Có | — | Đường dẫn POSIX tương đối dưới thư mục gốc dữ liệu |
| `thumbnail_path` | string hoặc null | Không | null | Cùng quy tắc với `image_path` |
| `width` | integer | Có | — | Chiều rộng ảnh gốc, lớn hơn 0 |
| `height` | integer | Có | — | Chiều cao ảnh gốc, lớn hơn 0 |
| `captions` | array[Caption] | Có | — | Tối thiểu một phần tử |
| license | string hoặc null | Không | null | Giấy phép đã xác định của nguồn dữ liệu |
| source_url | string hoặc null | Không | null | URL truy nguyên nguồn |

Schema Caption:

| Trường | Kiểu | Bắt buộc | Ràng buộc |
|---|---|---|---|
| text | string | Có | 1–10.000 ký tự; chứa ký tự khác khoảng trắng |
| language | string | Có | en, vi hoặc und |

Ràng buộc dữ liệu:

1. ID phải khớp mẫu `^[a-z0-9][a-z0-9_-]*:[A-Za-z0-9_-]+$`.
2. `image_path` và `thumbnail_path` không chứa dấu `\`, dấu `:`, thành phần rỗng, `.` hoặc `..`; không bắt đầu bằng `/`.
3. `width` và `height` phải là số nguyên JSON; chuỗi và boolean không hợp lệ.
4. Thuộc tính ngoài schema bị từ chối.
5. license chưa xác định phải bằng null; language chưa xác định sử dụng und.
6. `source_id` là định danh tại nguồn, độc lập với vị trí vector hoặc định danh số trong FAISS.

## 2. Kiểm tra dữ liệu

| Công cụ | Nội dung kiểm tra |
|---|---|
| `contracts/metadata.schema.json` | Kiểu dữ liệu, trường bắt buộc và ràng buộc cấu trúc |
| Pydantic Metadata | Ràng buộc schema, quan hệ `id=dataset:source_id` và an toàn đường dẫn |
| `scripts/validate_metadata.py` | Tính hợp lệ từng bản ghi, ID trùng lặp |
| validate_metadata.py với `--data-root` | Sự tồn tại của tệp và phạm vi đường dẫn dưới thư mục gốc |

```bash
python scripts/validate_metadata.py data/examples/metadata.jsonl --data-root data/examples
```

JSON Schema kiểm tra cấu trúc. Pydantic kiểm tra quan hệ giữa các trường và tính hợp lệ của đường dẫn.
Validator hiện chưa giải mã ảnh. Bước chuyển đổi dataset phải kiểm tra khả năng giải mã
và đối chiếu width/height với ảnh gốc.

Dữ liệu mẫu: `data/examples/metadata.jsonl`. Namespace: `demo`.

## 3. Ánh xạ dữ liệu nguồn

| Trường đích | COCO `captions` | Flickr30k |
|---|---|---|
| `source_id` | images[].id chuyển sang string | Tên tệp không gồm phần mở rộng |
| `image_path` | `images[].file_name` và thư mục split | Đường dẫn ảnh tương đối |
| `captions` | Gom annotation theo `image_id` | Gom chú thích theo tên tệp ảnh |
| `width`, `height` | Đối chiếu metadata với ảnh đã giải mã | Đọc kích thước ảnh đã giải mã |
| split | Split của bản phát hành được lựa chọn | Split theo tệp phân chia được lựa chọn |

Trước chuyển đổi, xác định bản phát hành, bộ chia dữ liệu và điều kiện sử dụng.
Nếu tích hợp nhiều bản phát hành, namespace phải phân biệt từng bản phát hành.
Split dùng để đánh giá phải tuân theo tập phân chia của bộ dữ liệu đã lựa chọn.

## 4. Dữ liệu đầu vào cho BM25

BM25 lập một document cho mỗi ảnh từ toàn bộ caption. Kết quả tham chiếu ID ảnh,
không tạo bản ghi kết quả riêng cho từng caption.
Giao diện sử dụng caption đầu tiên làm mô tả ngắn.

## 5. Bộ tệp bàn giao cho CLIP/FAISS

Đầu vào lập chỉ mục gồm `metadata.jsonl`, ảnh và thumbnail nếu có.
Bộ artifact được quản lý theo phiên bản:

| Tệp | Định dạng / nội dung |
|---|---|
| `embeddings.npy` | `float32`, `shape` (N, 512) đối với ViT-B/32; mỗi hàng được chuẩn hóa L2 |
| `image_ids.json` | Array gồm N ID ảnh; phần tử i ứng với hàng i của embeddings |
| `index.faiss` | Chỉ mục inner product trên vector đã chuẩn hóa |
| `manifest.json` | Phiên bản artifact; định danh encoder, weights, tokenizer, preprocess; dimension, N và checksums |

Ràng buộc tích hợp:

1. Tập ảnh và thứ tự vector phải khớp với `image_ids.json`.
2. Query encoder và image encoder phải sử dụng cấu hình mô hình tương thích.
3. Khi dùng `IndexIDMap`, phải có ánh xạ định danh số FAISS sang ID ảnh.
4. Chỉ mục của các encoder khác nhau phải được quản lý riêng, bao gồm trường hợp cùng dimension.
5. Caption được quản lý trong metadata, độc lập với tensor embedding ảnh.

## 6. Giao diện `RetrievalEngine`

Giao diện Python: `src/clip_retrieval/retrieval.py`.

| Thành phần | Quy định |
|---|---|
| `RankedImage`.`image_id` | ID ảnh theo metadata |
| `RankedImage`.score | Điểm xếp hạng |
| `RetrievalEngine`.`index_version` | Phiên bản chỉ mục |
| search_text(request) | Trả kết quả đã lọc, xếp hạng và giới hạn `top_k` trước phân trang |
| search_similar(`image_id`, request) | Cùng quy tắc với search_text; loại ảnh nguồn trước `top_k` |

## 7. Điểm tìm kiếm

| Phương pháp | score_type | score | similarity |
|---|---|---|---|
| CLIP | cosine_similarity | Tích vô hướng của hai vector đã chuẩn hóa L2 | Bằng score; thuộc [-1, 1] |
| BM25 | bm25_score | Điểm BM25 | null |

Điểm CLIP là cosine similarity, không phải xác suất.
Điểm BM25 và CLIP thuộc hai thang đo khác nhau. So sánh chất lượng bằng
Recall hoặc MRR.

## 8. Tài liệu tham chiếu

- [FastAPI response models](https://fastapi.tiangolo.com/tutorial/response-model/)
- [FastAPI additional responses](https://fastapi.tiangolo.com/advanced/additional-responses/)
- [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)
- [OpenAI CLIP](https://github.com/openai/CLIP)

Model Pydantic là nguồn xuất JSON Schema và OpenAPI.
Thay đổi schema phải cập nhật client, tài liệu và kiểm thử tương ứng.

Quy tắc chuyển đổi theo dataset: [Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md).

Đầu vào, artifact, manifest, lỗi và điều kiện nghiệm thu ML: [Đặc tả tích hợp ML và backend](05_ML_INTEGRATION_CONTRACT.md).
