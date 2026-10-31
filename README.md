# IT5410_CLIP_application

# CLIP Image Retrieval

Dự án tìm kiếm ảnh theo mô tả văn bản. Backend sử dụng FastAPI.
Bản hiện tại gồm API mock, schema metadata, TypeScript client và dữ liệu mẫu.
Các thành phần CLIP/FAISS, BM25 và frontend Next.js chưa được triển khai.

| Thành phần | Phiên bản |
|---|---|
| API | 2.0.0 |
| Metadata | 1.0.0 |

## 1. Cài đặt và khởi chạy

Sử dụng Python 3.11 hoặc 3.12. Thực hiện các lệnh tại thư mục chứa `pyproject.toml`.
Backend mock sử dụng dữ liệu có sẵn, không yêu cầu GPU, CUDA hoặc tài khoản API.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn clip_retrieval.backend.app:app --reload --port 8000
```

Với Python 3.11, thay py -3.12 bằng py -3.11 ở lệnh tạo virtualenv.
Các lệnh gọi trực tiếp Python trong virtualenv.

### Linux/macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m uvicorn clip_retrieval.backend.app:app --reload --port 8000
```

| Đường dẫn | Chức năng |
|---|---|
| http://localhost:8000/docs | Tra cứu và gọi thử API |
| http://localhost:8000/health | Trạng thái dịch vụ |
| http://localhost:8000/openapi.json | Đặc tả OpenAPI |

Health trong chế độ mock trả `data`.`mode=mock`, `data`.model_loaded=false và `data`.index_loaded=false.

### Dependency

| Tệp | Nội dung |
|---|---|
| `requirements.txt` | Dependency chạy ứng dụng |
| `requirements-dev.txt` | Dependency chạy ứng dụng và công cụ kiểm thử |
| `requirements-lock.txt` | Phiên bản dependency dùng cho cài đặt và CI |
| `requirements-ml.txt` | Dependency CLIP/FAISS cho phần ML; cài thêm khi build hoặc chạy engine thật |

Bản hiện tại đã được kiểm tra trên Windows/Python 3.12. Kết quả Linux/Python 3.12
được ghi nhận cho bản nền tảng API 1.0.0. CI được cấu hình cho Python 3.11 và 3.12.

## 2. Gọi API

Ví dụ PowerShell:

```powershell
$body = @{ query = 'a dog on the beach'; method = 'clip'; top_k = 3; offset = 0; limit = 2 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/search' -ContentType 'application/json' -Body $body
Invoke-RestMethod 'http://localhost:8000/images/demo:001'
```

| Nội dung | Quy định |
|---|---|
| Phản hồi thành công | success, statusCode, `data`, timestamp |
| Phản hồi lỗi | success, statusCode, message[], path, timestamp |
| Mã trạng thái | statusCode bằng mã HTTP. Validation trả 422 |
| Thời gian | UTC ISO 8601, ba chữ số phần mili giây, hậu tố Z |
| Header | X-Request-ID, X-API-Version. Phản hồi JSON có Cache-Control: no-store |
| Payload | Dữ liệu nghiệp vụ nằm trong `data` |

Client chuyển từ API 1.0.0 sang 2.0.0 đọc payload tại `response.data`.
Chi tiết endpoint, schema và mã trạng thái: [Đặc tả API](docs/03_ENDPOINT_CONTRACTS.md).

## 3. Cấu trúc dự án

| Đường dẫn | Nội dung |
|---|---|
| `src/clip_retrieval/contracts.py` | Model Pydantic và schema dùng chung |
| src/clip_retrieval/backend/ | Backend FastAPI mock |
| `src/clip_retrieval/retrieval.py` | Giao diện `RetrievalEngine` |
| contracts/ | OpenAPI, JSON Schema, TypeScript client và ví dụ |
| frontend/ | Thư mục frontend, chưa có ứng dụng Next.js |
| data/examples/ | Metadata và 4 ảnh SVG mẫu |
| data/raw/ | Dữ liệu nguồn |
| data/processed/ | Dữ liệu đã chuyển đổi |
| artifacts/ | Embedding, chỉ mục và tệp ánh xạ |
| scripts/ | Xuất schema và kiểm tra metadata |
| tests/ | Kiểm thử API và schema |
| `.github/workflows/ci.yml` | Cấu hình kiểm tra tự động |
| docs/ | Tài liệu kỹ thuật |

## 4. Kiểm tra trước khi tạo pull request

Chạy trong môi trường Python đã cài dependency của dự án:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/export_contracts.py --check
python scripts/validate_metadata.py data/examples/metadata.jsonl --data-root data/examples
```

Khi sửa model dữ liệu, xuất lại schema bằng `python scripts/export_contracts.py`.
Cập nhật TypeScript client, ví dụ, kiểm thử và tài liệu trong cùng pull request.

## 5. Tài liệu

- [Môi trường phát triển và quy ước mã nguồn](docs/01_SETUP.md)
- [Đặc tả dữ liệu](docs/02_DATA_API_CONTRACT.md)
- [Đặc tả API](docs/03_ENDPOINT_CONTRACTS.md)
- [Đặc tả ánh xạ dataset](docs/04_DATASET_MAPPING.md)
- [Đặc tả tích hợp ML và backend](docs/05_ML_INTEGRATION_CONTRACT.md)
- [Quy ước lưu trữ và chia sẻ dữ liệu](docs/06_SHARED_DATA.md)
