# Môi trường phát triển và quy ước mã nguồn

## 1. Môi trường

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.11 hoặc 3.12 |
| Môi trường Python | Virtualenv riêng cho dự án |
| Dependency | Cài từ `requirements-lock.txt` |
| GPU/CUDA | Không yêu cầu đối với backend mock |
| Dataset | Sử dụng data/examples trong chế độ mock |
| Git | Dùng để quản lý mã nguồn và bàn giao |

Lệnh cài đặt và khởi chạy: [README](../README.md).

## 2. Cấu hình

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| CLIP_MODE | mock | Chế độ thực thi. Giá trị real chưa được hỗ trợ |
| `CLIP_CORS_ORIGINS` | http://localhost:3000,http://127.0.0.1:3000 | Danh sách origin cho phép, phân cách bằng dấu phẩy |

`.env.example` cung cấp giá trị tham khảo. Ứng dụng đọc biến môi trường của tiến trình,
không tự nạp tệp `.env`. Thay đổi cấu hình có hiệu lực sau khi khởi động lại server.

PowerShell:

```powershell
$env:CLIP_CORS_ORIGINS='http://localhost:3000'
```

Linux/macOS:

```bash
export CLIP_CORS_ORIGINS=http://localhost:3000
```

## 3. Quy ước Python

- Module, biến và hàm: `snake_case`. Lớp: `PascalCase`.
- Giao diện và model trao đổi giữa các thành phần phải có type hint.
- Sử dụng Ruff để lint và định dạng mã. Độ dài dòng tối đa: 100 ký tự.
- OpenAPI và JSON Schema được xuất từ model trong `src/clip_retrieval/contracts.py`.
- Định danh ảnh công khai sử dụng trường id theo schema Metadata.
- Chế độ mock phải trả `mode=mock`. Ứng dụng từ chối `CLIP_MODE=real` khi chưa tích hợp mô hình và chỉ mục.

## 4. Quản lý thay đổi

| Nội dung | Quy định |
|---|---|
| Nhánh chính | main phải khởi chạy được và vượt qua các bước kiểm tra |
| Nhánh chức năng | feat/<task>-<description> |
| Nhánh sửa lỗi | fix/<description> |
| Nhánh tài liệu | docs/<description> |
| Tiền tố commit | feat:, fix:, docs:, test:, chore: |
| Pull request | Tạo vào dev, nêu thay đổi và kết quả kiểm tra, có ít nhất một thành viên approve |
| Đưa lên main | Chỉ qua pull request từ dev; leader (@hoangtung386, khai báo trong `.github/CODEOWNERS`) review, approve và merge |
| Kiểm tra bắt buộc | `test (3.11)`, `test (3.12)` theo `.github/workflows/ci.yml`; pull request vào main thêm `source-is-dev` theo `.github/workflows/main-source-guard.yml` |

Pull request thay đổi schema phải bao gồm model, OpenAPI/JSON Schema, TypeScript client,
ví dụ, tài liệu và kiểm thử liên quan. Thay đổi không tương thích phải
có phiên bản API mới. Metadata và API được quản lý phiên bản độc lập.

Credential, .env, virtualenv, dataset và tệp model/index lớn không thuộc phạm vi lưu trữ trong Git.
Đường dẫn trong metadata phải tương đối với thư mục gốc dữ liệu.
Branch protection của main và dev áp dụng cả với admin: cấm push trực tiếp, force-push và xóa nhánh;
chỉ leader được merge vào main.
