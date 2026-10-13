# Tích hợp frontend

## Trạng thái

Frontend Next.js thuộc hạng mục 1.5.1, phụ trách Thành viên 2.
Thư mục hiện chưa có ứng dụng.

## Giao diện backend

| Thuộc tính | Giá trị |
|---|---|
| Base URL | http://localhost:8000 |
| Phiên bản API | 2.0.0 |
| Chế độ hiện tại | mock |
| TypeScript client | ../contracts/client.ts |
| Đặc tả API | ../docs/03_ENDPOINT_CONTRACTS.md |

Các hàm client: `searchImages`, `getHealth`, `getImage`, `getSimilarImages`.
Các hàm trả `SuccessResponse<T>`. Payload nằm trong `data`.
Phản hồi lỗi được biểu diễn bằng `ApiError` gồm status, payload và `requestId`.

## Quy tắc tích hợp

- Sử dụng type và hàm gọi API từ `contracts/client.ts`.
- Phân giải `image_url` và `thumbnail_url` theo Base URL bằng hàm `imageUrl`.
- Giữ các tham số truy vấn khi tải trang tiếp theo. Đặt lại kết quả khi `index_version` thay đổi.
- Chế độ BM25 không sử dụng `min_similarity`.
- Frontend chỉ gọi HTTP API. Mô hình ML và mã Python được triển khai ở backend.
- Cấu hình `CLIP_CORS_ORIGINS` tại backend khi origin frontend thay đổi.

Chi tiết: [Đặc tả API](../docs/03_ENDPOINT_CONTRACTS.md).
