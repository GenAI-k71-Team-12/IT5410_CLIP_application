# Đặc tả API

| Thuộc tính | Giá trị |
|---|---|
| Hệ thống | CLIP Image Retrieval |
| Phiên bản API | 2.0.0 |
| Base URL | `http://localhost:8000` |
| Mã hóa | UTF-8 |
| Định dạng dữ liệu | JSON; riêng tài nguyên ảnh trả nội dung tệp |
| Xác thực | Không yêu cầu |
| Đặc tả OpenAPI | `contracts/openapi.json`; truy cập tại `/openapi.json` |
| Giao diện tra cứu | `/docs` |

## 1. Quy ước chung

1. Tên trường trong dữ liệu nghiệp vụ sử dụng `snake_case`. Envelope phản hồi sử dụng trường `statusCode`.
2. Mã trạng thái HTTP phải bằng giá trị `statusCode` trong phản hồi JSON.
3. Trường có giá trị `null` phải được giữ trong phản hồi.
4. Thuộc tính ngoài schema của body hoặc query bị từ chối với mã `422`.
5. Các endpoint GET sử dụng path và query để nhận tham số. Request body của GET không được xử lý.
6. Đường dẫn ảnh được phân giải theo Base URL. Phản hồi không chứa đường dẫn tệp nội bộ.
7. `/docs`, `/openapi.json` và CORS preflight giữ định dạng riêng.

### 1.1. Phản hồi thành công

| Trường | Kiểu | Giá trị / mô tả |
|---|---|---|
| success | boolean | `true` |
| statusCode | integer | `200` |
| `data` | object | Dữ liệu theo schema của endpoint |
| timestamp | string | Thời điểm tạo phản hồi; UTC, ISO 8601, ba chữ số phần mili giây, hậu tố `Z` |

```json
{
  "success": true,
  "statusCode": 200,
  "data": {
    "status": "ok",
    "mode": "mock",
    "ready": true,
    "model_loaded": false,
    "index_loaded": false,
    "api_version": "2.0.0"
  },
  "timestamp": "2026-10-13T03:50:00.000Z"
}
```

### 1.2. Phản hồi lỗi

| Trường | Kiểu | Giá trị / mô tả |
|---|---|---|
| success | boolean | `false` |
| statusCode | integer | Mã lỗi HTTP, từ `400` đến `599` |
| message | array[string] | Danh sách thông báo lỗi; tối thiểu một phần tử |
| path | string | Đường dẫn request, không bao gồm query string |
| timestamp | string | Cùng định dạng với phản hồi thành công |

```json
{
  "success": false,
  "statusCode": 422,
  "message": ["property extraField should not exist"],
  "path": "/search",
  "timestamp": "2026-10-13T03:50:00.000Z"
}
```

Thông báo validation có dạng `<vị trí>.<tên trường>: <thông báo>`.
Thuộc tính ngoài schema có thông báo `property <tên trường> should not exist`.
Lỗi nội bộ trả thông báo `Internal server error`; chi tiết ngoại lệ được ghi vào log.

### 1.3. Mã trạng thái

| Mã | Ý nghĩa | Điều kiện |
|---|---|---|
| 200 | OK | Xử lý thành công, bao gồm kết quả tìm kiếm rỗng |
| 304 | Not Modified | Tài nguyên ảnh không thay đổi theo điều kiện cache; không có body |
| 404 | Not Found | Route hoặc tài nguyên không tồn tại |
| 405 | Method Not Allowed | Phương thức HTTP không được hỗ trợ tại route |
| 415 | Unsupported Media Type | Thiếu hoặc sai Content-Type tại POST /search |
| 422 | Unprocessable Entity | Body, path hoặc query không hợp lệ |
| 500 | Internal Server Error | Lỗi xử lý nội bộ |
| 503 | Service Unavailable | Dành cho backend tích hợp ML khi chưa sẵn sàng. Chế độ mock chưa trả mã này |

## 2. Header

### 2.1. Request header

| Header | Áp dụng | Bắt buộc | Quy định |
|---|---|---|---|
| Accept | Endpoint JSON | Không | Giá trị khuyến nghị: `application/json` hoặc `*/*`. Server trả JSON và chưa hỗ trợ kiểm tra Accept |
| Content-Type | POST /search | Có | `application/json`; chấp nhận tham số `charset=utf-8` |
| If-None-Match | GET/HEAD /media/{path} | Không | ETag của tài nguyên đã nhận |
| If-Modified-Since | GET/HEAD /media/{path} | Không | Thời điểm sửa đổi tài nguyên đã nhận |

Endpoint GET không yêu cầu Content-Type. Hệ thống không yêu cầu Authorization.
X-Request-ID được cấp tại server; giá trị do client gửi không được sử dụng.
Server không từ chối các header ngoài danh sách trên.

### 2.2. Response header

| Header | Áp dụng | Quy định |
|---|---|---|
| Content-Type | Phản hồi có body | `application/json` cho dữ liệu/lỗi; `image/svg+xml` cho ảnh mẫu |
| X-Request-ID | JSON và tài nguyên ảnh, bao gồm lỗi | UUID được cấp mới cho mỗi request |
| X-API-Version | JSON và tài nguyên ảnh, bao gồm lỗi | `2.0.0` |
| Cache-Control | Phản hồi JSON, bao gồm lỗi | `no-store` |
| ETag | Phản hồi tài nguyên ảnh | Giá trị do StaticFiles xác định |
| Last-Modified | Phản hồi tài nguyên ảnh 200 | Thời điểm sửa đổi tệp |
| Allow | Phản hồi 405 | Danh sách phương thức được route hỗ trợ |

### 2.3. CORS

| Thuộc tính | Giá trị mặc định |
|---|---|
| Allowed origins | `http://localhost:3000`, `http://127.0.0.1:3000` |
| Allowed methods | GET, POST |
| Allowed request headers | Content-Type và các header được CORSMiddleware cho phép mặc định |
| Exposed response headers | X-Request-ID, X-API-Version |
| Cấu hình origins | Biến môi trường `CLIP_CORS_ORIGINS`; danh sách phân cách bằng dấu phẩy |

Access-Control-Allow-Origin được trả cho origin thuộc cấu hình cho phép.
OPTIONS preflight do CORSMiddleware xử lý; không áp dụng cấu trúc phản hồi JSON hoặc header X-Request-ID/X-API-Version.

## 3. Schema dữ liệu

### 3.1. ImageId

Kiểu `string`, độ dài tối đa 160 ký tự, biểu thức:

```text
^[a-z0-9][a-z0-9_-]*:[A-Za-z0-9_-]+$
```

Cấu trúc: `<dataset>:<source_id>`. Ví dụ: `demo:001`.
ID sai định dạng trả `422`. ID hợp lệ nhưng không tồn tại trả `404`.

### 3.2. Health

| Trường | Kiểu | Mô tả |
|---|---|---|
| status | string | `ok` hoặc `not_ready` |
| mode | string | `mock` hoặc `real` |
| ready | boolean | Trạng thái sẵn sàng của chế độ đang chạy |
| model_loaded | boolean | Trạng thái nạp mô hình |
| index_loaded | boolean | Trạng thái nạp chỉ mục |
| api_version | string | `2.0.0` |

### 3.3. `ImagePublic`

| Trường | Kiểu | Mô tả |
|---|---|---|
| id | ImageId | Định danh ảnh |
| dataset | string | Tên bộ dữ liệu |
| split | string | `train`, `val`, `test` hoặc `unspecified` |
| `width` | integer | Chiều rộng ảnh, lớn hơn 0 |
| `height` | integer | Chiều cao ảnh, lớn hơn 0 |
| `captions` | array[Caption] | Danh sách chú thích |
| `image_url` | string | URL ảnh |
| `thumbnail_url` | string | URL ảnh thu nhỏ |

`Caption` gồm `text: string` và `language: en | vi | und`.
`text` dài từ 1 đến 10.000 ký tự và phải chứa ký tự khác khoảng trắng.

### 3.4. `SearchResponse`

| Trường | Kiểu | Mô tả |
|---|---|---|
| request_id | string | Trùng với header X-Request-ID |
| api_version | string | `2.0.0` |
| mode | string | `mock` hoặc `real` |
| method | string | `clip` hoặc `bm25` |
| score_type | string | `cosine_similarity` hoặc `bm25_score` |
| query | string hoặc null | Nội dung truy vấn; null đối với tìm ảnh tương tự |
| `top_k` | integer | Số kết quả tối đa trước phân trang |
| `offset` | integer | Vị trí bắt đầu trang, tính từ 0 |
| `limit` | integer | Số kết quả tối đa trong trang |
| total | integer | Số kết quả sau khi giới hạn `top_k` |
| `has_more` | boolean | Có trang tiếp theo |
| `next_offset` | integer hoặc null | Offset trang tiếp theo; null khi hết kết quả |
| `index_version` | string | Phiên bản chỉ mục |
| results | array[SearchHit] | Danh sách kết quả |

`SearchHit` gồm:

| Trường | Kiểu | Mô tả |
|---|---|---|
| rank | integer | Thứ hạng trong toàn bộ tập kết quả, bắt đầu từ 1 |
| image | `ImagePublic` | Thông tin ảnh |
| score | number | Điểm xếp hạng hữu hạn |
| similarity | number hoặc null | CLIP: bằng score, thuộc [-1, 1]; BM25: null |

## 4. Endpoint

### 4.1. GET /health

**Chức năng:** Tra cứu trạng thái dịch vụ.

**Request:** Không có path parameter, query parameter hoặc body.

**Phản hồi 200:** `data` theo schema Health.

**Mã phản hồi:** 200, 405, 422, 500.

Trong chế độ mock: `status=ok`, `ready=true`, `model_loaded=false`, `index_loaded=false`.

### 4.2. POST /search

**Chức năng:** Tìm ảnh theo văn bản.

**Request header:** Content-Type: application/json.

**Request query:** Không có.

**Request body:** `SearchRequest`.

| Trường | Kiểu | Bắt buộc | Mặc định | Ràng buộc |
|---|---|---|---|---|
| query | string | Có | — | Loại bỏ khoảng trắng ở hai đầu; 1–500 ký tự; không rỗng sau xử lý |
| method | string | Không | clip | clip hoặc bm25 |
| `top_k` | integer | Không | 20 | 1–100 |
| `offset` | integer | Không | 0 | >= 0 |
| `limit` | integer | Không | 20 | 1–50 |
| `min_similarity` | number hoặc null | Không | null | Số hữu hạn thuộc [-1, 1]; chỉ được đặt khi method=clip |
| dataset | string hoặc null | Không | null | Lọc theo tên bộ dữ liệu chính xác |
| split | string hoặc null | Không | null | train, val, test hoặc unspecified |

Các trường integer trong JSON không chấp nhận chuỗi hoặc boolean.
`method=bm25` và `min_similarity` khác null trả `422`.
Dataset không tồn tại trả `200` với danh sách kết quả rỗng.

**Phản hồi 200:** `data` theo schema `SearchResponse`.

**Mã phản hồi:** 200, 405, 415, 422, 500, 503.

```http
POST /search HTTP/1.1
Accept: application/json
Content-Type: application/json

{"query":"a dog on the beach","method":"clip","top_k":3,"offset":0,"limit":2}
```

### 4.3. GET /images/{`image_id`}

**Chức năng:** Tra cứu thông tin công khai của ảnh.

| Tham số | Vị trí | Kiểu | Bắt buộc |
|---|---|---|---|
| `image_id` | path | ImageId | Có |

**Request:** Không có query parameter hoặc body.

**Phản hồi 200:** `data` theo schema `ImagePublic`.

**Mã phản hồi:** 200, 404, 405, 422, 500.

### 4.4. GET /images/{`image_id`}/similar

**Chức năng:** Tìm ảnh tương tự ảnh nguồn.

**Path parameter:** `image_id: ImageId`, bắt buộc.

**Query parameter:** SimilarRequest.

| Tham số | Kiểu | Mặc định | Ràng buộc |
|---|---|---|---|
| `top_k` | integer | 20 | 1–100 |
| `offset` | integer | 0 | >= 0 |
| `limit` | integer | 20 | 1–50 |
| `min_similarity` | number | Không đặt | Số hữu hạn thuộc [-1, 1] |

Giá trị query được truyền dưới dạng chuỗi số. Bỏ `min_similarity` tương đương null;
chuỗi `null` không hợp lệ. Request không có body.

**Phản hồi 200:** `data` theo schema `SearchResponse`; `method=clip`, `query=null`.
Ảnh nguồn phải được loại khỏi tập kết quả trước khi áp dụng `top_k`.

**Mã phản hồi:** 200, 404, 405, 422, 500, 503.

```http
GET /images/demo:001/similar?top_k=3&limit=2 HTTP/1.1
Accept: application/json
```

### 4.5. GET /media/{path}

**Chức năng:** Tải tài nguyên ảnh mẫu.

**Path parameter:** `path: string`, đường dẫn tương đối dưới thư mục ảnh mẫu.

**Phản hồi 200:** Nội dung SVG; Content-Type: image/svg+xml.

**Phản hồi 304:** Không có body.

**Phản hồi lỗi:** Theo cấu trúc tại mục 1.2.

**Mã phản hồi:** 200, 304, 404, 405, 500.

HEAD được hỗ trợ, trả header tương ứng GET và không có body.
HEAD không được hỗ trợ tại các endpoint JSON.

## 5. Xếp hạng và phân trang

Trình tự xử lý:

1. Lọc dataset và split.
2. Tính điểm và áp dụng `min_similarity` nếu được đặt.
3. Sắp xếp score giảm dần; nếu bằng điểm, sắp xếp ID tăng dần.
4. Giới hạn số kết quả theo `top_k`.
5. Lấy trang từ `offset` với tối đa `limit` phần tử.

Quy tắc phản hồi:

- `total` là số phần tử sau bước 4.
- `rank` là thứ hạng trong tập kết quả sau bước 4.
- `has_more = offset + len(results) < total`.
- `next_offset = offset + len(results)` nếu `has_more`=true; ngược lại bằng null.
- Offset vượt tập kết quả trả `200`, results=[], `has_more`=false, `next_offset`=null.
- Query, bộ lọc và `top_k` phải giữ nguyên giữa các trang của cùng truy vấn.
- Khi `index_version` thay đổi, client phải khởi tạo lại kết quả phân trang.

## 6. Ví dụ tích hợp COCO 2014

Dữ liệu ví dụ lấy từ captions_train2014.json: ảnh ID 57870, giấy phép tương ứng
và năm caption của ảnh.

### 6.1. Cấu trúc JSON gốc của COCO

Nguồn annotation: [annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip),
tệp annotations/captions_train2014.json sau giải nén. Quy định nguồn annotation và điều kiện dừng
chuyển đổi nằm tại mục 1.1 của [Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md).

File captions_train2014.json gồm bốn trường cấp gốc: info, images, licenses và annotations.
JSON dưới đây giữ cấu trúc nguồn và rút gọn các danh sách còn dữ liệu của ảnh 57870.

```json
{
  "info": {
    "description": "COCO 2014 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2014,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
  },
  "images": [
    {
      "license": 5,
      "file_name": "COCO_train2014_000000057870.jpg",
      "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg",
      "height": 480,
      "width": 640,
      "date_captured": "2013-11-14 16:28:13",
      "flickr_url": "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg",
      "id": 57870
    }
  ],
  "licenses": [
    {
      "url": "http://creativecommons.org/licenses/by-sa/2.0/",
      "id": 5,
      "name": "Attribution-ShareAlike License"
    }
  ],
  "annotations": [
    {
      "image_id": 57870,
      "id": 787980,
      "caption": "A restaurant has modern wooden tables and chairs."
    },
    {
      "image_id": 57870,
      "id": 789366,
      "caption": "A long restaurant table with rattan rounded back chairs."
    },
    {
      "image_id": 57870,
      "id": 789888,
      "caption": "a long table with a plant on top of it surrounded with wooden chairs "
    },
    {
      "image_id": 57870,
      "id": 791316,
      "caption": "A long table with a flower arrangement in the middle for meetings"
    },
    {
      "image_id": 57870,
      "id": 794853,
      "caption": "A table is adorned with wooden chairs with blue accents."
    }
  ]
}
```

Caption được ghép với ảnh qua `annotations[].image_id` = images[].id.
Giấy phép được tra qua images[].license = licenses[].id.

Đối chiếu cấu trúc và dữ liệu train 2014/2017: [Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md), mục 3.1.

### 6.2. Metadata sau chuyển đổi

Bản ghi Metadata 1.0.0 tương ứng:

```json
{
  "schema_version": "1.0.0",
  "id": "coco2014:57870",
  "dataset": "coco2014",
  "source_id": "57870",
  "split": "train",
  "image_path": "images/train2014/COCO_train2014_000000057870.jpg",
  "thumbnail_path": null,
  "width": 640,
  "height": 480,
  "captions": [
    {
      "text": "A restaurant has modern wooden tables and chairs.",
      "language": "en"
    },
    {
      "text": "A long restaurant table with rattan rounded back chairs.",
      "language": "en"
    },
    {
      "text": "a long table with a plant on top of it surrounded with wooden chairs ",
      "language": "en"
    },
    {
      "text": "A long table with a flower arrangement in the middle for meetings",
      "language": "en"
    },
    {
      "text": "A table is adorned with wooden chairs with blue accents.",
      "language": "en"
    }
  ],
  "license": "Attribution-ShareAlike License",
  "source_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg"
}
```

| Trường | Quy tắc áp dụng trong ví dụ |
|---|---|
| id | Namespace `coco2014` ghép với ID ảnh 57870 |
| `source_id` | Chuyển images[].id thành chuỗi |
| split | train, theo file captions_train2014.json |
| `image_path` | Đường dẫn tương đối theo cấu trúc thư mục đầu ra của dự án |
| `width`, `height` | Giữ giá trị trong annotation ảnh |
| `captions` | Gom toàn bộ annotation có `image_id`=57870, sắp theo annotation.id |
| license | Tên giấy phép tra theo images[].license=5 |
| `thumbnail_path` | null vì ví dụ chưa có thumbnail |

Caption được giữ nguyên, bao gồm khoảng trắng ở cuối nếu có trong nguồn.
Kích thước trong annotation phải được đối chiếu với tệp ảnh khi chuyển đổi.

### 6.3. Tra cứu ảnh qua API

Sau khi dataset được nạp vào backend:

```http
GET /images/coco2014:57870 HTTP/1.1
Accept: application/json
```

Header phản hồi:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store
X-API-Version: 2.0.0
X-Request-ID: 89225dda-1260-45ad-9c19-c05df6204fe8
```

Body theo `SuccessResponse<ImagePublic>`:

```json
{
  "success": true,
  "statusCode": 200,
  "data": {
    "id": "coco2014:57870",
    "dataset": "coco2014",
    "split": "train",
    "width": 640,
    "height": 480,
    "captions": [
      {
        "text": "A restaurant has modern wooden tables and chairs.",
        "language": "en"
      },
      {
        "text": "A long restaurant table with rattan rounded back chairs.",
        "language": "en"
      },
      {
        "text": "a long table with a plant on top of it surrounded with wooden chairs ",
        "language": "en"
      },
      {
        "text": "A long table with a flower arrangement in the middle for meetings",
        "language": "en"
      },
      {
        "text": "A table is adorned with wooden chairs with blue accents.",
        "language": "en"
      }
    ],
    "image_url": "/media/coco2014/train2014/COCO_train2014_000000057870.jpg",
    "thumbnail_url": "/media/coco2014/train2014/COCO_train2014_000000057870.jpg"
  },
  "timestamp": "2026-10-13T03:50:00.000Z"
}
```

`image_url` minh họa đường dẫn ảnh sau tích hợp dataset. Backend phải triển khai đường dẫn này.
Khi chưa có thumbnail, `thumbnail_url` trỏ đến ảnh gốc.
API không trả `image_path`, `source_id` hoặc license vì các trường này không thuộc `ImagePublic`.
timestamp và X-Request-ID trong ví dụ là giá trị mẫu. Server cấp giá trị mới cho mỗi request.

### 6.4. Tìm kiếm trong dataset

```http
POST /search HTTP/1.1
Accept: application/json
Content-Type: application/json
```

Body:

```json
{
  "query": "a restaurant with wooden tables and chairs",
  "method": "clip",
  "top_k": 20,
  "offset": 0,
  "limit": 10,
  "min_similarity": null,
  "dataset": "coco2014",
  "split": "train"
}
```

Backend lấy ảnh từ chỉ mục thuộc dataset `coco2014` và lọc split=train theo request.
Phản hồi có `data` theo `SearchResponse`. Mỗi hit gồm `ImagePublic` như mục 6.3 và điểm
do engine tính. Thứ hạng ảnh 57870 phụ thuộc kết quả tìm kiếm.

### 6.5. Phân công tích hợp

| Thành phần | Dữ liệu sử dụng |
|---|---|
| Xử lý dữ liệu | Chuyển annotation nguồn thành metadata như mục 6.2 |
| CLIP/FAISS | Tạo embedding ảnh; `image_ids.json` sử dụng ID `coco2014`:57870 |
| BM25 | Tạo một document từ các caption của ảnh; dùng cùng ID |
| Backend | Tra metadata bằng ID, tạo URL ảnh và trả `ImagePublic` |
| Frontend | Đọc `response.data`, hiển thị ảnh bằng `image_url` hoặc `thumbnail_url` |

Server mock hiện chỉ có namespace demo. Request với ID `coco2014`:57870 trả 404;
search có bộ lọc `coco2014` trả danh sách rỗng. Ví dụ áp dụng cho backend sau tích hợp dataset.

Contract của `RetrievalEngine` và quy trình tạo phản hồi từ kết quả ML: [Đặc tả tích hợp ML và backend](05_ML_INTEGRATION_CONTRACT.md).

## 7. Ví dụ tích hợp COCO 2017

### 7.1. Nguồn và annotation

Gói annotation: [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
Sử dụng annotations/captions_train2017.json sau giải nén. Áp dụng quy định nguồn
COCO tại mục 1.1 của [Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md).

Ví dụ giữ cấu trúc JSON nguồn và rút gọn còn ảnh 57870, giấy phép và toàn bộ caption
của ảnh. Dữ liệu lấy từ file captions_train2017.json đã đối chiếu.

```json
{
  "info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
  },
  "images": [
    {
      "license": 5,
      "file_name": "000000057870.jpg",
      "coco_url": "http://images.cocodataset.org/train2017/000000057870.jpg",
      "height": 480,
      "width": 640,
      "date_captured": "2013-11-14 16:28:13",
      "flickr_url": "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg",
      "id": 57870
    }
  ],
  "licenses": [
    {
      "url": "http://creativecommons.org/licenses/by-sa/2.0/",
      "id": 5,
      "name": "Attribution-ShareAlike License"
    }
  ],
  "annotations": [
    {
      "image_id": 57870,
      "id": 787980,
      "caption": "A restaurant has modern wooden tables and chairs."
    },
    {
      "image_id": 57870,
      "id": 789366,
      "caption": "A long restaurant table with rattan rounded back chairs."
    },
    {
      "image_id": 57870,
      "id": 789888,
      "caption": "a long table with a plant on top of it surrounded with wooden chairs "
    },
    {
      "image_id": 57870,
      "id": 791316,
      "caption": "A long table with a flower arrangement in the middle for meetings"
    },
    {
      "image_id": 57870,
      "id": 794853,
      "caption": "A table is adorned with wooden chairs with blue accents."
    }
  ]
}
```

### 7.2. Metadata sau chuyển đổi

Ghép annotations[].image_id với images[].id. Tên tệp ảnh lấy trực tiếp từ file_name.
Bản ghi Metadata 1.0.0:

```json
{
  "schema_version": "1.0.0",
  "id": "coco2017:57870",
  "dataset": "coco2017",
  "source_id": "57870",
  "split": "train",
  "image_path": "images/train2017/000000057870.jpg",
  "thumbnail_path": null,
  "width": 640,
  "height": 480,
  "captions": [
    {
      "text": "A restaurant has modern wooden tables and chairs.",
      "language": "en"
    },
    {
      "text": "A long restaurant table with rattan rounded back chairs.",
      "language": "en"
    },
    {
      "text": "a long table with a plant on top of it surrounded with wooden chairs ",
      "language": "en"
    },
    {
      "text": "A long table with a flower arrangement in the middle for meetings",
      "language": "en"
    },
    {
      "text": "A table is adorned with wooden chairs with blue accents.",
      "language": "en"
    }
  ],
  "license": "Attribution-ShareAlike License",
  "source_url": "http://images.cocodataset.org/train2017/000000057870.jpg"
}
```

Ảnh 57870 có trong cả hai bản 2014 và 2017. Với bản 2017, dataset=coco2017,
id=coco2017:57870 và file_name=000000057870.jpg. Không dùng tiền tố COCO_train2014
để tạo đường dẫn ảnh 2017.

### 7.3. Tra cứu ảnh qua API

```http
GET /images/coco2017:57870 HTTP/1.1
Accept: application/json
```

Header phản hồi:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store
X-API-Version: 2.0.0
X-Request-ID: 89225dda-1260-45ad-9c19-c05df6204fe8
```

Body theo `SuccessResponse<ImagePublic>`:

```json
{
  "success": true,
  "statusCode": 200,
  "data": {
    "id": "coco2017:57870",
    "dataset": "coco2017",
    "split": "train",
    "width": 640,
    "height": 480,
    "captions": [
      {
        "text": "A restaurant has modern wooden tables and chairs.",
        "language": "en"
      },
      {
        "text": "A long restaurant table with rattan rounded back chairs.",
        "language": "en"
      },
      {
        "text": "a long table with a plant on top of it surrounded with wooden chairs ",
        "language": "en"
      },
      {
        "text": "A long table with a flower arrangement in the middle for meetings",
        "language": "en"
      },
      {
        "text": "A table is adorned with wooden chairs with blue accents.",
        "language": "en"
      }
    ],
    "image_url": "/media/coco2017/train2017/000000057870.jpg",
    "thumbnail_url": "/media/coco2017/train2017/000000057870.jpg"
  },
  "timestamp": "2026-10-13T03:50:00.000Z"
}
```

URL media là đường dẫn minh họa cần triển khai khi tích hợp dataset.
thumbnail_url trỏ đến ảnh gốc khi chưa có thumbnail. Timestamp và X-Request-ID là giá trị mẫu.

### 7.4. Tìm kiếm trong dataset

```http
POST /search HTTP/1.1
Accept: application/json
Content-Type: application/json
```

Body:

```json
{
  "query": "a restaurant with wooden tables and chairs",
  "method": "clip",
  "top_k": 20,
  "offset": 0,
  "limit": 10,
  "min_similarity": null,
  "dataset": "coco2017",
  "split": "train"
}
```

ML/BM25 trả ID coco2017:57870 khi ảnh thuộc kết quả tìm kiếm. Backend ghép metadata,
phân trang và trả SearchResponse theo mục 3.4. Thứ hạng và score do engine tính.

## 8. Ví dụ tích hợp Flickr30k Karpathy

### 8.1. Nguồn tải và bố trí dữ liệu

| Thành phần | Nguồn | Sử dụng |
|---|---|---|
| Ảnh Flickr30k | [Flickr30k images trên Box](https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl) | Giải nén thư mục flickr30k-images |
| Annotation và split Karpathy | [caption_datasets.zip](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) | Sử dụng dataset_flickr30k.json trong gói |

Namespace của project là flickr30k_karpathy vì dùng annotation có split Karpathy.
Trường dataset trong JSON nguồn vẫn là flickr30k. Quy tắc namespace tại đây không sửa file nguồn.

```text
<dataset_root>/
  dataset_flickr30k.json
  flickr30k-images/
    1000092795.jpg
    ...
  metadata.jsonl
```

Hai gói được tải riêng. Annotation không chứa ảnh hoặc kích thước ảnh.
dataset_root là thư mục cha của flickr30k-images.

### 8.2. Cấu trúc JSON nguồn

Ví dụ giữ toàn bộ trường của ảnh đầu tiên và năm caption trong dataset_flickr30k.json.
Các ảnh khác được lược bỏ.

```json
{
  "dataset": "flickr30k",
  "images": [
    {
      "sentids": [
        0,
        1,
        2,
        3,
        4
      ],
      "imgid": 0,
      "sentences": [
        {
          "tokens": [
            "two",
            "young",
            "guys",
            "with",
            "shaggy",
            "hair",
            "look",
            "at",
            "their",
            "hands",
            "while",
            "hanging",
            "out",
            "in",
            "the",
            "yard"
          ],
          "raw": "Two young guys with shaggy hair look at their hands while hanging out in the yard.",
          "imgid": 0,
          "sentid": 0
        },
        {
          "tokens": [
            "two",
            "young",
            "white",
            "males",
            "are",
            "outside",
            "near",
            "many",
            "bushes"
          ],
          "raw": "Two young, White males are outside near many bushes.",
          "imgid": 0,
          "sentid": 1
        },
        {
          "tokens": [
            "two",
            "men",
            "in",
            "green",
            "shirts",
            "are",
            "standing",
            "in",
            "a",
            "yard"
          ],
          "raw": "Two men in green shirts are standing in a yard.",
          "imgid": 0,
          "sentid": 2
        },
        {
          "tokens": [
            "a",
            "man",
            "in",
            "a",
            "blue",
            "shirt",
            "standing",
            "in",
            "a",
            "garden"
          ],
          "raw": "A man in a blue shirt standing in a garden.",
          "imgid": 0,
          "sentid": 3
        },
        {
          "tokens": [
            "two",
            "friends",
            "enjoy",
            "time",
            "spent",
            "together"
          ],
          "raw": "Two friends enjoy time spent together.",
          "imgid": 0,
          "sentid": 4
        }
      ],
      "split": "train",
      "filename": "1000092795.jpg"
    }
  ]
}
```

Caption đã nằm trong sentences của từng ảnh; không cần ghép danh sách annotation riêng
như COCO. imgid=0 và sentid là định danh nội bộ của annotation. ID ảnh của project lấy
từ filename bỏ phần mở rộng.

### 8.3. Metadata sau chuyển đổi

Ghép filename=1000092795.jpg với <dataset_root>/flickr30k-images/1000092795.jpg.
Script mở ảnh để lấy width/height và chuyển toàn bộ sentences[].raw thành captions[].text.

Kích thước 640 × 480 trong hai ví dụ metadata/response dưới đây chỉ minh họa cấu trúc;
chưa được xác nhận từ tệp ảnh 1000092795.jpg. Script phải thay bằng kích thước đọc từ JPG,
không dùng các giá trị này làm mặc định.

```json
{
  "schema_version": "1.0.0",
  "id": "flickr30k_karpathy:1000092795",
  "dataset": "flickr30k_karpathy",
  "source_id": "1000092795",
  "split": "train",
  "image_path": "flickr30k-images/1000092795.jpg",
  "thumbnail_path": null,
  "width": 640,
  "height": 480,
  "captions": [
    {
      "text": "Two young guys with shaggy hair look at their hands while hanging out in the yard.",
      "language": "en"
    },
    {
      "text": "Two young, White males are outside near many bushes.",
      "language": "en"
    },
    {
      "text": "Two men in green shirts are standing in a yard.",
      "language": "en"
    },
    {
      "text": "A man in a blue shirt standing in a garden.",
      "language": "en"
    },
    {
      "text": "Two friends enjoy time spent together.",
      "language": "en"
    }
  ],
  "license": null,
  "source_url": null
}
```

license và source_url bằng null vì annotation này không có thông tin tương ứng.
Nếu dùng thư mục images thay vì flickr30k-images, image_path phải được cập nhật theo
vị trí ảnh tương đối với dataset_root; ID và caption giữ nguyên.

### 8.4. Tra cứu ảnh qua API

```http
GET /images/flickr30k_karpathy:1000092795 HTTP/1.1
Accept: application/json
```

Header phản hồi:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store
X-API-Version: 2.0.0
X-Request-ID: 89225dda-1260-45ad-9c19-c05df6204fe8
```

Body theo `SuccessResponse<ImagePublic>`, sử dụng kích thước minh họa tại mục 8.3:

```json
{
  "success": true,
  "statusCode": 200,
  "data": {
    "id": "flickr30k_karpathy:1000092795",
    "dataset": "flickr30k_karpathy",
    "split": "train",
    "width": 640,
    "height": 480,
    "captions": [
      {
        "text": "Two young guys with shaggy hair look at their hands while hanging out in the yard.",
        "language": "en"
      },
      {
        "text": "Two young, White males are outside near many bushes.",
        "language": "en"
      },
      {
        "text": "Two men in green shirts are standing in a yard.",
        "language": "en"
      },
      {
        "text": "A man in a blue shirt standing in a garden.",
        "language": "en"
      },
      {
        "text": "Two friends enjoy time spent together.",
        "language": "en"
      }
    ],
    "image_url": "/media/flickr30k_karpathy/1000092795.jpg",
    "thumbnail_url": "/media/flickr30k_karpathy/1000092795.jpg"
  },
  "timestamp": "2026-10-13T03:50:00.000Z"
}
```

URL media cần được backend triển khai khi tích hợp ảnh. thumbnail_url dùng ảnh gốc
khi chưa có thumbnail. Timestamp và X-Request-ID là giá trị mẫu.

### 8.5. Tìm kiếm và ảnh tương tự

```http
POST /search HTTP/1.1
Accept: application/json
Content-Type: application/json
```

Body:

```json
{
  "query": "two men standing in a yard",
  "method": "clip",
  "top_k": 20,
  "offset": 0,
  "limit": 10,
  "min_similarity": null,
  "dataset": "flickr30k_karpathy",
  "split": "train"
}
```

Tìm ảnh tương tự:

```http
GET /images/flickr30k_karpathy:1000092795/similar?top_k=20&limit=10 HTTP/1.1
Accept: application/json
```

Engine dùng embedding đã lưu của ảnh nguồn và loại ảnh nguồn khỏi kết quả.
Backend trả SearchResponse với method=clip, query=null và phân trang theo mục 5.
BM25 dùng cùng POST /search với method=bm25 và min_similarity=null.
