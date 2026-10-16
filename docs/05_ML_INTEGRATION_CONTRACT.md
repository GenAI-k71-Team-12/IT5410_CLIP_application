## 1. Phạm vi và trách nhiệm

| Thành phần | Trách nhiệm |
|---|---|
| Xử lý dữ liệu | Bàn giao metadata hợp lệ, ảnh, thông tin split/subset và báo cáo chuyển đổi |
| CLIP encoder | Encode ảnh và văn bản theo cùng cấu hình mô hình |
| Lập chỉ mục | Tạo embedding, index, ánh xạ ID và manifest |
| `RetrievalEngine` | Lọc, tính điểm, xếp hạng và trả tối đa `top_k` ID ảnh |
| BM25 | Lập document theo ảnh và trả kết quả cùng kiểu `RankedImage` |
| Backend | Kiểm tra request, gọi engine, ghép `ImagePublic`, phân trang và tạo response API |
| Frontend | Đọc `data`, hiển thị ảnh, điểm và trạng thái lỗi |

ML không tạo success, statusCode, timestamp, request_id hoặc URL ảnh.
Backend quản lý các trường này theo [Đặc tả API](03_ENDPOINT_CONTRACTS.md).

## 2. Đầu vào

| Đầu vào | Quy định |
|---|---|
| `dataset_root` | Thư mục gốc dùng để phân giải `image_path` và `thumbnail_path` |
| `metadata.jsonl` | Metadata 1.0.0, mỗi ảnh một bản ghi |
| dataset | Namespace đã chọn theo đặc tả ánh xạ dataset |
| splits | Danh sách split được lập chỉ mục |
| subset | Danh sách ID được chọn, nếu không lập chỉ mục toàn bộ phạm vi splits |
| `artifact_dir` | Thư mục đầu ra riêng cho mỗi phiên bản chỉ mục |

Baseline sử dụng một namespace trong một bộ artifact. Chỉ mục nhiều dataset cần
đặc tả tích hợp riêng. Không ghép embedding từ các cấu hình khác nhau.

Với COCO, nguồn annotation phải đáp ứng mục 1.1 của [Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md).
Dữ liệu từ URL khác hoặc chưa xác định nguồn không được chuyển metadata hoặc build index
cho đến khi được kiểm tra và cập nhật đặc tả.

Quy tắc đầu vào:

1. Metadata phải hợp lệ, ID duy nhất và ảnh tồn tại dưới `dataset_root`.
2. Ảnh phải giải mã được, width/height khớp metadata.
3. Tập ảnh được xác định trước khi encode, giữ nguyên split nguồn.
4. Sắp bản ghi được chọn theo Metadata.id tăng dần theo thứ tự chuỗi.
5. Snapshot metadata trong artifact chỉ chứa các bản ghi được chọn, theo thứ tự trên.
6. Build thất bại khi có bản ghi lỗi. Ảnh lỗi chỉ được loại ở bước chuyển đổi dữ liệu, có ghi nhận trong báo cáo.

Mỗi ảnh tạo một embedding trong `embeddings.npy`, độc lập với số caption.

## 3. Cấu hình encoder

### 3.1. Cấu hình cơ sở

| Thuộc tính | Quy định |
|---|---|
| Model | ViT-B/32 theo kế hoạch dự án |
| Số chiều baseline | 512; kiểm tra với output của model thực tế |
| Checkpoint | Pretrained CLIP; ghi định danh và SHA-256 trong manifest |
| Thư viện | `open_clip_torch` 3.3.0, model `ViT-B-32`, weights `openai` (`requirements-ml.txt`); triển khai tại `clip_retrieval.ml.encoder.ClipEncoder` |
| Tiền xử lý ảnh | Pipeline đi kèm checkpoint đã chọn |
| Tokenizer | Tokenizer đi kèm model/checkpoint đã chọn |
| Thực thi | Chế độ inference, không tính gradient |
| Thiết bị | CPU hoặc CUDA, khai báo khi build và khi nạp |
| Dữ liệu lưu | `float32`, chuẩn hóa L2 |

Thư viện, checkpoint và dependency phải được xác nhận trước khi build artifact.
Encoder cùng số chiều vẫn phải được kiểm tra tương thích về model và weights.
Thí nghiệm ViT-B/16 hoặc model khác tạo bộ artifact riêng.

### 3.2. Encode ảnh

Đầu vào là ảnh gốc đọc từ dataset_root/image_path. Chuyển ảnh sang RGB và áp dụng
pipeline của checkpoint. Xử lý bổ sung như xoay theo EXIF
phải được ghi trong cấu hình preprocess và áp dụng thống nhất khi encode.

Kết quả mỗi batch có `shape` (B, D), với B là số ảnh và D là số chiều của model.
Vector phải được chuyển sang `float32`, kiểm tra hữu hạn và chuẩn hóa L2 trước khi lưu.
Vector có chuẩn bằng 0 là lỗi build.

### 3.3. Encode văn bản

Đầu vào là `SearchRequest`.query đã qua validation. Không thêm prompt hoặc dịch tự động
trong baseline. Nếu áp dụng prompt/dịch ở phiên bản khác, phải ghi cấu hình và thay khóa cache.

Dùng tokenizer cùng checkpoint, kiểm tra số token bao gồm token đặc biệt.
Truy vấn vượt `context_length` phát sinh `QueryTooLong`; không tự cắt token.
OpenAI CLIP mặc định `context_length`=77. Giá trị sử dụng thực tế phải ghi trong manifest.

Embedding văn bản có `shape` (1, D), `float32` và chuẩn hóa L2 như embedding ảnh.

## 4. Artifact

### 4.1. Cấu trúc thư mục

```text
artifacts/<artifact_version>/
  manifest.json
  metadata.jsonl
  embeddings.npy
  image_ids.json
  index.faiss
```

Mọi đường dẫn trong manifest tương đối với `artifact_dir`. Không dùng đường dẫn tuyệt đối
hoặc thành phần .. . Đường dẫn ảnh trong snapshot metadata vẫn tương đối với `dataset_root`.

| Tệp | Contract |
|---|---|
| `metadata.jsonl` | Snapshot của N ảnh được lập chỉ mục, Metadata 1.0.0 |
| `embeddings.npy` | Mảng C-contiguous, `float32`, `shape` (N, D), không lưu object hoặc pickle |
| `image_ids.json` | JSON array gồm N string ID duy nhất |
| `index.faiss` | Chỉ mục có d=D, ntotal=N, metric inner product |
| `manifest.json` | Thông tin phiên bản, model, phạm vi dữ liệu và checksum |

Với mọi i từ 0 đến N-1:

```text
metadata[i].id = image_ids[i]
embeddings[i] = vector của ảnh image_ids[i]
FAISS row i = image_ids[i]
```

Baseline dùng `IndexFlatIP` với ID theo hàng. Nếu dùng `IndexIDMap`, phải bổ sung mapping
ID số FAISS sang ID ảnh, tên tệp và checksum vào phiên bản manifest tương ứng trước khi bàn giao.
Kết quả FAISS -1 không phải ID hợp lệ và không được đưa vào response.

### 4.2. Manifest

Cấu trúc manifest cơ sở:

```json
{
  "manifest_version": "1.0.0",
  "artifact_version": "coco2017-val-vitb32-v1",
  "metadata_schema_version": "1.0.0",
  "dataset": "coco2017",
  "splits": ["val"],
  "subset_ids_sha256": null,
  "created_at": "2026-10-13T03:50:00.000Z",
  "model": {
    "library": "TO_BE_SELECTED",
    "library_version": "TO_BE_SELECTED",
    "name": "ViT-B/32",
    "weights_id": "TO_BE_SELECTED",
    "weights_sha256": null,
    "tokenizer_id": "TO_BE_SELECTED",
    "tokenizer_config_sha256": null,
    "context_length": 77,
    "preprocess_id": "TO_BE_SELECTED",
    "preprocess_config_sha256": null
  },
  "embedding": {
    "dimension": 512,
    "dtype": "float32",
    "normalization": "l2",
    "count": 5000
  },
  "index": {
    "library": "faiss",
    "library_version": "TO_BE_SELECTED",
    "type": "IndexFlatIP",
    "metric": "inner_product",
    "id_mapping": "row"
  },
  "build": {
    "device": "cpu",
    "batch_size": 32,
    "seed": 0,
    "dependency_lock_sha256": null
  },
  "files": {
    "metadata": {"path": "metadata.jsonl", "sha256": null},
    "embeddings": {"path": "embeddings.npy", "sha256": null},
    "image_ids": {"path": "image_ids.json", "sha256": null},
    "index": {"path": "index.faiss", "sha256": null}
  }
}
```

Manifest trên là template. Namespace, split, count, batch_size và device là giá trị mẫu. Các giá trị `TO_BE_SELECTED` và checksum null bắt buộc
được thay bằng giá trị thực trước bàn giao, trừ `subset_ids_sha256` được phép null khi không chọn subset.

| Nhóm trường | Quy định |
|---|---|
| manifest_version | Phiên bản cấu trúc manifest, độc lập với API và metadata |
| `artifact_version` | Định danh bất biến cho một bộ dữ liệu/model/index; bằng engine.`index_version` |
| dataset, splits | Khớp snapshot metadata |
| `subset_ids_sha256` | SHA-256 của danh sách ID subset chuẩn hóa; null nếu toàn bộ phạm vi splits |
| created_at | UTC ISO 8601 với milliseconds, hậu tố Z |
| model | Định danh thư viện, checkpoint, tokenizer và preprocess dùng để build |
| embedding | D, N, `dtype` và normalization của tệp embeddings |
| index | Loại index, metric, phiên bản thư viện và cách ánh xạ ID |
| build | Thiết bị, batch size, seed và checksum file dependency ML |
| files | Tên tệp và SHA-256 nội dung byte của từng tệp |

Checksum checkpoint tính trên byte của tệp weights. Checksum cấu hình tokenizer/preprocess
và danh sách subset tính trên JSON UTF-8 chuẩn hóa với `sort_keys=true`, separators=(",", ":"),
không thêm newline. Cấu hình cần bao gồm các thông số thực tế và định danh tài nguyên
như vocabulary/merges, resize/crop/interpolation/mean/std; đủ để tái tạo pipeline.
Bản cấu hình chuẩn hóa phải được bàn giao để loader có thể đối chiếu.

Manifest không chứa checksum của chính nó. Các file cấu hình bổ sung phải có tên tệp
và checksum trong files khi triển khai. File manifest hiện chưa có schema/validator trong mã nguồn.

### 4.3. Build và cache offline

Lệnh build nhận metadata, `dataset_root`, dataset, splits, subset nếu có, model config,
batch_size, device và `artifact_dir`. Module `clip_retrieval.ml.build_index` (lệnh
`python -m clip_retrieval.ml.build_index --help`); kiểm tra artifact bằng
`python -m clip_retrieval.ml.artifact <artifact_dir> --check-model`. Artifact có thêm
`tokenizer_config.json` và `preprocess_config.json`, khai báo trong `files` của manifest.

Batch encode giữ thứ tự ID đã quy định. Cache embedding chỉ được tái sử dụng khi
checksum nội dung ảnh và fingerprint model/weights/preprocess cùng khớp.
Metadata đổi caption nhưng ảnh không đổi có thể giữ embedding ảnh; vẫn phải cập nhật snapshot,
manifest và BM25. Checkpoint hoặc preprocess thay đổi yêu cầu encode lại.

Ghi bộ artifact vào thư mục tạm, kiểm tra đầy đủ rồi công bố cả thư mục.
Không ghi đè bộ artifact đang phục vụ request. Build hoàn tất phải báo N, D,
số batch, thời gian và `artifact_version`.

## 5. Giao diện tìm kiếm

Giao diện hiện có tại `src/clip_retrieval/retrieval.py`:

```python
@dataclass(frozen=True)
class RankedImage:
    image_id: str
    score: float

class RetrievalEngine(Protocol):
    index_version: str

    def search_text(self, request: SearchRequest) -> list[RankedImage]: ...
    def search_similar(self, image_id: str, request: SearchRequest) -> list[RankedImage]: ...
```

Giao diện hoạt động đồng bộ. Tác vụ encode/search không chạy trực tiếp trên async event loop.
Bản triển khai: `clip_retrieval.ml.engine.ClipRetrievalEngine.from_artifact(artifact_dir, device)`.
Engine trả `EngineNotReady` cho method=bm25 cho tới khi thành phần BM25 được tích hợp.
Cấu hình thread, process hoặc worker được xác định khi tích hợp.

### 5.1. search_text

- method=clip: encode query và tìm theo cosine similarity.
- method=bm25: chuyển sang thành phần BM25 của cùng corpus.
- dataset/split khác null: lọc theo giá trị chính xác. Không có ảnh khớp trả [].
- `min_similarity` áp dụng cho CLIP, so sánh score >= ngưỡng.
- Trả tối đa `top_k` `RankedImage` sau lọc và xếp hạng.
- offset/limit không được dùng để cắt kết quả trong engine; backend phân trang.
- Thành phần được yêu cầu chưa sẵn sàng trả `EngineNotReady`; không tự chuyển phương pháp.

### 5.2. search_similar

- `image_id` phải tồn tại trong snapshot và có embedding tương ứng.
- Dùng embedding đã lưu của ảnh nguồn, không encode lại theo một pipeline khác.
- Chỉ hỗ trợ method=clip. Các bộ lọc trong `SearchRequest` nội bộ áp dụng như search_text.
- Loại ảnh nguồn trước ngưỡng/top_k; không tự giới hạn dataset ngoài bộ lọc đã truyền.
- query không được sử dụng. Backend dùng giá trị nội bộ "similar" để tạo `SearchRequest`
  và trả query=null trong response.
- Trả cùng kiểu `RankedImage`; offset/limit do backend xử lý.

### 5.3. Điểm và thứ tự

CLIP score là tích vô hướng giữa vector query và ảnh đã chuẩn hóa L2.
Không dùng logits đã nhân hệ số hoặc softmax làm cosine similarity.
Kết quả có score hữu hạn thuộc [-1, 1]. Sai lệch ngoài biên tối đa 1e-5 do `float32`
được clip về biên trước khi trả backend. Sai lệch lớn hơn là `RetrievalFailure`.

Sắp score giảm dần, hòa điểm sắp Metadata.id tăng dần. ID kết quả phải duy nhất,
tồn tại trong snapshot và không phải ID số nội bộ của FAISS.
Baseline phải cho kết quả tương đương tính điểm toàn bộ tập ảnh hợp lệ rồi lấy `top_k`.
Phải lọc trước tìm kiếm hoặc lấy đủ ứng viên để đạt `top_k` trên tập hợp lệ.
Lấy `top_k` toàn corpus rồi lọc không đáp ứng yêu cầu này. Trường hợp hòa điểm tại ranh giới `top_k` phải xử lý theo ID.

Index gần đúng dùng cho benchmark cần cấu hình và tiêu chí đánh giá riêng.

### 5.4. BM25

Một ảnh tạo một document bằng cách nối `captions[].text` theo thứ tự bằng newline.
Giữ toàn bộ caption, ID và corpus như snapshot. Ngôn ngữ, tokenizer, chuẩn hóa chữ,
stopwords, công thức BM25, thư viện/phiên bản và tham số phải được ghi trong cấu hình BM25
trước triển khai. Cấu hình BM25 chưa được xác nhận.

BM25 trả score hữu hạn theo công thức đã chọn và similarity=null ở API.
Có thể dựng lại BM25 khi startup từ snapshot và cấu hình; nếu lưu index riêng,
phải bàn giao định dạng, phiên bản và checksum, không dùng tệp pickle không xác định.

## 6. Cache truy vấn

Cache embedding văn bản dùng query đã trim và fingerprint model/weights/tokenizer/prompt.
Cache kết quả bổ sung `artifact_version`, method, dataset, split, `min_similarity` và `top_k`.
Cache trả danh sách trước phân trang; offset/limit không thuộc khóa cache này.
Cache similar bổ sung `image_id`. Thay đổi cấu hình BM25 phải đổi fingerprint BM25.

Cache có giới hạn số phần tử. Xóa cache không được làm thay đổi kết quả tìm kiếm.
Dung lượng/TTL được cấu hình khi triển khai. Không cache lỗi.

## 7. Nạp artifact và readiness

Backend cấu hình `dataset_root`, `artifact_dir`, device và các method cần phục vụ.
Nạp engine tại startup/lifespan, không nạp model theo từng request.

Loader phải kiểm tra:

1. Manifest được hỗ trợ và không còn giá trị placeholder.
2. Tệp tồn tại dưới `artifact_dir` và SHA-256 khớp.
3. Snapshot metadata hợp lệ, cùng namespace/split, có N bản ghi và N ID duy nhất.
4. image_ids khớp thứ tự snapshot; không có ID thừa hoặc thiếu.
5. Embedding có `shape` (N, D), `float32`, hữu hạn và N > 0.
6. Chuẩn L2 từng vector sai lệch không quá 1e-4 so với 1.
7. Index có d=D, ntotal=N, inner product và mapping đã quy định.
8. Vector trong `IndexFlatIP` tương ứng với embeddings, sai số tuyệt đối không quá 1e-5.
9. Checkpoint, tokenizer và preprocess được nạp khớp manifest.
10. BM25 sẵn sàng nếu nằm trong các method được cấu hình phục vụ.

Khi lỗi, engine không được nhận request. Nếu tiến trình backend tiếp tục chạy,
health trả 200 với `mode=real`, `ready=false`, `status=not_ready`; search trả 503.
model_loaded/index_loaded phản ánh riêng trạng thái nạp thành công từng thành phần.
Chỉ `ready=true` khi mọi thành phần cần phục vụ đã nạp và kiểm tra thành công.
Không tự fallback sang mock.

## 8. Lỗi tích hợp

| Lỗi nội bộ | Điều kiện | Xử lý backend |
|---|---|---|
| `QueryTooLong` | Query vượt `context_length` | HTTP 422, message mô tả giới hạn token |
| `ImageNotFound` | ID ảnh không có trong corpus phục vụ | HTTP 404 |
| `EngineNotReady` | Model/index/method chưa sẵn sàng | HTTP 503 |
| `ArtifactInvalid` | Tệp, checksum, `shape` hoặc mapping sai | Startup không ready, search 503; chi tiết ghi log |
| `RetrievalFailure` | Lỗi xử lý sau khi engine đã sẵn sàng | HTTP 500, message chung |

Engine không trả HTTPException hoặc JSON envelope. Backend chuyển lỗi nội bộ sang
`ErrorResponse`, cấp X-Request-ID và ghi log. Lỗi request Pydantic vẫn do backend xử lý.

## 9. Tạo phản hồi API

Backend nhận danh sách tối đa `top_k` kết quả từ engine và xử lý:

1. Kiểm ID duy nhất, ID tồn tại, score hữu hạn và thứ tự đúng.
2. total = len(pool).
3. Cắt pool[`offset`:`offset` + `limit`].
4. Ghép metadata thành `ImagePublic` và cấp rank=`offset`+i+1.
5. CLIP: similarity=score, score_type=cosine_similarity. BM25: similarity=null, score_type=bm25_score.
6. Tính `has_more` và `next_offset` theo đặc tả API.
7. Đặt `index_version`=engine.`index_version`, `mode=real` và tạo `SuccessResponse`.

Ví dụ `RankedImage` sử dụng ID dữ liệu mẫu và điểm minh họa:

```python
[
    RankedImage(image_id="demo:001", score=0.82),
    RankedImage(image_id="demo:003", score=0.71),
    RankedImage(image_id="demo:002", score=0.64),
]
```

Với `offset`=1, `limit`=1, backend trả một hit của demo:003, rank=2,
total=3, `has_more`=true, `next_offset`=2. Engine trả toàn bộ danh sách trước phân trang.

## 10. Điều kiện nghiệm thu

| Nhóm | Kiểm tra bắt buộc |
|---|---|
| Encoder | Ảnh/query trả đúng `shape` và `dtype`; hữu hạn; chuẩn L2 đạt dung sai; query vượt token bị từ chối |
| Mapping | Metadata, image_ids, embedding và FAISS row cùng ảnh; không bỏ sót hoặc trùng ID |
| Build | Tạo artifact bằng một lệnh đã bàn giao; đọc lại thành công; có log và manifest đầy đủ |
| Toàn vẹn | Sửa một tệp hoặc sai checkpoint làm loader từ chối artifact |
| Search | FAISS baseline khớp brute-force trên bộ kiểm tra cố định, so score với atol=1e-5 |
| Bộ lọc | Dataset/split/ngưỡng áp dụng trước `top_k`, bao gồm tập rỗng và `top_k` lớn hơn corpus |
| Similar | Loại ảnh nguồn trước `top_k`; ID thiếu trả lỗi xác định |
| Tie-break | Điểm bằng nhau có thứ tự ID ổn định, bao gồm ranh giới `top_k` |
| Phân trang | Không trùng/bỏ sót ID giữa các trang của cùng pool |
| BM25 | Một document mỗi ảnh, cùng corpus; similarity=null; method chưa nạp không fallback |
| Readiness | Artifact sai không ready; request trả 503; không dùng kết quả mock |
| Cache | Kết quả có/không cache tương đương; đổi artifact/model/bộ lọc không dùng kết quả cũ |

Bộ bàn giao ML phải có lệnh cài dependency, lệnh build index, lệnh kiểm tra artifact,
lệnh chạy kiểm thử, cấu hình đã chọn và kết quả kiểm tra. Dependency ML được khóa riêng
sau khi kiểm tra trên cấu hình CPU/GPU đã chọn.

## 11. Tài liệu tham chiếu

- [OpenAI CLIP](https://github.com/openai/CLIP): encoder, tokenizer và pipeline theo model.
- [FAISS Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started): vector `float32` và giao diện chỉ mục.
- [FAISS MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances): inner product trên vector chuẩn hóa cho cosine similarity.
- [Đặc tả dữ liệu](02_DATA_API_CONTRACT.md)
- [Đặc tả API](03_ENDPOINT_CONTRACTS.md)
- [Ánh xạ dataset](04_DATASET_MAPPING.md)
