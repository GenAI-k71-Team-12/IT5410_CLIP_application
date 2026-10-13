# Đặc tả ánh xạ dataset

| Thuộc tính | Giá trị |
|---|---|
| Phiên bản metadata | 1.0.0 |
| Phiên bản API | 2.0.0 |
| Phạm vi | COCO Captions 2017, COCO Captions 2014, Flickr30k |
| Trạng thái lựa chọn dataset | Chưa lựa chọn |
| Schema đích | [Đặc tả dữ liệu](02_DATA_API_CONTRACT.md) |
| Giao diện HTTP | [Đặc tả API](03_ENDPOINT_CONTRACTS.md) |

## 1. Phương án dữ liệu

Dữ liệu nguồn được chuyển thành Metadata 1.0.0 để sử dụng trong BM25, CLIP/FAISS và backend.
Các phương án sử dụng chung API 2.0.0 và các namespace sau:

| Phương án | dataset | `source_id` | split | Mẫu metadata |
|---|---|---|---|---|
| COCO Captions 2017, split gốc | `coco2017` | images[].id chuyển sang chuỗi thập phân | train hoặc val theo tệp annotation | [COCO 2017](../contracts/datasets/coco2017.metadata.example.jsonl) |
| COCO Captions 2014, split gốc | `coco2014` | images[].id chuyển sang chuỗi thập phân | train hoặc val theo tệp annotation | [COCO 2014](../contracts/datasets/coco2014.metadata.example.jsonl) |
| Flickr30k, chưa áp dụng bộ chia | flickr30k | Tên tệp ảnh bỏ phần mở rộng | unspecified | [Flickr30k](../contracts/datasets/flickr30k.metadata.example.jsonl) |
| Flickr30k, split Karpathy | `flickr30k_karpathy` | Tên tệp ảnh bỏ phần mở rộng | Theo images[].split trong annotation Karpathy | [Flickr30k Karpathy](../contracts/datasets/flickr30k_karpathy.metadata.example.jsonl) |

contracts/datasets chứa mẫu metadata và request. ID, kích thước và caption là dữ liệu minh họa;
không có tệp ảnh đi kèm.

### 1.1. Nguồn annotation COCO được chấp nhận

Nguồn annotation COCO của đặc tả:

| Bản phát hành | URL tải annotation | Tệp caption sử dụng sau giải nén |
|---|---|---|
| COCO 2014 | [annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) | annotations/captions_train2014.json, annotations/captions_val2014.json |
| COCO 2017 | [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) | annotations/captions_train2017.json, annotations/captions_val2017.json |

Quy định kiểm tra nguồn trước chuyển đổi:

1. URL tải annotation phải trùng một trong hai URL trên và đúng bản phát hành đã chọn.
2. Nếu tải từ URL khác, **dừng chuyển đổi metadata và build index**. Chỉ tiếp tục sau khi nhóm kiểm tra nguồn, cấu trúc và cập nhật đặc tả.
3. Nếu không xác định được URL nguồn, dừng và xác minh nguồn trước khi xử lý.
4. Ghi URL đã sử dụng vào `source_urls` của `dataset-selection.json`, cùng checksum gói ZIP và các tệp caption đã giải nén trong `source_checksums`.
5. Không dùng instances hoặc person_keypoints thay cho `captions` dù cùng nằm trong gói annotation.

Hai gói chỉ chứa annotation. Ảnh tải riêng, theo cùng bản phát hành.
Danh sách URL này áp dụng cho COCO; nguồn Flickr30k được quy định tại mục 4.

Bước chuyển đổi dữ liệu phải kiểm tra và ghi nhận URL nguồn.
Validator metadata hiện chưa kiểm tra URL tải.

## 2. COCO Captions 2017

### 2.1. Đầu vào

Nguồn: [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
Áp dụng quy định kiểm tra nguồn tại mục 1.1.

| Nội dung | Tệp / thư mục |
|---|---|
| Caption tập train | annotations/captions_train2017.json |
| Caption tập val | annotations/captions_val2017.json |
| Ảnh tập train | images/train2017/ |
| Ảnh tập val | images/val2017/ |

Vị trí images/ là quy ước tổ chức sau giải nén của dự án.
Đầu vào yêu cầu annotation caption. Annotation instance không chứa dữ liệu caption cần thiết.

### 2.2. Ánh xạ trường

| Trường Metadata | Quy tắc |
|---|---|
| `schema_version` | 1.0.0 |
| dataset | `coco2017` |
| `source_id` | str(image.id); không lấy annotation.id hoặc chỉ số hàng |
| id | `coco2017:<source_id>` |
| split | captions_train2017.json: train; captions_val2017.json: val |
| `image_path` | images/train2017/<image.`file_name`> hoặc images/val2017/<image.`file_name`> |
| `thumbnail_path` | null khi chưa tạo; sau khi tạo: thumbnails/<split>/`<source_id>`.jpg |
| `width`, `height` | image.`width`, image.`height`; phải đối chiếu với kích thước ảnh giải mã |
| `captions` | Gom annotations có `image_id`=image.id; text=annotation.caption, language=en |
| license | Tra image.license trong licenses[].id để lấy tên; null nếu không xác định |
| source_url | image.coco_url; nếu không có thì image.flickr_url; nếu không có thì null |

Gom annotation theo `image_id`, tương ứng với cấu trúc [COCO API chính thức](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).
ID ảnh và tên tệp có thể có cách biểu diễn khác nhau: image.id=1 tạo `source_id`="1",
trong khi `file_name`="000000000001.jpg" phải được giữ nguyên trong `image_path`.

### 2.3. Quy tắc chuyển đổi

1. Lập ánh xạ image.id sang danh sách caption; sắp caption theo annotation.id tăng dần.
2. Mỗi ảnh tạo đúng một bản ghi Metadata.
3. Caption được giữ nguyên nội dung; không dịch hoặc nối vào một caption duy nhất.
4. Annotation tham chiếu ảnh không tồn tại, ID trùng, caption rỗng hoặc ảnh không giải mã được phải được báo lỗi.
5. Ảnh không có caption không hợp lệ với Metadata 1.0.0. Phải loại khỏi tập đầu ra và ghi vào báo cáo chuyển đổi.
6. Tệp chỉ có thông tin ảnh test, không có caption, không được sử dụng trực tiếp cho BM25 hoặc đánh giá caption retrieval.
7. Số caption được lấy từ annotation thực tế; không tự bổ sung để đạt một số lượng cố định.

### 2.4. Tổ chức đầu ra

```text
data/processed/coco2017/
  metadata.jsonl
  dataset-selection.json
  conversion-report.json
  images/
    train2017/
    val2017/
  thumbnails/
    train/
    val/
```

`image_path` và `thumbnail_path` được phân giải tương đối từ data/processed/coco2017.
Có thể chỉ sử dụng tập val cho phạm vi demo; khi đó vẫn đặt split=val và ghi phạm vi này
trong `dataset-selection.json`. Giá trị split phải giữ theo annotation nguồn.

## 3. COCO Captions 2014

Nguồn: [annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).
Áp dụng quy định kiểm tra nguồn tại mục 1.1.

Áp dụng quy tắc mục 2 với các thay thế:

| Thuộc tính | Giá trị |
|---|---|
| dataset | `coco2014` |
| Annotation | captions_train2014.json, captions_val2014.json |
| `image_path` | images/train2014/<`file_name`> hoặc images/val2014/<`file_name`> |
| `dataset_root` | data/processed/coco2014 |
| id | `coco2014:<source_id>` |

Giữ nguyên `file_name` của annotation, bao gồm tiền tố COCO_train2014 hoặc COCO_val2014 nếu có.
Ảnh và annotation phải thuộc cùng bản phát hành.

Nếu chọn split Karpathy cho COCO, phải bổ sung phương án riêng với namespace
coco2014_karpathy, annotation và quy tắc restval đã được xác nhận. Phương án COCO 2014
trong tài liệu này chỉ áp dụng split gốc.

### 3.1. Đối chiếu annotation train 2014 và 2017

Kết quả đối chiếu ngày 13/10/2026 trên hai tệp captions_train2014.json và
captions_train2017.json do nhóm cung cấp. Phạm vi đối chiếu là annotation tập train;
không bao gồm tập val hoặc tệp ảnh.

| Nội dung | COCO train 2014 | COCO train 2017 |
|---|---|---|
| Trường cấp gốc | info, images, licenses, annotations | Cùng bốn trường |
| Trường trong images | id, `file_name`, `width`, `height`, license, coco_url, flickr_url, date_captured | Giống 2014 |
| Trường trong annotations | `image_id`, id, caption | Giống 2014 |
| Trường trong licenses | id, name, url | Giống 2014 |
| Số ảnh | 82.783 | 118.287 |
| Số caption | 414.113 | 591.753 |
| Số giấy phép | 8 | 8 |
| info.year | 2014 | 2017 |
| info.version | 1.0 | 1.0 |
| info.date_created | 2017/09/01 | 2017/09/01 |

Thứ tự trường cấp gốc khác nhau giữa hai tệp. Chương trình phải truy cập theo tên trường,
không phụ thuộc thứ tự. Danh sách licenses của hai tệp giống nhau.
info.year xác định bản dataset; info.date_created không được sử dụng thay cho năm phát hành.

Toàn bộ 82.783 ID ảnh train2014 có trong train2017. Tập train2017 có thêm
35.504 ID ảnh so với train2014. Kích thước width/height trong annotation
của các ảnh chung không thay đổi. Phạm vi kết quả: hai tệp annotation đã đối chiếu.

#### Tên tệp và URL ảnh

Cùng ảnh ID 57870:

```json
{
  "2014": {
    "id": 57870,
    "file_name": "COCO_train2014_000000057870.jpg",
    "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg"
  },
  "2017": {
    "id": 57870,
    "file_name": "000000057870.jpg",
    "coco_url": "http://images.cocodataset.org/train2017/000000057870.jpg"
  }
}
```

Script chuyển đổi phải lấy `file_name` trực tiếp từ annotation. Ảnh được tải từ gói cùng
bản phát hành với annotation; không dùng tên tệp 2014 để tìm ảnh trong thư mục 2017.

#### Số caption của mỗi ảnh

| Số caption / ảnh | Số ảnh train2014 | Số ảnh train2017 |
|---|---|---|
| 5 | 82.586 | 117.972 |
| 6 | 196 | 312 |
| 7 | 1 | 3 |

Script phải gom toàn bộ caption theo `image_id`, không giới hạn ở năm caption.
Không có ID ảnh hoặc ID annotation trùng trong mỗi tệp. Mọi `image_id` của annotation
đều tham chiếu ảnh tồn tại trong tệp tương ứng.

#### Áp dụng vào Metadata 1.0.0

Cấu trúc metadata và API 2.0.0 không thay đổi khi chuyển từ COCO 2014 sang 2017.
Cả hai phương án dùng chung phép ghép `annotations[].image_id` = images[].id.

| Trường | COCO 2014 | COCO 2017 |
|---|---|---|
| `schema_version` | 1.0.0 | 1.0.0 |
| dataset | `coco2014` | `coco2017` |
| `source_id` của ảnh ví dụ | 57870, kiểu string | 57870, kiểu string |
| id | `coco2014`:57870 | `coco2017`:57870 |
| split của tập train | train | train |
| `image_path` | images/train2014/COCO_train2014_000000057870.jpg | images/train2017/000000057870.jpg |
| `captions` | Gom theo `image_id` từ annotation 2014 | Gom theo `image_id` từ annotation 2017 |
| source_url | coco_url trong annotation 2014 | coco_url trong annotation 2017 |

Các namespace khác nhau có thể chứa cùng ảnh.
Nếu kết hợp cả hai bản trong một tập tìm kiếm hoặc thực nghiệm, phải xử lý ảnh trùng và
kiểm tra giao nhau giữa các split trước khi đánh giá.

## 4. Flickr30k

### 4.1. Đầu vào

| Thành phần | Link tải | Tệp / thư mục dùng để chuyển đổi |
|---|---|---|
| Ảnh Flickr30k | [Flickr30k images trên Box](https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl) | flickr30k-images |
| Annotation Karpathy | [caption_datasets.zip](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) | dataset_flickr30k.json |

Cặp ảnh/annotation này sử dụng namespace flickr30k_karpathy theo mục 4.3.
Ghi cả hai URL vào source_urls của dataset-selection.json và ghi checksum các gói tải
cùng file annotation vào source_checksums. Annotation dạng token tại mục 4.2 là phương án riêng.


- Thư mục ảnh Flickr30k.
- Tệp caption và mô tả định dạng của bản tải về.
- Tệp hoặc annotation phân chia train/val/test, nếu áp dụng.

Nguồn dữ liệu: [Flickr30k của nhóm tác giả](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html).
Nguồn annotation/split Karpathy: [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/).

Đầu vào phải có tệp ảnh tương ứng để đọc kích thước và phục vụ ảnh qua backend.

### 4.2. Ánh xạ caption dạng token

Áp dụng khi tệp đầu vào có cấu trúc mỗi dòng:

```text
<filename>#<caption_index><TAB><caption>
```

| Trường Metadata | Quy tắc |
|---|---|
| dataset | flickr30k khi chưa áp dụng bộ chia |
| `source_id` | Stem của `filename`; giữ nguyên số 0 đầu |
| id | <dataset>:`<source_id>` |
| split | unspecified khi chưa có bộ chia được xác nhận |
| `image_path` | images/<`filename`> |
| `thumbnail_path` | null hoặc thumbnails/`<source_id>`.jpg |
| `width`, `height` | Đọc từ ảnh gốc đã giải mã |
| `captions` | Gom theo `filename`, sắp caption_index tăng dần; text=caption, language=en |
| license | null nếu chưa xác định giấy phép của ảnh tương ứng |
| source_url | URL ảnh gốc nếu bản dữ liệu có cung cấp; ngược lại null |

Phải tách dòng ở dấu TAB đầu tiên và tách filename/caption_index ở dấu # cuối cùng.
Không dùng toàn bộ `filename`#caption_index làm `source_id`.
Tên tệp 000001.jpg tạo `source_id`="000001", không chuyển thành "1".
Dòng lỗi định dạng hoặc caption_index trùng trong cùng ảnh phải được báo lỗi.

### 4.3. Ánh xạ annotation Karpathy

Áp dụng cho `dataset_flickr30k.json` trong gói annotation đã lựa chọn.

| Trường Metadata | Nguồn / giá trị |
|---|---|
| dataset | `flickr30k_karpathy` |
| `source_id` | Stem của `images[].filename` |
| id | `flickr30k_karpathy`:`<source_id>` |
| split | images[].split; chỉ nhận train, val, test |
| `image_path` | images/<`images[].filename`> |
| `width`, `height` | Đọc ảnh gốc |
| `captions` | `images[].sentences`[].`raw`; language=en; giữ thứ tự `sentences` |
| `thumbnail_path` | null hoặc thumbnails/`<source_id>`.jpg |
| license, source_url | Áp dụng mục 4.2 |

Không lấy `imgid` hoặc `sentid` thay cho ID từ tên tệp.
Nếu annotation không có `raw`, việc ghép `tokens` phải được ghi thành quy tắc chuyển đổi riêng.
Giá trị split ngoài danh sách quy định phải được báo lỗi, không tự chuyển sang train.

### 4.4. Tổ chức đầu ra

```text
data/processed/flickr30k_karpathy/
  metadata.jsonl
  dataset-selection.json
  conversion-report.json
  images/
  thumbnails/
```

Khi chưa có bộ chia, sử dụng thư mục và namespace flickr30k.
Khi áp dụng bộ chia, mỗi ảnh phải thuộc đúng một split; các split không được giao nhau.
Subset phục vụ demo phải ghi danh sách ID hoặc checksum của danh sách ID và vẫn giữ split gốc.

### 4.5. Ánh xạ với thư mục `flickr30k-images`

Áp dụng cho thư mục ảnh giải nén có tên `flickr30k-images`.
Annotation sử dụng `dataset_flickr30k.json`, theo phương án Flickr30k Karpathy tại mục 4.3.

#### Thư mục gốc dữ liệu

Ví dụ bố trí:

```text
<dataset_root>/
  dataset_flickr30k.json
  flickr30k-images/
    1000092795.jpg
    10002456.jpg
    ...
  metadata.jsonl
```

`dataset_root` là thư mục cha của `flickr30k-images`. Annotation có thể được lưu ở vị trí
khác nếu script nhận đường dẫn riêng. Vị trí annotation không quyết định `dataset_root`.

#### Bản ghi nguồn

Trích các trường dùng để chuyển đổi từ ảnh đầu tiên trong annotation đã kiểm tra.
Trích đoạn giữ đủ năm caption; lược bỏ `tokens`, `sentid` và `sentids`.

```json
{
  "filename": "1000092795.jpg",
  "imgid": 0,
  "split": "train",
  "sentences": [
    {
      "raw": "Two young guys with shaggy hair look at their hands while hanging out in the yard."
    },
    {
      "raw": "Two young, White males are outside near many bushes."
    },
    {
      "raw": "Two men in green shirts are standing in a yard."
    },
    {
      "raw": "A man in a blue shirt standing in a garden."
    },
    {
      "raw": "Two friends enjoy time spent together."
    }
  ]
}
```

Script ghép ảnh bằng `filename`:

```text
images[].filename = 1000092795.jpg
Tệp ảnh = <dataset_root>/flickr30k-images/1000092795.jpg
```

`imgid`=0 là định danh trong annotation. Project sử dụng tên tệp bỏ phần mở rộng
làm `source_id` để ánh xạ trực tiếp đến ảnh.

#### Metadata đích

| Trường | Giá trị / cách xác định |
|---|---|
| `schema_version` | 1.0.0 |
| dataset | `flickr30k_karpathy`, theo quy ước project cho annotation có split Karpathy |
| `source_id` | "1000092795", lấy từ `filename` và giữ kiểu string |
| id | `flickr30k_karpathy`:1000092795 |
| split | train, giữ images[].split |
| `image_path` | flickr30k-images/1000092795.jpg |
| `thumbnail_path` | null khi chưa tạo thumbnail |
| `width`, `height` | Đọc kích thước pixel từ tệp ảnh đã giải mã |
| `captions` | Chuyển từng `sentences[].raw` thành {text: `raw`, language: "en"}, giữ thứ tự và đủ năm caption |
| license | null nếu chưa có thông tin giấy phép được xác định |
| source_url | null nếu chưa có URL ảnh nguồn được xác định |

width/height phải được đọc từ ảnh. Thiếu ảnh, lỗi giải mã hoặc kích thước không hợp lệ
làm bước chuyển đổi thất bại theo quy định đầu vào ML.

#### Quy tắc đường dẫn

| Bố trí ảnh | `dataset_root` | `image_path` |
|---|---|---|
| Giữ `flickr30k-images` trong thư mục dữ liệu | Thư mục cha của `flickr30k-images` | flickr30k-images/1000092795.jpg |
| Chuyển ảnh vào images trong thư mục dữ liệu | Thư mục cha của images | images/1000092795.jpg |

Cả hai bố trí dùng Metadata 1.0.0. `image_path` phải khớp vị trí tệp và được phân giải
từ `dataset_root`. Không lưu đường dẫn ổ đĩa Windows trong metadata.
Nếu thay đổi vị trí ảnh sau khi chuyển đổi, phải cập nhật đường dẫn và kiểm tra lại metadata.

Đối với bố trí giữ `flickr30k-images`, mục 4.5 thay quy ước images/ trong ví dụ tại mục 4.3–4.4.
Các trường ID, split và caption vẫn áp dụng cùng quy tắc.

## 5. Tích hợp API

`SearchRequest` sử dụng trường dataset để lọc bộ dữ liệu. Ví dụ với COCO 2017:

```json
{
  "query": "a dog on the beach",
  "method": "clip",
  "top_k": 20,
  "offset": 0,
  "limit": 10,
  "min_similarity": null,
  "dataset": "coco2017",
  "split": "val"
}
```

| Dataset | `SearchRequest`.dataset | `SearchRequest`.split | Ví dụ path ảnh |
|---|---|---|---|
| COCO 2017 | `coco2017` | train, val hoặc null | /images/coco2017:1 |
| COCO 2014 | `coco2014` | train, val hoặc null | /images/coco2014:1 |
| Flickr30k chưa chia | flickr30k | unspecified hoặc null | /images/flickr30k:000001 |
| Flickr30k Karpathy | `flickr30k_karpathy` | train, val, test hoặc null | /images/flickr30k_karpathy:000001 |

BM25 dùng cùng request; đặt method=bm25 và `min_similarity`=null.
Tìm ảnh tương tự sử dụng /images/{id}/similar; dataset được xác định từ ID ảnh nguồn,
không bổ sung dataset vào query của endpoint này.

`ImagePublic` giữ id, dataset, split, `width`, `height` và `captions` từ Metadata.
Backend tạo `image_url` và `thumbnail_url` từ cơ chế phục vụ ảnh; không trả `image_path`
hoặc `thumbnail_path` dưới dạng đường dẫn tệp nội bộ. Nếu chưa có thumbnail, backend có thể
trả URL ảnh gốc trong `thumbnail_url`. URL phải tải được từ môi trường frontend.

Request mẫu: contracts/datasets/*.search.example.json.
Server mock hiện chỉ nạp namespace demo. Gửi bộ lọc coco2017/flickr30k vào mock cho kết quả
rỗng; các ID minh họa tương ứng trả 404. Backend chưa hỗ trợ nạp các dataset này.

## 6. Thông tin lựa chọn và bàn giao

Sao chép [mẫu lựa chọn dataset](../contracts/datasets/dataset-selection.template.json)
thành `dataset-selection.json` trong thư mục đầu ra và điền:

| Trường | Nội dung |
|---|---|
| dataset | Namespace theo mục 1 |
| release | Bản phát hành cụ thể |
| source_annotation_format | coco_captions, flickr_token hoặc karpathy_json |
| split_protocol | Quy tắc phân chia: split gốc, Karpathy hoặc chưa chia |
| selected_splits | Các split có trong metadata đầu ra |
| subset | null nếu toàn bộ; nếu subset, ghi cách chọn và tệp danh sách ID |
| `dataset_root` | Thư mục gốc dùng để phân giải đường dẫn |
| metadata_file | Tệp metadata đầu ra |
| `source_urls` | Nguồn tải ảnh và annotation |
| `source_checksums` | Checksum các tệp annotation/bộ chia/danh sách subset |
| usage_terms_url | Nguồn điều kiện sử dụng |

`dataset-selection.json` là thông tin bàn giao, không thuộc schema Metadata và chưa được
server đọc tự động. Giá trị `CHUA_CHON` trong mẫu phải được thay bằng namespace đã lựa chọn.
Không thêm các trường này vào từng dòng metadata.

`conversion-report.json` ghi số ảnh và annotation đầu vào, số ảnh hợp lệ, số bản ghi
đầu ra, số ảnh theo split và danh sách lỗi hoặc bản ghi bị loại.

Ví dụ annotation nguồn, metadata và response API: mục 7–8 của [Đặc tả API](03_ENDPOINT_CONTRACTS.md).
