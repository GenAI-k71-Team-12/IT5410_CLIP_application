# Quy ước lưu trữ và chia sẻ dữ liệu

Tài liệu áp dụng cho mọi thành viên và agent khi tải, tạo, dùng hoặc phát hành dữ liệu của dự án.
Cấu trúc metadata và artifact theo [Đặc tả dữ liệu](02_DATA_API_CONTRACT.md),
[Đặc tả ánh xạ dataset](04_DATASET_MAPPING.md) và [Đặc tả tích hợp ML](05_ML_INTEGRATION_CONTRACT.md).

## 1. Nguyên tắc

| Nguyên tắc | Quy định |
|---|---|
| Git | Chứa code, contract, tài liệu và `data/examples`. Không chứa dataset, ảnh, embedding, index hoặc weights |
| Tái lập | Mọi dữ liệu dùng chung tạo lại được bằng lệnh đã ghi trong tài liệu và release notes |
| Định danh | Dữ liệu chuyển đổi định danh theo namespace `<dataset>`; artifact theo `artifact_version` |
| Bất biến | Phiên bản đã phát hành không được sửa hoặc ghi đè. Thay đổi thì tăng `v<N>` |
| Kiểm tra | Dữ liệu tải về phải qua lệnh kiểm tra trước khi dùng |

## 2. Bố trí thư mục local

Mọi đường dẫn tính từ thư mục gốc repo:

```text
data/
  examples/                  # Git: dữ liệu mẫu cho chế độ mock
  raw/<dataset>/             # Dữ liệu gốc tải từ nguồn chính thức
  processed/<dataset>/       # dataset_root của namespace
    metadata.jsonl
    dataset-selection.json
    conversion-report.json
    images/                  # bố trí theo đặc tả ánh xạ dataset
    thumbnails/
artifacts/
  <artifact_version>/        # manifest.json, metadata.jsonl, embeddings.npy, image_ids.json,
                             # index.faiss, tokenizer_config.json, preprocess_config.json
  .cache/                    # cache embedding ảnh theo fingerprint model; không chia sẻ
```

`data/raw`, `data/processed` và `artifacts` đã nằm trong `.gitignore`.

## 3. Nơi lưu từng loại dữ liệu

| Loại | Vị trí local | Chia sẻ | Người phát hành |
|---|---|---|---|
| Dữ liệu gốc (ảnh, annotation) | `data/raw/<dataset>/` | Không chia sẻ. Mỗi người tải từ URL chính thức trong đặc tả ánh xạ dataset | — |
| Metadata đã chuyển đổi | `data/processed/<dataset>/` | Release `data-<dataset>-v<N>` | Phụ trách dữ liệu (Quang) |
| Ảnh, thumbnail đã chuyển đổi | `data/processed/<dataset>/images/`, `thumbnails/` | Không đăng (repo public). Tạo lại bằng script chuyển đổi và script thumbnail | — |
| Artifact | `artifacts/<artifact_version>/` | Release `artifact-<artifact_version>` | Phụ trách ML (Quân) |
| Cache embedding | `artifacts/.cache/` | Không chia sẻ | — |
| Weights CLIP | Cache Hugging Face (`~/.cache/huggingface`, đổi bằng `HF_HOME`) | Tự tải khi chạy. Manifest ghi SHA-256 | — |

## 4. GitHub Releases

### 4.1. Tag và tệp

| Loại | Tag | Assets |
|---|---|---|
| Dữ liệu chuyển đổi | `data-<dataset>-v<N>` | `metadata.jsonl`, `dataset-selection.json`, `conversion-report.json` |
| Artifact | `artifact-<artifact_version>` | Toàn bộ tệp trong `artifacts/<artifact_version>/`, mỗi tệp một asset, không nén |

`artifact_version` có dạng `<dataset>-<split>-<model>-v<N>`, chữ thường:

| Thành phần | Quy định | Ví dụ |
|---|---|---|
| `<dataset>` | Namespace metadata | `coco2017` |
| `<split>` | Split được lập chỉ mục; nhiều split nối bằng `_` theo thứ tự chữ cái | `val`, `test_val` |
| `<model>` | Model viết gọn | `vitb32`, `vitb16` |
| `v<N>` | Tăng khi build lại với bất kỳ thay đổi nào | `v1` |

Ví dụ: `coco2017-val-vitb32-v1`. Subset không đưa vào tên; manifest ghi `subset_ids_sha256`.

### 4.2. Tải về và kiểm tra

```bash
gh release download data-coco2017-v1 -D data/processed/coco2017
gh release download artifact-coco2017-val-vitb32-v1 -D artifacts/coco2017-val-vitb32-v1
python -m clip_retrieval.ml.artifact artifacts/coco2017-val-vitb32-v1 --check-model
python scripts/validate_metadata.py data/processed/coco2017/metadata.jsonl --data-root data/processed/coco2017
```

Không có `gh`: tải asset từ trang Releases của repo và đặt vào đúng thư mục trên.
Lệnh kiểm tra artifact báo `Invalid artifact: ...` khi tệp, checksum, ánh xạ ID
hoặc model không khớp manifest. Khi đó không dùng artifact; tải lại hoặc báo người phát hành.
Lệnh `validate_metadata.py` cần ảnh đã được tạo theo mục 3.

### 4.3. Phát hành

1. Build từ commit đã có trên `dev` hoặc `main`, working tree sạch.
2. Build và kiểm tra:

   ```bash
   python -m clip_retrieval.ml.build_index --metadata data/processed/coco2017/metadata.jsonl \
     --dataset-root data/processed/coco2017 --dataset coco2017 --splits val \
     --artifact-version coco2017-val-vitb32-v1 --device cuda
   python -m clip_retrieval.ml.artifact artifacts/coco2017-val-vitb32-v1 --check-model
   ```

3. Tạo release, ghi trong notes: commit build, lệnh build, dòng kết quả `Built ...`, dòng `Valid artifact ...`:

   ```bash
   gh release create artifact-coco2017-val-vitb32-v1 artifacts/coco2017-val-vitb32-v1/* \
     --target <commit> --title "artifact coco2017-val-vitb32-v1" --notes-file release-notes.md
   ```

4. Cập nhật bảng mục 5 trong pull request kế tiếp và báo nhóm.

## 5. Phiên bản đang dùng

| Mục | Giá trị |
|---|---|
| Dataset | Chưa chốt (chờ phần tải và chuẩn hóa dataset) |
| Data release | Chưa phát hành |
| Artifact release | Chưa phát hành |

Thử nghiệm, đánh giá và demo dùng các phiên bản trong bảng này.
Kết quả thí nghiệm phải ghi kèm `artifact_version` và commit đã chạy.

## 6. Môi trường ML

Sau các lệnh cài đặt trong [README](../README.md):

```bash
python -m pip install -r requirements-ml.txt
```

Bản `torch` trên PyPI cho Windows là bản CPU. Muốn dùng GPU: cài `torch` bản CUDA cùng phiên bản
trong `requirements-ml.txt` theo hướng dẫn tại https://pytorch.org/get-started/locally/,
rồi mới cài `requirements-ml.txt`, và chạy build với `--device cuda`.
Test trong `tests/ml` tự bỏ qua khi chưa cài dependency ML.

## 7. Quy tắc cho agent

- Không commit, xóa hoặc sửa `data/raw`, `data/processed`, `artifacts`.
- Không sửa thư mục artifact đã tải về. Muốn thay đổi thì build `artifact_version` mới.
- Đọc phiên bản đang dùng ở mục 5 trước khi chạy thử nghiệm; không tự chọn artifact khác.
- Không tải dữ liệu từ nguồn khác URL trong đặc tả ánh xạ dataset.
- Không đăng ảnh, weights hoặc dữ liệu cá nhân lên release.
