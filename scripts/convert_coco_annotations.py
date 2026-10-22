"""Chuyển đổi annotation COCO Captions thành Metadata 1.0.0 (JSONL).

Áp dụng quy tắc tại docs/04_DATASET_MAPPING.md mục 2 (COCO Captions 2017) và mục 3
(COCO Captions 2014). Script chỉ dùng tệp captions_*.json; không dùng instances hoặc
person_keypoints (mục 1.1 quy tắc 5).

Mọi bản ghi đầu ra được kiểm tra bằng model Pydantic Metadata nên tự động tuân thủ
đặc tả dữ liệu 1.0.0.

Ví dụ:

    python scripts/convert_coco_annotations.py \\
        --annotation D:/study/nam_5/LLM/annotations/captions_val2017.json \\
        --annotation D:/study/nam_5/LLM/annotations/captions_train2017.json \\
        --dataset coco2017 --release 2017 \\
        --output-dir data/processed/coco2017
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clip_retrieval.contracts import Caption, Metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_TEMPLATE = REPO_ROOT / "contracts/datasets/dataset-selection.template.json"

# URL tải annotation được đặc tả chấp nhận tại docs/04_DATASET_MAPPING.md mục 1.1.
ALLOWED_SOURCE_URLS = {
    "2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
USAGE_TERMS_URLS = {
    "2014": "https://cocodataset.org/#termsofuse",
    "2017": "https://cocodataset.org/#termsofuse",
}
IMAGE_ROOT_URLS = {
    "2014": "http://images.cocodataset.org/zips/train2014.zip",
    "2017": "http://images.cocodataset.org/zips/train2017.zip",
}
IMAGE_ROOT_URLS_VAL = {
    "2014": "http://images.cocodataset.org/zips/val2014.zip",
    "2017": "http://images.cocodataset.org/zips/val2017.zip",
}

ANNOTATION_NAME = re.compile(r"^captions_(train|val)(2014|2017)\.json$")
SPLIT_VALUES = ("train", "val", "test", "unspecified")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ConversionError(Exception):
    """Lỗi dữ liệu nguồn làm bước chuyển đổi thất bại."""


def sha256_of(path: Path) -> str:
    """Checksum SHA-256 của tệp, đọc theo khối để không nạp toàn bộ vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_split_and_year(
    path: Path, override_split: str | None, override_year: int | None
) -> tuple[str, int]:
    """Suy ra split và năm phát hành từ tên tệp annotation."""
    match = ANNOTATION_NAME.match(path.name)
    if match:
        split, year = match.group(1), int(match.group(2))
    else:
        if override_split is None or override_year is None:
            raise ConversionError(
                f"Không suy ra được split/năm từ tên tệp '{path.name}'. "
                "Tên tệp phải có dạng captions_<split><year>.json, "
                "hoặc dùng --split và --release."
            )
        split, year = override_split, override_year
    if override_split is not None:
        split = override_split
    if override_year is not None:
        year = override_year
    if split not in SPLIT_VALUES:
        raise ConversionError(f"Split không hợp lệ '{split}'. Chỉ nhận {SPLIT_VALUES}.")
    return split, year


def load_coco(path: Path) -> dict[str, Any]:
    """Nạp tệp annotation COCO và kiểm tra các trường cấp gốc bắt buộc."""
    print(f"Đang nạp {path} ...", flush=True)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("info", "images", "licenses", "annotations"):
        if key not in payload:
            raise ConversionError(f"Tệp annotation '{path.name}' thiếu trường cấp gốc '{key}'.")
    return payload


def index_licenses(coco: dict[str, Any]) -> dict[int, str]:
    """Ánh xạ licenses[].id sang licenses[].name."""
    licenses: dict[int, str] = {}
    for item in coco["licenses"]:
        licenses[int(item["id"])] = str(item.get("name") or "") or None
    return licenses


def group_captions(coco: dict[str, Any]) -> tuple[dict[int, list[str]], list[str]]:
    """Gom caption theo image_id, sắp theo annotation.id tăng dần.

    Trả về danh sách lỗi thay vì ném ngoại lệ để báo cáo đầy đủ trong một lần chạy.
    """
    errors: list[str] = []
    by_image: dict[int, list[tuple[int, str]]] = defaultdict(list)
    seen_annotation_ids: set[int] = set()

    for annotation in coco["annotations"]:
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        text = str(annotation.get("caption", ""))
        if annotation_id in seen_annotation_ids:
            errors.append(f"annotation id trùng: {annotation_id}")
            continue
        seen_annotation_ids.add(annotation_id)
        if not text.strip():
            errors.append(f"caption rỗng: annotation {annotation_id}, image {image_id}")
            continue
        by_image[image_id].append((annotation_id, text))

    grouped = {
        image_id: [text for _, text in sorted(items, key=lambda pair: pair[0])]
        for image_id, items in by_image.items()
    }
    return grouped, errors


def build_records(
    images: list[dict[str, Any]],
    grouped: dict[int, list[str]],
    licenses: dict[int, str],
    dataset: str,
    split: str,
    year: int,
) -> tuple[list[Metadata], list[str], list[dict[str, Any]]]:
    """Tạo bản ghi Metadata từ images[] và caption đã gom.

    Trả về (bản ghi hợp lệ, lỗi, ảnh bị loại do không có caption).
    """
    errors: list[str] = []
    excluded: list[dict[str, Any]] = []
    records: list[Metadata] = []
    seen_image_ids: set[int] = set()

    for image in images:
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            errors.append(f"image id trùng trong images[]: {image_id}")
            continue
        seen_image_ids.add(image_id)

        captions = grouped.get(image_id)
        if not captions:
            excluded.append(
                {
                    "image_id": str(image_id),
                    "file_name": image.get("file_name"),
                    "reason": "Ảnh không có caption, không hợp lệ với Metadata 1.0.0",
                }
            )
            continue

        source_id = str(image_id)
        license_id = image.get("license")
        license_name = licenses.get(int(license_id)) if license_id is not None else None
        source_url = image.get("coco_url") or image.get("flickr_url") or None

        try:
            records.append(
                Metadata(
                    id=f"{dataset}:{source_id}",
                    dataset=dataset,
                    source_id=source_id,
                    split=split,
                    image_path=f"images/{split}{year}/{image['file_name']}",
                    thumbnail_path=None,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    captions=[Caption(text=text, language="en") for text in captions],
                    license=license_name,
                    source_url=source_url,
                )
            )
        except Exception as exc:  # noqa: BLE001 - lỗi Pydantic phải được báo cáo, không dừng cả lô
            errors.append(f"image {source_id}: {exc}")

    for image_id in grouped:
        if image_id not in seen_image_ids:
            errors.append(f"annotation tham chiếu ảnh không tồn tại: image_id={image_id}")

    return records, errors, excluded


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Đọc kích thước ảnh từ header, không cần thư viện ngoài.

    Hỗ trợ JPEG (marker SOF) và PNG (chunk IHDR). COCO có tệp mang đuôi .jpg
    nhưng thực chất là PNG, nên phải nhận dạng theo chữ ký tệp thay vì đuôi tệp.
    Trả về None nếu không đọc được kích thước.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
            if header.startswith(PNG_SIGNATURE):
                # 8 byte chữ ký, 4 byte độ dài chunk, "IHDR", rồi width/height 4 byte.
                if len(header) < 24:
                    return None
                return (
                    int.from_bytes(header[16:20], "big"),
                    int.from_bytes(header[20:24], "big"),
                )
            if not header.startswith(b"\xff\xd8"):
                return None
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if marker_start != b"\xff":
                    return None
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in (b"\xd8", b"\xd9"):
                    continue
                size_bytes = handle.read(2)
                if len(size_bytes) != 2:
                    return None
                segment_size = int.from_bytes(size_bytes, "big")
                if marker[0] in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }:
                    payload = handle.read(5)
                    if len(payload) != 5:
                        return None
                    height = int.from_bytes(payload[1:3], "big")
                    width = int.from_bytes(payload[3:5], "big")
                    return width, height
                handle.seek(segment_size - 2, 1)
    except OSError:
        return None


def verify_images(
    records: list[Metadata], dataset_root: Path, check_size: bool
) -> tuple[list[str], int]:
    """Kiểm tra ảnh tồn tại dưới dataset_root và kích thước có khớp metadata."""
    errors: list[str] = []
    missing = 0
    for record in records:
        path = dataset_root / record.image_path
        if not path.is_file():
            missing += 1
            errors.append(f"thiếu tệp ảnh: {record.image_path}")
            continue
        if check_size:
            size = read_image_size(path)
            if size is None:
                errors.append(f"không giải mã được ảnh: {record.image_path}")
            elif size != (record.width, record.height):
                errors.append(
                    f"kích thước không khớp: {record.image_path} "
                    f"ảnh={size[0]}x{size[1]} metadata={record.width}x{record.height}"
                )
    return errors, missing


def write_metadata(records: list[Metadata], path: Path) -> None:
    """Ghi metadata.jsonl, sắp theo Metadata.id tăng dần theo thứ tự chuỗi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":"))
            handle.write(f"{payload}\n")


def create_output_dirs(output_dir: Path, splits: list[str], year: int | None) -> None:
    """Tạo cấu trúc thư mục đầu ra theo docs/04_DATASET_MAPPING.md mục 2.4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if year is not None:
            (output_dir / "images" / f"{split}{year}").mkdir(parents=True, exist_ok=True)
        (output_dir / "thumbnails" / split).mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    """Trả về đường dẫn tương đối so với thư mục repo nếu nằm trong repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_dataset_selection(
    path: Path,
    dataset: str,
    release: str,
    splits: list[str],
    dataset_root: Path,
    metadata_file: Path,
    source_urls: list[str],
    source_checksums: dict[str, str],
) -> None:
    """Sinh dataset-selection.json từ mẫu trong contracts/datasets."""
    template = json.loads(SELECTION_TEMPLATE.read_text(encoding="utf-8"))
    template.update(
        {
            "dataset": dataset,
            "release": release,
            "source_annotation_format": "coco_captions",
            "split_protocol": "split gốc của bộ dữ liệu",
            "selected_splits": splits,
            "subset": None,
            "dataset_root": display_path(dataset_root),
            "metadata_file": display_path(metadata_file),
            "source_urls": source_urls,
            "source_checksums": source_checksums,
            "usage_terms_url": USAGE_TERMS_URLS.get(release),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_report(
    path: Path,
    dataset: str,
    release: str,
    inputs: list[dict[str, Any]],
    records: list[Metadata],
    errors: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    """Sinh conversion-report.json theo yêu cầu mục 6 đặc tả ánh xạ dataset."""
    records_by_split = Counter(record.split for record in records)
    captions_by_split = Counter(record.split for record in records for _ in record.captions)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": dataset,
        "release": release,
        "inputs": inputs,
        "totals": {
            "input_images": sum(item["image_count"] for item in inputs),
            "input_annotations": sum(item["annotation_count"] for item in inputs),
            "valid_images": len(records),
            "output_records": len(records),
            "excluded_images": len(excluded),
            "errors": len(errors),
        },
        "records_by_split": dict(sorted(records_by_split.items())),
        "captions_by_split": dict(sorted(captions_by_split.items())),
        "errors": errors,
        "excluded": excluded,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        type=Path,
        action="append",
        required=True,
        metavar="FILE",
        help="Tệp captions_<split><year>.json. Lặp lại để nạp nhiều split.",
    )
    parser.add_argument("--dataset", required=True, help="Namespace, ví dụ coco2017.")
    parser.add_argument("--release", type=int, help="Năm phát hành: 2014 hoặc 2017.")
    parser.add_argument(
        "--split",
        choices=SPLIT_VALUES,
        help="Ghi đè split suy ra từ tên tệp (áp dụng cho mọi tệp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục đầu ra chứa metadata.jsonl, dataset-selection.json, báo cáo.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Thư mục gốc dùng để phân giải image_path. Mặc định bằng thư mục đầu ra.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Kiểm tra tệp ảnh tồn tại dưới dataset_root.",
    )
    parser.add_argument(
        "--verify-size",
        action="store_true",
        help="Kiểm tra kích thước ảnh (JPEG/PNG) khớp width/height của metadata.",
    )
    parser.add_argument(
        "--source-url",
        action="append",
        default=[],
        help="URL tải nguồn đã dùng. Mặc định lấy URL được đặc tả chấp nhận.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Chỉ xử lý N ảnh đầu mỗi tệp. Chỉ dùng để kiểm thử nhanh.",
    )
    return parser.parse_args()


def use_utf8_console() -> None:
    """Ép stdout/stderr dùng UTF-8 để in được tiếng Việt trên console Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    use_utf8_console()
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    dataset_root: Path = (args.dataset_root or args.output_dir).resolve()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.dataset):
        print(f"Namespace không hợp lệ: '{args.dataset}'", file=sys.stderr)
        return 1

    all_records: list[Metadata] = []
    all_errors: list[str] = []
    all_excluded: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    source_urls: list[str] = list(args.source_url)
    source_checksums: dict[str, str] = {}
    splits: list[str] = []
    release_years: set[int] = set()

    for annotation_path in args.annotation:
        annotation_path = annotation_path.resolve()
        if not annotation_path.is_file():
            print(f"Không tìm thấy tệp: {annotation_path}", file=sys.stderr)
            return 1
        try:
            split, year = detect_split_and_year(annotation_path, args.split, args.release)
        except ConversionError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        release_years.add(year)
        if split not in splits:
            splits.append(split)

        coco = load_coco(annotation_path)
        info_year = coco["info"].get("year")
        if info_year is not None and int(info_year) != year:
            all_errors.append(
                f"{annotation_path.name}: info.year={info_year} khác năm suy ra từ tên tệp ({year})"
            )

        licenses = index_licenses(coco)
        grouped, group_errors = group_captions(coco)
        all_errors.extend(f"{annotation_path.name}: {message}" for message in group_errors)

        images = coco["images"]
        skipped = 0
        if args.limit is not None:
            keep = {int(image["id"]) for image in images[: args.limit]}
            skipped = len(images) - len(keep)
            images = [image for image in images if int(image["id"]) in keep]
            grouped = {key: value for key, value in grouped.items() if key in keep}

        records, errors, excluded = build_records(
            images, grouped, licenses, args.dataset, split, year
        )
        all_errors.extend(f"{annotation_path.name}: {message}" for message in errors)
        all_excluded.extend(excluded)
        all_records.extend(records)

        checksum = sha256_of(annotation_path)
        source_checksums[annotation_path.name] = checksum
        inputs.append(
            {
                "path": display_path(annotation_path),
                "split": split,
                "year": year,
                "bytes": annotation_path.stat().st_size,
                "sha256": checksum,
                "image_count": len(images),
                "annotation_count": len(coco["annotations"]),
                "license_count": len(coco["licenses"]),
                "output_records": len(records),
                "skipped_by_limit": skipped,
            }
        )

    if len(release_years) > 1:
        all_errors.append(
            f"các tệp thuộc nhiều bản phát hành: {sorted(release_years)}. "
            "Ảnh và annotation phải thuộc cùng bản phát hành."
        )

    # Chống trùng Metadata.id giữa các tệp (ví dụ trộn 2014 và 2017).
    seen_ids: set[str] = set()
    duplicated: list[str] = []
    for record in all_records:
        if record.id in seen_ids:
            duplicated.append(record.id)
        seen_ids.add(record.id)
    if duplicated:
        all_errors.append(f"Metadata.id trùng: {sorted(set(duplicated))[:10]}")

    if args.verify_images or args.verify_size:
        verify_errors, missing = verify_images(all_records, dataset_root, args.verify_size)
        all_errors.extend(verify_errors)
        print(f"Kiểm tra ảnh: {missing} tệp thiếu, {len(verify_errors)} lỗi.")

    all_records.sort(key=lambda record: record.id)
    metadata_file = output_dir / "metadata.jsonl"
    write_metadata(all_records, metadata_file)

    release = str(next(iter(release_years))) if len(release_years) == 1 else None
    create_output_dirs(output_dir, splits, next(iter(release_years)) if release_years else None)
    if release is not None:
        default_url = ALLOWED_SOURCE_URLS.get(release)
        if default_url and default_url not in source_urls:
            source_urls.append(default_url)
        for split in splits:
            url = (IMAGE_ROOT_URLS if split == "train" else IMAGE_ROOT_URLS_VAL).get(release)
            if url and url not in source_urls:
                source_urls.append(url)

    write_dataset_selection(
        output_dir / "dataset-selection.json",
        dataset=args.dataset,
        release=release,
        splits=splits,
        dataset_root=dataset_root,
        metadata_file=metadata_file,
        source_urls=source_urls,
        source_checksums=source_checksums,
    )
    write_report(
        output_dir / "conversion-report.json",
        dataset=args.dataset,
        release=release,
        inputs=inputs,
        records=all_records,
        errors=all_errors,
        excluded=all_excluded,
    )

    print()
    print(f"Metadata      : {metadata_file}")
    print(f"Bản ghi       : {len(all_records)}")
    print(f"Theo split    : {dict(Counter(r.split for r in all_records))}")
    print(f"Ảnh bị loại   : {len(all_excluded)}")
    print(f"Lỗi           : {len(all_errors)}")
    if all_errors:
        print("\nDanh sách lỗi (tối đa 20 dòng đầu):")
        for message in all_errors[:20]:
            print(f"  - {message}")
    return 2 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
