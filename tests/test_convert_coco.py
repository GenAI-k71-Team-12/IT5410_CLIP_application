"""Kiểm thử script chuyển đổi annotation COCO sang Metadata 1.0.0."""

import json
import shutil
import sys
from pathlib import Path

import pytest

import scripts.convert_coco_annotations as convert

FIXTURES = Path(__file__).parent / "fixtures"
TINY_SIZE = (13, 7)


def _write_annotation(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coco(images: list | None = None, annotations: list | None = None) -> dict:
    return {
        "info": {"year": 2017},
        "licenses": [{"id": 1, "name": "Attribution License"}],
        "images": images or [],
        "annotations": annotations or [],
    }


def _jpeg_with_lengthless_markers(width: int, height: int, path: Path) -> Path:
    """JPEG tối thiểu có RSTn/TEM trước SOF: các marker này không có trường độ dài."""
    sof = (
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + bytes([8])
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes([3])
        + b"\x01\x11\x00" * 3
    )
    payload = b"\xff\xd8" + b"\xff\xd0" + b"\xff\xd7" + b"\xff\x01" + sof + b"\xff\xd9"
    path.write_bytes(payload)
    return path


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(convert, "use_utf8_console", lambda: None)
    monkeypatch.setattr(sys, "argv", ["convert_coco_annotations", *argv])
    return convert.main()


# --- read_image_size: JPEG, PNG và trường hợp .jpg thực chất là PNG ----------


def test_read_image_size_reads_jpeg():
    assert convert.read_image_size(FIXTURES / "tiny.jpg") == TINY_SIZE


def test_read_image_size_reads_png():
    assert convert.read_image_size(FIXTURES / "tiny.png") == TINY_SIZE


def test_read_image_size_reads_png_saved_as_jpg(tmp_path):
    # COCO 2017 có tệp .jpg thực chất là PNG (train2017/000000320612.jpg).
    misnamed = tmp_path / "000000320612.jpg"
    shutil.copy(FIXTURES / "tiny.png", misnamed)
    assert convert.read_image_size(misnamed) == TINY_SIZE


def test_read_image_size_skips_lengthless_markers(tmp_path):
    path = _jpeg_with_lengthless_markers(*TINY_SIZE, tmp_path / "restart.jpg")
    assert convert.read_image_size(path) == TINY_SIZE


def test_read_image_size_returns_none_for_unknown_format(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("khong phai anh", encoding="utf-8")
    assert convert.read_image_size(path) is None


# --- detect_split_and_year: chỉ nhận COCO 2017 ------------------------------


def test_detect_split_and_year_reads_name():
    assert convert.detect_split_and_year(Path("captions_val2017.json"), None, None) == ("val", 2017)


def test_detect_split_and_year_rejects_2014():
    with pytest.raises(convert.ConversionError, match="2014"):
        convert.detect_split_and_year(Path("captions_train2014.json"), None, None)


def test_detect_split_and_year_rejects_2014_even_with_override():
    # --release không được che mất bản 2014 trong tên tệp.
    with pytest.raises(convert.ConversionError, match="2014"):
        convert.detect_split_and_year(Path("captions_train2014.json"), None, 2017)


# --- group_captions ---------------------------------------------------------


def test_group_captions_orders_by_annotation_id():
    coco = {
        "annotations": [
            {"id": 30, "image_id": 1, "caption": "thu ba"},
            {"id": 10, "image_id": 1, "caption": "thu nhat"},
            {"id": 20, "image_id": 1, "caption": "thu hai"},
        ]
    }
    grouped, errors = convert.group_captions(coco)
    assert grouped == {1: ["thu nhat", "thu hai", "thu ba"]}
    assert errors == []


def test_group_captions_reports_duplicate_and_empty_caption():
    coco = {
        "annotations": [
            {"id": 1, "image_id": 1, "caption": "hop le"},
            {"id": 1, "image_id": 1, "caption": "trung id"},
            {"id": 2, "image_id": 1, "caption": "   "},
        ]
    }
    grouped, errors = convert.group_captions(coco)
    assert grouped == {1: ["hop le"]}
    assert len(errors) == 2


# --- build_records ----------------------------------------------------------


def test_build_records_maps_coco_fields():
    images = [
        {
            "id": 7,
            "file_name": "000000000007.jpg",
            "width": 640,
            "height": 480,
            "license": 1,
            "coco_url": "http://images.cocodataset.org/val2017/000000000007.jpg",
        }
    ]
    records, errors, excluded = convert.build_records(
        images,
        {7: ["mot con meo", "mot con cho"]},
        {1: "Attribution License"},
        "coco2017",
        "val",
        2017,
    )

    assert errors == [] and excluded == []
    record = records[0]
    assert record.id == "coco2017:7"
    assert record.source_id == "7"
    assert record.split == "val"
    assert record.image_path == "images/val2017/000000000007.jpg"
    assert (record.width, record.height) == (640, 480)
    assert record.license == "Attribution License"
    assert record.source_url == "http://images.cocodataset.org/val2017/000000000007.jpg"
    assert [caption.text for caption in record.captions] == ["mot con meo", "mot con cho"]
    assert all(caption.language == "en" for caption in record.captions)


def test_build_records_excludes_image_without_caption():
    images = [{"id": 7, "file_name": "a.jpg", "width": 1, "height": 1}]
    records, errors, excluded = convert.build_records(images, {}, {}, "coco2017", "train", 2017)

    assert records == [] and errors == []
    assert len(excluded) == 1
    assert excluded[0]["image_id"] == "7"


def test_build_records_reports_annotation_with_unknown_image():
    records, errors, _ = convert.build_records([], {5: ["mo coi"]}, {}, "coco2017", "train", 2017)

    assert records == []
    assert any("image_id=5" in message for message in errors)


# --- write_metadata: ghi qua tệp tạm rồi replace ----------------------------


class _FakeRecord:
    """Bản ghi tối thiểu đủ để kiểm thử write_metadata."""

    def __init__(self, value: int) -> None:
        self.value = value

    def model_dump(self) -> dict:
        return {"id": self.value}


def test_write_metadata_replaces_file_atomically(tmp_path):
    target = tmp_path / "metadata.jsonl"
    convert.write_metadata([_FakeRecord(1), _FakeRecord(2)], target)

    assert target.read_text(encoding="utf-8").splitlines() == ['{"id":1}', '{"id":2}']
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_metadata_keeps_previous_file_when_interrupted(tmp_path, monkeypatch):
    target = tmp_path / "metadata.jsonl"
    target.write_text('{"giu": "nguyen"}\n', encoding="utf-8")

    real_dumps = json.dumps
    calls = {"count": 0}

    def flaky_dumps(payload, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("hỏng giữa chừng")
        return real_dumps(payload, **kwargs)

    monkeypatch.setattr(convert.json, "dumps", flaky_dumps)

    with pytest.raises(RuntimeError):
        convert.write_metadata([_FakeRecord(1), _FakeRecord(2)], target)

    assert target.read_text(encoding="utf-8") == '{"giu": "nguyen"}\n'
    assert list(tmp_path.glob("*.tmp")) == []


# --- main(): ghi đầu ra và không phá dữ liệu cũ khi có lỗi -------------------


def test_main_writes_metadata_and_selection(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    images_dir = output_dir / "images" / "train2017"
    images_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "tiny.jpg", images_dir / "000000000001.jpg")

    annotation = _write_annotation(
        tmp_path / "captions_train2017.json",
        _coco(
            images=[
                {
                    "id": 1,
                    "file_name": "000000000001.jpg",
                    "width": TINY_SIZE[0],
                    "height": TINY_SIZE[1],
                    "license": 1,
                    "coco_url": "http://images.cocodataset.org/train2017/000000000001.jpg",
                }
            ],
            annotations=[{"id": 10, "image_id": 1, "caption": "mot anh nho"}],
        ),
    )

    exit_code = _run_main(
        monkeypatch,
        [
            "--annotation",
            str(annotation),
            "--dataset",
            "coco2017",
            "--output-dir",
            str(output_dir),
            "--verify-images",
            "--verify-size",
        ],
    )

    assert exit_code == 0
    metadata = json.loads((output_dir / "metadata.jsonl").read_text(encoding="utf-8").strip())
    assert metadata["id"] == "coco2017:1"
    assert metadata["image_path"] == "images/train2017/000000000001.jpg"
    assert metadata["captions"] == [{"text": "mot anh nho", "language": "en"}]

    selection = json.loads((output_dir / "dataset-selection.json").read_text(encoding="utf-8"))
    assert selection["usage_terms_url"] == convert.USAGE_TERMS_URL
    assert selection["source_urls"] == [
        convert.SOURCE_ANNOTATION_URL,
        convert.IMAGE_ROOT_URLS["train"],
    ]


def test_main_keeps_previous_metadata_when_errors(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    metadata_file = output_dir / "metadata.jsonl"
    metadata_file.write_text('{"giu": "nguyen"}\n', encoding="utf-8")

    annotation = _write_annotation(
        tmp_path / "captions_val2017.json",
        _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 1, "height": 1}],
            annotations=[
                {"id": 1, "image_id": 1, "caption": "hop le"},
                {"id": 2, "image_id": 99, "caption": "mo coi"},
            ],
        ),
    )

    exit_code = _run_main(
        monkeypatch,
        [
            "--annotation",
            str(annotation),
            "--dataset",
            "coco2017",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exit_code == 2
    assert metadata_file.read_text(encoding="utf-8") == '{"giu": "nguyen"}\n'
    assert not (output_dir / "dataset-selection.json").exists()
    report = json.loads((output_dir / "conversion-report.json").read_text(encoding="utf-8"))
    assert report["totals"]["errors"] >= 1
