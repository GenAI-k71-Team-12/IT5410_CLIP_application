import json
import sys

import pytest

import scripts.make_thumbnails as thumbs


def _write_metadata(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_thumbnail_relpath_uses_split_and_source_id():
    assert thumbs.thumbnail_relpath("val", "1000") == "thumbnails/val/1000.jpg"


def test_iter_metadata_skips_blank_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert [r["id"] for r in thumbs.iter_metadata(path)] == ["a", "b"]


def test_process_skips_existing_and_counts_missing(tmp_path):
    root = tmp_path / "data"
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "thumbnails" / "val").mkdir(parents=True)
    (root / "thumbnails" / "val" / "1.jpg").write_bytes(b"old")
    records = [
        {
            "id": "coco2017:1",
            "source_id": "1",
            "split": "val",
            "image_path": "images/val2017/1.jpg",
        },
        {
            "id": "coco2017:2",
            "source_id": "2",
            "split": "val",
            "image_path": "images/val2017/2.jpg",
        },
    ]
    calls = []
    report = thumbs.process(
        records,
        data_root=root,
        max_size=320,
        quality=80,
        overwrite=False,
        workers=1,
        render=lambda src, dest, max_size, quality: calls.append(dest),
    )
    assert report.created == 0 and report.skipped == 1 and report.missing == 1
    assert calls == []


def test_update_metadata_sets_path_only_when_thumbnail_exists(tmp_path):
    root = tmp_path / "data"
    (root / "thumbnails" / "val").mkdir(parents=True)
    (root / "thumbnails" / "val" / "1.jpg").write_bytes(b"x")
    metadata = root / "metadata.jsonl"
    _write_metadata(
        metadata,
        [
            {
                "id": "coco2017:1",
                "source_id": "1",
                "split": "val",
                "image_path": "images/val2017/1.jpg",
                "thumbnail_path": None,
            },
            {
                "id": "coco2017:2",
                "source_id": "2",
                "split": "val",
                "image_path": "images/val2017/2.jpg",
                "thumbnail_path": None,
            },
        ],
    )
    assert thumbs.update_metadata(metadata, root) == 1
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["thumbnail_path"] == "thumbnails/val/1.jpg"
    assert rows[1]["thumbnail_path"] is None
    assert list(root.glob("*.tmp")) == []


def test_render_thumbnail_shrinks_keeping_aspect(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "src.jpg"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(src, "JPEG")
    dest = tmp_path / "out" / "thumb.jpg"
    thumbs.render_thumbnail(src, dest, max_size=100, quality=80)
    with Image.open(dest) as im:
        assert im.size == (100, 50)
        assert im.mode == "RGB"


def test_render_thumbnail_does_not_upscale(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGB", (50, 25), (0, 0, 0)).save(src, "PNG")
    dest = tmp_path / "thumb.jpg"
    thumbs.render_thumbnail(src, dest, max_size=320, quality=80)
    with Image.open(dest) as im:
        assert im.size == (50, 25)


def test_render_thumbnail_converts_alpha_to_rgb(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(src, "PNG")
    dest = tmp_path / "thumb.jpg"
    thumbs.render_thumbnail(src, dest, max_size=32, quality=80)
    with Image.open(dest) as im:
        assert im.mode == "RGB"
        assert im.size == (32, 32)


def test_process_reports_undecodable_file(tmp_path):
    pytest.importorskip("PIL")
    root = tmp_path / "data"
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "images" / "val2017" / "1.jpg").write_bytes(b"not an image")
    records = [
        {"id": "coco2017:1", "source_id": "1", "split": "val", "image_path": "images/val2017/1.jpg"}
    ]
    report = thumbs.process(
        records, data_root=root, max_size=320, quality=80, overwrite=False, workers=1
    )
    assert report.failed == 1 and report.created == 0
    assert report.errors and "1.jpg" in report.errors[0]


def test_main_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    root = tmp_path / "data"
    (root / "images" / "val2017").mkdir(parents=True)
    Image.new("RGB", (64, 48), (1, 2, 3)).save(root / "images" / "val2017" / "1.jpg", "JPEG")
    _write_metadata(
        root / "metadata.jsonl",
        [
            {
                "id": "coco2017:1",
                "source_id": "1",
                "split": "val",
                "image_path": "images/val2017/1.jpg",
                "thumbnail_path": None,
            },
            {
                "id": "coco2017:2",
                "source_id": "2",
                "split": "val",
                "image_path": "images/val2017/2.jpg",
                "thumbnail_path": None,
            },
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_thumbnails",
            "--metadata",
            str(root / "metadata.jsonl"),
            "--data-root",
            str(root),
            "--workers",
            "1",
        ],
    )
    assert thumbs.main() == 2
    assert (root / "thumbnails" / "val" / "1.jpg").is_file()
    report = json.loads((root / "thumbnail-report.json").read_text(encoding="utf-8"))
    assert report["created"] == 1 and report["missing"] == 1
