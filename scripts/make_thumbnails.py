"""Tạo thumbnail JPEG cho metadata COCO 2017 và cập nhật thumbnail_path."""

import argparse
import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_METADATA = Path("data/processed/coco2017/metadata.jsonl")
DEFAULT_DATA_ROOT = Path("data/processed/coco2017")
DEFAULT_MAX_SIZE = 320
DEFAULT_QUALITY = 80
DEFAULT_WORKERS = 4


def use_utf8_console() -> None:
    """Ép stdout/stderr về UTF-8 trên Windows; bỏ qua lỗi khi không hỗ trợ."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def thumbnail_relpath(split: str, source_id: str) -> str:
    """Đường dẫn thumbnail relative: thumbnails/<split>/<source_id>.jpg."""
    return f"thumbnails/{split}/{source_id}.jpg"


def iter_metadata(path: Path) -> Iterator[dict]:
    """Lặp từng record JSONL, bỏ qua dòng trống."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_thumbnail(src: Path, dest: Path, max_size: int, quality: int) -> None:
    """Render ảnh src thành thumbnail JPEG RGB tại dest (import Pillow bên trong)."""
    from PIL import Image, ImageOps

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_size, max_size), Image.LANCZOS)
        has_alpha = (
            im.mode == "RGBA" or im.mode == "LA" or (im.mode == "P" and "transparency" in im.info)
        )
        if has_alpha:
            im = im.convert("RGBA")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])
            im = background
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dest, "JPEG", quality=quality, optimize=True)


@dataclass
class ThumbnailReport:
    created: int = 0
    skipped: int = 0
    missing: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "created": self.created,
            "skipped": self.skipped,
            "missing": self.missing,
            "failed": self.failed,
            "errors": self.errors,
        }


def _apply_outcome(report: ThumbnailReport, kind: str, detail: object) -> None:
    if kind == "created":
        report.created += 1
    else:
        report.failed += 1
        report.errors.append(str(detail))


def process(
    records: list[dict],
    *,
    data_root: Path,
    max_size: int,
    quality: int,
    overwrite: bool,
    workers: int,
    render=render_thumbnail,
) -> ThumbnailReport:
    """Xử lý batch các record, trả về ThumbnailReport (không raise khi render lỗi)."""
    report = ThumbnailReport()
    jobs: list[tuple[dict, Path, Path]] = []
    for record in records:
        split = record["split"]
        source_id = record["source_id"]
        image_path = record["image_path"]
        dest = data_root / thumbnail_relpath(split, source_id)
        if dest.exists() and not overwrite:
            report.skipped += 1
            continue
        src = data_root / image_path
        if not src.is_file():
            report.missing += 1
            report.errors.append(f"Thiếu ảnh: {image_path}")
            continue
        jobs.append((record, src, dest))

    def _run(job: tuple[dict, Path, Path]) -> tuple[str, object]:
        _, src, dest = job
        try:
            render(src, dest, max_size, quality)
            return "created", dest
        except Exception as exc:  # noqa: BLE001
            return "failed", f"{src.name}: {exc}"

    if workers > 1 and jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for kind, detail in pool.map(_run, jobs):
                _apply_outcome(report, kind, detail)
    else:
        for job in jobs:
            kind, detail = _run(job)
            _apply_outcome(report, kind, detail)
    return report


def update_metadata(metadata_path: Path, data_root: Path) -> int:
    """Cập nhật thumbnail_path cho record có thumbnail; ghi .tmp rồi replace. Trả số record đổi."""
    changed = 0
    tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")
    with metadata_path.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                dst.write(line)
                continue
            record = json.loads(stripped)
            relpath = thumbnail_relpath(record["split"], record["source_id"])
            if (data_root / relpath).is_file() and record.get("thumbnail_path") != relpath:
                record["thumbnail_path"] = relpath
                changed += 1
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                dst.write(line)
    tmp_path.replace(metadata_path)
    return changed


def _apply_limit(records: list[dict], limit: int) -> list[dict]:
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for record in records:
        split = record["split"]
        if counts.get(split, 0) >= limit:
            continue
        counts[split] = counts.get(split, 0) + 1
        kept.append(record)
    return kept


def _write_report(path: Path, report: ThumbnailReport, max_size: int, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["max_size"] = max_size
    payload["quality"] = quality
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entry point. Thoát 0 sạch, 1 tham số sai, 2 có ảnh thiếu/hỏng; luôn ghi báo cáo."""
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Tạo thumbnail JPEG cho metadata COCO 2017")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-update-metadata", action="store_true")
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1

    # Lỗi tham số: file metadata không tồn tại.
    if not args.metadata.is_file():
        print(f"Không tìm thấy metadata: {args.metadata}")
        _write_report(
            args.data_root / "thumbnail-report.json",
            ThumbnailReport(),
            args.max_size,
            args.quality,
        )
        return 1

    splits = args.splits or None  # list rỗng == tất cả split
    report = ThumbnailReport()
    try:
        records = [
            record
            for record in iter_metadata(args.metadata)
            if splits is None or record["split"] in splits
        ]
        if args.limit > 0:
            records = _apply_limit(records, args.limit)
        report = process(
            records,
            data_root=args.data_root,
            max_size=args.max_size,
            quality=args.quality,
            overwrite=args.overwrite,
            workers=args.workers,
        )
        if not args.no_update_metadata:
            updated = update_metadata(args.metadata, args.data_root)
            print(f"Cập nhật metadata: {updated} record")
    except Exception as exc:  # noqa: BLE001
        report.failed += 1
        report.errors.append(str(exc))
        _write_report(args.data_root / "thumbnail-report.json", report, args.max_size, args.quality)
        return 1

    _write_report(args.data_root / "thumbnail-report.json", report, args.max_size, args.quality)
    print(
        f"Tạo: {report.created}, bỏ qua: {report.skipped}, "
        f"thiếu: {report.missing}, lỗi: {report.failed}"
    )
    return 0 if report.missing == 0 and report.failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
