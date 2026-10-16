"""Offline indexing: metadata + images -> artifacts/<artifact_version>/.

python -m clip_retrieval.ml.build_index --metadata data/processed/coco2017/metadata.jsonl \
    --dataset-root data/processed/coco2017 --dataset coco2017 --splits val \
    --artifact-version coco2017-val-vitb32-v1 --device cuda
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError

from clip_retrieval.contracts import Metadata
from clip_retrieval.ml.artifact import load_artifact, write_artifact
from clip_retrieval.ml.checksum import sha256_file, sha256_json
from clip_retrieval.ml.encoder import ClipEncoder, Encoder

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ["train", "val", "test", "unspecified"]


def read_metadata(path: Path) -> list[Metadata]:
    records, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = Metadata.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"Line {line_no}: {exc}") from exc
        if record.id in seen:
            raise ValueError(f"Line {line_no}: duplicate image id {record.id}")
        seen.add(record.id)
        records.append(record)
    return records


def select_records(
    records: list[Metadata], dataset: str, splits: list[str], subset_ids: list[str] | None
) -> list[Metadata]:
    selected = [r for r in records if r.dataset == dataset and r.split in splits]
    if subset_ids is not None:
        wanted = set(subset_ids)
        selected = [r for r in selected if r.id in wanted]
        if len(selected) != len(wanted):
            raise ValueError("Some subset IDs are not in the selected dataset/splits")
    if not selected:
        raise ValueError("No metadata records match the dataset/splits/subset")
    return sorted(selected, key=lambda record: record.id)


def image_file(dataset_root: Path, record: Metadata) -> Path:
    """Resolve the image under dataset_root and check its size against the metadata."""
    root = dataset_root.resolve()
    path = (root / record.image_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{record.id}: missing or unsafe image path {record.image_path}")
    with Image.open(path) as image:
        if image.size != (record.width, record.height):
            raise ValueError(
                f"{record.id}: image is {image.size}, metadata says {(record.width, record.height)}"
            )
    return path


def decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["sha256"].tolist(), data["vectors"], strict=True))


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(partial, sha256=np.array(list(cache)), vectors=np.stack(list(cache.values())))
    partial.replace(path)


def build_artifact(
    records: list[Metadata],
    dataset_root: Path,
    encoder: Encoder,
    artifacts_root: Path,
    artifact_version: str,
    *,
    dataset: str,
    splits: list[str],
    subset_ids: list[str] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> Path:
    started = time.perf_counter()
    target = artifacts_root / artifact_version
    if target.exists():
        raise FileExistsError(f"{target} exists; build a new artifact version instead")
    selected = select_records(records, dataset, splits, subset_ids)
    paths = [image_file(dataset_root, record) for record in selected]
    hashes = [sha256_file(path) for path in paths]

    # Same image bytes + same model fingerprint => same embedding.
    cache_path = cache_dir / f"{sha256_json(encoder.model_info)}.npz" if cache_dir else None
    cache = load_cache(cache_path) if cache_path else {}
    unique = dict(zip(hashes, paths, strict=True))
    missing = [(digest, path) for digest, path in unique.items() if digest not in cache]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = encoder.encode_images([decode_rgb(path) for _, path in batch])
        cache.update(zip([digest for digest, _ in batch], vectors, strict=True))
    if cache_path and missing:
        save_cache(cache_path, cache)
    embeddings = np.stack([cache[digest] for digest in hashes]).astype(np.float32)

    staging = artifacts_root / f".tmp-{artifact_version}"
    shutil.rmtree(staging, ignore_errors=True)
    write_artifact(
        staging,
        artifact_version=artifact_version,
        dataset=dataset,
        splits=sorted(set(splits)),
        subset_ids_sha256=None if subset_ids is None else sha256_json(sorted(set(subset_ids))),
        records=selected,
        embeddings=embeddings,
        encoder=encoder,
        build={
            "device": device,
            "batch_size": batch_size,
            "seed": 0,
            "dependency_lock_sha256": sha256_file(ROOT / "requirements-ml.txt"),
        },
    )
    load_artifact(staging)
    staging.rename(target)
    batches = -(-len(missing) // batch_size)
    print(
        f"Built {artifact_version}: N={len(selected)}, D={embeddings.shape[1]}, "
        f"encoded={len(missing)} in {batches} batches, cached={len(unique) - len(missing)}, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CLIP/FAISS artifact from metadata")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--subset", type=Path, help="text file with one image ID per line")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/.cache"))
    args = parser.parse_args()
    subset_ids = args.subset.read_text(encoding="utf-8").split() if args.subset else None
    build_artifact(
        read_metadata(args.metadata),
        args.dataset_root,
        ClipEncoder(args.model, args.pretrained, args.device),
        args.artifacts_root,
        args.artifact_version,
        dataset=args.dataset,
        splits=args.splits,
        subset_ids=subset_ids,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
