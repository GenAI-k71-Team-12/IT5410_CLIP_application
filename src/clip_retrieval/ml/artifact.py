"""Artifact writer and loader per docs/05_ML_INTEGRATION_CONTRACT.md sections 4 and 7.

Verify a downloaded artifact: python -m clip_retrieval.ml.artifact artifacts/<version> --check-model
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from clip_retrieval.contracts import Metadata, utc_timestamp
from clip_retrieval.ml import index as faiss_index
from clip_retrieval.ml.checksum import canonical_json, sha256_file
from clip_retrieval.ml.encoder import ClipEncoder, Encoder
from clip_retrieval.retrieval import ArtifactInvalid

MANIFEST_VERSION = "1.0.0"
DATA_FILES = {
    "metadata": "metadata.jsonl",
    "embeddings": "embeddings.npy",
    "image_ids": "image_ids.json",
    "index": "index.faiss",
}


@dataclass(frozen=True)
class Artifact:
    path: Path
    manifest: dict
    records: list[Metadata]
    image_ids: list[str]
    embeddings: np.ndarray
    index: faiss.Index


def write_artifact(
    out_dir: Path,
    *,
    artifact_version: str,
    dataset: str,
    splits: list[str],
    subset_ids_sha256: str | None,
    records: list[Metadata],
    embeddings: np.ndarray,
    encoder: Encoder,
    build: dict,
) -> None:
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "image_ids.json").write_text(json.dumps([r.id for r in records]), encoding="utf-8")
    faiss_index.save(faiss_index.build(embeddings), out_dir / "index.faiss")
    files = dict(DATA_FILES)
    for name, config in encoder.config_files.items():
        (out_dir / name).write_bytes(canonical_json(config))
        files[Path(name).stem] = name
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "artifact_version": artifact_version,
        "metadata_schema_version": "1.0.0",
        "dataset": dataset,
        "splits": splits,
        "subset_ids_sha256": subset_ids_sha256,
        "created_at": utc_timestamp(),
        "model": encoder.model_info,
        "embedding": {
            "dimension": int(embeddings.shape[1]),
            "dtype": "float32",
            "normalization": "l2",
            "count": len(records),
        },
        "index": {
            "library": "faiss",
            "library_version": faiss.__version__,
            "type": "IndexFlatIP",
            "metric": "inner_product",
            "id_mapping": "row",
        },
        "build": build,
        "files": {
            key: {"path": name, "sha256": sha256_file(out_dir / name)}
            for key, name in files.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_artifact(path: Path) -> Artifact:
    """Run checks 1-8 of the ML contract (section 7); any failure raises ArtifactInvalid."""
    try:
        return _load(Path(path))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactInvalid(f"{type(exc).__name__}: {exc}") from exc


def check_encoder(manifest: dict, encoder: Encoder) -> None:
    """Check 9: the loaded checkpoint, tokenizer and preprocess match the manifest."""
    expected, actual = manifest["model"], encoder.model_info
    differs = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    _require(not differs, f"Encoder does not match the artifact model: {differs}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactInvalid(message)


def _placeholders(value: object, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _placeholders(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _placeholders(child, key)
    elif value == "TO_BE_SELECTED" or (value is None and key != "subset_ids_sha256"):
        yield key


def _load(path: Path) -> Artifact:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "Unsupported manifest_version")
    placeholders = sorted(set(_placeholders(manifest)))
    _require(not placeholders, f"Manifest has placeholder values: {placeholders}")
    files = {}
    for key, entry in manifest["files"].items():
        relative = entry["path"]
        _require(
            ":" not in relative
            and "\\" not in relative
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            f"Unsafe artifact path: {relative}",
        )
        files[key] = path / relative
        _require(files[key].is_file(), f"Missing artifact file: {relative}")
        _require(sha256_file(files[key]) == entry["sha256"], f"Checksum mismatch: {relative}")
    for name in ("tokenizer_config", "preprocess_config"):
        _require(
            manifest["files"][name]["sha256"] == manifest["model"][f"{name}_sha256"],
            f"{name} file does not match the manifest model",
        )

    count, dimension = manifest["embedding"]["count"], manifest["embedding"]["dimension"]
    lines = files["metadata"].read_text(encoding="utf-8").splitlines()
    records = [Metadata.model_validate_json(line) for line in lines if line.strip()]
    ids = [record.id for record in records]
    _require(0 < len(records) == count, "Snapshot size does not match embedding.count")
    _require(ids == sorted(set(ids)), "Snapshot IDs must be unique and sorted")
    _require(
        all(r.dataset == manifest["dataset"] and r.split in manifest["splits"] for r in records),
        "Snapshot dataset/split does not match the manifest",
    )
    image_ids = json.loads(files["image_ids"].read_text(encoding="utf-8"))
    _require(image_ids == ids, "image_ids.json does not match the snapshot order")

    embeddings = np.load(files["embeddings"], allow_pickle=False)
    _require(
        embeddings.dtype == np.float32
        and embeddings.shape == (count, dimension)
        and embeddings.flags.c_contiguous,
        "Embeddings must be a C-contiguous float32 (N, D) array",
    )
    _require(bool(np.isfinite(embeddings).all()), "Embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    _require(bool(np.all(np.abs(norms - 1) <= 1e-4)), "Embeddings are not L2-normalized")

    index = faiss_index.load(files["index"])
    _require(
        isinstance(index, faiss.IndexFlatIP) and index.d == dimension and index.ntotal == count,
        "Index must be IndexFlatIP with d=D and ntotal=N",
    )
    _require(
        bool(np.allclose(index.reconstruct_n(0, count), embeddings, rtol=0, atol=1e-5)),
        "Index vectors differ from embeddings",
    )
    return Artifact(path, manifest, records, image_ids, embeddings, index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a CLIP/FAISS artifact directory")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--check-model", action="store_true", help="also load CLIP and compare")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        artifact = load_artifact(args.artifact_dir)
        if args.check_model:
            model = artifact.manifest["model"]
            encoder = ClipEncoder(model["name"], model["weights_id"], args.device)
            check_encoder(artifact.manifest, encoder)
    except ArtifactInvalid as exc:
        raise SystemExit(f"Invalid artifact: {exc}") from exc
    version, (count, dimension) = artifact.manifest["artifact_version"], artifact.embeddings.shape
    print(f"Valid artifact {version}: N={count}, D={dimension}")


if __name__ == "__main__":
    main()
