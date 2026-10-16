import json

import numpy as np
import pytest
from fakes import COLORS, CORPUS, FakeEncoder, build_toy, color_vector, write_dataset
from PIL import Image

from clip_retrieval.ml.artifact import load_artifact
from clip_retrieval.ml.artifact import main as verify_main
from clip_retrieval.ml.build_index import read_metadata
from clip_retrieval.ml.checksum import sha256_file, sha256_json
from clip_retrieval.retrieval import ArtifactInvalid


def test_rows_are_sorted_ids_matching_their_images(tmp_path):
    artifact = load_artifact(build_toy(tmp_path))
    colors = {image_id: color for image_id, color, _ in CORPUS}
    assert artifact.image_ids == sorted(colors) == [r.id for r in artifact.records]
    for row, image_id in enumerate(artifact.image_ids):
        expected = color_vector(COLORS[colors[image_id]])[0]
        np.testing.assert_allclose(artifact.embeddings[row], expected, atol=1e-6)
    manifest = artifact.manifest
    assert manifest["artifact_version"] == "toy-v1" and manifest["dataset"] == "toy"
    assert manifest["splits"] == ["test", "val"] and manifest["subset_ids_sha256"] is None
    assert manifest["model"] == FakeEncoder().model_info
    assert manifest["embedding"] == {
        "dimension": 4,
        "dtype": "float32",
        "normalization": "l2",
        "count": 7,
    }
    assert set(manifest["files"]) == {
        "metadata",
        "embeddings",
        "image_ids",
        "index",
        "tokenizer_config",
        "preprocess_config",
    }
    assert manifest["build"]["batch_size"] == 2
    assert len(manifest["build"]["dependency_lock_sha256"]) == 64


def test_split_and_subset_selection(tmp_path):
    records = write_dataset(tmp_path / "data")
    only_test = load_artifact(build_toy(tmp_path, "t-v1", records=records, splits=["test"]))
    assert only_test.image_ids == ["toy:c1", "toy:e1"]
    subset = load_artifact(
        build_toy(tmp_path, "s-v1", records=records, subset_ids=["toy:b1", "toy:a1"])
    )
    assert subset.image_ids == ["toy:a1", "toy:b1"]
    assert subset.manifest["subset_ids_sha256"] == sha256_json(["toy:a1", "toy:b1"])
    with pytest.raises(ValueError, match="subset"):
        build_toy(tmp_path, "bad-v1", records=records, subset_ids=["toy:zz"])


def test_cache_skips_unchanged_images_and_is_keyed_by_model(tmp_path):
    records = write_dataset(tmp_path / "data")
    encoder = FakeEncoder()
    first = build_toy(tmp_path, "toy-v1", encoder=encoder, records=records)
    assert encoder.images_encoded == 5  # three identical red squares share one hash
    second = build_toy(tmp_path, "toy-v2", encoder=encoder, records=records)
    assert encoder.images_encoded == 5
    np.testing.assert_array_equal(load_artifact(first).embeddings, load_artifact(second).embeddings)
    other = FakeEncoder(name="other-model")
    build_toy(tmp_path, "toy-v3", encoder=other, records=records)
    assert other.images_encoded == 5


def test_existing_artifact_version_is_never_overwritten(tmp_path):
    records = write_dataset(tmp_path / "data")
    build_toy(tmp_path, records=records)
    with pytest.raises(FileExistsError):
        build_toy(tmp_path, records=records)


def test_stale_staging_dir_from_interrupted_build_is_replaced(tmp_path):
    stale = tmp_path / "artifacts" / ".tmp-toy-v1"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("partial")
    path = build_toy(tmp_path)
    assert not stale.exists() and not (path / "junk").exists()


def test_unicode_paths_work(tmp_path):
    artifact = load_artifact(build_toy(tmp_path / "dữ liệu"))
    assert len(artifact.image_ids) == 7


def test_grayscale_rgba_and_palette_images_are_encoded(tmp_path):
    records = write_dataset(tmp_path / "data")
    Image.new("L", (8, 6), 90).save(tmp_path / "data/images/a1.png")
    Image.new("RGBA", (8, 6), (0, 0, 255, 40)).save(tmp_path / "data/images/b1.png")
    Image.new("RGB", (8, 6), (0, 255, 0)).convert("P").save(tmp_path / "data/images/d1.png")
    artifact = load_artifact(build_toy(tmp_path, records=records))
    assert artifact.embeddings.shape == (7, 4)


@pytest.mark.parametrize("change", [{"width": 9}, {"image_path": "images/missing.png"}])
def test_build_stops_on_bad_image(tmp_path, change):
    records = write_dataset(tmp_path / "data")
    records[0] = records[0].model_copy(update=change)
    with pytest.raises(ValueError):
        build_toy(tmp_path, records=records)
    assert not (tmp_path / "artifacts").exists()


def test_read_metadata_rejects_duplicate_ids(tmp_path):
    record = write_dataset(tmp_path / "data")[0]
    path = tmp_path / "metadata.jsonl"
    path.write_text(record.model_dump_json() + "\n" + record.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_metadata(path)


def rewrite_manifest(path, change):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    change(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def swap_first_ids(path):
    ids = json.loads((path / "image_ids.json").read_text(encoding="utf-8"))
    ids[0], ids[1] = ids[1], ids[0]
    (path / "image_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    digest = sha256_file(path / "image_ids.json")
    rewrite_manifest(path, lambda m: m["files"]["image_ids"].update(sha256=digest))


TAMPERS = {
    "edited embeddings": lambda p: (p / "embeddings.npy").write_bytes(
        (p / "embeddings.npy").read_bytes() + b"0"
    ),
    "missing index": lambda p: (p / "index.faiss").unlink(),
    "placeholder": lambda p: rewrite_manifest(
        p, lambda m: m["model"].update(library="TO_BE_SELECTED")
    ),
    "unsafe path": lambda p: rewrite_manifest(
        p, lambda m: m["files"]["index"].update(path="../index.faiss")
    ),
    "reordered ids": swap_first_ids,
}


@pytest.mark.parametrize("tamper", TAMPERS.values(), ids=TAMPERS.keys())
def test_loader_rejects_tampered_artifact(tmp_path, tamper):
    path = build_toy(tmp_path)
    tamper(path)
    with pytest.raises(ArtifactInvalid):
        load_artifact(path)


def test_verify_cli(tmp_path, monkeypatch, capsys):
    path = build_toy(tmp_path)
    monkeypatch.setattr("sys.argv", ["artifact", str(path)])
    verify_main()
    assert "Valid artifact toy-v1: N=7, D=4" in capsys.readouterr().out
    (path / "index.faiss").unlink()
    with pytest.raises(SystemExit, match="Invalid artifact"):
        verify_main()
