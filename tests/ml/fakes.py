"""Deterministic CLIP stand-in for ML tests: an image embeds as its mean colour."""

from pathlib import Path

import numpy as np
from PIL import Image

from clip_retrieval.contracts import Metadata
from clip_retrieval.ml.build_index import build_artifact
from clip_retrieval.ml.checksum import sha256_json
from clip_retrieval.ml.encoder import l2_normalize
from clip_retrieval.retrieval import QueryTooLong

COLORS = {
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "purple": (128, 0, 255),
}
# Three identical red squares share one file hash: cache de-duplication and score ties.
CORPUS = [
    ("toy:a1", "red", "val"),
    ("toy:a2", "red", "val"),
    ("toy:a3", "red", "val"),
    ("toy:b1", "orange", "val"),
    ("toy:c1", "blue", "test"),
    ("toy:d1", "green", "val"),
    ("toy:e1", "purple", "test"),
]
TOKENIZER_CONFIG = {"class": "FakeTokenizer", "context_length": 77}
PREPROCESS_CONFIG = {"mode": "RGB", "exif_transpose": False}


def color_vector(rgb) -> np.ndarray:
    return l2_normalize(np.array([[*rgb, 64.0]]))


class FakeEncoder:
    def __init__(self, name: str = "fake-rgb"):
        self.config_files = {
            "tokenizer_config.json": TOKENIZER_CONFIG,
            "preprocess_config.json": PREPROCESS_CONFIG,
        }
        self.model_info = {
            "library": "fake",
            "library_version": "0",
            "name": name,
            "weights_id": "fake",
            "weights_sha256": "0" * 64,
            "tokenizer_id": "fake",
            "tokenizer_config_sha256": sha256_json(TOKENIZER_CONFIG),
            "context_length": 77,
            "preprocess_id": "fake",
            "preprocess_config_sha256": sha256_json(PREPROCESS_CONFIG),
        }
        self.images_encoded = 0
        self.texts_encoded = 0

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.images_encoded += len(images)
        return np.concatenate([color_vector(np.asarray(i).mean(axis=(0, 1))) for i in images])

    def encode_text(self, text: str) -> np.ndarray:
        if len(text) > 40:
            raise QueryTooLong(text)
        self.texts_encoded += 1
        return color_vector(COLORS[text])


def write_dataset(root: Path, corpus=CORPUS) -> list[Metadata]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for image_id, color, split in corpus:
        dataset, source_id = image_id.split(":")
        relative = f"images/{source_id}.png"
        Image.new("RGB", (8, 6), COLORS[color]).save(root / relative)
        records.append(
            Metadata(
                id=image_id,
                dataset=dataset,
                source_id=source_id,
                split=split,
                image_path=relative,
                width=8,
                height=6,
                captions=[{"text": f"a {color} square", "language": "en"}],
            )
        )
    return records


def build_toy(tmp_path: Path, version="toy-v1", encoder=None, records=None, **options) -> Path:
    if records is None:
        records = write_dataset(tmp_path / "data")
    options.setdefault("splits", ["test", "val"])
    return build_artifact(
        records,
        tmp_path / "data",
        encoder or FakeEncoder(),
        tmp_path / "artifacts",
        version,
        dataset="toy",
        batch_size=2,
        cache_dir=tmp_path / "cache",
        **options,
    )
