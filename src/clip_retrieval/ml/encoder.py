"""CLIP encoder. torch/open_clip load inside ClipEncoder so the rest of the ML
package and its tests run without them."""

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from clip_retrieval.ml.checksum import sha256_file, sha256_json
from clip_retrieval.retrieval import QueryTooLong, RetrievalFailure


class Encoder(Protocol):
    model_info: dict  # the manifest "model" section
    config_files: dict[str, dict]  # canonical configs shipped inside the artifact

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise RetrievalFailure("Embedding must be finite with a non-zero norm")
    return np.ascontiguousarray(vectors / norms)


class ClipEncoder:
    def __init__(self, model: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        import open_clip
        from open_clip.pretrained import download_pretrained, get_pretrained_cfg
        from open_clip.tokenizer import default_bpe

        self.device = device
        weights_cfg = get_pretrained_cfg(model, pretrained)
        # open_clip only warns when weights were trained with QuickGELU (e.g. "openai").
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=weights_cfg.get("quick_gelu", False),
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        weights = download_pretrained(weights_cfg)
        tokenizer_config = {
            "class": type(self.tokenizer).__qualname__,
            "bpe_vocab_sha256": sha256_file(Path(default_bpe())),
            **open_clip.get_model_tokenize_cfg(self.model),
        }
        preprocess_config = {
            **open_clip.get_model_preprocess_cfg(self.model),
            "exif_transpose": False,
        }
        self.config_files = {
            "tokenizer_config.json": tokenizer_config,
            "preprocess_config.json": preprocess_config,
        }
        self.model_info = {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "name": model,
            "weights_id": pretrained,
            "weights_sha256": sha256_file(Path(weights)),
            "tokenizer_id": f"open_clip:{model}",
            "tokenizer_config_sha256": sha256_json(tokenizer_config),
            "context_length": self.tokenizer.context_length,
            "preprocess_id": f"open_clip:{model}/{pretrained}",
            "preprocess_config_sha256": sha256_json(preprocess_config),
        }

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
        return l2_normalize(features.float().cpu().numpy())

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        tokens = len(self.tokenizer.encode(text)) + 2  # + start/end-of-text tokens
        limit = self.tokenizer.context_length
        if tokens > limit:
            raise QueryTooLong(f"Query has {tokens} tokens; the limit is {limit}")
        with torch.no_grad():
            features = self.model.encode_text(self.tokenizer([text]).to(self.device))
        return l2_normalize(features.float().cpu().numpy())
