"""CLIP retrieval engine implementing clip_retrieval.retrieval.RetrievalEngine."""

import math
from functools import lru_cache
from pathlib import Path
from typing import Self

import numpy as np

from clip_retrieval.contracts import SearchRequest
from clip_retrieval.ml import index as faiss_index
from clip_retrieval.ml.artifact import Artifact, check_encoder, load_artifact
from clip_retrieval.ml.encoder import ClipEncoder, Encoder
from clip_retrieval.retrieval import EngineNotReady, ImageNotFound, RankedImage, RetrievalFailure

SCORE_TOLERANCE = 1e-5


class ClipRetrievalEngine:
    def __init__(self, artifact: Artifact, encoder: Encoder, cache_size: int = 1024):
        check_encoder(artifact.manifest, encoder)
        self.index_version: str = artifact.manifest["artifact_version"]
        self._artifact = artifact
        self._rows = {image_id: row for row, image_id in enumerate(artifact.image_ids)}
        self._datasets = np.array([record.dataset for record in artifact.records])
        self._splits = np.array([record.split for record in artifact.records])
        # An engine serves one artifact and one model, so cache keys need neither fingerprint.
        self._encode_text = lru_cache(cache_size)(encoder.encode_text)
        self._ranked = lru_cache(cache_size)(self._rank)

    @classmethod
    def from_artifact(cls, path: Path, device: str = "cpu", cache_size: int = 1024) -> Self:
        artifact = load_artifact(path)
        model = artifact.manifest["model"]
        return cls(artifact, ClipEncoder(model["name"], model["weights_id"], device), cache_size)

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        if request.method != "clip":
            raise EngineNotReady(f"Method {request.method} is not loaded")
        return list(
            self._ranked(
                "text",
                request.query,
                request.dataset,
                request.split,
                request.min_similarity,
                request.top_k,
            )
        )

    def search_similar(self, image_id: str, request: SearchRequest) -> list[RankedImage]:
        if image_id not in self._rows:
            raise ImageNotFound(image_id)
        return list(
            self._ranked(
                "image",
                image_id,
                request.dataset,
                request.split,
                request.min_similarity,
                request.top_k,
            )
        )

    def _rank(
        self,
        kind: str,
        key: str,
        dataset: str | None,
        split: str | None,
        min_similarity: float | None,
        top_k: int,
    ) -> tuple[RankedImage, ...]:
        mask = np.ones(len(self._rows), dtype=bool)
        if dataset is not None:
            mask &= self._datasets == dataset
        if split is not None:
            mask &= self._splits == split
        if kind == "image":
            source = self._rows[key]
            mask[source] = False
            query = self._artifact.embeddings[source : source + 1]
        else:
            query = self._encode_text(key)
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            return ()
        candidates = None if mask.all() else rows
        # Fetch past top_k until the boundary score is no longer tied, then break ties by ID.
        k = min(top_k + 1, len(rows))
        while True:
            scores, found = faiss_index.search(self._artifact.index, query, k, candidates)
            if k == len(rows) or scores[-1] < scores[top_k - 1]:
                break
            k = min(2 * k, len(rows))
        image_ids = self._artifact.image_ids
        ranked = sorted(
            zip(scores.tolist(), found.tolist(), strict=True),
            key=lambda pair: (-pair[0], image_ids[pair[1]]),
        )
        results = []
        for score, row in ranked[:top_k]:
            if not math.isfinite(score) or abs(score) > 1 + SCORE_TOLERANCE:
                raise RetrievalFailure(f"Score {score} is outside [-1, 1]")
            score = min(1.0, max(-1.0, score))
            if min_similarity is None or score >= min_similarity:
                results.append(RankedImage(image_ids[row], score))
        return tuple(results)
