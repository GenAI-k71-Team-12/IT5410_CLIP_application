"""Boundary for Quân's real implementation; never return FAISS row IDs to clients."""

from dataclasses import dataclass
from typing import Protocol

from clip_retrieval.contracts import SearchRequest


@dataclass(frozen=True)
class RankedImage:
    image_id: str
    score: float


class RetrievalEngine(Protocol):
    index_version: str

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        """Return filtered, sorted top_k hits BEFORE offset/limit slicing."""
        ...

    def search_similar(self, image_id: str, request: SearchRequest) -> list[RankedImage]:
        """Same rules; exclude source image before top_k truncation."""
        ...
