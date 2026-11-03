"""Process configuration for the mock and artifact-backed API."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class Settings:
    mode: str = "mock"
    dataset_root: Path | None = None
    artifact_dir: Path | None = None
    device: str = "cpu"
    methods: tuple[str, ...] = ("clip",)

    def __post_init__(self) -> None:
        if self.mode not in ("mock", "real"):
            raise ValueError("CLIP_MODE must be mock or real")
        if not self.methods or any(method not in ("clip", "bm25") for method in self.methods):
            raise ValueError("CLIP_METHODS must contain clip and/or bm25")

    @classmethod
    def from_env(cls, mode: str | None = None) -> Self:
        return cls(
            mode=mode or os.getenv("CLIP_MODE", "mock"),
            dataset_root=Path(value) if (value := os.getenv("CLIP_DATASET_ROOT")) else None,
            artifact_dir=Path(value) if (value := os.getenv("CLIP_ARTIFACT_DIR")) else None,
            device=os.getenv("CLIP_DEVICE", "cpu"),
            methods=tuple(item.strip() for item in os.getenv("CLIP_METHODS", "clip").split(",")),
        )
