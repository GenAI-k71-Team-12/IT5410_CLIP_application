"""Load one artifact and encoder per application lifespan."""

import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from clip_retrieval.backend.config import Settings
from clip_retrieval.contracts import Health, ImagePublic, Metadata
from clip_retrieval.retrieval import EngineNotReady, RetrievalEngine

ROOT = Path(__file__).resolve().parents[3]


class BackendRuntime:
    def __init__(self, settings: Settings, engine: RetrievalEngine | None = None) -> None:
        self.settings = settings
        self.engine = engine
        self.images: dict[str, ImagePublic] = {}
        self.media_paths: set[str] = set()
        self.model_loaded = False
        self.index_loaded = False
        self.ready = settings.mode == "mock"
        self.media_root = (
            ROOT / "data/examples/media" if settings.mode == "mock" else settings.dataset_root
        )
        if settings.mode == "mock":
            records = [
                Metadata.model_validate_json(line)
                for line in (ROOT / "data/examples/metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.set_catalog(records, mock=True)

    def set_catalog(self, records: list[Metadata], *, mock: bool = False) -> None:
        images, paths = {}, set()
        for item in records:
            image_path = f"{item.source_id}.svg" if mock else item.image_path
            thumbnail_path = image_path if mock else item.thumbnail_path or image_path
            for relative in (image_path, thumbnail_path):
                if self.media_root is None:
                    raise ValueError("CLIP_DATASET_ROOT is required")
                root = self.media_root.resolve()
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Missing or unsafe media for {item.id}")
                paths.add(relative)
            images[item.id] = ImagePublic(
                **item.model_dump(
                    include={"id", "dataset", "split", "width", "height", "captions"}
                ),
                image_url=f"/media/{quote(image_path, safe='/')}",
                thumbnail_url=f"/media/{quote(thumbnail_path, safe='/')}",
            )
        self.images = images
        self.media_paths = paths

    def start(self) -> None:
        if self.settings.mode == "mock":
            return
        self.ready = self.model_loaded = self.index_loaded = False
        self.images = {}
        self.media_paths = set()
        try:
            if self.settings.artifact_dir is None or self.media_root is None:
                raise ValueError(
                    "CLIP_ARTIFACT_DIR and CLIP_DATASET_ROOT are required in real mode"
                )
            from clip_retrieval.ml.artifact import load_artifact
            from clip_retrieval.ml.encoder import ClipEncoder
            from clip_retrieval.ml.engine import ClipRetrievalEngine

            artifact = load_artifact(self.settings.artifact_dir)
            self.index_loaded = True
            model = artifact.manifest["model"]
            encoder = ClipEncoder(model["name"], model["weights_id"], self.settings.device)
            self.model_loaded = True
            self.engine = ClipRetrievalEngine(artifact, encoder)
            self.set_catalog(artifact.records)
            if "bm25" in self.settings.methods:
                raise EngineNotReady("BM25 is not integrated")
            self.ready = True
        except Exception:
            self.engine = None
            self.images = {}
            self.media_paths = set()
            logging.getLogger(__name__).exception("Could not initialize retrieval backend")

    def stop(self) -> None:
        if self.settings.mode == "real":
            self.engine = None
            self.images = {}
            self.media_paths = set()
            self.ready = self.model_loaded = self.index_loaded = False

    def require_ready(self) -> None:
        if not self.ready:
            raise EngineNotReady("Retrieval engine is not ready")

    def health(self) -> Health:
        return Health(
            status="ok" if self.ready else "not_ready",
            mode=self.settings.mode,
            ready=self.ready,
            model_loaded=self.model_loaded,
            index_loaded=self.index_loaded,
        )


class CatalogMedia(StaticFiles):
    def __init__(self, runtime: BackendRuntime) -> None:
        self.runtime = runtime
        super().__init__(directory=runtime.media_root or Path("."), check_dir=False)

    async def check_config(self) -> None:
        # Failed startup is reported as 503 by get_response, rather than a StaticFiles 500.
        pass

    async def get_response(self, path: str, scope: Scope) -> Response:
        if not self.runtime.ready:
            raise HTTPException(503, "Retrieval engine is not ready")
        if Path(path).as_posix() not in self.runtime.media_paths:
            raise HTTPException(404, "Not Found")
        return await super().get_response(path, scope)
