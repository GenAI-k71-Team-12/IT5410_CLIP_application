"""API v2 with explicit mock mode and artifact-backed retrieval."""

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from clip_retrieval.backend.config import Settings
from clip_retrieval.backend.runtime import BackendRuntime, CatalogMedia
from clip_retrieval.contracts import (
    ErrorResponse,
    Health,
    ImageId,
    ImagePublic,
    NoQuery,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SuccessResponse,
)
from clip_retrieval.retrieval import (
    EngineNotReady,
    ImageNotFound,
    QueryTooLong,
    RankedImage,
    RetrievalEngine,
    RetrievalFailure,
)


def create_app(
    mode: str | None = None,
    engine: RetrievalEngine | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env(mode)
    runtime = BackendRuntime(settings, engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(runtime.start)
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="CLIP Image Retrieval", version="2.0.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.mount("/media", CatalogMedia(runtime), name="media")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        try:
            response = await call_next(request)
        except Exception as exc:
            response = await internal_error(request, exc)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-API-Version"] = "2.0.0"
        if response.headers.get("Content-Type", "").startswith("application/json"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def error(request: Request, status: int, messages: list[str], headers=None):
        body = ErrorResponse(statusCode=status, message=messages, path=request.url.path)
        return JSONResponse(
            status_code=status,
            content=body.model_dump(),
            headers={
                **(headers or {}),
                "X-Request-ID": request.state.request_id,
                "X-API-Version": "2.0.0",
                "Cache-Control": "no-store",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = []
        for item in exc.errors():
            field = ".".join(map(str, item["loc"]))
            if item["type"] == "extra_forbidden":
                property_name = field.removeprefix("body.").removeprefix("query.")
                messages.append(f"property {property_name} should not exist")
            else:
                messages.append(f"{field}: {item['msg']}")
        return error(request, 422, messages)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error(request, exc.status_code, [str(exc.detail)], exc.headers)

    @app.exception_handler(ImageNotFound)
    async def image_not_found(request: Request, exc: ImageNotFound):
        return error(request, 404, ["Image not found"])

    @app.exception_handler(EngineNotReady)
    async def engine_not_ready(request: Request, exc: EngineNotReady):
        return error(request, 503, ["Retrieval engine is not ready"])

    @app.exception_handler(QueryTooLong)
    async def query_too_long(request: Request, exc: QueryTooLong):
        return error(request, 422, [str(exc)])

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        logging.getLogger(__name__).error(
            "Unhandled error request_id=%s", request.state.request_id, exc_info=exc
        )
        return error(request, 500, ["Internal server error"])

    def common_headers(
        accept: Annotated[
            str | None,
            Header(
                description="Optional; use application/json or */*. JSON endpoints return JSON."
            ),
        ] = None,
    ):
        pass

    def json_body(request: Request):
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise HTTPException(415, "Content-Type must be application/json")

    header_docs = {
        "X-Request-ID": {
            "description": "Server-generated UUID for this request; also present on errors.",
            "schema": {"type": "string", "format": "uuid"},
        },
        "X-API-Version": {"schema": {"type": "string", "const": "2.0.0"}},
        "Cache-Control": {"schema": {"type": "string", "const": "no-store"}},
    }

    def responses(*statuses):
        return {
            200: {"description": "Success envelope", "headers": header_docs},
            **{status: {"model": ErrorResponse, "headers": header_docs} for status in statuses},
        }

    @app.get(
        "/health",
        response_model=SuccessResponse[Health],
        responses=responses(405, 422, 500),
        dependencies=[Depends(common_headers)],
        summary="Get API readiness",
    )
    def health(params: Annotated[NoQuery, Query()]):
        return SuccessResponse[Health](data=runtime.health())

    def require_image(image_id):
        runtime.require_ready()
        if image_id not in runtime.images:
            raise HTTPException(status_code=404, detail="Image not found")
        return runtime.images[image_id]

    def mock_rank(params: SearchRequest, exclude=None) -> list[RankedImage]:
        # Fixed scores are test fixtures; query does not change ordering in mock mode.
        pool = [
            RankedImage(item.id, 0.9 - i * 0.15)
            for i, item in enumerate(runtime.images.values())
            if item.id != exclude
            and (params.dataset is None or item.dataset == params.dataset)
            and (params.split is None or item.split == params.split)
        ]
        if params.method == "bm25":
            pool = [RankedImage(hit.image_id, round(hit.score * 10, 6)) for hit in pool]
        if params.min_similarity is not None:
            pool = [hit for hit in pool if hit.score >= params.min_similarity]
        pool.sort(key=lambda hit: (-hit.score, hit.image_id))
        return pool[: params.top_k]

    def page(
        params: SearchRequest,
        request: Request,
        ranked: list[RankedImage],
        query: str | None,
        mode: str = "mock",
        index_version: str = "mock-v1",
    ):
        ids = [hit.image_id for hit in ranked]
        if (
            len(ranked) > params.top_k
            or len(set(ids)) != len(ids)
            or any(image_id not in runtime.images for image_id in ids)
            or any(not math.isfinite(hit.score) for hit in ranked)
            or (params.method == "clip" and any(abs(hit.score) > 1 for hit in ranked))
            or ranked != sorted(ranked, key=lambda hit: (-hit.score, hit.image_id))
        ):
            raise RetrievalFailure("Invalid engine result pool")
        hits = [
            SearchHit(
                rank=i + params.offset + 1,
                image=runtime.images[hit.image_id],
                score=round(hit.score, 6),
                similarity=round(hit.score, 6) if params.method == "clip" else None,
            )
            for i, hit in enumerate(ranked[params.offset : params.offset + params.limit])
        ]
        end = params.offset + len(hits)
        return SearchResponse(
            request_id=request.state.request_id,
            mode=mode,
            method=params.method,
            score_type="cosine_similarity" if params.method == "clip" else "bm25_score",
            query=query,
            top_k=params.top_k,
            offset=params.offset,
            limit=params.limit,
            total=len(ranked),
            has_more=end < len(ranked),
            next_offset=end if end < len(ranked) else None,
            index_version=index_version,
            results=hits,
        )

    @app.post(
        "/search",
        response_model=SuccessResponse[SearchResponse],
        responses=responses(405, 415, 422, 500, 503),
        dependencies=[Depends(common_headers), Depends(json_body)],
        summary="Search images by text (CLIP or BM25)",
    )
    def search(params: SearchRequest, request: Request, query: Annotated[NoQuery, Query()]):
        runtime.require_ready()
        if settings.mode == "real":
            if params.method not in settings.methods or runtime.engine is None:
                raise EngineNotReady(f"Method {params.method} is not loaded")
            ranked = runtime.engine.search_text(params)
            return SuccessResponse[SearchResponse](
                data=page(
                    params, request, ranked, params.query, "real", runtime.engine.index_version
                )
            )
        return SuccessResponse[SearchResponse](
            data=page(params, request, mock_rank(params), params.query)
        )

    @app.get(
        "/images/{image_id}",
        response_model=SuccessResponse[ImagePublic],
        responses=responses(404, 405, 422, 500, 503),
        dependencies=[Depends(common_headers)],
        summary="Get public image metadata",
    )
    def image(image_id: ImageId, params: Annotated[NoQuery, Query()]):
        return SuccessResponse[ImagePublic](data=require_image(image_id))

    @app.get(
        "/images/{image_id}/similar",
        response_model=SuccessResponse[SearchResponse],
        responses=responses(404, 405, 422, 500, 503),
        dependencies=[Depends(common_headers)],
        summary="Find similar images, excluding the source",
    )
    def similar(
        image_id: ImageId,
        request: Request,
        filters: Annotated[SimilarRequest, Query()],
    ):
        require_image(image_id)
        params = SearchRequest(query="similar", **filters.model_dump())
        if settings.mode == "real" and "clip" not in settings.methods:
            raise EngineNotReady("Method clip is not loaded")
        if runtime.engine is None:
            data = page(params, request, mock_rank(params, exclude=image_id), None)
        else:
            ranked = runtime.engine.search_similar(image_id, params)
            if any(hit.image_id == image_id for hit in ranked):
                raise RetrievalFailure("Similar results contain the source image")
            data = page(params, request, ranked, None, "real", runtime.engine.index_version)
        return SuccessResponse[SearchResponse](data=data)

    # StaticFiles mounts are not exported by FastAPI; describe the media endpoint explicitly.
    default_openapi = app.openapi

    def openapi():
        schema = default_openapi()
        media_headers = {key: value for key, value in header_docs.items() if key != "Cache-Control"}
        schema["paths"]["/media/{path}"] = {
            "get": {
                "operationId": "get_media",
                "summary": "Download an image or thumbnail (raw bytes)",
                "parameters": [
                    {"name": "path", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {
                        "description": "Raw image; no JSON envelope",
                        "headers": {
                            **media_headers,
                            "ETag": {"schema": {"type": "string"}},
                            "Last-Modified": {"schema": {"type": "string"}},
                        },
                        "content": {
                            mime: {"schema": {"type": "string", "format": "binary"}}
                            for mime in ("image/svg+xml", "image/jpeg", "image/png", "image/webp")
                        },
                    },
                    "304": {
                        "description": "Conditional request: not modified; empty body",
                        "headers": media_headers,
                    },
                    **{
                        str(status): {
                            "description": "Error envelope",
                            "headers": header_docs,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        }
                        for status in (404, 405, 500, 503)
                    },
                },
            }
        }
        return schema

    app.openapi = openapi
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv(
            "CLIP_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID", "X-API-Version"],
    )

    return app


app = create_app()
