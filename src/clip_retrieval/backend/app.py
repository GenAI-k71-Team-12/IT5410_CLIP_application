"""Explicit fixture API for integration. No CLIP weights, FAISS or BM25 loaded."""

import json
import logging
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from clip_retrieval.contracts import (
    ErrorResponse,
    Health,
    ImageId,
    ImagePublic,
    Metadata,
    NoQuery,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SuccessResponse,
)

ROOT = Path(__file__).resolve().parents[3]


def create_app(mode: str | None = None) -> FastAPI:
    mode = mode or os.getenv("CLIP_MODE", "mock")
    if mode != "mock":
        raise RuntimeError(
            "Only mock mode is implemented; real engine must be integrated explicitly"
        )
    app = FastAPI(title="CLIP Image Retrieval — Mock Contract", version="2.0.0")
    records = [
        Metadata.model_validate(json.loads(line))
        for line in (ROOT / "data/examples/metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    images = {
        item.id: ImagePublic(
            id=item.id,
            dataset=item.dataset,
            split=item.split,
            width=item.width,
            height=item.height,
            captions=item.captions,
            image_url=f"/media/{item.source_id}.svg",
            thumbnail_url=f"/media/{item.source_id}.svg",
        )
        for item in records
    }
    app.mount("/media", StaticFiles(directory=ROOT / "data/examples/media"), name="media")

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
        return SuccessResponse[Health](
            data=Health(
                status="ok", mode="mock", ready=True, model_loaded=False, index_loaded=False
            )
        )

    def require_image(image_id):
        if image_id not in images:
            raise HTTPException(status_code=404, detail="Image not found")
        return images[image_id]

    def response(params: SearchRequest, request: Request, exclude=None):
        # Fixed scores are test fixtures; query does not change ordering in mock mode.
        pool = [
            (item, 0.9 - i * 0.15)
            for i, item in enumerate(images.values())
            if item.id != exclude
            and (params.dataset is None or item.dataset == params.dataset)
            and (params.split is None or item.split == params.split)
        ]
        if params.method == "bm25":
            pool = [(item, round(score * 10, 6)) for item, score in pool]
        if params.min_similarity is not None:
            pool = [(item, score) for item, score in pool if score >= params.min_similarity]
        pool.sort(key=lambda pair: (-pair[1], pair[0].id))
        pool = pool[: params.top_k]
        hits = [
            SearchHit(
                rank=i + params.offset + 1,
                image=item,
                score=round(score, 6),
                similarity=round(score, 6) if params.method == "clip" else None,
            )
            for i, (item, score) in enumerate(pool[params.offset : params.offset + params.limit])
        ]
        end = params.offset + len(hits)
        return SearchResponse(
            request_id=request.state.request_id,
            mode="mock",
            method=params.method,
            score_type="cosine_similarity" if params.method == "clip" else "bm25_score",
            query=None if exclude else params.query,
            top_k=params.top_k,
            offset=params.offset,
            limit=params.limit,
            total=len(pool),
            has_more=end < len(pool),
            next_offset=end if end < len(pool) else None,
            index_version="mock-v1",
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
        return SuccessResponse[SearchResponse](data=response(params, request))

    @app.get(
        "/images/{image_id}",
        response_model=SuccessResponse[ImagePublic],
        responses=responses(404, 405, 422, 500),
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
        return SuccessResponse[SearchResponse](data=response(params, request, exclude=image_id))

    # StaticFiles mounts are not exported by FastAPI; describe the media endpoint explicitly.
    default_openapi = app.openapi

    def openapi():
        schema = default_openapi()
        media_headers = {key: value for key, value in header_docs.items() if key != "Cache-Control"}
        schema["paths"]["/media/{path}"] = {
            "get": {
                "operationId": "get_media",
                "summary": "Download a fixture image (raw bytes)",
                "parameters": [
                    {"name": "path", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {
                        "description": "Raw SVG; no JSON envelope",
                        "headers": {
                            **media_headers,
                            "ETag": {"schema": {"type": "string"}},
                            "Last-Modified": {"schema": {"type": "string"}},
                        },
                        "content": {
                            "image/svg+xml": {"schema": {"type": "string", "format": "binary"}}
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
                        for status in (404, 405, 500)
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
