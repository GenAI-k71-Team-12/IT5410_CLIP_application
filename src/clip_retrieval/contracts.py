"""Authoritative API v2 and metadata v1 contracts. Export JSON Schema/OpenAPI from these models."""

from datetime import UTC, datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ImageId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*:[A-Za-z0-9_-]+$", max_length=160)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Method = Literal["clip", "bm25"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Caption(Contract):
    text: Annotated[str, Field(min_length=1, max_length=10000)]
    language: Literal["en", "vi", "und"]

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Caption must contain non-whitespace characters")
        return value


class Metadata(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: ImageId
    dataset: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    source_id: Annotated[str, Field(min_length=1, max_length=128)]
    split: Literal["train", "val", "test", "unspecified"]
    image_path: Annotated[str, Field(min_length=1)]
    thumbnail_path: str | None = None
    width: PositiveInt
    height: PositiveInt
    captions: list[Caption] = Field(min_length=1)
    license: str | None = None
    source_url: str | None = None

    @field_validator("image_path", "thumbnail_path")
    @classmethod
    def relative_path(cls, value: str | None) -> str | None:
        if value is not None:
            if (
                value.startswith("/")
                or "\\" in value
                or ":" in value
                or any(p in ("", ".", "..") for p in value.split("/"))
            ):
                raise ValueError("Use a safe relative POSIX path under the dataset root")
        return value

    @model_validator(mode="after")
    def identity(self):
        if self.id != f"{self.dataset}:{self.source_id}":
            raise ValueError("id must be dataset:source_id")
        return self


class ImagePublic(Contract):
    id: ImageId
    dataset: str
    split: Literal["train", "val", "test", "unspecified"]
    width: PositiveInt
    height: PositiveInt
    captions: list[Caption]
    image_url: str
    thumbnail_url: str


class SearchRequest(Contract):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    method: Method = "clip"
    top_k: Annotated[int, Field(strict=True, ge=1, le=100)] = 20
    offset: Annotated[int, Field(strict=True, ge=0)] = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=50)] = 20
    min_similarity: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None = None
    dataset: str | None = None
    split: Literal["train", "val", "test", "unspecified"] | None = None

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank")
        return value

    @model_validator(mode="after")
    def compatible_score(self):
        if self.method == "bm25" and self.min_similarity is not None:
            raise ValueError("min_similarity only applies to CLIP cosine similarity")
        return self


class NoQuery(Contract):
    """Endpoints without query parameters reject unknown fields."""


class SimilarRequest(Contract):
    # Query values arrive as strings, unlike JSON integers in SearchRequest.
    top_k: Annotated[int, Field(ge=1, le=100)] = 20
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=50)] = 20
    min_similarity: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None = None


class SearchHit(Contract):
    rank: PositiveInt
    image: ImagePublic
    score: Annotated[float, Field(allow_inf_nan=False)]
    similarity: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None


class SearchResponse(Contract):
    request_id: str
    api_version: Literal["2.0.0"] = "2.0.0"
    mode: Literal["mock", "real"]
    method: Method
    score_type: Literal["cosine_similarity", "bm25_score"]
    query: str | None
    top_k: int
    offset: int
    limit: int
    total: int
    has_more: bool
    next_offset: int | None
    index_version: str
    results: list[SearchHit]


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


Timestamp = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        json_schema_extra={"format": "date-time"},
    ),
]
Payload = TypeVar("Payload")


class SuccessResponse(Contract, Generic[Payload]):
    success: Literal[True] = True
    statusCode: Literal[200] = 200
    data: Payload
    timestamp: Timestamp = Field(default_factory=utc_timestamp)


class ErrorResponse(Contract):
    success: Literal[False] = False
    statusCode: Annotated[int, Field(ge=400, le=599)]
    message: list[str] = Field(min_length=1)
    path: str
    timestamp: Timestamp = Field(default_factory=utc_timestamp)


class Health(Contract):
    status: Literal["ok", "not_ready"]
    mode: Literal["mock", "real"]
    ready: bool
    model_loaded: bool
    index_loaded: bool
    api_version: Literal["2.0.0"] = "2.0.0"
