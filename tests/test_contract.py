import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from clip_retrieval.backend.app import create_app
from clip_retrieval.contracts import Metadata, SearchRequest

client = TestClient(create_app("mock"))


def test_health_is_explicit_mock():
    body = client.get("/health").json()["data"]
    assert body["mode"] == "mock" and body["ready"]
    assert not body["model_loaded"] and not body["index_loaded"]


def test_pagination_and_top_k():
    first = client.post("/search", json={"query": "a dog", "top_k": 3, "limit": 2}).json()["data"]
    second = client.post(
        "/search", json={"query": "a dog", "top_k": 3, "limit": 2, "offset": first["next_offset"]}
    ).json()["data"]
    assert first["total"] == 3 and first["has_more"]
    assert [h["rank"] for h in first["results"] + second["results"]] == [1, 2, 3]
    assert not second["has_more"] and second["next_offset"] is None
    assert len({h["image"]["id"] for h in first["results"] + second["results"]}) == 3


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 101},
        {"query": "x", "offset": -1},
        {"query": "x", "limit": 51},
        {"query": "x", "top_k": True},
        {"query": "x", "top_k": "10"},
        {"query": "x", "method": "unknown"},
        {"query": "x", "extra": 1},
        {"query": "x", "method": "bm25", "min_similarity": 0.5},
        {"query": "x", "min_similarity": 1.1},
    ],
)
def test_invalid_request(body):
    result = client.post("/search", json=body)
    assert result.status_code == 422
    assert result.json()["success"] is False
    assert result.json()["statusCode"] == 422
    assert result.json()["message"]
    assert result.headers["X-Request-ID"]


def test_filter_threshold_empty_and_beyond_end():
    assert (
        client.post("/search", json={"query": "x", "min_similarity": 0.8}).json()["data"]["total"]
        == 1
    )
    for args in ({"dataset": "absent"}, {"offset": 100}):
        body = client.post("/search", json={"query": "x", **args}).json()["data"]
        assert body["results"] == [] and not body["has_more"]


def test_bm25_distinct_score():
    body = client.post("/search", json={"query": "x", "method": "bm25"}).json()["data"]
    assert body["score_type"] == "bm25_score"
    assert all(hit["similarity"] is None for hit in body["results"])


def test_similar_excludes_source_and_media_loads():
    body = client.get("/images/demo:001/similar").json()["data"]
    assert all(hit["image"]["id"] != "demo:001" for hit in body["results"])
    image = client.get("/images/demo:001").json()["data"]
    assert client.get(image["image_url"]).status_code == 200
    assert client.get("/images/demo:missing").status_code == 404
    assert client.get("/media/missing.svg").json()["statusCode"] == 404


def test_real_mode_fails_closed():
    with pytest.raises(RuntimeError):
        create_app("real")


def test_metadata_jsonschema_and_semantics():
    path = Path("data/examples/metadata.jsonl")
    schema = Metadata.model_json_schema()
    Draft202012Validator.check_schema(schema)
    record = json.loads(path.read_text().splitlines()[0])
    Draft202012Validator(schema).validate(record)
    Metadata.model_validate(record)
    for changes in (
        {"image_path": "../secret"},
        {"width": 0},
        {"id": "demo:wrong"},
        {"captions": []},
        {"thumbnail_path": "C:\\secret"},
    ):
        with pytest.raises(ValidationError):
            Metadata.model_validate({**record, **changes})


def test_whitespace_query_normalized():
    assert SearchRequest(query="  a dog  ").query == "a dog"


@pytest.mark.parametrize(
    "method,path,kwargs",
    [
        ("GET", "/health", {}),
        ("POST", "/search", {"json": {"query": "dog"}}),
        ("GET", "/images/demo:001", {}),
        ("GET", "/images/demo:001/similar", {}),
    ],
)
def test_success_envelope(method, path, kwargs):
    before = datetime.now(UTC)
    result = client.request(method, path, **kwargs)
    after = datetime.now(UTC)
    body = result.json()
    assert set(body) == {"success", "statusCode", "data", "timestamp"}
    assert body["success"] is True
    assert body["statusCode"] == result.status_code == 200
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", body["timestamp"])
    assert before.timestamp() - 0.001 <= datetime.fromisoformat(body["timestamp"]).timestamp()
    assert datetime.fromisoformat(body["timestamp"]) <= after
    if "request_id" in body["data"]:
        assert body["data"]["request_id"] == result.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "method,path,kwargs,status",
    [
        ("POST", "/search", {"json": {"query": "dog", "extraField": 1}}, 422),
        ("POST", "/search", {"content": "{", "headers": {"Content-Type": "application/json"}}, 422),
        ("GET", "/images/demo:001/similar?limit=0", {}, 422),
        ("GET", "/images/demo:missing", {}, 404),
        ("GET", "/missing", {}, 404),
        ("GET", "/media/missing.svg", {}, 404),
        ("GET", "/search", {}, 405),
    ],
)
def test_error_envelope(method, path, kwargs, status):
    result = client.request(method, path, **kwargs)
    body = result.json()
    assert set(body) == {"success", "statusCode", "message", "path", "timestamp"}
    assert body["success"] is False
    assert body["statusCode"] == result.status_code == status
    assert body["path"] == path.split("?")[0]
    assert body["message"] and all(isinstance(message, str) for message in body["message"])
    assert body["timestamp"].endswith("Z")
    assert result.headers["X-Request-ID"]
    if "extraField" in str(kwargs):
        assert body["message"] == ["property extraField should not exist"]


def test_internal_error_does_not_leak_exception():
    app = create_app("mock")

    @app.get("/fail")
    def fail():
        raise RuntimeError("secret internal path")

    result = TestClient(app, raise_server_exceptions=False).get("/fail")
    assert result.status_code == 500
    assert result.json()["message"] == ["Internal server error"]
    assert result.json()["path"] == "/fail"
    assert result.headers["X-Request-ID"]


def test_openapi_and_examples_match_envelope():
    from jsonschema import FormatChecker
    from referencing import Registry, Resource

    schema = create_app("mock").openapi()
    registry = Registry().with_resource(
        "urn:api",
        Resource.from_contents(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                **schema,
            }
        ),
    )
    for filename, path, status in [
        ("health", "/health", "200"),
        ("search", "/search", "200"),
        ("bm25", "/search", "200"),
        ("image", "/images/{image_id}", "200"),
        ("similar", "/images/{image_id}/similar", "200"),
        ("error", "/search", "422"),
    ]:
        operation = schema["paths"][path]["post" if path == "/search" else "get"]
        ref = operation["responses"][status]["content"]["application/json"]["schema"]["$ref"]
        validator = Draft202012Validator(
            {"$ref": "urn:api" + ref}, registry=registry, format_checker=FormatChecker()
        )
        payload = json.loads(Path(f"contracts/examples/{filename}.response.json").read_text())
        validator.validate(payload)


@pytest.mark.parametrize(
    "path", ["/health", "/search", "/images/demo:001", "/images/demo:001/similar"]
)
def test_unknown_query_is_rejected(path):
    result = client.request(
        "POST" if path == "/search" else "GET",
        path + "?extraField=1",
        **({"json": {"query": "dog"}} if path == "/search" else {}),
    )
    assert result.status_code == 422
    assert result.json()["message"] == ["property extraField should not exist"]


@pytest.mark.parametrize("path", ["/images/invalid", "/images/invalid/similar"])
def test_invalid_image_id(path):
    result = client.get(path)
    assert result.status_code == 422
    assert any("path.image_id" in item for item in result.json()["message"])


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_search_requires_json_media_type(content_type):
    result = client.post(
        "/search",
        content='{"query":"dog"}',
        headers={"Content-Type": content_type} if content_type else {},
    )
    assert result.status_code == 415
    assert result.json()["statusCode"] == 415


def test_json_charset_is_accepted():
    result = client.post(
        "/search",
        content='{"query":"dog"}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert result.status_code == 200


@pytest.mark.parametrize(
    "path", ["/health", "/images/demo:001", "/images/demo:missing", "/missing"]
)
def test_common_headers_and_cors(path):
    from uuid import UUID

    result = client.get(path, headers={"Origin": "http://localhost:3000"})
    UUID(result.headers["X-Request-ID"])
    assert result.headers["X-API-Version"] == "2.0.0"
    assert result.headers["Cache-Control"] == "no-store"
    assert result.headers["Content-Type"] == "application/json"
    assert result.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    assert "X-API-Version" in result.headers["Access-Control-Expose-Headers"]


def test_media_conditional_headers_and_preflight():
    media = client.get("/media/001.svg")
    assert media.status_code == 200
    assert media.headers["Content-Type"].startswith("image/svg+xml")
    assert media.headers["X-API-Version"] == "2.0.0"
    cached = client.get("/media/001.svg", headers={"If-None-Match": media.headers["ETag"]})
    assert cached.status_code == 304 and cached.content == b""
    preflight = client.options(
        "/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


def test_internal_error_has_common_headers_and_cors():
    app = create_app("mock")

    @app.get("/fail")
    def fail():
        raise RuntimeError("private failure")

    result = TestClient(app).get("/fail", headers={"Origin": "http://localhost:3000"})
    assert result.status_code == 500
    assert result.headers["Cache-Control"] == "no-store"
    assert result.headers["X-API-Version"] == "2.0.0"
    assert result.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


def test_openapi_headers_and_endpoint_statuses():
    schema = create_app("mock").openapi()
    for item in schema["paths"].values():
        for operation in item.values():
            for response in operation["responses"].values():
                assert "X-Request-ID" in response["headers"]
                assert "X-API-Version" in response["headers"]
    assert "415" in schema["paths"]["/search"]["post"]["responses"]
    assert "304" in schema["paths"]["/media/{path}"]["get"]["responses"]
