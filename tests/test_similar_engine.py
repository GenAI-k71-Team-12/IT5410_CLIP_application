import pytest
from fastapi.testclient import TestClient

from clip_retrieval.backend.app import create_app
from clip_retrieval.contracts import SearchRequest
from clip_retrieval.retrieval import EngineNotReady, ImageNotFound, RankedImage, RetrievalFailure


class StubEngine:
    index_version = "stub-v1"

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str, SearchRequest]] = []

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        raise AssertionError("/similar must not call search_text")

    def search_similar(self, image_id: str, request: SearchRequest) -> list[RankedImage]:
        self.calls.append((image_id, request))
        if self.error:
            raise self.error
        return [
            RankedImage("demo:003", 0.81),
            RankedImage("demo:002", 0.42),
            RankedImage("demo:004", -0.1),
        ]


def test_similar_pages_engine_results():
    engine = StubEngine()
    client = TestClient(create_app(engine=engine))
    body = client.get("/images/demo:001/similar?top_k=3&limit=2&offset=1").json()["data"]
    image_id, request = engine.calls[0]
    assert image_id == "demo:001"
    assert request.method == "clip" and request.top_k == 3 and request.min_similarity is None
    assert body["mode"] == "real" and body["index_version"] == "stub-v1"
    assert body["query"] is None and body["score_type"] == "cosine_similarity"
    assert [(h["rank"], h["image"]["id"], h["similarity"]) for h in body["results"]] == [
        (2, "demo:002", 0.42),
        (3, "demo:004", -0.1),
    ]
    assert body["total"] == 3 and not body["has_more"] and body["next_offset"] is None


def test_similar_forwards_min_similarity():
    engine = StubEngine()
    TestClient(create_app(engine=engine)).get("/images/demo:001/similar?min_similarity=0.5")
    assert engine.calls[0][1].min_similarity == 0.5


@pytest.mark.parametrize(
    "error,status,message",
    [
        (ImageNotFound("demo:002"), 404, "Image not found"),
        (EngineNotReady("index not loaded"), 503, "Retrieval engine is not ready"),
        (RetrievalFailure("private detail"), 500, "Internal server error"),
    ],
)
def test_similar_maps_engine_errors(error, status, message):
    app = create_app(engine=StubEngine(error))
    result = TestClient(app, raise_server_exceptions=False).get("/images/demo:002/similar")
    body = result.json()
    assert result.status_code == body["statusCode"] == status
    assert body["success"] is False and body["message"] == [message]
    assert body["path"] == "/images/demo:002/similar"
    assert result.headers["X-Request-ID"] and result.headers["Cache-Control"] == "no-store"


def test_unknown_catalog_image_is_404_without_engine_call():
    engine = StubEngine()
    result = TestClient(create_app(engine=engine)).get("/images/demo:999/similar")
    assert result.status_code == 404 and engine.calls == []


def test_search_and_health_stay_mock_until_real_startup_exists():
    client = TestClient(create_app(engine=StubEngine()))
    assert client.post("/search", json={"query": "dog"}).json()["data"]["mode"] == "mock"
    assert client.get("/health").json()["data"]["mode"] == "mock"


def test_real_mode_without_engine_still_fails_closed():
    with pytest.raises(RuntimeError):
        create_app("real")
