# tests/test_backend_bm25.py
from pathlib import Path

from fastapi.testclient import TestClient

from clip_retrieval.backend.app import create_app
from clip_retrieval.contracts import SearchRequest
from clip_retrieval.keyword import Bm25RetrievalEngine
from clip_retrieval.retrieval import RankedImage


class StubBm25:
    index_version = "bm25-test"

    def __init__(self):
        self.calls: list[SearchRequest] = []

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        self.calls.append(request)
        return [RankedImage("demo:003", 3.5), RankedImage("demo:001", 1.25)]

    def search_similar(self, image_id, request):
        raise AssertionError("BM25 must not serve /similar")


class StubClip:
    index_version = "clip-test"

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        return [RankedImage("demo:002", 0.8)]

    def search_similar(self, image_id, request):
        raise AssertionError("not used")


def test_search_bm25_uses_injected_engine():
    engine = StubBm25()
    client = TestClient(create_app(bm25_engine=engine))
    response = client.post("/search", json={"query": "dog", "method": "bm25", "top_k": 2})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["method"] == "bm25"
    assert data["score_type"] == "bm25_score"
    assert data["mode"] == "real"
    assert data["index_version"] == "bm25-test"
    assert [hit["image"]["id"] for hit in data["results"]] == ["demo:003", "demo:001"]
    assert all(hit["similarity"] is None for hit in data["results"])
    assert data["results"][0]["score"] == 3.5
    assert engine.calls[0].query == "dog"


def test_search_clip_uses_clip_engine_when_injected():
    client = TestClient(create_app(engine=StubClip()))
    data = client.post("/search", json={"query": "beach", "method": "clip"}).json()["data"]
    assert data["score_type"] == "cosine_similarity"
    assert data["results"][0]["similarity"] == 0.8
    assert data["mode"] == "real"


def test_search_bm25_without_engine_keeps_mock():
    data = (
        TestClient(create_app("mock"))
        .post("/search", json={"query": "dog", "method": "bm25", "top_k": 2})
        .json()["data"]
    )
    assert data["score_type"] == "bm25_score"
    assert data["mode"] == "mock"
    assert data["results"]


def test_real_mode_accepts_bm25_engine_only():
    app = create_app("real", bm25_engine=StubBm25())
    data = TestClient(app).post("/search", json={"query": "dog", "method": "bm25"}).json()["data"]
    assert data["mode"] == "real"


def test_real_mode_without_matching_engine_returns_503():
    client = TestClient(create_app("real", bm25_engine=StubBm25()))
    response = client.post("/search", json={"query": "dog", "method": "clip"})
    assert response.status_code == 503


def test_real_mode_bm25_only_similar_returns_503():
    client = TestClient(create_app("real", bm25_engine=StubBm25()))
    assert client.get("/images/demo:001/similar").status_code == 503


def test_health_reports_real_mode_and_loaded_index():
    client = TestClient(create_app("real", bm25_engine=StubBm25()))
    health = client.get("/health").json()["data"]
    assert health["mode"] == "real"
    assert health["model_loaded"] is False
    assert health["index_loaded"] is True


def test_search_bm25_with_real_index_serves_example_catalog():
    engine = Bm25RetrievalEngine.from_metadata(Path("data/examples/metadata.jsonl"))
    client = TestClient(create_app("real", bm25_engine=engine))
    data = client.post("/search", json={"query": "dog", "method": "bm25", "top_k": 2}).json()[
        "data"
    ]
    assert data["mode"] == "real"
    assert data["score_type"] == "bm25_score"
    assert data["index_version"] == "bm25-v1-k1.2-b0.75"
    assert data["results"][0]["image"]["id"] == "demo:001"
    assert data["results"][0]["similarity"] is None
