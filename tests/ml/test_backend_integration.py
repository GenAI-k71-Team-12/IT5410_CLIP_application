import json

import pytest
from fakes import FakeEncoder, build_toy, write_dataset
from fastapi.testclient import TestClient

from clip_retrieval.backend.app import create_app
from clip_retrieval.backend.config import Settings
from clip_retrieval.contracts import SearchRequest
from clip_retrieval.ml.artifact import load_artifact
from clip_retrieval.ml.build_index import build_artifact
from clip_retrieval.ml.encoder import ClipEncoder
from clip_retrieval.retrieval import RankedImage


@pytest.fixture
def configured(tmp_path, monkeypatch):
    artifact_dir = build_toy(tmp_path)
    calls = []

    def load_encoder(*args):
        calls.append(args)
        return FakeEncoder()

    monkeypatch.setattr("clip_retrieval.ml.encoder.ClipEncoder", load_encoder)
    settings = Settings(mode="real", artifact_dir=artifact_dir, dataset_root=tmp_path / "data")
    return settings, calls


def test_startup_loads_once_and_shutdown_releases_engine(configured):
    settings, calls = configured
    app = create_app(settings=settings)
    assert calls == []
    with TestClient(app) as client:
        health = client.get("/health").json()["data"]
        assert health == {
            "status": "ok",
            "mode": "real",
            "ready": True,
            "model_loaded": True,
            "index_loaded": True,
            "api_version": "2.0.0",
        }
        for _ in range(2):
            assert client.post("/search", json={"query": "red"}).status_code == 200
        assert len(calls) == 1
    assert app.state.runtime.engine is None
    assert not app.state.runtime.ready


def test_search_matches_engine_and_paginates_same_pool(configured):
    settings, _ = configured
    app = create_app(settings=settings)
    with TestClient(app) as client:
        params = {"query": " red ", "top_k": 4, "limit": 2}
        first = client.post("/search", json=params)
        body = first.json()["data"]
        second = client.post("/search", json={**params, "offset": body["next_offset"]}).json()[
            "data"
        ]
        results = body["results"] + second["results"]
        expected = app.state.runtime.engine.search_text(SearchRequest(**params))
        assert [hit["image"]["id"] for hit in results] == [hit.image_id for hit in expected]
        assert [hit["rank"] for hit in results] == [1, 2, 3, 4]
        assert body["request_id"] == first.headers["X-Request-ID"]
        assert body["mode"] == "real" and body["index_version"] == "toy-v1"
        assert body["query"] == "red" and body["total"] == 4
        assert body["has_more"] and not second["has_more"] and second["next_offset"] is None
        assert all(hit["score"] == hit["similarity"] for hit in results)
        assert client.post("/search", json={**params, "offset": 50}).json()["data"]["results"] == []


def test_filters_and_unloaded_method(configured):
    settings, _ = configured
    with TestClient(create_app(settings=settings)) as client:
        body = client.post("/search", json={"query": "red", "split": "test", "top_k": 1}).json()[
            "data"
        ]
        assert body["results"][0]["image"]["id"] == "toy:e1"
        for filters in ({"dataset": "other"}, {"split": "test", "min_similarity": 0.99}):
            response = client.post("/search", json={"query": "red", **filters}).json()["data"]
            assert response["total"] == 0 and response["results"] == []
        assert client.post("/search", json={"query": "red", "method": "bm25"}).status_code == 503
        result = client.post("/search", json={"query": "x" * 41})
        assert result.status_code == 422 and result.json()["message"]


def test_metadata_media_and_similar_use_artifact_catalog(configured):
    settings, _ = configured
    (settings.dataset_root / "private.txt").write_text("not public")
    with TestClient(create_app(settings=settings)) as client:
        image = client.get("/images/toy:a1").json()["data"]
        assert image["id"] == "toy:a1" and image["captions"][0]["text"] == "a red square"
        assert "image_path" not in image and "source_id" not in image
        assert image["image_url"] == image["thumbnail_url"] == "/media/images/a1.png"
        media = client.get(image["image_url"])
        assert media.status_code == 200 and media.headers["Content-Type"] == "image/png"
        assert client.head(image["image_url"]).status_code == 200
        assert (
            client.get(
                image["image_url"], headers={"If-None-Match": media.headers["ETag"]}
            ).status_code
            == 304
        )
        assert client.get("/media/private.txt").status_code == 404
        assert client.get("/media/%2e%2e/private.txt").status_code == 404
        assert client.get("/images/demo:001").status_code == 404
        similar = client.get("/images/toy:a1/similar?top_k=2&limit=1").json()["data"]
        assert similar["query"] is None and similar["total"] == 2
        assert similar["results"][0]["image"]["id"] == "toy:a2"
        assert similar["has_more"] and similar["next_offset"] == 1


def test_thumbnail_and_encoded_media_path(tmp_path, monkeypatch):
    records = write_dataset(tmp_path / "data")
    (tmp_path / "data/images/a1.png").rename(tmp_path / "data/images/red square.png")
    records[0] = records[0].model_copy(
        update={
            "image_path": "images/red square.png",
            "thumbnail_path": "images/a2.png",
        }
    )
    artifact_dir = build_toy(tmp_path, records=records)
    monkeypatch.setattr("clip_retrieval.ml.encoder.ClipEncoder", lambda *args: FakeEncoder())
    with TestClient(
        create_app(
            settings=Settings(
                mode="real", dataset_root=tmp_path / "data", artifact_dir=artifact_dir
            )
        )
    ) as client:
        image = client.get("/images/toy:a1").json()["data"]
        assert image["image_url"] == "/media/images/red%20square.png"
        assert image["thumbnail_url"] == "/media/images/a2.png"
        assert client.get(image["image_url"]).status_code == 200
        assert client.get(image["thumbnail_url"]).status_code == 200


@pytest.mark.parametrize("failure", ["artifact", "model", "media", "bm25"])
def test_failed_startup_reports_component_state_and_never_serves_mock(
    configured, monkeypatch, failure
):
    settings, _ = configured
    if failure == "artifact":
        manifest_path = settings.artifact_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["index"]["metric"] = "l2"
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "model":

        def broken_encoder(*args):
            raise RuntimeError("private model failure")

        monkeypatch.setattr("clip_retrieval.ml.encoder.ClipEncoder", broken_encoder)
    elif failure == "media":
        (settings.dataset_root / "images/a1.png").unlink()
    else:
        settings = Settings(
            mode="real",
            dataset_root=settings.dataset_root,
            artifact_dir=settings.artifact_dir,
            methods=("clip", "bm25"),
        )
    with TestClient(create_app(settings=settings)) as client:
        health = client.get("/health").json()["data"]
        assert health["mode"] == "real" and health["status"] == "not_ready" and not health["ready"]
        assert health["index_loaded"] == (failure != "artifact")
        assert health["model_loaded"] == (failure in ("media", "bm25"))
        for path in ("/images/toy:a1", "/images/toy:a1/similar", "/media/images/a1.png"):
            assert client.get(path).status_code == 503
        result = client.post("/search", json={"query": "red"})
        assert result.status_code == 503 and result.json()["message"] == [
            "Retrieval engine is not ready"
        ]


@pytest.mark.parametrize(
    "hits",
    [
        [RankedImage("toy:a1", 0.5), RankedImage("toy:a1", 0.4)],
        [RankedImage("toy:missing", 0.5)],
        [RankedImage("toy:a1", float("nan"))],
        [RankedImage("toy:a1", 1.1)],
        [RankedImage("toy:a2", 0.5), RankedImage("toy:a1", 0.5)],
    ],
)
def test_invalid_engine_pool_is_500(configured, monkeypatch, hits):
    settings, _ = configured
    app = create_app(settings=settings)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.runtime.engine, "search_text", lambda request: hits)
        result = client.post("/search", json={"query": "red"})
        assert result.status_code == 500
        assert result.json()["message"] == ["Internal server error"]


def test_api_with_real_clip_encoder(tmp_path):
    pytest.importorskip("open_clip")
    encoder = ClipEncoder()
    records = write_dataset(tmp_path / "data")
    artifact_dir = build_artifact(
        records,
        tmp_path / "data",
        encoder,
        tmp_path / "artifacts",
        "toy-real-v1",
        dataset="toy",
        splits=["test", "val"],
    )
    expected_ids = load_artifact(artifact_dir).image_ids
    with TestClient(
        create_app(
            settings=Settings(
                mode="real", dataset_root=tmp_path / "data", artifact_dir=artifact_dir
            )
        )
    ) as client:
        assert client.get("/health").json()["data"]["ready"]
        result = client.post("/search", json={"query": "a red square", "top_k": 2}).json()["data"]
        assert result["mode"] == "real" and result["total"] == 2
        assert all(hit["image"]["id"] in expected_ids for hit in result["results"])
        image = result["results"][0]["image"]
        assert client.get(image["image_url"]).status_code == 200
        assert client.get(f"/images/{image['id']}/similar").status_code == 200
