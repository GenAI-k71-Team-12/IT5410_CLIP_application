import pytest
from fastapi.testclient import TestClient

from clip_retrieval.backend.app import create_app
from clip_retrieval.backend.config import Settings


def test_settings_read_process_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIP_MODE", "real")
    monkeypatch.setenv("CLIP_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CLIP_ARTIFACT_DIR", str(tmp_path / "artifact"))
    monkeypatch.setenv("CLIP_DEVICE", "cuda")
    monkeypatch.setenv("CLIP_METHODS", "clip, bm25")
    settings = Settings.from_env()
    assert settings == Settings(
        "real", tmp_path / "data", tmp_path / "artifact", "cuda", ("clip", "bm25")
    )
    assert Settings.from_env("mock").mode == "mock"


@pytest.mark.parametrize("kwargs", [{"mode": "typo"}, {"methods": ()}, {"methods": ("other",)}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_missing_real_configuration_has_no_demo_media():
    with TestClient(create_app(settings=Settings(mode="real"))) as client:
        health = client.get("/health").json()["data"]
        assert health["mode"] == "real" and not health["ready"]
        assert not health["model_loaded"] and not health["index_loaded"]
        assert client.get("/media/001.svg").status_code == 503
        assert client.get("/images/demo:001").status_code == 503
