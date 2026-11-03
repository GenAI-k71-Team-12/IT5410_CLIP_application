import numpy as np
import pytest
from PIL import Image

from clip_retrieval.ml.checksum import sha256_json
from clip_retrieval.ml.encoder import ClipEncoder
from clip_retrieval.retrieval import QueryTooLong


@pytest.fixture(scope="module")
def encoder():
    pytest.importorskip("open_clip")
    return ClipEncoder()


def test_embeddings_are_unit_float32(encoder):
    images = encoder.encode_images([Image.new("RGB", (64, 48), c) for c in ("red", "blue")])
    text = encoder.encode_text("a red square")
    assert images.shape == (2, 512) and text.shape == (1, 512)
    for vectors in (images, text):
        assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)


def test_openai_weights_run_with_their_quickgelu_activation(encoder):
    activations = {type(module).__name__ for module in encoder.model.modules()}
    assert "QuickGELU" in activations and "GELU" not in activations


def test_text_matches_its_colour(encoder):
    images = encoder.encode_images([Image.new("RGB", (64, 64), c) for c in ("red", "green")])
    scores = images @ encoder.encode_text("a photo of the color red")[0]
    assert scores[0] > scores[1]


def test_query_over_context_length_is_rejected_not_truncated(encoder):
    assert encoder.encode_text("dog " * 75).shape == (1, 512)  # 75 + start/end = 77
    with pytest.raises(QueryTooLong):
        encoder.encode_text("dog " * 76)


def test_vietnamese_query_is_encoded(encoder):
    assert encoder.encode_text("chó chạy trên bãi biển").shape == (1, 512)


def test_model_info_identifies_weights_tokenizer_and_preprocess(encoder):
    info = encoder.model_info
    assert info["library"] == "open_clip_torch" and info["name"] == "ViT-B-32"
    assert info["weights_id"] == "openai" and info["context_length"] == 77
    for key in ("weights_sha256", "tokenizer_config_sha256", "preprocess_config_sha256"):
        assert len(info[key]) == 64
    assert info["tokenizer_config_sha256"] == sha256_json(
        encoder.config_files["tokenizer_config.json"]
    )
    assert info["preprocess_config_sha256"] == sha256_json(
        encoder.config_files["preprocess_config.json"]
    )
