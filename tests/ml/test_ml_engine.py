import pytest
from fakes import FakeEncoder, build_toy

from clip_retrieval.contracts import SearchRequest
from clip_retrieval.ml.artifact import load_artifact
from clip_retrieval.ml.engine import ClipRetrievalEngine
from clip_retrieval.retrieval import ArtifactInvalid, EngineNotReady, ImageNotFound, QueryTooLong


def make_engine(tmp_path, **kwargs):
    return ClipRetrievalEngine(load_artifact(build_toy(tmp_path)), FakeEncoder(), **kwargs)


def ids(results):
    return [result.image_id for result in results]


def test_text_search_ranks_by_cosine_and_breaks_ties_by_id(tmp_path):
    engine = make_engine(tmp_path)
    results = engine.search_text(SearchRequest(query="red", top_k=4))
    assert ids(results) == ["toy:a1", "toy:a2", "toy:a3", "toy:b1"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6) and results[0].score <= 1.0
    assert engine.index_version == "toy-v1"


@pytest.mark.parametrize("top_k,expected", [(1, ["toy:a1"]), (2, ["toy:a1", "toy:a2"])])
def test_ties_at_top_k_boundary_keep_smallest_ids(tmp_path, top_k, expected):
    results = make_engine(tmp_path).search_text(SearchRequest(query="red", top_k=top_k))
    assert ids(results) == expected


def test_filters_apply_before_top_k(tmp_path):
    engine = make_engine(tmp_path)
    results = engine.search_text(SearchRequest(query="red", top_k=2, split="test"))
    assert ids(results) == ["toy:e1", "toy:c1"]


def test_threshold_empty_dataset_and_top_k_beyond_corpus(tmp_path):
    engine = make_engine(tmp_path)
    above = engine.search_text(SearchRequest(query="red", min_similarity=0.99))
    assert ids(above) == ["toy:a1", "toy:a2", "toy:a3"]
    assert engine.search_text(SearchRequest(query="red", dataset="other")) == []
    assert len(engine.search_text(SearchRequest(query="red", top_k=100))) == 7


def test_similar_excludes_source_before_top_k(tmp_path):
    engine = make_engine(tmp_path)
    request = SearchRequest(query="similar", top_k=2)
    assert ids(engine.search_similar("toy:a2", request)) == ["toy:a1", "toy:a3"]
    in_test = SearchRequest(query="similar", top_k=1, split="test")
    assert ids(engine.search_similar("toy:a1", in_test)) == ["toy:e1"]


def test_similar_unknown_image(tmp_path):
    with pytest.raises(ImageNotFound):
        make_engine(tmp_path).search_similar("toy:zz", SearchRequest(query="similar"))


def test_bm25_is_not_served_and_never_falls_back(tmp_path):
    with pytest.raises(EngineNotReady):
        make_engine(tmp_path).search_text(SearchRequest(query="red", method="bm25"))


def test_too_long_query_fails_even_when_filters_match_nothing(tmp_path):
    with pytest.raises(QueryTooLong):
        make_engine(tmp_path).search_text(SearchRequest(query="x" * 41, dataset="other"))


def test_cache_matches_uncached_and_survives_caller_mutation(tmp_path):
    artifact = load_artifact(build_toy(tmp_path))
    encoder = FakeEncoder()
    cached = ClipRetrievalEngine(artifact, encoder)
    uncached = ClipRetrievalEngine(artifact, FakeEncoder(), cache_size=0)
    request = SearchRequest(query="red", top_k=3)
    first = cached.search_text(request)
    first.clear()
    assert cached.search_text(request) == uncached.search_text(request)
    assert ids(cached.search_text(request)) == ["toy:a1", "toy:a2", "toy:a3"]
    cached.search_text(SearchRequest(query="red", top_k=2))
    assert encoder.texts_encoded == 1


def test_engine_rejects_encoder_that_differs_from_manifest(tmp_path):
    artifact = load_artifact(build_toy(tmp_path))
    with pytest.raises(ArtifactInvalid, match="name"):
        ClipRetrievalEngine(artifact, FakeEncoder(name="other-model"))
