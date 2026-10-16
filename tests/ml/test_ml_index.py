import numpy as np
import pytest

from clip_retrieval.ml import index as faiss_index


def unit_rows(n: int, d: int, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def brute_force(x, query, k, rows=None):
    rows = np.arange(len(x)) if rows is None else rows
    scores = x[rows] @ query[0]
    order = np.argsort(-scores, kind="stable")[:k]
    return scores[order], rows[order]


@pytest.mark.parametrize("rows", [None, np.array([1, 4, 9, 30, 77, 150])])
def test_search_matches_brute_force(rows):
    x = unit_rows(200, 32)
    index = faiss_index.build(x)
    for q in (0, 50, 199):
        scores, found = faiss_index.search(index, x[q : q + 1], 5, rows)
        expected_scores, expected_rows = brute_force(x, x[q : q + 1], 5, rows)
        np.testing.assert_allclose(scores, expected_scores, atol=1e-5)
        assert found.tolist() == expected_rows.tolist()


def test_k_beyond_candidates_returns_only_real_rows():
    x = unit_rows(5, 8)
    scores, found = faiss_index.search(faiss_index.build(x), x[:1], 50, np.array([2, 3]))
    assert sorted(found.tolist()) == [2, 3] and len(scores) == 2


def test_save_load_roundtrip_with_unicode_path(tmp_path):
    x = unit_rows(20, 8)
    path = tmp_path / "dữ liệu" / "index.faiss"
    path.parent.mkdir()
    faiss_index.save(faiss_index.build(x), path)
    loaded = faiss_index.load(path)
    assert loaded.ntotal == 20 and loaded.d == 8
    np.testing.assert_allclose(loaded.reconstruct_n(0, 20), x, atol=1e-6)
