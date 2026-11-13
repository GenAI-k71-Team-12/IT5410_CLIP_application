# tests/test_keyword.py
import json
from pathlib import Path

import pytest

from clip_retrieval.contracts import Metadata, SearchRequest
from clip_retrieval.keyword import (
    STOPWORDS,
    Bm25Index,
    Bm25RetrievalEngine,
    tokenize,
)
from clip_retrieval.retrieval import EngineNotReady


def record(
    image_id: str, captions: list[str], *, dataset: str = "demo", split: str = "test"
) -> Metadata:
    source_id = image_id.split(":", 1)[1]
    return Metadata.model_validate(
        {
            "id": image_id,
            "dataset": dataset,
            "source_id": source_id,
            "split": split,
            "image_path": f"media/{source_id}.svg",
            "width": 10,
            "height": 10,
            "captions": [{"text": text, "language": "en"} for text in captions],
        }
    )


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The DOG, a Cat!") == ["dog", "cat"]
    assert "the" in STOPWORDS


def test_tokenize_keeps_vietnamese_words():
    assert tokenize("Một con mèo") == ["một", "con", "mèo"]


def test_from_records_joins_all_captions():
    index = Bm25Index.from_records(
        [
            record("demo:1", ["a dog in a park", "the dog runs"]),
            record("demo:2", ["a cat on a mat", "the cat sleeps"]),
        ]
    )
    assert index.size == 2
    assert index.document_text("demo:1") == "a dog in a park\nthe dog runs"


def test_rank_returns_matching_document_first():
    index = Bm25Index.from_records(
        [
            record("demo:1", ["a person rides a horse"]),
            record("demo:2", ["a dog runs in the park"]),
        ]
    )
    hits = index.rank("horse", top_k=5)
    assert [hit.image_id for hit in hits] == ["demo:1"]
    assert hits[0].score > 0


def test_rank_ties_break_by_image_id():
    index = Bm25Index.from_records(
        [
            record("demo:b", ["dog"]),
            record("demo:a", ["dog"]),
        ]
    )
    assert [hit.image_id for hit in index.rank("dog", top_k=2)] == ["demo:a", "demo:b"]


def test_rank_filters_before_top_k():
    index = Bm25Index.from_records(
        [
            record("demo:1", ["dog"], split="test"),
            record("demo:2", ["dog"], split="val"),
        ]
    )
    hits = index.rank("dog", split="val", top_k=1)
    assert [hit.image_id for hit in hits] == ["demo:2"]


def test_rank_top_k_and_zero_score_docs_excluded():
    index = Bm25Index.from_records(
        [
            record("demo:1", ["dog dog dog"]),
            record("demo:2", ["dog"]),
            record("demo:3", ["cat"]),
        ]
    )
    hits = index.rank("dog", top_k=1)
    assert len(hits) == 1
    assert index.rank("xyzzy", top_k=5) == []


def test_rank_honors_k1_parameter():
    records = [record("demo:a", ["dog"]), record("demo:b", ["dog dog dog"])]
    assert [hit.image_id for hit in Bm25Index.from_records(records).rank("dog")] == [
        "demo:b",
        "demo:a",
    ]
    assert [hit.image_id for hit in Bm25Index.from_records(records, k1=0.0).rank("dog")] == [
        "demo:a",
        "demo:b",
    ]


def test_rank_honors_b_parameter():
    records = [record("demo:a", ["dog cat"]), record("demo:b", ["dog"])]
    assert [hit.image_id for hit in Bm25Index.from_records(records).rank("dog")] == [
        "demo:b",
        "demo:a",
    ]
    assert [hit.image_id for hit in Bm25Index.from_records(records, b=0.0).rank("dog")] == [
        "demo:a",
        "demo:b",
    ]


def test_rank_stopword_only_query_returns_empty():
    index = Bm25Index.from_records([record("demo:1", ["a dog"])])
    assert index.rank("the a an", top_k=5) == []


def test_empty_index_returns_empty():
    index = Bm25Index.from_records([])
    assert index.size == 0
    assert index.rank("dog", top_k=5) == []


def test_from_metadata_reads_jsonl(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    rows = [
        {
            "id": "demo:1",
            "dataset": "demo",
            "source_id": "1",
            "split": "test",
            "image_path": "media/1.svg",
            "width": 10,
            "height": 10,
            "captions": [{"text": "a horse", "language": "en"}],
        },
        {
            "id": "demo:2",
            "dataset": "demo",
            "source_id": "2",
            "split": "test",
            "image_path": "media/2.svg",
            "width": 10,
            "height": 10,
            "captions": [{"text": "a cat", "language": "en"}],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index = Bm25Index.from_metadata(path)
    assert index.size == 2
    assert index.rank("horse", top_k=1)[0].image_id == "demo:1"


def test_engine_rejects_non_bm25_request():
    engine = Bm25RetrievalEngine(Bm25Index.from_records([record("demo:1", ["dog"])]))
    assert engine.index_version == "bm25-v1-k1.2-b0.75"
    with pytest.raises(EngineNotReady):
        engine.search_text(SearchRequest(query="dog", method="clip"))
    with pytest.raises(EngineNotReady):
        engine.search_similar("demo:1", SearchRequest(query="similar"))


def test_engine_search_text_returns_ranked_images():
    engine = Bm25RetrievalEngine(
        Bm25Index.from_records(
            [
                record("demo:1", ["a horse"]),
                record("demo:2", ["a cat"]),
            ]
        )
    )
    hits = engine.search_text(SearchRequest(query="horse", method="bm25"))
    assert [hit.image_id for hit in hits] == ["demo:1"]
