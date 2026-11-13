# tests/test_evaluate_retrieval.py
import json
from pathlib import Path

import pytest

import scripts.evaluate_retrieval as evaluate_retrieval
from clip_retrieval.contracts import SearchRequest
from clip_retrieval.retrieval import RankedImage


def record(image_id: str, caption: str, *, split: str = "val") -> dict:
    source_id = image_id.split(":", 1)[1]
    return {
        "id": image_id,
        "dataset": "demo",
        "source_id": source_id,
        "split": split,
        "image_path": f"media/{source_id}.svg",
        "width": 10,
        "height": 10,
        "captions": [{"text": caption, "language": "en"}],
    }


def test_recall_at_k():
    rankings = [["a", "b", "c"], ["b", "a", "c"]]
    targets = ["a", "c"]
    assert evaluate_retrieval.recall_at_k(rankings, targets, 1) == 0.5
    assert evaluate_retrieval.recall_at_k(rankings, targets, 2) == 0.5
    assert evaluate_retrieval.recall_at_k(rankings, targets, 3) == 1.0


def test_mean_reciprocal_rank():
    rankings = [["a", "b"], ["b", "a"]]
    targets = ["a", "c"]
    assert evaluate_retrieval.mean_reciprocal_rank(rankings, targets) == pytest.approx(0.5)


class StubEngine:
    index_version = "stub"

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        if request.query == "a dog":
            return [RankedImage("demo:1", 2.0), RankedImage("demo:2", 1.0)]
        return [RankedImage("demo:2", 1.0)]

    def search_similar(self, image_id, request):
        raise AssertionError("not used")


def test_evaluate_returns_metrics():
    from clip_retrieval.contracts import Metadata

    records = [
        Metadata.model_validate(record("demo:1", "a dog")),
        Metadata.model_validate(record("demo:2", "a cat")),
    ]
    result = evaluate_retrieval.evaluate(StubEngine(), records, top_k=5)
    assert result["queries"] == 2
    assert result["recall@1"] == 1.0
    assert result["recall@5"] == 1.0
    assert result["mrr"] == pytest.approx(1.0)


def test_main_writes_json_for_bm25(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            json.dumps(record(f"demo:{i}", text))
            for i, text in [(1, "a dog in a park"), (2, "a cat on a mat")]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    code = evaluate_retrieval.main(
        [
            "--metadata",
            str(metadata),
            "--method",
            "bm25",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["methods"]["bm25"]["queries"] == 2
    assert set(payload["methods"]["bm25"]) >= {"recall@1", "recall@5", "recall@10", "mrr"}


def test_main_errors_on_empty_corpus(tmp_path: Path):
    metadata = tmp_path / "empty.jsonl"
    metadata.write_text("", encoding="utf-8")
    assert evaluate_retrieval.main(["--metadata", str(metadata), "--method", "bm25"]) == 2


def test_main_filters_queries_by_split(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    rows = [
        record("demo:1", "a dog in a park", split="val"),
        record("demo:2", "a cat on a mat", split="val"),
        record("demo:3", "a horse in a field", split="train"),
    ]
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "result.json"
    code = evaluate_retrieval.main(
        [
            "--metadata",
            str(metadata),
            "--method",
            "bm25",
            "--split",
            "val",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["methods"]["bm25"]["queries"] == 2


def test_main_clip_without_artifact_returns_3(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps(record("demo:1", "a dog")) + "\n", encoding="utf-8")
    assert evaluate_retrieval.main(["--metadata", str(metadata), "--method", "clip"]) == 3
