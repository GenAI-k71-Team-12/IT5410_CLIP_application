"""Evaluate Recall@1/5/10 and MRR for the BM25 or CLIP retrieval engine.

Usage:
    python scripts/evaluate_retrieval.py --method bm25 --limit 500
    python scripts/evaluate_retrieval.py --method bm25 --method clip \
        --artifact artifacts/<version> --output artifacts/eval.json

Only the standard library is imported at module level; CLIP dependencies are
imported inside the ``--method clip`` branch so BM25 evaluation works without
the ML extras.
"""

import argparse
import json
import sys
from pathlib import Path

from clip_retrieval.contracts import Metadata, SearchRequest

EXIT_OK = 0
EXIT_METADATA = 1
EXIT_EMPTY = 2
EXIT_CLIP_UNAVAILABLE = 3

CLIP_HINT = (
    "CLIP evaluation needs --artifact <dir> and the dependencies from "
    "requirements-ml.txt (torch, open_clip_torch, faiss-cpu, pillow)."
)


def recall_at_k(rankings: list[list[str]], targets: list[str], k: int) -> float:
    """Fraction of targets found in the first ``k`` ranked image IDs."""
    if not targets:
        return 0.0
    hits = sum(1 for ranked, target in zip(rankings, targets, strict=True) if target in ranked[:k])
    return hits / len(targets)


def mean_reciprocal_rank(rankings: list[list[str]], targets: list[str]) -> float:
    """Mean of ``1 / rank`` of the first hit, or 0 when the target is missing."""
    if not targets:
        return 0.0
    total = 0.0
    for ranked, target in zip(rankings, targets, strict=True):
        for rank, image_id in enumerate(ranked, start=1):
            if image_id == target:
                total += 1 / rank
                break
    return total / len(targets)


def evaluate(
    engine,
    records: list[Metadata],
    *,
    method: str = "bm25",
    top_k: int = 10,
    dataset: str | None = None,
    split: str | None = None,
) -> dict:
    """Run one query per record using its first caption and score the metrics."""
    rankings: list[list[str]] = []
    targets: list[str] = []
    for record in records:
        hits = engine.search_text(
            SearchRequest(
                query=record.captions[0].text,
                method=method,
                top_k=top_k,
                dataset=dataset,
                split=split,
            )
        )
        rankings.append([hit.image_id for hit in hits])
        targets.append(record.id)
    return {
        "queries": len(targets),
        "recall@1": recall_at_k(rankings, targets, 1),
        "recall@5": recall_at_k(rankings, targets, 5),
        "recall@10": recall_at_k(rankings, targets, 10),
        "mrr": mean_reciprocal_rank(rankings, targets),
    }


def _load_metadata(path: Path) -> list[Metadata]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(Metadata.model_validate_json(line))
    return records


def _build_bm25_engine(metadata_path: Path):
    from clip_retrieval.keyword import Bm25RetrievalEngine

    return Bm25RetrievalEngine.from_metadata(metadata_path)


def _build_clip_engine(artifact_dir: Path):
    try:
        from clip_retrieval.ml.artifact import load_artifact
        from clip_retrieval.ml.encoder import ClipEncoder
        from clip_retrieval.ml.engine import ClipRetrievalEngine
    except ModuleNotFoundError as exc:
        print(f"{CLIP_HINT}\nMissing dependency: {exc}", file=sys.stderr)
        return None
    try:
        artifact = load_artifact(artifact_dir)
        model = artifact.manifest["model"]
        encoder = ClipEncoder(model["name"], model["weights_id"], "cpu")
        return ClipRetrievalEngine(artifact, encoder)
    except Exception as exc:  # artifact files, checksums, model download, ...
        print(f"Could not load CLIP artifact: {exc}\n{CLIP_HINT}", file=sys.stderr)
        return None


def _print_table(results: dict[str, dict]) -> None:
    header = f"{'method':<10} | {'queries':>7} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<10} | {metrics['queries']:>7} | "
            f"{metrics['recall@1']:>6.4f} | {metrics['recall@5']:>6.4f} | "
            f"{metrics['recall@10']:>6.4f} | {metrics['mrr']:>6.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Evaluate Recall@K and MRR")
    parser.add_argument(
        "--metadata",
        default="data/processed/coco2017/metadata.jsonl",
        help="JSONL metadata snapshot used as the corpus",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=["bm25", "clip"],
        default=["bm25"],
        help="Method to evaluate; repeat to compare (default: bm25)",
    )
    parser.add_argument("--artifact", help="CLIP artifact directory (required for clip)")
    parser.add_argument(
        "--limit", type=int, default=0, help="Evaluate the first N records by id (0 = all)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Hits retrieved per query")
    parser.add_argument("--dataset", help="Only evaluate/retrieve inside this dataset")
    parser.add_argument("--split", help="Only evaluate/retrieve inside this split")
    parser.add_argument("--output", help="Write metrics as JSON to this path")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return EXIT_METADATA

    metadata_path = Path(args.metadata)
    if not metadata_path.is_file():
        print(f"Metadata file not found: {metadata_path}", file=sys.stderr)
        return EXIT_METADATA
    try:
        records = _load_metadata(metadata_path)
    except Exception as exc:
        print(f"Could not read metadata: {exc}", file=sys.stderr)
        return EXIT_METADATA

    if not records:
        print("Metadata corpus is empty", file=sys.stderr)
        return EXIT_EMPTY

    if args.dataset is not None:
        records = [record for record in records if record.dataset == args.dataset]
    if args.split is not None:
        records = [record for record in records if record.split == args.split]
    records.sort(key=lambda record: record.id)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        print("No records left after --dataset/--split/--limit", file=sys.stderr)
        return EXIT_EMPTY

    methods = list(dict.fromkeys(args.method))
    engine = None
    if "bm25" in methods:
        engine = _build_bm25_engine(metadata_path)

    results: dict[str, dict] = {}
    for method in methods:
        if method == "clip":
            if not args.artifact or not Path(args.artifact).is_dir():
                print(CLIP_HINT, file=sys.stderr)
                return EXIT_CLIP_UNAVAILABLE
            clip_engine = _build_clip_engine(Path(args.artifact))
            if clip_engine is None:
                return EXIT_CLIP_UNAVAILABLE
            current = clip_engine
        else:
            current = engine
        results[method] = evaluate(
            current,
            records,
            method=method,
            top_k=args.top_k,
            dataset=args.dataset,
            split=args.split,
        )

    _print_table(results)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "methods": results,
                    "config": {
                        "metadata": str(metadata_path),
                        "methods": methods,
                        "top_k": args.top_k,
                        "limit": args.limit,
                        "dataset": args.dataset,
                        "split": args.split,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
