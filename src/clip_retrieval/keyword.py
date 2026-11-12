"""Okapi BM25 keyword retrieval, standard library only.

The index is rebuilt from the metadata snapshot at startup; this baseline does
not persist an index file. The configuration is documented in
``docs/05_ML_INTEGRATION_CONTRACT.md`` §5.4.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from math import log
from pathlib import Path

from clip_retrieval.contracts import Metadata, SearchRequest
from clip_retrieval.retrieval import EngineNotReady, RankedImage

_WORD = re.compile(r"\w+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
    }
)

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
INDEX_VERSION = "bm25-v1-k1.2-b0.75"


def tokenize(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Lowercase ``text``, split on ``\\w+`` and drop stopwords."""
    return [word for word in _WORD.findall(text.lower()) if word not in stopwords]


class Bm25Index:
    """Okapi BM25 index backed by an inverted index.

    One document per image, keyed by :attr:`Metadata.id`. Documents are only
    scored through their postings, so a query never scans the whole corpus.
    """

    version = INDEX_VERSION

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._documents: dict[str, str] = {}
        self._meta: dict[str, tuple[str, str]] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._dataset_ids: dict[str, set[str]] = defaultdict(set)
        self._split_ids: dict[str, set[str]] = defaultdict(set)
        self._size = 0
        self._avgdl = 0.0

    @classmethod
    def from_records(
        cls,
        records: Iterable[Metadata],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> Bm25Index:
        """Build an index from metadata records, skipping images without captions."""
        index = cls(k1=k1, b=b)
        for record in records:
            if not record.captions:
                continue
            image_id = record.id
            text = "\n".join(caption.text for caption in record.captions)
            tokens = tokenize(text, stopwords)
            index._documents[image_id] = text
            index._meta[image_id] = (record.dataset, record.split)
            index._lengths[image_id] = len(tokens)
            index._dataset_ids[record.dataset].add(image_id)
            index._split_ids[record.split].add(image_id)
            for term, count in Counter(tokens).items():
                index._postings[term][image_id] = count
        index._size = len(index._documents)
        if index._size:
            index._avgdl = sum(index._lengths.values()) / index._size
        return index

    @classmethod
    def from_metadata(
        cls,
        path: Path,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> Bm25Index:
        """Build an index from a JSONL metadata file."""
        with path.open("r", encoding="utf-8") as handle:
            records = (Metadata.model_validate_json(line) for line in handle if line.strip())
            return cls.from_records(records, k1=k1, b=b, stopwords=stopwords)

    @property
    def size(self) -> int:
        """Number of indexed images."""
        return self._size

    def document_text(self, image_id: str) -> str:
        """Return the concatenated caption text stored for ``image_id``."""
        return self._documents[image_id]

    def _candidates(self, dataset: str | None, split: str | None) -> set[str] | None:
        """Return the image IDs allowed by the filters, or None when unfiltered."""
        if dataset is None and split is None:
            return None
        candidates: set[str] | None = None
        if dataset is not None:
            candidates = set(self._dataset_ids.get(dataset, ()))
        if split is not None:
            split_ids = set(self._split_ids.get(split, ()))
            candidates = split_ids if candidates is None else candidates & split_ids
        return candidates

    def rank(
        self,
        query: str,
        *,
        dataset: str | None = None,
        split: str | None = None,
        top_k: int = 20,
    ) -> list[RankedImage]:
        """Return at most ``top_k`` hits with a positive BM25 score.

        ``dataset``/``split`` are applied before truncation. Ties are ordered by
        ``image_id`` ascending so results are deterministic.
        """
        tokens = tokenize(query)
        if not tokens or self._size == 0 or top_k <= 0:
            return []
        candidates = self._candidates(dataset, split)
        if candidates is not None and not candidates:
            return []

        scores: dict[str, float] = defaultdict(float)
        for term in tokens:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = log(1 + (self._size - df + 0.5) / (df + 0.5))
            for image_id, term_frequency in postings.items():
                if candidates is not None and image_id not in candidates:
                    continue
                length_norm = self._lengths[image_id] / self._avgdl if self._avgdl else 1.0
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_norm)
                scores[image_id] += idf * term_frequency * (self._k1 + 1) / denominator

        ranked = sorted(
            ((image_id, score) for image_id, score in scores.items() if score > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return [RankedImage(image_id, score) for image_id, score in ranked[:top_k]]


class Bm25RetrievalEngine:
    """Adapt a :class:`Bm25Index` to the retrieval engine protocol."""

    def __init__(self, index: Bm25Index) -> None:
        self._index = index

    @classmethod
    def from_metadata(
        cls,
        path: Path,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> Bm25RetrievalEngine:
        """Build an engine from a JSONL metadata file."""
        index = Bm25Index.from_metadata(path, k1=k1, b=b, stopwords=stopwords)
        return cls(index)

    @property
    def index_version(self) -> str:
        return self._index.version

    def search_text(self, request: SearchRequest) -> list[RankedImage]:
        """Rank images for ``request.query``; only ``method="bm25"`` is served."""
        if request.method != "bm25":
            raise EngineNotReady(f"Method {request.method} is not loaded")
        return self._index.rank(
            request.query,
            dataset=request.dataset,
            split=request.split,
            top_k=request.top_k,
        )

    def search_similar(self, image_id: str, request: SearchRequest) -> list[RankedImage]:
        """BM25 has no similar-image search."""
        raise EngineNotReady("BM25 has no similar search")
