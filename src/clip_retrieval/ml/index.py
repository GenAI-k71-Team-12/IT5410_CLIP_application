"""Exact inner-product FAISS index over L2-normalized rows; FAISS IDs are row numbers.

Indexes go through bytes because faiss.write_index cannot open non-ASCII paths on Windows.
"""

from pathlib import Path

import faiss
import numpy as np


def build(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def save(index: faiss.Index, path: Path) -> None:
    path.write_bytes(faiss.serialize_index(index).tobytes())


def load(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))


def search(
    index: faiss.Index, query: np.ndarray, k: int, rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, rows) of the top-k matches for one (1, D) query, best first.

    `rows` restricts the candidates before ranking, so filters never starve top-k.
    """
    params = None
    if rows is not None:
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(rows.astype(np.int64)))
    scores, found = index.search(np.ascontiguousarray(query, dtype=np.float32), k, params=params)
    keep = found[0] >= 0
    return scores[0][keep], found[0][keep]
