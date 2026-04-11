from __future__ import annotations

import faiss
import numpy as np


class RefIndex:
    """Pre-stacked reference embeddings for fast batch matching.

    All reference embeddings are stacked into a single matrix and indexed
    with a FAISS ``IndexFlatIP`` for fast inner-product search.
    """

    def __init__(self, refs: dict[str, list[np.ndarray]]) -> None:
        names: list[str] = []
        rows: list[np.ndarray] = []
        for name, embs in refs.items():
            for e in embs:
                names.append(name)
                rows.append(e)
        self._names = names
        self._matrix = np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)

        if rows:
            dim = self._matrix.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(np.ascontiguousarray(self._matrix))
        else:
            self._index = None

    @property
    def n_refs(self) -> int:
        return len(self._names)

    def classify(self, embedding: np.ndarray, threshold: float) -> str | None:
        """Return the person whose reference is closest, or None if below threshold."""
        if self.n_refs == 0 or self._index is None:
            return None
        dists, indices = self._index.search(embedding.reshape(1, -1), 1)
        best_idx = int(indices[0][0])
        best_sim = float(dists[0][0])
        if best_sim >= threshold:
            return self._names[best_idx]
        return None

    def classify_batch(
        self, embeddings: np.ndarray, threshold: float
    ) -> list[str | None]:
        """Classify multiple embeddings in one FAISS search."""
        n = len(embeddings)
        if n == 0 or self.n_refs == 0 or self._index is None:
            return [None] * n
        dists, indices = self._index.search(np.ascontiguousarray(embeddings), 1)
        return [
            self._names[int(indices[i][0])] if float(dists[i][0]) >= threshold else None
            for i in range(n)
        ]
