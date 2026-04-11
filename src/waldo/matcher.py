from __future__ import annotations

import numpy as np


class RefIndex:
    """Pre-stacked reference embeddings for fast batch matching.

    All reference embeddings are stacked into a single matrix so that
    classification becomes a single BLAS matmul + argmax.
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

    @property
    def n_refs(self) -> int:
        return len(self._names)

    def classify(self, embedding: np.ndarray, threshold: float) -> str | None:
        """Return the person whose reference is closest, or None if below threshold."""
        if self.n_refs == 0:
            return None
        sims = self._matrix @ embedding
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= threshold:
            return self._names[best_idx]
        return None

    def classify_batch(
        self, embeddings: np.ndarray, threshold: float
    ) -> list[str | None]:
        """Classify multiple embeddings in one matmul."""
        n = len(embeddings)
        if n == 0 or self.n_refs == 0:
            return [None] * n
        # (total_refs, n) similarity matrix
        sim_matrix = self._matrix @ embeddings.T
        best_indices = np.argmax(sim_matrix, axis=0)
        best_sims = sim_matrix[best_indices, np.arange(n)]
        return [
            self._names[int(idx)] if float(sim) >= threshold else None
            for idx, sim in zip(best_indices, best_sims)
        ]
