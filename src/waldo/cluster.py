from __future__ import annotations

import logging

import faiss
import numpy as np

log = logging.getLogger(__name__)

_INITIAL_CAPACITY = 64


class FaceCluster:
    """Greedy online clustering of L2-normalised face embeddings.

    Centroids are stored in a pre-allocated matrix for BLAS-optimised
    similarity computation.  A FAISS ``IndexFlatIP`` accelerates
    nearest-centroid lookup.
    """

    def __init__(self, threshold: float = 0.55, *, dim: int = 512) -> None:
        self.threshold = threshold
        self._dim = dim
        self._capacity = _INITIAL_CAPACITY
        self._centroid_matrix = np.empty((self._capacity, dim), dtype=np.float32)
        self._counts = np.empty(self._capacity, dtype=np.int64)
        self._size = 0
        self._index = faiss.IndexFlatIP(dim)

    @property
    def n_clusters(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_mat = np.empty((new_cap, self._dim), dtype=np.float32)
        new_mat[: self._size] = self._centroid_matrix[: self._size]
        self._centroid_matrix = new_mat
        new_counts = np.empty(new_cap, dtype=np.int64)
        new_counts[: self._size] = self._counts[: self._size]
        self._counts = new_counts
        self._capacity = new_cap

    def _rebuild_index(self) -> None:
        """Reset and re-add all current centroids to the FAISS index."""
        self._index.reset()
        if self._size > 0:
            self._index.add(np.ascontiguousarray(self._centroid_matrix[: self._size]))

    def assign(self, embedding: np.ndarray) -> int:
        """Return the cluster index for this embedding (creating one if needed)."""
        if self._size == 0:
            # Infer dimension from the first embedding and reallocate if needed.
            if embedding.shape[0] != self._dim:
                self._dim = embedding.shape[0]
                self._centroid_matrix = np.empty(
                    (self._capacity, self._dim), dtype=np.float32
                )
                self._index = faiss.IndexFlatIP(self._dim)
            self._centroid_matrix[0] = embedding
            self._counts[0] = 1
            self._size = 1
            self._rebuild_index()
            return 0

        # Use FAISS to find nearest centroid
        _dists, indices = self._index.search(embedding.reshape(1, -1), 1)
        best_idx = int(indices[0][0])
        # Compute actual similarity via dot product for threshold check
        best_sim = float(self._centroid_matrix[best_idx] @ embedding)

        if best_sim >= self.threshold:
            n = int(self._counts[best_idx])
            new_c = (self._centroid_matrix[best_idx] * n + embedding) / (n + 1)
            norm = np.linalg.norm(new_c)
            if norm > 0:
                new_c /= norm
            self._centroid_matrix[best_idx] = new_c
            self._counts[best_idx] = n + 1
            self._rebuild_index()
            return best_idx

        if self._size >= self._capacity:
            self._grow()
        self._centroid_matrix[self._size] = embedding
        self._counts[self._size] = 1
        self._size += 1
        self._rebuild_index()
        return self._size - 1

    def assign_batch(self, embeddings: np.ndarray) -> list[int]:
        """Assign a batch of embeddings. Order-dependent — processes sequentially
        but each similarity computation is BLAS-optimised."""
        if len(embeddings) == 0:
            return []
        return [self.assign(e) for e in embeddings]

    def centroids_as_refs(self, min_count: int = 2) -> dict[str, list[np.ndarray]]:
        """Export clusters as a refs dict (same shape as load_references output).

        Clusters with fewer than ``min_count`` faces are dropped (likely noise).
        """
        refs: dict[str, list[np.ndarray]] = {}
        for i in range(self._size):
            n = int(self._counts[i])
            if n < min_count:
                log.debug("cluster person_%d has only %d face(s), skipping", i + 1, n)
                continue
            refs[f"person_{i + 1}"] = [self._centroid_matrix[i].copy()]
        return refs
