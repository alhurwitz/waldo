from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class FaceCluster:
    """Greedy online clustering of L2-normalised face embeddings.

    Each new embedding is compared (cosine similarity = dot product for
    normalised vectors) against existing cluster centroids. If the best
    match exceeds `threshold`, the embedding joins that cluster and the
    centroid is updated as a running mean (re-normalised). Otherwise a
    new cluster is created.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []  # L2-normed
        self._counts: list[int] = []

    @property
    def n_clusters(self) -> int:
        return len(self._centroids)

    def assign(self, embedding: np.ndarray) -> int:
        """Return the cluster index for this embedding (creating one if needed)."""
        if not self._centroids:
            self._centroids.append(embedding.copy())
            self._counts.append(1)
            return 0

        sims = np.array([float(np.dot(embedding, c)) for c in self._centroids])
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.threshold:
            # Update centroid as running mean, re-normalise.
            n = self._counts[best_idx]
            new_c = (self._centroids[best_idx] * n + embedding) / (n + 1)
            norm = np.linalg.norm(new_c)
            if norm > 0:
                new_c /= norm
            self._centroids[best_idx] = new_c
            self._counts[best_idx] = n + 1
            return best_idx

        # New cluster.
        self._centroids.append(embedding.copy())
        self._counts.append(1)
        return self.n_clusters - 1

    def centroids_as_refs(self, min_count: int = 2) -> dict[str, list[np.ndarray]]:
        """Export clusters as a refs dict (same shape as load_references output).

        Clusters with fewer than `min_count` faces are dropped (likely noise).
        """
        refs: dict[str, list[np.ndarray]] = {}
        for i, (c, n) in enumerate(zip(self._centroids, self._counts)):
            if n < min_count:
                log.debug("cluster person_%d has only %d face(s), skipping", i + 1, n)
                continue
            refs[f"person_{i + 1}"] = [c]
        return refs
