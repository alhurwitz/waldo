from __future__ import annotations

import numpy as np


def classify_face(
    embedding: np.ndarray,
    refs: dict[str, list[np.ndarray]],
    threshold: float,
) -> str | None:
    """Return the person name whose nearest reference is closest (by cosine
    similarity) and exceeds threshold, else None.

    Embeddings are assumed L2-normalized → cosine sim = dot product.
    """
    best_name: str | None = None
    best_sim = threshold
    for name, ref_list in refs.items():
        sims = [float(np.dot(embedding, r)) for r in ref_list]
        top = max(sims) if sims else -1.0
        if top >= best_sim:
            best_sim = top
            best_name = name
    return best_name
