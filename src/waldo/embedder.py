from __future__ import annotations

import logging
import threading

import numpy as np
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)


class Face(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    embedding: np.ndarray  # shape (512,), L2-normalized


class Embedder:
    """Thread-safe InsightFace buffalo_l wrapper."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        except Exception:
            log.warning("CoreML provider unavailable, falling back to CPU.")
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        with self._lock:
            results = self.app.get(frame_bgr)
        faces: list[Face] = []
        for r in results:
            emb = r.normed_embedding.astype(np.float32)
            x1, y1, x2, y2 = (float(v) for v in r.bbox)
            faces.append(Face(bbox=(x1, y1, x2, y2), embedding=emb))
        return faces
