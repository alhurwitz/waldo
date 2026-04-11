from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_EXTS
from .embedder import Embedder

log = logging.getLogger(__name__)


def load_references(refs_dir: Path, embedder: Embedder) -> dict[str, list[np.ndarray]]:
    """For each subdirectory of refs_dir (one per person), load and embed photos.

    A photo is used iff it contains exactly one detected face. Raises SystemExit
    if any person ends up with zero usable embeddings.
    """
    if not refs_dir.is_dir():
        raise SystemExit(f"refs dir not found: {refs_dir}")

    persons: dict[str, list[np.ndarray]] = {}
    for person_dir in sorted(p for p in refs_dir.iterdir() if p.is_dir()):
        embeddings: list[np.ndarray] = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("could not read %s", img_path)
                continue
            faces = embedder.detect(img)
            if len(faces) == 0:
                log.warning("no face in %s — skipping", img_path)
                continue
            if len(faces) > 1:
                log.warning("multiple faces in %s — ambiguous, skipping", img_path)
                continue
            embeddings.append(faces[0].embedding)
        if not embeddings:
            raise SystemExit(f"no usable reference photos for person '{person_dir.name}'")
        persons[person_dir.name] = embeddings
        log.info("loaded %d ref embeddings for %s", len(embeddings), person_dir.name)
    if not persons:
        raise SystemExit(f"no person subdirectories found in {refs_dir}")
    return persons
