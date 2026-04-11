from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def crop_face(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_fraction: float = 0.2,
    target_size: int | None = None,
) -> np.ndarray:
    """Extract a face crop from a frame with padding.

    Args:
        frame: BGR image (H, W, 3).
        bbox: (x1, y1, x2, y2) face bounding box.
        pad_fraction: Fraction of bbox size to pad on each side.
        target_size: If set, resize the crop to (target_size, target_size).

    Returns:
        BGR image of the cropped face region.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    pad_x = bw * pad_fraction
    pad_y = bh * pad_fraction

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    crop = frame[cy1:cy2, cx1:cx2]

    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return crop


def save_crops(
    refs_dir: Path,
    cluster_sources: list[list[tuple[str, tuple[float, float, float, float]]]],
    cluster_counts: list[int],
    frames: dict[str, np.ndarray],
    max_per_cluster: int = 5,
    min_count: int = 2,
) -> list[str]:
    """Save face crops to refs_dir/person_N/ folders.

    Args:
        refs_dir: Root directory for reference crops.
        cluster_sources: Per-cluster list of (source_id, bbox) from FaceCluster.
        cluster_counts: Total face count per cluster.
        frames: Dict mapping source_id to BGR frame.
        max_per_cluster: Maximum crops to save per person.
        min_count: Minimum faces in a cluster to save it.

    Returns:
        List of person folder names created (e.g. ["person_1", "person_3"]).
    """
    created: list[str] = []
    for i, (sources, count) in enumerate(zip(cluster_sources, cluster_counts)):
        if count < min_count:
            log.debug("cluster person_%d has only %d face(s), skipping", i + 1, count)
            continue

        name = f"person_{i + 1}"
        person_dir = refs_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        if len(sources) > max_per_cluster:
            step = len(sources) / max_per_cluster
            indices = [int(step * j) for j in range(max_per_cluster)]
            selected = [sources[idx] for idx in indices]
        else:
            selected = sources

        for j, (source_id, bbox) in enumerate(selected):
            frame = frames.get(source_id)
            if frame is None:
                continue
            crop = crop_face(frame, bbox)
            out_path = person_dir / f"face_{j + 1:03d}.jpg"
            cv2.imwrite(str(out_path), crop)

        created.append(name)
        log.info("saved %d crop(s) for %s", len(selected), name)

    return created
