from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2


def estimate_sample_count(video_path: Path, fps: float) -> int:
    """Estimate how many samples `sample_frames` will yield for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(src_fps / fps)))
        return (total + step - 1) // step if total else 0
    finally:
        cap.release()


def sample_frames(video_path: Path, fps: float) -> Iterator[tuple[float, "cv2.Mat"]]:
    """Yield (timestamp_seconds, frame) at approximately `fps` samples/sec."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(src_fps / fps)))
        idx = 0
        while idx < total or total == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            yield t, frame
            idx += step
    finally:
        cap.release()
