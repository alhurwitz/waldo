from __future__ import annotations

from pathlib import Path


def detect_scenes(video_path: Path) -> list[tuple[float, float]]:
    """Return list of (start_seconds, end_seconds) scene boundaries."""
    from scenedetect import detect, ContentDetector

    scene_list = detect(str(video_path), ContentDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
