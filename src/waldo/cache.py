from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class CachedFace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: tuple[float, float, float, float]
    embedding: np.ndarray

    @field_serializer("embedding")
    def _ser_emb(self, v: np.ndarray) -> list[float]:
        return v.astype(float).tolist()

    @field_validator("embedding", mode="before")
    @classmethod
    def _val_emb(cls, v):
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v, dtype=np.float32)


class CachedSample(BaseModel):
    t: float
    faces: list[CachedFace]


class VideoCache(BaseModel):
    video_mtime: float
    sampler_fps: float
    scenes: list[tuple[float, float]]
    samples: list[CachedSample]


def cache_path(videos_dir: Path, video: Path) -> Path:
    return videos_dir / ".waldo-cache" / f"{video.stem}.json"


def load_cache(path: Path) -> VideoCache | None:
    if not path.exists():
        return None
    try:
        return VideoCache.model_validate_json(path.read_text())
    except Exception:
        return None


def save_cache(path: Path, cache: VideoCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(cache.model_dump_json())
    tmp.replace(path)
