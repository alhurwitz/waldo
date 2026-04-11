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
    return videos_dir / ".waldo-cache" / f"{video.stem}.npz"


def load_cache(path: Path) -> VideoCache | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        video_mtime = float(data["video_mtime"])
        sampler_fps = float(data["sampler_fps"])
        scenes_arr = data["scenes"]
        scenes = [(float(r[0]), float(r[1])) for r in scenes_arr]
        timestamps = data["timestamps"]
        face_counts = data["face_counts"]
        embeddings = data["embeddings"]
        bboxes = data["bboxes"]

        samples: list[CachedSample] = []
        emb_idx = 0
        for i, t in enumerate(timestamps):
            n = int(face_counts[i])
            faces = [
                CachedFace(
                    bbox=tuple(bboxes[emb_idx + j]),
                    embedding=embeddings[emb_idx + j],
                )
                for j in range(n)
            ]
            emb_idx += n
            samples.append(CachedSample(t=float(t), faces=faces))

        return VideoCache(
            video_mtime=video_mtime,
            sampler_fps=sampler_fps,
            scenes=scenes,
            samples=samples,
        )
    except Exception:
        return None


def save_cache(path: Path, cache: VideoCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.array([s.t for s in cache.samples], dtype=np.float64)
    face_counts = np.array([len(s.faces) for s in cache.samples], dtype=np.int32)

    all_embeddings = [f.embedding for s in cache.samples for f in s.faces]
    all_bboxes = [f.bbox for s in cache.samples for f in s.faces]

    embeddings = np.array(all_embeddings, dtype=np.float32) if all_embeddings else np.empty((0, 512), dtype=np.float32)
    bboxes = np.array(all_bboxes, dtype=np.float32) if all_bboxes else np.empty((0, 4), dtype=np.float32)
    scenes = np.array(cache.scenes, dtype=np.float64) if cache.scenes else np.empty((0, 2), dtype=np.float64)

    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez_compressed(
        tmp,
        video_mtime=np.float64(cache.video_mtime),
        sampler_fps=np.float64(cache.sampler_fps),
        scenes=scenes,
        timestamps=timestamps,
        face_counts=face_counts,
        embeddings=embeddings,
        bboxes=bboxes,
    )
    tmp.replace(path)
