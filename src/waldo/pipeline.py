from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .cache import CachedFace, CachedSample, VideoCache, cache_path, load_cache, save_cache
from .cluster import FaceCluster
from .config import Config
from .cutter import cut_clip
from .embedder import Embedder
from .matcher import classify_face
from .references import IMG_EXTS
from .sampler import estimate_sample_count, sample_frames
from .scenes import detect_scenes
from .snapper import pad_within_scenes_and_union
from .timeline import build_intervals

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MAX_DET_WIDTH = 960


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= _MAX_DET_WIDTH:
        return frame
    scale = _MAX_DET_WIDTH / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def _ensure_cache(
    video: Path,
    cfg: Config,
    embedder: Embedder,
    progress: Progress | None = None,
) -> VideoCache:
    path = cache_path(cfg.input_dir, video)
    mtime = video.stat().st_mtime
    cached = load_cache(path)
    if (
        cached
        and cached.video_mtime == mtime
        and cached.sampler_fps == cfg.fps
        and cached.scenes
    ):
        return cached

    log.info("scanning %s", video.name)

    with ThreadPoolExecutor(max_workers=1) as pool:
        scenes_future: Future[list[tuple[float, float]]] = pool.submit(detect_scenes, video)

        total = estimate_sample_count(video, cfg.fps)
        task_id = progress.add_task(f"  {video.name}", total=total) if progress else None
        samples: list[CachedSample] = []
        for t, frame in sample_frames(video, cfg.fps):
            small = _downscale(frame)
            faces = embedder.detect(small)
            samples.append(
                CachedSample(
                    t=t,
                    faces=[CachedFace(bbox=f.bbox, embedding=f.embedding) for f in faces],
                )
            )
            if progress is not None and task_id is not None:
                progress.advance(task_id)

        scenes = scenes_future.result()

    if progress is not None and task_id is not None:
        progress.remove_task(task_id)

    cache = VideoCache(video_mtime=mtime, sampler_fps=cfg.fps, scenes=scenes, samples=samples)
    save_cache(path, cache)
    return cache


def _process_video(
    video: Path,
    cfg: Config,
    refs: dict[str, list[np.ndarray]],
    embedder: Embedder,
    progress: Progress | None = None,
) -> list[Path]:
    cache = _ensure_cache(video, cfg, embedder, progress=progress)

    per_person: dict[str, list[tuple[float, bool]]] = {p: [] for p in refs}
    for sample in cache.samples:
        present_now = {p: False for p in refs}
        for face in sample.faces:
            who = classify_face(face.embedding, refs, cfg.threshold)
            if who:
                present_now[who] = True
        for p, present in present_now.items():
            per_person[p].append((sample.t, present))

    intervals = {
        p: build_intervals(samples, cfg.gap, cfg.min_len)
        for p, samples in per_person.items()
    }
    clip_scenes = pad_within_scenes_and_union(intervals, cache.scenes, cfg.pad)

    out_files: list[Path] = []
    for cs in clip_scenes:
        if len(cs.persons) >= 2:
            sub = "together"
        else:
            sub = next(iter(cs.persons))
        out_dir = cfg.output_dir / sub
        try:
            out_files.append(cut_clip(video, cs.start, cs.end, out_dir))
        except Exception as e:
            log.warning("ffmpeg failed for %s [%s-%s]: %s", video.name, cs.start, cs.end, e)
    return out_files


# ---------------------------------------------------------------------------
# Photo processing
# ---------------------------------------------------------------------------

def _process_photo(
    img_path: Path,
    refs: dict[str, list[np.ndarray]],
    output_dir: Path,
    embedder: Embedder,
    threshold: float,
) -> str | None:
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("could not read %s", img_path)
        return None
    faces = embedder.detect(img)
    if not faces:
        return None

    found: set[str] = set()
    for face in faces:
        who = classify_face(face.embedding, refs, threshold)
        if who:
            found.add(who)
    if not found:
        return None

    tag = "together" if len(found) >= 2 else next(iter(found))
    dest_dir = output_dir / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / img_path.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{img_path.stem}_{n}{img_path.suffix}"
        n += 1
    shutil.copy2(img_path, dest)
    return tag


# ---------------------------------------------------------------------------
# Auto-cluster mode: discover all faces, cluster, then sort
# ---------------------------------------------------------------------------

def _auto_cluster(cfg: Config, embedder: Embedder) -> dict[str, list[np.ndarray]]:
    """First pass: scan all media, collect face embeddings, cluster them,
    and return a refs dict mapping person_N → [centroid]."""
    all_files = sorted(p for p in cfg.input_dir.rglob("*") if p.is_file())
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    photos = [f for f in all_files if f.suffix.lower() in IMG_EXTS]

    cluster = FaceCluster(threshold=cfg.threshold)
    log.info("auto-cluster: collecting face embeddings...")

    with _make_progress() as progress:
        # Videos — use cached embeddings or scan.
        if videos:
            task = progress.add_task("clustering videos", total=len(videos))
            for video in videos:
                try:
                    cache = _ensure_cache(video, cfg, embedder, progress=progress)
                    for sample in cache.samples:
                        for face in sample.faces:
                            cluster.assign(face.embedding)
                except Exception as e:
                    log.warning("skipping %s: %s", video.name, e)
                progress.advance(task)

        # Photos.
        if photos:
            task = progress.add_task("clustering photos", total=len(photos))
            for img_path in photos:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        for face in embedder.detect(img):
                            cluster.assign(face.embedding)
                except Exception as e:
                    log.warning("skipping %s: %s", img_path.name, e)
                progress.advance(task)

    refs = cluster.centroids_as_refs(min_count=2)
    log.info("auto-cluster: found %d person(s) across %d cluster(s)",
             len(refs), cluster.n_clusters)
    return refs


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
    )


def run(cfg: Config) -> None:
    embedder = Embedder()

    # Decide ref source: explicit refs dir or auto-cluster.
    if cfg.refs_dir is not None:
        from .references import load_references
        refs_arrays = load_references(cfg.refs_dir, embedder)
    else:
        log.info("no --refs provided, auto-clustering faces by person")
        refs_arrays = _auto_cluster(cfg, embedder)
        if not refs_arrays:
            log.warning("no faces found to cluster")
            return

    # Discover all media files recursively.
    all_files = sorted(p for p in cfg.input_dir.rglob("*") if p.is_file())
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    photos = [f for f in all_files if f.suffix.lower() in IMG_EXTS]

    if not videos and not photos:
        log.warning("no videos or images found in %s", cfg.input_dir)
        return

    with _make_progress() as progress:
        # --- Videos ---
        if videos:
            log.info("found %d video(s)", len(videos))
            videos_task = progress.add_task("videos", total=len(videos))

            def _do_video(v: Path) -> None:
                try:
                    _process_video(v, cfg, refs_arrays, embedder, progress=progress)
                except Exception as e:
                    log.exception("failed processing %s: %s", v, e)
                progress.advance(videos_task)

            if cfg.workers <= 1:
                for v in videos:
                    _do_video(v)
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    list(pool.map(_do_video, videos))

        # --- Photos ---
        if photos:
            log.info("found %d photo(s)", len(photos))
            photos_task = progress.add_task("photos", total=len(photos))

            def _do_photo(p: Path) -> None:
                try:
                    _process_photo(p, refs_arrays, cfg.output_dir, embedder, cfg.threshold)
                except Exception as e:
                    log.exception("failed processing %s: %s", p, e)
                progress.advance(photos_task)

            if cfg.workers <= 1:
                for p in photos:
                    _do_photo(p)
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    list(pool.map(_do_photo, photos))
