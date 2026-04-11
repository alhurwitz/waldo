from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Raise SystemExit if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and try again."
        )


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}-{m:02d}-{s:05.2f}"


def _safe_path(p: Path) -> str:
    """Ensure a path string can't be misinterpreted as an ffmpeg flag."""
    s = str(p)
    if s.startswith("-"):
        return f"./{s}"
    return s


def cut_clip(video: Path, start: float, end: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{video.stem}_{_hms(start)}_{_hms(end)}.mp4"
    out = out_dir / name
    # Avoid clobbering if timestamps collide.
    n = 1
    while out.exists():
        out = out_dir / f"{video.stem}_{_hms(start)}_{_hms(end)}_{n}.mp4"
        n += 1
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", _safe_path(video),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        _safe_path(out),
    ]
    log.info("cutting %s", out.name)
    subprocess.run(cmd, check=True)
    return out
