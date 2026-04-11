from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}-{m:02d}-{s:02d}"


def cut_clip(video: Path, start: float, end: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{video.stem}_{_hms(start)}_{_hms(end)}.mp4"
    out = out_dir / name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    log.info("cutting %s", out.name)
    subprocess.run(cmd, check=True)
    return out
