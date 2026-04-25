"""Stitch per-person clips into a compilation with auto transitions."""
from __future__ import annotations

import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

_RANDOM_XFADE_KINDS = (
    "fade", "fadeblack", "slideleft", "slideright",
    "wipeleft", "wiperight", "dissolve",
)


def _build_filter_complex(
    durations: list[float],
    transition: str,
    transition_duration: float,
    rng: random.Random | None = None,
) -> str:
    """Return the ffmpeg filter_complex string for the given clips.

    Empty string when there are fewer than 2 clips (caller takes a shortcut).
    Output streams are always labelled [v] and [a].
    """
    n = len(durations)
    if n < 2:
        return ""

    if transition == "cut":
        pads = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        return f"{pads}concat=n={n}:v=1:a=1[v][a]"

    if transition == "crossfade":
        kinds = ["fade"] * (n - 1)
    elif transition == "fade":
        kinds = ["fadeblack"] * (n - 1)
    elif transition == "random":
        rng = rng or random.Random()
        kinds = [rng.choice(_RANDOM_XFADE_KINDS) for _ in range(n - 1)]
    else:
        raise ValueError(f"unknown transition: {transition!r}")

    d = transition_duration
    video_lines: list[str] = []
    audio_lines: list[str] = []
    cum = 0.0
    for k in range(n - 1):
        cum += durations[k]
        offset = cum - (k + 1) * d
        v_in = "[0:v]" if k == 0 else f"[v0{k}]"
        a_in = "[0:a]" if k == 0 else f"[a0{k}]"
        v_out = "[v]" if k == n - 2 else f"[v0{k+1}]"
        a_out = "[a]" if k == n - 2 else f"[a0{k+1}]"
        video_lines.append(
            f"{v_in}[{k+1}:v]xfade=transition={kinds[k]}:"
            f"duration={d:.3f}:offset={offset:.3f}{v_out}"
        )
        audio_lines.append(f"{a_in}[{k+1}:a]acrossfade=d={d:.3f}{a_out}")

    return ";".join(video_lines + audio_lines)


_RESERVED_NAMES = frozenset({"together"})


def discover_clips(output_dir: Path, persons: list[str]) -> list[Path]:
    """Return all .mp4 clips under output_dir/<person>/ for each person.

    Skips reserved folders (together, anything starting with `_`).
    Warns and skips persons whose folder is missing or empty.
    Result is sorted alphabetically by full path; the assembler shuffles.
    """
    clips: list[Path] = []
    for name in persons:
        if name in _RESERVED_NAMES or name.startswith("_"):
            continue
        folder = output_dir / name
        if not folder.is_dir():
            log.warning("no folder for person %r at %s — skipping", name, folder)
            continue
        person_clips = sorted(folder.glob("*.mp4"))
        if not person_clips:
            log.warning("no .mp4 clips in %s — skipping", folder)
            continue
        clips.extend(person_clips)
    return sorted(clips)
