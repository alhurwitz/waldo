from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from .config import Config
from .pipeline import run as run_pipeline

app = typer.Typer(add_completion=False, help="Where's Waldo, but for real faces. Find people in videos and photos using face recognition.")


@app.command()
def scan(
    input: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Folder to scan (recursively). Videos (.mp4/.mov/.avi/.mkv/.webm/.m4v) get clipped; images (.jpg/.png/.bmp/.webp) get copied.",
    ),
    refs: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Folder of reference photos. One subfolder per person (e.g. refs/grandma, refs/grandpa). If omitted, faces are auto-clustered and sorted into person_1/, person_2/, etc.",
    ),
    output: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
        help="Where to write results. Organized into <output>/<person>/ and <output>/together/.",
    ),
    fps: float = typer.Option(
        2.0,
        help="Frames per second to sample from videos for face detection. Lower = faster but may miss brief appearances. Higher = slower but more accurate.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Cosine-similarity cutoff (0-1). With --refs: min similarity to count as a match. Without --refs: min similarity to group two faces as the same person.",
    ),
    gap: float = typer.Option(
        2.0,
        help="Bridge gaps shorter than this many seconds between detections of the same person — treats them as one continuous appearance.",
    ),
    min_len: float = typer.Option(
        0.5,
        help="Minimum appearance duration in seconds. A person must be on screen at least this long for a clip to be produced. Raise to suppress brief appearances.",
    ),
    pad: float = typer.Option(
        15.0,
        help="Seconds of padding before and after each appearance in a clip, clamped to scene boundaries. 15 = ~30s window around a brief appearance.",
    ),
    workers: int = typer.Option(
        1,
        help="Number of files to process concurrently. Set to 2-4 for a speedup on large folders.",
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = Config(
        input_dir=input,
        refs_dir=refs,
        output_dir=output,
        fps=fps,
        threshold=threshold,
        gap=gap,
        min_len=min_len,
        pad=pad,
        workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)
    run_pipeline(cfg)


if __name__ == "__main__":
    app()
