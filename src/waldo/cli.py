from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

import typer

from .config import Config

app = typer.Typer(
    add_completion=False,
    help="Where's Waldo, but for real faces. Find people in videos and photos using face recognition.",
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def scan(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to scan for faces.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to save clustered face crops (e.g. refs/).",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Clustering similarity threshold (0-1)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
) -> None:
    """Detect and cluster all faces in input media. Saves face crops to --refs folder."""
    _setup_logging()

    if refs.is_dir() and any(refs.iterdir()):
        overwrite = typer.confirm(f"Refs folder {refs} already exists. Overwrite?")
        if not overwrite:
            raise typer.Abort()
        import shutil
        shutil.rmtree(refs)

    cfg = Config(input_dir=input, refs_dir=refs, fps=fps, threshold=threshold, workers=workers)

    from .pipeline import scan_faces
    created = scan_faces(cfg)

    if created:
        typer.echo(f"\nSaved {len(created)} cluster(s) to {refs}/")
        typer.echo(f"Next: rename folders manually, or run: waldo identify --refs {refs}")
    else:
        typer.echo("No faces found.")


@app.command()
def identify(
    refs: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Refs folder with face crop subfolders to identify.",
    ),
) -> None:
    """Interactively name face clusters. Opens an HTML page and prompts in the terminal."""
    _setup_logging()

    from .identify import generate_html, run_identify_prompts

    html_path = generate_html(refs, refs / ".identify.html")
    webbrowser.open(html_path.as_uri())

    stats = run_identify_prompts(refs)
    typer.echo(f"\nDone! {stats['identified']} identified, {stats['skipped']} skipped, {stats['deleted']} deleted.")
    typer.echo(f"Next: waldo extract --input <media> --refs {refs} --output <output>")


@app.command()
def extract(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to process.",
    ),
    refs: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Refs folder with named subfolders (one per person).",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to write clips and photos.",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Match similarity threshold (0-1)."),
    gap: float = typer.Option(2.0, help="Bridge gaps shorter than this (seconds)."),
    min_len: float = typer.Option(0.5, help="Minimum appearance duration (seconds)."),
    pad: float = typer.Option(15.0, help="Padding around appearances (seconds)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
) -> None:
    """Match faces against references and cut clips / copy photos."""
    _setup_logging()
    cfg = Config(
        input_dir=input, refs_dir=refs, output_dir=output,
        fps=fps, threshold=threshold, gap=gap, min_len=min_len, pad=pad, workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)

    from .pipeline import extract_clips
    extract_clips(cfg)


@app.command()
def run(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to process.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Refs folder — will be created by scan, used by extract.",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to write clips and photos.",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Similarity threshold (0-1)."),
    gap: float = typer.Option(2.0, help="Bridge gaps shorter than this (seconds)."),
    min_len: float = typer.Option(0.5, help="Minimum appearance duration (seconds)."),
    pad: float = typer.Option(15.0, help="Padding around appearances (seconds)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
    auto: bool = typer.Option(False, help="Skip interactive identification — use generic person_N names."),
) -> None:
    """Run the full pipeline: scan → identify → extract."""
    _setup_logging()
    cfg = Config(
        input_dir=input, refs_dir=refs, output_dir=output,
        fps=fps, threshold=threshold, gap=gap, min_len=min_len, pad=pad, workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)

    from .pipeline import run as run_pipeline
    run_pipeline(cfg, auto=auto)


if __name__ == "__main__":
    app()
