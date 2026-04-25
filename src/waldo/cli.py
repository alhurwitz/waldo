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
        help="Directory containing videos and/or photos to scan. "
             "Searched recursively — all .mp4, .mov, .avi, .mkv, .webm, .m4v, "
             ".jpg, .jpeg, .png, .bmp, and .tiff files are included.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Output directory for clustered face crops. Each discovered person "
             "gets a subfolder (person_1/, person_2/, ...) containing cropped "
             "face images. You can rename these folders before running 'identify' "
             "or 'extract'. Will be created if it doesn't exist.",
    ),
    fps: float = typer.Option(
        2.0,
        help="How many frames per second to sample from videos. Higher values "
             "find more faces but take longer. 2.0 is a good balance; try 0.5 "
             "for long videos or 5.0 for short clips where you need precision.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Cosine similarity threshold for grouping faces into clusters "
             "(0.0-1.0). Lower values merge more aggressively (fewer clusters, "
             "risk mixing people). Higher values split more (more clusters, "
             "risk splitting one person). Default 0.5 works well for most cases; "
             "try 0.4 if clusters are too fragmented or 0.6 if people are getting merged.",
    ),
    workers: int = typer.Option(
        1,
        help="Number of files to process concurrently. Face detection (the "
             "slowest step) runs in parallel across files. Set to the number of "
             "CPU cores for maximum throughput. Default 1 processes files sequentially.",
    ),
) -> None:
    """Detect and cluster all faces in input media.

    Scans all videos and photos in --input, detects faces using InsightFace,
    clusters them by identity, and saves cropped face images to --refs.

    After scanning, use 'waldo identify' to name the clusters, then
    'waldo extract' to cut clips and sort photos by person.
    """
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
        help="Refs directory created by 'waldo scan', containing person_N/ "
             "subfolders with cropped face images. Each subfolder will be "
             "presented for you to name, skip, or delete.",
    ),
) -> None:
    """Interactively name face clusters.

    Opens an HTML page showing face crops for each cluster and prompts you
    in the terminal to assign names (e.g. 'grandma'), skip, or delete
    clusters. Renamed folders are used by 'waldo extract' to label output.
    """
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
        help="Directory containing videos and/or photos to process. "
             "Searched recursively for supported media formats.",
    ),
    refs: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Refs directory with named subfolders (one per person). Each "
             "subfolder should contain reference face images — either from "
             "'waldo scan' + 'waldo identify', or manually curated.",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Output directory for results. Video clips are cut with ffmpeg "
             "and saved as <person>/<clip>.mp4. Photos are copied into "
             "<person>/ subfolders. When multiple people appear together, "
             "output goes to a 'together/' subfolder. Created if it doesn't exist.",
    ),
    fps: float = typer.Option(
        2.0,
        help="Frames per second to sample from videos for face matching. "
             "Higher values improve detection accuracy at the cost of speed. "
             "Must match the fps used during 'scan' if relying on cached embeddings.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Cosine similarity threshold for matching detected faces to "
             "reference identities (0.0-1.0). Lower values are more permissive "
             "(more matches, more false positives). Higher values are stricter "
             "(fewer matches, may miss appearances). Default 0.5 balances "
             "precision and recall.",
    ),
    gap: float = typer.Option(
        2.0,
        help="Maximum gap in seconds between appearances to bridge into a "
             "single clip. If a person disappears for less than this duration, "
             "the two appearances are merged. Increase for scenes with "
             "intermittent occlusion; decrease for tighter cuts.",
    ),
    min_len: float = typer.Option(
        0.5,
        help="Minimum appearance duration in seconds. Brief detections shorter "
             "than this are discarded as noise (e.g. a face in a crowd for one "
             "frame). Increase to filter out fleeting appearances.",
    ),
    pad: float = typer.Option(
        15.0,
        help="Seconds of padding added before and after each appearance, "
             "snapped to scene boundaries (so padding never crosses a cut). "
             "Gives context around the person's appearance. Set to 0 for "
             "tight cuts with no extra context.",
    ),
    workers: int = typer.Option(
        1,
        help="Number of files to process concurrently. Video processing "
             "(decode + detect + cut) and photo processing run in parallel. "
             "Set to the number of CPU cores for maximum throughput.",
    ),
) -> None:
    """Match faces against references and cut video clips / copy photos.

    For each video, samples frames, matches detected faces against the
    reference identities in --refs, builds a timeline of appearances,
    and uses ffmpeg to cut clips. Photos are classified and copied to
    per-person subfolders. Requires ffmpeg on PATH.
    """
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
        help="Directory containing videos and/or photos to process. "
             "Searched recursively for supported media formats.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Refs directory for face clusters. Created during scan, used "
             "during extract. If it already exists and contains subfolders, "
             "the scan step is skipped and extraction runs directly.",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Output directory for video clips and sorted photos. "
             "Organized into subfolders by person name (or 'together/' "
             "for co-appearances). Created if it doesn't exist.",
    ),
    fps: float = typer.Option(
        2.0,
        help="Frames per second to sample from videos. Used for both the "
             "scan and extract phases. Higher values improve accuracy but "
             "increase processing time. 2.0 works well for most content.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Cosine similarity threshold (0.0-1.0) used for both clustering "
             "faces during scan and matching faces during extract. Lower "
             "values are more permissive, higher values are stricter.",
    ),
    gap: float = typer.Option(
        2.0,
        help="Maximum gap in seconds between appearances to bridge into a "
             "single clip. Appearances separated by less than this are merged.",
    ),
    min_len: float = typer.Option(
        0.5,
        help="Minimum appearance duration in seconds. Detections shorter "
             "than this are discarded as noise.",
    ),
    pad: float = typer.Option(
        15.0,
        help="Seconds of padding before/after each appearance, snapped to "
             "scene boundaries. Set to 0 for tight cuts.",
    ),
    workers: int = typer.Option(
        1,
        help="Number of files to process concurrently. Parallelizes face "
             "detection and video processing. Set to the number of CPU "
             "cores for maximum throughput.",
    ),
    auto: bool = typer.Option(
        False,
        help="Skip interactive identification and use auto-generated names "
             "(person_1, person_2, ...). Useful for batch processing or when "
             "you don't need to label people by name.",
    ),
) -> None:
    """Run the full pipeline: scan → identify → extract.

    One command to do everything: detect and cluster faces, interactively
    name them (unless --auto), then cut video clips and sort photos by
    person. If --refs already contains identified faces, skips straight
    to extraction.
    """
    _setup_logging()
    cfg = Config(
        input_dir=input, refs_dir=refs, output_dir=output,
        fps=fps, threshold=threshold, gap=gap, min_len=min_len, pad=pad, workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)

    from .pipeline import run as run_pipeline
    run_pipeline(cfg, auto=auto)


@app.command(name="compile")
def compile_cmd(
    output: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Output directory previously populated by 'waldo extract', "
             "containing per-person subfolders with .mp4 clips.",
    ),
    persons: str = typer.Option(
        None,
        help="Comma-separated person names to include. Each must match a "
             "subfolder under --output. Mutually exclusive with --all.",
    ),
    all: bool = typer.Option(
        False, "--all",
        help="Compile every person folder under --output (excluding "
             "together/ and _compilations/). Mutually exclusive with --persons.",
    ),
    transition: str = typer.Option(
        "crossfade",
        help="Transition style between clips: crossfade, fade, cut, or random. "
             "All transitions except 'cut' use a 0.5s overlap.",
    ),
) -> None:
    """Stitch per-person clips into a single compilation video.

    Reads <output>/<person>/*.mp4 for the requested persons, shuffles them,
    and renders <output>/_compilations/<persons>.mp4 with the chosen transition.
    Re-running overwrites the output. together/ clips are skipped (their
    person mapping is lost during extract).
    """
    _setup_logging()

    if (persons is None) == (not all):
        raise typer.BadParameter(
            "exactly one of --persons or --all must be provided "
            "(they are mutually exclusive)"
        )

    from . import assembler
    from .cutter import check_ffmpeg

    check_ffmpeg()

    if all:
        names = sorted(
            d.name for d in output.iterdir()
            if d.is_dir() and d.name != "together" and not d.name.startswith("_")
        )
        out_stem = "all"
    else:
        names = [n.strip() for n in persons.split(",") if n.strip()]
        out_stem = "_".join(names)

    if not names:
        typer.echo("No persons selected.", err=True)
        raise typer.Exit(code=1)

    clips = assembler.discover_clips(output, names)
    if not clips:
        typer.echo(f"No clips found for: {', '.join(names)}", err=True)
        raise typer.Exit(code=1)

    out_path = output / "_compilations" / f"{out_stem}.mp4"
    assembler.assemble(clips, out_path, transition=transition)
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
