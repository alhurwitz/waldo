"""Integration tests for assembler — gated on ffmpeg/ffprobe availability."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    ffmpeg is None or ffprobe is None,
    reason="requires ffmpeg and ffprobe on PATH",
)


def _make_clip(path: Path, duration: float, color: str = "red") -> Path:
    """Synthesize a short test clip with color video + sine audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={color}:s=160x90:d={duration}:r=24",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ])
    return float(out.strip())


# ---------- _probe_duration ----------

def test_probe_duration_returns_clip_length(tmp_path):
    from waldo.assembler import _probe_duration
    clip = _make_clip(tmp_path / "a.mp4", duration=2.0)
    d = _probe_duration(clip)
    assert d == pytest.approx(2.0, abs=0.2)


def test_probe_duration_raises_on_missing_file(tmp_path):
    from waldo.assembler import _probe_duration
    with pytest.raises(subprocess.CalledProcessError):
        _probe_duration(tmp_path / "nope.mp4")


# ---------- assemble ----------

def test_assemble_cut_concatenates_durations(tmp_path):
    from waldo.assembler import assemble
    a = _make_clip(tmp_path / "a.mp4", 2.0, "red")
    b = _make_clip(tmp_path / "b.mp4", 2.0, "green")
    c = _make_clip(tmp_path / "c.mp4", 2.0, "blue")
    out = tmp_path / "compilation.mp4"
    assemble([a, b, c], out, transition="cut", transition_duration=0.5)
    assert out.exists() and out.stat().st_size > 0
    assert _ffprobe_duration(out) == pytest.approx(6.0, abs=0.5)


def test_assemble_crossfade_subtracts_overlaps(tmp_path):
    from waldo.assembler import assemble
    a = _make_clip(tmp_path / "a.mp4", 2.0, "red")
    b = _make_clip(tmp_path / "b.mp4", 2.0, "green")
    c = _make_clip(tmp_path / "c.mp4", 2.0, "blue")
    out = tmp_path / "x.mp4"
    assemble([a, b, c], out, transition="crossfade", transition_duration=0.5)
    # Expected: 6.0 - 2 transitions * 0.5 = 5.0
    assert _ffprobe_duration(out) == pytest.approx(5.0, abs=0.5)


def test_assemble_fade_renders(tmp_path):
    from waldo.assembler import assemble
    a = _make_clip(tmp_path / "a.mp4", 2.0, "red")
    b = _make_clip(tmp_path / "b.mp4", 2.0, "green")
    out = tmp_path / "f.mp4"
    assemble([a, b], out, transition="fade", transition_duration=0.5)
    assert _ffprobe_duration(out) == pytest.approx(3.5, abs=0.5)


def test_assemble_random_renders(tmp_path):
    from waldo.assembler import assemble
    a = _make_clip(tmp_path / "a.mp4", 2.0, "red")
    b = _make_clip(tmp_path / "b.mp4", 2.0, "green")
    c = _make_clip(tmp_path / "c.mp4", 2.0, "blue")
    out = tmp_path / "r.mp4"
    assemble([a, b, c], out, transition="random", transition_duration=0.5)
    assert _ffprobe_duration(out) == pytest.approx(5.0, abs=0.5)


def test_assemble_single_clip_copies(tmp_path):
    from waldo.assembler import assemble
    a = _make_clip(tmp_path / "a.mp4", 2.0, "red")
    out = tmp_path / "single.mp4"
    assemble([a], out, transition="crossfade", transition_duration=0.5)
    assert out.exists()
    assert _ffprobe_duration(out) == pytest.approx(2.0, abs=0.2)


def test_assemble_empty_raises(tmp_path):
    from waldo.assembler import assemble
    with pytest.raises(ValueError):
        assemble([], tmp_path / "empty.mp4", transition="cut", transition_duration=0.5)
