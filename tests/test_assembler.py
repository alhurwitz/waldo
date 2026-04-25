"""Unit tests for waldo.assembler — pure helpers, no ffmpeg."""
from __future__ import annotations

import random

import pytest

from waldo.assembler import _build_filter_complex, discover_clips


# ---------- _build_filter_complex ----------

def test_filter_complex_empty_for_zero_or_one_clip():
    assert _build_filter_complex([], "crossfade", 0.5) == ""
    assert _build_filter_complex([10.0], "crossfade", 0.5) == ""


def test_filter_complex_cut_two_clips():
    g = _build_filter_complex([10.0, 5.0], "cut", 0.5)
    assert g == "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"


def test_filter_complex_cut_four_clips():
    g = _build_filter_complex([3.0, 4.0, 5.0, 6.0], "cut", 0.5)
    assert g == "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a]concat=n=4:v=1:a=1[v][a]"


def test_filter_complex_crossfade_two_clips():
    g = _build_filter_complex([10.0, 5.0], "crossfade", 0.5)
    expected = (
        "[0:v][1:v]xfade=transition=fade:duration=0.500:offset=9.500[v];"
        "[0:a][1:a]acrossfade=d=0.500[a]"
    )
    assert g == expected


def test_filter_complex_crossfade_three_clips_chained_offsets():
    """3 clips → 2 xfades, offsets are cumulative minus k*D."""
    g = _build_filter_complex([10.0, 5.0, 8.0], "crossfade", 0.5)
    # offset_1 = 10 - 0.5 = 9.5  (between v0 and v1)
    # offset_2 = 10 + 5 - 1.0 = 14.0  (between [v01] and v2)
    expected = (
        "[0:v][1:v]xfade=transition=fade:duration=0.500:offset=9.500[v01];"
        "[v01][2:v]xfade=transition=fade:duration=0.500:offset=14.000[v];"
        "[0:a][1:a]acrossfade=d=0.500[a01];"
        "[a01][2:a]acrossfade=d=0.500[a]"
    )
    assert g == expected


def test_filter_complex_fade_uses_fadeblack():
    g = _build_filter_complex([10.0, 5.0], "fade", 0.5)
    assert "transition=fadeblack" in g
    assert "transition=fade:" not in g  # not the plain fade


def test_filter_complex_random_picks_from_known_set():
    """Random mode picks an xfade transition from a fixed allow-list per cut."""
    rng = random.Random(0)
    g = _build_filter_complex([10.0, 5.0, 8.0], "random", 0.5, rng=rng)
    # Two xfade segments → two transition= tokens
    assert g.count("xfade=transition=") == 2
    allowed = {"fade", "fadeblack", "slideleft", "slideright",
               "wipeleft", "wiperight", "dissolve"}
    for line in g.split(";"):
        if "xfade=transition=" in line:
            kind = line.split("xfade=transition=")[1].split(":")[0]
            assert kind in allowed


def test_filter_complex_unknown_transition_raises():
    with pytest.raises(ValueError):
        _build_filter_complex([10.0, 5.0], "starwipe", 0.5)
