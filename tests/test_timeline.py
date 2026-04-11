from waldo.timeline import build_intervals


def test_empty():
    assert build_intervals([], gap=2.0, min_len=0.5) == []


def test_drop_blip():
    samples = [(0.0, False), (1.0, True), (1.2, False), (5.0, False)]
    assert build_intervals(samples, gap=2.0, min_len=0.5) == []


def test_merge_consecutive():
    samples = [(0.0, True), (0.5, True), (1.0, True), (1.5, False)]
    out = build_intervals(samples, gap=2.0, min_len=0.5)
    assert out == [(0.0, 1.5)]


def test_bridge_gap():
    samples = [
        (0.0, True), (0.5, True), (1.0, False),
        (2.0, False), (2.5, True), (3.0, True), (3.5, False),
    ]
    out = build_intervals(samples, gap=2.0, min_len=0.5)
    assert len(out) == 1
    assert out[0][0] == 0.0
    assert out[0][1] >= 3.0


def test_no_bridge_when_gap_too_big():
    samples = [
        (0.0, True), (0.5, True), (1.0, False),
        (10.0, False), (10.5, True), (11.0, True), (11.5, False),
    ]
    out = build_intervals(samples, gap=2.0, min_len=0.5)
    assert len(out) == 2
