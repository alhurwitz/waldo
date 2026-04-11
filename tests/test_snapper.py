from waldo.snapper import pad_within_scenes_and_union


def test_pad_within_scene():
    scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    intervals = {"grandma": [(12.0, 13.0)]}
    out = pad_within_scenes_and_union(intervals, scenes, pad=15.0)
    assert len(out) == 1
    # Padded to [-2, 28] but clamped to scene [10, 20].
    assert out[0].start == 10.0
    assert out[0].end == 20.0
    assert out[0].persons == frozenset({"grandma"})


def test_no_scenes_pads_without_clamping():
    intervals = {"grandma": [(5.0, 6.0)]}
    out = pad_within_scenes_and_union(intervals, scenes=[], pad=3.0)
    assert len(out) == 1
    assert out[0].start == 2.0
    assert out[0].end == 9.0


def test_pad_clamps_to_zero():
    scenes = [(0.0, 100.0)]
    intervals = {"grandma": [(1.0, 2.0)]}
    out = pad_within_scenes_and_union(intervals, scenes, pad=10.0)
    assert out[0].start == 0.0


def test_union_both_in_same_scene():
    scenes = [(0.0, 10.0), (10.0, 20.0)]
    intervals = {"grandma": [(11.0, 12.0)], "grandpa": [(15.0, 16.0)]}
    out = pad_within_scenes_and_union(intervals, scenes, pad=15.0)
    assert len(out) == 1
    assert out[0].persons == frozenset({"grandma", "grandpa"})


def test_separate_scenes_stay_separate():
    scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    intervals = {"grandma": [(1.0, 2.0)], "grandpa": [(25.0, 26.0)]}
    out = pad_within_scenes_and_union(intervals, scenes, pad=2.0)
    assert len(out) == 2
    persons = {tuple(sorted(c.persons)) for c in out}
    assert persons == {("grandma",), ("grandpa",)}


def test_empty_intervals():
    scenes = [(0.0, 10.0)]
    out = pad_within_scenes_and_union({}, scenes, pad=5.0)
    assert out == []


def test_no_match_intervals():
    scenes = [(0.0, 10.0)]
    out = pad_within_scenes_and_union({"grandma": []}, scenes, pad=5.0)
    assert out == []
