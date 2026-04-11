from waldo.snapper import expand_and_union


def test_expand_to_scene():
    scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    intervals = {"grandma": [(12.0, 13.0)]}
    out = expand_and_union(intervals, scenes)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (10.0, 20.0)
    assert out[0].persons == frozenset({"grandma"})


def test_union_both_in_same_scene():
    scenes = [(0.0, 10.0), (10.0, 20.0)]
    intervals = {"grandma": [(11.0, 12.0)], "grandpa": [(15.0, 16.0)]}
    out = expand_and_union(intervals, scenes)
    assert len(out) == 1
    assert out[0].persons == frozenset({"grandma", "grandpa"})


def test_separate_scenes():
    scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    intervals = {"grandma": [(1.0, 2.0)], "grandpa": [(25.0, 26.0)]}
    out = expand_and_union(intervals, scenes)
    assert len(out) == 2
    persons = {tuple(sorted(c.persons)) for c in out}
    assert persons == {("grandma",), ("grandpa",)}
