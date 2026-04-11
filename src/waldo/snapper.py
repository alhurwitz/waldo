from __future__ import annotations

from pydantic import BaseModel


class ClipScene(BaseModel):
    start: float
    end: float
    persons: frozenset[str]

    model_config = {"frozen": True}


def _scene_bounds_for(
    interval: tuple[float, float], scenes: list[tuple[float, float]]
) -> tuple[float, float] | None:
    """Return (min_start, max_end) of all scenes overlapping this interval."""
    s, e = interval
    overlapping = [(ss, se) for ss, se in scenes if ss < e and se > s]
    if not overlapping:
        return None
    return (min(o[0] for o in overlapping), max(o[1] for o in overlapping))


def pad_within_scenes_and_union(
    intervals_per_person: dict[str, list[tuple[float, float]]],
    scenes: list[tuple[float, float]],
    pad: float,
) -> list[ClipScene]:
    """Pad each interval by `pad` seconds on either side, clamp the window to
    the boundaries of whichever scene(s) the appearance falls in, then union
    overlapping windows across persons.
    """
    tagged: list[tuple[float, float, str]] = []
    for person, intervals in intervals_per_person.items():
        for s, e in intervals:
            padded_s = max(0.0, s - pad)
            padded_e = e + pad
            bounds = _scene_bounds_for((s, e), scenes)
            if bounds is not None:
                bs, be = bounds
                padded_s = max(padded_s, bs)
                padded_e = min(padded_e, be)
            if padded_e <= padded_s:
                continue
            tagged.append((padded_s, padded_e, person))

    if not tagged:
        return []

    tagged.sort(key=lambda x: x[0])
    result: list[ClipScene] = []
    for s, e, p in tagged:
        if result and s <= result[-1].end:
            prev = result[-1]
            result[-1] = ClipScene(
                start=prev.start,
                end=max(prev.end, e),
                persons=frozenset(prev.persons | {p}),
            )
        else:
            result.append(ClipScene(start=s, end=e, persons=frozenset({p})))
    return result


def pad_and_union(
    intervals_per_person: dict[str, list[tuple[float, float]]],
    pad: float,
) -> list[ClipScene]:
    """Pad each interval by `pad` seconds on either side, then union overlapping
    intervals across persons, tagging each clip with the persons present.
    """
    tagged: list[tuple[float, float, str]] = []
    for person, intervals in intervals_per_person.items():
        for s, e in intervals:
            tagged.append((max(0.0, s - pad), e + pad, person))

    if not tagged:
        return []

    tagged.sort(key=lambda x: x[0])
    result: list[ClipScene] = []
    for s, e, p in tagged:
        if result and s <= result[-1].end:
            prev = result[-1]
            result[-1] = ClipScene(
                start=prev.start,
                end=max(prev.end, e),
                persons=frozenset(prev.persons | {p}),
            )
        else:
            result.append(ClipScene(start=s, end=e, persons=frozenset({p})))
    return result


def expand_and_union(
    intervals_per_person: dict[str, list[tuple[float, float]]],
    scenes: list[tuple[float, float]],
) -> list[ClipScene]:
    """For each person's intervals, snap each interval out to fully cover any
    scene it overlaps. Then union overlapping snapped intervals across persons,
    tagging each resulting clip with the set of persons present in it.
    """
    if not scenes:
        # No scene info: return per-person intervals as-is, unioned naively.
        scenes = sorted({(s, e) for ivs in intervals_per_person.values() for s, e in ivs})

    # Per-person: list of (scene_start, scene_end) covered.
    snapped: list[tuple[float, float, str]] = []
    for person, intervals in intervals_per_person.items():
        covered: set[tuple[float, float]] = set()
        for s, e in intervals:
            for ss, se in scenes:
                if ss < e and se > s:  # overlap
                    covered.add((ss, se))
        for ss, se in covered:
            snapped.append((ss, se, person))

    if not snapped:
        return []

    # Group by scene span.
    by_scene: dict[tuple[float, float], set[str]] = {}
    for s, e, p in snapped:
        by_scene.setdefault((s, e), set()).add(p)

    # Sort by start, then merge any overlapping/adjacent groups (unioning persons).
    ordered = sorted(by_scene.items(), key=lambda kv: kv[0])
    result: list[ClipScene] = []
    for (s, e), persons in ordered:
        if result and s <= result[-1].end:
            prev = result[-1]
            result[-1] = ClipScene(
                start=prev.start,
                end=max(prev.end, e),
                persons=frozenset(prev.persons | persons),
            )
        else:
            result.append(ClipScene(start=s, end=e, persons=frozenset(persons)))
    return result
