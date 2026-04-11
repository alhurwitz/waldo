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

    If scenes is empty (single-scene video / no cuts detected), padding is
    applied without clamping.
    """
    tagged: list[tuple[float, float, str]] = []
    for person, intervals in intervals_per_person.items():
        for s, e in intervals:
            padded_s = max(0.0, s - pad)
            padded_e = e + pad
            if scenes:
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
