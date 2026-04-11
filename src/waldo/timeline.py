from __future__ import annotations


def build_intervals(
    samples: list[tuple[float, bool]],
    gap: float,
    min_len: float,
) -> list[tuple[float, float]]:
    """Convert a list of (timestamp, matched) samples into intervals.

    - Merge consecutive matched samples (assigning each match a span up to the
      midpoint to the next sample).
    - Bridge gaps between intervals shorter than `gap`.
    - Drop intervals shorter than `min_len`.
    """
    if not samples:
        return []
    samples = sorted(samples)

    # Build raw intervals: each matched sample contributes [t_i, t_{i+1}].
    raw: list[tuple[float, float]] = []
    for i, (t, matched) in enumerate(samples):
        if not matched:
            continue
        end = samples[i + 1][0] if i + 1 < len(samples) else t
        if end <= t:
            end = t
        raw.append((t, end))

    if not raw:
        return []

    # Merge contiguous/overlapping raw intervals.
    merged: list[tuple[float, float]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Bridge gaps shorter than `gap`.
    bridged: list[tuple[float, float]] = []
    for s, e in merged:
        if bridged and s - bridged[-1][1] < gap:
            bridged[-1] = (bridged[-1][0], e)
        else:
            bridged.append((s, e))

    # Drop blips.
    return [(s, e) for s, e in bridged if (e - s) >= min_len]
