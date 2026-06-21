"""Intra/inter-band de-duplication of catalogue events (Section 4.3, B.5).

De-duplication prevents counting each sub-peak of one complex flare as a separate
event (research doc ``03 Section 4.3``):

* **Within a band** -- two detections belong to the *same* flare unless they are
  separated by a clean return-to-baseline (the first event's ``t_end``) **plus**
  a guard time ``guard_s``. Sub-peaks that re-trigger before that are merged.
* **Across bands** -- one master event carries both soft and hard flags
  (association handles the pairing; here we additionally collapse residual
  duplicates whose intervals overlap within the guard).

:func:`deduplicate` is a pure function over a list of :class:`FlareEvent`.
"""

from __future__ import annotations

from .schema import FlareEvent

__all__ = ["deduplicate", "should_merge"]


def should_merge(a: FlareEvent, b: FlareEvent, guard_s: float) -> bool:
    """True if ``b`` is part of the same flare as the earlier ``a`` (Section 4.3).

    ``b`` (the later-starting event) is merged into ``a`` unless it begins after
    ``a`` has cleanly ended *and* a guard interval has elapsed, i.e. merge when::

        b.t_start <= a.t_end + guard_s

    O(1).
    """
    return b.t_start <= a.t_end + float(guard_s)


def deduplicate(events: list[FlareEvent], guard_s: float = 90.0) -> list[FlareEvent]:
    """Collapse re-triggered sub-peaks into single events (Section 4.3).

    Events are processed in start-time order; each is either merged into the
    currently-open event (if it re-triggers within ``guard_s`` of that event's
    end) or starts a new one. Merging extends the interval to the union, keeps
    the **stronger** GOES class and peak, ORs the band/Neupert flags, unions the
    detector lists, and takes the max confidence. Returns a new list (the input
    is not mutated). O(n log n) for the sort, O(n) for the sweep.

    Parameters
    ----------
    events:
        Candidate events (any band mix).
    guard_s:
        Re-detection guard (default :data:`flarecast.constants.DEDUP_GUARD_S`).
    """
    if not events:
        return []

    ordered = sorted(events, key=lambda e: e.t_start)
    merged: list[FlareEvent] = [_copy_event(ordered[0])]

    for ev in ordered[1:]:
        cur = merged[-1]
        if should_merge(cur, ev, guard_s):
            merged[-1] = _merge_into(cur, ev)
        else:
            merged.append(_copy_event(ev))
    return merged


def _copy_event(e: FlareEvent) -> FlareEvent:
    """Shallow copy of an event with independent flag/detector containers. O(1)."""
    return FlareEvent(
        event_id=e.event_id,
        t_start=e.t_start,
        t_peak=e.t_peak,
        t_end=e.t_end,
        goes_class=e.goes_class,
        soft=dict(e.soft) if e.soft else None,
        hard=dict(e.hard) if e.hard else None,
        flags=dict(e.flags),
        confidence=e.confidence,
        detectors=list(e.detectors),
        ref_match=dict(e.ref_match) if e.ref_match else None,
    )


def _merge_into(a: FlareEvent, b: FlareEvent) -> FlareEvent:
    """Merge ``b`` into ``a`` (same physical flare). Returns a new event. O(1)."""
    from ..detect.classify import class_to_flux  # local import: avoid cycle

    # Interval = union.
    t_start = min(a.t_start, b.t_start)
    t_end = max(a.t_end, b.t_end)

    # Keep the stronger peak / class.
    fa = _safe_flux(a.goes_class, class_to_flux)
    fb = _safe_flux(b.goes_class, class_to_flux)
    if fb > fa:
        goes_class = b.goes_class
        t_peak = b.t_peak
    else:
        goes_class = a.goes_class
        t_peak = a.t_peak

    # OR flags / counters; union detectors; max confidence.
    flags = dict(a.flags)
    for k, v in b.flags.items():
        if isinstance(v, bool):
            flags[k] = bool(flags.get(k, False)) or v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flags[k] = (flags.get(k, 0) or 0) + v
        else:
            flags.setdefault(k, v)

    detectors = list(dict.fromkeys([*a.detectors, *b.detectors]))
    soft = _merge_subrecord(a.soft, b.soft)
    hard = _merge_subrecord(a.hard, b.hard)
    confidence = max(a.confidence, b.confidence)
    ref_match = a.ref_match or b.ref_match

    return FlareEvent(
        event_id=a.event_id,
        t_start=t_start,
        t_peak=t_peak,
        t_end=t_end,
        goes_class=goes_class,
        soft=soft,
        hard=hard,
        flags=flags,
        confidence=confidence,
        detectors=detectors,
        ref_match=dict(ref_match) if ref_match else None,
    )


def _merge_subrecord(a: dict | None, b: dict | None) -> dict | None:
    """Merge two per-band sub-records, preferring present/detected data. O(1)."""
    if a is None:
        return dict(b) if b else None
    if b is None:
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        if out.get(k) is None:
            out[k] = v
    out["detected"] = bool(a.get("detected")) or bool(b.get("detected"))
    return out


def _safe_flux(cls: str | None, conv) -> float:
    """GOES class -> flux for strength comparison; ``-1`` on failure. O(1)."""
    if not cls:
        return -1.0
    try:
        return conv(cls)
    except (ValueError, KeyError):
        return -1.0
