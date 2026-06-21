"""Pre-peak label construction for the sliding-window forecast.

Governing research: ``docs/research/04-forecasting-models.md`` Section 4
(*Problem Formulation -> Labels*) and ARCHITECTURE.md Section 5.3.

The contract (Appendix B.6)::

    def build_labels(catalogue, lightcurves, horizon_min, class_threshold="C",
                     strict_pre_peak=True, mask_in_event=True) -> (X[n,30], y[n])

**Definition (research doc 04 Section 4, "Labels").** Ground truth is the set of
flare **peaks** (lead time is defined relative to the peak). For a window ending
at step ``t`` and a flare peak ``p`` of class ``>= class_threshold``:

* the window is **positive** for horizon ``N`` iff ``0 < p - t <= N`` (a flare
  of sufficient class peaks within the next ``N`` minutes), and
* with ``strict_pre_peak=True`` we additionally require ``t < p`` so a
  "forecast" can never fire *at or after* the peak (it would be a nowcast).
  ``0 < p - t`` already enforces this; the flag is kept for contract fidelity
  and to make the intent explicit.

**In-event masking** (``mask_in_event=True``). Samples that fall *inside* a
flare (between its onset/start and end) are neither clean precursors nor true
negatives, so they are **excluded** from the training set entirely
(ARCHITECTURE.md Section 5.3: "Mask in-flare/decay samples from the negative
class ... so they don't confuse the precursor signal"). A positive pre-peak
window is kept even if it nominally overlaps the rising edge, because the
positive label *is* the precursor signal; masking only removes samples that
would otherwise be labelled **negative** while in-event.

Minimal event interface
------------------------
To avoid a hard dependency on Workstream 3's exact ``FlareEvent`` import, an
"event" here is duck-typed. :func:`build_labels` accepts a list of any of:

* an object with attributes ``t_peak`` (epoch seconds, float) and
  ``goes_class`` (e.g. ``"M2.5"``); optional ``t_start`` / ``t_end`` (epoch
  seconds) bound the in-event mask. This matches WS3's ``FlareEvent`` and the
  Section 9.3 JSON record (peak/start/end + ``goes_class``);
* a mapping with keys ``"t_peak"`` / ``"goes_class"`` (+ optional
  ``"t_start"`` / ``"t_end"``);
* a ``(t_peak, goes_class)`` tuple.

Timestamps are interpreted in epoch **seconds**; the lightcurve time column is
matched leniently (see :func:`flarecast.forecast.features._window_rows`).
"""

from __future__ import annotations

from typing import Any

from flarecast.constants import GOES_CLASS_LADDER
from flarecast.forecast.features import FeatureExtractor, _window_rows

__all__ = ["build_labels", "event_peak", "event_class", "class_at_least"]


# ---------------------------------------------------------------------------
# Minimal event accessors (duck-typed; no WS3 import).
# ---------------------------------------------------------------------------
def _getattr_or_key(ev: Any, name: str, default: Any = None) -> Any:
    """Fetch ``name`` from an object attribute, mapping key, else ``default``."""
    if isinstance(ev, dict):
        return ev.get(name, default)
    return getattr(ev, name, default)


def event_peak(ev: Any) -> float:
    """Return the flare peak time (epoch seconds) for a duck-typed event."""
    if isinstance(ev, (tuple, list)) and not isinstance(ev, str):
        return float(ev[0])
    p = _getattr_or_key(ev, "t_peak")
    if p is None:
        raise KeyError(f"event {ev!r} has no t_peak")
    return float(p)


def event_class(ev: Any) -> str:
    """Return the GOES class string (e.g. ``"M2.5"``) for a duck-typed event."""
    if isinstance(ev, (tuple, list)) and not isinstance(ev, str):
        return str(ev[1]) if len(ev) > 1 else "C1.0"
    c = _getattr_or_key(ev, "goes_class")
    if c is None:
        c = _getattr_or_key(ev, "cls")
    return str(c) if c is not None else "C1.0"


def _event_bounds(ev: Any) -> tuple[float | None, float | None]:
    """Return (t_start, t_end) in epoch seconds, or (None, None) if absent."""
    ts = _getattr_or_key(ev, "t_start")
    te = _getattr_or_key(ev, "t_end")
    ts = float(ts) if ts is not None else None
    te = float(te) if te is not None else None
    return ts, te


def _class_letter_and_mant(cls: str) -> tuple[str, float]:
    """Parse ``"M2.5"`` -> ("M", 2.5); tolerant of ``"M"`` / lowercase / junk."""
    cls = (cls or "").strip().upper()
    if not cls:
        return "A", 1.0
    letter = cls[0]
    if letter not in GOES_CLASS_LADDER:
        return "A", 1.0
    rest = cls[1:].strip()
    try:
        mant = float(rest) if rest else 1.0
    except ValueError:
        mant = 1.0
    return letter, mant


def class_at_least(cls: str, threshold: str) -> bool:
    """True iff GOES class ``cls`` is >= ``threshold`` on the A<B<C<M<X ladder.

    Compares the letter decade first, then the mantissa within the same decade,
    so ``"C5.0" >= "C"`` and ``"M1.0" >= "C"`` are both True while
    ``"B9.0" >= "C"`` is False. A bare threshold letter (``"C"``) means
    "anywhere in the C decade or above" (mantissa 0).
    """
    cl, cm = _class_letter_and_mant(cls)
    tl, tm = _class_letter_and_mant(threshold)
    # A bare threshold like "C" has no mantissa intent -> floor at 0.
    if len((threshold or "").strip()) <= 1:
        tm = 0.0
    ci = GOES_CLASS_LADDER.index(cl)
    ti = GOES_CLASS_LADDER.index(tl)
    if ci != ti:
        return ci > ti
    return cm >= tm


# ---------------------------------------------------------------------------
# Label builder.
# ---------------------------------------------------------------------------
def build_labels(
    catalogue: list[Any],
    lightcurves: Any,
    horizon_min: float,
    class_threshold: str = "C",
    strict_pre_peak: bool = True,
    mask_in_event: bool = True,
):
    """Build ``(X, y)`` feature/label arrays from a catalogue + light curves.

    Parameters
    ----------
    catalogue:
        List of duck-typed flare events (see module docstring). Only events of
        class ``>= class_threshold`` define positive windows; *all* events
        (regardless of class) contribute to the in-event mask so a sub-threshold
        flare's decay is still excluded from the negatives.
    lightcurves:
        Window object (pandas ``DataFrame`` or column-mapping) with soft/hard
        and time columns, oldest-first. Each row is one aggregated step.
    horizon_min:
        Forecast horizon ``N`` in minutes. A row at time ``t`` is positive iff a
        qualifying peak lies in ``(t, t + N*60]`` seconds.
    class_threshold:
        Minimum GOES class for a *positive* (default ``"C"``).
    strict_pre_peak:
        If True (default), require ``t < p`` for positives (already implied by
        ``0 < p - t``); kept for contract fidelity.
    mask_in_event:
        If True (default), drop rows whose time falls inside any event's
        ``[t_start, t_end]`` *and* that are not themselves positive pre-peak
        windows.

    Returns
    -------
    (X, y):
        ``X`` is a ``numpy.ndarray`` of shape ``(n_kept, N_FEATURES)`` and ``y``
        a ``numpy.ndarray`` of shape ``(n_kept,)`` with values in ``{0, 1}``.
        numpy is imported lazily (the heavy work -- labelling -- is pure
        python); a numpy-free caller can read :func:`build_labels_lists`.
    """
    import numpy as np  # lazy; only for the return container.

    X_rows, y_rows = build_labels_lists(
        catalogue,
        lightcurves,
        horizon_min,
        class_threshold=class_threshold,
        strict_pre_peak=strict_pre_peak,
        mask_in_event=mask_in_event,
    )
    if not X_rows:
        from flarecast.constants import N_FEATURES

        return (
            np.zeros((0, N_FEATURES), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(X_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.int64),
    )


def build_labels_lists(
    catalogue: list[Any],
    lightcurves: Any,
    horizon_min: float,
    class_threshold: str = "C",
    strict_pre_peak: bool = True,
    mask_in_event: bool = True,
) -> tuple[list[list[float]], list[int]]:
    """Pure-python core of :func:`build_labels` (returns lists, no numpy).

    Exposed so the labelling/masking logic is unit-testable with zero optional
    dependencies. Streams the light curve once through a single
    :class:`FeatureExtractor` (so features are causal and incrementally built),
    emitting one feature row per retained step.
    """
    sxr, hxr, times = _window_rows(lightcurves)
    horizon_s = float(horizon_min) * 60.0

    # Pre-extract peaks (qualifying) and event bounds (all) -- sorted for a
    # simple two-pointer-free scan (catalogues are small).
    qualifying_peaks: list[float] = []
    all_bounds: list[tuple[float, float]] = []
    for ev in catalogue or []:
        try:
            p = event_peak(ev)
        except (KeyError, TypeError, ValueError):
            continue
        if class_at_least(event_class(ev), class_threshold):
            qualifying_peaks.append(p)
        ts, te = _event_bounds(ev)
        if ts is None and te is None:
            # No explicit bounds: treat a small symmetric guard around the peak
            # as "in event" so decay/rise near an unbounded peak is still
            # maskable. Use the horizon as a conservative half-width cap of
            # 600 s (10 min), independent of N, so masking does not erase the
            # entire pre-window.
            guard = 600.0
            all_bounds.append((p - guard, p + guard))
        else:
            lo = ts if ts is not None else p
            hi = te if te is not None else p
            if hi < lo:
                lo, hi = hi, lo
            all_bounds.append((lo, hi))
    qualifying_peaks.sort()
    all_bounds.sort()

    def _is_positive(t: float) -> bool:
        # Any qualifying peak in (t, t+horizon]; strict_pre_peak => p>t (implied).
        for p in qualifying_peaks:
            d = p - t
            if d <= 0:
                continue
            if d <= horizon_s:
                if strict_pre_peak and not (t < p):
                    continue
                return True
            # peaks are sorted ascending; once d>horizon we can stop early.
            if d > horizon_s:
                break
        return False

    def _in_event(t: float) -> bool:
        for lo, hi in all_bounds:
            if lo <= t <= hi:
                return True
            if lo > t:
                break
        return False

    ex = FeatureExtractor()
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for s, h, t in zip(sxr, hxr, times, strict=True):
        feat = ex.update(s, h, t, horizon_min)
        pos = _is_positive(t)
        if mask_in_event and not pos and _in_event(t):
            # In-event negative -> drop (do not feed the extractor's history a
            # gap: the extractor already advanced above, preserving causality).
            continue
        X_rows.append(feat)
        y_rows.append(1 if pos else 0)
    return X_rows, y_rows
