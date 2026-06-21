"""Lead-time quantification and the lead-time-vs-false-alarm sweep.

Governing research: ``docs/research/04-forecasting-models.md`` Section 5
(*Lead-Time Quantification*) and ARCHITECTURE.md Section 5.4 / 10.1.

Contract (Appendix B.6)::

    def lead_time(prob_series, t, peak_times, theta,
                  w_min=7200.0, k_of_m=(2,3)) -> Array            # LT per flare (s)
    def lt_vs_far(prob_series, t, peak_times, thetas) -> DF       # theta, median_lt, tpr, far

**Definition (research doc 04 Section 5).** For each true flare with peak time
``p``, the lead time is ``LT = p - t_alert`` where ``t_alert`` is the **first**
probability crossing of threshold ``theta`` inside the pre-peak window
``[p - W, p)`` that is **confirmed** by ``k`` of the next ``m`` samples also
being above ``theta`` (the anti-flicker rule,
:data:`flarecast.constants.LEADTIME_K_OF_M`). ``W`` defaults to
:data:`flarecast.constants.LEADTIME_WINDOW_S` (7200 s = 120 min). If the
probability never crosses (confirmed) before ``p`` the forecast is **missed**
(LT undefined / ``NaN``), which counts against TPR.

This module is **pure standard library** (``math`` / ``statistics``). The
batch ``lead_time`` returns a numpy array to satisfy the ``Array`` contract
(numpy imported lazily); :func:`lead_time_list` returns a plain list of
``float | None`` for numpy-free callers/tests. ``lt_vs_far`` returns a pandas
``DataFrame`` when pandas is available, else a list of row-dicts (same columns),
so the sweep is usable offline without pandas.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from flarecast.constants import (
    LEADTIME_K_OF_M,
    LEADTIME_REPORT_BUCKETS_MIN,
    LEADTIME_WINDOW_S,
)

__all__ = [
    "lead_time",
    "lead_time_list",
    "lt_vs_far",
    "lead_time_report",
]


def _as_float_list(x: Any) -> list[float]:
    return [float(v) for v in x]


def _confirmed_crossing_time(
    times: list[float],
    probs: list[float],
    idx: int,
    theta: float,
    k: int,
    m: int,
) -> bool:
    """True iff index ``idx`` is a confirmed crossing: it and ``k-1`` of the
    next ``m-1`` samples (so ``k`` of a window of ``m`` starting at ``idx``) are
    ``>= theta``. The window is truncated at the end of the series.
    """
    if probs[idx] < theta:
        return False
    window_end = min(idx + m, len(probs))
    above = sum(1 for j in range(idx, window_end) if probs[j] >= theta)
    return above >= k


def lead_time_list(
    prob_series: Any,
    t: Any,
    peak_times: Any,
    theta: float,
    w_min: float = LEADTIME_WINDOW_S,
    k_of_m: tuple[int, int] = LEADTIME_K_OF_M,
) -> list[float | None]:
    """Pure-python lead time per true flare (seconds), ``None`` if missed.

    Parameters
    ----------
    prob_series:
        Forecast probability at each step (aligned with ``t``).
    t:
        Step timestamps (epoch seconds), non-decreasing.
    peak_times:
        True flare peak times (epoch seconds).
    theta:
        Probability threshold for an alert.
    w_min:
        Pre-peak look-back window ``W`` in **seconds** (the parameter keeps the
        contract's name; the default is the seconds constant
        :data:`flarecast.constants.LEADTIME_WINDOW_S`).
    k_of_m:
        Anti-flicker ``(k, m)``: require ``k`` of ``m`` consecutive samples
        (from the crossing) above ``theta`` to confirm.
    """
    probs = _as_float_list(prob_series)
    times = _as_float_list(t)
    peaks = _as_float_list(peak_times)
    if len(probs) != len(times):
        raise ValueError(
            f"prob_series ({len(probs)}) and t ({len(times)}) length mismatch"
        )
    k, m = int(k_of_m[0]), int(k_of_m[1])
    W = float(w_min)

    out: list[float | None] = []
    for p in peaks:
        lo = p - W
        t_alert: float | None = None
        for i, ti in enumerate(times):
            if ti < lo:
                continue
            if ti >= p:
                # Strictly pre-peak window [p - W, p).
                break
            if _confirmed_crossing_time(times, probs, i, theta, k, m):
                t_alert = ti
                break
        out.append((p - t_alert) if t_alert is not None else None)
    return out


def lead_time(
    prob_series: Any,
    t: Any,
    peak_times: Any,
    theta: float,
    w_min: float = LEADTIME_WINDOW_S,
    k_of_m: tuple[int, int] = LEADTIME_K_OF_M,
):
    """numpy twin of :func:`lead_time_list`: LT per true flare (seconds).

    Missed forecasts are encoded as ``NaN`` so the result is a homogeneous
    float array (the contract's ``Array``). numpy is imported lazily.
    """
    import numpy as np  # lazy

    vals = lead_time_list(prob_series, t, peak_times, theta, w_min, k_of_m)
    return np.asarray(
        [v if v is not None else float("nan") for v in vals], dtype=np.float64
    )


def _far_at_theta(
    probs: list[float],
    times: list[float],
    peaks: list[float],
    theta: float,
    k: int,
    m: int,
    assoc_window_s: float,
) -> tuple[float, float]:
    """Compute (tpr, far) for a single threshold over the whole series.

    An **alert** is a confirmed crossing (k-of-m). An alert is a *hit* if a true
    peak lies within ``[t_alert, t_alert + assoc_window_s]`` (the alert precedes
    the peak it warns of); otherwise it is a *false alarm*. TPR = fraction of
    true peaks that had at least one preceding confirmed alert within ``W``;
    FAR = false_alarms / total_alerts.
    """
    n = len(probs)
    # Collect confirmed alert times (de-bounced: skip while still above theta to
    # avoid counting every sample of one sustained alert as a new alarm).
    alert_times: list[float] = []
    i = 0
    while i < n:
        if _confirmed_crossing_time(times, probs, i, theta, k, m):
            alert_times.append(times[i])
            # Advance past the contiguous above-theta run (one alarm episode).
            j = i
            while j < n and probs[j] >= theta:
                j += 1
            i = max(j, i + 1)
        else:
            i += 1

    # Hits / false alarms.
    false_alarms = 0
    for at in alert_times:
        if not any(at <= p <= at + assoc_window_s for p in peaks):
            false_alarms += 1
    total_alerts = len(alert_times)
    far_val = false_alarms / total_alerts if total_alerts else 0.0

    # TPR: a peak is detected if some confirmed alert precedes it within W.
    W = assoc_window_s
    detected = 0
    for p in peaks:
        if any(p - W <= at < p for at in alert_times):
            detected += 1
    tpr = detected / len(peaks) if peaks else 0.0
    return tpr, far_val


def lt_vs_far(
    prob_series: Any,
    t: Any,
    peak_times: Any,
    thetas: Any,
    w_min: float = LEADTIME_WINDOW_S,
    k_of_m: tuple[int, int] = LEADTIME_K_OF_M,
):
    """Sweep ``theta`` -> table of ``(theta, median_lt, tpr, far)``.

    *The* operating-point selection plot for the judges (research doc 04
    Section 5): lower ``theta`` => earlier alerts (larger median LT) but more
    false alarms. ``median_lt`` is in **seconds** over the hit flares
    (``NaN`` if none).

    Returns a pandas ``DataFrame`` (columns ``theta, median_lt, tpr, far``)
    when pandas is importable, else a list of dicts with the same keys so the
    sweep works offline without pandas.
    """
    probs = _as_float_list(prob_series)
    times = _as_float_list(t)
    peaks = _as_float_list(peak_times)
    k, m = int(k_of_m[0]), int(k_of_m[1])
    W = float(w_min)

    rows: list[dict[str, float]] = []
    for theta in _as_float_list(thetas):
        lts = [v for v in lead_time_list(probs, times, peaks, theta, W, k_of_m)
               if v is not None]
        med = median(lts) if lts else float("nan")
        tpr, far_val = _far_at_theta(probs, times, peaks, theta, k, m, W)
        rows.append({
            "theta": theta,
            "median_lt": med,
            "tpr": tpr,
            "far": far_val,
        })

    try:
        import pandas as pd  # lazy/optional
        return pd.DataFrame(rows, columns=["theta", "median_lt", "tpr", "far"])
    except ImportError:
        return rows


def lead_time_report(
    prob_series: Any,
    t: Any,
    peak_times: Any,
    theta: float,
    w_min: float = LEADTIME_WINDOW_S,
    k_of_m: tuple[int, int] = LEADTIME_K_OF_M,
    buckets_min: tuple[int, ...] = LEADTIME_REPORT_BUCKETS_MIN,
) -> dict:
    """Summarize the lead-time distribution at a fixed operating point.

    Returns median / IQR (seconds), the number of hits vs misses, and the
    fraction of hits with ``LT >= b`` minutes for each ``b`` in ``buckets_min``
    (the operationally-meaningful cushions, default
    :data:`flarecast.constants.LEADTIME_REPORT_BUCKETS_MIN` = {5,10,15,30}).
    """
    lts_all = lead_time_list(prob_series, t, peak_times, theta, w_min, k_of_m)
    hits = [v for v in lts_all if v is not None]
    n_total = len(lts_all)
    n_hit = len(hits)
    n_miss = n_total - n_hit

    def _pctile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return float("nan")
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = q * (len(sorted_vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    s = sorted(hits)
    frac_ge = {
        b: (sum(1 for v in hits if v >= b * 60.0) / n_hit) if n_hit else 0.0
        for b in buckets_min
    }
    return {
        "theta": float(theta),
        "median_lt_s": median(hits) if hits else float("nan"),
        "iqr_lt_s": (_pctile(s, 0.75) - _pctile(s, 0.25)) if hits else float("nan"),
        "n_hit": n_hit,
        "n_miss": n_miss,
        "tpr": (n_hit / n_total) if n_total else 0.0,
        "frac_ge_min": frac_ge,
    }
