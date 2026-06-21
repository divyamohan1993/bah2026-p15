"""Leakage-free temporal cross-validation (blocked rolling-origin + embargo).

Governing research: ``docs/research/04-forecasting-models.md`` Section 4
(*Cross-validation -- NO leakage*) and ARCHITECTURE.md Section 5.4.

The contract (Appendix B.6)::

    def blocked_splits(t, n_splits=5, embargo_min=120.0) -> list[(train_idx, test_idx)]

**Why not random k-fold.** Adjacent sliding windows overlap, so one flare's
pre-window can land in both train and test under a random shuffle -- a
documented leakage failure mode (research doc 04 Section 4). We therefore use
**blocked rolling-origin / walk-forward** splits over *time*: each fold trains
on a contiguous past block and tests on the contiguous block immediately
after, with an **embargo gap** (:data:`flarecast.constants.CV_EMBARGO_MIN`
minutes, here ``embargo_min``) removed from the *end of train* so no training
window peeks into a test flare. The embargo must be ``>= N + max window``; the
default 120 min comfortably covers the standard horizons {15, 30, 60} min plus
the longest (15 min) feature window.

This module is **pure standard library** (it yields plain index lists). numpy
is accepted for the ``t`` argument but never required: timestamps are read via
iteration, and the returned ``train_idx`` / ``test_idx`` are Python ``list[int]``
(numpy arrays index fine with a list).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from flarecast.constants import CV_EMBARGO_MIN

__all__ = ["blocked_splits", "rolling_origin_splits"]


def _as_float_list(t: Any) -> list[float]:
    """Coerce ``t`` (numpy array / list / any iterable) to ``list[float]``."""
    if t is None:
        return []
    # numpy array, list, tuple, range, ... all iterable.
    try:
        return [float(x) for x in t]
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"blocked_splits: cannot iterate t of type {type(t)!r}") from exc


def blocked_splits(
    t: Any,
    n_splits: int = 5,
    embargo_min: float = CV_EMBARGO_MIN,
    groups: Iterable[Any] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Rolling-origin temporal CV splits with an embargo gap.

    The ordered time axis is partitioned into ``n_splits + 1`` contiguous
    chunks ``C0, C1, ..., C_{n_splits}``. Fold ``i`` (1-based) tests on chunk
    ``C_i`` and trains on everything strictly before it (``C0..C_{i-1}``), minus
    the embargo: any training index whose timestamp lies within ``embargo_min``
    minutes *before* the first test timestamp is dropped. This guarantees a
    clean temporal gap between train and test so no overlapping window leaks.

    Parameters
    ----------
    t:
        Per-sample timestamps (epoch seconds), in **non-decreasing** order
        (the natural order of a streamed light curve). Accepts a numpy array,
        list, or any iterable of numbers.
    n_splits:
        Number of test folds (default 5). Produces up to ``n_splits`` folds; a
        fold whose train set is empty after the embargo is skipped.
    embargo_min:
        Embargo/purge width in minutes (default
        :data:`flarecast.constants.CV_EMBARGO_MIN` = 120). Train samples within
        this many minutes before the test block start are removed.
    groups:
        Optional per-sample event-group id (e.g. flare id). When supplied, any
        group that has at least one sample in the test fold is **entirely**
        removed from that fold's train set (event-grouped CV: "all windows of
        one flare go to the same fold", research doc 04 Section 4). This is a
        belt-and-braces complement to the time embargo.

    Returns
    -------
    list of (train_idx, test_idx):
        Each a ``list[int]`` of positional indices into ``t``.
    """
    times = _as_float_list(t)
    n = len(times)
    if n == 0 or n_splits < 1:
        return []

    grp_list: list[Any] | None = None
    if groups is not None:
        grp_list = list(groups)
        if len(grp_list) != n:
            raise ValueError(
                f"groups length {len(grp_list)} != number of samples {n}"
            )

    embargo_s = float(embargo_min) * 60.0

    # Contiguous, near-equal chunk boundaries over the ordered index range.
    # We want n_splits test blocks each preceded by some train data, so split
    # the index range into (n_splits + 1) parts and use parts 1..n_splits as
    # the successive test blocks (part 0 is the initial train-only seed).
    n_chunks = n_splits + 1
    # Boundaries via integer interpolation so chunks differ by <=1 in size.
    bounds = [round(i * n / n_chunks) for i in range(n_chunks + 1)]

    splits: list[tuple[list[int], list[int]]] = []
    for i in range(1, n_chunks):
        test_lo, test_hi = bounds[i], bounds[i + 1]
        if test_hi <= test_lo:
            continue
        test_idx = list(range(test_lo, test_hi))
        first_test_t = times[test_lo]

        # Train = everything strictly before the test block start index.
        train_candidates = range(0, test_lo)

        # Embargo: drop every train sample within embargo_s of the test start,
        # so the surviving train/test time gap is strictly greater than the
        # embargo (the conservative, leak-safe purge). A sample whose timestamp
        # is < first_test_t - embargo_s is kept; one exactly at the boundary is
        # dropped.
        embargo_cutoff = first_test_t - embargo_s
        train_idx = [j for j in train_candidates if times[j] < embargo_cutoff]

        # Event-group purge: remove any train sample sharing a group with the
        # test fold (prevents a flare's windows straddling the gap by index).
        if grp_list is not None and train_idx:
            test_groups = {grp_list[k] for k in test_idx}
            train_idx = [j for j in train_idx if grp_list[j] not in test_groups]

        if not train_idx:
            # No usable history after the embargo -> skip this (early) fold.
            continue
        splits.append((train_idx, test_idx))

    return splits


# Alias under the more descriptive name used in the prose / build plan.
rolling_origin_splits = blocked_splits
