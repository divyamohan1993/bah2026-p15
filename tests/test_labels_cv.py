"""Tests for forecast labels (strict pre-peak + masking) and CV embargo.

Covers ARCHITECTURE.md Section 5.3-5.4 / research doc 04 Section 4:

* labels are **strictly pre-peak** -- no positive window may fall at or after a
  flare peak (else a "forecast" is really a nowcast), and in-event samples are
  masked out of the negatives;
* the blocked rolling-origin CV **embargo** removes overlap between train and
  test -- there are **no shared or adjacent indices across the gap**, and the
  time gap is at least the embargo.

Pure python (stdlib only): a tiny stub event list and an inline light-curve
dict stand in for WS3's catalogue and WS1's synth, so the tests have zero
cross-workstream dependencies. The numpy-returning ``build_labels`` /
``blocked_splits`` array forms are checked under ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest
from flarecast.constants import CV_EMBARGO_MIN
from flarecast.forecast.cv import blocked_splits
from flarecast.forecast.labels import (
    build_labels_lists,
    class_at_least,
    event_class,
    event_peak,
)


# ---------------------------------------------------------------------------
# Minimal stub event (duck-typed: t_peak + goes_class, optional bounds).
# ---------------------------------------------------------------------------
class _StubEvent:
    def __init__(self, t_peak, goes_class, t_start=None, t_end=None):
        self.t_peak = t_peak
        self.goes_class = goes_class
        self.t_start = t_start
        self.t_end = t_end


def _flat_lightcurve(n=30, dt=60.0):
    return {
        "t": [float(i) * dt for i in range(n)],
        "sxr": [1e-6] * n,
        "hxr": [5.0] * n,
    }


# ---------------------------------------------------------------------------
# Class-ladder comparison.
# ---------------------------------------------------------------------------
def test_class_at_least_ladder():
    assert class_at_least("M1.0", "C")
    assert class_at_least("C5.0", "C")
    assert class_at_least("X1.0", "M")
    assert class_at_least("C1.0", "C")        # bare threshold = decade floor
    assert not class_at_least("B9.0", "C")
    assert not class_at_least("C9.9", "M")
    # Mantissa within a decade.
    assert class_at_least("M5.0", "M2.5")
    assert not class_at_least("M2.0", "M5.0")


def test_event_accessors_duck_typed():
    # object, dict, and tuple all work.
    assert event_peak(_StubEvent(600.0, "C2.0")) == 600.0
    assert event_class(_StubEvent(600.0, "C2.0")) == "C2.0"
    assert event_peak({"t_peak": 700.0, "goes_class": "M1.0"}) == 700.0
    assert event_class({"t_peak": 700.0, "goes_class": "M1.0"}) == "M1.0"
    assert event_peak((800.0, "X1.0")) == 800.0
    assert event_class((800.0, "X1.0")) == "X1.0"


# ---------------------------------------------------------------------------
# Strict pre-peak labels.
# ---------------------------------------------------------------------------
def test_labels_are_strictly_pre_peak():
    """No positive label may occur at or after the flare peak."""
    lc = _flat_lightcurve(n=30)
    peak_t = 600.0  # step 10
    cat = [_StubEvent(peak_t, "C2.0")]
    # mask_in_event=False so we can inspect the full label series row-by-row.
    X, y = build_labels_lists(
        cat, lc, horizon_min=15.0, class_threshold="C", mask_in_event=False
    )
    times = lc["t"]
    assert len(y) == len(times)
    for ti, yi in zip(times, y, strict=True):
        if yi == 1:
            assert ti < peak_t, f"positive label at/after peak (t={ti}, peak={peak_t})"
    # The exact at-peak step must be negative.
    peak_idx = times.index(peak_t)
    assert y[peak_idx] == 0


def test_positive_window_is_within_horizon():
    """Positives appear only within the horizon before the peak."""
    lc = _flat_lightcurve(n=40)
    peak_t = 1800.0  # step 30
    cat = [_StubEvent(peak_t, "C5.0")]
    horizon_min = 10.0  # 600 s
    X, y = build_labels_lists(
        cat, lc, horizon_min=horizon_min, class_threshold="C", mask_in_event=False
    )
    horizon_s = horizon_min * 60.0
    for ti, yi in zip(lc["t"], y, strict=True):
        if yi == 1:
            assert 0 < (peak_t - ti) <= horizon_s
        else:
            # negatives are outside the (t, t+N] pre-peak band.
            assert not (0 < (peak_t - ti) <= horizon_s)


def test_sub_threshold_flare_yields_no_positives():
    """A B-class flare does not create >=C positives, but still masks in-event."""
    lc = _flat_lightcurve(n=30)
    cat = [_StubEvent(600.0, "B5.0", t_start=300.0, t_end=900.0)]
    X, y = build_labels_lists(cat, lc, horizon_min=15.0, class_threshold="C")
    assert sum(y) == 0  # no >=C positive


def test_in_event_masking_removes_in_flare_negatives():
    """With bounded events, in-event negative samples are dropped entirely."""
    lc = _flat_lightcurve(n=30)
    # Flare fully in the past relative to most steps so its pre-window is gone,
    # leaving only in-event (decay) samples to be masked.
    cat = [_StubEvent(120.0, "C2.0", t_start=60.0, t_end=600.0)]
    X_masked, y_masked = build_labels_lists(
        cat, lc, horizon_min=15.0, class_threshold="C", mask_in_event=True
    )
    X_unmasked, y_unmasked = build_labels_lists(
        cat, lc, horizon_min=15.0, class_threshold="C", mask_in_event=False
    )
    # Masking can only remove rows, never add.
    assert len(y_masked) < len(y_unmasked)
    # Every retained masked row that is negative must be OUTSIDE [60, 600].
    times = lc["t"]
    # Reconstruct kept times by aligning the unmasked series.
    # Simpler invariant: count of in-event negatives removed > 0.
    n_in_event_neg = sum(
        1 for ti, yi in zip(times, y_unmasked, strict=True) if yi == 0 and 60.0 <= ti <= 600.0
    )
    assert n_in_event_neg > 0
    assert len(y_unmasked) - len(y_masked) == n_in_event_neg


def test_positive_kept_even_if_in_event_window_overlaps():
    """A pre-peak positive is never dropped by masking (it IS the signal)."""
    lc = _flat_lightcurve(n=30)
    # Bounds that overlap the pre-peak band; positives must survive.
    cat = [_StubEvent(600.0, "M1.0", t_start=0.0, t_end=1200.0)]
    X, y = build_labels_lists(
        cat, lc, horizon_min=15.0, class_threshold="C", mask_in_event=True
    )
    assert sum(y) >= 1  # at least one positive survived masking


# ---------------------------------------------------------------------------
# CV embargo / leakage prevention.
# ---------------------------------------------------------------------------
def test_blocked_splits_no_shared_or_adjacent_indices():
    """Train and test share no indices and are separated by the embargo gap."""
    n = 200
    dt = 60.0
    t = [float(i) * dt for i in range(n)]
    embargo_min = 5.0  # 300 s = 5 steps
    splits = blocked_splits(t, n_splits=4, embargo_min=embargo_min)
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        train_set = set(train_idx)
        test_set = set(test_idx)
        # No shared indices.
        assert not (train_set & test_set)
        # Train strictly precedes test.
        assert max(train_idx) < min(test_idx)
        # Adjacent indices across the gap are removed by the embargo: the index
        # gap must exceed the embargo span (in steps).
        embargo_steps = int(embargo_min * 60.0 / dt)
        assert (min(test_idx) - max(train_idx)) > embargo_steps
        # And the *time* gap is at least the embargo.
        gap_s = t[min(test_idx)] - t[max(train_idx)]
        assert gap_s >= embargo_min * 60.0


def test_blocked_splits_are_temporally_ordered_walk_forward():
    """Successive folds test progressively later blocks (rolling origin)."""
    n = 300
    t = [float(i) * 60.0 for i in range(n)]
    splits = blocked_splits(t, n_splits=5, embargo_min=2.0)
    test_starts = [min(te) for _tr, te in splits]
    assert test_starts == sorted(test_starts)
    # Each fold's train set lies entirely before its test set.
    for tr, te in splits:
        assert max(tr) < min(te)


def test_embargo_can_remove_entire_early_fold():
    """A very large embargo wipes the (short) early train sets -> fewer folds.

    With the default 120-min embargo and a 1-hour series at 1-min cadence, early
    folds have no surviving training history and are skipped, but the splitter
    must not crash and any returned fold must still be valid.
    """
    n = 60  # 1 hour at 60 s cadence
    t = [float(i) * 60.0 for i in range(n)]
    splits = blocked_splits(t, n_splits=5, embargo_min=CV_EMBARGO_MIN)
    # Default embargo (120 min) exceeds the whole hour -> all folds dropped.
    assert splits == []


def test_event_grouped_purge_keeps_flare_in_one_fold():
    """Event-grouped CV removes any train sample sharing a group with the test
    fold, so a flare's windows never straddle the gap."""
    n = 100
    t = [float(i) * 60.0 for i in range(n)]
    # Assign group ids: a "flare group" spanning the train/test boundary.
    groups = [i // 10 for i in range(n)]  # 10 contiguous groups of 10.
    splits = blocked_splits(t, n_splits=4, embargo_min=0.0, groups=groups)
    for tr, te in splits:
        test_groups = {groups[k] for k in te}
        train_groups = {groups[j] for j in tr}
        assert not (test_groups & train_groups), "a group straddles train/test"


# ---------------------------------------------------------------------------
# numpy array forms (skipped cleanly without numpy).
# ---------------------------------------------------------------------------
def test_build_labels_returns_numpy_arrays():
    pytest.importorskip("numpy")
    from flarecast.constants import N_FEATURES
    from flarecast.forecast.labels import build_labels

    lc = _flat_lightcurve(n=30)
    cat = [_StubEvent(600.0, "M1.0")]
    X, y = build_labels(cat, lc, horizon_min=15.0, class_threshold="C")
    assert X.ndim == 2 and X.shape[1] == N_FEATURES
    assert y.ndim == 1 and y.shape[0] == X.shape[0]
    assert {int(v) for v in y} <= {0, 1}


def test_blocked_splits_accepts_numpy_time_array():
    np = pytest.importorskip("numpy")
    t = np.arange(0, 200, 1, dtype=float) * 60.0
    splits = blocked_splits(t, n_splits=4, embargo_min=2.0)
    assert len(splits) >= 1
    for tr, te in splits:
        # Returned indices index a numpy array fine.
        assert len(np.asarray(t)[tr]) == len(tr)
        assert max(tr) < min(te)
