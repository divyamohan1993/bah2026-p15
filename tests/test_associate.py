"""Tests for Neupert-aware soft+hard association (research 03 Section 4.2).

A hard burst slightly *before* a soft rise (the Neupert ordering) must associate
into one master event; a hard burst far from any soft rise must not. Also checks
the asymmetric MATCH_SCORE window and noisy-OR confidence fusion. Pure stdlib --
detections are constructed directly as ``DetectionState`` records.
"""

from __future__ import annotations

from flarecast.catalog.associate import Associator, match_score
from flarecast.catalog.confidence import fuse_confidence, noisy_or
from flarecast.constants import ASSOC_TAU_MATCH
from flarecast.types import DetectionState


def _soft(onset_t, peak_t, peak_flux=2.5e-5):
    return DetectionState(
        onset=True,
        in_event=True,
        statistic=1.0,
        onset_time=onset_t,
        meta={
            "t_peak": peak_t,
            "peak_flux": peak_flux,
            "goes_class": "M2.5",
            "detectors": ["CUSUM"],
        },
    )


def _hard(onset_t, peak_t):
    return DetectionState(
        onset=True,
        in_event=True,
        statistic=8.0,
        onset_time=onset_t,
        meta={"t_peak": peak_t, "peak_counts": 850.0, "detectors": ["FOCuS"]},
    )


# ---------------------------------------------------------------------------
# MATCH_SCORE
# ---------------------------------------------------------------------------
def test_match_score_high_when_hard_leads_soft():
    """Hard onset/peak just before soft (Neupert) -> high score above tau."""
    soft = _soft(onset_t=1000.0, peak_t=1100.0)
    hard = _hard(onset_t=940.0, peak_t=990.0)  # leads onset by 60 s, peak first
    score = match_score(soft, hard)
    assert score >= ASSOC_TAU_MATCH
    assert score > 0.7  # strong proximity + Neupert peak-lag bonus


def test_match_score_low_when_far_apart():
    """A hard burst ~13 min before the soft rise scores below tau."""
    soft = _soft(onset_t=1000.0, peak_t=1100.0)
    hard = _hard(onset_t=200.0, peak_t=250.0)  # 800 s before -> outside w_lead
    score = match_score(soft, hard)
    assert score < ASSOC_TAU_MATCH


def test_match_score_asymmetric_window():
    """A hard burst *after* the soft onset is penalised more than one before."""
    soft = _soft(onset_t=1000.0, peak_t=1100.0)
    before = _hard(onset_t=850.0, peak_t=900.0)  # 150 s before
    after = _hard(onset_t=1150.0, peak_t=1200.0)  # 150 s after (un-Neupert-like)
    s_before = match_score(soft, before)
    s_after = match_score(soft, after)
    assert s_before > s_after


def test_match_score_zero_without_times():
    soft = DetectionState(onset=True, in_event=True, statistic=1.0, onset_time=None)
    hard = _hard(onset_t=940.0, peak_t=990.0)
    assert match_score(soft, hard) == 0.0


# ---------------------------------------------------------------------------
# Associator (streaming)
# ---------------------------------------------------------------------------
def test_associator_merges_hard_before_soft():
    """A hard burst slightly before a soft rise is merged into one event."""
    assoc = Associator()
    # Hard arrives first (leads), then the soft rise.
    assert assoc.add(_hard(940.0, 990.0), "hard", 940.0) is None
    event = assoc.add(_soft(1000.0, 1100.0), "soft", 1000.0)
    assert event is not None, "soft+hard should have associated into one event"
    assert event.flags["soft"] is True
    assert event.flags["hard"] is True
    assert event.flags["neupert_consistent"] is True  # HXR peak precedes SXR
    assert "CUSUM" in event.detectors and "FOCuS" in event.detectors
    assert event.goes_class == "M2.5"
    # Master start is the earliest band onset (the hard lead).
    assert event.t_start == 940.0


def test_associator_does_not_merge_far_apart():
    """A far-apart hard burst and soft rise stay separate events."""
    assoc = Associator()
    assert assoc.add(_hard(200.0, 250.0), "hard", 200.0) is None
    # The soft rise is too far from the hard burst to associate immediately.
    assert assoc.add(_soft(1000.0, 1100.0), "soft", 1000.0) is None
    # Flushing yields two independent single-band events.
    solo = assoc.flush()
    assert len(solo) == 2
    bands = sorted(("hard" if e.flags["hard"] and not e.flags["soft"] else "soft") for e in solo)
    assert bands == ["hard", "soft"]
    # The hard-only event is flagged sxr_weak (a non-thermal-dominated flare).
    hard_only = next(e for e in solo if e.flags["hard"] and not e.flags["soft"])
    assert hard_only.flags["sxr_weak"] is True


def test_associator_solo_soft_event_on_flush():
    assoc = Associator()
    assert assoc.add(_soft(500.0, 600.0), "soft", 500.0) is None
    solo = assoc.flush()
    assert len(solo) == 1
    assert solo[0].flags["soft"] is True
    assert solo[0].flags["hard"] is False


# ---------------------------------------------------------------------------
# Confidence fusion
# ---------------------------------------------------------------------------
def test_noisy_or_basic():
    assert noisy_or([]) == 0.0
    assert abs(noisy_or([0.5]) - 0.5) < 1e-12
    # Two independent 0.5 detectors -> 1 - 0.25 = 0.75.
    assert abs(noisy_or([0.5, 0.5]) - 0.75) < 1e-12


def test_noisy_or_clamps():
    # Out-of-range probabilities are clamped to [0, 1].
    assert abs(noisy_or([1.5, -0.3]) - 1.0) < 1e-12


def test_fuse_confidence_bonuses_increase_score():
    base = fuse_confidence([0.6])
    with_bonus = fuse_confidence([0.6], cross_band_agreement=True, neupert_consistent=True)
    assert with_bonus > base
    assert 0.0 <= with_bonus <= 1.0
