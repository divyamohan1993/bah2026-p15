"""Consensus labeling across catalogues (ARCHITECTURE.md Section 3.9 / 06 §6).

Verifies weighted voting: a 2-of-3 catalogue agreement is **confirmed** while a
lone candidate is **candidate**, association is LTT-aware, an HXR-only event is
kept confirmed with ``sxr_weak``, and thresholds come from the ``CONSENSUS_*``
constants.

Pure standard library; no network, no numpy.
"""

from __future__ import annotations

from flarecast.constants import (
    CONSENSUS_CANDIDATE_THRESH,
    CONSENSUS_CONFIRM_THRESH,
)
from flarecast.fusion.consensus import CatalogueEntry, consensus_label


def test_two_of_three_agreement_confirmed_lone_candidate():
    """2-of-3 catalogues agreeing -> confirmed; a lone entry -> candidate.

    Uses the majority/weighted setup where two equally-reliable catalogues clear
    the confirm threshold (research 06 Section 6.3: majority is the unweighted
    special case at tau_hi=0.5).
    """
    rho = {"GOES": 1.0, "STIX": 1.0, "GBM": 1.0}
    # Two catalogues report the same physical flare (peaks within tolerance).
    goes = [CatalogueEntry(catalog="GOES", t_peak=1000.0, goes_class="M2.5")]
    stix = [CatalogueEntry(catalog="STIX", t_peak=1010.0, is_hxr=True)]
    # A third catalogue reports an unrelated, far-apart lone event.
    gbm_lone = [CatalogueEntry(catalog="GBM", t_peak=9000.0, is_hxr=True)]

    labels = consensus_label(
        [goes, stix, gbm_lone], rho, tau_hi=0.5, tau_lo=0.25
    )
    by_time = {round(L.t_peak): L for L in labels}

    confirmed = by_time[1005]
    assert confirmed.state == "confirmed", f"got {confirmed.state}"
    assert set(confirmed.voters) == {"GOES", "STIX"}
    assert confirmed.confidence >= 0.5
    assert confirmed.goes_class == "M2.5"  # GOES class preferred

    lone = by_time[9000]
    assert lone.state == "candidate", f"got {lone.state}"
    assert lone.voters == ["GBM"]
    assert lone.confidence < CONSENSUS_CONFIRM_THRESH


def test_weighted_confidence_uses_reliabilities():
    """conf = sum(rho voters) / sum(rho all); thresholds default to CONSENSUS_*."""
    rho = {"GOES": 0.98, "STIX": 0.95, "GBM": 0.90}  # total 2.83
    goes = [CatalogueEntry(catalog="GOES", t_peak=500.0, goes_class="C5.0")]
    stix = [CatalogueEntry(catalog="STIX", t_peak=505.0, is_hxr=True)]
    labels = consensus_label([goes, stix], rho)  # default tau_hi=0.7
    assert len(labels) == 1
    conf = labels[0].confidence
    assert abs(conf - (0.98 + 0.95) / 2.83) < 1e-9
    # Default tau_hi == the frozen confirm threshold.
    assert CONSENSUS_CONFIRM_THRESH == 0.7
    assert CONSENSUS_CANDIDATE_THRESH == 0.3


def test_lone_low_reliability_rejected():
    """A single low-reliability voter below tau_lo is rejected."""
    rho = {"GOES": 0.98, "STIX": 0.95, "ECALLISTO": 0.2}
    lone = [CatalogueEntry(catalog="ECALLISTO", t_peak=42.0)]
    labels = consensus_label([[], [], lone], rho)
    assert len(labels) == 1
    # 0.2 / 2.13 ~ 0.094 < tau_lo (0.3) -> rejected.
    assert labels[0].state == "rejected"


def test_hxr_only_event_kept_confirmed_sxr_weak():
    """HXR-present / SXR-absent -> confirmed with sxr_weak (forecast signal)."""
    rho = {"GOES": 0.98, "STIX": 0.95, "GBM": 0.90}
    # Only STIX reports it, and only in HXR (no soft-band counterpart).
    stix = [CatalogueEntry(catalog="STIX", t_peak=700.0, is_hxr=True, is_sxr=False)]
    labels = consensus_label([[], stix, []], rho, tau_lo=0.3, tau_hi=0.7)
    assert len(labels) == 1
    lab = labels[0]
    assert lab.sxr_weak is True
    # Even though lone (conf ~ 0.336 >= tau_lo), the HXR-only conflict rule keeps
    # it confirmed rather than candidate (research 06 Section 6.3).
    assert lab.state == "confirmed"


def test_association_is_ltt_aware():
    """Entries from different vantages associate after LTT correction.

    A STIX entry timestamped ~240 s *earlier* than GOES (because Solar Orbiter
    saw the flare earlier) should align with the GOES entry once both are
    LTT-corrected to the Earth frame, forming a single confirmed event.
    """
    rho = {"GOES": 1.0, "STIX": 1.0, "GBM": 1.0}
    t_goes = 10_000.0
    # STIX at ~0.5 AU sees it ~+249 s earlier in raw time; store that raw time.
    t_stix_raw = t_goes - 249.5
    goes = [CatalogueEntry(catalog="GOES", t_peak=t_goes, vantage_r_au=1.0,
                           goes_class="M1.0")]
    stix = [CatalogueEntry(catalog="STIX", t_peak=t_stix_raw, vantage_r_au=0.5,
                           is_hxr=True)]
    labels = consensus_label([goes, stix], rho, tau_hi=0.5, peak_tol_s=180.0)
    # Without LTT they'd be ~249 s apart (> 180 s tol) -> two events; with LTT
    # they collapse into one confirmed event.
    assert len(labels) == 1, "LTT-aware association should merge the two entries"
    assert labels[0].state == "confirmed"
    assert set(labels[0].voters) == {"GOES", "STIX"}


def test_candidate_promoted_by_own_detection():
    """A candidate is promoted to confirmed when our own fused detection agrees."""
    rho = {"GOES": 0.98, "STIX": 0.95, "GBM": 0.90}  # total 2.83
    lone = [CatalogueEntry(catalog="GBM", t_peak=2000.0, is_hxr=True, is_sxr=True)]
    # 0.90/2.83 ~ 0.318 -> candidate by threshold.
    labels = consensus_label(
        [[], [], lone], rho, own_detection_times=[2010.0]
    )
    assert len(labels) == 1
    assert labels[0].state == "confirmed", "own detection should promote candidate"
