"""Noisy-OR confidence fusion across detectors / bands (Section 4.6, B.5).

The master-catalogue confidence combines the firing detectors into one score via
a **noisy-OR** (research doc ``03 Section 4.6``)::

    confidence = 1 - prod_d (1 - p_d)

where ``p_d`` is each detector's calibrated detection probability. Independent
pieces of evidence reinforce each other: any one strong detector already gives
high confidence, and agreeing detectors push it toward 1. Cross-band agreement
and Neupert consistency are added as bonus evidence terms by the caller.

``noisy_or`` is O(#detectors) = O(1) for the fixed, small detector set.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["noisy_or", "fuse_confidence"]


def noisy_or(probs: list[float]) -> float:
    """Noisy-OR combination ``1 - prod(1 - p)`` of detection probabilities.

    Each ``p`` is clamped to ``[0, 1]``. An empty list returns ``0.0`` (no
    evidence). O(len(probs)).
    """
    prod = 1.0
    any_seen = False
    for p in probs:
        any_seen = True
        pc = 0.0 if p < 0.0 else (1.0 if p > 1.0 else float(p))
        prod *= 1.0 - pc
    if not any_seen:
        return 0.0
    return 1.0 - prod


def fuse_confidence(
    detector_probs: list[float],
    *,
    cross_band_agreement: bool = False,
    neupert_consistent: bool = False,
    agreement_bonus: float = 0.5,
    neupert_bonus: float = 0.5,
    extra: Iterable[float] | None = None,
) -> float:
    """Fuse detector probabilities plus physical-prior bonuses (Section 4.6).

    Builds the full evidence list -- the per-detector probabilities, an optional
    cross-band-agreement term, an optional Neupert-consistency term, and any
    ``extra`` evidence -- and combines them with :func:`noisy_or`. The bonuses
    enter as additional noisy-OR evidence (so they can only *increase*
    confidence, never push it down), reflecting that two-band agreement and a
    consistent Neupert lead are independent corroborations. Returns a value in
    ``[0, 1]``. O(1) for the fixed detector set.

    Parameters
    ----------
    detector_probs:
        Calibrated per-detector detection probabilities.
    cross_band_agreement:
        True if both soft and hard bands detected the event.
    neupert_consistent:
        True if the HXR-leads-SXR Neupert ordering held.
    agreement_bonus, neupert_bonus:
        Evidence weights (in ``[0, 1]``) contributed by the two priors.
    extra:
        Any additional evidence probabilities to fold in.
    """
    evidence = list(detector_probs)
    if cross_band_agreement:
        evidence.append(agreement_bonus)
    if neupert_consistent:
        evidence.append(neupert_bonus)
    if extra is not None:
        evidence.extend(extra)
    return noisy_or(evidence)
