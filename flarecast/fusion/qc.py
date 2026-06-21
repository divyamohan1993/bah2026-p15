"""Quality control: despike, housekeeping gating, spectral test, bitmask.

Per-sample, per-source QC (ARCHITECTURE.md Section 3.11, research doc
``06 Section 7``). Fusion is only as trustworthy as its flags, so QC travels
with the data as a :class:`~flarecast.types.QCBit` bitmask (conditions coexist,
e.g. ``FILLED | NEAR_SAA``). Multi-stage particle-hit rejection:

1. **housekeeping gating** -- drop samples flagged by orbit/HK as SAA, eclipse,
   off-point, attenuator transitions (:func:`housekeeping_gate`);
2. **iterative bidirectional median sigma-clip** -- mark a sample exceeding the
   local median by > k*sigma, excluding already-marked spikes each pass
   (:func:`median_sigma_despike`);
3. **X-ray spectral-shape test** -- a true flare brightens coherently across the
   band; a particle hit deposits in a few channels with a non-solar signature
   (:func:`spectral_shape_test`);
4. cross-sensor veto (the strongest test) is implemented elsewhere by the Kalman
   innovation gate (:mod:`flarecast.fusion.fuse`) + IPN (:mod:`.stereo`).

:func:`qc_bitmask` assembles the integer bitmask from flag tokens.

Pure standard library (a numpy fast path is provided where helpful but never
required); fully testable offline.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..constants import (
    QC_BAD,
    QC_FILLED,
    QC_GOOD,
    QC_INTERPOLATED,
    QC_NEAR_SAA,
    QC_SATURATED,
    QC_SPIKE_REJECTED,
    QC_SUSPECT,
    SDD_SATURATION_CPS,
)
from ..types import QCBit

__all__ = [
    "median_sigma_despike",
    "spectral_shape_test",
    "qc_bitmask",
    "housekeeping_gate",
    "FLAG_TO_BIT",
]

# Map flag tokens (case-insensitive) -> QCBit integer values. Accepts both the
# QCFlag-style names and the specific-hazard names from the DDL comment in
# ARCHITECTURE.md Section 9.4.
FLAG_TO_BIT: dict[str, int] = {
    "good": QC_GOOD,
    "interpolated": QC_INTERPOLATED,
    "interp": QC_INTERPOLATED,
    "filled": QC_FILLED,
    "suspect": QC_SUSPECT,
    "bad": QC_BAD,
    "near_saa": QC_NEAR_SAA,
    "saa": QC_NEAR_SAA,
    "saturated": QC_SATURATED,
    "saturation": QC_SATURATED,
    "data_gap": QCBit.DATA_GAP.value,
    "gap": QCBit.DATA_GAP.value,
    "spike_rejected": QC_SPIKE_REJECTED,
    "spike": QC_SPIKE_REJECTED,
}


def qc_bitmask(flags: Sequence[str]) -> int:
    """Assemble a QC bitmask integer from a list of flag tokens.

    Implements Appendix B.3 ``qc.qc_bitmask``. Tokens are matched
    case-insensitively against :data:`FLAG_TO_BIT`; unknown tokens are ignored
    (defensive). Conditions coexist via bitwise OR, e.g.
    ``qc_bitmask(["FILLED", "near_SAA"]) == QC_FILLED | QC_NEAR_SAA``.
    """
    mask = 0
    for f in flags:
        bit = FLAG_TO_BIT.get(str(f).strip().lower())
        if bit is not None:
            mask |= bit
    return mask


def median_sigma_despike(
    x: Sequence[float],
    k: float = 8.0,
    max_iter: int = 3,
) -> tuple[list[float], list[bool]]:
    """Iterative bidirectional median sigma-clip despiker.

    Implements Appendix B.3 ``qc.median_sigma_despike``. Each pass computes the
    global median and a MAD-based robust sigma over the *not-yet-marked*
    samples, then marks any sample exceeding the median by more than ``k`` sigma
    (research 06 Section 7.3 uses 5-11 sigma; default 8). Marked spikes are
    excluded from the statistics on subsequent passes, and replaced in the
    returned clean series by the running median so they cannot contaminate
    downstream smoothing.

    A single-sample multi-decade jump inconsistent with flare rise times is a
    particle spike, not a flare -- the width-gate despiker in the detector stack
    handles the complementary "is it sustained?" test.

    Parameters
    ----------
    x:
        Input series.
    k:
        Sigma multiplier for the clip threshold.
    max_iter:
        Maximum clipping passes.

    Returns
    -------
    (clean, mask)
        ``clean`` is the despiked series (spikes replaced by the local median);
        ``mask[i]`` is ``True`` where sample ``i`` was flagged a spike.
    """
    n = len(x)
    vals = [float(v) for v in x]
    mask = [False] * n
    if n == 0:
        return vals, mask

    for _ in range(max(1, max_iter)):
        kept = [vals[i] for i in range(n) if not mask[i]]
        if len(kept) < 3:
            break
        med = _median(kept)
        sigma = _mad_sigma(kept, med)
        if sigma <= 0.0:
            # Degenerate case: the bulk of the data is identical (MAD == 0) so
            # the robust scale collapses. Fall back to the mean absolute
            # deviation, which is still non-zero when a lone spike exists, so an
            # obvious outlier in otherwise-flat data is not silently kept.
            mad_mean = sum(abs(v - med) for v in kept) / len(kept)
            if mad_mean <= 0.0:
                break  # genuinely constant data -> no spikes possible
            sigma = 1.2533 * mad_mean  # ~ Gaussian sigma from mean abs dev
        thr = k * sigma
        changed = False
        for i in range(n):
            if mask[i]:
                continue
            if abs(vals[i] - med) > thr:
                mask[i] = True
                changed = True
        if not changed:
            break

    # Build the clean series: spikes -> running median of recent good values.
    clean = list(vals)
    last_good = _median([vals[i] for i in range(n) if not mask[i]]) if any(
        not m for m in mask
    ) else (vals[0] if vals else 0.0)
    for i in range(n):
        if mask[i]:
            clean[i] = last_good
        else:
            last_good = vals[i]
    return clean, mask


def spectral_shape_test(
    spectrum: Sequence[float],
    response: Sequence[float],
    min_corr: float = 0.3,
    max_single_channel_frac: float = 0.8,
) -> bool:
    """X-ray spectral-shape consistency test (solar vs particle hit).

    Implements Appendix B.3 ``qc.spectral_shape_test``. Returns ``True`` if the
    measured ``spectrum`` is **consistent with a solar X-ray flare** (a coherent
    brightening across the band shaped like the instrument ``response``), and
    ``False`` if it looks like a particle hit (energy dumped into one/few
    channels with a non-solar signature) -- research 06 Section 7.3.

    Two cheap, robust diagnostics:

    * **band coherence**: the (positive) correlation between the measured
      spectrum shape and the expected response shape must exceed ``min_corr``;
    * **single-channel dominance**: no single channel may hold more than
      ``max_single_channel_frac`` of the total counts (a particle hit spikes one
      channel).

    Parameters
    ----------
    spectrum:
        Measured counts per energy channel.
    response:
        Expected per-channel response/shape for a nominal flare spectrum.
    min_corr:
        Minimum Pearson correlation (shape match) to accept.
    max_single_channel_frac:
        Maximum fraction of total counts allowed in any single channel.

    Returns
    -------
    bool
        ``True`` = solar-consistent (keep); ``False`` = reject as non-solar.
    """
    n = len(spectrum)
    if n == 0 or len(response) != n:
        # Cannot test -> do not reject on this basis (defensive: let other QC
        # stages decide).
        return True
    total = sum(max(v, 0.0) for v in spectrum)
    if total <= 0.0:
        return False  # no counts at all is not a solar brightening
    # Single-channel dominance.
    if max(max(v, 0.0) for v in spectrum) / total > max_single_channel_frac:
        return False
    # Shape coherence (Pearson correlation between spectrum and response).
    corr = _pearson(spectrum, response)
    return corr >= min_corr


def housekeeping_gate(
    value: float,
    *,
    in_saa: bool = False,
    in_eclipse: bool = False,
    off_point: bool = False,
    attenuator_moving: bool = False,
    saturation_ceiling: float = SDD_SATURATION_CPS,
) -> tuple[str, int]:
    """Gate one sample on housekeeping/orbit state -> ``(qc_flag, bitmask)``.

    Stage 1 of despiking (research 06 Section 7.3): samples flagged by
    orbit/housekeeping as SAA, eclipse, off-point, or during attenuator
    transitions are excluded (``BAD``), and a value pinned at the detector
    ceiling is marked ``SUSPECT | SATURATED`` (a very large flare on the wrong
    gain state -- do not read the paralyzable turnover as a flux drop).

    Returns the human-readable :class:`~flarecast.types.QCFlag`-style token and
    the coexisting-conditions integer bitmask.
    """
    flags: list[str] = []
    bad = False
    if in_saa:
        flags.append("near_saa")
        bad = True
    if in_eclipse or off_point or attenuator_moving:
        bad = True
    saturated = value >= saturation_ceiling
    if saturated:
        flags.append("saturated")

    if bad:
        flags.append("bad")
        return "BAD", qc_bitmask(flags)
    if saturated:
        flags.append("suspect")
        return "SUSPECT", qc_bitmask(flags)
    flags.append("good")
    return "GOOD", qc_bitmask(flags)


# ---------------------------------------------------------------------------
# Small statistics helpers (pure python)
# ---------------------------------------------------------------------------
def _median(seq: Sequence[float]) -> float:
    s = sorted(seq)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad_sigma(seq: Sequence[float], med: float) -> float:
    abs_dev = [abs(v - med) for v in seq]
    return 1.4826 * _median(abs_dev)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    denom = math.sqrt(va * vb)
    if denom <= 0.0:
        return 0.0
    return cov / denom
