"""Cross-calibration transfer functions (ARCHITECTURE.md Section 3.7).

Different instruments report different numbers for the same photons. To assign
standard GOES A-X classes and to fuse counts from different HXR detectors, we
put everyone on a common scale via a log-space transfer function fit on the
overlap set of jointly observed flares (research doc ``06 Section 3``)::

    log10 y_ref = a + b * log10 x_inst + eps

with ``b`` the gain (ideally ~1), ``a`` the bias/offset (ideally ~0), and
``eps`` the scatter that becomes the post-cal transfer residual added in
quadrature to each source's statistical sigma -- exactly what the fusion stage
consumes as a weight.

* :func:`fit_transfer_function` -- robust (IRLS / Huber) log-space straight-line
  fit; returns ``{a, b, resid, ...}``. ``tests/test_xcal.py`` asserts it
  recovers ``b ~ 1`` and ``a ~ 0`` on a synthetic overlapping series.
* :func:`solexs_to_goes` -- apply the SoLEXS -> GOES W m^-2 transfer.
* :func:`hxr_to_reference` -- apply the HEL1OS/GBM -> STIX reference-band
  transfer, chained per instrument.

Implemented in **pure standard library** least squares (the published anchor is
``a ~ 0, b ~ 1``), so it is fully testable offline. numpy is not imported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "fit_transfer_function",
    "apply_transfer_function",
    "solexs_to_goes",
    "hxr_to_reference",
    "DEFAULT_SOLEXS_GOES_COEFFS",
    "DEFAULT_HXR_REF_COEFFS",
]

# Published cross-cal anchor: SoLEXS vs GOES-XRS / Chandrayaan-2-XSM agree
# within ~10-15%, anchoring a ~ 0, b ~ 1 in log space (research 06 Section 3.1).
DEFAULT_SOLEXS_GOES_COEFFS: dict[str, float] = {"a": 0.0, "b": 1.0, "resid": 0.06}

# HXR pairwise transfers chained to STIX as reference (GBM all-sky anchor).
# Per-instrument {a, b, resid} in log10 space, identity for STIX itself.
DEFAULT_HXR_REF_COEFFS: dict[str, dict[str, float]] = {
    "STIX": {"a": 0.0, "b": 1.0, "resid": 0.0},
    "HEL1OS": {"a": 0.0, "b": 1.0, "resid": 0.10},
    "GBM": {"a": 0.0, "b": 1.0, "resid": 0.08},
    "KONUS": {"a": 0.0, "b": 1.0, "resid": 0.12},
}

_LOG_FLOOR = 1e-300  # clamp before log10 so non-positive inputs do not blow up


def _safe_log10(x: float) -> float:
    return math.log10(max(float(x), _LOG_FLOOR))


def fit_transfer_function(
    x: Sequence[float],
    y: Sequence[float],
    robust: bool = True,
) -> dict:
    """Fit ``log10 y = a + b * log10 x`` (robust by default).

    Implements Appendix B.3 ``xcal.fit_transfer_function``. Both axes are
    log10-transformed (flux is log-distributed). With ``robust=True`` the fit is
    iteratively reweighted least squares with Huber weights, so a few saturated
    or particle-contaminated points cannot tilt the slope (research 06
    Section 3.1 recommends Huber / Theil-Sen / ODR). With ``robust=False`` it is
    ordinary least squares.

    Parameters
    ----------
    x:
        Instrument values (e.g. SoLEXS-band-synthesized flux, must be > 0).
    y:
        Reference values (e.g. GOES 1-8 A flux, must be > 0).
    robust:
        Use Huber IRLS (default) instead of plain OLS.

    Returns
    -------
    dict
        ``{"a", "b", "resid", "n", "rms_log", "r2"}`` where ``a`` is the offset,
        ``b`` the slope/gain, ``resid`` the robust 1-sigma residual in log10
        (== ``rms_log`` for OLS), ``n`` the point count, and ``r2`` the
        coefficient of determination in log space.

    Raises
    ------
    ValueError
        If fewer than two points are supplied or lengths mismatch.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("fit_transfer_function: x/y length mismatch")
    if n < 2:
        raise ValueError("fit_transfer_function: need >= 2 points")

    lx = [_safe_log10(v) for v in x]
    ly = [_safe_log10(v) for v in y]

    a, b = _wls_line(lx, ly, [1.0] * n)

    if robust:
        # IRLS with Huber weights; a handful of passes converges for a line.
        for _ in range(10):
            resid = [ly[i] - (a + b * lx[i]) for i in range(n)]
            scale = _mad_scale(resid)
            if scale <= 0.0:
                break
            weights = [_huber_weight(r / scale) for r in resid]
            a_new, b_new = _wls_line(lx, ly, weights)
            if abs(a_new - a) < 1e-9 and abs(b_new - b) < 1e-9:
                a, b = a_new, b_new
                break
            a, b = a_new, b_new

    # Residual statistics (unweighted, on the final line).
    resid = [ly[i] - (a + b * lx[i]) for i in range(n)]
    rms_log = math.sqrt(sum(r * r for r in resid) / n)
    robust_sigma = _mad_scale(resid) if robust else rms_log
    if robust_sigma <= 0.0:
        robust_sigma = rms_log

    # R^2 in log space.
    mean_y = sum(ly) / n
    ss_tot = sum((v - mean_y) ** 2 for v in ly)
    ss_res = sum(r * r for r in resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return {
        "a": a,
        "b": b,
        "resid": robust_sigma,
        "rms_log": rms_log,
        "n": n,
        "r2": r2,
    }


def apply_transfer_function(
    value_native: float, coeffs: dict
) -> tuple[float, float]:
    """Apply a fitted log-space transfer to one value.

    ``log10 y = a + b * log10 x`` -> ``y = 10 ** (a + b log10 x)``. The returned
    sigma is the transfer residual propagated into linear space::

        sigma_y = ln(10) * y * resid

    Returns ``(value_ref, sigma_ref)``.
    """
    a = coeffs.get("a", 0.0)
    b = coeffs.get("b", 1.0)
    resid = coeffs.get("resid", 0.0)
    log_y = a + b * _safe_log10(value_native)
    y = 10.0 ** log_y
    sigma_y = math.log(10.0) * y * resid
    return y, sigma_y


def solexs_to_goes(
    value_native: float,
    hardness: float | None = None,
    coeffs: dict | None = None,
) -> tuple[float, float]:
    """SoLEXS (GOES-band-synthesized flux) -> GOES W m^-2 + sigma.

    Implements Appendix B.3 ``xcal.solexs_to_goes``. Step A (synthesize the GOES
    1-8 A band from the SoLEXS photon spectrum) is assumed done upstream; this
    is Step B, the empirical bias/gain transfer in log space. ``coeffs`` may be
    a fitted ``{a, b, resid}`` from :func:`fit_transfer_function`; if ``None``
    the published anchor ``a~0, b~1`` is used.

    ``hardness`` (optional spectral-hardness proxy) is accepted for API
    completeness and, when provided, applies a small first-order spectral
    correction to the offset (harder spectra deposit relatively less in the
    soft GOES band). Returns ``(flux_W_m2, sigma_W_m2)`` on the GOES scale.
    """
    cfs = dict(coeffs) if coeffs is not None else dict(DEFAULT_SOLEXS_GOES_COEFFS)
    if hardness is not None:
        # First-order, bounded spectral nudge to the offset; harmless at the
        # nominal hardness of 1.0 (no shift) and never flips the gain.
        cfs["a"] = cfs.get("a", 0.0) - 0.05 * (float(hardness) - 1.0)
    # Output is on the canonical SXR scale (UNIT_SXR == "W m^-2").
    flux, sigma = apply_transfer_function(value_native, cfs)
    return flux, sigma


def hxr_to_reference(
    value_native: float,
    instrument: str,
    coeffs: dict | None = None,
) -> tuple[float, float]:
    """HXR counts/s -> common 25-50 keV reference-band counts/s + sigma.

    Implements Appendix B.3 ``xcal.hxr_to_reference``. Each instrument
    (HEL1OS, GBM, Konus, ...) is mapped onto the STIX reference band via its
    pairwise log-space transfer (research 06 Section 3.2). ``coeffs`` overrides
    the per-instrument default; ``instrument`` selects the default entry from
    :data:`DEFAULT_HXR_REF_COEFFS` (case-insensitive). Unknown instruments fall
    back to identity (with a conservative residual) rather than raising, so the
    pipeline degrades gracefully.

    Returns ``(counts_ref, sigma_ref)``.
    """
    if coeffs is None:
        key = instrument.strip().upper()
        cfs = DEFAULT_HXR_REF_COEFFS.get(
            key, {"a": 0.0, "b": 1.0, "resid": 0.15}
        )
    else:
        cfs = coeffs
    # Output is on the canonical HXR scale (UNIT_HXR == "counts/s").
    counts, sigma = apply_transfer_function(value_native, cfs)
    return counts, sigma


# ---------------------------------------------------------------------------
# Pure-python weighted line fit + robust helpers
# ---------------------------------------------------------------------------
def _wls_line(
    x: Sequence[float], y: Sequence[float], w: Sequence[float]
) -> tuple[float, float]:
    """Weighted least-squares fit of ``y = a + b x``; returns ``(a, b)``."""
    sw = sum(w)
    if sw <= 0.0:
        return 0.0, 1.0
    sx = sum(wi * xi for wi, xi in zip(w, x, strict=True))
    sy = sum(wi * yi for wi, yi in zip(w, y, strict=True))
    sxx = sum(wi * xi * xi for wi, xi in zip(w, x, strict=True))
    sxy = sum(wi * xi * yi for wi, xi, yi in zip(w, x, y, strict=True))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-30:
        # Degenerate (all x equal): fall back to unit slope through the mean.
        return (sy / sw), 1.0
    b = (sw * sxy - sx * sy) / denom
    a = (sy - b * sx) / sw
    return a, b


def _mad_scale(resid: Sequence[float]) -> float:
    """Robust scale estimate ~ 1-sigma via the median absolute deviation."""
    n = len(resid)
    if n == 0:
        return 0.0
    srt = sorted(resid)
    med = _median_sorted(srt)
    abs_dev = sorted(abs(r - med) for r in resid)
    mad = _median_sorted(abs_dev)
    return 1.4826 * mad  # consistent with Gaussian sigma


def _median_sorted(srt: Sequence[float]) -> float:
    n = len(srt)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return srt[mid]
    return 0.5 * (srt[mid - 1] + srt[mid])


def _huber_weight(t: float, c: float = 1.345) -> float:
    """Huber weight for a standardized residual ``t`` (psi(t)/t)."""
    at = abs(t)
    if at <= c:
        return 1.0
    return c / at
