"""Cross-calibration transfer functions (ARCHITECTURE.md Section 3.7 / 06 §3).

Verifies the log-space transfer fit recovers slope ~ 1 and offset ~ 0 on a
synthetic overlapping series (the published SoLEXS<->GOES<->XSM anchor is
``a ~ 0, b ~ 1``), is robust to outliers, and that the apply helpers round-trip.

Pure standard library; no network, no numpy.
"""

from __future__ import annotations

import math

from flarecast.fusion.xcal import (
    apply_transfer_function,
    fit_transfer_function,
    hxr_to_reference,
    solexs_to_goes,
)


def _synthetic_overlap(a_true: float, b_true: float, n: int = 40):
    """Build x (instrument) and y = 10**(a + b log10 x) reference series."""
    xs = [10.0 ** (-7.0 + 4.0 * i / (n - 1)) for i in range(n)]  # 1e-7 .. 1e-3
    ys = [10.0 ** (a_true + b_true * math.log10(x)) for x in xs]
    return xs, ys


def test_identity_recovers_slope_one_offset_zero():
    """y == x -> b ~ 1, a ~ 0 within tolerance."""
    xs, ys = _synthetic_overlap(0.0, 1.0)
    fit = fit_transfer_function(xs, ys, robust=True)
    assert abs(fit["b"] - 1.0) < 1e-3, f"slope {fit['b']} not ~ 1"
    assert abs(fit["a"] - 0.0) < 1e-3, f"offset {fit['a']} not ~ 0"
    assert fit["resid"] < 1e-3
    assert fit["r2"] > 0.999
    assert fit["n"] == len(xs)


def test_known_gain_and_offset_recovered():
    """A planted (a=0.3, b=0.9) transfer is recovered within tolerance."""
    xs, ys = _synthetic_overlap(0.3, 0.9)
    fit = fit_transfer_function(xs, ys, robust=False)
    assert abs(fit["b"] - 0.9) < 1e-2, f"slope {fit['b']} not ~ 0.9"
    assert abs(fit["a"] - 0.3) < 1e-2, f"offset {fit['a']} not ~ 0.3"


def test_robust_fit_resists_outliers():
    """A few corrupted points do not tilt the robust slope away from ~1."""
    xs, ys = _synthetic_overlap(0.0, 1.0, n=40)
    # Corrupt 4 points with multi-decade spikes (saturation / particle hits).
    for i in (5, 15, 25, 35):
        ys[i] *= 1000.0
    robust = fit_transfer_function(xs, ys, robust=True)
    ols = fit_transfer_function(xs, ys, robust=False)
    # Robust slope stays near 1; OLS is dragged noticeably off.
    assert abs(robust["b"] - 1.0) < 0.1, f"robust slope {robust['b']} drifted"
    assert abs(robust["b"] - 1.0) < abs(ols["b"] - 1.0), (
        "robust fit should beat OLS under outliers"
    )


def test_apply_transfer_function_roundtrip():
    """Applying an identity transfer returns the input value unchanged."""
    val, sigma = apply_transfer_function(2.5e-5, {"a": 0.0, "b": 1.0, "resid": 0.0})
    assert abs(val - 2.5e-5) / 2.5e-5 < 1e-9
    assert sigma == 0.0
    # A residual produces a positive linear-space sigma proportional to value.
    val2, sigma2 = apply_transfer_function(2.5e-5, {"a": 0.0, "b": 1.0, "resid": 0.1})
    assert sigma2 > 0.0
    assert abs(sigma2 - math.log(10.0) * val2 * 0.1) < 1e-12


def test_solexs_to_goes_default_anchor():
    """The default SoLEXS->GOES transfer is ~ identity (published anchor)."""
    flux, sigma = solexs_to_goes(2.0e-5)
    assert abs(flux - 2.0e-5) / 2.0e-5 < 1e-6, "default transfer not ~ identity"
    assert sigma > 0.0  # carries the transfer residual
    # Fitted coeffs override the default.
    xs, ys = _synthetic_overlap(0.0, 1.0)
    fit = fit_transfer_function(xs, ys)
    flux2, _ = solexs_to_goes(2.0e-5, coeffs=fit)
    assert abs(flux2 - 2.0e-5) / 2.0e-5 < 1e-3


def test_hxr_to_reference_per_instrument():
    """HXR transfer is ~ identity for STIX (the reference) and defined for others."""
    counts_stix, sigma_stix = hxr_to_reference(500.0, "STIX")
    assert abs(counts_stix - 500.0) < 1e-6
    assert sigma_stix == 0.0  # STIX is the reference -> zero transfer residual
    counts_h, sigma_h = hxr_to_reference(500.0, "HEL1OS")
    assert abs(counts_h - 500.0) / 500.0 < 1e-6  # default gain ~ 1
    assert sigma_h > 0.0  # but carries a transfer residual
    # Unknown instrument falls back to identity (graceful), not an exception.
    counts_u, sigma_u = hxr_to_reference(500.0, "MysteryDetector")
    assert abs(counts_u - 500.0) / 500.0 < 1e-6
    assert sigma_u > 0.0


def test_fit_requires_two_points():
    """Too few points raises a clear error."""
    import pytest

    with pytest.raises(ValueError):
        fit_transfer_function([1.0], [1.0])
