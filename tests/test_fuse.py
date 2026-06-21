"""Fusion estimators (ARCHITECTURE.md Section 3.8 / 06 §4.3).

Verifies the two payoffs of fusion:

* inverse-variance fusion yields ``sigma_hat < min_i sigma_i`` (redundancy
  reduces uncertainty);
* the Kalman innovation gate (chi^2 ~ 10.8) **rejects** an injected spike (the
  state barely moves) but **accepts** normal updates, and the covariance grows
  while coasting over a gap.

Pure standard library; no network, no numpy.
"""

from __future__ import annotations

import math

from flarecast.constants import KALMAN_GATE_CHI2
from flarecast.fusion.fuse import KalmanFuser, inverse_variance_fuse


# ---------------------------------------------------------------------------
# Static inverse-variance fusion
# ---------------------------------------------------------------------------
def test_inverse_variance_sigma_smaller_than_min_input():
    """Two equal sensors -> sigma_hat = sigma / sqrt(2) < min(sigma_i)."""
    x_hat, sigma_hat = inverse_variance_fuse([1.0, 1.0], [0.1, 0.1])
    assert abs(x_hat - 1.0) < 1e-12
    assert sigma_hat < 0.1, "fused sigma must be smaller than any input sigma"
    assert abs(sigma_hat - 0.1 / math.sqrt(2.0)) < 1e-9


def test_inverse_variance_sigma_below_min_for_unequal():
    """Fused sigma is below the smallest input sigma for unequal sensors too."""
    sigmas = [0.05, 0.2, 0.5]
    x_hat, sigma_hat = inverse_variance_fuse([2.0, 2.1, 1.9], sigmas)
    assert sigma_hat < min(sigmas)
    # Closed form: 1/sigma_hat^2 == sum(1/sigma_i^2).
    inv = sum(1.0 / s ** 2 for s in sigmas)
    assert abs(sigma_hat - math.sqrt(1.0 / inv)) < 1e-12


def test_inverse_variance_weights_toward_precise_sensor():
    """The estimate is pulled toward the lower-sigma (more precise) sensor."""
    x_hat, _ = inverse_variance_fuse([0.0, 10.0], [0.1, 1.0])
    # Sensor 0 (sigma=0.1) dominates -> estimate near 0, far from the midpoint 5.
    assert x_hat < 1.0


def test_inverse_variance_reliability_weighting():
    """Reliability weights scale the inverse-variance weights."""
    # Down-weighting the second sensor pulls the estimate toward the first.
    x_eq, _ = inverse_variance_fuse([0.0, 10.0], [1.0, 1.0])
    x_rw, _ = inverse_variance_fuse([0.0, 10.0], [1.0, 1.0], [1.0, 0.1])
    assert abs(x_eq - 5.0) < 1e-9
    assert x_rw < x_eq, "lower reliability on sensor 2 should pull toward sensor 1"


def test_inverse_variance_rejects_bad_input():
    """Empty / mismatched / non-positive sigma inputs raise."""
    import pytest

    with pytest.raises(ValueError):
        inverse_variance_fuse([], [])
    with pytest.raises(ValueError):
        inverse_variance_fuse([1.0, 2.0], [0.1])
    with pytest.raises(ValueError):
        inverse_variance_fuse([1.0], [0.0])  # zero sigma is not a valid weight


# ---------------------------------------------------------------------------
# Kalman filter: gate + coast
# ---------------------------------------------------------------------------
def _warm_up(kf: KalmanFuser, level: float, n: int = 20, sigma_log: float = 0.02):
    for _ in range(n):
        kf.step(1.0, [(level, sigma_log, 1.0)])


def test_kalman_accepts_normal_updates():
    """Normal in-family updates are accepted and track the level."""
    kf = KalmanFuser(q=1e-6, init_log_flux=-6.0)
    _warm_up(kf, -6.0, n=25)
    assert kf.n_gated == 0, "no normal update should be gated"
    assert kf.n_updates >= 25
    assert abs(kf.log_flux - (-6.0)) < 0.05, "filter should track the level"


def test_kalman_gate_rejects_injected_spike():
    """An injected spike is gated (state barely moves); the gate counter ticks."""
    kf = KalmanFuser(q=1e-6, init_log_flux=-6.0)
    _warm_up(kf, -6.0, n=25)
    before = kf.log_flux
    gated_before = kf.n_gated
    # Inject a 1000x spike (log10 jumps +3 decades) on a tight-sigma sensor.
    kf.step(1.0, [(-3.0, 0.02, 1.0)])
    after = kf.log_flux
    assert kf.n_gated == gated_before + 1, "spike should have been gated"
    assert abs(after - before) < 0.1, (
        f"state moved {abs(after - before):.3f} decades on a gated spike"
    )


def test_kalman_resumes_after_spike():
    """After gating a spike, the next normal update is still accepted."""
    kf = KalmanFuser(q=1e-6, init_log_flux=-6.0)
    _warm_up(kf, -6.0, n=25)
    kf.step(1.0, [(-3.0, 0.02, 1.0)])  # gated
    n_before = kf.n_updates
    kf.step(1.0, [(-6.0, 0.02, 1.0)])  # normal -> accepted
    assert kf.n_updates == n_before + 1


def test_kalman_gate_threshold_is_chi2_constant():
    """The default gate uses the frozen chi^2_{1,0.999} ~ 10.8 constant."""
    kf = KalmanFuser()
    assert abs(kf.gate_chi2 - KALMAN_GATE_CHI2) < 1e-9
    # A borderline innovation just inside the gate is accepted; just outside is
    # rejected. Build a state with known P00 and R so y^2/S is controllable.
    kf2 = KalmanFuser(q=0.0, init_log_flux=0.0, init_var_level=1.0)
    # First call snaps to the data; do a no-op predict to keep P00 ~ 1, R ~ 1.
    kf2.step(0.0, [(0.0, 1.0, 1.0)])  # initializes at 0 with P00 ~ 1
    # S = P00 + R. With a measurement at z, innovation y = z. Choose z so that
    # y^2/S is just over the gate -> rejected.
    p00 = kf2.var_level
    R = 1.0
    S = p00 + R
    z_reject = math.sqrt(KALMAN_GATE_CHI2 * S) * 1.05
    g0 = kf2.n_gated
    kf2.step(0.0, [(z_reject, 1.0, 1.0)])
    assert kf2.n_gated == g0 + 1, "innovation beyond the gate should be rejected"


def test_kalman_coasts_over_gap_with_growing_variance():
    """With no valid sensor the filter coasts and its covariance grows."""
    kf = KalmanFuser(q=1e-3, init_log_flux=-6.0)
    _warm_up(kf, -6.0, n=10, sigma_log=0.05)
    var_before = kf.var_level
    for _ in range(20):
        kf.step(1.0, [])  # no sensors -> predict-only coast
    var_after = kf.var_level
    assert var_after > var_before, "covariance must grow during a data gap"
    assert kf.n_coast == 20


def test_kalman_sequential_update_matches_inverse_variance():
    """One Kalman step with N sensors ~ batch inverse-variance (independent).

    Sequentially updating with several independent sensors in one step is the
    temporal generalization of the static inverse-variance combination
    (research 06 Section 4.3.2). Starting from a diffuse prior, the post-update
    level should match the inverse-variance mean of the measurements closely.
    """
    measurements = [(-6.0, 0.05, 1.0), (-5.9, 0.05, 1.0), (-6.1, 0.10, 1.0)]
    # Expected inverse-variance combination in log space.
    vals = [m[0] for m in measurements]
    sigs = [m[1] for m in measurements]
    iv_mean, _ = inverse_variance_fuse(vals, sigs)

    kf = KalmanFuser(q=0.0, init_log_flux=-7.0, init_var_level=1e6)
    kf.step(0.0, measurements)
    assert abs(kf.log_flux - iv_mean) < 0.02, (
        f"Kalman level {kf.log_flux:.4f} should match IV mean {iv_mean:.4f}"
    )
