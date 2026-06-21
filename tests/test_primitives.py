"""Tests for O(1) streaming primitives (ARCHITECTURE.md B.4 / research 03 Section 1).

Pure standard library: correctness checks against :mod:`statistics` plus explicit
verification that each primitive's state stays **constant-size** as the stream
grows (the O(1)-state contract). No numpy, no network, no ``flarecast.synth``.
"""

from __future__ import annotations

import random
import statistics

from flarecast.detect.primitives import (
    EMA,
    EWMV,
    HampelDespiker,
    P2Quantile,
    RingBuffer,
    SlopeEstimator,
    Welford,
)


def _gaussian(n, mu=10.0, sigma=2.0, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def test_ema_converges_to_constant():
    ema = EMA(alpha=0.9, x0=0.0)
    for _ in range(500):
        ema.update(5.0)
    assert abs(ema.value - 5.0) < 1e-6


def test_ema_bias_correction_helps_cold_start():
    ema = EMA(alpha=0.95, x0=0.0)
    ema.update(10.0)
    # Raw EMA underestimates after one sample; bias-corrected is exact here.
    assert ema.value < 10.0
    assert abs(ema.value_bias_corrected() - 10.0) < 1e-9


def test_ema_rejects_bad_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            EMA(bad)
        except ValueError:
            continue
        raise AssertionError(f"EMA accepted invalid alpha {bad}")


# ---------------------------------------------------------------------------
# EWMV
# ---------------------------------------------------------------------------
def test_ewmv_tracks_mean_and_variance():
    data = _gaussian(5000, mu=10.0, sigma=2.0, seed=1)
    ewmv = EWMV(alpha=0.99, x0=data[0])
    for x in data:
        ewmv.update(x)
    assert abs(ewmv.mean() - 10.0) < 0.5
    # EW sd should be in the right ballpark of the true sigma=2.
    assert 1.0 < ewmv.sd() < 3.0


# ---------------------------------------------------------------------------
# Welford
# ---------------------------------------------------------------------------
def test_welford_matches_statistics_exactly():
    data = _gaussian(2000, mu=-3.0, sigma=5.0, seed=2)
    w = Welford()
    for x in data:
        w.update(x)
    assert abs(w.mean() - statistics.mean(data)) < 1e-9
    assert abs(w.variance() - statistics.variance(data)) < 1e-6


def test_welford_variance_zero_for_single_sample():
    w = Welford()
    w.update(42.0)
    assert w.variance() == 0.0
    assert w.mean() == 42.0


# ---------------------------------------------------------------------------
# P2Quantile
# ---------------------------------------------------------------------------
def test_p2_median_accurate():
    data = _gaussian(5000, mu=10.0, sigma=2.0, seed=3)
    p2 = P2Quantile(0.5)
    for x in data:
        p2.update(x)
    true_med = statistics.median(data)
    assert abs(p2.value - true_med) / true_med < 0.02  # within 2%


def test_p2_high_quantile_accurate():
    data = _gaussian(20000, mu=0.0, sigma=1.0, seed=4)
    p2 = P2Quantile(0.9)
    for x in data:
        p2.update(x)
    s = sorted(data)
    true_q = s[int(0.9 * len(s))]
    assert abs(p2.value - true_q) < 0.1


def test_p2_rejects_bad_q():
    for bad in (0.0, 1.0, -0.5, 2.0):
        try:
            P2Quantile(bad)
        except ValueError:
            continue
        raise AssertionError(f"P2Quantile accepted invalid q {bad}")


# ---------------------------------------------------------------------------
# HampelDespiker
# ---------------------------------------------------------------------------
def test_hampel_flags_and_replaces_spike():
    hd = HampelDespiker(window=7, k=3.0)
    sig = [10.0] * 20
    sig[10] = 10000.0  # a single huge spike
    flags = []
    cleaned = []
    for x in sig:
        cv, is_out = hd.update(x)
        flags.append(is_out)
        cleaned.append(cv)
    assert flags[10] is True
    # The despiked replacement is the window median (~10), not the spike.
    assert abs(cleaned[10] - 10.0) < 1e-6
    # A flat inlier is not flagged.
    assert flags[5] is False


def test_hampel_passes_clean_signal():
    # A window large enough for a stable MAD estimate (small windows bias MAD
    # low and over-flag) flags only a small fraction of a clean Gaussian stream.
    hd = HampelDespiker(window=21, k=3.0)
    data = _gaussian(5000, mu=100.0, sigma=1.0, seed=5)
    n_flagged = sum(1 for x in data if hd.update(x)[1])
    # At k=3 sigma a clean Gaussian stream should flag only a few percent.
    assert n_flagged < 0.05 * len(data)


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------
def test_ring_buffer_fixed_capacity():
    rb = RingBuffer(3)
    for x in [1, 2, 3, 4, 5]:
        rb.push(x)
    assert list(rb.last(3)) == [3.0, 4.0, 5.0]
    assert len(rb) == 3
    assert rb.is_full


def test_ring_buffer_partial():
    rb = RingBuffer(5)
    rb.push(1.0)
    rb.push(2.0)
    assert rb.last(5) == [1.0, 2.0]
    assert not rb.is_full


# ---------------------------------------------------------------------------
# SlopeEstimator
# ---------------------------------------------------------------------------
def test_slope_estimator_exact_on_line():
    se = SlopeEstimator(window=10)
    slope = 0.0
    for t in range(50):
        slope = se.update(2.0 * t + 1.0, float(t))
    assert abs(slope - 2.0) < 1e-9


def test_slope_estimator_negative_slope():
    se = SlopeEstimator(window=8)
    slope = 0.0
    for t in range(40):
        slope = se.update(-3.0 * t + 7.0, float(t))
    assert abs(slope - (-3.0)) < 1e-9


# ---------------------------------------------------------------------------
# O(1) STATE INVARIANT -- the heart of the contract
# ---------------------------------------------------------------------------
def _state_signature(obj) -> tuple[int, int]:
    """Return ``(n_scalar_slots, total_container_length)`` for a primitive.

    The O(1)-state contract means the number of stored attributes is fixed and
    every contained collection (deque/list) is bounded by a constant capacity --
    so this signature must be *identical* regardless of how many samples have
    streamed through. (We deliberately do not use ``sys.getsizeof`` on the whole
    object: CPython's container over-allocation makes byte counts noisy; the
    honest invariant is that container *lengths* never grow with the stream.)
    """
    n_scalar = 0
    total_len = 0
    for name in getattr(type(obj), "__slots__", ()):  # noqa: SLF001 (introspection)
        try:
            v = getattr(obj, name)
        except AttributeError:
            continue
        n_scalar += 1
        try:
            total_len += len(v)
        except TypeError:
            pass
    return n_scalar, total_len


def test_primitive_state_is_constant_size():
    """State signature after 100 samples must equal that after 100k samples."""
    factories = [
        lambda: EMA(0.9),
        lambda: EWMV(0.9),
        lambda: Welford(),
        lambda: P2Quantile(0.5),
        lambda: HampelDespiker(7),
        lambda: SlopeEstimator(10),
        lambda: RingBuffer(64),
    ]
    rng = random.Random(7)
    for make in factories:
        a = make()
        b = make()
        for _ in range(100):
            _update_any(a, rng)
        for _ in range(100_000):
            _update_any(b, rng)
        sa = _state_signature(a)
        sb = _state_signature(b)
        assert sa == sb, (
            f"{type(a).__name__} state signature grew from {sa} to {sb} -> not O(1) state"
        )
        # Every contained collection must be bounded by a small constant, not by
        # the 100k samples streamed.
        assert sb[1] < 1000, f"{type(a).__name__} retains {sb[1]} elements"


def _update_any(obj, rng) -> None:
    """Call the right ``update``/``push`` signature for any primitive."""
    x = rng.gauss(10.0, 2.0)
    if isinstance(obj, SlopeEstimator):
        obj.update(x, rng.random() * 1000.0)
    elif isinstance(obj, RingBuffer):
        obj.push(x)
    else:
        obj.update(x)
