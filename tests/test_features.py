"""Tests for the forecast feature extractor (Workstream 4).

Covers the two contractual invariants from ARCHITECTURE.md / research doc 04:

* the streaming feature vector has length exactly
  :data:`flarecast.constants.N_FEATURES`, and
* **causality / no leakage** -- a *future* spike must never change a feature
  value emitted for an *earlier* step (research doc 04 Section 2.1; the
  documented leakage failure mode for sliding-window forecasting).

These tests are **pure python** (stdlib only); the numpy batch wrappers are
exercised in a separate test guarded by ``pytest.importorskip("numpy")`` so the
file runs (and the rest passes) with no optional dependencies installed.
"""

from __future__ import annotations

import math

import pytest
from flarecast.constants import N_FEATURES, TCN_LOOKBACK_STEPS, TCN_N_CHANNELS
from flarecast.forecast.features import FEATURE_NAMES, FeatureExtractor


# ---------------------------------------------------------------------------
# Inline synthetic helpers (no WS1/WS3 dependency).
# ---------------------------------------------------------------------------
def _quiet_then_flare(n: int = 24, spike_at: int | None = None):
    """Return (sxr, hxr, t) lists: quiet baseline with an optional flare spike.

    A quiet soft baseline of 1e-6 W/m^2 and hard rate of 5 c/s; if ``spike_at``
    is given, inject a sharp soft+hard enhancement at that step (a flare).
    """
    sxr = [1e-6] * n
    hxr = [5.0] * n
    t = [float(i) * 60.0 for i in range(n)]
    if spike_at is not None:
        sxr[spike_at] = 1e-4
        hxr[spike_at] = 500.0
    return sxr, hxr, t


# ---------------------------------------------------------------------------
# 1. Vector length == N_FEATURES.
# ---------------------------------------------------------------------------
def test_feature_names_length_matches_contract():
    assert len(FEATURE_NAMES) == N_FEATURES
    # Names are unique (no accidental duplicate dimension).
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_streaming_vector_length_is_n_features():
    ex = FeatureExtractor(cadence_s=60.0)
    vec = ex.update(sxr=1e-6, hxr=10.0, t=0.0, n_horizon=30.0)
    assert isinstance(vec, list)
    assert len(vec) == N_FEATURES
    # Every entry is a finite float (defensive guarantee for downstream models).
    assert all(isinstance(v, float) and math.isfinite(v) for v in vec)


def test_vector_length_stable_across_many_updates():
    ex = FeatureExtractor(cadence_s=60.0)
    sxr, hxr, t = _quiet_then_flare(n=50, spike_at=20)
    for i in range(50):
        vec = ex.update(sxr[i], hxr[i], t[i], n_horizon=15.0)
        assert len(vec) == N_FEATURES
        assert all(math.isfinite(v) for v in vec)


def test_horizon_is_passed_through_as_last_feature():
    # Feature #30 (index 29) is N_horizon, emitted verbatim (research 04 §5.2).
    ex = FeatureExtractor()
    vec = ex.update(1e-6, 5.0, 0.0, n_horizon=42.0)
    assert FEATURE_NAMES[-1] == "N_horizon"
    assert vec[-1] == 42.0


# ---------------------------------------------------------------------------
# 2. CAUSALITY: a future spike does not change a past feature value.
# ---------------------------------------------------------------------------
def test_future_spike_does_not_change_past_features():
    """Two streams identical up to step k; one gets a huge spike *after* k.

    The first k feature rows must be bit-identical between the two streams: a
    causal extractor cannot see the future. This is the core no-leakage test.
    """
    k = 10
    n = 20
    # Stream A stays quiet throughout.
    sxrA, hxrA, t = _quiet_then_flare(n=n, spike_at=None)
    # Stream B is identical for steps 0..k-1, then spikes at step k+2.
    sxrB, hxrB, _ = _quiet_then_flare(n=n, spike_at=k + 2)

    exA = FeatureExtractor()
    exB = FeatureExtractor()
    rowsA = [exA.update(sxrA[i], hxrA[i], t[i], 30.0) for i in range(n)]
    rowsB = [exB.update(sxrB[i], hxrB[i], t[i], 30.0) for i in range(n)]

    # Past (steps before the spike, here all of 0..k inclusive since the spike
    # is at k+2) must be identical.
    for i in range(k + 1):
        for j in range(N_FEATURES):
            assert rowsA[i][j] == rowsB[i][j], (
                f"causality violated at step {i}, feature {FEATURE_NAMES[j]}: "
                f"{rowsA[i][j]} != {rowsB[i][j]}"
            )


def test_spike_does_influence_subsequent_features():
    """Sanity counterpart: the extractor is *not* inert -- a spike DOES change
    later feature rows (otherwise the causality test would pass trivially)."""
    n = 20
    spike = 8
    sxrA, hxrA, t = _quiet_then_flare(n=n, spike_at=None)
    sxrB, hxrB, _ = _quiet_then_flare(n=n, spike_at=spike)
    exA = FeatureExtractor()
    exB = FeatureExtractor()
    rowsA = [exA.update(sxrA[i], hxrA[i], t[i], 30.0) for i in range(n)]
    rowsB = [exB.update(sxrB[i], hxrB[i], t[i], 30.0) for i in range(n)]
    # At/after the spike the rows must differ on at least one feature.
    assert any(
        abs(rowsA[spike][j] - rowsB[spike][j]) > 1e-9 for j in range(N_FEATURES)
    )


def test_neupert_features_present_and_ordered():
    # The three Neupert features occupy indices 16-18 in the §5.2 order.
    assert FEATURE_NAMES[16] == "neupert_corr"
    assert FEATURE_NAMES[17] == "neupert_resid"
    assert FEATURE_NAMES[18] == "hxr_leads_flag"


def test_neupert_residual_spikes_at_impulsive_onset():
    """The leaky-integrator Neupert residual (index 17) should be larger during
    a sharp soft rise than during quiet times (it measures deviation from the
    HXR-integral coupling)."""
    n = 30
    sxr = [1e-6] * n
    hxr = [5.0] * n
    t = [float(i) * 60.0 for i in range(n)]
    # Sharp soft rise with no matching hard injection -> integrator under-
    # predicts -> residual grows.
    for i in range(15, 20):
        sxr[i] = 1e-5
    ex = FeatureExtractor()
    resids = []
    for i in range(n):
        vec = ex.update(sxr[i], hxr[i], t[i], 30.0)
        resids.append(vec[17])
    quiet_resid = max(resids[:10])
    onset_resid = max(resids[15:22])
    assert onset_resid > quiet_resid


def test_irregular_cadence_is_handled():
    # Non-uniform timestamps must not crash and must stay finite (dt is read
    # from the timestamps, falling back to the nominal cadence on the first
    # sample / non-positive gaps).
    ex = FeatureExtractor(cadence_s=60.0)
    ts = [0.0, 30.0, 90.0, 300.0, 305.0]
    for ti in ts:
        vec = ex.update(1e-6, 5.0, ti, 30.0)
        assert len(vec) == N_FEATURES
        assert all(math.isfinite(v) for v in vec)


def test_flare_history_drives_self_excitation_features():
    """Supplying recent flare onsets should raise the Hawkes-style history
    features (flares_last_6h #24, decayed_flare_history #25) above zero."""
    ex = FeatureExtractor()
    now = 10_000.0
    history = [now - 600.0, now - 1800.0]  # two flares in the last 30 min
    vec = ex.update(1e-6, 5.0, now, 30.0, flare_history=history)
    assert vec[24] == 2.0          # flares_last_6h
    assert vec[25] > 0.0           # decayed_flare_history
    assert vec[23] <= 10.0         # time_since_last_flare (min) ~ 10 min, capped


# ---------------------------------------------------------------------------
# 3. numpy batch wrappers (skipped cleanly when numpy is absent).
# ---------------------------------------------------------------------------
def test_extract_features_batch_shape_and_equivalence():
    np = pytest.importorskip("numpy")
    from flarecast.forecast.features import extract_features

    n = 25
    sxr = [1e-6 * (1.0 + 0.1 * i) for i in range(n)]
    hxr = [5.0 + i for i in range(n)]
    t = [float(i) * 60.0 for i in range(n)]
    window = {"t": t, "sxr": sxr, "hxr": hxr}

    batch = extract_features(window, n_horizon=30.0)
    assert batch.shape == (N_FEATURES,)
    assert batch.dtype == np.float64

    # Batch wrapper == final streaming update on the same input (bit-identical).
    ex = FeatureExtractor()
    last = None
    for i in range(n):
        last = ex.update(sxr[i], hxr[i], t[i], 30.0)
    assert np.allclose(batch, np.asarray(last))


def test_build_tcn_tensor_shape():
    pytest.importorskip("numpy")
    from flarecast.forecast.features import build_tcn_tensor

    n = 40
    window = {
        "t": [float(i) * 60.0 for i in range(n)],
        "sxr": [1e-6] * n,
        "hxr": [5.0] * n,
    }
    tensor = build_tcn_tensor(window)
    assert tensor.shape == (TCN_LOOKBACK_STEPS, TCN_N_CHANNELS)


def test_build_tcn_tensor_front_padded_when_short():
    np = pytest.importorskip("numpy")
    from flarecast.forecast.features import build_tcn_tensor

    n = 5  # shorter than lookback -> front-padded with zeros.
    window = {
        "t": [float(i) * 60.0 for i in range(n)],
        "sxr": [1e-6] * n,
        "hxr": [5.0] * n,
    }
    tensor = build_tcn_tensor(window)
    assert tensor.shape == (TCN_LOOKBACK_STEPS, TCN_N_CHANNELS)
    # The leading rows are the zero pad; the last n rows carry data.
    assert np.allclose(tensor[: TCN_LOOKBACK_STEPS - n], 0.0)
