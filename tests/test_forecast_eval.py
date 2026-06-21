"""Tests for forecast evaluation metrics, baselines, lead time, and models.

Covers ARCHITECTURE.md Section 10.2 / research doc 04 Section 5-6:

* TSS / HSS / BSS / POD / FAR formulas on **hand-checked confusion matrices**;
* a **perfect** predictor scores TSS = 1, a **random** predictor scores ~0;
* **AUC of a separable set ~ 1** (rank/trapezoid method, pure python);
* the mandatory baselines (climatology / persistence / Hawkes) behave sensibly;
* lead time = peak - first confirmed crossing, with k-of-m anti-flicker.

The metrics, baselines and lead-time tests are **pure python** (stdlib only).
The optional GBT/TCN model paths are guarded by ``pytest.importorskip`` so they
skip cleanly when lightgbm / torch are not installed.
"""

from __future__ import annotations

import random

import pytest
from flarecast.constants import N_FEATURES
from flarecast.forecast import evaluate as ev
from flarecast.forecast.baselines import (
    ClimatologyBaseline,
    HawkesBaseline,
    PersistenceBaseline,
)
from flarecast.forecast.leadtime import lead_time_list, lead_time_report, lt_vs_far


# ===========================================================================
# Confusion-matrix-based skill scores (hand-checked).
# ===========================================================================
def test_confusion_counts_hand_checked():
    # y_true / y_pred chosen so tp=2, fp=1, fn=1, tn=2.
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert ev.confusion_counts(y_true, y_pred) == (2, 1, 1, 2)


def test_tss_hand_checked():
    # tp=2 fp=1 fn=1 tn=2 -> TPR=2/3, FPR=1/3, TSS = 1/3.
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert ev.tss(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_pod_far_hand_checked():
    # POD = TP/(TP+FN) = 2/3 ; FAR = FP/(FP+TP) = 1/3.
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    p, f = ev.pod_far(y_true, y_pred)
    assert p == pytest.approx(2.0 / 3.0)
    assert f == pytest.approx(1.0 / 3.0)


def test_hss_hand_checked():
    # HSS = 2(TP*TN - FP*FN)/[(TP+FN)(FN+TN)+(TP+FP)(FP+TN)]
    #     = 2(2*2 - 1*1)/[(3)(3)+(3)(3)] = 2*3/18 = 1/3.
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert ev.hss(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_perfect_predictor_tss_is_one():
    y = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    assert ev.tss(y, y) == pytest.approx(1.0)
    assert ev.hss(y, y) == pytest.approx(1.0)
    p, f = ev.pod_far(y, y)
    assert p == pytest.approx(1.0)
    assert f == pytest.approx(0.0)


def test_inverted_predictor_tss_is_minus_one():
    y = [0, 0, 1, 1, 0, 1]
    y_inv = [1 - v for v in y]
    assert ev.tss(y, y_inv) == pytest.approx(-1.0)


def test_random_predictor_tss_near_zero():
    random.seed(7)
    n = 5000
    y = [1 if random.random() < 0.3 else 0 for _ in range(n)]
    yp = [1 if random.random() < 0.3 else 0 for _ in range(n)]
    assert abs(ev.tss(y, yp)) < 0.05


def test_far_is_verification_far_not_fpr():
    # FAR = FP/(FP+TP). With tp=1, fp=3 -> FAR = 3/4 (independent of TN).
    y_true = [1, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    assert ev.far(y_true, y_pred) == pytest.approx(3.0 / 4.0)


# ===========================================================================
# Probabilistic scores.
# ===========================================================================
def test_brier_score_hand_checked():
    # 0.5 prob everywhere vs labels [0,1] -> ((0.5)^2 + (0.5)^2)/2 = 0.25.
    assert ev.brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)


def test_bss_perfect_beats_climatology():
    y = [0, 0, 1, 1]
    p_perfect = [0.0, 0.0, 1.0, 1.0]
    assert ev.bss(y, p_perfect, p_clim=0.5) == pytest.approx(1.0)


def test_bss_climatology_against_itself_is_zero():
    y = [0, 0, 1, 1]
    base = 0.5
    p_clim_pred = [base] * 4
    assert ev.bss(y, p_clim_pred, p_clim=base) == pytest.approx(0.0)


def test_bss_negative_when_worse_than_climatology():
    y = [0, 0, 1, 1]
    # Anti-correlated probabilities -> worse than climatology -> BSS < 0.
    p_bad = [1.0, 1.0, 0.0, 0.0]
    assert ev.bss(y, p_bad, p_clim=0.5) < 0.0


# ===========================================================================
# AUCs (rank / trapezoid, pure python).
# ===========================================================================
def test_roc_auc_separable_is_one():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert ev.roc_auc(y, p) == pytest.approx(1.0)


def test_roc_auc_inverted_is_zero():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert ev.roc_auc(y, p) == pytest.approx(0.0)


def test_roc_auc_ties_give_half():
    # All identical scores -> no discrimination -> 0.5 (tie-aware rank stat).
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert ev.roc_auc(y, p) == pytest.approx(0.5)


def test_roc_auc_random_near_half():
    random.seed(11)
    n = 5000
    y = [1 if random.random() < 0.3 else 0 for _ in range(n)]
    p = [random.random() for _ in range(n)]
    assert ev.roc_auc(y, p) == pytest.approx(0.5, abs=0.05)


def test_roc_auc_single_class_returns_half():
    assert ev.roc_auc([1, 1, 1], [0.2, 0.5, 0.9]) == pytest.approx(0.5)
    assert ev.roc_auc([0, 0, 0], [0.2, 0.5, 0.9]) == pytest.approx(0.5)


def test_pr_auc_separable_is_one():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert ev.pr_auc(y, p) == pytest.approx(1.0, abs=1e-9)


def test_pr_auc_no_positives_is_zero():
    assert ev.pr_auc([0, 0, 0], [0.1, 0.2, 0.3]) == pytest.approx(0.0)


# ===========================================================================
# Calibration.
# ===========================================================================
def test_reliability_perfectly_calibrated_on_diagonal():
    # Two clusters: prob 0.0 with all-0 labels, prob 1.0 with all-1 labels.
    y = [0, 0, 0, 1, 1, 1]
    p = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    mean_pred, obs_freq = ev.reliability(y, p, n_bins=10)
    for mp, of in zip(mean_pred, obs_freq, strict=True):
        assert of == pytest.approx(mp, abs=1e-9)


def test_ece_zero_when_calibrated_one_when_anticalibrated():
    y = [0, 0, 1, 1]
    p_cal = [0.0, 0.0, 1.0, 1.0]
    assert ev.ece(y, p_cal) == pytest.approx(0.0)
    p_anti = [1.0, 1.0, 0.0, 0.0]
    assert ev.ece(y, p_anti) == pytest.approx(1.0)


# ===========================================================================
# Full report battery.
# ===========================================================================
def test_report_has_full_battery_and_picks_threshold():
    random.seed(3)
    n = 400
    y = [1 if random.random() < 0.25 else 0 for _ in range(n)]
    # Informative probability: correlated with the label.
    p = [min(0.99, max(0.01, 0.7 if yi else 0.2) + 0.1 * (random.random() - 0.5))
         for yi in y]
    rep = ev.report(y, p)
    for key in ("theta", "tp", "fp", "fn", "tn", "pod", "far", "tss", "hss",
                "bss", "brier", "roc_auc", "pr_auc", "ece", "n", "n_pos",
                "base_rate"):
        assert key in rep
    assert rep["n"] == n
    assert 0.0 <= rep["base_rate"] <= 1.0
    # Informative predictor should beat random discrimination.
    assert rep["roc_auc"] > 0.6


def test_report_relative_to_baselines():
    y = [0, 0, 1, 1, 0, 1, 0, 0]
    p_model = [0.1, 0.2, 0.8, 0.9, 0.1, 0.7, 0.2, 0.1]
    p_clim = [0.375] * len(y)  # base rate
    rep = ev.report(y, p_model, baselines={"climatology": p_clim})
    assert "climatology_tss" in rep
    assert "climatology_skill_vs" in rep
    # Model strictly better than climatology here -> positive skill.
    assert rep["climatology_skill_vs"] > 0.0


# ===========================================================================
# Baselines.
# ===========================================================================
def _row(**kw):
    r = [0.0] * N_FEATURES
    for idx, val in kw.items():
        r[int(idx)] = val
    return r


def test_climatology_predicts_base_rate():
    y = [0, 0, 1, 1, 0, 1]  # base rate 0.5
    clim = ClimatologyBaseline().fit(y)
    preds = clim.predict_proba([_row() for _ in range(4)])
    assert all(p == pytest.approx(0.5) for p in preds)


def test_climatology_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ClimatologyBaseline().predict_proba([_row()])


def test_persistence_flaring_now_high_quiet_low():
    pers = PersistenceBaseline()
    flaring = _row(**{"2": 2.0})   # S_over_baseline = 2.0 -> flaring
    quiet = _row(**{"2": 1.0})     # S_over_baseline = 1.0 -> quiet
    p = pers.predict_proba([flaring, quiet])
    assert p[0] > 0.9
    assert p[1] < 0.1


def test_decayed_persistence_grades_with_history():
    pers = PersistenceBaseline(decayed=True)
    much = _row(**{"25": 3.0})     # large decayed_flare_history
    little = _row(**{"25": 0.0})
    p = pers.predict_proba([much, little])
    assert p[0] > p[1]


def test_hawkes_probability_in_unit_interval_and_rises_with_history():
    hk = HawkesBaseline().fit([0.0, 3600.0, 7200.0, 10800.0])
    base = _row(**{"29": 30.0})                 # N_horizon = 30 min, no history
    excited = _row(**{"25": 5.0, "29": 30.0})   # recent decayed history
    p = hk.predict_proba([base, excited])
    assert all(0.0 < pi < 1.0 for pi in p)
    assert p[1] > p[0]


# ===========================================================================
# Lead time.
# ===========================================================================
def test_lead_time_equals_peak_minus_first_confirmed_crossing():
    t = [float(i) * 60.0 for i in range(20)]
    # Crosses theta=0.5 at step 8 (t=480) and stays high -> confirmed.
    probs = [0.1] * 8 + [0.6, 0.7] + [0.8] * 10
    peaks = [600.0]
    lt = lead_time_list(probs, t, peaks, theta=0.5)
    assert lt[0] == pytest.approx(600.0 - 480.0)  # 120 s


def test_lead_time_missed_when_never_crosses():
    t = [float(i) * 60.0 for i in range(20)]
    probs = [0.1] * 20
    peaks = [600.0]
    lt = lead_time_list(probs, t, peaks, theta=0.5)
    assert lt[0] is None


def test_lead_time_flicker_suppressed_by_k_of_m():
    """A single isolated spike (1 of 3) must NOT confirm an alert."""
    t = [float(i) * 60.0 for i in range(20)]
    # One lone spike at step 5 then quiet -> with k_of_m=(2,3) not confirmed.
    probs = [0.1] * 5 + [0.9] + [0.1] * 14
    peaks = [900.0]
    lt = lead_time_list(probs, t, peaks, theta=0.5, k_of_m=(2, 3))
    assert lt[0] is None
    # But a sustained crossing (2 of 3) confirms.
    probs2 = [0.1] * 5 + [0.9, 0.9] + [0.1] * 13
    lt2 = lead_time_list(probs2, t, peaks, theta=0.5, k_of_m=(2, 3))
    assert lt2[0] is not None


def test_lead_time_only_counts_pre_peak_window():
    """A crossing at/after the peak does not count as a lead-time alert."""
    t = [float(i) * 60.0 for i in range(20)]
    peak = 300.0  # step 5
    # Probability only rises AFTER the peak -> no pre-peak alert.
    probs = [0.1] * 6 + [0.9] * 14
    lt = lead_time_list(probs, t, [peak], theta=0.5)
    assert lt[0] is None


def test_lt_vs_far_monotone_trend_and_columns():
    """Lower theta -> earlier/more alerts (>= median LT, >= TPR)."""
    t = [float(i) * 60.0 for i in range(60)]
    # A slow ramp to 1.0 around a peak at step 50.
    probs = [min(1.0, i / 50.0) for i in range(60)]
    peaks = [3000.0]  # step 50
    sweep = lt_vs_far(probs, t, peaks, thetas=[0.2, 0.5, 0.8])
    # Normalize to a list of dicts whether pandas is present or not.
    rows = sweep.to_dict("records") if hasattr(sweep, "to_dict") else sweep
    for r in rows:
        assert set(r.keys()) >= {"theta", "median_lt", "tpr", "far"}
    # Lower theta -> not-smaller median lead time among detected peaks.
    by_theta = {r["theta"]: r for r in rows}
    if by_theta[0.2]["median_lt"] == by_theta[0.2]["median_lt"] and \
       by_theta[0.8]["median_lt"] == by_theta[0.8]["median_lt"]:
        assert by_theta[0.2]["median_lt"] >= by_theta[0.8]["median_lt"]


def test_lead_time_report_buckets():
    t = [float(i) * 60.0 for i in range(40)]
    # Crosses 20 min (1200 s) before a peak at step 30 (t=1800): cross at step10.
    probs = [0.1] * 10 + [0.9] * 30
    peaks = [1800.0]
    rep = lead_time_report(probs, t, peaks, theta=0.5)
    assert rep["n_hit"] == 1
    assert rep["n_miss"] == 0
    # LT = 1800 - 600 = 1200 s = 20 min -> >= 5,10,15 buckets but not >= 30.
    assert rep["frac_ge_min"][5] == pytest.approx(1.0)
    assert rep["frac_ge_min"][15] == pytest.approx(1.0)
    assert rep["frac_ge_min"][30] == pytest.approx(0.0)


# ===========================================================================
# Optional model backends (skipped cleanly when libs absent).
# ===========================================================================
def test_gbt_fit_predict_calibrate_with_backend():
    np = pytest.importorskip("numpy")
    # Skip unless at least one GBT backend is importable.
    have_lgb = True
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        have_lgb = False
    have_skl = True
    try:
        import sklearn  # noqa: F401
    except ImportError:
        have_skl = False
    if not (have_lgb or have_skl):
        pytest.skip("no GBT backend (lightgbm/sklearn) installed")

    from flarecast.forecast.model_gbt import GBTForecaster

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, N_FEATURES))
    # Label separable on feature index 2 (S_over_baseline).
    y = (X[:, 2] + 0.3 * rng.normal(size=n) > 0.5).astype(int)
    g = GBTForecaster(n_estimators=50).fit(X, y)
    p = g.predict_proba(X[:10])
    assert p.shape == (10,)
    assert ((p > 0.0) & (p < 1.0)).all()
    # Discrimination should be well above chance on a separable signal.
    assert ev.roc_auc(y.tolist(), g.predict_proba(X).tolist()) > 0.8
    # Calibration runs (requires sklearn isotonic).
    if have_skl:
        g.calibrate(X[:200], y[:200], method="isotonic")
        pc = g.predict_proba(X[:10])
        assert pc.shape == (10,)


def test_gbt_predict_before_fit_raises():
    np = pytest.importorskip("numpy")
    from flarecast.forecast.model_gbt import GBTForecaster

    g = GBTForecaster()
    with pytest.raises(RuntimeError):
        g.predict_proba(np.zeros((1, N_FEATURES)))


def test_gbt_save_load_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("no GBT backend installed")
    from flarecast.forecast.model_gbt import GBTForecaster

    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, N_FEATURES))
    y = (X[:, 2] > 0).astype(int)
    g = GBTForecaster(n_estimators=30).fit(X, y)
    path = str(tmp_path / "gbt.pkl")
    g.save(path)
    g2 = GBTForecaster.load(path)
    p1 = g.predict_proba(X[:5])
    p2 = g2.predict_proba(X[:5])
    assert np.allclose(p1, p2)


def test_tcn_fit_predict_with_torch():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from flarecast.constants import TCN_LOOKBACK_STEPS, TCN_N_CHANNELS
    from flarecast.forecast.model_tcn import TCNForecaster

    rng = np.random.default_rng(0)
    n = 32
    X = rng.normal(size=(n, TCN_LOOKBACK_STEPS, TCN_N_CHANNELS)).astype("float32")
    y = (rng.random(n) > 0.5).astype("float32")
    horizons = [5, 15, 30]
    tcn = TCNForecaster().fit(X, y, horizons=horizons, epochs=1, batch_size=8)
    p = tcn.predict_proba(X[:4])
    assert p.shape == (4, len(horizons))
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_focal_loss_runs_with_torch():
    pytest.importorskip("torch")
    import torch
    from flarecast.forecast.model_tcn import focal_loss_with_logits

    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = focal_loss_with_logits(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
