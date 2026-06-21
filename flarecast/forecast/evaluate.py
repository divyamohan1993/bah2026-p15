"""Pure-python evaluation metrics for rare-event flare forecasting.

Governing research: ``docs/research/04-forecasting-models.md`` Section 6
(*Evaluation Metrics for Rare Events*) and ARCHITECTURE.md Section 10.2.

Flares are rare, so accuracy / ROC-AUC mislead; the flare-forecasting standard
battery is implemented here (Appendix B.6)::

    tss, hss, bss, pod_far, reliability, report

plus :func:`roc_auc` and :func:`pr_auc` (computed with the **rank / trapezoid**
method so the whole module is **pure standard library** -- ``math`` only, no
numpy/sklearn) and :func:`ece` (expected calibration error). ``report``
runs the full battery and, when baseline probabilities are supplied, reports
skill relative to them.

Conventions. ``y_true`` is a sequence of 0/1 labels. ``y_pred`` for the
threshold metrics is a sequence of 0/1 decisions; ``p_pred`` for the
probabilistic metrics is a sequence of probabilities in ``[0, 1]``. Inputs may
be Python lists or numpy arrays (both iterate the same way); outputs are plain
``float`` / ``dict`` so nothing here requires numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "confusion_counts",
    "tss",
    "hss",
    "bss",
    "pod",
    "far",
    "pod_far",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "reliability",
    "ece",
    "report",
]


def _as_int_list(y: Any) -> list[int]:
    return [1 if float(v) >= 0.5 else 0 for v in y]


def _as_float_list(p: Any) -> list[float]:
    return [float(v) for v in p]


# ---------------------------------------------------------------------------
# Confusion matrix.
# ---------------------------------------------------------------------------
def confusion_counts(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` from 0/1 truth and 0/1 predictions."""
    yt = _as_int_list(y_true)
    yp = _as_int_list(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"length mismatch: y_true={len(yt)} y_pred={len(yp)}")
    tp = fp = fn = tn = 0
    for a, b in zip(yt, yp, strict=True):
        if a == 1 and b == 1:
            tp += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 1 and b == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Threshold (deterministic) skill scores.
# ---------------------------------------------------------------------------
def pod(y_true: Any, y_pred: Any) -> float:
    """Probability Of Detection = TPR = recall = TP / (TP + FN)."""
    tp, _fp, fn, _tn = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else 0.0


def far(y_true: Any, y_pred: Any) -> float:
    """False Alarm Ratio = FP / (FP + TP) (the problem's FAR axis, §10.1).

    NOTE: this is the *forecast-verification* FAR = FP/(FP+TP) (1 - precision),
    as defined in ARCHITECTURE.md Section 6 / 10.1 -- **not** the false-positive
    rate FP/(FP+TN). The probability-of-false-detection FP/(FP+TN) used by TSS
    is computed separately inside :func:`tss`.
    """
    tp, fp, _fn, _tn = confusion_counts(y_true, y_pred)
    denom = fp + tp
    return fp / denom if denom else 0.0


def pod_far(y_true: Any, y_pred: Any) -> tuple[float, float]:
    """Return ``(POD, FAR)`` -- the operational detection / false-alarm pair."""
    return pod(y_true, y_pred), far(y_true, y_pred)


def tss(y_true: Any, y_pred: Any) -> float:
    """True Skill Statistic (Hanssen-Kuipers) = TPR - FPR = POD - POFD.

    The **primary** flare-forecasting metric (research doc 04 Section 6):
    insensitive to class ratio, range ``[-1, 1]``. A perfect predictor scores
    ``1`` (TPR=1, FPR=0); a predictor uncorrelated with truth scores ``~0``.
    """
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr - fpr


def hss(y_true: Any, y_pred: Any) -> float:
    """Heidke Skill Score: skill vs a random forecast with the same marginals.

    ``HSS = 2(TP*TN - FP*FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]``. Range
    ``(-inf, 1]``; 0 = no skill over random, 1 = perfect. Ratio-sensitive
    (reported alongside TSS, research doc 04 Section 6).
    """
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denom == 0:
        return 0.0
    return 2.0 * (tp * tn - fp * fn) / denom


# ---------------------------------------------------------------------------
# Probabilistic scores.
# ---------------------------------------------------------------------------
def brier_score(y_true: Any, p_pred: Any) -> float:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    yt = [float(v) for v in y_true]
    pp = _as_float_list(p_pred)
    if len(yt) != len(pp):
        raise ValueError(f"length mismatch: y_true={len(yt)} p_pred={len(pp)}")
    if not yt:
        return 0.0
    return sum((p - y) ** 2 for y, p in zip(yt, pp, strict=True)) / len(yt)


def bss(y_true: Any, p_pred: Any, p_clim: float) -> float:
    """Brier Skill Score = ``1 - BS / BS_climatology`` (vs constant ``p_clim``).

    Probabilistic skill **versus climatology** (research doc 04 Section 6):
    ``> 0`` means the forecast beats always predicting the base rate ``p_clim``.
    Decomposes into reliability + resolution - uncertainty; here we report the
    scalar skill score.
    """
    yt = [float(v) for v in y_true]
    if not yt:
        return 0.0
    bs = brier_score(yt, p_pred)
    bs_clim = sum((p_clim - y) ** 2 for y in yt) / len(yt)
    if bs_clim <= 0:
        # Degenerate climatology (all-0 or all-1 truth with matching p_clim).
        return 0.0
    return 1.0 - bs / bs_clim


# ---------------------------------------------------------------------------
# Threshold-free discrimination (rank / trapezoid AUC -- pure python).
# ---------------------------------------------------------------------------
def roc_auc(y_true: Any, p_pred: Any) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (pure python, tie-aware).

    ``AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``
    using average ranks for ties -- algebraically identical to the trapezoidal
    area under the ROC curve, with no numpy. Returns ``0.5`` when one class is
    absent (undefined discrimination). A perfectly separable set returns ``1``.
    """
    yt = _as_int_list(y_true)
    pp = _as_float_list(p_pred)
    if len(yt) != len(pp):
        raise ValueError(f"length mismatch: y_true={len(yt)} p_pred={len(pp)}")
    n_pos = sum(yt)
    n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks (1-based) of scores, ties share the mean rank.
    order = sorted(range(len(pp)), key=lambda i: pp[i])
    ranks = [0.0] * len(pp)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pp[order[j + 1]] == pp[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average of positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(ranks[i] for i in range(len(yt)) if yt[i] == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def pr_auc(y_true: Any, p_pred: Any) -> float:
    """Precision-Recall AUC (average precision; trapezoid over recall steps).

    More informative than ROC under heavy imbalance (research doc 04 Section 6:
    focuses on the positive/flare class). Computed by sweeping every distinct
    score threshold high->low, accumulating precision/recall, and integrating
    precision over recall with the trapezoid rule. Pure python.
    """
    yt = _as_int_list(y_true)
    pp = _as_float_list(p_pred)
    if len(yt) != len(pp):
        raise ValueError(f"length mismatch: y_true={len(yt)} p_pred={len(pp)}")
    n_pos = sum(yt)
    if n_pos == 0:
        return 0.0

    # Sort by descending score; walk thresholds, emitting (recall, precision).
    order = sorted(range(len(pp)), key=lambda i: pp[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    prev_precision = 1.0  # precision at recall 0 is conventionally 1.
    i = 0
    while i < len(order):
        # Advance through all samples sharing the current score (a single
        # threshold yields one operating point).
        j = i
        while j + 1 < len(order) and pp[order[j + 1]] == pp[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if yt[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        # Trapezoid between the previous and current recall.
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall = recall
        prev_precision = precision
        i = j + 1
    return max(0.0, min(1.0, area))


# ---------------------------------------------------------------------------
# Calibration.
# ---------------------------------------------------------------------------
def reliability(y_true: Any, p_pred: Any, n_bins: int = 10):
    """Reliability-diagram points: (mean predicted prob, observed frequency).

    Bins predictions into ``n_bins`` equal-width probability bins on ``[0, 1]``
    and returns, for each non-empty bin, the mean predicted probability and the
    observed positive frequency. A well-calibrated forecaster lies on the
    diagonal (research doc 04 Section 6; required since the deliverable is a
    probability).

    Returns
    -------
    (mean_pred, obs_freq):
        Two equal-length ``list[float]`` (one entry per non-empty bin), ordered
        by increasing probability. (The contract types these as arrays; lists
        are returned so the module stays numpy-free -- numpy arrays accept lists
        directly.)
    """
    yt = [float(v) for v in y_true]
    pp = _as_float_list(p_pred)
    if len(yt) != len(pp):
        raise ValueError(f"length mismatch: y_true={len(yt)} p_pred={len(pp)}")
    n_bins = max(1, int(n_bins))
    sums_p = [0.0] * n_bins
    sums_y = [0.0] * n_bins
    counts = [0] * n_bins
    for y, p in zip(yt, pp, strict=True):
        p = min(1.0, max(0.0, p))
        b = min(n_bins - 1, int(p * n_bins))
        sums_p[b] += p
        sums_y[b] += y
        counts[b] += 1
    mean_pred: list[float] = []
    obs_freq: list[float] = []
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        mean_pred.append(sums_p[b] / counts[b])
        obs_freq.append(sums_y[b] / counts[b])
    return mean_pred, obs_freq


def ece(y_true: Any, p_pred: Any, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |conf - accuracy|.

    Summary scalar of the reliability diagram (research doc 04 Section 6). 0 =
    perfectly calibrated.
    """
    yt = [float(v) for v in y_true]
    pp = _as_float_list(p_pred)
    if len(yt) != len(pp):
        raise ValueError(f"length mismatch: y_true={len(yt)} p_pred={len(pp)}")
    n = len(yt)
    if n == 0:
        return 0.0
    n_bins = max(1, int(n_bins))
    sums_p = [0.0] * n_bins
    sums_y = [0.0] * n_bins
    counts = [0] * n_bins
    for y, p in zip(yt, pp, strict=True):
        p = min(1.0, max(0.0, p))
        b = min(n_bins - 1, int(p * n_bins))
        sums_p[b] += p
        sums_y[b] += y
        counts[b] += 1
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        conf = sums_p[b] / counts[b]
        acc = sums_y[b] / counts[b]
        total += (counts[b] / n) * abs(conf - acc)
    return total


# ---------------------------------------------------------------------------
# Operating-point selection + full report.
# ---------------------------------------------------------------------------
def best_threshold_by_tss(y_true: Any, p_pred: Any) -> tuple[float, float]:
    """Return ``(theta, tss)`` maximizing TSS over candidate thresholds.

    Sweeps every distinct predicted-probability value as a candidate threshold
    (decision ``p >= theta``) and returns the threshold with the highest TSS --
    the TSS-max operating-point rule (research doc 04 Section 6).
    """
    pp = _as_float_list(p_pred)
    if not pp:
        return 0.5, 0.0
    candidates = sorted(set(pp))
    # Include a threshold just above the max so the all-negative point is seen.
    candidates.append(max(pp) + 1e-9)
    best_theta = candidates[0]
    best_score = -2.0
    for theta in candidates:
        preds = [1 if p >= theta else 0 for p in pp]
        score = tss(y_true, preds)
        if score > best_score:
            best_score = score
            best_theta = theta
    return best_theta, best_score


def report(y_true: Any, p_pred: Any, theta: float | None = None,
           p_clim: float | None = None,
           baselines: dict[str, Sequence[float]] | None = None) -> dict:
    """Run the full metric battery and return a dictionary of results.

    Parameters
    ----------
    y_true:
        0/1 labels.
    p_pred:
        Predicted probabilities in ``[0, 1]``.
    theta:
        Decision threshold for the deterministic metrics (POD/FAR/TSS/HSS and
        the confusion matrix). If ``None``, the TSS-maximizing threshold is
        chosen automatically and reported under ``"theta"``.
    p_clim:
        Climatology probability for BSS. If ``None``, the empirical base rate
        ``mean(y_true)`` is used.
    baselines:
        Optional mapping ``name -> baseline_probabilities`` (same length as
        ``p_pred``). For each, the report adds ``"<name>_bss"`` (skill of the
        *model* vs that baseline's Brier score) and the baseline's own TSS at
        its TSS-max threshold, so the headline numbers are always shown
        relative to persistence / climatology / Hawkes (research doc 04 §6).

    Returns
    -------
    dict
        Keys: ``theta, tp, fp, fn, tn, pod, far, tss, hss, bss, brier,
        roc_auc, pr_auc, ece, n, n_pos, base_rate`` and any baseline-relative
        entries.
    """
    yt = [float(v) for v in y_true]
    pp = _as_float_list(p_pred)
    n = len(yt)
    n_pos = int(sum(1 for v in yt if v >= 0.5))
    base_rate = (n_pos / n) if n else 0.0

    if theta is None:
        theta, _ = best_threshold_by_tss(yt, pp)
    preds = [1 if p >= theta else 0 for p in pp]
    tp, fp, fn, tn = confusion_counts(yt, preds)

    if p_clim is None:
        p_clim = min(1.0 - 1e-6, max(1e-6, base_rate))

    out: dict[str, Any] = {
        "theta": float(theta),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pod": pod(yt, preds),
        "far": far(yt, preds),
        "tss": tss(yt, preds),
        "hss": hss(yt, preds),
        "bss": bss(yt, pp, p_clim),
        "brier": brier_score(yt, pp),
        "roc_auc": roc_auc(yt, pp),
        "pr_auc": pr_auc(yt, pp),
        "ece": ece(yt, pp),
        "n": n,
        "n_pos": n_pos,
        "base_rate": base_rate,
    }

    if baselines:
        bs_model = brier_score(yt, pp)
        for name, bp in baselines.items():
            bp_list = _as_float_list(bp)
            if len(bp_list) != n:
                continue
            bs_base = brier_score(yt, bp_list)
            # Skill of the model relative to this baseline (1 - BS_model/BS_base).
            skill = (1.0 - bs_model / bs_base) if bs_base > 0 else 0.0
            b_theta, b_tss = best_threshold_by_tss(yt, bp_list)
            out[f"{name}_brier"] = bs_base
            out[f"{name}_skill_vs"] = skill
            out[f"{name}_tss"] = b_tss
            out[f"{name}_theta"] = b_theta
    return out
