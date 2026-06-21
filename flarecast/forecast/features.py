"""Streaming + batch feature extraction for the forecast model.

Governing research: ``docs/research/04-forecasting-models.md`` Section 2
(*Engineered Feature Vector*) and ARCHITECTURE.md Section 5.2 (*the 30-dim
streaming feature vector*). This module implements **both** halves of the
Appendix B.6 contract:

* :class:`FeatureExtractor` -- the **pure-python**, O(1)-per-step streaming
  extractor used on the edge / realtime hot path. ``update(...)`` maintains a
  bounded amount of state (a handful of EMAs, two Welford accumulators, two
  short ring buffers, a Hawkes-style decayed history accumulator) and returns a
  ``list[float]`` of length :data:`flarecast.constants.N_FEATURES`. It imports
  nothing beyond the standard library so it is fully testable offline and can
  be transliterated to the Cloudflare Durable Object.
* :func:`extract_features` -- the numpy **batch** twin. It feeds a window of
  samples through a fresh :class:`FeatureExtractor` and returns the final
  ``numpy.ndarray`` of shape ``(N_FEATURES,)``. numpy is imported lazily inside
  the function so the module imports cleanly without numpy installed.
* :func:`build_tcn_tensor` -- the raw ``[L, C]`` look-back tensor builder for
  the Tier-2 TCN (``TCN_LOOKBACK_STEPS`` x ``TCN_N_CHANNELS``), numpy-guarded.

**Causality is the central invariant** (research doc 04 Section 2.1, "Practical
notes": *use causal windows only -- no future samples -> no leakage*). Every
statistic is a function of the current and *past* samples only; a future spike
can never change a feature value emitted for an earlier step. ``test_features``
asserts this directly.

The 30 dimensions, in the exact order of ARCHITECTURE.md Section 5.2 (which is
also the order of :data:`FEATURE_NAMES`):

======  =================================  =========================================
Index   Name                               Meaning
======  =================================  =========================================
0       ``logS``                           EMA-smoothed log10 soft (SoLEXS) flux.
1       ``logH``                           EMA-smoothed log10 hard (HEL1OS) flux.
2       ``S_over_baseline``                S / quiet-Sun rolling baseline (>1 rising).
3       ``H_over_baseline``                H / quiet-Sun rolling baseline.
4       ``sxr_slope_short``                d(logS)/dt over a short EMA pair (precursor).
5       ``sxr_slope_long``                 d(logS)/dt over a long EMA pair.
6       ``sxr_curvature``                  2nd derivative of logS (onset sharpening).
7       ``hxr_slope_short``                d(logH)/dt over a short EMA pair.
8       ``slope_accel``                    sxr_slope_short - sxr_slope_long.
9       ``ema_ratio_S``                    EMA_short(S)/EMA_long(S) (>1 rising).
10      ``ema_ratio_H``                    EMA_short(H)/EMA_long(H).
11      ``ema_ratio_cross``                EMA_short(S)/EMA_long(H) (cross divergence).
12      ``hardness_ratio``                 H / S (non-thermal vs thermal).
13      ``d_hardness``                     d(hardness)/dt (spectral hardening).
14      ``temp_proxy``                     hot/cool soft-band ratio proxy (here logS-scaled).
15      ``d_temp_proxy``                   derivative of the temperature proxy.
16      ``neupert_corr``                   rolling corr(H, dS/dt) -- Neupert (highest value).
17      ``neupert_resid``                  ||H - alpha*dS/dt|| leaky-integrator residual.
18      ``hxr_leads_flag``                 H rising while dS/dt rising and S below local max.
19      ``sxr_var``                        rolling Welford variance of logS.
20      ``hxr_var``                        rolling Welford variance of logH.
21      ``hxr_burst_count``                # of >k-sigma excursions of H in the window.
22      ``hxr_peak_over_baseline``         max(H)/baseline within the ring-buffer window.
23      ``time_since_last_flare``          minutes since last flare onset (capped).
24      ``flares_last_6h``                 count of flares in the trailing 6 h.
25      ``decayed_flare_history``          sum exp(-dt/tau) over recent flares (Hawkes).
26      ``time_since_last_microflare``     minutes since last HXR burst (capped).
27      ``quiet_duration``                 minutes since flux last near quiet baseline.
28      ``solar_cycle_phase``              slow background-rate proxy in [0, 1].
29      ``N_horizon``                      forecast horizon N (minutes) -- horizon as a feature.
======  =================================  =========================================

The three **Neupert features** (indices 16-18) are the physical core of a
short-horizon forecast (ARCHITECTURE.md Section 1.2): index 16 is the rolling
Pearson correlation between the hard channel and the soft-flux derivative;
index 18 is the residual of the leaky-integrator coupling
``dF_SXR/dt = c*F_HXR - F_SXR/tau_cool`` with ``tau_cool`` =
:data:`flarecast.constants.NEUPERT_TAU_COOL_S`; index 18 (``hxr_leads_flag``)
is the leading-indicator boolean. Index 17's residual uses the *current*
integrator prediction so it spikes at impulsive onset.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from flarecast.constants import (
    HISTORY_DECAY_TAU_S,
    N_FEATURES,
    NEUPERT_TAU_COOL_S,
    TCN_LOOKBACK_STEPS,
    TCN_N_CHANNELS,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "extract_features",
    "build_tcn_tensor",
]

# ---------------------------------------------------------------------------
# Public feature-name vector (length == N_FEATURES, exact §5.2 order).
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "logS",                      # 0
    "logH",                      # 1
    "S_over_baseline",           # 2
    "H_over_baseline",           # 3
    "sxr_slope_short",           # 4
    "sxr_slope_long",            # 5
    "sxr_curvature",             # 6
    "hxr_slope_short",           # 7
    "slope_accel",               # 8
    "ema_ratio_S",               # 9
    "ema_ratio_H",               # 10
    "ema_ratio_cross",           # 11
    "hardness_ratio",            # 12
    "d_hardness",                # 13
    "temp_proxy",                # 14
    "d_temp_proxy",              # 15
    "neupert_corr",              # 16
    "neupert_resid",             # 17
    "hxr_leads_flag",            # 18
    "sxr_var",                   # 19
    "hxr_var",                   # 20
    "hxr_burst_count",           # 21
    "hxr_peak_over_baseline",    # 22
    "time_since_last_flare",     # 23
    "flares_last_6h",            # 24
    "decayed_flare_history",     # 25
    "time_since_last_microflare",  # 26
    "quiet_duration",            # 27
    "solar_cycle_phase",         # 28
    "N_horizon",                 # 29
]

assert len(FEATURE_NAMES) == N_FEATURES, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected N_FEATURES="
    f"{N_FEATURES}"
)

# ---------------------------------------------------------------------------
# Internal numeric guards / defaults.
# ---------------------------------------------------------------------------
#: Floor applied to flux before log10 so quiet-Sun zeros don't blow up.
_FLUX_FLOOR: float = 1e-12
#: Cap (minutes) for the "time since ..." features so a long quiet period
#: doesn't dominate tree splits; 24 h is generous for a minutes-ahead forecast.
_TIME_CAP_MIN: float = 24.0 * 60.0
#: k-sigma threshold for counting HXR bursts / micro-flares in the window.
_BURST_K_SIGMA: float = 3.0
#: Fractional enhancement over baseline that defines "not quiet" for
#: ``quiet_duration`` (10% above the rolling quiet baseline).
_QUIET_REL_TOL: float = 0.10
#: Length of the short raw ring buffer (in steps) used for burst counting and
#: the rolling Neupert correlation. Bounded -> O(1) state.
_RING_LEN: int = 30
#: Reference solar-cycle length (seconds) for the (weak) phase proxy. ~11 yr.
_SOLAR_CYCLE_S: float = 11.0 * 365.25 * 86400.0
#: Reference epoch (Unix seconds) for the cycle-phase ramp -- 2020-01-01,
#: roughly the Solar Cycle 25 minimum, so the phase grows through the mission.
_CYCLE_REF_EPOCH_S: float = 1577836800.0


def _safe_log10(x: float) -> float:
    """log10 with a positive floor (X-ray flux is log-distributed)."""
    return math.log10(x if x > _FLUX_FLOOR else _FLUX_FLOOR)


class _EMA:
    """One-pole exponential moving average; O(1) time, one float state.

    ``alpha`` is the *retention* factor: ``mu <- alpha*mu + (1-alpha)*x``. A
    larger ``alpha`` (closer to 1) means a longer effective window
    ``~1/(1-alpha)``. The first sample seeds the state so there is no warm-up
    transient toward zero.
    """

    __slots__ = ("alpha", "mu", "_seeded")

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mu = 0.0
        self._seeded = False

    def update(self, x: float) -> float:
        if not self._seeded:
            self.mu = x
            self._seeded = True
        else:
            self.mu = self.alpha * self.mu + (1.0 - self.alpha) * x
        return self.mu


class _Welford:
    """Online (Welford) mean+variance; O(1) time, three floats of state.

    Numerically stable single-pass variance (ARCHITECTURE.md primitives table).
    Returns the *population* variance (denominator n), which is well-defined
    from the first sample and avoids a NaN on n==1.
    """

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class _RollingCorr:
    """Rolling Pearson correlation over a bounded window; O(window) update.

    Holds two fixed-capacity deques (``deque(maxlen=...)`` -> O(1) push and
    automatic eviction). ``update`` returns ``corr(x, y)`` over the retained
    window. The window is bounded (``_RING_LEN``), so this is O(1) *state* and
    O(window)=O(1) work per step -- consistent with the "O(window) const"
    SlopeEstimator entry in the primitives table.
    """

    __slots__ = ("xs", "ys")

    def __init__(self, maxlen: int) -> None:
        self.xs: deque[float] = deque(maxlen=maxlen)
        self.ys: deque[float] = deque(maxlen=maxlen)

    def update(self, x: float, y: float) -> float:
        self.xs.append(x)
        self.ys.append(y)
        n = len(self.xs)
        if n < 2:
            return 0.0
        mx = sum(self.xs) / n
        my = sum(self.ys) / n
        sxy = sxx = syy = 0.0
        for xi, yi in zip(self.xs, self.ys, strict=True):
            dx = xi - mx
            dy = yi - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        denom = math.sqrt(sxx * syy)
        if denom <= 0.0:
            return 0.0
        c = sxy / denom
        # Clamp against floating-point overshoot beyond [-1, 1].
        return max(-1.0, min(1.0, c))


class FeatureExtractor:
    """Pure-python streaming extractor: O(1)/step -> ``list[float]`` of N_FEATURES.

    This is the **edge / realtime** path (ARCHITECTURE.md Section 5.6: "feature
    computation is O(1) with running EMAs + Welford variance + ring-buffer").
    Construct once per stream, then call :meth:`update` for each aggregated
    sample (default cadence 60 s). State size is independent of the number of
    samples processed.

    Parameters
    ----------
    cadence_s:
        Nominal spacing between successive :meth:`update` calls, in seconds.
        Used to translate EMA windows (expressed in minutes) into retention
        factors and to scale derivatives to per-minute units. The actual ``dt``
        between calls is taken from the supplied timestamps when available, so
        an irregular cadence is handled gracefully; ``cadence_s`` is the
        fallback for the first sample.
    """

    def __init__(self, cadence_s: float = 60.0) -> None:
        self.cadence_s = float(cadence_s) if cadence_s and cadence_s > 0 else 60.0

        # --- EMA windows (minutes) -> retention alphas (research doc 04 §2.1).
        # alpha = exp(-cadence / tau) gives an effective window ~ tau.
        def _alpha_for_window_min(win_min: float) -> float:
            tau_s = max(win_min * 60.0, self.cadence_s)
            return math.exp(-self.cadence_s / tau_s)

        # Levels (short smoothing) + multi-scale slope/EMA pairs.
        self._ema_S_short = _EMA(_alpha_for_window_min(2.0))
        self._ema_S_long = _EMA(_alpha_for_window_min(15.0))
        self._ema_H_short = _EMA(_alpha_for_window_min(2.0))
        self._ema_H_long = _EMA(_alpha_for_window_min(15.0))
        # Quiet-Sun baselines: very long EMA, *gated* so a flare does not
        # contaminate the baseline (ARCHITECTURE.md Section 4.3 critical note).
        self._base_S = _EMA(_alpha_for_window_min(120.0))
        self._base_H = _EMA(_alpha_for_window_min(120.0))
        # Hardness EMA (for d_hardness) and temp-proxy EMA (for d_temp_proxy).
        self._ema_hardness = _EMA(_alpha_for_window_min(2.0))
        self._ema_temp = _EMA(_alpha_for_window_min(2.0))

        # Welford rolling variance of logS / logH.
        self._var_S = _Welford()
        self._var_H = _Welford()

        # Rolling Neupert correlation corr(H, dS/dt).
        self._neupert_corr = _RollingCorr(_RING_LEN)

        # Short raw ring buffers for burst stats (bounded -> O(1) state).
        self._ring_H: deque[float] = deque(maxlen=_RING_LEN)

        # Leaky-integrator (Neupert) predicted SXR state, in *linear* flux.
        # dF/dt = c*H - F/tau_cool ; we track F_pred and compare to observed S.
        self._neupert_F_pred: float | None = None
        self._neupert_c: float = 1.0  # coupling gain (data-free default).

        # Previous-step memories for derivatives (all causal: value at t-1).
        self._prev_t: float | None = None
        self._prev_logS: float | None = None
        self._prev_logS_short: float | None = None  # for curvature
        self._prev_slope_S_short: float | None = None
        self._prev_logH: float | None = None
        self._prev_hardness: float | None = None
        self._prev_temp: float | None = None
        self._prev_S_lin: float | None = None  # observed linear S at t-1

        # Running local-max of S (causal) for the hxr_leads_flag.
        self._local_max_S: float = -math.inf

        # Burst / micro-flare bookkeeping.
        self._last_microflare_t: float | None = None

        # Quiet-duration / last-near-baseline timestamp.
        self._last_quiet_t: float | None = None

        # Step counter (warm-up handling).
        self._n: int = 0

    # ------------------------------------------------------------------
    # Streaming update.
    # ------------------------------------------------------------------
    def update(
        self,
        sxr: float,
        hxr: float,
        t: float,
        n_horizon: float,
        flare_history: list[float] | None = None,
    ) -> list[float]:
        """Ingest one sample; return the N_FEATURES-length feature vector.

        Parameters
        ----------
        sxr, hxr:
            Soft (SoLEXS) and hard (HEL1OS) channel values for this step, in
            their canonical linear units. Non-finite or negative inputs are
            floored, so a despiked gap-filled sample never crashes the hot path.
        t:
            Epoch seconds (UTC, ``t_earth`` frame) of this sample. Used for
            ``dt``, time-since features, decayed history and the cycle phase.
        n_horizon:
            Forecast horizon N in **minutes**, emitted verbatim as the last
            feature so a single model serves multiple horizons (§5.2 #30).
        flare_history:
            Optional list of past flare *onset* epoch-seconds (<= ``t``). Drives
            the Hawkes-style history features (#24-26). The caller owns this
            list; we only read it. If ``None``, history features are zeroed
            (capped time-since).

        Returns
        -------
        list[float]
            Length :data:`flarecast.constants.N_FEATURES`.
        """
        self._n += 1

        # --- Sanitize inputs -------------------------------------------------
        s_ok = isinstance(sxr, (int, float)) and math.isfinite(sxr) and sxr > 0
        s_lin = sxr if s_ok else _FLUX_FLOOR
        h_ok = isinstance(hxr, (int, float)) and math.isfinite(hxr) and hxr >= 0
        h_lin = hxr if h_ok else 0.0
        t = float(t)

        # --- dt (seconds, minutes) ------------------------------------------
        if self._prev_t is None:
            dt_s = self.cadence_s
        else:
            dt_s = t - self._prev_t
            if not math.isfinite(dt_s) or dt_s <= 0:
                dt_s = self.cadence_s
        dt_min = dt_s / 60.0

        # --- Levels (log10) + EMA smoothing ---------------------------------
        logS_raw = _safe_log10(s_lin)
        logH_raw = _safe_log10(h_lin + 1.0)  # +1 so zero counts -> 0, not -inf.

        logS = self._ema_S_short.update(logS_raw)
        logH = self._ema_H_short.update(logH_raw)
        emaS_long = self._ema_S_long.update(logS_raw)
        emaH_long = self._ema_H_long.update(logH_raw)

        # --- Baselines (gated): only let the quiet baseline track when we are
        # NOT clearly in a flare, else the detector "adapts away" the signal
        # (ARCHITECTURE.md Section 4.3). Gate when S is >2x its baseline.
        prev_base_S = self._base_S.mu if self._base_S._seeded else s_lin
        prev_base_H = self._base_H.mu if self._base_H._seeded else max(h_lin, 1e-6)
        gate_open = s_lin <= 2.0 * prev_base_S
        if gate_open or not self._base_S._seeded:
            base_S = self._base_S.update(s_lin)
            base_H = self._base_H.update(h_lin)
        else:
            base_S = prev_base_S
            base_H = prev_base_H
        base_S = base_S if base_S > _FLUX_FLOOR else _FLUX_FLOOR
        base_H_safe = base_H if base_H > 1e-9 else 1e-9

        s_over_base = s_lin / base_S
        h_over_base = (h_lin + 1.0) / (base_H_safe + 1.0)

        # --- Slopes (per-minute, log-domain) --------------------------------
        # short slope from short EMA, long slope from long EMA -- both causal.
        if self._prev_logS_short is None:
            sxr_slope_short = 0.0
        else:
            sxr_slope_short = (logS - self._prev_logS_short) / dt_min
        if self._prev_logS is None:
            sxr_slope_long = 0.0
        else:
            sxr_slope_long = (emaS_long - self._prev_logS) / dt_min
        if self._prev_logH is None:
            hxr_slope_short = 0.0
        else:
            hxr_slope_short = (logH - self._prev_logH) / dt_min

        # Curvature = change in short slope (2nd derivative of logS).
        if self._prev_slope_S_short is None:
            sxr_curvature = 0.0
        else:
            sxr_curvature = (sxr_slope_short - self._prev_slope_S_short) / dt_min

        slope_accel = sxr_slope_short - sxr_slope_long

        # --- EMA ratios (linear-domain ratios of the smoothed levels) -------
        # Use 10**(log EMA) to get back to flux-scale ratios; >1 => rising.
        emaS_short_lin = 10.0 ** logS
        emaS_long_lin = 10.0 ** emaS_long
        emaH_short_lin = 10.0 ** logH
        emaH_long_lin = 10.0 ** emaH_long
        ema_ratio_S = emaS_short_lin / emaS_long_lin if emaS_long_lin > 0 else 1.0
        ema_ratio_H = emaH_short_lin / emaH_long_lin if emaH_long_lin > 0 else 1.0
        ema_ratio_cross = emaS_short_lin / emaH_long_lin if emaH_long_lin > 0 else 1.0

        # --- Hardness & temperature proxy -----------------------------------
        # Hardness = hard / soft (non-thermal vs thermal). Smoothed for d/dt.
        hardness_raw = (h_lin + 1.0) / (s_lin + _FLUX_FLOOR)
        hardness = self._ema_hardness.update(hardness_raw)
        if self._prev_hardness is None:
            d_hardness = 0.0
        else:
            d_hardness = (hardness - self._prev_hardness) / dt_min
        # Temperature proxy: in a Sun-as-a-star single-band synthetic we do not
        # have two SoLEXS sub-bands, so use a monotone proxy of the soft level
        # (hotter plasma -> brighter SXR). Smoothed; derivative tracked.
        temp_raw = logS_raw
        temp_proxy = self._ema_temp.update(temp_raw)
        if self._prev_temp is None:
            d_temp_proxy = 0.0
        else:
            d_temp_proxy = (temp_proxy - self._prev_temp) / dt_min

        # --- Neupert features (the highest-value block) ---------------------
        # dS/dt in *linear* flux (the Neupert relation is on linear flux).
        if self._prev_S_lin is None:
            dS_dt = 0.0
        else:
            dS_dt = (s_lin - self._prev_S_lin) / dt_s
        # (16) rolling corr(H, dS/dt).
        neupert_corr = self._neupert_corr.update(h_lin, dS_dt)
        # (17) leaky-integrator residual: predict F_SXR from H, compare to S.
        # dF/dt = c*H - F/tau_cool  (Euler step). Residual = |S - F_pred|
        # normalized by baseline so it is dimensionless and spikes at onset.
        if self._neupert_F_pred is None:
            self._neupert_F_pred = s_lin
        f_pred = self._neupert_F_pred
        f_pred = f_pred + dt_s * (self._neupert_c * h_lin - f_pred / NEUPERT_TAU_COOL_S)
        if f_pred < 0:
            f_pred = 0.0
        neupert_resid = abs(s_lin - f_pred) / (base_S + _FLUX_FLOOR)
        self._neupert_F_pred = f_pred
        # (18) hxr_leads_flag: H rising AND dS/dt rising AND S below local max.
        hxr_leads_flag = 1.0 if (
            hxr_slope_short > 0.0 and dS_dt > 0.0 and s_lin < self._local_max_S
        ) else 0.0

        # --- Variance (Welford) ---------------------------------------------
        self._var_S.update(logS_raw)
        self._var_H.update(logH_raw)
        sxr_var = self._var_S.variance
        hxr_var = self._var_H.variance

        # --- Burst detection over the short ring buffer ---------------------
        self._ring_H.append(h_lin)
        h_std = self._var_H.std
        h_mean = self._var_H.mean
        # Count >k-sigma excursions of (log) H in the window. Use the log ring
        # for scale-stability: compare each entry's deviation in log space.
        burst_count = 0
        if h_std > 0:
            thresh = h_mean + _BURST_K_SIGMA * h_std
            for v in self._ring_H:
                if _safe_log10(v + 1.0) > thresh:
                    burst_count += 1
        hxr_burst_count = float(burst_count)
        # A fresh burst on *this* sample updates the micro-flare clock.
        if h_std > 0 and logH_raw > h_mean + _BURST_K_SIGMA * h_std:
            self._last_microflare_t = t
        # Peak over baseline within the window.
        ring_max = max(self._ring_H) if self._ring_H else h_lin
        hxr_peak_over_baseline = (ring_max + 1.0) / (base_H_safe + 1.0)

        # --- History / self-excitation (Hawkes-flavored) --------------------
        tsl_flare_min = _TIME_CAP_MIN
        flares_last_6h = 0.0
        decayed_history = 0.0
        if flare_history:
            six_h_s = 6.0 * 3600.0
            last_onset = None
            for ot in flare_history:
                if ot is None:
                    continue
                dt_hist = t - float(ot)
                if dt_hist < 0:
                    # Strictly causal: ignore any "future" onsets defensively.
                    continue
                if last_onset is None or float(ot) > last_onset:
                    last_onset = float(ot)
                if dt_hist <= six_h_s:
                    flares_last_6h += 1.0
                decayed_history += math.exp(-dt_hist / HISTORY_DECAY_TAU_S)
            if last_onset is not None:
                tsl_flare_min = min((t - last_onset) / 60.0, _TIME_CAP_MIN)

        if self._last_microflare_t is None:
            tsl_micro_min = _TIME_CAP_MIN
        else:
            tsl_micro_min = min((t - self._last_microflare_t) / 60.0, _TIME_CAP_MIN)

        # --- Quiet duration --------------------------------------------------
        # "Near quiet" = within _QUIET_REL_TOL of the rolling soft baseline.
        if s_lin <= base_S * (1.0 + _QUIET_REL_TOL):
            self._last_quiet_t = t
        if self._last_quiet_t is None:
            quiet_duration_min = 0.0
        else:
            quiet_duration_min = min((t - self._last_quiet_t) / 60.0, _TIME_CAP_MIN)

        # --- Solar-cycle phase (slow background-rate proxy, in [0,1]) -------
        phase = ((t - _CYCLE_REF_EPOCH_S) / _SOLAR_CYCLE_S) % 1.0
        if phase < 0:
            phase += 1.0
        solar_cycle_phase = phase

        # --- Assemble the vector (exact §5.2 order) -------------------------
        features: list[float] = [
            logS,                       # 0
            logH,                       # 1
            s_over_base,                # 2
            h_over_base,                # 3
            sxr_slope_short,            # 4
            sxr_slope_long,             # 5
            sxr_curvature,              # 6
            hxr_slope_short,            # 7
            slope_accel,                # 8
            ema_ratio_S,                # 9
            ema_ratio_H,                # 10
            ema_ratio_cross,            # 11
            hardness,                   # 12
            d_hardness,                 # 13
            temp_proxy,                 # 14
            d_temp_proxy,               # 15
            neupert_corr,               # 16
            neupert_resid,              # 17
            hxr_leads_flag,             # 18
            sxr_var,                    # 19
            hxr_var,                    # 20
            hxr_burst_count,            # 21
            hxr_peak_over_baseline,     # 22
            tsl_flare_min,              # 23
            flares_last_6h,             # 24
            decayed_history,            # 25
            tsl_micro_min,              # 26
            quiet_duration_min,         # 27
            solar_cycle_phase,          # 28
            float(n_horizon),           # 29
        ]

        # --- Roll memories forward (AFTER computing this step's features so the
        # vector for step t never depends on step t+1: strict causality). ----
        self._prev_t = t
        self._prev_logS = emaS_long
        self._prev_logS_short = logS
        self._prev_slope_S_short = sxr_slope_short
        self._prev_logH = logH
        self._prev_hardness = hardness
        self._prev_temp = temp_proxy
        self._prev_S_lin = s_lin
        if s_lin > self._local_max_S:
            self._local_max_S = s_lin

        # Defensive: guarantee length & finiteness for downstream models.
        if len(features) != N_FEATURES:  # pragma: no cover - structural guard
            raise AssertionError(
                f"feature vector length {len(features)} != N_FEATURES {N_FEATURES}"
            )
        for i, v in enumerate(features):
            if not math.isfinite(v):
                features[i] = 0.0
        return features


# ---------------------------------------------------------------------------
# Batch / numpy wrappers (numpy is imported lazily so the module imports
# cleanly without numpy installed -- the streaming path above needs only
# stdlib).
# ---------------------------------------------------------------------------
def _window_rows(window: Any) -> tuple[list[float], list[float], list[float]]:
    """Extract (sxr, hxr, t) columns from a window object.

    Accepts a pandas-like ``DataFrame`` (duck-typed via ``columns`` + indexing)
    or a plain mapping of column-name -> sequence. Column names are matched
    leniently so the same code works on the synth generator output
    (``sxr_long`` / ``hxr_8_30``) and on a generic ``{"sxr": ..., "hxr": ...}``
    mapping. This keeps the forecast module decoupled from WS1's exact schema.
    """
    def _col(names: tuple[str, ...]) -> list[float]:
        # DataFrame path.
        cols = getattr(window, "columns", None)
        if cols is not None:
            colset = set(map(str, cols))
            for nm in names:
                if nm in colset:
                    return [float(v) for v in list(window[nm])]
            raise KeyError(f"window missing any of columns {names}; has {sorted(colset)}")
        # Mapping path.
        if isinstance(window, dict):
            for nm in names:
                if nm in window:
                    return [float(v) for v in window[nm]]
            raise KeyError(f"window missing any of keys {names}; has {sorted(window)}")
        raise TypeError(f"unsupported window type {type(window)!r}")

    sxr = _col(("sxr", "sxr_long", "S", "soft", "solexs"))
    hxr = _col(("hxr", "hxr_8_30", "H", "hard", "hel1os"))
    try:
        t = _col(("t", "time", "epoch", "t_earth"))
    except KeyError:
        # Synthesize a uniform 60 s grid if no time column is supplied.
        t = [float(i) * 60.0 for i in range(len(sxr))]
    return sxr, hxr, t


def extract_features(window: Any, n_horizon: float):
    """Batch feature extraction -> ``numpy.ndarray`` of shape ``(N_FEATURES,)``.

    Feeds every row of ``window`` (oldest-first) through a fresh
    :class:`FeatureExtractor` and returns the **final** step's feature vector as
    a float64 numpy array. This is the numpy twin of the streaming extractor; on
    the same input the last :meth:`FeatureExtractor.update` value and this
    array are identical.

    numpy is imported here (not at module top) so ``import
    flarecast.forecast.features`` succeeds without numpy installed; the test for
    this function uses ``pytest.importorskip("numpy")``.

    Parameters
    ----------
    window:
        A pandas ``DataFrame`` or column-mapping with soft/hard (and optional
        time) columns; see :func:`_window_rows` for accepted names.
    n_horizon:
        Forecast horizon N in minutes (emitted as the last feature).
    """
    import numpy as np  # lazy

    sxr, hxr, t = _window_rows(window)
    if not sxr:
        return np.zeros(N_FEATURES, dtype=np.float64)
    ex = FeatureExtractor()
    vec: list[float] = [0.0] * N_FEATURES
    for s, h, ti in zip(sxr, hxr, t, strict=True):
        vec = ex.update(s, h, ti, n_horizon)
    return np.asarray(vec, dtype=np.float64)


def build_tcn_tensor(window: Any, lookback: int = TCN_LOOKBACK_STEPS,
                     n_channels: int = TCN_N_CHANNELS):
    """Build the raw ``[L, C]`` look-back tensor for the Tier-2 TCN.

    Produces a ``numpy.ndarray`` of shape ``(lookback, n_channels)`` with the
    most recent ``lookback`` steps (zero-padded at the front if the window is
    shorter). Channels, in order (ARCHITECTURE.md Section 5.2 / research 04
    Section 2.2): ``logS, logH, hardness, d(logS)/dt, d(logH)/dt`` and, to fill
    out to ``TCN_N_CHANNELS=8``: ``S_over_baseline, H_over_baseline,
    neupert_resid``. All channels are strictly causal.

    numpy is imported lazily; the test uses ``pytest.importorskip("numpy")``.
    """
    import numpy as np  # lazy

    sxr, hxr, t = _window_rows(window)
    n = len(sxr)
    out = np.zeros((lookback, n_channels), dtype=np.float64)
    if n == 0:
        return out

    ex = FeatureExtractor()
    rows: list[list[float]] = []
    for s, h, ti in zip(sxr, hxr, t, strict=True):
        f = ex.update(s, h, ti, 0.0)
        # Map the 8 TCN channels out of the engineered vector (causal subset).
        rows.append([
            f[0],   # logS
            f[1],   # logH
            f[12],  # hardness_ratio
            f[4],   # d(logS)/dt  (sxr_slope_short)
            f[7],   # d(logH)/dt  (hxr_slope_short)
            f[2],   # S_over_baseline
            f[3],   # H_over_baseline
            f[17],  # neupert_resid
        ][:n_channels])

    tail = rows[-lookback:]
    start = lookback - len(tail)
    for i, r in enumerate(tail):
        out[start + i, : len(r)] = r
    return out
