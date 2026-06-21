"""Mandatory forecast baselines: climatology, persistence, Hawkes.

Governing research: ``docs/research/04-forecasting-models.md`` Section 6
(*Baselines to beat (mandatory)*) and ARCHITECTURE.md Section 5.5 / 10.2.

All forecast skill is reported **relative to these** -- raw TSS alone is not
credible (research doc 04 Section 0/6; the 2025 SWPC verification shows
persistence is shockingly hard to beat). Each baseline implements the Appendix
B.6 ``fit`` / ``predict_proba`` surface and is **pure standard library** so it
is fully testable offline.

Feature-column convention. ``predict_proba`` receives the same ``X`` matrix the
models see -- rows of the :data:`flarecast.forecast.features.FEATURE_NAMES`
vector. The baselines read a small number of those columns by their fixed index
(documented at each use): index 2 ``S_over_baseline`` (a "flaring now" proxy),
index 24 ``flares_last_6h`` and index 25 ``decayed_flare_history``
(Hawkes-style self-excitation), index 29 ``N_horizon``. ``X`` may be a numpy
array or a list of rows; both index identically with ``row[i]``.
"""

from __future__ import annotations

import math
from typing import Any

from flarecast.constants import HISTORY_DECAY_TAU_S

__all__ = [
    "ClimatologyBaseline",
    "PersistenceBaseline",
    "HawkesBaseline",
]

# Fixed feature indices this module relies on (see FEATURE_NAMES).
_IDX_S_OVER_BASE = 2
_IDX_FLARES_6H = 24
_IDX_DECAYED_HISTORY = 25

#: "Flaring now" threshold on ``S_over_baseline`` (flux >= 1.5x quiet baseline).
_FLARING_REL = 1.5


def _n_rows(X: Any) -> int:
    """Row count of a numpy array or list-of-rows."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)


def _row(X: Any, i: int):
    """Row ``i`` of a numpy array or list-of-rows."""
    return X[i]


def _col_value(row: Any, idx: int, default: float = 0.0) -> float:
    """Safely read ``row[idx]`` (numpy row or list) as a float."""
    try:
        v = float(row[idx])
    except (IndexError, TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class ClimatologyBaseline:
    """Constant base-rate forecaster: ``P = mean(y_train)``.

    The simplest skill floor (research doc 04 Section 6): predict the training
    base rate of "flare within N" for every sample, ignoring the inputs. BSS is
    defined *against* exactly this probability, so the climatology baseline is
    also the reference for :func:`flarecast.forecast.evaluate.bss`.
    """

    def __init__(self) -> None:
        self.base_rate_: float = 0.0
        self._fitted = False

    def fit(self, y: Any) -> ClimatologyBaseline:
        vals = [float(v) for v in y]
        self.base_rate_ = (sum(vals) / len(vals)) if vals else 0.0
        # Clamp strictly inside (0, 1) so downstream log-losses stay finite.
        self.base_rate_ = min(1.0 - 1e-6, max(1e-6, self.base_rate_))
        self._fitted = True
        return self

    def predict_proba(self, X: Any) -> list[float]:
        if not self._fitted:
            raise RuntimeError("ClimatologyBaseline.predict_proba before fit()")
        return [self.base_rate_] * _n_rows(X)


class PersistenceBaseline:
    """"Flaring now => predict a flare" (with an optional decayed variant).

    Plain persistence (research doc 04 Section 6): output a high probability
    when the Sun is *currently* flaring (here: ``S_over_baseline`` >=
    :data:`_FLARING_REL`), a low probability otherwise -- no training needed.

    Decayed variant (``decayed=True``): instead of a hard now/not-now flag, map
    the Hawkes-style ``decayed_flare_history`` feature (sum of ``exp(-dt/tau)``
    over recent flares) through a saturating transform so recent activity yields
    a graded probability that decays with time -- the "decayed persistence"
    explicitly called for in the research doc. ``tau_min`` rescales the decayed
    score; it defaults to 360 min (6 h) to match
    :data:`flarecast.constants.HISTORY_DECAY_TAU_S`.
    """

    def __init__(self, decayed: bool = False, tau_min: float = 360.0) -> None:
        self.decayed = bool(decayed)
        self.tau_min = float(tau_min)
        # High/low probabilities for the hard persistence decision.
        self._p_hi = 1.0 - 1e-3
        self._p_lo = 1e-3
        # Reference scale for the decayed score's saturation (1 recent flare
        # at lag 0 contributes ~1.0 to decayed_flare_history).
        self._decay_scale = max(self.tau_min / (HISTORY_DECAY_TAU_S / 60.0), 1e-6)

    def fit(self, X: Any = None, y: Any = None) -> PersistenceBaseline:
        """Persistence is parameter-free; ``fit`` is a no-op returning self."""
        return self

    def predict_proba(self, X: Any) -> list[float]:
        out: list[float] = []
        for i in range(_n_rows(X)):
            row = _row(X, i)
            if self.decayed:
                # Graded probability from decayed flare history.
                score = _col_value(row, _IDX_DECAYED_HISTORY, 0.0)
                # Saturating map: 1 - exp(-score / scale) in [0, 1).
                p = 1.0 - math.exp(-max(score, 0.0) / self._decay_scale)
                p = min(self._p_hi, max(self._p_lo, p))
            else:
                flaring = _col_value(row, _IDX_S_OVER_BASE, 0.0) >= _FLARING_REL
                p = self._p_hi if flaring else self._p_lo
            out.append(p)
        return out


class HawkesBaseline:
    """Self-exciting point-process event-rate baseline.

    A 1-D Hawkes intensity ``lambda(t) = mu + Sum_i alpha * exp(-(t - t_i)/tau)``
    over past flare event times (research doc 04 Section 3.3 / 6). ``fit``
    estimates the background rate ``mu`` from the mean event spacing and uses a
    fixed excitation kernel (``alpha``, ``tau`` from
    :data:`flarecast.constants.HISTORY_DECAY_TAU_S`); ``predict_proba`` converts
    the per-sample intensity to a probability of "at least one event in the next
    horizon" via the Poisson tail ``1 - exp(-lambda * N)``.

    Because this baseline only sees flare *history*, ``predict_proba`` reads the
    streaming ``decayed_flare_history`` feature (index 25) as the already-decayed
    excitation term and ``N_horizon`` (index 29) as the horizon, so it needs no
    separate event stream at inference time. ``fit(event_times)`` calibrates the
    background ``mu`` and the excitation gain ``alpha`` from the training events.
    """

    def __init__(self) -> None:
        # tau in *minutes* (the decayed-history feature uses HISTORY_DECAY_TAU_S).
        self.tau_min: float = HISTORY_DECAY_TAU_S / 60.0
        self.mu_per_min: float = 1e-4   # background event rate [events/min].
        self.alpha: float = 0.5         # excitation gain per decayed-history unit.
        self._fitted = False

    def fit(self, event_times: Any) -> HawkesBaseline:
        """Estimate background rate ``mu`` from training event times.

        Parameters
        ----------
        event_times:
            Iterable of flare event epoch-seconds (onsets or peaks). ``mu`` is
            set to the mean event rate (events per minute) over the observed
            span; ``alpha`` is left at its prior (a fuller MLE is out of scope
            for a baseline). An empty / singleton input falls back to the prior.
        """
        ts = sorted(float(x) for x in (event_times or []))
        if len(ts) >= 2:
            span_min = (ts[-1] - ts[0]) / 60.0
            if span_min > 0:
                self.mu_per_min = max(len(ts) / span_min, 1e-9)
        self._fitted = True
        return self

    def predict_proba(self, X: Any) -> list[float]:
        out: list[float] = []
        for i in range(_n_rows(X)):
            row = _row(X, i)
            decayed = _col_value(row, _IDX_DECAYED_HISTORY, 0.0)
            n_horizon_min = _col_value(row, -1, 0.0)  # last col = N_horizon.
            if n_horizon_min <= 0:
                n_horizon_min = 30.0
            # Intensity = background + excitation from recent (decayed) flares.
            lam = self.mu_per_min + self.alpha * self.mu_per_min * max(decayed, 0.0)
            # P(>=1 event in next N) under a (locally constant) Poisson rate.
            p = 1.0 - math.exp(-lam * n_horizon_min)
            out.append(min(1.0 - 1e-6, max(1e-6, p)))
        return out
