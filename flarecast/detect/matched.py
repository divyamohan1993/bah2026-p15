"""SPIIR-style IIR FRED matched filter (ARCHITECTURE.md Section 4.4, B.4).

Soft X-ray flares have a characteristic **fast-rise / exponential-decay (FRED)**
shape. The optimal linear detector for a known shape is a matched filter, but an
exact FIR matched filter costs O(L) per sample. Approximating the FRED template
by a difference of two exponentials makes each exponential a one-pole recursive
filter (an EMA), so the matched filter collapses to a **sum of a few EMAs** --
**O(1) time, O(1) state**. This is the SPIIR trick from low-latency
gravitational-wave matched filtering (Hooper et al. 2012); here it is a *shape
confirmer / feature* for gradual soft flares, not the primary trigger (research
doc ``03 Section 2.8``)::

    r_t  = alpha_decay * r_{t-1} + (1 - alpha_decay) * x_t     # slow pole (decay)
    u_t  = alpha_rise  * u_{t-1} + (1 - alpha_rise)  * x_t     # fast pole (rise)
    mf_t = c1 * r_t - c2 * u_t                                  # "FREDness"
"""

from __future__ import annotations

from .primitives import EMA

__all__ = ["IIRMatchedFilter"]


class IIRMatchedFilter:
    """IIR (SPIIR) approximation of a FRED matched filter (``03 Section 2.8``).

    The statistic ``mf = c1 * slow_ema - c2 * fast_ema`` peaks during the
    fast-rise / exponential-decay structure of a soft flare and stays near zero
    for flat baselines and symmetric noise, so it discriminates gradual soft
    flares from impulsive (spiky) hard structure and from non-flare ramps.

    Complexity: **O(1) time**, **O(P) = O(1) state** (two one-pole EMAs).

    Parameters
    ----------
    alpha_rise:
        Forgetting factor of the *fast* pole (short time constant, the rise).
    alpha_decay:
        Forgetting factor of the *slow* pole (long time constant, the decay);
        should satisfy ``alpha_decay > alpha_rise`` so the slow pole genuinely
        lags the fast one.
    c1, c2:
        Mixing weights of the slow and fast poles (template amplitudes).
    """

    __slots__ = ("c1", "c2", "_slow", "_fast")

    def __init__(
        self,
        alpha_rise: float,
        alpha_decay: float,
        c1: float = 1.0,
        c2: float = 1.0,
    ) -> None:
        if alpha_decay <= alpha_rise:
            raise ValueError(
                f"alpha_decay ({alpha_decay}) must exceed alpha_rise "
                f"({alpha_rise}) so the slow pole lags the fast one"
            )
        self.c1: float = float(c1)
        self.c2: float = float(c2)
        self._fast: EMA = EMA(alpha_rise)
        self._slow: EMA = EMA(alpha_decay)

    def update(self, x: float) -> float:
        """Fold one sample in; return the FREDness statistic. O(1)."""
        slow = self._slow.update(x)
        fast = self._fast.update(x)
        return self.c1 * slow - self.c2 * fast

    @property
    def value(self) -> float:
        """Current FREDness statistic without updating."""
        return self.c1 * self._slow.value - self.c2 * self._fast.value

    def reset(self, x0: float = 0.0) -> None:
        """Re-seed both poles to ``x0`` (e.g. after a data gap). O(1)."""
        self._fast.m = float(x0)
        self._slow.m = float(x0)
