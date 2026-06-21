"""Hard-band Poisson onset detectors (ARCHITECTURE.md Section 4.4, B.4).

Hard X-ray HEL1OS data are low-count **Poisson**; Gaussian thresholds are
invalid (variance equals the mean, fluctuations are asymmetric). This module
provides the two recommended count-channel detectors (research doc
``03 Section 2.5``):

``PoissonFOCuS``
    The **primary HEL1OS onset detector**. FOCuS (Functional Online CUSUM,
    Ward et al. 2023 -- developed for onboard GRB/CubeSat triggers) tests *all*
    post-change magnitudes and *all* window sizes simultaneously, so the flare
    magnitude need not be guessed in advance (flares span orders of magnitude).
    It runs in **amortised O(1)** via a pruned piecewise representation of the
    cost -- here, a curve list maintained as a convex hull of cumulative-count
    partial sums.

``PoissonCUSUM``
    A Poisson CUSUM confirm for a *fixed* rate jump ``lam0 -> lam1 = rho*lam0``;
    O(1) time, one float of state.

Both return a :class:`flarecast.types.DetectionState` from their O(1) ``update``.
"""

from __future__ import annotations

import math

from ..constants import POISSON_CUSUM_H, POISSON_LAM1_RATIO
from ..types import DetectionState

__all__ = ["PoissonFOCuS", "PoissonCUSUM"]


def _poisson_llr(csum: float, n: int, mu0: float) -> float:
    """Poisson generalized log-likelihood ratio for a segment vs background.

    For ``csum`` counts observed over ``n`` bins with background rate ``mu0``
    per bin, the MLE post-change rate is ``lam_hat = csum / n`` and the GLR
    statistic is::

        LLR = csum * ln(lam_hat / mu0) - (csum - n * mu0)

    which is ``0`` when ``lam_hat <= mu0`` (we only test rate *increases*).
    O(1).
    """
    if n <= 0 or mu0 <= 0.0:
        return 0.0
    lam = csum / n
    if lam <= mu0:
        return 0.0
    return csum * math.log(lam / mu0) - (csum - n * mu0)


class PoissonFOCuS:
    """Poisson-FOCuS hard-band onset detector (research doc ``03 Section 2.5b``).

    Maintains a list of candidate change points. Each candidate ``tau`` is
    summarised by the cumulative counts and bin-count accumulated since ``tau``;
    equivalently we keep the running cumulative count ``C_t`` and, for each
    retained candidate, the snapshot ``(C_tau, t_tau)`` taken at that change
    point. The best statistic at time ``t`` is

        ``max over retained tau of  LLR(C_t - C_tau, t - t_tau, mu0)``.

    **Pruning (the FOCuS trick).** A candidate ``tau_i`` can be discarded if a
    later candidate ``tau_j`` always yields at least as large an excess rate for
    every future ``t``. In cumulative-sum space this is exactly the condition
    that the points ``(t_tau, C_tau)`` must form a **lower convex hull**: any
    interior point sits above a chord and can never give the steepest (highest
    average-rate) segment, so it is removed. The hull holds only a handful of
    points in practice, giving **amortised O(1)** time and O(1) state.

    The background ``mu0`` (counts per bin) is supplied per sample by the caller
    (a long, gated EMA of the counts -- research doc ``03 Section 2.5``).

    Parameters
    ----------
    threshold:
        Alarm threshold on the FOCuS statistic (set from a target ARL_0 / false
        alarm rate). An alarm fires when ``stat > threshold``.
    max_curves:
        Safety cap on the retained-candidate list so a pathological stream can
        never grow the state without bound (keeps worst-case per-sample work
        bounded). The convex-hull pruning normally keeps the list far shorter.
    """

    __slots__ = ("thr", "_csum", "_t", "_hull_c", "_hull_t", "_max", "_in_event")

    def __init__(self, threshold: float, max_curves: int = 64) -> None:
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {threshold!r}")
        if max_curves < 2:
            raise ValueError(f"max_curves must be >= 2, got {max_curves!r}")
        self.thr: float = float(threshold)
        self._max: int = int(max_curves)
        # Running cumulative count and bin index (integer time in bins).
        self._csum: float = 0.0
        self._t: int = 0
        # Convex hull of candidate change points as parallel arrays of the
        # cumulative-count snapshot and the bin index at each candidate. The
        # first entry is the "anchor" (state at the last reset / stream start).
        self._hull_c: list[float] = [0.0]
        self._hull_t: list[int] = [0]
        self._in_event: bool = False

    def update(self, count: float, mu0: float) -> DetectionState:
        """Fold one count bin in; return a :class:`DetectionState`. Amortised O(1).

        Parameters
        ----------
        count:
            Counts observed in this bin (Poisson; ``>= 0``).
        mu0:
            Background rate (counts per bin), e.g. a gated long EMA of counts.
            A non-positive ``mu0`` is floored to a small positive value.
        """
        c = float(count)
        if c < 0.0:
            c = 0.0
        b0 = mu0 if mu0 > 0.0 else 1e-9

        self._t += 1
        self._csum += c
        t = self._t
        csum_now = self._csum

        hull_c = self._hull_c
        hull_t = self._hull_t

        # --- Add the new candidate change point (the point just before t) and
        #     restore the lower-convex-hull property. We keep points such that
        #     the average rate of the *most recent* segment is maximal; a point
        #     is interior (prunable) when the slope to it from its predecessor
        #     is >= the slope from it to the newcomer.
        new_c = csum_now
        new_t = t
        while len(hull_c) >= 2:
            c1, t1 = hull_c[-2], hull_t[-2]
            c2, t2 = hull_c[-1], hull_t[-1]
            # slope (avg rate) of segment (1->2) vs (2->new)
            # cross-product form avoids division: (c2-c1)*(new_t-t2) vs (new_c-c2)*(t2-t1)
            left = (c2 - c1) * (new_t - t2)
            right = (new_c - c2) * (t2 - t1)
            if left >= right:
                hull_c.pop()
                hull_t.pop()
            else:
                break
        hull_c.append(new_c)
        hull_t.append(new_t)

        # Bound the state size (defensive; hull is normally tiny).
        if len(hull_c) > self._max:
            del hull_c[0 : len(hull_c) - self._max]
            del hull_t[0 : len(hull_t) - self._max]

        # --- Evaluate the FOCuS statistic: best LLR of the segment from each
        #     retained candidate up to now. The hull is ordered by time; the
        #     maximum-rate segment ending at ``t`` is found by scanning the hull
        #     (a handful of points) for the steepest chord to (t, csum_now).
        stat = 0.0
        best_tau_t: int | None = None
        # The most recent segments (largest tau) give the steepest rate on a
        # lower hull; scan from the front to find the changepoint maximising LLR.
        for i in range(len(hull_t) - 1):
            seg_counts = csum_now - hull_c[i]
            seg_bins = t - hull_t[i]
            llr = _poisson_llr(seg_counts, seg_bins, b0)
            if llr > stat:
                stat = llr
                best_tau_t = hull_t[i]

        if stat > self.thr:
            onset_time = float(best_tau_t) if best_tau_t is not None else None
            self._in_event = True
            return DetectionState(
                onset=True,
                in_event=True,
                statistic=stat,
                onset_time=onset_time,
                meta={"detector": "PoissonFOCuS", "n_curves": len(hull_t)},
            )

        return DetectionState(
            onset=False,
            in_event=self._in_event,
            statistic=stat,
            onset_time=None,
            meta={"detector": "PoissonFOCuS", "n_curves": len(hull_t)},
        )

    def reset(self, in_event: bool = False) -> None:
        """Collapse the candidate list back to the current state. O(1).

        Called at flare end so accumulated evidence does not bleed into the next
        event. The cumulative count and time keep running (they are differences),
        but the hull is re-anchored at "now".
        """
        self._hull_c = [self._csum]
        self._hull_t = [self._t]
        self._in_event = bool(in_event)

    @property
    def n_curves(self) -> int:
        """Number of retained candidate change points (hull size)."""
        return len(self._hull_t)


class PoissonCUSUM:
    """Poisson CUSUM for a fixed rate jump ``lam0 -> lam1`` (``03 Section 2.5a``).

    The LLR contribution of one count bin ``x`` for a jump from the (current)
    background ``lam0 = mu0`` to ``lam1 = lam1_ratio * mu0`` is
    ``x * ln(lam1/lam0) - (lam1 - lam0)``; accumulating with a reflecting floor
    at zero gives the Poisson CUSUM::

        S_t = max(0, S_{t-1} + x_t * ln(lam1/lam0) - (lam1 - lam0))
        alarm when S_t >= h

    O(1) time, one float of state. Used as a *confirm* behind Poisson-FOCuS.

    Parameters
    ----------
    lam1_ratio:
        Smallest flare rate of interest as a multiple of background
        (default :data:`flarecast.constants.POISSON_LAM1_RATIO`, ``~1.8``).
    h:
        Decision threshold (default :data:`flarecast.constants.POISSON_CUSUM_H`).
    """

    __slots__ = ("rho", "h", "S", "_t", "_t0", "_in_event")

    def __init__(self, lam1_ratio: float = POISSON_LAM1_RATIO, h: float = POISSON_CUSUM_H) -> None:
        if lam1_ratio <= 1.0:
            raise ValueError(f"lam1_ratio must be > 1, got {lam1_ratio!r}")
        if h <= 0.0:
            raise ValueError(f"h must be > 0, got {h!r}")
        self.rho: float = float(lam1_ratio)
        self.h: float = float(h)
        self.S: float = 0.0
        self._t: int = 0  # internal bin index (no timestamp in signature)
        self._t0: int | None = None
        self._in_event: bool = False

    def update(self, count: float, mu0: float) -> DetectionState:
        """Fold one count bin in; return a :class:`DetectionState`. O(1).

        The contract signature carries no timestamp, so onset time is reported
        as the **bin index** at which the statistic last left zero (the MLE
        change point in the count stream); the caller maps bins to wall-clock.

        Parameters
        ----------
        count:
            Counts observed this bin (``>= 0``).
        mu0:
            Current background rate ``lam0`` (counts per bin); floored positive.
        """
        c = float(count)
        if c < 0.0:
            c = 0.0
        lam0 = mu0 if mu0 > 0.0 else 1e-9
        lam1 = self.rho * lam0

        self._t += 1
        increment = c * math.log(lam1 / lam0) - (lam1 - lam0)
        prev = self.S
        self.S = max(0.0, self.S + increment)

        # Mark the provisional change point on the transition off zero.
        if prev <= 0.0 and self.S > 0.0:
            self._t0 = self._t

        if self.S >= self.h:
            onset_time = float(self._t0) if self._t0 is not None else float(self._t)
            self.S = 0.0
            self._in_event = True
            return DetectionState(
                onset=True,
                in_event=True,
                statistic=0.0,
                onset_time=onset_time,
                meta={"detector": "PoissonCUSUM"},
            )

        return DetectionState(
            onset=False,
            in_event=self._in_event,
            statistic=self.S,
            onset_time=None,
            meta={"detector": "PoissonCUSUM"},
        )

    def reset(self, in_event: bool = False) -> None:
        """Reset the statistic and latched in-event flag. O(1)."""
        self.S = 0.0
        self._t0 = None
        self._in_event = bool(in_event)

    @property
    def statistic(self) -> float:
        """Current Poisson CUSUM statistic ``S`` (no update)."""
        return self.S
