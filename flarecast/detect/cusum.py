"""Soft-band CUSUM onset detector (ARCHITECTURE.md Section 4.4, B.4).

The CUSUM (cumulative sum) test is the recommended **primary soft-band onset
detector** (research doc ``03 Section 2.2``): O(1) time, a single float of state,
an analytic false-alarm rate via ARL curves, and -- crucially for the
catalogue -- a free maximum-likelihood **onset time** (the instant the statistic
last reset to zero before the alarm).

We implement the one-sided *upper* CUSUM (we only care about flux *increases*)::

    S_t = max(0, S_{t-1} + (x_t - baseline_t) - k_slack * sigma_t)
    alarm when S_t > h * sigma_t
    on alarm: record onset_time = last-reset time, reset S_t = 0
"""

from __future__ import annotations

from ..constants import CUSUM_H, CUSUM_K_SLACK
from ..types import DetectionState

__all__ = ["CUSUMDetector"]


class CUSUMDetector:
    """One-sided upper CUSUM for a flux increase (research doc ``03 Section 2.2``).

    ``update`` runs in **O(1) time** and the whole detector is **O(1) state**
    (one running statistic ``S`` plus the provisional change-point time). The
    slack ``k_slack`` is the smallest shift (in sigma units) the detector
    chases; ``h`` is the decision interval that trades ARL_0 (mean time between
    false alarms) against detection delay.

    Onset-time semantics
    --------------------
    Whenever ``S`` is at zero, the *next* sample's time becomes the provisional
    change point ``t0``. The MLE flare-start reported on ``onset`` is that
    ``t0`` -- the last time the cumulative evidence was reset before crossing the
    threshold (research doc ``03 Section 2.2``: "the time CUSUM last reset to 0
    before the alarm is the maximum-likelihood change point").

    Parameters
    ----------
    k_slack:
        Slack ``k_c = delta / (2 sigma)`` in sigma units
        (default :data:`flarecast.constants.CUSUM_K_SLACK`).
    h:
        Decision interval ``h_c`` in sigma units
        (default :data:`flarecast.constants.CUSUM_H`).
    """

    __slots__ = ("k", "h", "S", "_t0", "_in_event")

    def __init__(self, k_slack: float = CUSUM_K_SLACK, h: float = CUSUM_H) -> None:
        if k_slack < 0.0:
            raise ValueError(f"k_slack must be >= 0, got {k_slack!r}")
        if h <= 0.0:
            raise ValueError(f"h must be > 0, got {h!r}")
        self.k: float = float(k_slack)
        self.h: float = float(h)
        self.S: float = 0.0
        self._t0: float | None = None
        self._in_event: bool = False

    def update(self, x: float, baseline: float, sigma: float, t: float) -> DetectionState:
        """Fold one sample in and return a :class:`DetectionState`. O(1).

        Parameters
        ----------
        x:
            Current sample value.
        baseline:
            Robust baseline ``b_t`` (e.g. P^2 median) to subtract.
        sigma:
            Robust scale ``sigma_t`` (e.g. ``1.4826 * MAD`` or EWMV sd). A
            non-positive sigma is floored to a tiny positive number so the
            detector degrades gracefully instead of dividing by zero.
        t:
            Sample timestamp (epoch seconds UTC).
        """
        # Floor sigma so a flat/degenerate scale never breaks the arithmetic.
        s = sigma if sigma > 0.0 else 1e-12

        # The instant the statistic sits at zero is the provisional change point.
        if self.S <= 0.0:
            self._t0 = t

        self.S = max(0.0, self.S + (x - baseline) - self.k * s)

        threshold = self.h * s
        if self.S > threshold:
            onset_time = self._t0
            self.S = 0.0
            self._in_event = True
            return DetectionState(
                onset=True,
                in_event=True,
                statistic=0.0,
                onset_time=onset_time,
                meta={"detector": "CUSUM", "threshold": threshold},
            )

        # No alarm this sample. ``in_event`` stays latched until the caller (the
        # band stack / FSM) clears it via :meth:`reset` at flare end.
        return DetectionState(
            onset=False,
            in_event=self._in_event,
            statistic=self.S,
            onset_time=None,
            meta={"detector": "CUSUM", "threshold": threshold},
        )

    def reset(self, in_event: bool = False) -> None:
        """Reset the statistic and the latched in-event flag. O(1).

        Called by the band stack when the FSM declares the flare ended so the
        next genuine rise opens a fresh change-point search.
        """
        self.S = 0.0
        self._t0 = None
        self._in_event = bool(in_event)

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic ``S`` (no update)."""
        return self.S
