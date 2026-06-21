"""Adaptive k-sigma threshold with hysteresis + M-of-N persistence (B.4).

The adaptive threshold is the workhorse cross-check trigger (research doc
``03 Section 2.1``). It fires when the sample rises a configurable number of
robust sigmas above a robust baseline, with two false-alarm killers layered on:

* **Hysteresis** -- a high ON threshold ``k_on`` to *enter* the alarmed state
  and a lower OFF threshold ``k_off`` to *leave* it, preventing chattering at
  the flare's decay.
* **M-of-N persistence** -- require ``m`` of the last ``n`` samples to be over
  the ON threshold before declaring onset (the operational GOES philosophy of
  requiring a sustained increase), which multiplies the effective false-alarm
  interval enormously.

``update`` is O(1) time and O(1) state (a fixed-length boolean ring of length
``n`` plus the latched alarm flag).
"""

from __future__ import annotations

from collections import deque

from ..constants import ADAPTIVE_K_OFF, ADAPTIVE_K_ON, ADAPTIVE_M, ADAPTIVE_N
from ..types import DetectionState

__all__ = ["AdaptiveThreshold"]


class AdaptiveThreshold:
    """k-sigma threshold with hysteresis and M-of-N persistence (``03 Section 2.1``).

    Complexity: **O(1) time**, **O(1) state**. The persistence window is a
    ``deque(maxlen=n)`` of booleans with a running true-count, so each update is
    constant-time regardless of stream length.

    Parameters
    ----------
    k_on:
        ON threshold in sigmas (default :data:`flarecast.constants.ADAPTIVE_K_ON`).
    k_off:
        OFF (hysteresis) threshold in sigmas
        (default :data:`flarecast.constants.ADAPTIVE_K_OFF`); must be ``<= k_on``.
    m, n:
        Persistence rule: require ``m`` of the last ``n`` samples over ``k_on``
        to declare onset (defaults
        :data:`flarecast.constants.ADAPTIVE_M` / :data:`~flarecast.constants.ADAPTIVE_N`).
    """

    __slots__ = ("k_on", "k_off", "m", "n", "_hist", "_true", "_in_event")

    def __init__(
        self,
        k_on: float = ADAPTIVE_K_ON,
        k_off: float = ADAPTIVE_K_OFF,
        m: int = ADAPTIVE_M,
        n: int = ADAPTIVE_N,
    ) -> None:
        if k_off > k_on:
            raise ValueError(f"k_off ({k_off}) must be <= k_on ({k_on})")
        if not 1 <= m <= n:
            raise ValueError(f"require 1 <= m <= n, got m={m}, n={n}")
        self.k_on: float = float(k_on)
        self.k_off: float = float(k_off)
        self.m: int = int(m)
        self.n: int = int(n)
        self._hist: deque[bool] = deque(maxlen=self.n)
        self._true: int = 0  # running count of True in _hist
        self._in_event: bool = False

    def update(self, x: float, baseline: float, sigma: float) -> DetectionState:
        """Fold one sample in; return a :class:`DetectionState`. O(1).

        Parameters
        ----------
        x:
            Current sample value.
        baseline:
            Robust baseline ``b_t``.
        sigma:
            Robust scale ``sigma_t``; floored to a tiny positive value.
        """
        s = sigma if sigma > 0.0 else 1e-12
        on_level = baseline + self.k_on * s
        off_level = baseline + self.k_off * s

        over_on = x > on_level

        # Maintain the M-of-N ring with a running True-count (O(1)).
        if len(self._hist) == self.n:
            if self._hist[0]:
                self._true -= 1
        self._hist.append(over_on)
        if over_on:
            self._true += 1

        onset = False
        if not self._in_event:
            # Enter the alarmed state on M-of-N persistence over the ON level.
            if self._true >= self.m:
                self._in_event = True
                onset = True
        else:
            # Stay alarmed until the sample falls back below the OFF level
            # (hysteresis) -- avoids decay-phase chatter.
            if x < off_level:
                self._in_event = False

        # Statistic: standardized excess over baseline (sigmas above baseline),
        # a convenient confidence proxy for downstream fusion.
        statistic = (x - baseline) / s
        return DetectionState(
            onset=onset,
            in_event=self._in_event,
            statistic=statistic,
            onset_time=None,
            meta={
                "detector": "AdaptiveThreshold",
                "k_on": self.k_on,
                "k_off": self.k_off,
                "m_of_n": (self.m, self.n),
                "over_on": over_on,
            },
        )

    def reset(self, in_event: bool = False) -> None:
        """Clear the persistence window and latched alarm flag. O(1)."""
        self._hist.clear()
        self._true = 0
        self._in_event = bool(in_event)
