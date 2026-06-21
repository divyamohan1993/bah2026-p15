"""GOES-style soft-band flare finite-state machine (Section 4.4, B.4).

NOAA/SWPC's operational flare definition is a small finite-state rule on the
1-min long-channel flux, directly portable to SoLEXS (research doc
``03 Section 2.7``). This FSM *frames* a flare -- emitting canonical
**start / peak / end** -- once an onset is suspected:

* **Start** = the first of :data:`~flarecast.constants.FSM_START_CONSEC_MIN`
  consecutive rising samples whose last sample is at least
  :data:`~flarecast.constants.FSM_START_RISE_FACTOR` x the first
  (e.g. 4 consecutive minutes of increase with minute-4 >= 1.4 x minute-1).
  An external ``onset`` (from CUSUM / adaptive threshold) also starts it.
* **Peak** = the maximum flux during the event; declared once the flux has
  fallen back below the peak (i.e. the maximum is confirmed past).
* **End** = decay to the midpoint ``(x_peak + x_start) / 2`` (the GOES rule).

``step`` is O(1) time and O(1) state (a tiny rise-history ring plus a handful of
scalars).
"""

from __future__ import annotations

from collections import deque

from ..constants import FSM_START_CONSEC_MIN, FSM_START_RISE_FACTOR
from ..types import DetectionPhase, DetectionState

__all__ = ["FlareFSM"]


class FlareFSM:
    """Soft-band start/peak/end finite-state machine (research doc ``03 2.7``).

    States map to :class:`flarecast.types.DetectionPhase`:
    ``QUIET`` -> ``RISING`` -> ``PEAK`` -> ``DECAYING`` -> ``QUIET``. The
    returned :class:`DetectionState` carries the boolean transitions
    (``onset`` / ``peak`` / ``end``) plus the current ``phase`` and the
    ``t_start`` / ``t_peak`` / ``x_start`` / ``x_peak`` in ``meta``.

    Complexity: **O(1) time**, **O(1) state** (the rise-history ring has fixed
    length ``consec_min``).

    Parameters
    ----------
    consec_min:
        Consecutive rising samples required for the GOES start rule
        (default :data:`flarecast.constants.FSM_START_CONSEC_MIN`).
    rise_factor:
        Required ratio of the last to the first sample over the start window
        (default :data:`flarecast.constants.FSM_START_RISE_FACTOR`).
    """

    __slots__ = (
        "consec_min",
        "rise_factor",
        "min_rise_ratio",
        "_phase",
        "_hist",
        "_x_start",
        "_t_start",
        "_x_peak",
        "_t_peak",
        "_peak_emitted",
        "_qualified",
        "_onset_emitted",
    )

    def __init__(
        self,
        consec_min: int = FSM_START_CONSEC_MIN,
        rise_factor: float = FSM_START_RISE_FACTOR,
        min_rise_ratio: float | None = None,
    ) -> None:
        if consec_min < 2:
            raise ValueError(f"consec_min must be >= 2, got {consec_min!r}")
        if rise_factor <= 1.0:
            raise ValueError(f"rise_factor must be > 1, got {rise_factor!r}")
        self.consec_min: int = int(consec_min)
        self.rise_factor: float = float(rise_factor)
        # An event is only *confirmed* (and its onset/peak/end emitted) once its
        # peak exceeds its start by this ratio; otherwise it is a noise blip and
        # is discarded silently. Defaults to the GOES rise factor so a CUSUM-
        # triggered start must still grow like a real flare to be catalogued.
        self.min_rise_ratio: float = float(
            rise_factor if min_rise_ratio is None else min_rise_ratio
        )
        if self.min_rise_ratio < 1.0:
            raise ValueError("min_rise_ratio must be >= 1")
        self._phase: DetectionPhase = DetectionPhase.QUIET
        # Ring of the last ``consec_min`` (value, time) pairs for the start rule.
        self._hist: deque[tuple[float, float]] = deque(maxlen=self.consec_min)
        self._x_start: float | None = None
        self._t_start: float | None = None
        self._x_peak: float | None = None
        self._t_peak: float | None = None
        self._peak_emitted: bool = False
        self._qualified: bool = False  # peak has met min_rise_ratio
        self._onset_emitted: bool = False  # onset already reported for this event

    def step(self, x: float, t: float, onset: bool) -> DetectionState:
        """Advance the FSM by one sample; return a :class:`DetectionState`. O(1).

        Parameters
        ----------
        x:
            Current (smoothed) soft-band sample value.
        t:
            Sample timestamp (epoch seconds UTC).
        onset:
            External onset hint (CUSUM / adaptive threshold). The FSM starts on
            *either* this hint or its own GOES consecutive-rise rule.
        """
        x = float(x)
        t = float(t)
        self._hist.append((x, t))

        fired_onset = False
        fired_peak = False
        fired_end = False

        goes_rule = self._goes_start_rule()

        if self._phase is DetectionPhase.QUIET:
            if onset or goes_rule:
                # Anchor the event at the *rise origin* -- the oldest sample in
                # the history ring (lowest of the recent rise) -- so both the
                # reported start time and the rise-factor qualification use the
                # true pre-rise level, whether the start came from the GOES rule
                # or an external onset hint that arrived mid-rise.
                if self._hist:
                    self._x_start, self._t_start = self._hist[0]
                else:
                    self._x_start, self._t_start = x, t
                self._x_peak, self._t_peak = x, t
                self._phase = DetectionPhase.RISING
                self._peak_emitted = False
                self._onset_emitted = False
                # The GOES rule already guarantees the rise factor was met.
                self._qualified = goes_rule or self._meets_rise(x)
                if self._qualified:
                    fired_onset = True
                    self._onset_emitted = True

        elif self._phase in (DetectionPhase.RISING, DetectionPhase.PEAK, DetectionPhase.DECAYING):
            assert self._x_peak is not None and self._x_start is not None
            if x > self._x_peak:
                # New high: still rising; any provisional peak is superseded.
                self._x_peak, self._t_peak = x, t
                self._phase = DetectionPhase.RISING
                self._peak_emitted = False
            else:
                # x has fallen below the running maximum -> we are decaying.
                # Declare the (confirmed) peak exactly once, on this transition.
                if self._qualified and not self._peak_emitted:
                    fired_peak = True
                    self._peak_emitted = True
                self._phase = DetectionPhase.DECAYING

            # An event becomes confirmed once its peak meets the rise factor; the
            # onset is then emitted (late) with the true rise-origin start time.
            if not self._qualified and self._meets_rise(self._x_peak):
                self._qualified = True
            if self._qualified and not self._onset_emitted:
                fired_onset = True
                self._onset_emitted = True
                # If we already slipped into DECAYING before qualifying (a sharp
                # one-sample spike that met the ratio at its top), emit the peak
                # alongside the onset so the event is well-formed.
                if self._phase is DetectionPhase.DECAYING and not self._peak_emitted:
                    fired_peak = True
                    self._peak_emitted = True

            # GOES end rule: decay to the start/peak midpoint. A *qualified* event
            # emits ``end``; an unqualified blip is simply discarded at this point.
            midpoint = 0.5 * (self._x_peak + self._x_start)
            if self._phase is DetectionPhase.DECAYING and x <= midpoint:
                if self._qualified:
                    fired_end = True
                else:
                    # Noise blip that never grew into a flare: drop it silently.
                    self._reset_event()
                    return DetectionState(
                        onset=False,
                        in_event=False,
                        statistic=0.0,
                        meta=self._quiet_meta(),
                    )

        state = DetectionState(
            onset=fired_onset,
            in_event=self._phase is not DetectionPhase.QUIET,
            statistic=(x - self._x_start) if self._x_start is not None else 0.0,
            onset_time=self._t_start if fired_onset else None,
            peak=fired_peak,
            end=fired_end,
            meta={
                "detector": "FlareFSM",
                "phase": self._phase.value,
                "x_start": self._x_start,
                "t_start": self._t_start,
                "x_peak": self._x_peak,
                "t_peak": self._t_peak,
                "qualified": self._qualified,
            },
        )

        if fired_end:
            # Return to rest after reporting the end on this sample.
            self._reset_event()

        return state

    def _meets_rise(self, x_peak: float) -> bool:
        """True if a peak meets the qualifying rise over the event's start. O(1)."""
        if self._x_start is None:
            return False
        if self._x_start <= 0.0:
            return x_peak > 0.0
        return x_peak >= self.min_rise_ratio * self._x_start

    def _quiet_meta(self) -> dict:
        """Metadata dict for a quiescent return value. O(1)."""
        return {
            "detector": "FlareFSM",
            "phase": self._phase.value,
            "x_start": None,
            "t_start": None,
            "x_peak": None,
            "t_peak": None,
            "qualified": False,
        }

    def suppress(self, x: float, t: float) -> None:
        """Note a sample while the FSM is intentionally held quiescent. O(1).

        Used by the band stack during warm-up / data-gap / re-detection-guard
        windows: it keeps the FSM at rest and clears the rise-history ring so a
        partial monotonic run accumulated during the suppressed window cannot
        immediately satisfy the GOES start rule once suppression lifts. Only
        valid when the FSM is in ``QUIET`` (no open event to corrupt).
        """
        if self._phase is DetectionPhase.QUIET:
            self._hist.clear()

    def _goes_start_rule(self) -> bool:
        """True if the last ``consec_min`` samples satisfy the GOES start rule.

        Requires the window to be full, strictly monotonically increasing, and
        the last sample at least ``rise_factor`` x the first. O(consec_min)=O(1).
        """
        if len(self._hist) < self.consec_min:
            return False
        vals = [v for v, _ in self._hist]
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                return False
        if vals[0] <= 0.0:
            # Cannot apply a multiplicative rule to a non-positive baseline;
            # monotonic increase alone suffices in that degenerate case.
            return True
        return vals[-1] >= self.rise_factor * vals[0]

    def _reset_event(self) -> None:
        """Return the FSM to QUIET, clearing event state. O(1)."""
        self._phase = DetectionPhase.QUIET
        self._hist.clear()
        self._x_start = None
        self._t_start = None
        self._x_peak = None
        self._t_peak = None
        self._peak_emitted = False
        self._qualified = False
        self._onset_emitted = False

    @property
    def phase(self) -> DetectionPhase:
        """Current FSM phase."""
        return self._phase
