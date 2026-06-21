"""Composed per-band detection stacks (ARCHITECTURE.md Section 4.2/4.4, B.4).

These classes wire the O(1) primitives and detectors into one
``.update(sample) -> DetectionState`` per band, exactly as the two-band stack in
research doc ``03 Section 7`` prescribes:

Soft band (:class:`SoftBandDetector`)
    Hampel despike -> gap gate -> robust baseline (P^2 median + MAD, **gated**
    during a flare) -> CUSUM primary onset (with adaptive k-sigma + M-of-N
    cross-check and an IIR FRED matched-filter confirmer) -> GOES-style FSM for
    canonical start/peak/end -> GOES-equivalent class at the peak.

Hard band (:class:`HardBandDetector`)
    Width-gate despike (cosmic-ray rejection) -> gap gate -> Poisson background
    ``mu0`` (gated EMA of counts) -> Poisson-FOCuS primary onset (with a Poisson
    CUSUM confirm).

Every per-sample update is **O(1) time and O(1) state**: each composed piece is
itself O(1), and there is a fixed, small number of them.

The *critical subtlety* (research doc ``03 Section 1.2 / 2.1``) is the **gated
baseline**: while a flare is in progress the baseline / scale estimators are
frozen so the flare does not contaminate the very baseline the detector measures
its excess against.
"""

from __future__ import annotations

from ..constants import (
    CUSUM_H,
    CUSUM_K_SLACK,
    DEDUP_GUARD_S,
    DEFAULT_EMA_ALPHA,
    HAMPEL_K,
    POISSON_CUSUM_H,
    POISSON_LAM1_RATIO,
    SPIKE_WIDTH_MIN_BINS,
)
from ..types import DetectionState
from .classify import classify_flux
from .cusum import CUSUMDetector
from .focus import PoissonCUSUM, PoissonFOCuS
from .fsm import FlareFSM
from .matched import IIRMatchedFilter
from .primitives import EMA, HampelDespiker, P2Quantile
from .threshold import AdaptiveThreshold

__all__ = ["SoftBandDetector", "HardBandDetector"]

_MAD_SCALE = 1.4826  # MAD -> Gaussian sigma


class SoftBandDetector:
    """Soft-band (SoLEXS / GOES-long) composed detector (research doc ``03 Section 7``).

    ``update(x, t)`` returns a :class:`DetectionState` whose ``onset`` /
    ``peak`` / ``end`` come from the framing FSM, whose ``in_event`` reflects the
    flare being open, and whose ``meta`` carries the contributing-detector flags,
    the running statistics, and -- on ``peak`` -- the GOES-equivalent class and
    peak flux. **O(1)** per sample.

    Parameters
    ----------
    cadence_s:
        Sample spacing (s); recorded in ``meta`` and used for gap detection.
    cusum_k, cusum_h:
        CUSUM slack and decision interval.
    hampel_window, hampel_k:
        Despiker window and threshold.
    gap_factor:
        A timestamp jump greater than ``gap_factor * cadence_s`` is treated as a
        data gap: state is held and alarms suppressed for one sample after
        resume (research doc ``03 Section 5``).
    median_min_scale:
        Floor on the robust scale so a perfectly flat baseline cannot make the
        thresholds collapse to zero.
    """

    def __init__(
        self,
        cadence_s: float = 1.0,
        *,
        cusum_k: float = CUSUM_K_SLACK,
        cusum_h: float = CUSUM_H,
        hampel_window: int = 7,
        hampel_k: float = HAMPEL_K,
        gap_factor: float = 3.0,
        median_min_scale: float = 1e-12,
        matched_alpha_rise: float = 0.5,
        matched_alpha_decay: float = 0.97,
        warmup_samples: int = 30,
        guard_s: float = DEDUP_GUARD_S,
        **_params: object,
    ) -> None:
        self.cadence_s = float(cadence_s)
        self.gap_factor = float(gap_factor)
        self.median_min_scale = float(median_min_scale)
        self.warmup_samples = int(warmup_samples)
        self.guard_s = float(guard_s)

        self._despiker = HampelDespiker(hampel_window, k=hampel_k)
        self._median = P2Quantile(0.5)
        self._mad = P2Quantile(0.5)  # tracks median of |x - median| -> MAD
        self._cusum = CUSUMDetector(k_slack=cusum_k, h=cusum_h)
        self._thresh = AdaptiveThreshold()
        self._matched = IIRMatchedFilter(matched_alpha_rise, matched_alpha_decay)
        self._fsm = FlareFSM()

        self._last_t: float | None = None
        self._in_event: bool = False
        self._suppress_next: bool = False
        self._spike_count: int = 0
        self._n_seen: int = 0
        self._guard_until: float | None = None  # re-detection guard after end
        self._recovering: bool = False  # decay-tail still above baseline
        self._recover_baseline: float = 0.0
        self._recover_until: float | None = None  # safety cap on recovery

    #: During tail recovery the baseline is allowed to re-adapt (so it tracks the
    #: decaying tail back down); onsets stay suppressed until the excess over the
    #: re-converged baseline falls below this many sigmas -- a clean
    #: return-to-baseline de-duplication condition (research doc 03 Section 4.3).
    RECOVERY_RESUME_SIGMAS: float = 5.0

    def update(self, x: float, t: float) -> DetectionState:
        """Process one soft-band sample; return a :class:`DetectionState`. O(1)."""
        x = float(x)
        t = float(t)
        self._n_seen += 1
        warming = self._n_seen <= self.warmup_samples

        # --- Gap gate: hold state across a timestamp jump, suppress one sample.
        gap = False
        if self._last_t is not None and (t - self._last_t) > self.gap_factor * self.cadence_s:
            gap = True
            self._suppress_next = True
        self._last_t = t

        # --- Despike (Hampel). A despiked sample is replaced by the median.
        clean, is_spike = self._despiker.update(x)
        if is_spike:
            self._spike_count += 1

        # --- Robust baseline (P^2 median + MAD), FROZEN while a flare is open
        #     *or* its decay tail is still recovering, so neither contaminates the
        #     quiet-Sun baseline (research doc 03 Section 1.2). The frozen value
        #     and scale are exactly what the "clean return-to-baseline" test
        #     below compares against.
        if not self._in_event and not self._recovering:
            med = self._median.update(clean)
            mad = self._mad.update(abs(clean - med))
        else:
            med = self._median.value
            mad = self._mad.value
        sigma = max(_MAD_SCALE * mad, self.median_min_scale)

        # --- Recovery exit: the decay tail is back at quiet-Sun once the flux
        #     returns to within a few (frozen) sigmas of the pre-flare baseline.
        #     A generous time cap is a safety valve so a never-recovering tail
        #     (e.g. an elevated SEP background) cannot wedge detection off forever.
        if self._recovering:
            recovered = (clean - self._recover_baseline) <= (self.RECOVERY_RESUME_SIGMAS * sigma)
            timed_out = self._recover_until is not None and t >= self._recover_until
            if recovered or timed_out:
                self._recovering = False

        # --- Primary + cross-check onset detectors.
        cusum_state = self._cusum.update(clean, med, sigma, t)
        thr_state = self._thresh.update(clean, med, sigma)
        fredness = self._matched.update(clean)

        # During warm-up the robust baseline/scale are not yet stable, so keep
        # the change detectors reset and emit no onset (avoids cold-start false
        # alarms; research doc 03 Section 1.1 bias-correction / Section 5).
        if warming:
            self._cusum.reset(in_event=False)
            self._thresh.reset(in_event=False)

        # Re-detection guard: after a flare ends, suppress new onsets for
        # ``guard_s`` AND until the decay tail has recovered (research doc
        # 03 Section 4.3). Only applies while no event is currently open.
        in_guard = not self._in_event and (
            self._recovering or (self._guard_until is not None and t < self._guard_until)
        )

        suppressed = self._suppress_next or gap or warming or in_guard
        onset_hint = (cusum_state.onset or thr_state.onset) and not suppressed
        self._suppress_next = False  # only the first post-gap sample is gated

        # --- FSM framing (start / peak / end). Use the despiked sample.
        # While suppressed (warm-up / gap / guard) and at rest, keep the FSM
        # quiescent so neither the onset hint nor its internal GOES start rule
        # can open a spurious event; its rise history is cleared so a clean
        # post-guard rise still starts normally.
        if suppressed and not self._in_event:
            self._fsm.suppress(clean, t)
            fsm_state = DetectionState(
                onset=False,
                in_event=False,
                statistic=0.0,
                meta={
                    "detector": "FlareFSM",
                    "phase": self._fsm.phase.value,
                    "x_start": None,
                    "t_start": None,
                    "x_peak": None,
                    "t_peak": None,
                },
            )
        else:
            fsm_state = self._fsm.step(clean, t, onset_hint)
        # The FSM reports ``in_event=True`` on the very sample it emits ``end``
        # (the event was still open up to that sample); for the stack's guard
        # logic the event is finished, so treat an end sample as not-in-event.
        self._in_event = fsm_state.in_event and not fsm_state.end

        # On flare end, release the gated detectors, arm the time guard, and
        # enter the tail-recovery state (keep the baseline frozen + onsets off
        # until the decaying tail returns near the pre-flare baseline).
        if fsm_state.end:
            self._cusum.reset(in_event=False)
            self._thresh.reset(in_event=False)
            self._guard_until = t + self.guard_s
            self._recovering = True
            self._recover_baseline = med
            # Safety cap: at most ~100x the post-end guard before forcing resume.
            self._recover_until = t + 100.0 * self.guard_s

        meta: dict[str, object] = {
            "band": "soft",
            "cadence_s": self.cadence_s,
            "baseline": med,
            "sigma": sigma,
            "cusum_S": cusum_state.statistic,
            "cusum_onset": cusum_state.onset,
            "threshold_onset": thr_state.onset,
            "fredness": fredness,
            "phase": fsm_state.meta["phase"] if fsm_state.meta else None,
            "spike_rejected": self._spike_count,
            "data_gap": gap,
            "detectors": self._fired_detectors(cusum_state, thr_state),
        }

        # On the peak, attach the GOES-equivalent class + peak flux for the
        # catalogue (the input is treated as a GOES-scale flux in W m^-2).
        if fsm_state.peak and fsm_state.meta is not None:
            x_peak = fsm_state.meta.get("x_peak")
            if isinstance(x_peak, (int, float)):
                meta["peak_flux"] = float(x_peak)
                meta["goes_class"] = classify_flux(float(x_peak))

        return DetectionState(
            onset=fsm_state.onset,
            in_event=self._in_event,
            statistic=cusum_state.statistic,
            onset_time=fsm_state.onset_time,
            peak=fsm_state.peak,
            end=fsm_state.end,
            meta=meta,
        )

    @staticmethod
    def _fired_detectors(cusum_state: DetectionState, thr_state: DetectionState) -> list[str]:
        fired: list[str] = []
        if cusum_state.onset:
            fired.append("CUSUM")
        if thr_state.onset:
            fired.append("threshold")
        return fired

    @property
    def in_event(self) -> bool:
        """True while a soft-band flare is open."""
        return self._in_event


class HardBandDetector:
    """Hard-band (HEL1OS) composed detector (research doc ``03 Section 7``).

    ``update(count, t)`` returns a :class:`DetectionState`. The primary onset is
    Poisson-FOCuS (multi-scale, correct low-count Poisson statistics), confirmed
    by a Poisson CUSUM. A **width gate** removes one-bin cosmic-ray excursions
    *before* the detectors see them: a Hampel-flagged outlier is held back until
    the excursion has persisted for ``spike_width_min`` consecutive bins (a real
    flare persists; a cosmic ray does not). **O(1)** per sample.

    Parameters
    ----------
    cadence_s:
        Bin spacing (s); recorded in ``meta`` and used for gap detection.
    bg_alpha:
        Forgetting factor of the gated background-rate EMA ``mu0``.
    focus_threshold:
        Poisson-FOCuS alarm threshold (set from a target ARL_0).
    lam1_ratio, cusum_h:
        Poisson CUSUM confirm parameters.
    spike_width_min:
        Minimum sustained width (bins) for an excursion to be a flare and not a
        cosmic ray (default :data:`flarecast.constants.SPIKE_WIDTH_MIN_BINS`).
    hampel_window, hampel_k:
        Despiker window and threshold used to *flag* candidate spikes.
    gap_factor:
        Timestamp-jump multiple that counts as a data gap.
    """

    def __init__(
        self,
        cadence_s: float = 1.0,
        *,
        bg_alpha: float = DEFAULT_EMA_ALPHA,
        focus_threshold: float = 10.0,
        lam1_ratio: float = POISSON_LAM1_RATIO,
        cusum_h: float = POISSON_CUSUM_H,
        spike_width_min: int = SPIKE_WIDTH_MIN_BINS,
        hampel_window: int = 7,
        hampel_k: float = HAMPEL_K,
        gap_factor: float = 3.0,
        mu0_floor: float = 1e-6,
        warmup_samples: int = 30,
        guard_s: float = DEDUP_GUARD_S,
        **_params: object,
    ) -> None:
        self.cadence_s = float(cadence_s)
        self.gap_factor = float(gap_factor)
        self.spike_width_min = int(spike_width_min)
        self.mu0_floor = float(mu0_floor)
        self.warmup_samples = int(warmup_samples)
        self.guard_s = float(guard_s)

        self._bg = EMA(bg_alpha)
        self._bg_seeded = False
        self._despiker = HampelDespiker(hampel_window, k=hampel_k)
        self._focus = PoissonFOCuS(threshold=focus_threshold)
        self._pcusum = PoissonCUSUM(lam1_ratio=lam1_ratio, h=cusum_h)

        self._last_t: float | None = None
        self._in_event: bool = False
        self._suppress_next: bool = False
        self._spike_count: int = 0
        self._excursion_run: int = 0  # consecutive flagged-outlier bins
        self._n_seen: int = 0
        self._guard_until: float | None = None
        self._low_run: int = 0  # consecutive at-background bins (for end)
        self._mu0_frozen: float = 0.0  # background frozen at burst onset

    #: Counts must sit at/below ``END_FACTOR x mu0`` for ``END_CONSEC`` consecutive
    #: bins before a burst is declared over (a sustained return-to-background, so
    #: a single Poisson dip mid-burst does not split one burst into many).
    END_FACTOR: float = 1.5
    END_CONSEC: int = 5

    def update(self, count: float, t: float) -> DetectionState:
        """Process one hard-band count bin; return a :class:`DetectionState`. O(1)."""
        c = float(count)
        if c < 0.0:
            c = 0.0
        t = float(t)
        self._n_seen += 1
        warming = self._n_seen <= self.warmup_samples

        # Seed the background EMA to the first observed count so mu0 starts near
        # the truth instead of crawling up from zero (a cold-start false-alarm
        # source). Research doc 03 Section 1.1 (bias correction / seeding).
        if not self._bg_seeded:
            self._bg.m = c
            self._bg_seeded = True

        # --- Gap gate.
        gap = False
        if self._last_t is not None and (t - self._last_t) > self.gap_factor * self.cadence_s:
            gap = True
            self._suppress_next = True
        self._last_t = t

        # --- Width-gated despike (cosmic-ray rejection). A Hampel-flagged bin is
        #     a candidate cosmic ray; we only let an excursion through to the
        #     detectors once it has persisted spike_width_min bins.
        _clean, is_outlier = self._despiker.update(c)
        if is_outlier:
            self._excursion_run += 1
        else:
            self._excursion_run = 0

        gated_count = c
        rejected = False
        if is_outlier and self._excursion_run < self.spike_width_min:
            # Too short to be a flare yet: substitute the background so a single
            # cosmic ray cannot spike FOCuS/CUSUM.
            gated_count = self._bg.value
            rejected = True
            self._spike_count += 1

        # --- Background rate mu0. While a burst is open it is frozen at the
        #     pre-onset value (gated EMA; research doc 03 Section 1.2) so the
        #     burst counts cannot contaminate the background the detectors test
        #     against. Otherwise the gated EMA tracks the quiet background.
        if self._in_event:
            mu0 = max(self._mu0_frozen, self.mu0_floor)
        else:
            if not rejected:
                self._bg.update(gated_count)
            mu0 = max(self._bg.value, self.mu0_floor)

        # --- Poisson detectors.
        focus_state = self._focus.update(gated_count, mu0)
        pcusum_state = self._pcusum.update(gated_count, mu0)

        # During warm-up the background is still settling; keep the Poisson
        # detectors reset and emit no onset (avoids cold-start false alarms).
        if warming:
            self._focus.reset(in_event=False)
            self._pcusum.reset(in_event=False)

        # Re-detection guard after a burst ends (research doc 03 Section 4.3).
        in_guard = not self._in_event and self._guard_until is not None and t < self._guard_until

        suppressed = self._suppress_next or gap or warming or in_guard
        self._suppress_next = False

        # Only the *first* detection opens the burst; while already in_event we
        # do not re-fire onset (one onset per physical burst).
        onset = (focus_state.onset or pcusum_state.onset) and not suppressed and not self._in_event
        if onset:
            self._in_event = True
            self._mu0_frozen = mu0  # freeze background at burst onset
            self._low_run = 0

        # End-of-burst: require a *sustained* return to background (END_CONSEC
        # consecutive at-background bins) so a single Poisson dip mid-burst does
        # not prematurely end (and then re-trigger) the burst.
        if self._in_event:
            if gated_count <= self.END_FACTOR * mu0:
                self._low_run += 1
            else:
                self._low_run = 0
            if self._low_run >= self.END_CONSEC:
                self._in_event = False
                self._low_run = 0
                self._focus.reset(in_event=False)
                self._pcusum.reset(in_event=False)
                self._guard_until = t + self.guard_s
                # Re-seed the quiet-background EMA to the current (low) level so
                # it resumes tracking from background, not the stale frozen value.
                self._bg.m = gated_count

        # Onset time: prefer FOCuS's MLE change-point bin, else CUSUM's.
        onset_time = None
        if onset:
            if focus_state.onset and focus_state.onset_time is not None:
                onset_time = focus_state.onset_time
            elif pcusum_state.onset and pcusum_state.onset_time is not None:
                onset_time = pcusum_state.onset_time

        fired = []
        if focus_state.onset:
            fired.append("FOCuS")
        if pcusum_state.onset:
            fired.append("PoissonCUSUM")

        meta = {
            "band": "hard",
            "cadence_s": self.cadence_s,
            "mu0": mu0,
            "focus_stat": focus_state.statistic,
            "focus_onset": focus_state.onset,
            "pcusum_onset": pcusum_state.onset,
            "spike_rejected": self._spike_count,
            "data_gap": gap,
            "detectors": fired,
        }

        return DetectionState(
            onset=onset,
            in_event=self._in_event,
            statistic=focus_state.statistic,
            onset_time=onset_time,
            peak=False,
            end=False,
            meta=meta,
        )

    @property
    def in_event(self) -> bool:
        """True while a hard-band burst is open."""
        return self._in_event
