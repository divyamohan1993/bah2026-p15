"""Tests for the GOES-style soft-band FSM (research 03 Section 2.7).

Inline synthetic FRED (fast-rise / exponential-decay) profiles, pure stdlib.
Verifies the GOES start rule (consecutive rises + rise factor), single peak at
the true maximum, the midpoint end rule, that noise blips below the rise factor
are discarded, and that the standalone FSM and the composed SoftBandDetector both
frame a single clean flare.
"""

from __future__ import annotations

import math
import random

from flarecast.detect.fsm import FlareFSM
from flarecast.detect.stack import SoftBandDetector
from flarecast.types import DetectionPhase


def _fred(t, t_peak, rise_s, decay_s, amp, base=0.0):
    """A FRED bump: Gaussian rise to the peak, exponential decay after."""
    if t < t_peak:
        return base + amp * math.exp(-((t - t_peak) ** 2) / (2 * rise_s * rise_s))
    return base + amp * math.exp(-(t - t_peak) / decay_s)


def test_fsm_start_peak_end_on_fred():
    """A clean FRED yields exactly one onset, one peak (at the max) and one end."""
    fsm = FlareFSM(consec_min=4, rise_factor=1.4)
    # base 10, peak +40 at t=40, then decay. A steep rise (rise_s=3) makes the
    # 4-sample window satisfy the GOES 1.4x rule (as a real impulsive flare does).
    onset = peak = end = 0
    t_peak_reported = None
    x_peak_reported = None
    for t in range(120):
        x = _fred(t, 40, 3.0, 25.0, 40.0, base=10.0)
        st = fsm.step(x, float(t), onset=False)  # rely on the FSM's GOES rule
        onset += int(st.onset)
        if st.peak:
            peak += 1
            t_peak_reported = st.meta["t_peak"]
            x_peak_reported = st.meta["x_peak"]
        end += int(st.end)
    assert onset == 1, f"expected 1 onset, got {onset}"
    assert peak == 1, f"expected 1 peak, got {peak}"
    assert end == 1, f"expected 1 end, got {end}"
    # Peak reported at (or adjacent to) the true maximum time t=40.
    assert t_peak_reported is not None and abs(t_peak_reported - 40) <= 2
    assert x_peak_reported is not None and x_peak_reported > 45.0  # ~base+amp


def test_fsm_end_at_midpoint():
    """The end fires when the decay crosses (peak + start)/2 (the GOES rule)."""
    fsm = FlareFSM(consec_min=4, rise_factor=1.4)
    x_start = x_peak = None
    end_value = None
    for t in range(200):
        x = _fred(t, 30, 6.0, 20.0, 30.0, base=10.0)
        st = fsm.step(x, float(t), onset=False)
        if st.onset:
            x_start = st.meta["x_start"]
        if st.end:
            x_peak = st.meta["x_peak"]
            end_value = x
            break
    assert x_start is not None and x_peak is not None and end_value is not None
    midpoint = 0.5 * (x_peak + x_start)
    # End is declared at the first sample at/below the midpoint.
    assert end_value <= midpoint + 1e-6


def test_fsm_returns_to_quiet_after_end():
    fsm = FlareFSM(consec_min=4, rise_factor=1.4)
    for t in range(200):
        x = _fred(t, 30, 6.0, 20.0, 30.0, base=10.0)
        fsm.step(x, float(t), onset=False)
    assert fsm.phase is DetectionPhase.QUIET


def test_fsm_start_rule_requires_rise_factor():
    """A monotonic rise that fails the 1.4x factor does not start a flare."""
    fsm = FlareFSM(consec_min=4, rise_factor=1.4)
    # 4 increasing samples but only a 1.15x total rise -> below the factor.
    onsets = 0
    for t, x in enumerate([100.0, 105.0, 110.0, 115.0, 110.0, 105.0, 100.0]):
        onsets += int(fsm.step(x, float(t), onset=False).onset)
    assert onsets == 0


def test_fsm_discards_noise_blip_via_qualification():
    """A CUSUM-hinted start that never grows by the rise factor is discarded."""
    fsm = FlareFSM(consec_min=4, rise_factor=1.4, min_rise_ratio=1.4)
    events = 0
    # Feed an onset hint but a signal that wiggles without a real rise.
    rng = random.Random(0)
    for t in range(100):
        x = 10.0 + rng.gauss(0.0, 0.05)
        st = fsm.step(x, float(t), onset=(t == 10))  # spurious hint at t=10
        events += int(st.onset) + int(st.peak) + int(st.end)
    assert events == 0, "an unqualified noise blip leaked an event"


def test_fsm_explicit_onset_hint_starts_qualified_event():
    """An onset hint plus a real subsequent rise produces one well-formed event."""
    fsm = FlareFSM(consec_min=4, rise_factor=1.4)
    onset = peak = end = 0
    for t in range(120):
        x = _fred(t, 40, 3.0, 25.0, 40.0, base=10.0)
        # Provide the hint only at the very first rising sample; the FSM must
        # still confirm via the rise factor (anchored at the rise origin) before
        # emitting.
        st = fsm.step(x, float(t), onset=(t == 35))
        onset += int(st.onset)
        peak += int(st.peak)
        end += int(st.end)
    assert (onset, peak, end) == (1, 1, 1)


def test_soft_stack_single_clean_flare():
    """The composed SoftBandDetector frames one clean flare on a noisy FRED."""
    rng = random.Random(3)
    soft = SoftBandDetector(cadence_s=1.0, warmup_samples=60)
    onsets = peaks = ends = 0
    peak_class = None
    for i in range(800):
        val = 1e-6 + _fred(i, 300, 25.0, 80.0, 2e-5) + rng.gauss(0.0, 2e-8)
        st = soft.update(max(val, 1e-9), float(i))
        onsets += int(st.onset)
        if st.peak:
            peaks += 1
            peak_class = st.meta.get("goes_class")
        ends += int(st.end)
    assert onsets == 1, f"expected 1 onset, got {onsets}"
    assert peaks == 1, f"expected 1 peak, got {peaks}"
    assert ends == 1, f"expected 1 end, got {ends}"
    # Peak flux ~2e-5 -> an M-class flare.
    assert peak_class is not None and peak_class.startswith("M")
