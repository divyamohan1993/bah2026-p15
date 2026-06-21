"""Tests for CUSUM and Poisson-FOCuS onset detectors (research 03 Section 2).

Inline synthetic signals (pure stdlib): a quiet Gaussian/Poisson baseline plus
an injected step/ramp (soft) and a Poisson rate jump (hard). Also verifies that
FOCuS's amortised pruning stays O(1) (bounded curve list) and matches a
brute-force changepoint search, and that the hard-band width gate rejects a
single-bin cosmic-ray spike while still firing on a sustained rate jump.
"""

from __future__ import annotations

import math
import random

from flarecast.detect.cusum import CUSUMDetector
from flarecast.detect.focus import PoissonCUSUM, PoissonFOCuS, _poisson_llr
from flarecast.detect.primitives import P2Quantile
from flarecast.detect.stack import HardBandDetector


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------
def test_cusum_fires_on_step_and_reports_sane_onset():
    """A clean upward step is detected and the MLE onset is near the step."""
    rng = random.Random(1)
    baseline, sigma = 0.0, 1.0
    cu = CUSUMDetector(k_slack=0.5, h=5.0)
    step_t = 200
    fired_t = None
    onset_t = None
    for t in range(400):
        x = rng.gauss(baseline, sigma) + (8.0 if t >= step_t else 0.0)
        st = cu.update(x, baseline, sigma, float(t))
        if st.onset:
            fired_t = t
            onset_t = st.onset_time
            break
    assert fired_t is not None, "CUSUM never fired on a clear step"
    # Detected within a few samples of the step (CUSUM has tiny detection delay).
    assert step_t <= fired_t <= step_t + 5
    # The reported MLE onset is at/just before the step, never far before it.
    # (Pre-step noise can nudge the statistic off zero a few samples early, so
    # the change point can sit slightly before the true step -- that is expected
    # CUSUM behaviour, not an error.)
    assert onset_t is not None
    assert step_t - 8 <= onset_t <= fired_t


def test_cusum_fires_on_gradual_ramp():
    """CUSUM catches a slow ramp that a single-sample threshold would miss."""
    baseline, sigma = 0.0, 1.0
    cu = CUSUMDetector(k_slack=0.5, h=5.0)
    fired = False
    for t in range(300):
        # Gentle deterministic ramp of +0.1 sigma/sample after t=100.
        x = baseline + (0.1 * (t - 100) if t >= 100 else 0.0)
        st = cu.update(x, baseline, sigma, float(t))
        if st.onset:
            fired = True
            assert st.onset_time is not None and st.onset_time >= 95
            break
    assert fired, "CUSUM failed to catch a gradual ramp"


def test_cusum_quiet_baseline_low_false_alarm():
    """On a converged robust baseline, quiet noise yields few CUSUM alarms."""
    rng = random.Random(2)
    med = P2Quantile(0.5)
    mad = P2Quantile(0.5)
    cu = CUSUMDetector(k_slack=0.5, h=5.0)
    alarms = 0
    for t in range(3000):
        x = rng.gauss(0.0, 1.0)
        m = med.update(x)
        a = mad.update(abs(x - m))
        sigma = max(1.4826 * a, 1e-9)
        if cu.update(x, m, sigma, float(t)).onset:
            alarms += 1
    # h=5 sigma is a deliberately conservative ARL0; only a handful of alarms.
    assert alarms < 30


def test_cusum_onset_time_is_last_reset():
    """The onset time equals the time the statistic last left zero."""
    cu = CUSUMDetector(k_slack=0.5, h=3.0)
    # Quiet, then a clean sustained jump; with zero noise the change point is t0.
    onset_t = None
    fired_t = None
    for t in range(100):
        x = 0.0 if t < 50 else 10.0
        st = cu.update(x, 0.0, 1.0, float(t))
        if st.onset:
            onset_t = st.onset_time
            fired_t = t
            break
    assert onset_t == 50.0  # the exact change point
    assert fired_t == 50  # detected on the very first jumped sample


# ---------------------------------------------------------------------------
# Poisson-FOCuS
# ---------------------------------------------------------------------------
def _brute_focus(counts, mu0):
    """O(t^2) reference: max Poisson LLR over all changepoints."""
    csum = [0.0]
    for c in counts:
        csum.append(csum[-1] + c)
    n = len(counts)
    best = 0.0
    for tau in range(n):
        best = max(best, _poisson_llr(csum[n] - csum[tau], n - tau, mu0))
    return best


def test_focus_matches_bruteforce_and_stays_bounded():
    """FOCuS's pruned statistic equals brute force; its curve list stays small."""
    rng = random.Random(7)
    mu0 = 5.0
    counts = []
    for i in range(200):
        lam = 20.0 if i >= 100 else mu0
        counts.append(max(0.0, rng.gauss(lam, math.sqrt(lam))))

    f = PoissonFOCuS(threshold=1e9)  # never resets -> pure-statistic comparison
    max_hull = 0
    for i, c in enumerate(counts):
        st = f.update(c, mu0)
        max_hull = max(max_hull, f.n_curves)
        bf = _brute_focus(counts[: i + 1], mu0)
        assert abs(st.statistic - bf) < 1e-6, f"FOCuS != brute at {i}"
    # The convex-hull pruning keeps the candidate list short (amortised O(1)).
    assert max_hull < 40


def test_focus_fires_on_rate_jump():
    """FOCuS alarms shortly after a genuine Poisson rate increase."""
    rng = random.Random(11)
    mu0 = 5.0
    f = PoissonFOCuS(threshold=15.0)
    jump_t = 120
    fired_t = None
    for i in range(300):
        lam = 25.0 if i >= jump_t else mu0
        c = max(0.0, rng.gauss(lam, math.sqrt(lam)))
        st = f.update(c, mu0)
        if st.onset:
            fired_t = i
            assert st.onset_time is not None
            break
    assert fired_t is not None, "FOCuS never fired on a rate jump"
    assert jump_t <= fired_t <= jump_t + 10


def test_hard_stack_width_gate_rejects_cosmic_ray():
    """A single-bin cosmic ray is rejected by the width gate; a real burst fires.

    FOCuS itself is multi-scale and *will* react to a huge one-bin excess, so the
    cosmic-ray rejection lives in the HardBandDetector's width gate (research doc
    03 Section 2.6/7): an excursion must persist >= SPIKE_WIDTH_MIN_BINS bins to
    reach the detector.
    """
    rng = random.Random(5)
    hard = HardBandDetector(cadence_s=1.0, focus_threshold=12.0)
    onsets = []
    cr_bin = 100
    burst_lo, burst_hi = 200, 235
    for i in range(400):
        lam = 40.0 if burst_lo <= i < burst_hi else 5.0
        c = max(0.0, rng.gauss(lam, math.sqrt(lam)))
        if i == cr_bin:
            c = 600.0  # one-bin cosmic ray
        st = hard.update(c, float(i))
        if st.onset:
            onsets.append(i)
    # At least one spike was rejected by the width gate...
    assert hard._spike_count >= 1  # noqa: SLF001 (assert internal counter)
    # ...and no onset fired on the isolated cosmic-ray bin...
    assert all(abs(o - cr_bin) > 2 for o in onsets), (
        f"cosmic-ray bin {cr_bin} wrongly triggered an onset: {onsets}"
    )
    # ...while the sustained burst was detected.
    assert any(burst_lo <= o <= burst_hi + 5 for o in onsets), (
        f"sustained burst was missed: {onsets}"
    )


# ---------------------------------------------------------------------------
# Poisson CUSUM
# ---------------------------------------------------------------------------
def test_poisson_cusum_fires_on_rate_jump():
    rng = random.Random(3)
    mu0 = 5.0
    pc = PoissonCUSUM(lam1_ratio=1.8, h=5.0)
    jump_t = 100
    fired = False
    for i in range(250):
        lam = 20.0 if i >= jump_t else mu0
        c = max(0.0, rng.gauss(lam, math.sqrt(lam)))
        st = pc.update(c, mu0)
        if st.onset:
            fired = True
            assert st.onset_time is not None
            break
    assert fired, "Poisson CUSUM never fired on a rate jump"
