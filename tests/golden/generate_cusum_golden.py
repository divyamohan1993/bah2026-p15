"""Generate ``cusum_golden.json`` -- the cross-substrate parity contract.

ARCHITECTURE.md Appendix B.8 fixes a hard invariant::

    edge/src/detector.ts EMA+CUSUM math and flarecast/detect/{primitives,cusum}.py
    produce the SAME onset decisions on the same input sequence (same alpha, k, h).

This module is the *single generator* of the golden vector both substrates are
scored against. It drives the **real** frozen-foundation classes
(:class:`flarecast.detect.primitives.EMA`, :class:`~flarecast.detect.primitives.EWMV`,
and :class:`flarecast.detect.cusum.CUSUMDetector`) through the minimal hot-path
pipeline that the edge Durable Object (``edge/src/detector.ts``) mirrors
byte-for-byte, and writes the input vector, the per-step CUSUM statistic, the
per-step baseline/sigma, and the onset indices to ``cusum_golden.json``.

The pipeline (identical on both substrates -- this is the contract)
---------------------------------------------------------------------
For each sample ``x_t`` at integer time ``t``:

1. **Gated baseline / scale.** While *not* in an event, fold ``x_t`` into an EMA
   baseline and an EWMV mean+variance; while *in* an event, FREEZE both so the
   flare cannot contaminate the very baseline it is measured against
   (ARCHITECTURE.md Section 4.3 "gated baseline" subtlety). ``sigma`` is the
   EWMV standard deviation, floored to a small positive constant.
2. **CUSUM update.** Feed ``(x_t, baseline, sigma, t)`` to the one-sided upper
   :class:`CUSUMDetector` (``k = CUSUM_K_SLACK``, ``h = CUSUM_H``). It returns an
   onset flag, the MLE onset time, and the running statistic ``S``.
3. **Event latch (one onset per flare).** The CUSUM alarm only counts as a
   reported onset when *not* already in an event; while ``in_event`` is latched
   the CUSUM is still updated (so its statistic evolves identically on both
   substrates) but repeat alarms are not re-reported. ``in_event`` is cleared by
   the return-to-baseline rule below so the baseline can re-adapt for the next
   flare. Both substrates use the *same* rule, so their gating -- and therefore
   their statistics and onsets -- agree exactly.

Determinism
-----------
The input vector is constructed from pure-stdlib arithmetic with a fixed seed,
so re-running this script reproduces ``cusum_golden.json`` byte-for-byte. The
parity tests (``tests/test_parity.py`` for Python, ``edge/test/parity.mjs`` for
TS/JS) *load* the committed JSON and assert their implementation reproduces it;
they do not call this generator, so the golden is a frozen regression anchor.

Run ``python tests/golden/generate_cusum_golden.py`` to (re)generate.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any

from flarecast.constants import CUSUM_H, CUSUM_K_SLACK, DEFAULT_EMA_ALPHA
from flarecast.detect.cusum import CUSUMDetector
from flarecast.detect.primitives import EMA, EWMV

# --- Pipeline constants: these ARE the cross-substrate contract. -----------
#: EMA / EWMV forgetting factor for the baseline + scale estimators.
PARITY_ALPHA: float = DEFAULT_EMA_ALPHA
#: CUSUM slack (sigma units).
PARITY_K: float = CUSUM_K_SLACK
#: CUSUM decision interval (sigma units).
PARITY_H: float = CUSUM_H
#: Floor on sigma so a flat/degenerate scale never breaks the arithmetic.
PARITY_SIGMA_FLOOR: float = 1e-12
#: Return-to-baseline event-exit rule: clear ``in_event`` once the sample falls
#: back within this many (frozen) sigmas of the frozen baseline.
PARITY_EXIT_SIGMAS: float = 1.0

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "cusum_golden.json")


def build_input_vector() -> list[float]:
    """Construct the fixed input vector (deterministic, pure stdlib).

    A quiet baseline (~1.0) with small fixed pseudo-random jitter, into which
    three Gaussian-shaped excursions of increasing amplitude are injected so the
    CUSUM fires multiple, well-separated onsets with clean returns to baseline in
    between. No numpy -- the same arithmetic runs anywhere.
    """
    rng = random.Random(20260620)  # fixed seed -> reproducible
    n = 240
    base = 1.0
    x = [base + rng.uniform(-0.02, 0.02) for _ in range(n)]

    # Three flares: (center, amplitude, width). Increasing amplitude.
    flares = [(60, 0.6, 6.0), (120, 1.5, 7.0), (180, 4.0, 8.0)]
    for center, amp, width in flares:
        for i in range(n):
            x[i] += amp * math.exp(-0.5 * ((i - center) / width) ** 2)
    return [round(v, 9) for v in x]


def run_pipeline(x: list[float]) -> dict[str, Any]:
    """Run the minimal EMA+EWMV+CUSUM hot-path pipeline over ``x``.

    Returns the per-step statistic / baseline / sigma / in_event arrays and the
    list of onset indices + onset times -- exactly the fields the TS mirror must
    reproduce. See the module docstring for the (contractual) pipeline steps.
    """
    ema = EMA(PARITY_ALPHA, x0=x[0] if x else 0.0)
    ewmv = EWMV(PARITY_ALPHA, x0=x[0] if x else 0.0)
    cusum = CUSUMDetector(k_slack=PARITY_K, h=PARITY_H)

    in_event = False
    frozen_baseline = 0.0
    frozen_sigma = PARITY_SIGMA_FLOOR

    statistic: list[float] = []
    baseline_out: list[float] = []
    sigma_out: list[float] = []
    in_event_out: list[bool] = []
    onset_indices: list[int] = []
    onset_times: list[float] = []

    for i, xi in enumerate(x):
        t = float(i)

        # 1. Gated baseline / scale (frozen while in an event).
        if not in_event:
            baseline = ema.update(xi)
            _mean, var = ewmv.update(xi)
            sigma = math.sqrt(var) if var > 0.0 else 0.0
            sigma = sigma if sigma > PARITY_SIGMA_FLOOR else PARITY_SIGMA_FLOOR
            frozen_baseline = baseline
            frozen_sigma = sigma
        else:
            baseline = frozen_baseline
            sigma = frozen_sigma

        # 2. CUSUM update (the load-bearing parity math). Always called so the
        #    statistic evolves identically on both substrates.
        state = cusum.update(xi, baseline, sigma, t)

        # 3a. Onset is reported only on the FIRST alarm of a burst (one onset per
        #     physical flare; cf. HardBandDetector "not self._in_event").
        if state.onset and not in_event:
            in_event = True
            onset_indices.append(i)
            onset_times.append(
                float(state.onset_time) if state.onset_time is not None else t
            )

        # 3b. Event-exit: clean return to (frozen) baseline re-arms detection.
        elif in_event and (xi - frozen_baseline) <= PARITY_EXIT_SIGMAS * frozen_sigma:
            in_event = False
            cusum.reset(in_event=False)

        statistic.append(round(state.statistic, 9))
        baseline_out.append(round(baseline, 9))
        sigma_out.append(round(sigma, 9))
        in_event_out.append(in_event)

    return {
        "statistic": statistic,
        "baseline": baseline_out,
        "sigma": sigma_out,
        "in_event": in_event_out,
        "onset_indices": onset_indices,
        "onset_times": onset_times,
    }


def build_golden() -> dict[str, Any]:
    """Assemble the full golden document (params + input + expected outputs)."""
    x = build_input_vector()
    out = run_pipeline(x)
    return {
        "_comment": (
            "Cross-substrate parity golden (ARCHITECTURE.md Appendix B.8). "
            "Regenerate with: python tests/golden/generate_cusum_golden.py. "
            "Python (tests/test_parity.py) and TS/JS (edge/test/parity.mjs) "
            "both reproduce these arrays from the same input + params."
        ),
        "pipeline": "EMA-baseline + EWMV-sigma + one-sided-upper CUSUM (gated)",
        "params": {
            "alpha": PARITY_ALPHA,
            "k_slack": PARITY_K,
            "h": PARITY_H,
            "sigma_floor": PARITY_SIGMA_FLOOR,
            "exit_sigmas": PARITY_EXIT_SIGMAS,
        },
        "n": len(x),
        "input": x,
        "expected": out,
    }


def main() -> int:
    golden = build_golden()
    with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2, sort_keys=False)
        fh.write("\n")
    n_onsets = len(golden["expected"]["onset_indices"])
    print(
        f"wrote {GOLDEN_PATH}: n={golden['n']} samples, "
        f"{n_onsets} onsets at {golden['expected']['onset_indices']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
