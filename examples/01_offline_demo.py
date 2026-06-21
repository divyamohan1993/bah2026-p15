#!/usr/bin/env python3
"""Example 01 -- end-to-end OFFLINE pipeline: synth -> detect -> catalogue.

The keystone offline proof (ARCHITECTURE.md Section 6 / Appendix A): generate
physics-based SoLEXS-like *soft* + HEL1OS-like *hard* X-ray light curves with a
ground-truth flare list, stream them through the **O(1)** dual-band detectors
(CUSUM soft onset + GOES-style FSM; Poisson-FOCuS hard onset), associate the
per-band detections into a Neupert-aware **master catalogue**, and print a
detected-vs-truth table. Runs with **zero network and zero credentials**.

Run::

    python examples/01_offline_demo.py
    python examples/01_offline_demo.py --hours 12 --flares 6 --seed 7

This reuses :func:`flarecast.cli.main.run_offline_pipeline` (the same flow the
``flarecast demo`` CLI runs) so the example stays faithful to the shipped
pipeline. The catalogue is persisted to SQLite + JSON under a temp dir.
"""

from __future__ import annotations

import argparse

from flarecast.cli.main import _print_demo, run_offline_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0, help="synthetic duration (h)")
    ap.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
    ap.add_argument("--flares", type=int, default=8, help="flare count")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (determinism)")
    ap.add_argument("--horizon", type=float, default=30.0, help="forecast horizon (min)")
    ap.add_argument("--no-forecast", action="store_true", help="skip the forecast block")
    args = ap.parse_args()

    res = run_offline_pipeline(
        duration_s=args.hours * 3600.0,
        cadence_s=args.cadence,
        n_flares=args.flares,
        seed=args.seed,
        horizon_min=args.horizon,
        do_forecast=not args.no_forecast,
    )
    _print_demo(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
