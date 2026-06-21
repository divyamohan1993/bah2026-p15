#!/usr/bin/env python3
"""Example 04 -- train + evaluate the flare forecaster (offline).

The full forecasting deliverable (ARCHITECTURE.md Section 5 / 10): from a
synthetic master catalogue + light curves, build **strict pre-peak, in-event
masked** labels for "flare >= class C within the next N minutes", run
**leakage-free blocked rolling-origin CV** (with an embargo), train a
**gradient-boosted-tree** forecaster (LightGBM, sklearn fallback) and the
**mandatory baselines** (climatology + persistence), and print the rare-event
metric battery (TSS / HSS / BSS / POD / FAR / ROC-AUC) plus the **lead-time
distribution and the LT-vs-FAR operating-point sweep**.

Pure offline: numpy is a core dep; LightGBM is used opportunistically and
falls back to scikit-learn, then to baselines-only -- the example never errors
for a missing optional model.

Run::

    python examples/04_train_forecast.py
    python examples/04_train_forecast.py --hours 24 --horizon 30 --threshold C
"""

from __future__ import annotations

import argparse

from flarecast.cli.main import (
    OfflinePipelineResult,
    _build_catalogue,
    _offline_forecast,
    _pivot_soft_hard,
    _print_forecast_only,
    _run_detection,
)
from flarecast.synth.generator import generate_flare_lightcurves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0, help="synthetic duration (h)")
    ap.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
    ap.add_argument("--flares", type=int, default=10, help="flare count")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--horizon", type=float, default=30.0, help="forecast horizon (min)")
    ap.add_argument("--threshold", default="C", help="minimum positive class")
    args = ap.parse_args()

    # synth -> dual-band detection -> master catalogue (the label source).
    samples, truth = generate_flare_lightcurves(
        duration_s=args.hours * 3600.0,
        cadence_s=args.cadence,
        n_flares=args.flares,
        seed=args.seed,
    )
    times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, args.cadence)
    events = _build_catalogue(soft_flares, hard_flares)

    # labels -> CV -> GBT + baselines -> metrics + lead time + LT-vs-FAR.
    fs = _offline_forecast(
        samples, events, truth, times, soft, hard,
        horizon_min=args.horizon, class_threshold=args.threshold, seed=args.seed,
    )

    print("=" * 70)
    print("Aditya FlareCast - forecast training + evaluation (offline)")
    print("=" * 70)
    print(
        f"synthetic flares: {len(truth)}   master catalogue events: {len(events)}"
    )
    print("-" * 70)
    _print_forecast_only(
        OfflinePipelineResult(
            samples=samples, truth=truth, soft_flares=soft_flares,
            hard_flares=hard_flares, events=events, match_rows=[],
            forecast_summary=fs, out_dir="", sqlite_path="", json_path="",
        )
    )
    print()
    print("OK - forecaster trained + evaluated OFFLINE (GBT + baselines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
