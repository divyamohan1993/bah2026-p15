#!/usr/bin/env python3
"""Offline synthetic-data demo -- the keystone proof (ARCHITECTURE.md Section 6).

Generates one day of physics-based SoLEXS-like *soft* and HEL1OS-like *hard*
X-ray light curves plus a ground-truth flare event list, then prints summary
statistics and the truth table -- using **pure Python standard library only**
(no numpy, no pandas, no network, no credentials). This is Workstream 1's proof
that the whole pipeline can run offline.

Run::

    python examples/00_synth_demo.py
    python examples/00_synth_demo.py --seed 7 --hours 12

The hard channel is built to satisfy the **Neupert effect** (HXR ~ d/dt SXR),
so the impulsive hard bursts lead each soft peak by a minute or two -- the
forecasting lever (research 01 Section 6). Large flares additionally trip the
SoLEXS SDD1 paralyzable saturation flag (research 01 Section 1.4).
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

# Import only the pure-python synth core; nothing here needs numpy/pandas.
from flarecast.synth.generator import (
    STREAM_HXR_HIGH,
    STREAM_HXR_LOW,
    STREAM_SXR_LONG,
    STREAM_SXR_SHORT,
    generate_flare_lightcurves,
)


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (min, mean, max, stdev) of a value list (pure python)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    vmin = min(values)
    vmax = max(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return (vmin, mean, vmax, math.sqrt(var))


def _fmt_hms(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (determinism).")
    parser.add_argument("--hours", type=float, default=24.0, help="Duration in hours.")
    parser.add_argument("--cadence", type=float, default=60.0, help="Cadence in seconds.")
    parser.add_argument("--flares", type=int, default=None, help="Flare count (default auto).")
    args = parser.parse_args()

    duration_s = args.hours * 3600.0

    samples, truth = generate_flare_lightcurves(
        duration_s=duration_s,
        cadence_s=args.cadence,
        n_flares=args.flares,
        seed=args.seed,
    )

    # bucket samples by stream for per-channel stats.
    by_stream: dict[str, list[float]] = {}
    for s in samples:
        by_stream.setdefault(s.stream, []).append(s.value)

    n_timestamps = len(by_stream.get(STREAM_SXR_LONG, []))

    print("=" * 70)
    print("Aditya FlareCast - offline synthetic light-curve demo (pure Python)")
    print("=" * 70)
    print(
        f"duration={args.hours:g} h  cadence={args.cadence:g} s  seed={args.seed}  "
        f"timestamps={n_timestamps}  total_samples={len(samples)}"
    )
    print()

    # ---- per-channel summary stats ----
    print("Per-channel summary")
    print("-" * 70)
    print(f"{'stream':<24}{'unit':<10}{'min':>11}{'mean':>11}{'max':>11}")
    channel_units = {
        STREAM_SXR_LONG: "W m^-2",
        STREAM_SXR_SHORT: "W m^-2",
        STREAM_HXR_LOW: "counts/s",
        STREAM_HXR_HIGH: "counts/s",
    }
    for stream in (STREAM_SXR_LONG, STREAM_SXR_SHORT, STREAM_HXR_LOW, STREAM_HXR_HIGH):
        vals = by_stream.get(stream, [])
        vmin, mean, vmax, _ = _stats(vals)
        unit = channel_units[stream]
        if unit == "W m^-2":
            print(f"{stream:<24}{unit:<10}{vmin:>11.2e}{mean:>11.2e}{vmax:>11.2e}")
        else:
            print(f"{stream:<24}{unit:<10}{vmin:>11.0f}{mean:>11.0f}{vmax:>11.0f}")
    print()

    # ---- truth event table ----
    print(f"Ground-truth flare events: {len(truth)}")
    print("-" * 70)
    if truth:
        print(
            f"{'#':>2}  {'class':<6}{'start':>10}{'peak':>10}{'end':>10}"
            f"{'peak flux':>12}{'HXR lead':>10}{'spikes':>8}{'sat':>5}"
        )
        for i, e in enumerate(truth, 1):
            print(
                f"{i:>2}  {e.goes_class:<6}{_fmt_hms(e.t_start):>10}"
                f"{_fmt_hms(e.t_peak):>10}{_fmt_hms(e.t_end):>10}"
                f"{e.peak_flux_wm2:>12.2e}{_fmt_hms(e.hxr_lead_s):>10}"
                f"{e.n_hxr_spikes:>8}{('Y' if e.saturated else '-'):>5}"
            )
    print()

    # ---- class-mix breakdown ----
    counts: dict[str, int] = {}
    for e in truth:
        counts[e.goes_class[0].upper()] = counts.get(e.goes_class[0].upper(), 0) + 1
    if counts:
        mix = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"class mix (by letter): {mix}")

    # ---- a peek at the canonical FluxSample record ----
    if samples:
        s0 = next((s for s in samples if s.stream == STREAM_SXR_LONG), samples[0])
        utc = datetime.fromtimestamp(s0.t, tz=timezone.utc).isoformat() if s0.t else "n/a"
        print()
        print("example FluxSample (soft long channel, first timestamp):")
        print(
            f"  stream={s0.stream} t={s0.t:g}s ({utc}) value={s0.value:.3e} "
            f"{s0.unit} quantity={s0.quantity} cls={s0.cls} qc={s0.qc}"
        )

    print()
    print("OK - synthetic generator ran offline with zero network / zero deps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
