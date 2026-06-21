#!/usr/bin/env python3
"""Example 02 -- live GOES XRS nowcast (live -> cache -> synth fallback).

GOES XRS is the **live operational anchor** (ARCHITECTURE.md Section 1.3): a
sub-minute, public, no-auth JSON feed that defines the A-X scale. This example
pulls it through :class:`flarecast.ingest.goes.GOESFetcher`, whose resolution
order is **live SWPC JSON -> bundled cached sample -> physics-based synthetic
generator**, so it runs identically online or fully offline. The fetched soft
flux is then streamed through the **O(1)** soft-band detector (CUSUM + GOES
FSM) to nowcast onsets/peaks, and the latest GOES class is reported.

Run::

    python examples/02_goes_live_nowcast.py            # try live, fall back offline
    python examples/02_goes_live_nowcast.py --offline  # force cache -> synth

The printed ``tier`` line shows which source actually served the data
(``live`` / ``cache`` / ``synth``) -- in a no-network sandbox it will be
``cache`` (the bundled ``examples/data/xrays-1-day.sample.json``) or ``synth``.
"""

from __future__ import annotations

import argparse

from flarecast.cli.main import (
    _build_catalogue,
    _hms,
    _pivot_soft_hard,
    _relabel_goes_as_synth,
    _run_detection,
)
from flarecast.ingest.goes import GOESFetcher


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="skip the live tier")
    ap.add_argument("--cadence", type=float, default=60.0, help="assumed cadence (s)")
    args = ap.parse_args()

    # channel="both" so we get the long (class) + short channels.
    fetcher = GOESFetcher(channel="both", allow_network=not args.offline)
    samples = list(fetcher.fetch(0.0, float("inf")))

    print("=" * 70)
    print("Aditya FlareCast - GOES XRS live nowcast (live -> cache -> synth)")
    print("=" * 70)
    print(f"data tier : {fetcher.last_source}")
    print(f"samples   : {len(samples)}")

    if not samples:
        print("no GOES samples available (no live feed, no cache, no synth).")
        return 0

    longs = [s for s in samples if s.stream.endswith("-long")]
    longs.sort(key=lambda s: s.t)
    if longs:
        latest = longs[-1]
        print(
            f"latest    : t={latest.t:g}s  flux={latest.value:.3e} {latest.unit}  "
            f"class={latest.cls or '-'}"
        )

    # Relabel GOES long/short -> synth stream ids so the shared pivot + detectors
    # accept them (GOES has no hard band, so only the soft path runs).
    times, soft, hard = _pivot_soft_hard(_relabel_goes_as_synth(samples))
    if not times:
        # defensive: build a soft-only stream straight from the long channel.
        soft = {s.t: s.value for s in longs}
        hard = {}
        times = sorted(soft)

    soft_flares, hard_flares = _run_detection(times, soft, hard, args.cadence)
    events = _build_catalogue(soft_flares, hard_flares)

    print()
    print(f"soft-band nowcast detections: {len(soft_flares)}")
    print("-" * 70)
    print(f"{'#':>2}  {'onset':>10}{'peak':>10}{'end':>10}{'class':>8}")
    for i, bf in enumerate(soft_flares, 1):
        print(
            f"{i:>2}  {_hms(bf.onset_time):>10}{_hms(bf.peak_time):>10}"
            f"{_hms(bf.end_time):>10}{(bf.goes_class or '-'):>8}"
        )
    print()
    print(f"master catalogue events: {len(events)}")
    print()
    print("OK - GOES nowcast ran (offline-capable: live -> cache -> synth).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
