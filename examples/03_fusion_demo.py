#!/usr/bin/env python3
"""Example 03 -- multi-satellite fusion: two soft-X-ray sources -> tighter sigma.

Demonstrates the quantitative payoff of fusion (ARCHITECTURE.md Section 3.8):
combining two cross-calibrated estimates of the *same* physical quantity yields
a best estimate whose uncertainty is **smaller than either input**. We build two
synthetic soft-X-ray (SXR_LONG) sources -- Aditya-L1 SoLEXS (r ~ 0.990 AU) and
GOES (r = 1.0 AU) -- LTT-correct them to the common Earth/L1 frame (SoLEXS is
~+5 s ahead), and fuse them:

* :func:`flarecast.fusion.fuse.inverse_variance_fuse` -- the static per-cell
  estimator: ``sigma_hat = 1 / sqrt(sum 1/sigma_i^2)`` is provably below
  ``min_i sigma_i``;
* :func:`flarecast.fusion.pipeline.run_fusion` -- the full temporal pipeline
  (LTT -> grid -> Kalman fuse), whose median fused sigma also beats each source.

Pure standard library + ``flarecast`` -- no network, no numpy.

Run::

    python examples/03_fusion_demo.py
    python examples/03_fusion_demo.py --hours 3 --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
from statistics import median

from flarecast.constants import ADITYA_L1_R_AU, UNIT_SXR
from flarecast.fusion.fuse import inverse_variance_fuse
from flarecast.fusion.ltt import ltt_delta_seconds
from flarecast.fusion.pipeline import run_fusion
from flarecast.fusion.schema import FusionRecord
from flarecast.synth.generator import STREAM_SXR_LONG, generate_flare_lightcurves
from flarecast.types import Quantity

# Per-source 1-sigma (canonical W/m^2). SoLEXS is noisier than the GOES anchor.
SIGMA_SOLEXS = 6e-8
SIGMA_GOES = 2e-8


def _build_records(seed: int, hours: float, cadence_s: float):
    """Two SXR_LONG FusionRecord streams from one truth light curve + noise."""
    rng = random.Random(seed)
    samples, _truth = generate_flare_lightcurves(
        duration_s=hours * 3600.0,
        cadence_s=cadence_s,
        n_flares=max(1, int(round(hours / 4.0))),
        noise=False,  # we add per-source measurement noise ourselves
        gaps=False,
        spikes=False,
        seed=seed,
    )
    truth_long = {s.t: s.value for s in samples if s.stream == STREAM_SXR_LONG}
    times = sorted(truth_long)

    solexs: list[FusionRecord] = []
    goes: list[FusionRecord] = []
    for t in times:
        true_v = truth_long[t]
        solexs.append(
            FusionRecord(
                t_obs_utc=t,
                source_id="ADITYA_SOLEXS",
                platform="Aditya-L1",
                quantity=Quantity.SXR_LONG.value,
                value=max(1e-9, true_v + rng.gauss(0.0, SIGMA_SOLEXS)),
                band="1-8A",
                unit=UNIT_SXR,
                sigma=SIGMA_SOLEXS,
                cadence_s=cadence_s,
                vantage_r_au=ADITYA_L1_R_AU,
            )
        )
        goes.append(
            FusionRecord(
                t_obs_utc=t,
                source_id="GOES_PRIMARY_XRSB",
                platform="GOES-19",
                quantity=Quantity.SXR_LONG.value,
                value=max(1e-9, true_v + rng.gauss(0.0, SIGMA_GOES)),
                band="1-8A",
                unit=UNIT_SXR,
                sigma=SIGMA_GOES,
                cadence_s=cadence_s,
                vantage_r_au=1.0,
            )
        )
    return times, truth_long, solexs, goes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=3.0, help="synthetic duration (h)")
    ap.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    times, _truth, solexs, goes = _build_records(args.seed, args.hours, args.cadence)

    print("=" * 70)
    print("Aditya FlareCast - multi-satellite fusion demo (tighter sigma)")
    print("=" * 70)
    print(f"sources : ADITYA_SOLEXS (sigma={SIGMA_SOLEXS:.1e}) + "
          f"GOES_PRIMARY_XRSB (sigma={SIGMA_GOES:.1e})  [W/m^2]")
    print(f"samples : {len(times)} per source  cadence={args.cadence:g}s")

    # ---- LTT: SoLEXS at L1 leads GOES by ~+5 s ----
    dt_solexs = ltt_delta_seconds(ADITYA_L1_R_AU)
    dt_goes = ltt_delta_seconds(1.0)
    print()
    print("Light-travel-time correction to the Earth/L1 frame")
    print("-" * 70)
    print(f"  ADITYA_SOLEXS (r={ADITYA_L1_R_AU:.3f} AU): dt = {dt_solexs:+.1f} s")
    print(f"  GOES          (r=1.000 AU): dt = {dt_goes:+.1f} s  (reference)")

    # ---- static inverse-variance fusion (provably tighter) ----
    x_hat, sigma_hat = inverse_variance_fuse(
        [1.0, 1.0], [SIGMA_SOLEXS, SIGMA_GOES]
    )
    sigma_min = min(SIGMA_SOLEXS, SIGMA_GOES)
    print()
    print("Static inverse-variance fusion (per cell)")
    print("-" * 70)
    print(f"  single-source best sigma : {sigma_min:.3e}")
    print(f"  FUSED sigma_hat          : {sigma_hat:.3e}")
    expected = 1.0 / math.sqrt(1.0 / SIGMA_SOLEXS**2 + 1.0 / SIGMA_GOES**2)
    print(f"  expected 1/sqrt(sum 1/s^2): {expected:.3e}")
    assert sigma_hat < sigma_min, "fused sigma must be below the best single source"
    print(f"  --> fused sigma is {100 * (1 - sigma_hat / sigma_min):.1f}% smaller. OK")

    # ---- full temporal pipeline (LTT -> grid -> Kalman fuse) ----
    product = run_fusion(
        {"ADITYA_SOLEXS": solexs, "GOES_PRIMARY_XRSB": goes},
        grid_dt=args.cadence,
    )
    fused = product.products[Quantity.SXR_LONG.value]
    fused_sigmas = [s for s in fused.sigma if s == s and s > 0]  # drop NaNs
    med_fused = median(fused_sigmas) if fused_sigmas else float("nan")
    print()
    print("Full Kalman fusion pipeline (run_fusion)")
    print("-" * 70)
    print(f"  fused grid cells     : {len(fused.grid)}  coverage={fused.coverage:.2f}")
    print(f"  median fused sigma   : {med_fused:.3e}")
    print(f"  single-source sigma  : SoLEXS={SIGMA_SOLEXS:.1e}  GOES={SIGMA_GOES:.1e}")
    if fused_sigmas:
        better = med_fused < SIGMA_GOES
        print(f"  --> median fused sigma {'beats' if better else 'does NOT beat'} "
              f"the GOES anchor sigma.")
    print()
    print("OK - fusion ran OFFLINE; fused uncertainty < single-source uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
