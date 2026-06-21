# Aditya FlareCast

**Automated soft + hard X-ray solar-flare nowcasting & forecasting.**
ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 15.

Solar flares are sudden bursts of X-ray/EUV radiation released by magnetic
reconnection in the solar corona; the strongest drive space-weather storms
that disrupt satellites, GPS, HF communications, and power grids. **Aditya
FlareCast** is an automated pipeline that watches the Sun in two X-ray bands
from ISRO's Aditya-L1 mission — **SoLEXS** (soft X-rays, the thermal/gradual
phase) and **HEL1OS** (hard X-rays, the impulsive/non-thermal phase) — to
**(a) nowcast** flares (real-time detection + A–X classification), **(b)
forecast** them minutes ahead with a calibrated probability and a quantifiable
lead time, and **(c) visualize** the light curves with live alerts.

## The pitch

- **Multi-satellite by design.** Aditya-L1 cannot image, see the far side, or
  supply pre-2024 history, so SoLEXS/HEL1OS are anchored on the live **GOES
  XRS** feed and fused with a 30+ satellite constellation (STIX, Fermi GBM,
  SDO, STEREO, ACE, GONG, radio, …) for cross-calibration, gap-fill, far-side
  coverage, and consensus labels.
- **Nowcast → forecast bridge via the Neupert effect.** The hard X-ray light
  curve tracks the *time-derivative* of the soft X-ray curve and **leads** the
  soft-band peak by ~1–3 minutes, so HEL1OS gives an early-warning signal —
  the physical engine of the short-horizon forecast.
- **Fast / O(1) hot path.** Every detector primitive is constant-time and
  constant-state per sample (EMA, CUSUM, Poisson-FOCuS, hash-bucket catalogue
  index), so the same detector math runs identically offline (this Python
  reference) and on the Cloudflare edge production target.
- **Runs fully offline.** A physics-based synthetic generator produces
  SoLEXS-like and HEL1OS-like light curves plus a ground-truth event list, so
  the entire pipeline is demoable and testable with **zero network and zero
  credentials**.

## Architecture at a glance

```
ingest → fusion → nowcast detection → master catalogue → forecast → API → UI/alerts
```

Two physical substrates run the *same* detector logic: the offline-capable
Python reference package (`flarecast`) and the Cloudflare edge deployment
(`edge/`). For the full design — data model, schemas, module contracts, and
the O(1) complexity justification — see **[ARCHITECTURE.md](ARCHITECTURE.md)**
(the single source of truth; its appendices are the normative build contract).

## Quickstart (offline demo)

```bash
# 1. Install with the forecaster + serving extras (recommended):
pip install -e ".[ml,api]"

# 2. Run the end-to-end offline demo (synthetic data, no network):
python -m flarecast.cli.main demo

# 3. Launch the API + live dashboard (offline synthetic/cached data):
flarecast serve            # then open http://127.0.0.1:8000/  (API docs at /docs)
```

`flarecast serve` mounts the static dashboard (`dashboard/`) at `/` and exposes
the O(1) read API under `/api/*` plus an SSE live feed at `/api/stream`. The
dashboard is fully **offline and dependency-free** (vanilla JS on `<canvas>`,
no CDN): dual soft+hard light curves with GOES A–X colour bands, a forecast
probability gauge, a red alert banner on a nowcast/forecast trigger, and a
recent-flares catalogue table.

Optional capability bundles (see `pyproject.toml`):

```bash
pip install -e .           # minimal offline core (numpy, pandas) — demo + core tests
pip install -e ".[real]"   # live + archive data sources (astropy, sunpy, …)
pip install -e ".[ml]"     # forecasters (lightgbm, scikit-learn, torch, onnx)
pip install -e ".[api]"    # serving layer (fastapi, uvicorn)
pip install -e ".[dev]"    # tooling (pytest, ruff)
```

> The offline core (`requirements.txt`) is intentionally tiny so the demo and
> core tests run anywhere. Everything else is opt-in. The `demo` and
> `examples/01`–`04` run on the core (numpy); `serve` and `examples/05` need
> the `api` extra; the GBT forecaster uses the `ml` extra (it falls back to
> baselines-only if absent).

### Runnable examples

Each script under `examples/` runs offline (no network, no credentials):

| Example | What it shows |
|---|---|
| `examples/00_synth_demo.py` | Generate synthetic soft+hard light curves + a truth event list (pure stdlib). |
| `examples/01_offline_demo.py` | Full pipeline: synth → dual-band detection → master catalogue → detected-vs-truth table. |
| `examples/02_goes_live_nowcast.py` | GOES XRS nowcast via `GOESFetcher` (live → cached sample → synth fallback) → soft-band detection. |
| `examples/03_fusion_demo.py` | Two synthetic SXR sources → LTT → inverse-variance + Kalman fusion → fused σ < single-source σ. |
| `examples/04_train_forecast.py` | Labels → leakage-free CV → GBT + baselines → TSS/HSS/BSS/POD/FAR + lead-time & LT-vs-FAR sweep. |
| `examples/05_serve_dashboard.py` | Seed the store + serve the API/dashboard; non-blocking self-check by default, `--serve` to run uvicorn. |

## Repository layout

```
flarecast/            the Python package
├── __init__.py       version + minimal, import-safe top level (lazy run_pipeline)
├── types.py          shared dataclasses/enums (FluxSample, Quantity, QCFlag, …)
├── constants.py      physical + config constants (LTT, GOES bands, QC bits, …)
├── ingest/           data access: GOES live + Fido archive + PRADAN + fallback
├── synth/            physics-based synthetic light-curve generator (offline keystone)
├── fusion/           multi-satellite fusion: LTT, cross-cal, gap-fill, Kalman
├── detect/           O(1) streaming detection (CUSUM, Poisson-FOCuS, FSM, classify)
├── catalog/          master flare catalogue + O(1) hash-bucket index
├── forecast/         30-dim features, LightGBM/TCN, lead time, evaluation
├── api/              FastAPI O(1) reads + SSE live push
└── cli/              `flarecast` command-line entry point
dashboard/            static UI: light curves, probability gauge, alert banner
edge/                 Cloudflare Workers edge deployment (DO + KV + D1 + R2)
examples/             runnable examples + small bundled sample data
tests/                offline test suite (synthetic data only)
docs/research/        the six research deliverables (design inputs)
ARCHITECTURE.md       authoritative architecture & build contract
```

See [ARCHITECTURE.md, Appendix A](ARCHITECTURE.md) for the exact, normative
tree with a one-line purpose per file.

## Status & roadmap

Early development. The project is organized into independent, parallelizable
workstreams (see [ARCHITECTURE.md, Appendix C](ARCHITECTURE.md)):

| Workstream | Scope | Status |
|---|---|---|
| **WS0 — Foundation** | `types.py`, `constants.py`, `__init__.py`, packaging | **Done** (this skeleton) |
| WS1 — Synthetic data + ingest | offline generator, GOES/Fido/PRADAN fetchers | In progress |
| WS2 — Fusion | LTT, cross-cal, gap-fill, Kalman, consensus | In progress |
| WS3 — Detection + catalogue | O(1) detectors, classification, master catalogue | In progress |
| WS4 — Forecasting | 30-dim features, LightGBM/TCN, lead time, metrics | In progress |
| WS5 — API + dashboard + CLI | FastAPI/SSE, visual interface, `flarecast` CLI | In progress |
| WS6 — Edge + packaging + tests | Cloudflare Workers, cross-substrate invariant, CI | In progress |

Anything marked *in progress* may not yet exist or may be a stub; the package
top level is deliberately import-safe so each workstream can build and import
in parallel against the frozen `flarecast.types` / `flarecast.constants`
contract.

## License

MIT (see `pyproject.toml`).
