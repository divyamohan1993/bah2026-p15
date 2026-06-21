# Aditya FlareCast — Cloudflare edge

The **O(1) production target** (ARCHITECTURE.md Section 7). A Workers **Cron
Trigger** pulls the NOAA SWPC GOES XRS JSON each minute, routes the newest
soft-band sample to a per-stream **Durable Object** running the *same*
EMA + EWMV + CUSUM detector math as the Python reference, and serves O(1) reads
from **KV** plus a Hibernatable **WebSocket** live stream. The queryable flare
catalogue lives in **D1**; bulk/raw artifacts (FITS, Parquet, ONNX) live in
**R2**.

```
src/index.ts     front Worker: O(1) KV reads (/api/latest|alert|forecast),
                 indexed D1 /api/at, /api/stream WebSocket upgrade, scheduled()
src/cron.ts      ingestOnce(): fetch SWPC JSON -> normalize -> route to the DO
src/detector.ts  DetectorDO: O(1) streaming detector + Hibernatable WS fan-out
src/types.ts     FluxSample / DetectionState TS mirrors of flarecast.types
src/constants.ts edge subset of flarecast.constants (parity-locked values)
schema.sql       D1 DDL — mirrors flare_catalogue (§9.4) + a samples rollup
wrangler.toml    Workers config: cron, KV/D1/R2 + DetectorDO bindings, migration
```

## Bindings (declared in `wrangler.toml`)

| Binding    | Type                   | Purpose                                            |
|------------|------------------------|----------------------------------------------------|
| `FLARE_KV` | KV namespace           | `latest:` / `alert:` / `forecast:` hot cache (O(1))|
| `FLARE_DB` | D1 (SQLite)            | `flare_catalogue` — queryable analytics path       |
| `FLARE_R2` | R2 bucket              | raw FITS / Parquet light curves / ONNX models      |
| `DETECTOR` | Durable Object         | per-stream online detector (`DetectorDO`) + WS      |

## Prerequisites

- Node 20+ and npm.
- A Cloudflare account; `wrangler` is installed as a dev dependency.

```bash
cd edge
npm install
```

## Deploy

`wrangler` reads `wrangler.toml`. Create the backing resources first, paste the
ids it prints into `wrangler.toml` (replacing the `REPLACE_WITH_*` placeholders),
then deploy.

```bash
# 1. Authenticate.
npx wrangler login

# 2. Create the KV namespace (paste `id` into [[kv_namespaces]] in wrangler.toml;
#    add a preview id for `wrangler dev`).
npx wrangler kv namespace create FLARE_KV
npx wrangler kv namespace create FLARE_KV --preview

# 3. Create the D1 database (paste `database_id` into [[d1_databases]]), then
#    load the catalogue schema.
npx wrangler d1 create aditya-flarecast-db
npx wrangler d1 execute FLARE_DB --remote --file schema.sql
#    (local dev copy: npx wrangler d1 execute FLARE_DB --local --file schema.sql)

# 4. Create the R2 bucket (name must match [[r2_buckets]].bucket_name).
npx wrangler r2 bucket create aditya-flarecast-raw

# 5. Deploy (Cron + DO + bindings). The DetectorDO migration (tag v1,
#    new_sqlite_classes) applies automatically on first deploy.
npx wrangler deploy
```

> The Durable Object binding `DETECTOR -> DetectorDO` and its
> `[[migrations]]` (`new_sqlite_classes = ["DetectorDO"]`) are required because
> `DetectorDO` uses the Hibernatable WebSocket API and SQLite-backed DO storage.

### Local development

```bash
npm run dev        # wrangler dev — local Worker + Miniflare KV/D1/R2/DO
```

The SWPC endpoint can be overridden for staging/tests via the `SWPC_XRAYS_URL`
var in `wrangler.toml` (defaults in code to the public 1-day XRS feed).

## Cross-substrate parity (the hard invariant)

The edge detector (`src/detector.ts`) and the Python detector
(`flarecast/detect/{primitives,cusum}.py`) must produce **identical** onset
decisions on the same input with the same `alpha`/`k`/`h`
(ARCHITECTURE.md Appendix B.8). That invariant is anchored by one committed
golden vector, `tests/golden/cusum_golden.json`, generated once from the Python
reference pipeline.

```bash
npm test           # runs node test/parity.mjs
```

`test/parity.mjs` re-runs the JS mirror of the gated EMA + EWMV + CUSUM hot path
over the stored golden input and asserts it reproduces every stored array
exactly (exit 0 on success, 1 on any mismatch). The Python half of the same
contract is `tests/test_parity.py`. **This check is mandatory in CI**
(`.github/workflows/ci.yml`, job `edge`).

## Type checking

```bash
npm run typecheck  # tsc --noEmit against src/ with @cloudflare/workers-types
```
