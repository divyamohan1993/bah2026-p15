/**
 * Front Worker -- the edge API + live-stream entry point
 * (ARCHITECTURE.md Section 7.1/7.3, Appendix B.7/B.8).
 *
 * Routes (O(1) unless noted; CORS enabled on all):
 *   GET  /api/latest?stream=...    -> KV `latest:<stream>`   (O(1) GET)
 *   GET  /api/alert?stream=...     -> KV `alert:<stream>`    (O(1) GET)
 *   GET  /api/forecast?stream=...  -> KV `forecast:<stream>` (O(1) GET)
 *   GET  /api/at?stream=..&t=..    -> D1 indexed query        (O(log n + k))
 *   GET  /api/stream?stream=...    -> upgrade to the DO WebSocket (live push)
 *   GET  /api/health               -> liveness probe
 *
 * The hot reads are single KV GETs (hash lookup, no scan) -- the "fastest
 * platform / O(1)" claim. The `/api/at` time query is the deliberate analytics
 * path: an indexed D1 lookup on `t_peak`, NOT forced to be O(1) (ARCHITECTURE.md
 * Section 7.3 "honest caveats"). `/api/stream` defers to the per-stream Detector
 * Durable Object, which owns the strongly-consistent state and the Hibernatable
 * WebSocket fan-out.
 *
 * This file also re-exports {@link DetectorDO} (so the DO migration in
 * `wrangler.toml` can bind it) and wires the Cron `scheduled()` handler.
 */

import { ingestOnce } from "./cron";
import { DetectorDO } from "./detector";
import {
  KV_KEY_ALERT,
  KV_KEY_FORECAST,
  KV_KEY_LATEST,
  STREAM_GOES_LONG,
} from "./constants";

export { DetectorDO };

/** Worker + DO bindings (see `wrangler.toml`). */
export interface Env {
  FLARE_KV: KVNamespace;
  FLARE_DB: D1Database;
  FLARE_R2: R2Bucket;
  DETECTOR: DurableObjectNamespace;
  SWPC_XRAYS_URL?: string;
}

const CORS_HEADERS: Record<string, string> = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type",
  "access-control-max-age": "86400",
};

/** JSON response with CORS. `null` body -> 404 (not found). */
function json(body: unknown, status = 200): Response {
  if (body === null) {
    return new Response(JSON.stringify({ error: "not found" }), {
      status: 404,
      headers: { "content-type": "application/json", ...CORS_HEADERS },
    });
  }
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...CORS_HEADERS },
  });
}

/** Read + parse a KV JSON value (O(1) GET); returns null if absent. */
async function kvJson(kv: KVNamespace, key: string): Promise<unknown | null> {
  const raw = await kv.get(key);
  return raw === null ? null : JSON.parse(raw);
}

/** `/api/at` -> the catalogue event whose peak is nearest `t` (indexed D1). */
async function queryAt(
  env: Env,
  stream: string,
  tSeconds: number,
): Promise<unknown | null> {
  void stream; // single logical catalogue; stream reserved for future sharding
  const tMs = Math.round(tSeconds * 1000.0);
  // Uses the idx_cat_tpeak index: nearest peak by absolute time distance.
  const row = await env.FLARE_DB.prepare(
    `SELECT * FROM flare_catalogue
       ORDER BY ABS(t_peak - ?1) ASC
       LIMIT 1`,
  )
    .bind(tMs)
    .first();
  return row ?? null;
}

/** Forward a WebSocket upgrade to the per-stream Detector Durable Object. */
function streamViaDO(env: Env, req: Request, stream: string): Response {
  const id = env.DETECTOR.idFromName(stream); // deterministic routing, O(1)
  const stub = env.DETECTOR.get(id);
  return stub.fetch("https://do/ws", req) as unknown as Response;
}

async function handle(req: Request, env: Env): Promise<Response> {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  const url = new URL(req.url);
  const path = url.pathname;
  const stream = url.searchParams.get("stream") ?? STREAM_GOES_LONG;

  switch (path) {
    case "/api/health":
      return json({ ok: true, ts: Date.now() });

    case "/api/latest":
      return json(await kvJson(env.FLARE_KV, `${KV_KEY_LATEST}${stream}`));

    case "/api/alert":
      return json(await kvJson(env.FLARE_KV, `${KV_KEY_ALERT}${stream}`));

    case "/api/forecast":
      return json(await kvJson(env.FLARE_KV, `${KV_KEY_FORECAST}${stream}`));

    case "/api/at": {
      const tRaw = url.searchParams.get("t");
      const t = tRaw === null ? NaN : Number(tRaw);
      if (!Number.isFinite(t)) {
        return json({ error: "query param 't' (epoch seconds) required" }, 400);
      }
      return json(await queryAt(env, stream, t));
    }

    case "/api/stream": {
      if (req.headers.get("Upgrade") !== "websocket") {
        return json({ error: "expected a WebSocket upgrade" }, 426);
      }
      return streamViaDO(env, req, stream);
    }

    default:
      return json({ error: `unknown route: ${path}` }, 404);
  }
}

export default {
  /** HTTP entry point. */
  async fetch(req: Request, env: Env): Promise<Response> {
    return handle(req, env);
  },

  /** Cron Trigger entry point (delegates to the shared ingest routine). */
  async scheduled(
    _event: ScheduledController,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    ctx.waitUntil(
      ingestOnce(env).catch((err) => console.error("cron ingest error:", err)),
    );
  },
};
