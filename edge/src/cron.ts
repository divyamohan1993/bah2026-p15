/**
 * Cron ingestion handler (ARCHITECTURE.md Section 7.1, Appendix B.8).
 *
 * Workers Cron Triggers fire `scheduled()` each cadence (see `wrangler.toml`
 * `[triggers] crons`). Each tick: fetch the NOAA SWPC GOES XRS JSON
 * (`SWPC_XRAYS_URL`), normalise every record to a {@link FluxSample}, and route
 * each soft-band sample to the per-stream Detector Durable Object (which runs
 * the O(1) EMA+CUSUM math and writes `alert:*`). The most recent sample per
 * stream is also written to `latest:*` in KV for the O(1) front-Worker reads.
 *
 * This mirrors the Python live hot path (`flarecast.ingest.goes.GOESFetcher`
 * + the SoLEXS/GOES soft-band detector), so the edge nowcast is a faithful
 * production copy of the offline reference.
 */

import {
  KV_KEY_LATEST,
  STREAM_GOES_LONG,
  STREAM_GOES_SHORT,
  SWPC_ENERGY_LONG,
  SWPC_XRAYS_URL,
} from "./constants";
import { classifyFlux } from "./detector";
import type { FluxSample } from "./types";

/** The Worker bindings available to the cron handler (see `wrangler.toml`). */
export interface Env {
  FLARE_KV: KVNamespace;
  FLARE_DB: D1Database;
  FLARE_R2: R2Bucket;
  DETECTOR: DurableObjectNamespace;
  /** Optional override (tests / staging) for the SWPC endpoint. */
  SWPC_XRAYS_URL?: string;
}

/** One raw SWPC GOES XRS JSON record (the public feed's shape). */
interface SwpcXrayRecord {
  time_tag: string; // ISO-8601 UTC
  satellite?: number;
  flux: number; // W m^-2
  observed_flux?: number;
  electron_correction?: number;
  electron_contaminaton?: boolean;
  energy: string; // "0.1-0.8nm" (long) | "0.05-0.4nm" (short)
}

/**
 * Normalise one SWPC record to a {@link FluxSample}.
 * Mirrors `flarecast.ingest.normalize.normalize_swpc`: the long channel
 * (0.1-0.8 nm = 1-8 A) is `SXR_LONG`, the short channel is `SXR_SHORT`; flux is
 * already on the GOES scale (W m^-2).
 */
export function normalizeSwpc(rec: SwpcXrayRecord): FluxSample {
  const isLong = rec.energy === SWPC_ENERGY_LONG;
  const stream = isLong ? STREAM_GOES_LONG : STREAM_GOES_SHORT;
  const quantity = isLong ? "SXR_LONG" : "SXR_SHORT";
  const t = Date.parse(rec.time_tag) / 1000.0; // epoch seconds UTC
  const value = Number(rec.flux);
  return {
    stream,
    t,
    value,
    unit: "W m^-2",
    source: "SWPC",
    quantity,
    cls: isLong ? classifyFlux(value) : undefined,
    qc: 0,
  };
}

/** Fetch + parse the SWPC XRS JSON, defensively (bad feed -> empty list). */
export async function fetchSwpcXrays(env: Env): Promise<SwpcXrayRecord[]> {
  const url = env.SWPC_XRAYS_URL ?? SWPC_XRAYS_URL;
  const resp = await fetch(url, {
    headers: { accept: "application/json" },
    cf: { cacheTtl: 30 },
  });
  if (!resp.ok) {
    console.warn(`SWPC fetch failed: ${resp.status} ${resp.statusText}`);
    return [];
  }
  const data = (await resp.json()) as SwpcXrayRecord[];
  if (!Array.isArray(data)) return [];
  return data.filter(
    (r) => r && typeof r.flux === "number" && Number.isFinite(r.flux) && r.flux > 0,
  );
}

/** Route a sample to its per-stream Detector Durable Object (`/ingest`). */
async function routeToDetector(env: Env, sample: FluxSample): Promise<void> {
  const id = env.DETECTOR.idFromName(sample.stream); // deterministic, O(1)
  const stub = env.DETECTOR.get(id);
  await stub.fetch("https://do/ingest", {
    method: "POST",
    body: JSON.stringify(sample),
    headers: { "content-type": "application/json" },
  });
}

/**
 * One ingest tick: fetch, normalise, route the latest soft-band samples to the
 * DO, and write `latest:*` to KV. Routes only the *newest* sample per stream to
 * the detector each tick (the DO holds the streaming state; replaying the whole
 * day every minute would double-count) while still caching latest values.
 */
export async function ingestOnce(env: Env): Promise<{ ingested: number }> {
  const records = await fetchSwpcXrays(env);
  if (records.length === 0) return { ingested: 0 };

  const samples = records.map(normalizeSwpc);

  // Keep the latest sample per stream (records arrive in ascending time order).
  const latestByStream = new Map<string, FluxSample>();
  for (const s of samples) {
    const prev = latestByStream.get(s.stream);
    if (!prev || s.t >= prev.t) latestByStream.set(s.stream, s);
  }

  let ingested = 0;
  for (const [stream, sample] of latestByStream) {
    // Cache the latest value for O(1) front-Worker reads.
    await env.FLARE_KV.put(
      `${KV_KEY_LATEST}${stream}`,
      JSON.stringify({
        stream: sample.stream,
        t: sample.t,
        value: sample.value,
        unit: sample.unit,
        cls: sample.cls ?? classifyFlux(sample.value),
      }),
    );
    // Drive the streaming detector with the newest soft-band sample only.
    if (sample.quantity === "SXR_LONG") {
      await routeToDetector(env, sample);
      ingested += 1;
    }
  }
  return { ingested };
}

/** The Cron Trigger entry point (registered in `wrangler.toml`). */
export default {
  async scheduled(
    _event: ScheduledController,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    ctx.waitUntil(
      ingestOnce(env).catch((err) => {
        console.error("cron ingest error:", err);
      }),
    );
  },
};
