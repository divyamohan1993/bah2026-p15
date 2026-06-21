/**
 * `DetectorDO` -- the per-stream streaming detector Durable Object.
 *
 * ARCHITECTURE.md Section 7.1 / Appendix B.8: the edge hot path is a single-
 * instance, in-memory, strongly-consistent online detector. It holds **O(1)
 * state** (a handful of floats) and does **O(1) work per sample**: a gated
 * EMA baseline + EWMV variance feeding a one-sided upper CUSUM -- the *same
 * math* as `flarecast/detect/{primitives,cusum}.py`. The cross-substrate parity
 * invariant (Appendix B.8) is locked by a golden vector that both this file's
 * logic and the Python reference reproduce exactly (see `edge/test/parity.mjs`
 * and `tests/test_parity.py`).
 *
 * Live fan-out uses a **Hibernatable WebSocket**: the DO can sleep while still
 * holding client sockets, so idle subscribers cost nothing
 * (ARCHITECTURE.md Section 7.1, "DO WebSocket (Hibernatable)").
 *
 * The pure math (EMA / EWMV / CUSUM / `parityStep`) is exported as plain
 * functions so the Node parity harness can import and validate it without the
 * Workers runtime. The class merely wraps that math in the DO lifecycle.
 */

import {
  CUSUM_H,
  CUSUM_K_SLACK,
  DEFAULT_EMA_ALPHA,
  GOES_CLASS_A_WM2,
  GOES_CLASS_THRESHOLDS_WM2,
  GOES_CLASS_X_WM2,
  KV_KEY_ALERT,
  KV_KEY_LATEST,
  PARITY_EXIT_SIGMAS,
  PARITY_SIGMA_FLOOR,
} from "./constants";
import type { DetectionState, FluxSample } from "./types";

// ---------------------------------------------------------------------------
// O(1) streaming primitives -- byte-for-byte mirrors of
// flarecast/detect/primitives.py (EMA, EWMV) and flarecast/detect/cusum.py.
// ---------------------------------------------------------------------------

/** Recursive EMA: `m = alpha*m + (1-alpha)*x` (mirrors primitives.EMA). */
export class EMA {
  private a: number;
  m: number;
  constructor(alpha: number, x0 = 0.0) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new RangeError(`alpha must be in (0, 1), got ${alpha}`);
    }
    this.a = alpha;
    this.m = x0;
  }
  update(x: number): number {
    this.m = this.a * this.m + (1.0 - this.a) * x;
    return this.m;
  }
  get value(): number {
    return this.m;
  }
}

/**
 * Exponentially-weighted mean + variance (mirrors primitives.EWMV)::
 *
 *   d = x - m;  m += (1-alpha)*d;  S = alpha*(S + (1-alpha)*d*d)
 */
export class EWMV {
  private a: number;
  m: number;
  S: number;
  constructor(alpha: number, x0 = 0.0) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new RangeError(`alpha must be in (0, 1), got ${alpha}`);
    }
    this.a = alpha;
    this.m = x0;
    this.S = 0.0;
  }
  update(x: number): [number, number] {
    const d = x - this.m;
    this.m = this.m + (1.0 - this.a) * d;
    this.S = this.a * (this.S + (1.0 - this.a) * d * d);
    return [this.m, this.S];
  }
  sd(): number {
    return this.S > 0.0 ? Math.sqrt(this.S) : 0.0;
  }
}

/**
 * One-sided upper CUSUM for a flux increase (mirrors cusum.CUSUMDetector)::
 *
 *   S_t = max(0, S_{t-1} + (x_t - baseline) - k*sigma)
 *   alarm when S_t > h*sigma; on alarm record onset_time = last-reset, reset S=0
 */
export class CUSUMDetector {
  private k: number;
  private h: number;
  S: number;
  private t0: number | null;
  private inEvent: boolean;
  constructor(kSlack: number = CUSUM_K_SLACK, h: number = CUSUM_H) {
    if (kSlack < 0.0) throw new RangeError(`k_slack must be >= 0, got ${kSlack}`);
    if (h <= 0.0) throw new RangeError(`h must be > 0, got ${h}`);
    this.k = kSlack;
    this.h = h;
    this.S = 0.0;
    this.t0 = null;
    this.inEvent = false;
  }
  update(x: number, baseline: number, sigma: number, t: number): DetectionState {
    // Floor sigma so a flat/degenerate scale never breaks the arithmetic.
    const s = sigma > 0.0 ? sigma : 1e-12;

    // The instant the statistic sits at zero is the provisional change point.
    if (this.S <= 0.0) this.t0 = t;

    this.S = Math.max(0.0, this.S + (x - baseline) - this.k * s);

    const threshold = this.h * s;
    if (this.S > threshold) {
      const onsetTime = this.t0;
      this.S = 0.0;
      this.inEvent = true;
      return { onset: true, inEvent: true, statistic: 0.0, onsetTime };
    }
    return {
      onset: false,
      inEvent: this.inEvent,
      statistic: this.S,
      onsetTime: null,
    };
  }
  reset(inEvent = false): void {
    this.S = 0.0;
    this.t0 = null;
    this.inEvent = inEvent;
  }
}

// ---------------------------------------------------------------------------
// GOES A-X classification -- mirrors flarecast/detect/classify.classify_flux.
// ---------------------------------------------------------------------------

/**
 * Return the GOES class string for a peak 1-8 A flux (mirrors
 * `flarecast.detect.classify.classify_flux`, ARCHITECTURE.md Section 4.5).
 * Closed-form, O(1). Examples: 2.5e-5 -> "M2.5", 1e-4 -> "X1.0",
 * 5e-8 -> "A5.0", 1e-9 -> "A0.1", <=0 -> "A0.0".
 */
export function classifyFlux(fluxWm2: number): string {
  const f = fluxWm2;
  if (f <= 0.0) return "A0.0";

  const e = Math.floor(Math.log10(f));

  if (e >= -4) {
    // X-class is open-ended: mantissa measured against the X floor (1e-4).
    const mant = f / GOES_CLASS_X_WM2;
    return `X${mant.toFixed(1)}`;
  }
  if (e < -8) {
    // Below the nominal A decade floor: clamp to 'A', mantissa vs 1e-8.
    const mant = f / GOES_CLASS_A_WM2;
    return `A${mant.toFixed(1)}`;
  }
  const expToLetter: Record<number, string> = {
    [-8]: "A",
    [-7]: "B",
    [-6]: "C",
    [-5]: "M",
    [-4]: "X",
  };
  const letter = expToLetter[e];
  const mant = f / Math.pow(10.0, e);
  return `${letter}${mant.toFixed(1)}`;
}

/** Inverse of {@link classifyFlux}: GOES class string -> peak flux (W m^-2). */
export function classToFlux(cls: string): number {
  if (!cls) throw new Error("empty class string");
  const letter = cls[0].toUpperCase();
  const floor = GOES_CLASS_THRESHOLDS_WM2[letter];
  if (floor === undefined) throw new Error(`unknown GOES class letter: ${cls}`);
  const mantStr = cls.slice(1).trim();
  const mant = mantStr ? parseFloat(mantStr) : 1.0;
  return mant * floor;
}

// ---------------------------------------------------------------------------
// The minimal hot-path pipeline -- the cross-substrate parity contract.
// ---------------------------------------------------------------------------

/** O(1) streaming state for {@link parityStep} (held in DO memory). */
export interface DetectorState {
  ema: EMA;
  ewmv: EWMV;
  cusum: CUSUMDetector;
  inEvent: boolean;
  frozenBaseline: number;
  frozenSigma: number;
  seeded: boolean;
}

/** Construct a fresh detector state seeded at `x0`. O(1). */
export function makeDetectorState(
  x0: number,
  alpha: number = DEFAULT_EMA_ALPHA,
  kSlack: number = CUSUM_K_SLACK,
  h: number = CUSUM_H,
): DetectorState {
  return {
    ema: new EMA(alpha, x0),
    ewmv: new EWMV(alpha, x0),
    cusum: new CUSUMDetector(kSlack, h),
    inEvent: false,
    frozenBaseline: 0.0,
    frozenSigma: PARITY_SIGMA_FLOOR,
    seeded: false,
  };
}

/** Per-step result of the parity pipeline (one sample). */
export interface ParityStepResult {
  statistic: number;
  baseline: number;
  sigma: number;
  inEvent: boolean;
  onset: boolean;
  onsetTime: number | null;
}

/**
 * Fold one sample `(x, t)` through the gated EMA+EWMV+CUSUM pipeline.
 *
 * This is the SAME algorithm as `tests/golden/generate_cusum_golden.run_pipeline`
 * (ARCHITECTURE.md Appendix B.8). It mutates `st` in place (O(1)) and returns
 * the per-step statistic / baseline / sigma / onset for the caller to record or
 * broadcast.
 *
 * Steps (the contract):
 *  1. Gated baseline/scale: fold into EMA+EWMV while NOT in an event; freeze
 *     both while in an event so the flare cannot contaminate the baseline.
 *  2. CUSUM update on `(x, baseline, sigma, t)`.
 *  3. One-onset-per-flare latch + return-to-baseline exit.
 */
export function parityStep(st: DetectorState, x: number, t: number): ParityStepResult {
  let baseline: number;
  let sigma: number;

  // 1. Gated baseline / scale (frozen while in an event).
  if (!st.inEvent) {
    baseline = st.ema.update(x);
    const [, varEw] = st.ewmv.update(x);
    let sd = varEw > 0.0 ? Math.sqrt(varEw) : 0.0;
    if (!(sd > PARITY_SIGMA_FLOOR)) sd = PARITY_SIGMA_FLOOR;
    sigma = sd;
    st.frozenBaseline = baseline;
    st.frozenSigma = sigma;
  } else {
    baseline = st.frozenBaseline;
    sigma = st.frozenSigma;
  }

  // 2. CUSUM update (the load-bearing parity math).
  const state = st.cusum.update(x, baseline, sigma, t);

  let onset = false;
  let onsetTime: number | null = null;

  // 3a. Onset reported only on the FIRST alarm of a burst (one per flare).
  if (state.onset && !st.inEvent) {
    st.inEvent = true;
    onset = true;
    onsetTime = state.onsetTime !== null ? state.onsetTime : t;
  } else if (
    // 3b. Event-exit: clean return to (frozen) baseline re-arms detection.
    st.inEvent &&
    x - st.frozenBaseline <= PARITY_EXIT_SIGMAS * st.frozenSigma
  ) {
    st.inEvent = false;
    st.cusum.reset(false);
  }

  return {
    statistic: state.statistic,
    baseline,
    sigma,
    inEvent: st.inEvent,
    onset,
    onsetTime,
  };
}

// ---------------------------------------------------------------------------
// Cloudflare runtime ambient types (minimal; the real types come from
// @cloudflare/workers-types at build time -- declared here so this file is
// self-describing and the Node parity harness can import the pure math without
// the Workers lib present).
// ---------------------------------------------------------------------------
interface Env {
  FLARE_KV: KVNamespace;
  FLARE_DB: D1Database;
  FLARE_R2: R2Bucket;
  DETECTOR: DurableObjectNamespace;
}

// ---------------------------------------------------------------------------
// DetectorDO -- the Durable Object.
// ---------------------------------------------------------------------------

/** A persisted snapshot of the O(1) detector state (survives DO eviction). */
interface PersistedState {
  emaM: number;
  ewmvM: number;
  ewmvS: number;
  cusumS: number;
  inEvent: boolean;
  frozenBaseline: number;
  frozenSigma: number;
  seeded: boolean;
}

export class DetectorDO {
  private state: DurableObjectState;
  private env: Env;
  private det: DetectorState;
  private loaded = false;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
    this.det = makeDetectorState(0.0);
  }

  /** Lazily rehydrate the O(1) detector state from durable storage. */
  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return;
    const p = await this.state.storage.get<PersistedState>("det");
    if (p) {
      this.det.ema.m = p.emaM;
      this.det.ewmv.m = p.ewmvM;
      this.det.ewmv.S = p.ewmvS;
      this.det.cusum.S = p.cusumS;
      this.det.inEvent = p.inEvent;
      this.det.frozenBaseline = p.frozenBaseline;
      this.det.frozenSigma = p.frozenSigma;
      this.det.seeded = p.seeded;
    }
    this.loaded = true;
  }

  /** Persist the (constant-size) detector state. */
  private async persist(): Promise<void> {
    const p: PersistedState = {
      emaM: this.det.ema.m,
      ewmvM: this.det.ewmv.m,
      ewmvS: this.det.ewmv.S,
      cusumS: this.det.cusum.S,
      inEvent: this.det.inEvent,
      frozenBaseline: this.det.frozenBaseline,
      frozenSigma: this.det.frozenSigma,
      seeded: this.det.seeded,
    };
    await this.state.storage.put("det", p);
  }

  /** DO entry point: `/ingest` (POST sample) and `/ws` (WebSocket upgrade). */
  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname.endsWith("/ws")) {
      return this.handleWebSocket(req);
    }
    if (url.pathname.endsWith("/ingest")) {
      const sample = (await req.json()) as FluxSample;
      return this.ingest(sample);
    }
    return new Response("not found", { status: 404 });
  }

  /**
   * Fold one sample through the detector (O(1)), update KV (`latest:*`, and on
   * onset `alert:*`), broadcast to subscribers, and persist. Returns the
   * detection result as JSON.
   */
  async ingest(sample: FluxSample): Promise<Response> {
    await this.ensureLoaded();

    // Seed the estimators to the first observed value (cold-start guard).
    if (!this.det.seeded) {
      this.det.ema.m = sample.value;
      this.det.ewmv.m = sample.value;
      this.det.seeded = true;
    }

    const step = parityStep(this.det, sample.value, sample.t);

    const latest = {
      stream: sample.stream,
      t: sample.t,
      value: sample.value,
      unit: sample.unit,
      statistic: step.statistic,
      baseline: step.baseline,
      in_event: step.inEvent,
      cls: classifyFlux(sample.value),
    };
    await this.env.FLARE_KV.put(
      `${KV_KEY_LATEST}${sample.stream}`,
      JSON.stringify(latest),
    );

    if (step.onset) {
      const alert = {
        stream: sample.stream,
        onset: true,
        onset_time: step.onsetTime,
        t: sample.t,
        value: sample.value,
        cls: classifyFlux(sample.value),
        detector: "CUSUM",
      };
      await this.env.FLARE_KV.put(
        `${KV_KEY_ALERT}${sample.stream}`,
        JSON.stringify(alert),
      );
      this.broadcast(JSON.stringify({ type: "alert", ...alert }));
    }

    this.broadcast(JSON.stringify({ type: "sample", ...latest }));
    await this.persist();

    return new Response(JSON.stringify({ ...latest, onset: step.onset }), {
      headers: { "content-type": "application/json" },
    });
  }

  /** Accept a Hibernatable WebSocket so idle subscribers cost nothing. */
  private handleWebSocket(req: Request): Response {
    const upgrade = req.headers.get("Upgrade");
    if (upgrade !== "websocket") {
      return new Response("expected websocket", { status: 426 });
    }
    const pair = new WebSocketPair();
    const [client, server] = [pair[0], pair[1]];
    // Hibernation API: the runtime can evict the DO while keeping the socket.
    this.state.acceptWebSocket(server);
    return new Response(null, { status: 101, webSocket: client });
  }

  /** Fan out a message to every connected (incl. hibernating) subscriber. */
  private broadcast(message: string): void {
    for (const ws of this.state.getWebSockets()) {
      try {
        ws.send(message);
      } catch {
        // Drop sockets that error on send; the runtime reaps closed ones.
      }
    }
  }

  /** Hibernation hook: a closing socket needs no special handling here. */
  async webSocketClose(ws: WebSocket): Promise<void> {
    try {
      ws.close();
    } catch {
      // already closed
    }
  }
}
