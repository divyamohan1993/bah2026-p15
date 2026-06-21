/**
 * Cross-substrate parity harness -- the TypeScript/JS half of the contract
 * (ARCHITECTURE.md Appendix B.8; the Python half is tests/test_parity.py).
 *
 * The architecture fixes a hard invariant: the edge detector
 * (edge/src/detector.ts) and the Python detector
 * (flarecast/detect/{primitives,cusum}.py) produce the SAME onset decisions on
 * the same input sequence with the same alpha/k/h. That invariant is anchored
 * by a single committed golden vector, tests/golden/cusum_golden.json, generated
 * once from the Python reference pipeline.
 *
 * This file loads that SAME golden, re-runs the JS mirror of the gated
 * EMA + EWMV + CUSUM hot-path pipeline over the stored input, and asserts it
 * reproduces every stored array exactly (after the generator's 9-dp rounding).
 * Python and TypeScript are therefore validated against one shared source of
 * truth -- the definition of the parity invariant.
 *
 * The math below is a byte-for-byte mirror of the pure functions exported from
 * edge/src/detector.ts (EMA, EWMV, CUSUMDetector, parityStep), re-implemented
 * inline so the harness is pure Node ESM with no build step / no TS toolchain
 * required: run with `node edge/test/parity.mjs` (this is what `cd edge &&
 * npm test` and the edge CI job invoke). Keeping the math inline here AND in
 * detector.ts is intentional: if the two drift, this harness stops reproducing
 * the golden and fails, exactly like the Python side.
 *
 * Exit code 0 on success, 1 on any mismatch (so CI fails loudly).
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The ONE shared source of truth, loaded (never regenerated) by both substrates.
const GOLDEN_PATH = join(__dirname, "..", "..", "tests", "golden", "cusum_golden.json");

// ---------------------------------------------------------------------------
// O(1) streaming primitives -- byte-for-byte mirrors of edge/src/detector.ts
// (which themselves mirror flarecast/detect/{primitives,cusum}.py).
// ---------------------------------------------------------------------------

/** Recursive EMA: m = alpha*m + (1-alpha)*x. */
class EMA {
  constructor(alpha, x0 = 0.0) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new RangeError(`alpha must be in (0, 1), got ${alpha}`);
    }
    this.a = alpha;
    this.m = x0;
  }
  update(x) {
    this.m = this.a * this.m + (1.0 - this.a) * x;
    return this.m;
  }
}

/** EWMV mean + variance: d=x-m; m+=(1-a)*d; S=a*(S+(1-a)*d*d). */
class EWMV {
  constructor(alpha, x0 = 0.0) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new RangeError(`alpha must be in (0, 1), got ${alpha}`);
    }
    this.a = alpha;
    this.m = x0;
    this.S = 0.0;
  }
  update(x) {
    const d = x - this.m;
    this.m = this.m + (1.0 - this.a) * d;
    this.S = this.a * (this.S + (1.0 - this.a) * d * d);
    return [this.m, this.S];
  }
}

/** One-sided upper CUSUM (mirrors cusum.CUSUMDetector / detector.ts). */
class CUSUMDetector {
  constructor(kSlack, h) {
    if (kSlack < 0.0) throw new RangeError(`k_slack must be >= 0, got ${kSlack}`);
    if (h <= 0.0) throw new RangeError(`h must be > 0, got ${h}`);
    this.k = kSlack;
    this.h = h;
    this.S = 0.0;
    this.t0 = null;
    this.inEvent = false;
  }
  update(x, baseline, sigma, t) {
    const s = sigma > 0.0 ? sigma : 1e-12;
    if (this.S <= 0.0) this.t0 = t;
    this.S = Math.max(0.0, this.S + (x - baseline) - this.k * s);
    const threshold = this.h * s;
    if (this.S > threshold) {
      const onsetTime = this.t0;
      this.S = 0.0;
      this.inEvent = true;
      return { onset: true, inEvent: true, statistic: 0.0, onsetTime };
    }
    return { onset: false, inEvent: this.inEvent, statistic: this.S, onsetTime: null };
  }
  reset(inEvent = false) {
    this.S = 0.0;
    this.t0 = null;
    this.inEvent = inEvent;
  }
}

// ---------------------------------------------------------------------------
// The minimal hot-path pipeline -- the cross-substrate parity contract.
// Mirrors tests/golden/generate_cusum_golden.run_pipeline AND
// edge/src/detector.parityStep.
// ---------------------------------------------------------------------------

/** Round to 9 dp, matching the generator's `round(v, 9)` before it stores. */
function round9(v) {
  // Number.prototype.toFixed uses round-half-away-from-zero like Python's
  // round() does not, but the generator's values are far from .5 ulps at 9 dp;
  // we match its decimal string exactly via toFixed(9) -> Number, which is what
  // a JSON byte-for-byte comparison against the rounded golden needs.
  return Number(v.toFixed(9));
}

function runPipeline(x, params) {
  const { alpha, k_slack: kSlack, h, sigma_floor: sigmaFloor, exit_sigmas: exitSigmas } =
    params;

  const ema = new EMA(alpha, x.length ? x[0] : 0.0);
  const ewmv = new EWMV(alpha, x.length ? x[0] : 0.0);
  const cusum = new CUSUMDetector(kSlack, h);

  let inEvent = false;
  let frozenBaseline = 0.0;
  let frozenSigma = sigmaFloor;

  const statistic = [];
  const baselineOut = [];
  const sigmaOut = [];
  const inEventOut = [];
  const onsetIndices = [];
  const onsetTimes = [];

  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const t = i; // unit time steps, like the generator's float(i)

    let baseline;
    let sigma;

    // 1. Gated baseline / scale (frozen while in an event).
    if (!inEvent) {
      baseline = ema.update(xi);
      const [, varEw] = ewmv.update(xi);
      let sd = varEw > 0.0 ? Math.sqrt(varEw) : 0.0;
      if (!(sd > sigmaFloor)) sd = sigmaFloor;
      sigma = sd;
      frozenBaseline = baseline;
      frozenSigma = sigma;
    } else {
      baseline = frozenBaseline;
      sigma = frozenSigma;
    }

    // 2. CUSUM update (the load-bearing parity math).
    const state = cusum.update(xi, baseline, sigma, t);

    // 3a. Onset reported only on the FIRST alarm of a burst (one per flare).
    if (state.onset && !inEvent) {
      inEvent = true;
      onsetIndices.push(i);
      onsetTimes.push(state.onsetTime !== null ? state.onsetTime : t);
      // 3b. Event-exit: clean return to (frozen) baseline re-arms detection.
    } else if (inEvent && xi - frozenBaseline <= exitSigmas * frozenSigma) {
      inEvent = false;
      cusum.reset(false);
    }

    statistic.push(round9(state.statistic));
    baselineOut.push(round9(baseline));
    sigmaOut.push(round9(sigma));
    inEventOut.push(inEvent);
  }

  return {
    statistic,
    baseline: baselineOut,
    sigma: sigmaOut,
    in_event: inEventOut,
    onset_indices: onsetIndices,
    onset_times: onsetTimes,
  };
}

// ---------------------------------------------------------------------------
// Assertions.
// ---------------------------------------------------------------------------

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  ok   - ${name}`);
  } else {
    failures++;
    console.error(`  FAIL - ${name}${detail ? `: ${detail}` : ""}`);
  }
}

/** Compare two numeric arrays element-wise (already rounded to 9 dp on both). */
function arraysEqual(a, b) {
  if (a.length !== b.length) {
    return { ok: false, detail: `length ${a.length} != ${b.length}` };
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return { ok: false, detail: `index ${i}: ${a[i]} != ${b[i]}` };
    }
  }
  return { ok: true, detail: "" };
}

function main() {
  const golden = JSON.parse(readFileSync(GOLDEN_PATH, "utf-8"));
  console.log(`edge parity harness: loaded ${GOLDEN_PATH}`);
  console.log(
    `  golden: n=${golden.n}, params=${JSON.stringify(golden.params)}, ` +
      `onsets=${JSON.stringify(golden.expected.onset_indices)}`,
  );

  const exp = golden.expected;
  const got = runPipeline(golden.input, golden.params);

  check("n matches input length", golden.n === golden.input.length);

  const stat = arraysEqual(got.statistic, exp.statistic);
  check("statistic[] reproduced", stat.ok, stat.detail);

  const base = arraysEqual(got.baseline, exp.baseline);
  check("baseline[] reproduced", base.ok, base.detail);

  const sig = arraysEqual(got.sigma, exp.sigma);
  check("sigma[] reproduced", sig.ok, sig.detail);

  const inEvtOk =
    got.in_event.length === exp.in_event.length &&
    got.in_event.every((v, i) => v === exp.in_event[i]);
  check("in_event[] reproduced", inEvtOk);

  const onsetIdx = arraysEqual(got.onset_indices, exp.onset_indices);
  check("onset_indices reproduced", onsetIdx.ok, onsetIdx.detail);

  const onsetT = arraysEqual(got.onset_times, exp.onset_times);
  check("onset_times reproduced", onsetT.ok, onsetT.detail);

  // The fixture injects three flares; the JS mirror must find exactly three.
  check(
    "exactly three onsets detected",
    got.onset_indices.length === 3,
    `got ${got.onset_indices.length}`,
  );

  if (failures === 0) {
    console.log(
      "\nOK - edge (TS/JS) detector reproduces the Python golden EXACTLY " +
        "(cross-substrate parity holds).",
    );
    process.exit(0);
  } else {
    console.error(`\nFAILED - ${failures} parity check(s) did not match the golden.`);
    process.exit(1);
  }
}

main();
