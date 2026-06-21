/**
 * Edge constants -- the TypeScript mirror of the subset of
 * `flarecast/constants.py` the edge hot path needs (ARCHITECTURE.md Section 4 /
 * Appendix B). Values MUST match the Python source of truth so the two
 * substrates agree (the parity invariant, Appendix B.8).
 */

// --- CUSUM / EMA detector defaults (constants.py Section "Detector defaults") ---
/** CUSUM slack k_c in sigma units (`flarecast.constants.CUSUM_K_SLACK`). */
export const CUSUM_K_SLACK = 0.5;
/** CUSUM decision interval h_c in sigma units (`flarecast.constants.CUSUM_H`). */
export const CUSUM_H = 5.0;
/** EMA/EWMV forgetting factor (`flarecast.constants.DEFAULT_EMA_ALPHA`). */
export const DEFAULT_EMA_ALPHA = 0.99;

// --- Parity-pipeline constants (tests/golden/generate_cusum_golden.py) ---
/** Floor on sigma so a flat/degenerate scale never breaks the arithmetic. */
export const PARITY_SIGMA_FLOOR = 1e-12;
/** Return-to-baseline event-exit threshold in (frozen) sigmas. */
export const PARITY_EXIT_SIGMAS = 1.0;

// --- GOES A-X classification decade edges (constants.py Section 4.5) ---
export const GOES_CLASS_A_WM2 = 1e-8;
export const GOES_CLASS_B_WM2 = 1e-7;
export const GOES_CLASS_C_WM2 = 1e-6;
export const GOES_CLASS_M_WM2 = 1e-5;
export const GOES_CLASS_X_WM2 = 1e-4;

/** Lower-bound flux of each GOES class letter (mirrors GOES_CLASS_THRESHOLDS_WM2). */
export const GOES_CLASS_THRESHOLDS_WM2: Record<string, number> = {
  A: GOES_CLASS_A_WM2,
  B: GOES_CLASS_B_WM2,
  C: GOES_CLASS_C_WM2,
  M: GOES_CLASS_M_WM2,
  X: GOES_CLASS_X_WM2,
};

// --- Data access (constants.py Section "Data access") ---
/** NOAA SWPC real-time GOES XRS JSON (live nowcast anchor, no auth). */
export const SWPC_XRAYS_URL =
  "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json";
/** NOAA SWPC GOES X-ray flare event list. */
export const SWPC_FLARES_URL =
  "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json";

// --- KV key namespaces (ARCHITECTURE.md Section 7.1 hot-state keys) ---
/** `latest:<stream>` -> last sample + statistic (O(1) read). */
export const KV_KEY_LATEST = "latest:";
/** `alert:<stream>` -> most recent onset alert (O(1) read). */
export const KV_KEY_ALERT = "alert:";
/** `forecast:<stream>` -> latest forecast record (O(1) read). */
export const KV_KEY_FORECAST = "forecast:";

// --- Canonical SWPC GOES stream ids (match the Python ingest normaliser) ---
/** SWPC long channel (1-8 A) -> the primary soft-band nowcast stream. */
export const STREAM_GOES_LONG = "goes-primary-long";
/** SWPC short channel (0.5-4 A). */
export const STREAM_GOES_SHORT = "goes-primary-short";

/** SWPC `energy` field values that identify each XRS channel. */
export const SWPC_ENERGY_LONG = "0.1-0.8nm";
export const SWPC_ENERGY_SHORT = "0.05-0.4nm";
