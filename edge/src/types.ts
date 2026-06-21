/**
 * Shared edge types -- the TypeScript mirror of `flarecast/types.py` +
 * `flarecast/constants.py` (ARCHITECTURE.md Appendix B.0 / B.8, Section 9).
 *
 * Field names and the QC integer values here MUST match the Python side
 * byte-for-byte: the same `FluxSample` flows through both substrates and the
 * same QC bitmask is written to D1 (`schema.sql` / ARCHITECTURE.md Section 9.4).
 * Keep this file dependency-free so it can be imported by the Worker, the
 * Durable Object, and the Node parity harness alike.
 */

/** Physical quantity a sample estimates (mirrors `flarecast.types.Quantity`). */
export type Quantity =
  | "SXR_LONG"
  | "SXR_SHORT"
  | "HXR"
  | "EUV"
  | "MAGSCALAR"
  | "RADIO"
  | "PROTON";

/**
 * Canonical ingest record: one sample of one stream
 * (mirrors `flarecast.types.FluxSample`, ARCHITECTURE.md Section 9.1 / B.8).
 */
export interface FluxSample {
  /** Stream id, e.g. "goes-primary-long", "solexs-sxr-long". */
  stream: string;
  /** Epoch seconds UTC (observed). */
  t: number;
  /** Measurement in the canonical unit for the quantity. */
  value: number;
  /** Canonical unit string, e.g. "W m^-2" or "counts/s". */
  unit: string;
  /** Provider, e.g. "SWPC", "AdityaL1-SoLEXS", "synth". */
  source: string;
  /** One of the {@link Quantity} values (stored as a plain string). */
  quantity: string;
  /** Derived GOES flare class, e.g. "C3.1" (optional). */
  cls?: string;
  /** QC bitmask (see {@link QCBit}); 0 means "unset / not yet QC'd". */
  qc?: number;
}

/**
 * QC bitmask values for {@link FluxSample.qc}
 * (mirror `flarecast.constants.QC_*` / `flarecast.types.QCBit`,
 * ARCHITECTURE.md Section 9.4). Stored as a bitmask so multiple conditions
 * coexist on one sample, e.g. `QCBit.FILLED | QCBit.NEAR_SAA`. These integer
 * values are shared verbatim with the Python/D1 substrate.
 */
export const QCBit = {
  GOOD: 1,
  INTERPOLATED: 2,
  FILLED: 4,
  SUSPECT: 8,
  BAD: 16,
  NEAR_SAA: 32,
  SATURATED: 64,
  DATA_GAP: 128,
  SPIKE_REJECTED: 256,
} as const;

export type QCBitName = keyof typeof QCBit;

/**
 * Result returned by the streaming detector update
 * (mirrors `flarecast.types.DetectionState`). The edge DO returns this shape
 * from its O(1) `update` step.
 */
export interface DetectionState {
  /** Onset fired on this sample. */
  onset: boolean;
  /** Currently inside a flare. */
  inEvent: boolean;
  /** The detector's running statistic (CUSUM S). */
  statistic: number;
  /** MLE / edge onset time (epoch seconds UTC) when `onset` fires, else null. */
  onsetTime: number | null;
}

/** A master-catalogue flare row as written to D1 (mirrors §9.4 columns). */
export interface FlareRow {
  event_id: string;
  t_start: number; // epoch ms
  t_peak: number; // epoch ms
  t_end: number; // epoch ms
  goes_class: string;
  soft_detected: number; // 0/1
  hard_detected: number; // 0/1
  peak_flux_goes: number | null;
  peak_counts: number | null;
  detector_used: string | null;
  neupert_ok: number; // 0/1
  confidence: number;
  detectors: string; // JSON array string
  qc_bitmask: number;
  ref_catalog: string | null;
  ref_id: string | null;
  ref_dt_s: number | null;
}
