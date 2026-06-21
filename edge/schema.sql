-- Aditya FlareCast — D1 (serverless SQLite) DDL for the edge analytics path.
--
-- This MUST mirror, byte-for-byte, the offline SQLite schema and the Python
-- catalogue record so the two substrates agree (ARCHITECTURE.md Section 9.4;
-- flarecast/catalog/schema.py `FlareEvent.sql_columns` / `to_sql_row`).
--
-- Column order and names below match `FlareEvent.sql_columns()` exactly:
--   (event_id, t_start, t_peak, t_end, goes_class, soft_detected,
--    hard_detected, peak_flux_goes, peak_counts, detector_used, neupert_ok,
--    confidence, detectors, qc_bitmask, ref_catalog, ref_id, ref_dt_s)
--
-- Times are epoch MILLISECONDS (UTC, t_earth frame): the Python dataclass works
-- in epoch seconds and `to_sql_row()` multiplies by 1000 to this integer unit.
--
-- Apply with:  wrangler d1 execute FLARE_DB --remote --file schema.sql
--    (local:   wrangler d1 execute FLARE_DB --local  --file schema.sql)

-- Master flare catalogue (one row = one physical flare). The front Worker's
-- /api/at query is an indexed lookup on t_peak (ARCHITECTURE.md Section 7.3:
-- the deliberate O(log n + k) analytics path, NOT forced to be O(1)).
CREATE TABLE IF NOT EXISTS flare_catalogue (
  event_id       TEXT PRIMARY KEY,   -- stable UUID hex (catalog.new_event_id)
  t_start        INTEGER,            -- epoch ms (UTC, t_earth frame) — earliest band onset
  t_peak         INTEGER,            -- epoch ms — soft-band peak (canonical)
  t_end          INTEGER,            -- epoch ms — soft-band FSM-midpoint end
  goes_class     TEXT,               -- GOES class string, e.g. "M2.5"
  soft_detected  INTEGER,            -- 0/1 — SoLEXS/GOES soft band fired
  hard_detected  INTEGER,            -- 0/1 — HEL1OS hard band fired
  peak_flux_goes REAL,               -- W m^-2 — GOES-equivalent soft peak flux
  peak_counts    REAL,               -- HXR counts/s — hard-band peak
  detector_used  TEXT,               -- soft detector path, e.g. "SDD1"|"SDD2"
  neupert_ok     INTEGER,            -- 0/1 — Neupert-consistent (HXR ~ d/dt SXR)
  confidence     REAL,               -- noisy-OR fused detector confidence [0,1]
  detectors      TEXT,               -- JSON array of firing detectors, e.g. ["CUSUM","FOCuS"]
  qc_bitmask     INTEGER,            -- QC bits (see legend below)
  ref_catalog    TEXT,               -- reference-catalogue source, e.g. "GOES/HEK"
  ref_id         TEXT,               -- matched reference event id
  ref_dt_s       REAL                -- peak time offset to the reference match (s)
);

-- Range / backtest queries by peak time (the analytics path's primary index;
-- ARCHITECTURE.md Section 7.1 "D1 SQLite ... INDEX peak_time").
CREATE INDEX IF NOT EXISTS idx_cat_tpeak ON flare_catalogue(t_peak);
-- Secondary index for per-class queries (POD-by-class, class filters).
CREATE INDEX IF NOT EXISTS idx_cat_class ON flare_catalogue(goes_class);

-- QC bitmask bits (shared edge + offline; flarecast/constants.py / types.QCBit,
-- ARCHITECTURE.md Section 9.4):
--   1=GOOD 2=INTERPOLATED 4=FILLED 8=SUSPECT 16=BAD
--   32=near_SAA 64=saturated 128=data_gap 256=spike_rejected

-- Optional per-stream sample rollup: a compact downsampled history of the live
-- light curve for backtest plotting, so the dashboard's catalogue/light-curve
-- view has a queryable series without re-reading raw R2 objects. The O(1) hot
-- path uses KV `latest:<stream>`; this table is the durable analytics mirror.
CREATE TABLE IF NOT EXISTS samples (
  stream     TEXT NOT NULL,          -- stream id, e.g. "goes-primary-long"
  t          INTEGER NOT NULL,       -- epoch ms (UTC) of the sample
  value      REAL NOT NULL,          -- canonical-unit value (W m^-2 for SXR)
  unit       TEXT,                   -- canonical unit, e.g. "W m^-2"
  statistic  REAL,                   -- detector statistic (CUSUM S) at this sample
  baseline   REAL,                   -- gated baseline at this sample
  in_event   INTEGER,                -- 0/1 — detector was inside a flare
  cls        TEXT,                   -- derived GOES class at this sample
  qc_bitmask INTEGER DEFAULT 0,      -- QC bits (legend above)
  PRIMARY KEY (stream, t)
);

-- Time-range queries over the rollup (per stream, by time).
CREATE INDEX IF NOT EXISTS idx_samples_stream_t ON samples(stream, t);
