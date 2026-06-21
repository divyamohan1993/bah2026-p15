"""Physical and configuration constants for ``flarecast``.

Every value here is grounded in ARCHITECTURE.md (the source of truth) or the
research deliverables under ``docs/research/``; the relevant section is cited
inline. Like :mod:`flarecast.types`, this module is **pure standard library**
(no numpy / pandas) so it is always importable, including offline.

Defaults that also appear in the Appendix B function/class signatures
(``CUSUMDetector(k_slack=0.5, h=5.0)`` etc.) are mirrored here as named
constants so a single value governs both the contract default and any caller
that wants to reference it explicitly. **When the prose and an appendix
disagree, the appendix wins** -- these constants follow the appendix.
"""

from __future__ import annotations

from typing import Final

# ===========================================================================
# Physical constants (ARCHITECTURE.md Section 3.5)
# ===========================================================================
#: Speed of light in vacuum [km/s] (used for light-travel-time correction).
C_KM_S: Final[float] = 299792.458

#: One astronomical unit [km].
AU_KM: Final[float] = 1.495978707e8

#: Sun--Earth L1 standoff distance sunward of Earth [km] (~1.5e6 km).
L1_DISTANCE_KM: Final[float] = 1.5e6

#: Heliocentric distance of Aditya-L1 [AU] = 1 AU - L1 standoff.
#: (1 AU - 1.5e6 km) / 1 AU -> ~0.98997 AU.
ADITYA_L1_R_AU: Final[float] = (AU_KM - L1_DISTANCE_KM) / AU_KM

#: Reference (fusion key) heliocentric distance [AU] -- Earth / GOES frame.
EARTH_R_AU: Final[float] = 1.0

#: Light-travel-time lead of Aditya-L1 over the Earth/GOES frame [s].
#: dt = (r_earth - r_sc) / c. With r_sc ~ 0.990 AU this is ~ +5.0 s, i.e. L1
#: sees flares ~5 s before GOES (ARCHITECTURE.md Section 3.5 table).
ADITYA_L1_LTT_LEAD_S: Final[float] = (AU_KM - ADITYA_L1_R_AU * AU_KM) / C_KM_S

#: J2000.0 epoch as Unix epoch seconds (2000-01-01T12:00:00 TT ~ UTC here).
#: FusionRecord.t_obs_utc is documented "s since J2000" (Section 9.2); this is
#: the offset to convert to/from Unix epoch seconds used elsewhere.
J2000_UNIX_S: Final[float] = 946728000.0


# ===========================================================================
# GOES soft X-ray (1--8 A) flare classification (ARCHITECTURE.md Section 4.5)
# ===========================================================================
# Class = order of magnitude of peak 1--8 A flux [W m^-2]; sub-class is the
# linear mantissa (e.g. M2.5 = 2.5e-5 W m^-2).
#   A < 1e-7
#   B [1e-7, 1e-6)
#   C [1e-6, 1e-5)
#   M [1e-5, 1e-4)
#   X >= 1e-4
GOES_CLASS_A_WM2: Final[float] = 1e-8  # nominal A-class decade floor
GOES_CLASS_B_WM2: Final[float] = 1e-7
GOES_CLASS_C_WM2: Final[float] = 1e-6
GOES_CLASS_M_WM2: Final[float] = 1e-5
GOES_CLASS_X_WM2: Final[float] = 1e-4

#: Lower-bound flux [W m^-2] of each GOES class letter, in ascending order.
#: ``classify_flux`` (flarecast.detect.classify) uses these decade edges.
GOES_CLASS_THRESHOLDS_WM2: Final[dict[str, float]] = {
    "A": GOES_CLASS_A_WM2,
    "B": GOES_CLASS_B_WM2,
    "C": GOES_CLASS_C_WM2,
    "M": GOES_CLASS_M_WM2,
    "X": GOES_CLASS_X_WM2,
}

#: The canonical A/B/C/M/X class ladder in ascending intensity order.
GOES_CLASS_LADDER: Final[tuple[str, ...]] = ("A", "B", "C", "M", "X")


# ===========================================================================
# QC bitmask flag values (ARCHITECTURE.md Section 9.4) -- shared edge + offline
# ===========================================================================
# Mirrors flarecast.types.QCBit; provided here as plain ints for code paths
# that prefer module-level constants (and for the TS/edge side to copy).
QC_GOOD: Final[int] = 1
QC_INTERPOLATED: Final[int] = 2
QC_FILLED: Final[int] = 4
QC_SUSPECT: Final[int] = 8
QC_BAD: Final[int] = 16
QC_NEAR_SAA: Final[int] = 32
QC_SATURATED: Final[int] = 64
QC_DATA_GAP: Final[int] = 128
QC_SPIKE_REJECTED: Final[int] = 256


# ===========================================================================
# Canonical units (ARCHITECTURE.md Section 9.2)
# ===========================================================================
UNIT_SXR: Final[str] = "W m^-2"      # GOES scale for soft X-ray flux
UNIT_HXR: Final[str] = "counts/s"    # reference-band hard X-ray count rate
UNIT_EUV: Final[str] = "W m^-2"
UNIT_RADIO: Final[str] = "sfu"       # per frequency bin
UNIT_PROTON: Final[str] = "pfu"      # per energy channel


# ===========================================================================
# Streaming grid & cadence (ARCHITECTURE.md Section 3.6)
# ===========================================================================
#: Master nowcast grid spacing in t_earth_utc [s] (uniform 1 s).
NOWCAST_GRID_DT_S: Final[float] = 1.0

#: Feature / forecast aggregate cadence [s] (1 min) built by aggregation.
FEATURE_GRID_DT_S: Final[float] = 60.0

#: Maximum gap (as a multiple of cadence) across which numeric interpolation
#: is permitted; never spline impulsive HXR (Section 3.6).
MAX_INTERP_GAP_CADENCES: Final[int] = 3


# ===========================================================================
# Detector defaults (ARCHITECTURE.md Section 4 / Appendix B.4)
# ===========================================================================
# --- CUSUM (soft-band primary onset), Appendix B.4 CUSUMDetector ---
#: Slack k_c = delta / (2 sigma); smallest shift to chase (~0.5 sigma).
CUSUM_K_SLACK: Final[float] = 0.5
#: Decision interval h_c [sigma units] (4--5); ARL0 vs detection-delay knob.
CUSUM_H: Final[float] = 5.0

# --- Adaptive threshold + M-of-N persistence + hysteresis (B.4) ---
#: ON threshold [sigma] (4--5).
ADAPTIVE_K_ON: Final[float] = 5.0
#: OFF threshold [sigma] (1--2); hysteresis to avoid chattering at flare end.
ADAPTIVE_K_OFF: Final[float] = 2.0
#: M-of-N persistence: require M of the last N samples over threshold.
ADAPTIVE_M: Final[int] = 3
ADAPTIVE_N: Final[int] = 4

# --- Poisson hard-band (HEL1OS), Appendix B.4 PoissonCUSUM ---
#: Smallest flare rate of interest as a ratio of background: lam1 = rho*lam0,
#: rho ~ 1.5--2 (research doc 03 Section 4.2).
POISSON_LAM1_RATIO: Final[float] = 1.8
#: Poisson CUSUM decision threshold h.
POISSON_CUSUM_H: Final[float] = 5.0

# --- Width-gate despike (cosmic-ray rejection), Section 11.1 / research 03 ---
#: Minimum sustained width [samples] for an HXR excursion to be a flare and
#: not a one-bin cosmic-ray hit.
SPIKE_WIDTH_MIN_BINS: Final[int] = 3

# --- Hampel despiker, Appendix B.4 HampelDespiker ---
HAMPEL_K: Final[float] = 3.0

# --- GOES-style soft-band FSM start rule (ARCHITECTURE.md Section 4.4) ---
#: Start = first of 4 consecutive minutes of monotonic increase with minute-4
#: flux >= 1.4x minute-1.
FSM_START_CONSEC_MIN: Final[int] = 4
FSM_START_RISE_FACTOR: Final[float] = 1.4

# --- SDD detector arbitration (ARCHITECTURE.md Section 4.5) ---
#: SoLEXS SDD1 saturation ceiling [counts/s]; above this switch to SDD2.
SDD_SATURATION_CPS: Final[float] = 1e5

# --- EMA baseline (research doc 03 Section 1.1) ---
#: Default EMA forgetting factor; effective window ~ 1/(1-alpha).
DEFAULT_EMA_ALPHA: Final[float] = 0.99

# --- Kalman fusion innovation gate (ARCHITECTURE.md Section 3.8) ---
#: chi^2_{1, 0.999} ~ 10.8 innovation gate for automatic outlier rejection.
KALMAN_GATE_CHI2: Final[float] = 10.8

# --- Neupert leaky integrator (ARCHITECTURE.md Section 1.2) ---
#: SXR cooling timescale tau_cool [s] in dF_SXR/dt = c*F_HXR - F_SXR/tau_cool.
NEUPERT_TAU_COOL_S: Final[float] = 240.0


# ===========================================================================
# Master-catalogue association & indexing (ARCHITECTURE.md Section 4.6 / 4.7)
# ===========================================================================
#: Asymmetric association window, hard-before-soft (Neupert lead) [s].
ASSOC_W_LEAD_S: Final[float] = 300.0    # ~5 min
#: Asymmetric association window, long soft decay (lag) [s].
ASSOC_W_LAG_S: Final[float] = 900.0     # ~10-15 min
#: MATCH_SCORE acceptance threshold (Appendix B.5 Associator.tau_match).
ASSOC_TAU_MATCH: Final[float] = 0.5
#: De-duplication guard time [s] (Appendix B.5 deduplicate.guard_s).
DEDUP_GUARD_S: Final[float] = 90.0
#: Hash-bucket index width [s] for O(1) by-time lookup (Section 4.7 / B.5).
CATALOG_BUCKET_S: Final[float] = 3600.0


# ===========================================================================
# Consensus labeling (ARCHITECTURE.md Section 3.9)
# ===========================================================================
#: conf >= this -> "confirmed".
CONSENSUS_CONFIRM_THRESH: Final[float] = 0.7
#: conf >= this (and < confirm) -> "candidate"; below -> "rejected".
CONSENSUS_CANDIDATE_THRESH: Final[float] = 0.3
#: Temporal association tolerance for consensus voting (+/- peak) [s].
CONSENSUS_PEAK_TOL_S: Final[float] = 180.0
#: Spatial association tolerance for consensus voting [deg].
CONSENSUS_LOC_TOL_DEG: Final[float] = 10.0


# ===========================================================================
# Forecast defaults (ARCHITECTURE.md Section 5 / Appendix B.6)
# ===========================================================================
#: Standard forecast horizons N [minutes] (binary "flare >= class in next N").
FORECAST_HORIZONS_MIN: Final[tuple[int, ...]] = (15, 30, 60)
#: Multi-horizon probability-curve horizons h [minutes] (deliverable framing).
FORECAST_P_CURVE_HORIZONS_MIN: Final[tuple[int, ...]] = (5, 15, 30, 60, 120)
#: Default operational class threshold for the positive label.
FORECAST_DEFAULT_CLASS_THRESHOLD: Final[str] = "C"
#: Dimensionality of the streaming feature vector (Section 5.2).
N_FEATURES: Final[int] = 30
#: TCN raw-input look-back length L [steps] and channel count C (Section 5.2).
TCN_LOOKBACK_STEPS: Final[int] = 120
TCN_N_CHANNELS: Final[int] = 8
#: Walk-forward CV embargo [minutes] (Appendix B.6 blocked_splits default).
CV_EMBARGO_MIN: Final[float] = 120.0
#: Hawkes / decayed-history timescale tau [s] (~6 h) for flare history feature.
HISTORY_DECAY_TAU_S: Final[float] = 6 * 3600.0
#: Lead-time evaluation window before peak [s] (Appendix B.6 lead_time.w_min).
LEADTIME_WINDOW_S: Final[float] = 7200.0
#: Lead-time anti-flicker k-of-m crossing requirement (Appendix B.6).
LEADTIME_K_OF_M: Final[tuple[int, int]] = (2, 3)
#: Lead-time reporting buckets [minutes] (fraction with LT >= each).
LEADTIME_REPORT_BUCKETS_MIN: Final[tuple[int, ...]] = (5, 10, 15, 30)
#: Default synthetic flare class mix (Appendix B.2 generate_flare_lightcurves).
DEFAULT_CLASS_MIX: Final[dict[str, float]] = {"C": 0.7, "M": 0.25, "X": 0.05}


# ===========================================================================
# Data access (ARCHITECTURE.md Section 6)
# ===========================================================================
#: NOAA SWPC real-time GOES XRS JSON (live nowcast anchor, no auth).
SWPC_XRAYS_URL: Final[str] = (
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
)
#: NOAA SWPC GOES X-ray flare event list.
SWPC_FLARES_URL: Final[str] = (
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
)
#: Default network timeout for live fetchers [s] before offline fallback.
DEFAULT_FETCH_TIMEOUT_S: Final[float] = 10.0
