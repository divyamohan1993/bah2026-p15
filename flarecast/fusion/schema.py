"""Unified multi-source fusion record + canonical-unit conversion.

This module implements the long/tidy ``FusionRecord`` from
ARCHITECTURE.md Section 9.2 and research doc ``06 Section 1.2``, plus the
``to_canonical_unit`` helper (Appendix B.3). One ``FusionRecord`` row is one
sample of one *quantity* from one *source*; the record is **append-only and
lossless** -- the raw ``value_native``/``unit`` are never overwritten, every
correction (LTT, cross-cal, QC, fill) is an additional column.

It is **pure standard library** (no numpy / pandas) so it always imports,
including offline, exactly like :mod:`flarecast.types` and
:mod:`flarecast.constants` which it builds on. Canonical units are taken
verbatim from :mod:`flarecast.constants` (``UNIT_*``) so a single value
governs the contract everywhere.

The fusion key is :attr:`FusionRecord.t_earth_utc` -- the LTT-corrected time
the photons would have reached the common ~1 AU Earth/L1 reference frame (see
:mod:`flarecast.fusion.ltt`). Fusing on the raw ``t_obs_utc`` would smear and
misalign peaks by up to minutes (Solar Orbiter ~240 s) and is never done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import (
    UNIT_EUV,
    UNIT_HXR,
    UNIT_PROTON,
    UNIT_RADIO,
    UNIT_SXR,
)
from ..types import QCFlag, Quantity

__all__ = [
    "FusionRecord",
    "CANONICAL_UNITS",
    "canonical_unit_for",
    "to_canonical_unit",
]


# ---------------------------------------------------------------------------
# Canonical units per quantity (ARCHITECTURE.md Section 9.2 / research 06 1.3)
# ---------------------------------------------------------------------------
#: Map each :class:`~flarecast.types.Quantity` to its canonical unit token.
#: Sourced from the frozen ``UNIT_*`` constants so the fusion layer and the
#: rest of the package never drift. MAGSCALAR is a derived feature (not fused
#: as a redundant measurement) and carries a dimensionless token.
CANONICAL_UNITS: dict[str, str] = {
    Quantity.SXR_LONG.value: UNIT_SXR,     # "W m^-2" on the GOES scale
    Quantity.SXR_SHORT.value: UNIT_SXR,
    Quantity.HXR.value: UNIT_HXR,          # "counts/s" in a reference band
    Quantity.EUV.value: UNIT_EUV,          # "W m^-2"
    Quantity.RADIO.value: UNIT_RADIO,      # "sfu" per frequency bin
    Quantity.PROTON.value: UNIT_PROTON,    # "pfu" per energy channel
    Quantity.MAGSCALAR.value: "1",         # derived scalar (feature, not fused)
}


# ---------------------------------------------------------------------------
# Unit-conversion table -> canonical
# ---------------------------------------------------------------------------
# Multiplicative factors to bring a recognized native unit onto the canonical
# unit for its quantity. The conversions are deliberately conservative: only
# obvious, unambiguous unit aliases / SI prefixes are converted. Anything not
# recognized is returned unchanged (the cross-calibration stage, not unit
# conversion, is responsible for instrument scale -- research 06 Section 3).
_SXR_FACTORS: dict[str, float] = {
    "w m^-2": 1.0,
    "w/m^2": 1.0,
    "w m-2": 1.0,
    "w m^2": 1.0,            # tolerate the common typo for W m^-2
    "watt/m^2": 1.0,
    "mw m^-2": 1e-3,        # milliwatt
    "uw m^-2": 1e-6,        # microwatt
    "erg s^-1 cm^-2": 1e-3,  # 1 erg/s/cm^2 = 1e-3 W/m^2
    "erg/s/cm^2": 1e-3,
}
_HXR_FACTORS: dict[str, float] = {
    "counts/s": 1.0,
    "counts s^-1": 1.0,
    "count/s": 1.0,
    "cps": 1.0,
    "cts/s": 1.0,
    "kcps": 1e3,
    "counts/min": 1.0 / 60.0,
}
_EUV_FACTORS: dict[str, float] = {
    "w m^-2": 1.0,
    "w/m^2": 1.0,
    "mw m^-2": 1e-3,
    "uw m^-2": 1e-6,
}
_RADIO_FACTORS: dict[str, float] = {
    "sfu": 1.0,
    "solar flux unit": 1.0,
    "jy": 1e-4,            # 1 sfu = 1e4 Jy -> 1 Jy = 1e-4 sfu
}
_PROTON_FACTORS: dict[str, float] = {
    "pfu": 1.0,
    "particles cm^-2 s^-1 sr^-1": 1.0,
    "cm^-2 s^-1 sr^-1": 1.0,
}

_FACTOR_TABLES: dict[str, dict[str, float]] = {
    Quantity.SXR_LONG.value: _SXR_FACTORS,
    Quantity.SXR_SHORT.value: _SXR_FACTORS,
    Quantity.HXR.value: _HXR_FACTORS,
    Quantity.EUV.value: _EUV_FACTORS,
    Quantity.RADIO.value: _RADIO_FACTORS,
    Quantity.PROTON.value: _PROTON_FACTORS,
}


def canonical_unit_for(quantity: str) -> str:
    """Return the canonical unit token for ``quantity``.

    ``quantity`` may be a :class:`~flarecast.types.Quantity` member or its
    string value. Unknown quantities return the dimensionless token ``"1"``.
    """
    key = quantity.value if isinstance(quantity, Quantity) else str(quantity)
    return CANONICAL_UNITS.get(key, "1")


def to_canonical_unit(value: float, unit: str, quantity: str) -> tuple[float, str]:
    """Convert ``value`` in ``unit`` to the canonical unit for ``quantity``.

    Implements Appendix B.3 ``schema.to_canonical_unit``. Returns
    ``(value_canonical, canonical_unit_token)``.

    Only unambiguous unit aliases / SI-prefix rescalings are applied (see the
    module factor tables). If ``unit`` is already canonical, or is not a
    recognized alias, the value is returned **unchanged** with the canonical
    token -- residual instrument-scale differences are removed later by the
    cross-calibration transfer functions (research 06 Section 3), not here.

    Parameters
    ----------
    value:
        Measurement in ``unit``.
    unit:
        Native unit token (case/spacing-insensitive).
    quantity:
        A :class:`~flarecast.types.Quantity` member or its string value.
    """
    canon = canonical_unit_for(quantity)
    key = quantity.value if isinstance(quantity, Quantity) else str(quantity)
    table = _FACTOR_TABLES.get(key)
    if table is None:
        return float(value), canon
    norm = unit.strip().lower().replace(" ", " ") if unit else ""
    norm = " ".join(norm.split())  # collapse internal whitespace
    factor = table.get(norm)
    if factor is None:
        # Unrecognized -> assume already on canonical scale (lossless: native
        # value is preserved separately on the FusionRecord).
        return float(value), canon
    return float(value) * factor, canon


# ---------------------------------------------------------------------------
# The canonical fusion record (long/tidy form) -- ARCHITECTURE.md Section 9.2
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FusionRecord:
    """One sample of one quantity from one source (ARCHITECTURE.md 9.2).

    Field names, order, and defaults mirror the normative dataclass in
    ARCHITECTURE.md Section 9.2 and research doc ``06 Section 1.2``. The record
    is append-only and lossless: ``value_native`` / ``unit`` retain the raw
    instrument reading for audit, while ``value`` carries the canonical-unit
    (and, after :mod:`~flarecast.fusion.xcal`, cross-calibrated) measurement.

    The required, no-default fields come first (raw timestamp, identity,
    quantity, value); everything that the fusion pipeline *fills in* as it runs
    (the LTT-corrected fusion key, native audit columns, uncertainty, vantage
    geometry, QC, provenance, reproducibility) has a sensible default so a
    fresh record can be built from a minimal ingest and progressively enriched.

    Attributes
    ----------
    t_obs_utc:
        Raw on-board timestamp converted to UTC (TAI-aware), seconds. The
        contract documents this as "seconds since J2000"; the
        :data:`flarecast.constants.J2000_UNIX_S` offset converts to/from Unix
        epoch seconds used elsewhere. The math here is offset-agnostic (LTT and
        fusion only ever *difference* times).
    t_earth_utc:
        LTT-corrected time in the common Earth/L1 (~1 AU) frame -- **the fusion
        key**. Set by :func:`flarecast.fusion.ltt.light_travel_correction`.
    source_id:
        Stream identifier, e.g. ``"ADITYA_SOLEXS"``, ``"GOES18_XRSB"``,
        ``"SOLO_STIX_25-50keV"``.
    platform:
        Spacecraft / platform name, e.g. ``"Aditya-L1"``, ``"GOES-18"``.
    quantity:
        One of the :class:`~flarecast.types.Quantity` string values.
    value:
        Measurement in the canonical unit for ``quantity`` (GOES W m^-2 for
        SXR; reference counts/s for HXR), after cross-calibration.
    band:
        Energy / wavelength band token, e.g. ``"1-8A"``, ``"25-50keV"``.
    value_native:
        Original instrument-unit value (lossless audit).
    unit:
        Canonical unit token for ``value``.
    sigma:
        1-sigma uncertainty in the canonical unit -- **the fusion weight
        source**.
    cadence_s:
        Nominal sample spacing of this stream (1, 3, 60, 720, ...).
    vantage_r_au:
        Heliocentric distance of the spacecraft [AU] (for LTT).
    vantage_xyz_hci:
        Spacecraft position, Heliocentric Inertial [km] (for IPN geometry).
    sub_sc_lon_deg:
        Sub-spacecraft Carrington/Stonyhurst longitude [deg] (for stereoscopy).
    qc_flag:
        Human-readable QC state (:class:`~flarecast.types.QCFlag` value).
    qc_bitmask:
        Integer coexisting-conditions bitmask (:class:`~flarecast.types.QCBit`).
    provenance:
        Provenance string, e.g. ``"measured"``, ``"filled:GOES18"``,
        ``"interp:linear"``, ``"calibrated:v2.1"``.
    src_weight:
        Static reliability weight r_i for this source/quantity (registry).
    cal_version:
        Calibration / transfer-function version applied.
    ingest_hash:
        Content hash of the raw record (reproducibility).
    """

    # --- required identity / measurement ---
    t_obs_utc: float
    source_id: str
    platform: str
    quantity: str
    value: float

    # --- fusion key (filled by LTT stage) ---
    t_earth_utc: float = 0.0

    # --- lossless audit + uncertainty ---
    band: str = ""
    value_native: float = 0.0
    unit: str = ""
    sigma: float = 0.0

    # --- cadence + vantage geometry ---
    cadence_s: float = 1.0
    vantage_r_au: float = 1.0
    vantage_xyz_hci: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sub_sc_lon_deg: float = 0.0

    # --- QC / provenance / reproducibility ---
    qc_flag: str = QCFlag.GOOD.value
    qc_bitmask: int = 0
    provenance: str = "measured"
    src_weight: float = 1.0
    cal_version: str = ""
    ingest_hash: str = ""

    # Optional free-form metadata (energy sub-band, detector id, n averaged...).
    meta: dict | None = field(default=None)

    def __post_init__(self) -> None:
        # Default the lossless native columns from the canonical value/unit on
        # first construction (an honest no-op when nothing was supplied) so a
        # minimally-built record is still self-consistent and auditable.
        if not self.unit:
            self.unit = canonical_unit_for(self.quantity)
        if self.value_native == 0.0 and self.value != 0.0:
            self.value_native = self.value
