"""Per-source reliability / cadence / latency / failure-mode registry.

Implements Appendix B.3 ``registry`` (``SourceInfo`` + ``SourceRegistry``) and
research doc ``06 Section 7.2``. The registry is the table that makes fusion
**source-agnostic**: new platforms are added with no code changes. Each entry
holds, per (source, quantity):

* ``reliability`` -- static r_i in (0, 1] from historical agreement with the
  consensus catalogue (research 06 Section 7.2); used as the inverse-variance
  weight multiplier ``w_i = r_i / sigma_i^2`` (Section 4.3.1).
* ``cadence_s`` -- nominal sample spacing (1 s, 60 s, 720 s, ...).
* ``latency_budget_s`` -- how stale this source may be and still feed a
  *real-time nowcast* without look-ahead leakage. Aditya-L1 arrives ~1 day
  late, so it is a label/training source, not a live nowcast feature
  (research 06 Section 2.4 -- the no-look-ahead guard).
* ``typical_sigma`` -- representative 1-sigma (canonical unit) used as a prior
  when a sample lacks a measured uncertainty.
* ``failure_modes`` -- known hazards (SAA windows, attenuator states,
  saturation, pile-up, data gaps) for QC and operator awareness.

Reliability weights are seeded from literature agreement and are intended to be
*adaptively decayed* when a source is repeatedly gated by the Kalman innovation
test (Section 4.3.2) and restored when it agrees again -- :class:`SourceRegistry`
exposes :meth:`decay_reliability` / :meth:`restore_reliability` for that loop.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..constants import (
    FEATURE_GRID_DT_S,
    NOWCAST_GRID_DT_S,
)
from ..types import Quantity

__all__ = [
    "SourceInfo",
    "SourceRegistry",
    "default_registry",
    "DEFAULT_SOURCES",
]

# One sidereal day in seconds -- the ~1-day Aditya-L1 / archive science latency.
_ONE_DAY_S = 86_400.0


@dataclass(slots=True)
class SourceInfo:
    """Static registry entry for one (source, quantity).

    Field names and order follow Appendix B.3 ``registry.SourceInfo``. Extra
    descriptive fields (``platform``, ``band``, ``vantage_r_au``) carry the
    geometry/identity the LTT and stereoscopy stages need; they default so a
    minimal ``SourceInfo(source_id, quantity, reliability, cadence_s,
    latency_budget_s, typical_sigma, failure_modes)`` matches the contract.

    Attributes
    ----------
    source_id:
        Stream identifier, e.g. ``"ADITYA_SOLEXS"``, ``"GOES_PRIMARY_XRSB"``.
    quantity:
        :class:`~flarecast.types.Quantity` string value this source estimates.
    reliability:
        Static reliability weight r_i in (0, 1].
    cadence_s:
        Nominal sample spacing [s].
    latency_budget_s:
        Max staleness [s] for real-time nowcast eligibility (no-look-ahead).
    typical_sigma:
        Representative 1-sigma in the canonical unit.
    failure_modes:
        List of known hazard tokens.
    platform:
        Spacecraft / platform name (for LTT named-platform lookup).
    band:
        Energy / wavelength band token.
    vantage_r_au:
        Representative heliocentric distance [AU] (LTT default).
    """

    source_id: str
    quantity: str
    reliability: float
    cadence_s: float
    latency_budget_s: float
    typical_sigma: float
    failure_modes: list[str] = field(default_factory=list)
    platform: str = ""
    band: str = ""
    vantage_r_au: float = 1.0

    def is_realtime_eligible(self, age_s: float) -> bool:
        """``True`` if a sample this stale may feed the real-time nowcast.

        Enforces the per-source ``latency_budget`` no-look-ahead guard
        (research 06 Section 2.4): slower-arriving sources (Aditya-L1 at ~1 day)
        are usable for post-hoc labels/training but never as live features.
        """
        return age_s <= self.latency_budget_s


class SourceRegistry:
    """Mutable table of :class:`SourceInfo` keyed by ``source_id``.

    Implements Appendix B.3 ``registry.SourceRegistry`` (``get`` / ``register``)
    plus the adaptive-reliability loop hooks from research 06 Section 7.2.
    """

    def __init__(self, sources: list[SourceInfo] | None = None) -> None:
        self._by_id: dict[str, SourceInfo] = {}
        for info in sources or []:
            self.register(info)

    # --- contract surface -------------------------------------------------
    def register(self, info: SourceInfo) -> None:
        """Add or replace a source entry."""
        self._by_id[info.source_id] = info

    def get(self, source_id: str) -> SourceInfo:
        """Return the entry for ``source_id`` (raises :class:`KeyError`)."""
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            raise KeyError(
                f"source_id {source_id!r} not in registry; register a "
                f"SourceInfo first (known: {sorted(self._by_id)})"
            ) from exc

    # --- convenience ------------------------------------------------------
    def has(self, source_id: str) -> bool:
        """``True`` if ``source_id`` is registered."""
        return source_id in self._by_id

    def get_or_none(self, source_id: str) -> SourceInfo | None:
        """Return the entry for ``source_id`` or ``None`` if absent."""
        return self._by_id.get(source_id)

    def sources_for_quantity(self, quantity: str) -> list[SourceInfo]:
        """All registered sources estimating ``quantity``."""
        q = quantity.value if isinstance(quantity, Quantity) else str(quantity)
        return [s for s in self._by_id.values() if s.quantity == q]

    def all(self) -> list[SourceInfo]:
        """All registered entries (insertion order)."""
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._by_id

    # --- adaptive reliability (research 06 Section 7.2) -------------------
    def decay_reliability(self, source_id: str, factor: float = 0.9,
                          floor: float = 0.05) -> float:
        """Multiplicatively decay a source's reliability after gating.

        Called when the Kalman innovation gate (Section 4.3.2) repeatedly
        rejects this source -- the system *learns* to trust it less. Clamped at
        ``floor`` so a source is never zeroed out (it can recover). Returns the
        new reliability.
        """
        info = self.get(source_id)
        new_r = max(floor, info.reliability * factor)
        self._by_id[source_id] = replace(info, reliability=new_r)
        return new_r

    def restore_reliability(self, source_id: str, rate: float = 0.05,
                            ceiling: float = 1.0) -> float:
        """Nudge a source's reliability back up after it agrees again.

        Inverse of :meth:`decay_reliability`; restores trust gradually toward
        ``ceiling``. Returns the new reliability.
        """
        info = self.get(source_id)
        new_r = min(ceiling, info.reliability + rate * (ceiling - info.reliability))
        self._by_id[source_id] = replace(info, reliability=new_r)
        return new_r


# ---------------------------------------------------------------------------
# Seed table -- the core fusion constellation (census 02 / fusion 06 Section 7.2)
# ---------------------------------------------------------------------------
# Reliabilities are literature-agreement priors (GOES high for SXR class/time;
# STIX/GBM high for HXR; Aditya primary but ~1-day latency). Latency budgets
# distinguish live nowcast feeds (GOES, STIX QL, GBM ~ minutes) from the ~1-day
# Aditya-L1 / archive science streams (research 06 Section 2.4). typical_sigma
# is in the canonical unit (W m^-2 for SXR, counts/s for HXR).
DEFAULT_SOURCES: list[SourceInfo] = [
    # --- Soft X-ray flux (W m^-2 on the GOES scale) ---
    SourceInfo(
        source_id="ADITYA_SOLEXS",
        quantity=Quantity.SXR_LONG.value,
        reliability=0.90,
        cadence_s=NOWCAST_GRID_DT_S,            # 1 s during flares
        latency_budget_s=_ONE_DAY_S,            # ~1-day PRADAN latency
        typical_sigma=5e-8,
        failure_modes=[
            "saa", "eclipse", "off_point", "sdd1_saturation", "telemetry_gap",
        ],
        platform="Aditya-L1",
        band="1-8A",
        vantage_r_au=0.990,
    ),
    SourceInfo(
        source_id="GOES_PRIMARY_XRSB",
        quantity=Quantity.SXR_LONG.value,
        reliability=0.98,                       # canonical A-X class anchor
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=180.0,                 # live SWPC JSON ~ minutes
        typical_sigma=2e-8,
        failure_modes=["saturation_xclass", "spacecraft_eclipse"],
        platform="GOES-19",
        band="1-8A",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="CHANDRAYAAN2_XSM",
        quantity=Quantity.SXR_LONG.value,
        reliability=0.85,                       # lunar-vantage soft anchor
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=6e-8,
        failure_modes=["lunar_eclipse", "off_point", "saa"],
        platform="Chandrayaan-2",
        band="1-15keV",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="GOES_PRIMARY_XRSA",
        quantity=Quantity.SXR_SHORT.value,
        reliability=0.96,
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=180.0,
        typical_sigma=2e-9,
        failure_modes=["saturation_xclass"],
        platform="GOES-19",
        band="0.5-4A",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="ADITYA_SOLEXS_SHORT",
        quantity=Quantity.SXR_SHORT.value,
        reliability=0.88,
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=5e-9,
        failure_modes=["saa", "eclipse", "sdd1_saturation"],
        platform="Aditya-L1",
        band="0.5-4A",
        vantage_r_au=0.990,
    ),
    # --- Hard X-ray (counts/s in the 25-50 keV reference band) ---
    SourceInfo(
        source_id="ADITYA_HEL1OS",
        quantity=Quantity.HXR.value,
        reliability=0.85,
        cadence_s=NOWCAST_GRID_DT_S,            # 1 s light curves
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=20.0,
        failure_modes=["saa", "cosmic_ray_spikes", "pile_up", "sep_storm"],
        platform="Aditya-L1",
        band="8-30keV",
        vantage_r_au=0.990,
    ),
    SourceInfo(
        source_id="SOLO_STIX_25-50keV",
        quantity=Quantity.HXR.value,
        reliability=0.95,                       # imaging HXR reference
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=3600.0,                # STIX quicklook ~ minutes-hour
        typical_sigma=15.0,
        failure_modes=["attenuator_state", "data_gap", "eccentric_ltt"],
        platform="Solar Orbiter",
        band="25-50keV",
        vantage_r_au=0.50,
    ),
    SourceInfo(
        source_id="FERMI_GBM",
        quantity=Quantity.HXR.value,
        reliability=0.90,                       # all-sky high-cadence anchor
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=3600.0,
        typical_sigma=25.0,
        failure_modes=["saturation_largest_flares", "saa", "data_gap", "occultation"],
        platform="Fermi",
        band="8-40000keV",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="KONUS_WIND",
        quantity=Quantity.HXR.value,
        reliability=0.88,                       # L1 IPN node, total-sky
        cadence_s=3.0,                          # 3 s waiting mode
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=30.0,
        failure_modes=["trigger_only", "data_gap"],
        platform="Wind",
        band="20-1500keV",
        vantage_r_au=1.0,
    ),
    # --- EUV irradiance (W m^-2) ---
    SourceInfo(
        source_id="SDO_EVE",
        quantity=Quantity.EUV.value,
        reliability=0.90,
        cadence_s=10.0,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=1e-5,
        failure_modes=["eclipse_season", "calibration_drift"],
        platform="SDO",
        band="6-106nm",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="STEREOA_EUVI",
        quantity=Quantity.EUV.value,
        reliability=0.80,                       # far-side / different longitude
        cadence_s=180.0,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=2e-5,
        failure_modes=["downlink_cadence", "data_gap"],
        platform="STEREO-A",
        band="17.1-30.4nm",
        vantage_r_au=0.96,
    ),
    # --- Radio dynamic spectra (sfu) ---
    SourceInfo(
        source_id="WIND_WAVES",
        quantity=Quantity.RADIO.value,
        reliability=0.75,
        cadence_s=NOWCAST_GRID_DT_S,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=10.0,
        failure_modes=["rfi", "data_gap"],
        platform="Wind",
        band="4kHz-14MHz",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="ECALLISTO",
        quantity=Quantity.RADIO.value,
        reliability=0.65,                       # ground network, RFI-prone
        cadence_s=0.25,
        latency_budget_s=_ONE_DAY_S,
        typical_sigma=15.0,
        failure_modes=["rfi", "night_gap", "station_dropout"],
        platform="e-CALLISTO",
        band="45-870MHz",
        vantage_r_au=1.0,
    ),
    # --- In-situ protons (pfu) -- own clock, never LTT-corrected ---
    SourceInfo(
        source_id="GOES_SEISS",
        quantity=Quantity.PROTON.value,
        reliability=0.92,                       # operational S-scale definition
        cadence_s=FEATURE_GRID_DT_S,            # ~ 1 min
        latency_budget_s=180.0,
        typical_sigma=0.1,
        failure_modes=["instrument_saturation"],
        platform="GOES-19",
        band=">10MeV",
        vantage_r_au=1.0,
    ),
    SourceInfo(
        source_id="ACE_EPAM",
        quantity=Quantity.PROTON.value,
        reliability=0.85,
        cadence_s=FEATURE_GRID_DT_S,
        latency_budget_s=180.0,
        typical_sigma=0.2,
        failure_modes=["degraded_detector", "data_gap"],
        platform="ACE",
        band="0.05-5MeV",
        vantage_r_au=1.0,
    ),
]


def default_registry() -> SourceRegistry:
    """Build a :class:`SourceRegistry` seeded with the core constellation.

    Returns a fresh, independently-mutable registry each call (so adaptive
    reliability updates in one pipeline run never leak into another).
    """
    return SourceRegistry([replace(s) for s in DEFAULT_SOURCES])
