"""Consensus labeling: weighted voting across catalogues (Section 3.9 / 06 §6).

Single catalogues disagree (GOES vs RHESSI held 25,691 vs 121,430 events over
2002-2017). Trustworthy supervised labels are built by **voting** across
catalogues (research doc ``06 Section 6``):

1. LTT-correct every catalogue's times to ``t_earth_utc`` (so one physical
   flare's entries actually overlap; :mod:`flarecast.fusion.ltt`);
2. **associate** entries by temporal overlap (peak within ``peak_tol_s``) and,
   when available, source location (within ``loc_tol_deg``) into candidate
   physical events;
3. score ``conf(e) = sum(rho_j v_j) / sum(rho_j)`` over catalogue reliabilities
   ``rho_j``; label **confirmed** if ``conf >= tau_hi``, **candidate** if
   ``>= tau_lo``, else **rejected**.

Conflict rules: an HXR-present / SXR-absent event is kept **confirmed** with an
``sxr_weak`` flag (a valuable forecast signal); a single-catalogue event is a
**candidate**, promotable only if our own fused detection or IPN/stereoscopy
corroborates. Thresholds come from ``CONSENSUS_*`` constants.

Pure standard library; catalogue entries are plain dicts and a "DF" is a list of
:class:`ConsensusLabel` (the offline reference container).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..constants import (
    CONSENSUS_CANDIDATE_THRESH,
    CONSENSUS_CONFIRM_THRESH,
    CONSENSUS_LOC_TOL_DEG,
    CONSENSUS_PEAK_TOL_S,
)
from .ltt import light_travel_correction

__all__ = [
    "CatalogueEntry",
    "ConsensusLabel",
    "consensus_label",
    "associate_entries",
]


@dataclass(slots=True)
class CatalogueEntry:
    """One flare entry from one catalogue (a "voter"'s ballot).

    Attributes
    ----------
    catalog:
        Catalogue name (``"GOES"``, ``"HEK"``, ``"RHESSI"``, ``"STIX"``,
        ``"GBM"``, ...) -- selects the reliability ``rho_j``.
    t_peak:
        Peak time [s]. If ``vantage_r_au`` differs from the reference it is
        LTT-corrected to ``t_earth_utc`` before association.
    event_id:
        Catalogue-native event id (provenance).
    goes_class:
        Reported GOES class (``"M2.5"``) if any.
    lon_deg / lat_deg:
        Source location [deg] if known (``None`` if not).
    is_hxr / is_sxr:
        Whether this catalogue reports the event in hard / soft X-rays.
    vantage_r_au:
        Heliocentric distance of the reporting platform [AU] (for LTT).
    """

    catalog: str
    t_peak: float
    event_id: str = ""
    goes_class: str | None = None
    lon_deg: float | None = None
    lat_deg: float | None = None
    is_hxr: bool = False
    is_sxr: bool = True
    vantage_r_au: float = 1.0


@dataclass(slots=True)
class ConsensusLabel:
    """A consensus master-catalogue label for one physical event.

    Attributes
    ----------
    t_peak:
        Consensus peak time [s] in the Earth/L1 frame (mean of voters).
    confidence:
        ``conf(e)`` in [0, 1].
    state:
        ``"confirmed"`` | ``"candidate"`` | ``"rejected"``.
    goes_class:
        Reconciled GOES class (GOES catalogue preferred).
    voters:
        Catalogue names that voted *present* for this event.
    n_catalogues:
        Total number of distinct catalogues contributing an entry.
    sxr_weak:
        ``True`` if HXR-present but SXR-absent (non-thermal-dominated flare).
    lon_deg / lat_deg:
        Consensus source location if any voter supplied one.
    members:
        The contributing :class:`CatalogueEntry` objects (provenance).
    """

    t_peak: float
    confidence: float
    state: str
    goes_class: str | None = None
    voters: list[str] = field(default_factory=list)
    n_catalogues: int = 0
    sxr_weak: bool = False
    lon_deg: float | None = None
    lat_deg: float | None = None
    members: list[CatalogueEntry] = field(default_factory=list)


def _to_entries(catalogue) -> list[CatalogueEntry]:
    """Coerce a catalogue (list of dicts or CatalogueEntry) to entries."""
    out: list[CatalogueEntry] = []
    for row in catalogue:
        if isinstance(row, CatalogueEntry):
            out.append(row)
        elif isinstance(row, Mapping):
            out.append(
                CatalogueEntry(
                    catalog=row.get("catalog", row.get("catalogue", "")),
                    t_peak=float(row["t_peak"]),
                    event_id=str(row.get("event_id", "")),
                    goes_class=row.get("goes_class"),
                    lon_deg=row.get("lon_deg"),
                    lat_deg=row.get("lat_deg"),
                    is_hxr=bool(row.get("is_hxr", False)),
                    is_sxr=bool(row.get("is_sxr", True)),
                    vantage_r_au=float(row.get("vantage_r_au", 1.0)),
                )
            )
        else:
            raise TypeError(
                f"consensus: catalogue rows must be dict or CatalogueEntry, "
                f"got {type(row)!r}"
            )
    return out


def associate_entries(
    catalogues: Sequence[Sequence],
    peak_tol_s: float = CONSENSUS_PEAK_TOL_S,
    loc_tol_deg: float = CONSENSUS_LOC_TOL_DEG,
) -> list[list[CatalogueEntry]]:
    """Group LTT-corrected catalogue entries into candidate physical events.

    Greedy single-linkage association by peak time (within ``peak_tol_s``) and,
    when both have a location, by angular proximity (within ``loc_tol_deg``).
    Times are first LTT-corrected to the Earth/L1 frame so a single physical
    flare's entries from different vantages actually overlap (research 06
    Section 6.2). Returns a list of clusters (each a list of entries).
    """
    # Flatten + LTT-correct.
    entries: list[CatalogueEntry] = []
    for cat in catalogues:
        for e in _to_entries(cat):
            te = light_travel_correction(e.t_peak, e.vantage_r_au)
            entries.append(
                CatalogueEntry(
                    catalog=e.catalog,
                    t_peak=te,
                    event_id=e.event_id,
                    goes_class=e.goes_class,
                    lon_deg=e.lon_deg,
                    lat_deg=e.lat_deg,
                    is_hxr=e.is_hxr,
                    is_sxr=e.is_sxr,
                    vantage_r_au=e.vantage_r_au,
                )
            )
    entries.sort(key=lambda e: e.t_peak)

    clusters: list[list[CatalogueEntry]] = []
    for e in entries:
        placed = False
        for cluster in clusters:
            # Compare against the cluster's current mean peak time.
            mean_t = sum(c.t_peak for c in cluster) / len(cluster)
            if abs(e.t_peak - mean_t) > peak_tol_s:
                continue
            # Location proximity (only if both have a location).
            if e.lon_deg is not None and e.lat_deg is not None:
                loc_ok = True
                for c in cluster:
                    if c.lon_deg is not None and c.lat_deg is not None:
                        if _ang_sep(
                            e.lon_deg, e.lat_deg, c.lon_deg, c.lat_deg
                        ) > loc_tol_deg:
                            loc_ok = False
                            break
                if not loc_ok:
                    continue
            cluster.append(e)
            placed = True
            break
        if not placed:
            clusters.append([e])
    return clusters


def consensus_label(
    catalogues: Sequence[Sequence],
    reliabilities: Mapping[str, float],
    tau_hi: float = CONSENSUS_CONFIRM_THRESH,
    tau_lo: float = CONSENSUS_CANDIDATE_THRESH,
    peak_tol_s: float = CONSENSUS_PEAK_TOL_S,
    *,
    loc_tol_deg: float = CONSENSUS_LOC_TOL_DEG,
    own_detection_times: Sequence[float] | None = None,
) -> list[ConsensusLabel]:
    """Weighted voting across catalogues -> master labels (Appendix B.3).

    Implements ``consensus.consensus_label``. For each associated event,
    ``conf = sum_{j in voters} rho_j / sum_all rho_j`` where the denominator is
    the total reliability of the catalogues that *could* have voted (those
    present in ``reliabilities``). Labels:

    * ``conf >= tau_hi`` -> **confirmed**;
    * ``tau_lo <= conf < tau_hi`` -> **candidate** (promoted to confirmed if an
      ``own_detection_times`` entry corroborates within ``peak_tol_s``);
    * else -> **rejected**.

    HXR-present / SXR-absent events are kept **confirmed** with ``sxr_weak=True``
    (research 06 Section 6.3). GOES class is preferred for the reconciled class.

    Parameters
    ----------
    catalogues:
        Sequence of catalogues; each a list of dicts or :class:`CatalogueEntry`.
    reliabilities:
        Map ``catalog_name -> rho_j`` in (0, 1].
    tau_hi / tau_lo:
        Confirm / candidate thresholds (default ``CONSENSUS_*``).
    peak_tol_s / loc_tol_deg:
        Association tolerances.
    own_detection_times:
        Optional peak times [s] of our own fused detections, used to promote a
        candidate to confirmed (research 06 Section 6.3).

    Returns
    -------
    list[ConsensusLabel]
        One label per associated physical event, sorted by peak time.
    """
    total_rho = sum(reliabilities.values())
    if total_rho <= 0.0:
        raise ValueError("consensus_label: reliabilities must sum to > 0")

    clusters = associate_entries(catalogues, peak_tol_s, loc_tol_deg)
    own = list(own_detection_times or [])

    labels: list[ConsensusLabel] = []
    for cluster in clusters:
        # Distinct voting catalogues (a catalogue votes once per event).
        voters = sorted({e.catalog for e in cluster})
        voted_rho = sum(reliabilities.get(v, 0.0) for v in voters)
        conf = voted_rho / total_rho
        mean_t = sum(e.t_peak for e in cluster) / len(cluster)

        any_hxr = any(e.is_hxr for e in cluster)
        any_sxr = any(e.is_sxr for e in cluster)
        sxr_weak = any_hxr and not any_sxr

        # Base label from thresholds.
        if conf >= tau_hi:
            state = "confirmed"
        elif conf >= tau_lo:
            state = "candidate"
        else:
            state = "rejected"

        # Conflict rule: HXR present, SXR absent -> keep as confirmed HXR event.
        if sxr_weak and state == "candidate":
            state = "confirmed"

        # Promotion: a candidate corroborated by our own fused detection.
        if state == "candidate" and own:
            if any(abs(mean_t - t) <= peak_tol_s for t in own):
                state = "confirmed"

        # Reconcile class: prefer the GOES catalogue, else first available.
        goes_class = None
        for e in cluster:
            if e.catalog.upper().startswith("GOES") and e.goes_class:
                goes_class = e.goes_class
                break
        if goes_class is None:
            for e in cluster:
                if e.goes_class:
                    goes_class = e.goes_class
                    break

        # Consensus location (first voter that supplied one).
        lon = lat = None
        for e in cluster:
            if e.lon_deg is not None and e.lat_deg is not None:
                lon, lat = e.lon_deg, e.lat_deg
                break

        labels.append(
            ConsensusLabel(
                t_peak=mean_t,
                confidence=conf,
                state=state,
                goes_class=goes_class,
                voters=voters,
                n_catalogues=len(voters),
                sxr_weak=sxr_weak,
                lon_deg=lon,
                lat_deg=lat,
                members=list(cluster),
            )
        )

    labels.sort(key=lambda x: x.t_peak)
    return labels


def _ang_sep(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Great-circle angular separation [deg] between two lon/lat points."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    cos_sep = (
        math.sin(p1) * math.sin(p2)
        + math.cos(p1) * math.cos(p2) * math.cos(dl)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))
