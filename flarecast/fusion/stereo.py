"""Multi-viewpoint stereoscopy + IPN timing triangulation (Section 3.10 / 06 §5).

For impulsive HXR/gamma bursts seen by **non-imaging** detectors (HEL1OS,
Fermi GBM, Konus-Wind, GECAM), the Interplanetary Network (IPN) locates the
source purely from **arrival-time differences** across widely separated
spacecraft (research doc ``06 Section 5.4``). For a baseline ``D12`` between two
spacecraft, the same burst arrives ``dt`` apart and the source direction makes
angle ``theta`` with the baseline vector::

    cos(theta) = c * dt / D12

confining the source to an **annulus** on the sky of half-width::

    dtheta ~ c * sigma_dt / (D12 * sin(theta))

Longer baselines -> tighter annulus. Three+ spacecraft -> intersecting annuli ->
an IPN **error box**. IPN is also a powerful **false-alarm killer**: a real solar
transient triangulates consistently across separated spacecraft; a local
particle hit does not (research 06 Section 5.4 / 7.3).

* :func:`ipn_annulus` -- one spacecraft pair -> :class:`Annulus`.
* :func:`localize_burst` -- a trigger table of raw sub-second arrival times ->
  list of annuli (the IPN constraint set).
* :func:`baseline_km` / :func:`tag_far_side` -- geometry + far-side tagging.

Pure standard library (the math is scalar/3-vector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..constants import C_KM_S

__all__ = [
    "Annulus",
    "TriggerObservation",
    "ipn_annulus",
    "localize_burst",
    "baseline_km",
    "tag_far_side",
    "is_consistent_solar",
]


@dataclass(slots=True)
class Annulus:
    """An IPN annulus (ring) constraint on the sky.

    Field names follow Appendix B.3 ``stereo.Annulus``. The ring is centered on
    the baseline direction; the source lies on (or, within ``halfwidth_deg``,
    near) the small circle of angular radius ``radius_deg`` about that center.

    Attributes
    ----------
    center_lon_lat:
        ``(lon_deg, lat_deg)`` of the baseline direction (annulus center).
    radius_deg:
        Annulus angular radius ``theta`` [deg] from ``cos theta = c dt / D``.
    halfwidth_deg:
        1-sigma annulus half-width ``dtheta`` [deg] from the timing uncertainty.
    pair:
        ``(source_id_1, source_id_2)`` that produced the annulus (provenance).
    """

    center_lon_lat: tuple[float, float]
    radius_deg: float
    halfwidth_deg: float
    pair: tuple[str, str] = ("", "")


@dataclass(slots=True)
class TriggerObservation:
    """One spacecraft's view of an impulsive burst (raw sub-grid timing).

    The IPN uses **raw sub-second arrival times**, kept in a side table -- never
    the resampled grid (research 06 Section 2.3 / 5.4).

    Attributes
    ----------
    source_id:
        Detector / spacecraft id (e.g. ``"ADITYA_HEL1OS"``, ``"FERMI_GBM"``).
    t_arrival_s:
        Burst arrival time at this spacecraft [s] (its own clock, sub-second).
    xyz_hci_km:
        Spacecraft position in Heliocentric Inertial coords [km].
    sigma_t_s:
        1-sigma cross-correlation timing uncertainty [s].
    sub_sc_lon_deg:
        Sub-spacecraft Carrington/Stonyhurst longitude [deg] (far-side tagging).
    """

    source_id: str
    t_arrival_s: float
    xyz_hci_km: tuple[float, float, float]
    sigma_t_s: float = 1e-3
    sub_sc_lon_deg: float = 0.0


def baseline_km(
    xyz1: Sequence[float], xyz2: Sequence[float]
) -> float:
    """Euclidean distance between two HCI positions [km]."""
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(xyz1, xyz2))
    )


def _xyz_to_lon_lat(xyz: Sequence[float]) -> tuple[float, float]:
    """Direction of a 3-vector -> ``(lon_deg, lat_deg)``."""
    x, y, z = xyz[0], xyz[1], xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        return 0.0, 0.0
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    return lon, lat


def ipn_annulus(
    dt_s: float,
    baseline_km: float,
    sigma_dt_s: float,
    baseline_dir: tuple[float, float],
) -> Annulus:
    """One spacecraft pair -> IPN annulus (Appendix B.3 ``stereo.ipn_annulus``).

    ``cos theta = c * dt / D12`` (clipped to [-1, 1]); the half-width is
    ``dtheta = c * sigma_dt / (D12 * sin theta)`` (research 06 Section 5.4).

    Parameters
    ----------
    dt_s:
        Arrival-time difference between the two spacecraft [s].
    baseline_km:
        Baseline length ``D12`` [km] (must be > 0).
    sigma_dt_s:
        1-sigma timing (cross-correlation) uncertainty [s].
    baseline_dir:
        ``(lon_deg, lat_deg)`` of the baseline vector (annulus center).

    Returns
    -------
    Annulus

    Raises
    ------
    ValueError
        If ``baseline_km <= 0``.
    """
    if baseline_km <= 0.0:
        raise ValueError("ipn_annulus: baseline_km must be > 0")
    cos_theta = C_KM_S * dt_s / baseline_km
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        # Source near the baseline axis -> half-width is ill-conditioned; cap.
        halfwidth = 90.0
    else:
        halfwidth = math.degrees(C_KM_S * sigma_dt_s / (baseline_km * sin_theta))
    return Annulus(
        center_lon_lat=baseline_dir,
        radius_deg=math.degrees(theta),
        halfwidth_deg=halfwidth,
    )


def localize_burst(trigger_table: Sequence[TriggerObservation]) -> list[Annulus]:
    """IPN-localize a burst from a trigger table (Appendix B.3).

    Forms an :class:`Annulus` for every spacecraft pair from the raw sub-second
    arrival times. With N >= 3 spacecraft the resulting annuli intersect in an
    IPN error box; this function returns the annulus set (the box construction /
    imager fusion is a downstream step). ``dt`` for each pair is the difference
    of the two recorded arrival times; ``sigma_dt`` is the quadrature sum of the
    two per-detector timing uncertainties.

    Returns an empty list for fewer than two observations.
    """
    obs = list(trigger_table)
    annuli: list[Annulus] = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            o1, o2 = obs[i], obs[j]
            d12 = baseline_km(o1.xyz_hci_km, o2.xyz_hci_km)
            if d12 <= 0.0:
                continue
            dt = o2.t_arrival_s - o1.t_arrival_s
            sigma_dt = math.hypot(o1.sigma_t_s, o2.sigma_t_s)
            # Baseline direction = unit vector from o1 toward o2.
            diff = tuple(b - a for a, b in zip(o1.xyz_hci_km, o2.xyz_hci_km))
            ann = ipn_annulus(dt, d12, sigma_dt, _xyz_to_lon_lat(diff))
            ann.pair = (o1.source_id, o2.source_id)
            annuli.append(ann)
    return annuli


def is_consistent_solar(
    trigger_table: Sequence[TriggerObservation],
    max_residual_s: float | None = None,
) -> bool:
    """Cross-vantage veto: does the burst triangulate consistently?

    A real solar transient yields arrival-time differences consistent with a
    single sky direction across all pairs; a local particle hit does not
    (research 06 Section 5.4 -- the geometry-based false-alarm killer). This is a
    lightweight consistency check: every pairwise ``|c*dt|`` must be physically
    realizable (``<= D12`` within timing error), i.e. ``|cos theta| <= 1``.

    Returns ``True`` if all pairs are physically consistent, ``False`` if any
    pair implies a superluminal delay beyond its timing uncertainty (a spike).
    """
    obs = list(trigger_table)
    if len(obs) < 2:
        return True  # cannot veto with a single detector
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            o1, o2 = obs[i], obs[j]
            d12 = baseline_km(o1.xyz_hci_km, o2.xyz_hci_km)
            if d12 <= 0.0:
                continue
            dt = o2.t_arrival_s - o1.t_arrival_s
            sigma_dt = math.hypot(o1.sigma_t_s, o2.sigma_t_s)
            tol = max_residual_s if max_residual_s is not None else 3.0 * sigma_dt
            # Max physically realizable delay is D12/c.
            max_dt = d12 / C_KM_S
            if abs(dt) > max_dt + tol:
                return False
    return True


def tag_far_side(
    sub_earth_lon_deg: float,
    source_lon_deg: float,
    limb_margin_deg: float = 90.0,
) -> bool:
    """Tag a flare as **far-side** (occulted from the Earth/L1 line).

    A source whose Carrington/Stonyhurst longitude is more than ``limb_margin_deg``
    from the sub-Earth longitude is behind the limb as seen from Earth/L1
    (research 06 Section 5.1). Such detections (from STEREO-A / Solar Orbiter)
    enter the catalogue as occulted events and give a multi-day pre-rotation
    forecast warning. Longitudes wrap at +-180 deg.

    Returns ``True`` if the source is far-side.
    """
    diff = abs(_wrap180(source_lon_deg - sub_earth_lon_deg))
    return diff > limb_margin_deg


def _wrap180(angle_deg: float) -> float:
    """Wrap an angle to (-180, 180] degrees."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a if a != -180.0 else 180.0
