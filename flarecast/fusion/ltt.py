"""Light-travel-time (LTT) correction to the common Earth/L1 frame.

Mandatory **before** fusion (ARCHITECTURE.md Section 3.5, research doc
``06 Section 1.4``). A UTC timestamp on two spacecraft does *not* mean they saw
the same photons: they sit at different heliocentric distances, so a flash
emitted at solar time ``t_sun`` reaches spacecraft *k* at

    t_k = t_sun + (r_k - R_sun) / c .

To bring spacecraft *k* onto the Earth reference frame (reference distance
``r_ref ~ 1 AU``) we **add** the differential path delay:

    Δt_(k->earth) = (r_ref - r_k) / c ,
    t_earth_utc   = t_obs_utc + Δt_(k->earth) .

``Δt > 0`` means *k* is closer to the Sun than Earth (saw the flare earlier),
so we push its clock forward to align with the Earth frame. The
``R_sun`` term cancels when differencing two spacecraft, so it is dropped (we
work in the heliocentric frame). Constants come from :mod:`flarecast.constants`
(``C_KM_S``, ``AU_KM``, ``EARTH_R_AU``, ``ADITYA_L1_R_AU``).

Pure standard library -- the arithmetic is scalar.

Worked numbers reproduced by this module (Section 3.5 table):

========================  ============  ====================
Spacecraft                r_k [AU]      Δt to Earth frame
========================  ============  ====================
Aditya-L1                 ~0.990        +5.0 s
Earth / GOES (reference)  1.000         0
STEREO-A                  ~0.96         +20 s
Solar Orbiter (~0.5 AU)   0.50          ~+249 s (EAR_TDEL)
Parker (perihelion)       ~0.046        +476 s
========================  ============  ====================

In-situ **PROTON** streams keep their own clock (particle transport delay along
the Parker spiral, not c) and are never LTT-corrected -- see
:func:`light_travel_correction_record`.
"""

from __future__ import annotations

from ..constants import (
    ADITYA_L1_R_AU,
    AU_KM,
    C_KM_S,
    EARTH_R_AU,
)
from ..types import Quantity

__all__ = [
    "ltt_delta_seconds",
    "light_travel_correction",
    "light_travel_correction_record",
    "NAMED_PLATFORM_R_AU",
    "delta_seconds_for_platform",
    "correct_for_platform",
]


# ---------------------------------------------------------------------------
# Representative heliocentric distances for named platforms [AU]
# ---------------------------------------------------------------------------
# These are the canonical *representative* distances from ARCHITECTURE.md
# Section 3.5 / research 06 Section 1.4. For eccentric orbits (Solar Orbiter,
# Parker, STEREO) the true r_k(t) is time-dependent and must come from SPICE /
# mission ephemerides (or STIX ``EAR_TDEL``) -- these constants are only
# order-of-magnitude anchors and sanity defaults, never a substitute for the
# per-sample ``vantage_r_au`` when it is available.
NAMED_PLATFORM_R_AU: dict[str, float] = {
    "ADITYA-L1": ADITYA_L1_R_AU,    # ~0.990 AU  -> +5.0 s
    "L1": ADITYA_L1_R_AU,
    "EARTH": EARTH_R_AU,            # reference frame -> 0 s
    "GOES": EARTH_R_AU,
    "STEREO-A": 0.96,              # -> ~+20 s
    "SOLAR ORBITER": 0.50,         # typical -> ~+249 s (matches EAR_TDEL)
    "SOLO": 0.50,
    "PARKER": 0.046,               # perihelion -> ~+476 s
    "PSP": 0.046,
}


def ltt_delta_seconds(vantage_r_au: float, r_ref_au: float = EARTH_R_AU) -> float:
    """Differential light-travel delay to the reference frame [s].

    Implements Appendix B.3 ``ltt.ltt_delta_seconds``::

        Δt = (r_ref - r_k) * AU_KM / c

    Positive when the spacecraft is closer to the Sun than the reference
    (it saw the flare earlier; add this to push its clock to the reference
    frame). For Aditya-L1 (``vantage_r_au = ADITYA_L1_R_AU``) this is ~+5.0 s.

    Parameters
    ----------
    vantage_r_au:
        Heliocentric distance of the spacecraft [AU].
    r_ref_au:
        Reference heliocentric distance [AU] (default Earth/L1 = 1 AU).
    """
    return (r_ref_au - vantage_r_au) * AU_KM / C_KM_S


def light_travel_correction(
    t_obs_utc: float,
    vantage_r_au: float,
    r_ref_au: float = EARTH_R_AU,
) -> float:
    """LTT-correct an observed time to the Earth/L1 reference frame [s].

    Implements Appendix B.3 ``ltt.light_travel_correction``::

        t_earth_utc = t_obs_utc + (r_ref - r_k) * AU_KM / c

    Returns the fusion key ``t_earth_utc``. The argument name ``vantage_r_au``
    matches the ``FusionRecord`` field; callers passing the documented
    ``r_sc_au`` keyword are equivalent (same heliocentric distance).

    Parameters
    ----------
    t_obs_utc:
        Raw observed timestamp [s] (Unix or J2000 epoch -- only differences
        matter, so the offset is irrelevant here).
    vantage_r_au:
        Heliocentric distance of the spacecraft [AU].
    r_ref_au:
        Reference heliocentric distance [AU] (default Earth/L1 = 1 AU).
    """
    return t_obs_utc + ltt_delta_seconds(vantage_r_au, r_ref_au)


def delta_seconds_for_platform(
    platform: str, r_ref_au: float = EARTH_R_AU
) -> float:
    """LTT delay [s] for a *named* platform using its representative distance.

    Convenience helper over :data:`NAMED_PLATFORM_R_AU` for L1, STEREO-A,
    Solar Orbiter, Parker, etc. Matching is case-insensitive and
    whitespace-tolerant. Raises :class:`KeyError` for unknown platforms (the
    caller should supply ``vantage_r_au`` from ephemerides instead).
    """
    key = " ".join(platform.strip().upper().split())
    if key not in NAMED_PLATFORM_R_AU:
        raise KeyError(
            f"unknown platform {platform!r}; pass vantage_r_au from "
            f"ephemerides (known: {sorted(NAMED_PLATFORM_R_AU)})"
        )
    return ltt_delta_seconds(NAMED_PLATFORM_R_AU[key], r_ref_au)


def correct_for_platform(
    t_obs_utc: float, platform: str, r_ref_au: float = EARTH_R_AU
) -> float:
    """LTT-correct ``t_obs_utc`` for a named platform -> ``t_earth_utc`` [s]."""
    return t_obs_utc + delta_seconds_for_platform(platform, r_ref_au)


def light_travel_correction_record(record, r_ref_au: float = EARTH_R_AU):
    """Set ``record.t_earth_utc`` from ``t_obs_utc`` and ``vantage_r_au``.

    Mutates and returns the :class:`~flarecast.fusion.schema.FusionRecord` in
    place. **In-situ PROTON streams are exempt**: their photons-vs-particles
    timing model differs (transport delay along field lines, not c), so we copy
    ``t_obs_utc`` unchanged onto the fusion key and never fuse their timing with
    photon timing (research 06 Section 1.4).
    """
    if record.quantity == Quantity.PROTON.value:
        record.t_earth_utc = record.t_obs_utc
        return record
    record.t_earth_utc = light_travel_correction(
        record.t_obs_utc, record.vantage_r_au, r_ref_au
    )
    return record
