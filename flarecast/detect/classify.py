"""GOES A-X flare classification + SoLEXS SDD arbitration (Section 4.5, B.4).

A flare's class is the **order of magnitude of its peak 1-8 A soft-X-ray flux**
(W m^-2), with a linear 1-9 sub-class (mantissa) within each decade
(research doc ``03 Section 3``)::

    A  F < 1e-7        B  1e-7 <= F < 1e-6      C  1e-6 <= F < 1e-5
    M  1e-5 <= F < 1e-4                          X  F >= 1e-4

So ``M2.5`` means ``2.5e-5 W m^-2``. Classification is pure arithmetic -> O(1).

The module also handles **SoLEXS detector arbitration**: SDD1 (large aperture)
saturates above ~1e5 counts/s, so above that ceiling the unsaturated SDD2 (small
aperture, captured the May-2024 X8.7) must be read before computing peak flux
(research doc ``03 Section 3.3`` / ``Section 7``).
"""

from __future__ import annotations

import math

from ..constants import (
    GOES_CLASS_A_WM2,
    GOES_CLASS_THRESHOLDS_WM2,
    GOES_CLASS_X_WM2,
    SDD_SATURATION_CPS,
)

__all__ = ["classify_flux", "class_to_flux", "select_sdd", "sdd_arbitration"]

# Decade exponent -> class letter for the canonical A..X decades.
_EXP_TO_LETTER: dict[int, str] = {-8: "A", -7: "B", -6: "C", -5: "M", -4: "X"}


def classify_flux(flux_wm2: float) -> str:
    """Return the GOES class string for a peak 1-8 A flux (``03 Section 3.2``).

    Closed-form, **O(1)**. Examples: ``2.5e-5 -> "M2.5"``, ``1e-4 -> "X1.0"``,
    ``9.9e-5 -> "M9.9"``, ``1.2e-3 -> "X12.0"`` (X is open-ended), ``5e-8 ->
    "A5.0"``, ``1e-9 -> "A0.1"`` (clamped to the A decade).

    A non-positive flux returns ``"A0.0"`` (no measurable emission).
    """
    f = float(flux_wm2)
    if f <= 0.0:
        return "A0.0"

    e = math.floor(math.log10(f))

    if e >= -4:
        # X-class is open-ended: mantissa is measured against the X floor 1e-4,
        # so X10 = 1e-3, X12 = 1.2e-3, etc.
        mant = f / GOES_CLASS_X_WM2
        return f"X{mant:.1f}"

    if e < -8:
        # Below the nominal A decade floor: clamp to 'A' and express the
        # mantissa relative to the A decade (1e-8) so e.g. 1e-9 -> "A0.1".
        mant = f / GOES_CLASS_A_WM2
        return f"A{mant:.1f}"

    letter = _EXP_TO_LETTER[e]
    mant = f / (10.0**e)
    return f"{letter}{mant:.1f}"


def class_to_flux(cls: str) -> float:
    """Inverse of :func:`classify_flux`: GOES class string -> peak flux (W m^-2).

    ``"M2.5" -> 2.5e-5``, ``"X1.0" -> 1e-4``, ``"C5" -> 5e-6``. The mantissa is
    optional (defaults to ``1.0``: ``"M" -> 1e-5``). O(1).

    Raises
    ------
    ValueError
        If the letter is not one of A/B/C/M/X or the mantissa is unparseable.
    """
    if not cls:
        raise ValueError("empty class string")
    letter = cls[0].upper()
    if letter not in GOES_CLASS_THRESHOLDS_WM2:
        raise ValueError(f"unknown GOES class letter: {cls!r}")
    mant_str = cls[1:].strip()
    mant = float(mant_str) if mant_str else 1.0
    return mant * GOES_CLASS_THRESHOLDS_WM2[letter]


def select_sdd(
    sdd1_cps: float, sdd2_cps: float, saturation: float = SDD_SATURATION_CPS
) -> tuple[str, float]:
    """Pick the unsaturated SoLEXS SDD and its count rate (``03 Section 3.3/7``).

    SDD1 (large aperture) is preferred for quiet/small flares for its
    sensitivity, but it saturates (paralyzable turnover) above ``saturation``
    counts/s; above that we must read SDD2 (small aperture, stays linear to
    X-class). O(1).

    Returns
    -------
    (detector, value)
        ``("SDD1", sdd1_cps)`` when SDD1 is below saturation, else
        ``("SDD2", sdd2_cps)``. The caller flags ``saturated`` when SDD1 was
        over the ceiling, and reports a lower bound if *both* saturate.
    """
    if sdd1_cps < saturation:
        return "SDD1", float(sdd1_cps)
    return "SDD2", float(sdd2_cps)


def sdd_arbitration(
    sdd1_cps: float, sdd2_cps: float, saturation: float = SDD_SATURATION_CPS
) -> dict:
    """Full SDD arbitration result (research doc ``03 Section 3.3 / 7``).

    A richer wrapper around :func:`select_sdd` that also reports the saturation
    state needed for the catalogue's ``soft.saturated`` flag and the
    lower-bound case (``>=Xn``) when both detectors saturate. O(1).

    Returns a dict with keys ``detector_used`` (``"SDD1"``/``"SDD2"``),
    ``value`` (the chosen count rate), ``saturated`` (bool: SDD1 over ceiling),
    and ``both_saturated`` (bool: even SDD2 is over ceiling -> class is a lower
    bound).
    """
    s1_sat = sdd1_cps >= saturation
    s2_sat = sdd2_cps >= saturation
    detector, value = select_sdd(sdd1_cps, sdd2_cps, saturation)
    return {
        "detector_used": detector,
        "value": value,
        "saturated": s1_sat,
        "both_saturated": s1_sat and s2_sat,
    }
