"""Tests for GOES classification + SDD arbitration (research 03 Section 3).

Boundary cases across A/B/C/M/X, sub-class mantissa (the canonical
``2.5e-5 -> "M2.5"``), open-ended X, the inverse helper, and the SoLEXS SDD1->SDD2
saturation arbitration. Pure stdlib.
"""

from __future__ import annotations

from flarecast.constants import SDD_SATURATION_CPS
from flarecast.detect.classify import (
    class_to_flux,
    classify_flux,
    sdd_arbitration,
    select_sdd,
)


def test_classify_canonical_example():
    """The documented example: 2.5e-5 W/m^2 -> 'M2.5'."""
    assert classify_flux(2.5e-5) == "M2.5"


def test_classify_decade_floors():
    assert classify_flux(1e-7) == "B1.0"
    assert classify_flux(1e-6) == "C1.0"
    assert classify_flux(1e-5) == "M1.0"
    assert classify_flux(1e-4) == "X1.0"


def test_classify_each_letter():
    assert classify_flux(5e-8) == "A5.0"  # A class
    assert classify_flux(3.0e-7) == "B3.0"  # B class
    assert classify_flux(3.1e-6) == "C3.1"  # C class
    assert classify_flux(7.2e-5) == "M7.2"  # M class
    assert classify_flux(2.0e-4) == "X2.0"  # X class


def test_classify_just_below_decade_boundary():
    # 9.9e-5 is still M (below the 1e-4 X floor).
    assert classify_flux(9.9e-5) == "M9.9"
    # 9.9e-6 is still C.
    assert classify_flux(9.9e-6) == "C9.9"


def test_classify_x_open_ended():
    # X is open-ended: mantissa measured against the 1e-4 X floor.
    assert classify_flux(1e-3) == "X10.0"
    assert classify_flux(1.2e-3) == "X12.0"


def test_classify_below_a_floor_clamped():
    # Below the nominal A decade: clamp to A, mantissa vs the A decade (1e-8).
    assert classify_flux(1e-9) == "A0.1"


def test_classify_nonpositive():
    assert classify_flux(0.0) == "A0.0"
    assert classify_flux(-1.0) == "A0.0"


def test_class_to_flux_inverse():
    assert abs(class_to_flux("M2.5") - 2.5e-5) < 1e-12
    assert abs(class_to_flux("X1.0") - 1e-4) < 1e-12
    assert abs(class_to_flux("C5") - 5e-6) < 1e-12
    assert abs(class_to_flux("M") - 1e-5) < 1e-12  # mantissa defaults to 1.0


def test_class_to_flux_roundtrip():
    for f in (5e-8, 3.1e-6, 2.5e-5, 7.2e-5, 2.0e-4):
        cls = classify_flux(f)
        back = class_to_flux(cls)
        # Round-trip is accurate to the one-decimal mantissa quantisation.
        assert abs(back - f) / f < 0.05


def test_class_to_flux_rejects_bad():
    for bad in ("", "Z1.0", "Q"):
        try:
            class_to_flux(bad)
        except ValueError:
            continue
        raise AssertionError(f"class_to_flux accepted invalid class {bad!r}")


# ---------------------------------------------------------------------------
# SDD arbitration
# ---------------------------------------------------------------------------
def test_select_sdd_below_saturation_uses_sdd1():
    det, val = select_sdd(5e4, 1e3)
    assert det == "SDD1"
    assert val == 5e4


def test_select_sdd_above_saturation_switches_to_sdd2():
    det, val = select_sdd(2e5, 4e3)
    assert det == "SDD2"
    assert val == 4e3


def test_select_sdd_at_exact_ceiling_switches():
    # At/above the ceiling SDD1 is saturated -> use SDD2.
    det, _ = select_sdd(SDD_SATURATION_CPS, 1e3)
    assert det == "SDD2"


def test_sdd_arbitration_flags_saturation():
    res = sdd_arbitration(2e5, 4e3)
    assert res["detector_used"] == "SDD2"
    assert res["saturated"] is True
    assert res["both_saturated"] is False


def test_sdd_arbitration_both_saturated_lower_bound():
    res = sdd_arbitration(2e5, 2e5)
    assert res["saturated"] is True
    assert res["both_saturated"] is True  # report a >=Xn lower bound downstream


def test_sdd_arbitration_quiet_uses_sdd1():
    res = sdd_arbitration(1e3, 1e2)
    assert res["detector_used"] == "SDD1"
    assert res["saturated"] is False
