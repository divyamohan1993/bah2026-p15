"""Light-travel-time correction physics (ARCHITECTURE.md Section 3.5 / 06 §1.4).

Verifies the LTT numbers and sign convention the whole fusion layer depends on:

* Aditya-L1 lead over the Earth/GOES frame is ~ +5.0 s.
* Solar Orbiter lead is ~ +240 s (order-of-magnitude vs the published STIX
  ``EAR_TDEL`` ~ +239.9 s).
* The sign convention is correct: a spacecraft *closer* to the Sun saw the flare
  *earlier*, so ``Delta t > 0`` is added to push its clock to the Earth frame.

Pure standard library; no network, no numpy.
"""

from __future__ import annotations

from flarecast.constants import (
    ADITYA_L1_LTT_LEAD_S,
    ADITYA_L1_R_AU,
    AU_KM,
    C_KM_S,
    EARTH_R_AU,
)
from flarecast.fusion import ltt
from flarecast.types import Quantity


def test_l1_lead_is_about_five_seconds():
    """Aditya-L1 sees flares ~5.0 s before GOES (Section 3.5 table)."""
    dt = ltt.ltt_delta_seconds(ADITYA_L1_R_AU)
    assert 4.8 < dt < 5.2, f"L1 LTT lead {dt} not ~ +5.0 s"
    # Must equal the frozen constant exactly.
    assert abs(dt - ADITYA_L1_LTT_LEAD_S) < 1e-9
    # And the named-platform helper agrees.
    assert abs(ltt.delta_seconds_for_platform("Aditya-L1") - dt) < 1e-9
    assert abs(ltt.delta_seconds_for_platform("L1") - dt) < 1e-9


def test_solar_orbiter_lead_order_of_magnitude():
    """Solar Orbiter (~0.5 AU) lead ~ +240-250 s vs published EAR_TDEL ~ +239.9 s."""
    dt = ltt.delta_seconds_for_platform("Solar Orbiter")
    # Representative 0.5 AU -> ~+249.5 s; accept the ~+240 s neighbourhood.
    assert 200.0 < dt < 300.0, f"SolO LTT lead {dt} not order ~ +240 s"
    # Sanity vs the literature anchor (EAR_TDEL ~ 239.9 s) within ~15%.
    assert abs(dt - 239.9) / 239.9 < 0.15
    # Direct distance form matches.
    dt_direct = ltt.ltt_delta_seconds(0.50)
    assert abs(dt - dt_direct) < 1e-9


def test_earth_reference_is_zero():
    """The Earth/GOES reference frame has zero LTT correction."""
    assert abs(ltt.ltt_delta_seconds(EARTH_R_AU)) < 1e-12
    assert abs(ltt.delta_seconds_for_platform("Earth")) < 1e-12
    assert abs(ltt.delta_seconds_for_platform("GOES")) < 1e-12


def test_sign_convention_closer_is_earlier():
    """Closer to the Sun -> positive delay added (saw the flare earlier)."""
    # 0.5 AU (closer than Earth) -> positive; 1.5 AU (farther) -> negative.
    assert ltt.ltt_delta_seconds(0.5) > 0.0
    assert ltt.ltt_delta_seconds(1.5) < 0.0
    # Monotonic: the closer the spacecraft, the larger the (earlier) lead.
    assert ltt.ltt_delta_seconds(0.3) > ltt.ltt_delta_seconds(0.7)


def test_light_travel_correction_adds_delta():
    """``light_travel_correction`` adds the delta to the observed time."""
    t_obs = 1_000_000.0
    t_earth = ltt.light_travel_correction(t_obs, ADITYA_L1_R_AU)
    assert abs((t_earth - t_obs) - ADITYA_L1_LTT_LEAD_S) < 1e-9
    # Explicit reference distance argument is honoured.
    t_earth2 = ltt.light_travel_correction(t_obs, 0.5, r_ref_au=EARTH_R_AU)
    assert abs((t_earth2 - t_obs) - ltt.ltt_delta_seconds(0.5)) < 1e-9


def test_formula_matches_closed_form():
    """The implementation matches Delta t = (r_ref - r_sc) * AU / c exactly."""
    for r_au in (0.046, 0.28, 0.5, 0.96, 0.99, 1.0, 1.01):
        expect = (EARTH_R_AU - r_au) * AU_KM / C_KM_S
        assert abs(ltt.ltt_delta_seconds(r_au) - expect) < 1e-9


def test_parker_and_stereo_match_table():
    """Parker (~+476 s) and STEREO-A (~+20 s) match the Section 3.5 table."""
    parker = ltt.delta_seconds_for_platform("Parker")
    stereo = ltt.delta_seconds_for_platform("STEREO-A")
    assert 400.0 < parker < 520.0, f"Parker {parker} not ~ +476 s"
    assert 15.0 < stereo < 25.0, f"STEREO-A {stereo} not ~ +20 s"


def test_proton_streams_keep_their_own_clock():
    """In-situ PROTON records are exempt from LTT (own transport clock)."""

    class _Rec:
        # Minimal stand-in with the attributes the helper touches.
        quantity = Quantity.PROTON.value
        t_obs_utc = 12345.0
        vantage_r_au = 0.5
        t_earth_utc = 0.0

    rec = _Rec()
    ltt.light_travel_correction_record(rec)
    assert rec.t_earth_utc == rec.t_obs_utc, "PROTON must keep its own clock"


def test_unknown_platform_raises():
    """A platform without an ephemeris entry raises (caller must pass r_au)."""
    import pytest

    with pytest.raises(KeyError):
        ltt.delta_seconds_for_platform("Nonexistent Probe")
