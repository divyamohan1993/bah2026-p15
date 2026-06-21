"""Workstream 1 tests: synthetic generator + ingest fallback chain (B.1/B.2).

These tests are **pure standard library** and perform **no network I/O**: they
prove the offline keystone works deterministically (ARCHITECTURE.md Section 6).

Coverage:

* the synthetic generator produces well-formed :class:`FluxSample` records and a
  truth event list, **deterministically** given a fixed seed (Appendix B.2);
* the :class:`GOESFetcher` falls back **live -> cache -> synth** when the network
  is forced off (Appendix B.1 / Section 6 offline-fallback strategy);
* the SWPC normalizer maps a record to a :class:`FluxSample` with the correct
  GOES class (Appendix B.1, Section 4.5).

numpy / pandas are optional accelerators; the parts that use pandas are guarded
with ``pytest.importorskip("pandas")`` so the suite passes (pass *or* skip) in a
minimal environment.
"""

from __future__ import annotations

import urllib.error

import pytest
from flarecast.constants import DEFAULT_CLASS_MIX, UNIT_HXR, UNIT_SXR
from flarecast.ingest.base import FALLBACK_ERRORS, Fetcher, with_fallback
from flarecast.ingest.goes import GOESFetcher, flare_class
from flarecast.ingest.normalize import (
    SWPC_ENERGY_LONG,
    SWPC_ENERGY_SHORT,
    goes_class,
    normalize_generic,
    normalize_swpc,
    parse_swpc_time,
)
from flarecast.synth.generator import (
    STREAM_HXR_HIGH,
    STREAM_HXR_LOW,
    STREAM_SXR_LONG,
    STREAM_SXR_SHORT,
    TruthEvent,
    generate_flare_lightcurves,
)
from flarecast.types import FluxSample, Quantity

# Window covering the bundled xrays-1-day.sample.json (2026-06-18 09:00-12:00Z).
_SAMPLE_T0 = parse_swpc_time("2026-06-18T09:00:00Z")
_SAMPLE_T1 = parse_swpc_time("2026-06-18T12:00:00Z")


# ===========================================================================
# 1. Synthetic generator: well-formed + deterministic (B.2)
# ===========================================================================
def test_synth_returns_samples_and_truth():
    samples, truth = generate_flare_lightcurves(
        duration_s=86400.0, cadence_s=60.0, seed=42
    )
    assert isinstance(samples, list)
    assert isinstance(truth, list)
    assert len(samples) > 0
    assert len(truth) > 0
    assert all(isinstance(s, FluxSample) for s in samples)
    assert all(isinstance(e, TruthEvent) for e in truth)


def test_synth_samples_are_well_formed():
    samples, _ = generate_flare_lightcurves(duration_s=43200.0, cadence_s=60.0, seed=1)
    valid_quantities = {q.value for q in Quantity}
    valid_units = {UNIT_SXR, UNIT_HXR}
    for s in samples:
        assert s.stream
        assert isinstance(s.t, float)
        assert isinstance(s.value, float)
        # flux/counts must be finite and non-negative.
        assert s.value == s.value  # not NaN
        assert s.value >= 0.0
        assert s.unit in valid_units
        assert s.source == "synth"
        assert s.quantity in valid_quantities
        assert s.qc != 0  # QC stamped


def test_synth_emits_all_four_streams():
    samples, _ = generate_flare_lightcurves(duration_s=43200.0, cadence_s=60.0, seed=3)
    streams = {s.stream for s in samples}
    assert streams == {
        STREAM_SXR_LONG,
        STREAM_SXR_SHORT,
        STREAM_HXR_LOW,
        STREAM_HXR_HIGH,
    }
    # soft channels carry W m^-2; hard channels carry counts/s.
    for s in samples:
        if s.stream in (STREAM_SXR_LONG, STREAM_SXR_SHORT):
            assert s.unit == UNIT_SXR
        else:
            assert s.unit == UNIT_HXR


def test_synth_truth_event_ordering_and_class():
    _, truth = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=42)
    # truth events sorted by peak time.
    peaks = [e.t_peak for e in truth]
    assert peaks == sorted(peaks)
    for e in truth:
        # start <= peak <= end, all finite.
        assert e.t_start <= e.t_peak <= e.t_end
        # class letter is one of the GOES ladder.
        assert e.goes_class[0].upper() in {"A", "B", "C", "M", "X"}
        assert e.peak_flux_wm2 > 0
        # Neupert lead is positive (HXR leads the soft peak).
        assert e.hxr_lead_s > 0


def test_synth_is_deterministic_with_fixed_seed():
    s1, t1 = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=7)
    s2, t2 = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=7)
    assert len(s1) == len(s2)
    # byte-for-byte identical sample streams.
    for a, b in zip(s1, s2, strict=True):
        assert a.stream == b.stream
        assert a.t == b.t
        assert a.value == b.value
        assert a.cls == b.cls
    # identical truth lists.
    assert [e.to_dict() for e in t1] == [e.to_dict() for e in t2]


def test_synth_different_seeds_differ():
    s1, _ = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=1)
    s2, _ = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=2)
    vals1 = [s.value for s in s1]
    vals2 = [s.value for s in s2]
    assert vals1 != vals2


def test_synth_honors_class_mix_all_x():
    # a pure-X mix must yield only X-class truth events.
    _, truth = generate_flare_lightcurves(
        duration_s=86400.0, cadence_s=60.0, n_flares=8, class_mix={"X": 1.0}, seed=11
    )
    assert len(truth) == 8
    assert all(e.goes_class[0].upper() == "X" for e in truth)


def test_synth_class_mix_default_is_c_dominated():
    # with the default mix and many flares, C should dominate (70%).
    _, truth = generate_flare_lightcurves(
        duration_s=86400.0,
        cadence_s=60.0,
        n_flares=60,
        class_mix=dict(DEFAULT_CLASS_MIX),
        seed=2024,
    )
    letters = [e.goes_class[0].upper() for e in truth]
    c_frac = letters.count("C") / len(letters)
    # generous bound around the 0.70 expectation for n=60.
    assert c_frac > 0.5


def test_synth_neupert_off_zeroes_hard_excess():
    # with neupert=False the hard channels carry only background (+ optional
    # noise); there should be no large flare-driven excess.
    samples, _ = generate_flare_lightcurves(
        duration_s=43200.0,
        cadence_s=60.0,
        n_flares=3,
        neupert=False,
        noise=False,
        spikes=False,
        gaps=False,
        seed=5,
    )
    hard = [s.value for s in samples if s.stream == STREAM_HXR_LOW]
    # no Neupert coupling => hard stays near its quiet background, no big peaks.
    assert max(hard) < 10 * (min(hard) + 1.0)


def test_synth_saturation_flag_for_big_flares():
    # a strong X flare should drive the SDD1 saturation flag.
    _, truth = generate_flare_lightcurves(
        duration_s=86400.0, cadence_s=60.0, n_flares=5, class_mix={"X": 1.0}, seed=3
    )
    assert any(e.saturated for e in truth)


# ===========================================================================
# 2. GOES fetcher fallback: live -> cache -> synth (B.1 / Section 6)
# ===========================================================================
def test_goes_falls_back_to_cache_when_offline():
    # allow_network=False forces skipping the live tier; the bundled sample is
    # used. No network is touched.
    f = GOESFetcher(channel="both", allow_network=False)
    samples = list(f.fetch(_SAMPLE_T0, _SAMPLE_T1))
    assert f.last_source == "cache"
    assert len(samples) > 0
    # both channels present.
    streams = {s.stream for s in samples}
    assert any(s.endswith("-long") for s in streams)
    assert any(s.endswith("-short") for s in streams)


def test_goes_cache_contains_clear_flare_with_class():
    f = GOESFetcher(channel="long", allow_network=False)
    longs = list(f.fetch(_SAMPLE_T0, _SAMPLE_T1))
    assert f.last_source == "cache"
    peak = max(longs, key=lambda s: s.value)
    # the bundled sample peaks at ~C5.5 (a clear C flare).
    assert peak.cls is not None
    assert peak.cls.startswith("C")
    assert peak.value >= 1e-6  # at least C-class


def test_goes_falls_back_to_synth_when_no_cache():
    # point at a non-existent cache so the chain proceeds to synth.
    f = GOESFetcher(
        channel="both",
        allow_network=False,
        cache_path="does-not-exist.sample.json",
        synth_seed=123,
    )
    samples = list(f.fetch(0.0, 86400.0))
    assert f.last_source == "synth"
    assert len(samples) > 0
    # synth GOES fallback emits GOES-labelled soft streams only.
    assert all("goes-" in s.stream for s in samples)
    assert all(s.quantity in (Quantity.SXR_LONG.value, Quantity.SXR_SHORT.value) for s in samples)


def test_goes_synth_fallback_is_deterministic():
    kw = {
        "channel": "long",
        "allow_network": False,
        "cache_path": "does-not-exist.sample.json",
        "synth_seed": 777,
    }
    a = list(GOESFetcher(**kw).fetch(0.0, 86400.0))
    b = list(GOESFetcher(**kw).fetch(0.0, 86400.0))
    assert [s.value for s in a] == [s.value for s in b]


def test_goes_fetch_latest_from_cache():
    f = GOESFetcher(channel="long", allow_network=False)
    latest = f.fetch_latest()
    assert isinstance(latest, FluxSample)
    assert latest.stream.endswith("-long")
    assert latest.cls is not None


def test_with_fallback_skips_broken_live_source():
    # a broken live fetcher must be skipped; the offline GOES (cache) serves.
    class _BrokenLive:
        def fetch(self, t0, t1):
            raise urllib.error.URLError("network forced off")
            yield  # pragma: no cover

    chain = with_fallback(_BrokenLive(), GOESFetcher(channel="long", allow_network=False))
    out = list(chain.fetch(_SAMPLE_T0, _SAMPLE_T1))
    assert len(out) > 0
    assert chain.last_source == "GOESFetcher"


def test_with_fallback_requires_at_least_one_fetcher():
    with pytest.raises(ValueError):
        with_fallback()


def test_fetchers_satisfy_protocol():
    assert isinstance(GOESFetcher(allow_network=False), Fetcher)
    assert isinstance(
        with_fallback(GOESFetcher(allow_network=False)), Fetcher
    )


def test_fallback_error_set_covers_network_and_io():
    # the fallback must treat URL/timeout/OS errors as "try the next source".
    assert urllib.error.URLError in FALLBACK_ERRORS
    assert OSError in FALLBACK_ERRORS
    assert TimeoutError in FALLBACK_ERRORS


# ===========================================================================
# 3. SWPC normalize -> FluxSample with correct class (B.1 / Section 4.5)
# ===========================================================================
def test_normalize_swpc_long_channel_class():
    rec = {
        "time_tag": "2026-06-20T12:00:00Z",
        "satellite": 16,
        "flux": 2.5e-5,
        "observed_flux": 2.5e-5,
        "energy": SWPC_ENERGY_LONG,
    }
    s = normalize_swpc(rec)
    assert s.stream == "goes-16-long"
    assert s.quantity == Quantity.SXR_LONG.value
    assert s.unit == UNIT_SXR
    assert s.source == "SWPC"
    assert s.value == pytest.approx(2.5e-5)
    # 2.5e-5 W/m^2 is exactly M2.5.
    assert s.cls == "M2.5"


def test_normalize_swpc_short_channel_has_no_class():
    rec = {
        "time_tag": "2026-06-20T12:00:00Z",
        "satellite": 16,
        "flux": 4.0e-6,
        "energy": SWPC_ENERGY_SHORT,
    }
    s = normalize_swpc(rec)
    assert s.quantity == Quantity.SXR_SHORT.value
    # the short channel does not define the GOES class.
    assert s.cls is None


@pytest.mark.parametrize(
    "flux,expected_letter",
    [
        (5.0e-8, "A"),
        (3.0e-7, "B"),
        (5.5e-6, "C"),
        (2.5e-5, "M"),
        (1.1e-4, "X"),
    ],
)
def test_goes_class_boundaries(flux, expected_letter):
    assert goes_class(flux)[0] == expected_letter
    assert flare_class(flux)[0] == expected_letter


def test_goes_class_quiet_for_nonpositive():
    assert goes_class(0.0) == "Q"
    assert goes_class(-1.0) == "Q"
    assert goes_class(None) == "Q"


def test_normalize_swpc_rejects_unknown_energy():
    rec = {"time_tag": "2026-06-20T12:00:00Z", "flux": 1e-6, "energy": "9-9nm"}
    with pytest.raises(ValueError):
        normalize_swpc(rec)


def test_normalize_swpc_flags_electron_contamination():
    rec = {
        "time_tag": "2026-06-20T12:00:00Z",
        "satellite": 16,
        "flux": 1.0e-6,
        "energy": SWPC_ENERGY_LONG,
        "electron_contaminaton": True,
    }
    s = normalize_swpc(rec)
    from flarecast.types import QCBit

    assert s.qc & QCBit.SUSPECT.value


def test_normalize_generic_sxr_long_gets_class():
    rec = {"t": 1_700_000_000.0, "value": 1.2e-5}
    s = normalize_generic(
        rec, stream="solexs-sxr", quantity=Quantity.SXR_LONG.value, unit=UNIT_SXR, source="AdityaL1-SoLEXS"
    )
    assert s.cls == "M1.2"
    assert s.source == "AdityaL1-SoLEXS"


def test_normalize_generic_hxr_has_no_class():
    rec = {"t": 1_700_000_000.0, "value": 850.0}
    s = normalize_generic(
        rec, stream="hel1os-hxr", quantity=Quantity.HXR.value, unit=UNIT_HXR, source="AdityaL1-HEL1OS"
    )
    assert s.cls is None
    assert s.unit == UNIT_HXR


def test_parse_swpc_time_roundtrip():
    t = parse_swpc_time("2026-06-20T12:00:00Z")
    assert isinstance(t, float)
    # 2026-06-20T12:00:00Z is a fixed epoch; sanity bound (> 2025-01-01).
    assert t > 1_735_689_600.0


# ===========================================================================
# 4. Optional pandas accelerator path (skipped if pandas absent)
# ===========================================================================
def test_as_dataframe_optional_pandas():
    pytest.importorskip("pandas")
    from flarecast.synth.generator import as_dataframe

    samples, _ = generate_flare_lightcurves(duration_s=7200.0, cadence_s=60.0, seed=9)
    df = as_dataframe(samples)
    assert list(df.columns) == ["t", "sxr_long", "sxr_short", "hxr_8_30", "hxr_30_70"]
    assert len(df) > 0


def test_truth_to_dataframe_optional_pandas():
    pytest.importorskip("pandas")
    from flarecast.synth.generator import truth_to_dataframe

    _, truth = generate_flare_lightcurves(duration_s=86400.0, cadence_s=60.0, seed=9)
    tdf = truth_to_dataframe(truth)
    assert "goes_class" in tdf.columns
    assert len(tdf) == len(truth)
