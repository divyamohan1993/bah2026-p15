"""Shared pytest fixtures (ARCHITECTURE.md Appendix C, Workstream 6).

These fixtures give every workstream's tests a common, **fully offline**
(no network, no credentials, fixed seed) source of small synthetic inputs so
the suite is deterministic and fast. They are intentionally import-safe: only
the pure-stdlib parts of ``flarecast`` (synth core, dataclasses) are touched at
collection time, so ``conftest`` import never pulls numpy/torch/network.

Fixtures
--------
``synthetic_streams``
    A small soft+hard :class:`~flarecast.types.FluxSample` corpus from
    :func:`flarecast.synth.generate_flare_lightcurves` (fixed seed), split into
    convenient per-band lists.
``tiny_catalogue``
    A few hand-built :class:`~flarecast.catalog.schema.FlareEvent` records
    covering the C / M / X classes and the soft-only / both-bands cases.
``golden_cusum_input``
    The fixed input float vector that backs the cross-substrate parity golden
    (``tests/golden/cusum_golden.json``) -- handy for tests that want a
    deterministic CUSUM driving sequence without re-deriving it.
"""

from __future__ import annotations

import pytest

from flarecast.catalog.schema import FlareEvent
from flarecast.synth import generate_flare_lightcurves
from flarecast.types import FluxSample

# Stream id constants (mirror flarecast.synth.generator).
_SXR_LONG = "solexs-sxr-long"
_SXR_SHORT = "solexs-sxr-short"
_HXR_LOW = "hel1os-hxr-8-30keV"
_HXR_HIGH = "hel1os-hxr-30-70keV"

#: Fixed seed so the synthetic corpus is byte-for-byte reproducible.
SYNTH_SEED = 20260620


@pytest.fixture(scope="session")
def synthetic_streams() -> dict[str, list[FluxSample]]:
    """Small, deterministic soft+hard ``FluxSample`` streams (fixed seed).

    Returns a dict with keys:

    ``all``
        Every sample (all four streams), sorted as the generator emits them.
    ``soft_long`` / ``soft_short``
        SoLEXS soft-band samples (1-8 A long, 0.5-4 A short).
    ``hard_low`` / ``hard_high``
        HEL1OS hard-band count streams (8-30 keV, 30-70 keV).
    ``truth``
        The ground-truth ``list[TruthEvent]`` (start/peak/end/class per flare).

    Short (~20 min at 1 s) with a couple of flares so it is fast yet exercises
    onset/peak/decay across both bands. Session-scoped: generated once.
    """
    samples, truth = generate_flare_lightcurves(
        duration_s=1200.0,
        cadence_s=1.0,
        n_flares=2,
        neupert=True,
        noise=True,
        gaps=False,
        spikes=False,
        seed=SYNTH_SEED,
    )
    by_stream: dict[str, list[FluxSample]] = {
        _SXR_LONG: [],
        _SXR_SHORT: [],
        _HXR_LOW: [],
        _HXR_HIGH: [],
    }
    for s in samples:
        by_stream.setdefault(s.stream, []).append(s)
    return {
        "all": samples,
        "soft_long": by_stream[_SXR_LONG],
        "soft_short": by_stream[_SXR_SHORT],
        "hard_low": by_stream[_HXR_LOW],
        "hard_high": by_stream[_HXR_HIGH],
        "truth": truth,
    }


@pytest.fixture
def tiny_catalogue() -> list[FlareEvent]:
    """A few representative master-catalogue :class:`FlareEvent` records.

    Covers a C-class soft-only event, an M-class both-bands Neupert-consistent
    event, and an X-class saturated event -- enough to exercise serialization,
    SQL-row mapping, indexing, and class queries without any heavy machinery.
    Times are epoch seconds UTC; ids are stable (not random) so assertions can
    reference them.
    """
    return [
        FlareEvent(
            event_id="evt-c-0001",
            t_start=1_000.0,
            t_peak=1_180.0,
            t_end=1_600.0,
            goes_class="C3.1",
            soft={
                "detected": True,
                "t_start": 1_000.0,
                "t_peak": 1_180.0,
                "t_end": 1_600.0,
                "peak_flux_native": 1.2e3,
                "peak_flux_goes_equiv": 3.1e-6,
                "detector_used": "SDD1",
                "saturated": False,
            },
            hard=None,
            flags={"soft": True, "hard": False, "neupert_consistent": False},
            confidence=0.62,
            detectors=["CUSUM"],
        ),
        FlareEvent(
            event_id="evt-m-0002",
            t_start=5_000.0,
            t_peak=5_240.0,
            t_end=5_900.0,
            goes_class="M2.5",
            soft={
                "detected": True,
                "t_start": 5_000.0,
                "t_peak": 5_240.0,
                "t_end": 5_900.0,
                "peak_flux_native": 4.8e4,
                "peak_flux_goes_equiv": 2.5e-5,
                "detector_used": "SDD1",
                "saturated": False,
            },
            hard={
                "detected": True,
                "t_start": 4_950.0,
                "t_peak": 5_120.0,
                "peak_counts": 850.0,
                "energy_band": "8-30keV",
            },
            flags={"soft": True, "hard": True, "neupert_consistent": True},
            confidence=0.93,
            detectors=["CUSUM", "FOCuS", "threshold"],
            ref_match={"catalog": "GOES/HEK", "id": "SOL-demo-M2.5", "dt_s": 41.0},
        ),
        FlareEvent(
            event_id="evt-x-0003",
            t_start=9_000.0,
            t_peak=9_300.0,
            t_end=10_200.0,
            goes_class="X1.0",
            soft={
                "detected": True,
                "t_start": 9_000.0,
                "t_peak": 9_300.0,
                "t_end": 10_200.0,
                "peak_flux_native": 1.5e5,
                "peak_flux_goes_equiv": 1.0e-4,
                "detector_used": "SDD2",
                "saturated": True,
            },
            hard={
                "detected": True,
                "t_start": 8_950.0,
                "t_peak": 9_150.0,
                "peak_counts": 5200.0,
                "energy_band": "8-30keV",
            },
            flags={
                "soft": True,
                "hard": True,
                "neupert_consistent": True,
                "data_gap_during": False,
                "spike_rejected": 2,
            },
            confidence=0.98,
            detectors=["CUSUM", "FOCuS"],
        ),
    ]


@pytest.fixture
def golden_cusum_input() -> list[float]:
    """The fixed CUSUM driving vector that backs the parity golden.

    Imported lazily from the golden generator so there is a single definition of
    the contract input (``tests/golden/generate_cusum_golden.py``).
    """
    from tests.golden.generate_cusum_golden import build_input_vector

    return build_input_vector()
