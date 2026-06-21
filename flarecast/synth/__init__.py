"""Physics-based synthetic light-curve generator (offline keystone).

Workstream 1 (ARCHITECTURE.md Appendix C / B.2). This sub-package produces
SoLEXS-like soft and HEL1OS-like hard X-ray streams plus a ground-truth event
list so the entire pipeline runs and is testable with zero network and zero
credentials (ARCHITECTURE.md Section 6).

The core (``profiles``, ``noise``, ``generator``) is **pure standard library**;
numpy/pandas are optional accelerators only. Public entry point:
:func:`generate_flare_lightcurves`.
"""

from __future__ import annotations

from .generator import (
    STREAM_HXR_HIGH,
    STREAM_HXR_LOW,
    STREAM_SXR_LONG,
    STREAM_SXR_SHORT,
    TruthEvent,
    as_dataframe,
    generate_flare_lightcurves,
    truth_to_dataframe,
)

__all__ = [
    "generate_flare_lightcurves",
    "TruthEvent",
    "as_dataframe",
    "truth_to_dataframe",
    "STREAM_SXR_LONG",
    "STREAM_SXR_SHORT",
    "STREAM_HXR_LOW",
    "STREAM_HXR_HIGH",
]
