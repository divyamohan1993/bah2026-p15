"""Shared types for the ``flarecast`` package.

This module is the single source of truth for the cross-workstream data
contracts described in ARCHITECTURE.md Appendix B.0 and Section 9. It is
imported by *every* other workstream (ingest, fusion, detect, catalog,
forecast, api, cli), so it is deliberately kept **pure standard library** --
no numpy / pandas / scipy imports at module top -- so that ``import flarecast``
and ``from flarecast.types import FluxSample`` always succeed even in a
minimal / offline environment.

The public symbols below are *normative*: keep their names, fields, field
order, and defaults exactly as written so modules compose without collisions.
See ARCHITECTURE.md "Appendix B.0 -- Shared types".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any

__all__ = [
    "Quantity",
    "QCFlag",
    "QCBit",
    "DetectionPhase",
    "FluxSample",
    "DetectionState",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Quantity(str, Enum):
    """Physical quantity a sample estimates (ARCHITECTURE.md Section 9.2).

    Fusion only ever combines sources that estimate the *same* quantity
    (after cross-calibration). Different quantities are features, not
    redundant measurements. Subclassing ``str`` makes members JSON-friendly
    and directly comparable to the plain strings stored on ``FluxSample`` /
    ``FusionRecord``.
    """

    SXR_LONG = "SXR_LONG"
    SXR_SHORT = "SXR_SHORT"
    HXR = "HXR"
    EUV = "EUV"
    MAGSCALAR = "MAGSCALAR"
    RADIO = "RADIO"
    PROTON = "PROTON"


class QCFlag(str, Enum):
    """Per-sample quality-control state (ARCHITECTURE.md Section 3.11 / 9.4).

    Ordered worst-to-best handling: ``GOOD`` (full weight) ->
    ``INTERPOLATED`` (inflated sigma) -> ``FILLED`` (filler sigma combined
    with transfer residual, provenance recorded) -> ``SUSPECT`` (gated but
    logged) -> ``BAD`` (excluded). This enum is the human-readable label used
    on ``FusionRecord.qc_flag``; the integer ``FluxSample.qc`` field carries
    the coexisting-conditions bitmask (see :class:`QCBit`).
    """

    GOOD = "GOOD"
    INTERPOLATED = "INTERPOLATED"
    FILLED = "FILLED"
    SUSPECT = "SUSPECT"
    BAD = "BAD"


class QCBit(IntFlag):
    """QC bitmask values for ``FluxSample.qc`` (ARCHITECTURE.md Section 9.4).

    Stored as a bitmask so multiple conditions coexist on one sample, e.g.
    ``FILLED | NEAR_SAA``. The low five bits mirror the :class:`QCFlag`
    states; the high bits record specific hazards. These integer values are
    shared verbatim by the edge (Cloudflare D1) and offline substrates and
    MUST match the DDL comment in ARCHITECTURE.md Section 9.4.
    """

    GOOD = 1
    INTERPOLATED = 2
    FILLED = 4
    SUSPECT = 8
    BAD = 16
    NEAR_SAA = 32
    SATURATED = 64
    DATA_GAP = 128
    SPIKE_REJECTED = 256


class DetectionPhase(str, Enum):
    """Phase of the soft-band flare finite-state machine (ARCHITECTURE.md 4.4).

    Convenience enum for the GOES-style start/peak/end FSM in
    ``flarecast.detect.fsm``. ``DetectionState`` exposes phase transitions as
    the boolean flags ``onset`` / ``peak`` / ``end``; this enum lets callers
    that prefer an explicit state machine name the resting/active phases.
    """

    QUIET = "QUIET"
    RISING = "RISING"
    PEAK = "PEAK"
    DECAYING = "DECAYING"


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FluxSample:
    """Canonical ingest record: one sample of one stream (ARCHITECTURE.md 9.1).

    Every fetcher in ``flarecast.ingest`` and the synthetic generator in
    ``flarecast.synth`` normalize their output to this record so the rest of
    the pipeline is source-agnostic.

    Attributes
    ----------
    stream:
        Stream identifier, e.g. ``"goes-primary-long"``, ``"solexs-sxr"``,
        ``"hel1os-hxr-8-30keV"``.
    t:
        Epoch seconds UTC (observed on-board time).
    value:
        Measurement in the canonical unit for the quantity.
    unit:
        Canonical unit string, e.g. ``"W m^-2"`` or ``"counts/s"``.
    source:
        Provider, e.g. ``"SWPC"``, ``"AdityaL1-SoLEXS"``, ``"synth"``.
    quantity:
        One of the :class:`Quantity` values (stored as a plain string), e.g.
        ``"SXR_LONG"``, ``"SXR_SHORT"``, ``"HXR"``.
    cls:
        Derived GOES flare class, e.g. ``"C3.1"`` (``None`` if not derived).
    qc:
        QC bitmask (see :class:`QCBit`); ``0`` means "unset / not yet QC'd".
    meta:
        Free-form per-sample metadata (detector id, energy band, ...).
    """

    stream: str
    t: float
    value: float
    unit: str
    source: str
    quantity: str
    cls: str | None = None
    qc: int = 0
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class DetectionState:
    """Result returned by every detector ``.update()`` (ARCHITECTURE.md B.0).

    Streaming detectors (CUSUM, Poisson-FOCuS, adaptive threshold, the FSM,
    and the composed band detectors) all return this record from their O(1)
    ``update`` step so they compose uniformly.

    Attributes
    ----------
    onset:
        ``True`` if an onset fired on this sample.
    in_event:
        ``True`` while currently inside a flare.
    statistic:
        The detector's running statistic (CUSUM ``S``, FOCuS statistic, ...).
    onset_time:
        MLE / edge onset time (epoch seconds UTC) when ``onset`` fires, else
        ``None``.
    peak:
        ``True`` if the FSM declared the peak on this sample.
    end:
        ``True`` if the FSM declared the flare end on this sample.
    meta:
        Free-form detector metadata.
    """

    onset: bool
    in_event: bool
    statistic: float
    onset_time: float | None = None
    peak: bool = False
    end: bool = False
    meta: dict[str, Any] | None = None
