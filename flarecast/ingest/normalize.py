"""Per-source raw-record -> :class:`FluxSample` adapters (ARCHITECTURE.md B.1).

Every fetcher normalizes its provider-specific records into the one canonical
:class:`~flarecast.types.FluxSample` so the rest of the pipeline is
source-agnostic (ARCHITECTURE.md Section 2, Section 9.1). This module holds the
adapters plus the GOES soft-X-ray class mapping (Section 4.5) used to stamp the
``cls`` field on the long-channel samples.

Pure standard library. The headline adapter is :func:`normalize_swpc` for the
NOAA SWPC GOES XRS JSON (the live nowcast anchor); :func:`normalize_generic`
covers any flat ``{time, value}`` record from other sources.

SWPC record shape (verified field semantics, ARCHITECTURE.md Section 6 /
research 05 Section 1.2)::

    {"time_tag": "2026-06-20T12:00:00Z", "satellite": 16,
     "flux": 3.1e-6, "observed_flux": 3.1e-6,
     "electron_correction": 0.0, "electron_contaminaton": false,
     "energy": "0.1-0.8nm"}

Each timestamp appears twice, once per ``energy`` band: ``"0.1-0.8nm"``
(1-8 A, the *long* channel that defines flare class -> :data:`Quantity.SXR_LONG`)
and ``"0.05-0.4nm"`` (0.5-4 A, the *short* channel -> :data:`Quantity.SXR_SHORT`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..constants import GOES_CLASS_LADDER, GOES_CLASS_THRESHOLDS_WM2, UNIT_SXR
from ..types import FluxSample, QCBit, Quantity

__all__ = [
    "normalize_swpc",
    "normalize_generic",
    "goes_class",
    "parse_swpc_time",
    "SWPC_ENERGY_LONG",
    "SWPC_ENERGY_SHORT",
]

#: SWPC ``energy`` strings for the two GOES XRS channels.
SWPC_ENERGY_LONG = "0.1-0.8nm"     # 1-8 A long channel (defines flare class)
SWPC_ENERGY_SHORT = "0.05-0.4nm"   # 0.5-4 A short channel

#: Map a SWPC ``energy`` string to (stream-suffix, quantity).
_ENERGY_MAP = {
    SWPC_ENERGY_LONG: ("long", Quantity.SXR_LONG.value),
    SWPC_ENERGY_SHORT: ("short", Quantity.SXR_SHORT.value),
}


def goes_class(flux_wm2: float | None) -> str:
    """Map a GOES 1-8 A flux [W m^-2] to its class string (ARCHITECTURE 4.5).

    Closed-form, O(1) (research 05 Section 1.2; same math as
    ``flarecast.detect.classify.classify_flux`` and
    ``flarecast.ingest.goes.flare_class``). Uses the decade thresholds in
    :data:`~flarecast.constants.GOES_CLASS_THRESHOLDS_WM2`.

    Returns ``"Q"`` for quiet / missing / non-positive flux; otherwise
    ``"<LETTER><mantissa>"`` (one decimal), or ``"A<x.x>"`` below the A floor.

    Parameters
    ----------
    flux_wm2:
        Long-channel flux in W m^-2.
    """
    if flux_wm2 is None or flux_wm2 <= 0:
        return "Q"
    for letter in reversed(GOES_CLASS_LADDER):  # X, M, C, B, A
        base = GOES_CLASS_THRESHOLDS_WM2[letter]
        if flux_wm2 >= base:
            return f"{letter}{flux_wm2 / base:.1f}"
    a_base = GOES_CLASS_THRESHOLDS_WM2["A"]
    return f"A<{flux_wm2 / a_base:.1f}"


def parse_swpc_time(time_tag: str) -> float:
    """Parse a SWPC ISO-8601 ``time_tag`` to epoch seconds UTC.

    Accepts the trailing ``Z`` (UTC) form SWPC emits, e.g.
    ``"2026-06-20T12:00:00Z"``, as well as offset-aware ISO strings. A naive
    timestamp is assumed to be UTC.

    Parameters
    ----------
    time_tag:
        ISO-8601 timestamp string.

    Returns
    -------
    Epoch seconds (float, UTC).
    """
    s = time_tag.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def normalize_swpc(record: dict[str, Any]) -> FluxSample:
    """Adapt one SWPC GOES XRS JSON record to a :class:`FluxSample` (B.1).

    The ``energy`` field selects the channel: ``"0.1-0.8nm"`` -> long
    (``SXR_LONG``, class stamped on ``cls``); ``"0.05-0.4nm"`` -> short
    (``SXR_SHORT``). The stream id is ``"goes-<sat>-<long|short>"``. The flux is
    already in W m^-2 (the GOES scale), so the canonical unit is
    :data:`~flarecast.constants.UNIT_SXR`.

    Parameters
    ----------
    record:
        One SWPC record dict (see module docstring for the schema).

    Returns
    -------
    A :class:`FluxSample`. ``cls`` is the GOES class for long-channel records
    (``None`` for the short channel, which does not define the class).

    Raises
    ------
    KeyError / ValueError
        If required fields (``time_tag``, ``energy``) are missing or the energy
        band is unrecognised -- callers parsing an external feed should catch
        these and fall back (input-hardening, ARCHITECTURE.md Section 11.2).
    """
    energy = record.get("energy")
    if energy not in _ENERGY_MAP:
        raise ValueError(f"unrecognised SWPC energy band {energy!r}")
    suffix, quantity = _ENERGY_MAP[energy]

    time_tag = record.get("time_tag")
    if not time_tag:
        raise KeyError("SWPC record missing 'time_tag'")
    t = parse_swpc_time(time_tag)

    # prefer the corrected 'flux'; fall back to 'observed_flux'.
    flux = record.get("flux")
    if flux is None:
        flux = record.get("observed_flux")
    value = float(flux) if flux is not None else float("nan")

    sat = record.get("satellite", "primary")
    stream = f"goes-{sat}-{suffix}"

    # class only meaningful for the long channel.
    cls = goes_class(value) if quantity == Quantity.SXR_LONG.value else None

    # QC: GOOD by default; flag electron contamination as SUSPECT.
    qc = QCBit.GOOD.value
    if record.get("electron_contaminaton") or record.get("electron_contamination"):
        qc |= QCBit.SUSPECT.value

    meta: dict[str, Any] = {
        "energy": energy,
        "satellite": sat,
    }
    if record.get("observed_flux") is not None:
        meta["observed_flux"] = record.get("observed_flux")
    if record.get("electron_correction") is not None:
        meta["electron_correction"] = record.get("electron_correction")

    return FluxSample(
        stream=stream,
        t=t,
        value=value,
        unit=UNIT_SXR,
        source="SWPC",
        quantity=quantity,
        cls=cls,
        qc=qc,
        meta=meta,
    )


def normalize_generic(
    record: dict[str, Any],
    stream: str,
    quantity: str,
    unit: str,
    source: str,
    *,
    time_key: str = "t",
    value_key: str = "value",
) -> FluxSample:
    """Adapt a flat ``{time, value}`` record to a :class:`FluxSample` (B.1).

    A generic adapter for sources that already provide a simple time/value pair
    (or a cached snapshot of one). The time may be given as epoch seconds
    (numeric) or an ISO-8601 string (parsed via :func:`parse_swpc_time`).

    Parameters
    ----------
    record:
        Raw record dict.
    stream:
        Canonical stream id to stamp.
    quantity:
        One of the :class:`~flarecast.types.Quantity` values (as a string).
    unit:
        Canonical unit string for the quantity.
    source:
        Provider name (e.g. ``"AdityaL1-SoLEXS"``).
    time_key:
        Key holding the timestamp (default ``"t"``).
    value_key:
        Key holding the measurement (default ``"value"``).

    Returns
    -------
    A :class:`FluxSample` (``cls`` stamped only when ``quantity`` is
    ``SXR_LONG`` and the unit is the GOES scale).
    """
    raw_t = record.get(time_key)
    if raw_t is None:
        raise KeyError(f"record missing time key {time_key!r}")
    if isinstance(raw_t, str):
        t = parse_swpc_time(raw_t)
    else:
        t = float(raw_t)

    raw_v = record.get(value_key)
    value = float(raw_v) if raw_v is not None else float("nan")

    cls = None
    if quantity == Quantity.SXR_LONG.value and unit == UNIT_SXR:
        cls = goes_class(value)

    return FluxSample(
        stream=stream,
        t=t,
        value=value,
        unit=unit,
        source=source,
        quantity=quantity,
        cls=cls,
        qc=QCBit.GOOD.value,
        meta={k: v for k, v in record.items() if k not in (time_key, value_key)} or None,
    )
