"""Master-catalogue flare-event record (ARCHITECTURE.md Section 9.3, B.5).

:class:`FlareEvent` is the single physical-flare record produced by associating
the per-band detections (soft SoLEXS/GOES + hard HEL1OS) into one event. It
carries the canonical start/peak/end, the GOES class, per-band sub-records, the
fused confidence, the firing detectors, and an optional reference-catalogue
match. The (de)serialization helpers mirror the JSON shape in Section 9.3 and
the SQLite/D1 DDL in Section 9.4 exactly so the offline and edge substrates
agree byte-for-byte.

Times on the dataclass are **epoch seconds UTC** (the pipeline's working unit);
:meth:`FlareEvent.to_sql_row` converts to **epoch milliseconds** to match the
``flare_catalogue`` DDL (Section 9.4).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = ["FlareEvent", "new_event_id"]


def new_event_id() -> str:
    """Return a fresh stable event id (UUID4 hex string). O(1)."""
    return uuid.uuid4().hex


@dataclass(slots=True)
class FlareEvent:
    """One physical flare in the master catalogue (Section 9.3 / B.5).

    Attributes
    ----------
    event_id:
        Stable unique key (UUID hex).
    t_start, t_peak, t_end:
        Canonical start / peak / end, **epoch seconds UTC** (t_earth frame).
        ``t_start`` is the earliest band onset (CUSUM last-reset); ``t_peak`` /
        ``t_end`` are the soft-band peak and FSM-midpoint end.
    goes_class:
        GOES class string from the soft GOES-equivalent peak, e.g. ``"M2.5"``.
    soft:
        SoLEXS sub-record dict (``None`` if not detected): keys ``detected``,
        ``t_start``, ``t_peak``, ``t_end``, ``peak_flux_native``,
        ``peak_flux_goes_equiv``, ``detector_used``, ``saturated``.
    hard:
        HEL1OS sub-record dict (``None`` if not detected): keys ``detected``,
        ``t_start``, ``t_peak``, ``peak_counts``, ``energy_band``.
    flags:
        Dict of booleans/counters: ``soft``, ``hard``, ``neupert_consistent``,
        ``data_gap_during``, ``spike_rejected``, ``sxr_weak``.
    confidence:
        Noisy-OR fused detector confidence in ``[0, 1]``.
    detectors:
        Names of the detectors that fired, e.g. ``["CUSUM", "FOCuS"]``.
    ref_match:
        Optional reference-catalogue match dict (``catalog``, ``id``, ``dt_s``),
        filled post-hoc; ``None`` until matched.
    """

    event_id: str
    t_start: float
    t_peak: float
    t_end: float
    goes_class: str
    soft: dict[str, Any] | None
    hard: dict[str, Any] | None
    flags: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    detectors: list[str] = field(default_factory=list)
    ref_match: dict[str, Any] | None = None

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict mirroring Section 9.3. O(1)."""
        return {
            "event_id": self.event_id,
            "t_start": self.t_start,
            "t_peak": self.t_peak,
            "t_end": self.t_end,
            "goes_class": self.goes_class,
            "soft": self.soft,
            "hard": self.hard,
            "flags": dict(self.flags),
            "confidence": self.confidence,
            "detectors": list(self.detectors),
            "ref_match": self.ref_match,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FlareEvent:
        """Reconstruct a :class:`FlareEvent` from a :meth:`to_json` dict. O(1)."""
        return cls(
            event_id=d["event_id"],
            t_start=float(d["t_start"]),
            t_peak=float(d["t_peak"]),
            t_end=float(d["t_end"]),
            goes_class=d["goes_class"],
            soft=d.get("soft"),
            hard=d.get("hard"),
            flags=dict(d.get("flags") or {}),
            confidence=float(d.get("confidence", 0.0)),
            detectors=list(d.get("detectors") or []),
            ref_match=d.get("ref_match"),
        )

    def to_sql_row(self) -> tuple:
        """Return a row tuple matching the ``flare_catalogue`` DDL (Section 9.4).

        Column order::

            (event_id, t_start, t_peak, t_end, goes_class, soft_detected,
             hard_detected, peak_flux_goes, peak_counts, detector_used,
             neupert_ok, confidence, detectors, qc_bitmask, ref_catalog,
             ref_id, ref_dt_s)

        Times are converted to **epoch milliseconds** (the DDL's integer unit);
        ``detectors`` is JSON-encoded; ``soft``/``hard`` sub-records are flattened
        to the scalar columns the DDL keeps. O(1).
        """
        soft = self.soft or {}
        hard = self.hard or {}
        flags = self.flags or {}
        ref = self.ref_match or {}
        return (
            self.event_id,
            _to_ms(self.t_start),
            _to_ms(self.t_peak),
            _to_ms(self.t_end),
            self.goes_class,
            1 if flags.get("soft") or soft.get("detected") else 0,
            1 if flags.get("hard") or hard.get("detected") else 0,
            _opt_float(soft.get("peak_flux_goes_equiv")),
            _opt_float(hard.get("peak_counts")),
            soft.get("detector_used"),
            1 if flags.get("neupert_consistent") else 0,
            float(self.confidence),
            json.dumps(list(self.detectors)),
            _qc_bitmask(flags),
            ref.get("catalog"),
            ref.get("id"),
            _opt_float(ref.get("dt_s")),
        )

    @classmethod
    def sql_columns(cls) -> tuple[str, ...]:
        """Column names matching :meth:`to_sql_row` / the DDL (Section 9.4)."""
        return (
            "event_id",
            "t_start",
            "t_peak",
            "t_end",
            "goes_class",
            "soft_detected",
            "hard_detected",
            "peak_flux_goes",
            "peak_counts",
            "detector_used",
            "neupert_ok",
            "confidence",
            "detectors",
            "qc_bitmask",
            "ref_catalog",
            "ref_id",
            "ref_dt_s",
        )


def _to_ms(t_seconds: float) -> int:
    """Epoch seconds -> epoch milliseconds (DDL integer unit). O(1)."""
    return int(round(float(t_seconds) * 1000.0))


def _opt_float(v: Any) -> float | None:
    """Coerce to float or pass through ``None``. O(1)."""
    return None if v is None else float(v)


def _qc_bitmask(flags: dict[str, Any]) -> int:
    """Derive the QC bitmask integer from event flags (Section 9.4 bits). O(1).

    Sets the ``data_gap`` (128) and ``spike_rejected`` (256) bits when the
    corresponding flags indicate those conditions occurred during the event.
    """
    bits = 0
    if flags.get("data_gap_during"):
        bits |= 128
    if flags.get("spike_rejected"):
        bits |= 256
    return bits
