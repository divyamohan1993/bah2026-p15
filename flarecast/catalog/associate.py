"""Neupert-aware soft+hard association into master events (Section 4.6, B.5).

Detection runs independently per band; this module associates a soft (SoLEXS /
GOES-long) detection with a hard (HEL1OS) detection into one physical flare using
the **Neupert prior** (research doc ``03 Section 4.1/4.2``):

* hard X-rays **lead** soft X-rays -- ``HXR onset <= SXR onset`` and
  ``HXR peak < SXR peak`` (lead of seconds-minutes);
* so the association window is **asymmetric**: wide (``w_lead_s``, ~5 min) on the
  hard-before-soft side, wide (``w_lag_s``, ~10-15 min) on the long-soft-decay
  side.

:func:`match_score` returns a ``MATCH_SCORE`` in ``[0, 1]`` combining temporal
proximity and Neupert peak-lag consistency; pairs scoring at least
``tau_match`` are merged. :class:`Associator` is the streaming front-end that
maintains a small set of open events (scanning only recent ones -> O(1) per
detection) and emits closed :class:`FlareEvent` records.
"""

from __future__ import annotations

from typing import Any

from ..constants import (
    ASSOC_TAU_MATCH,
    ASSOC_W_LAG_S,
    ASSOC_W_LEAD_S,
)
from ..detect.classify import classify_flux
from ..types import DetectionState
from .schema import FlareEvent, new_event_id

__all__ = ["match_score", "Associator"]


def _peak_time(det: DetectionState) -> float | None:
    """Best available peak time of a detection (meta ``t_peak`` else onset). O(1)."""
    if det.meta is not None:
        tp = det.meta.get("t_peak")
        if isinstance(tp, (int, float)):
            return float(tp)
    return det.onset_time


def _onset_time(det: DetectionState) -> float | None:
    """Onset time of a detection (``onset_time`` else meta ``t_start``). O(1)."""
    if det.onset_time is not None:
        return float(det.onset_time)
    if det.meta is not None:
        ts = det.meta.get("t_start")
        if isinstance(ts, (int, float)):
            return float(ts)
    return None


def match_score(
    soft: DetectionState,
    hard: DetectionState,
    w_lead_s: float = ASSOC_W_LEAD_S,
    w_lag_s: float = ASSOC_W_LAG_S,
) -> float:
    """Neupert-aware MATCH_SCORE for a soft+hard pair in ``[0, 1]`` (Section 4.6).

    The score blends two terms:

    * **Onset proximity** with an *asymmetric* tolerance -- the hard onset is
      expected at or before the soft onset, so a hard-before-soft offset is
      tolerated out to ``w_lead_s`` while a hard-*after*-soft offset is tolerated
      only out to the shorter implied window (a soft-leads-hard ordering is
      un-Neupert-like and decays faster). The proximity term is
      ``max(0, 1 - dt / window)`` for the applicable window.
    * **Peak-lag consistency** -- a bonus when ``t_peak^HXR < t_peak^SXR``
      (the canonical Neupert ordering), zero otherwise.

    Returns ``0.0`` if either onset time is unavailable. O(1).

    Parameters
    ----------
    soft, hard:
        The soft- and hard-band detections to score.
    w_lead_s:
        Tolerance (s) on the hard-before-soft side (default
        :data:`flarecast.constants.ASSOC_W_LEAD_S`).
    w_lag_s:
        Tolerance (s) on the long soft-decay side (default
        :data:`flarecast.constants.ASSOC_W_LAG_S`).
    """
    s_on = _onset_time(soft)
    h_on = _onset_time(hard)
    if s_on is None or h_on is None:
        return 0.0

    # Signed lead: positive when hard precedes soft (the Neupert expectation).
    lead = s_on - h_on
    if lead >= 0.0:
        # Hard at/before soft: tolerated out to w_lead_s.
        window = w_lead_s
        dt = lead
    else:
        # Hard after soft: less Neupert-like; tolerate out to the (shorter) lag
        # window but treat soft-leads-hard as the off-prior direction.
        window = w_lag_s
        dt = -lead
    if window <= 0.0:
        proximity = 1.0 if dt == 0.0 else 0.0
    else:
        proximity = max(0.0, 1.0 - dt / window)

    # Peak-lag consistency bonus (Neupert: HXR peak precedes SXR peak).
    s_pk = _peak_time(soft)
    h_pk = _peak_time(hard)
    neupert = 0.0
    if s_pk is not None and h_pk is not None and h_pk < s_pk:
        neupert = 1.0

    # Weighted blend: proximity dominates, Neupert ordering is corroboration.
    return 0.7 * proximity + 0.3 * neupert


class _OpenEvent:
    """Mutable builder accumulating one band's detections before emission."""

    __slots__ = ("soft", "hard", "t_first", "t_last", "spike_rejected", "data_gap")

    def __init__(self) -> None:
        self.soft: dict[str, Any] | None = None
        self.hard: dict[str, Any] | None = None
        self.t_first: float = float("inf")
        self.t_last: float = float("-inf")
        self.spike_rejected: int = 0
        self.data_gap: bool = False


class Associator:
    """Streaming Neupert-aware soft+hard associator (research doc ``03 Section 4.2``).

    Feed it per-band detections via :meth:`add`; it maintains a small set of open
    events, pairs soft and hard detections whose :func:`match_score` reaches
    ``tau_match``, and returns a closed :class:`FlareEvent` when an event is
    complete (its soft band reached ``end``, or it ages out of the association
    window). Because only events within the lead/lag window are ever considered,
    the per-detection work is O(1).

    Parameters
    ----------
    tau_match:
        MATCH_SCORE acceptance threshold (default
        :data:`flarecast.constants.ASSOC_TAU_MATCH`).
    w_lead_s, w_lag_s:
        Asymmetric association windows (defaults from constants).
    """

    def __init__(
        self,
        tau_match: float = ASSOC_TAU_MATCH,
        **windows: float,
    ) -> None:
        self.tau_match: float = float(tau_match)
        self.w_lead_s: float = float(windows.get("w_lead_s", ASSOC_W_LEAD_S))
        self.w_lag_s: float = float(windows.get("w_lag_s", ASSOC_W_LAG_S))
        # Pending per-band detections awaiting a partner (kept as DetectionState
        # snapshots with their representative time).
        self._pending_soft: list[tuple[float, DetectionState]] = []
        self._pending_hard: list[tuple[float, DetectionState]] = []

    def add(self, det: DetectionState, band: str, t: float) -> FlareEvent | None:
        """Add a band detection; emit a :class:`FlareEvent` when one closes.

        Only ``onset`` detections open/extend associations here (peak/end refine
        the soft sub-record). Returns the merged event when an association
        completes, else ``None``. O(1) (scans only in-window pending partners).

        Parameters
        ----------
        det:
            The detection (a :class:`DetectionState`).
        band:
            ``"soft"`` or ``"hard"``.
        t:
            The detection's representative time (epoch seconds UTC).
        """
        band = band.lower()
        if not det.onset:
            return None
        t = float(t)

        if band == "soft":
            partner = self._pop_best_partner(det, self._pending_hard, soft_is_self=True)
            if partner is not None:
                return self._emit(soft=det, hard=partner)
            self._pending_soft.append((t, det))
            self._evict(self._pending_soft, t)
            return None

        if band == "hard":
            partner = self._pop_best_partner(det, self._pending_soft, soft_is_self=False)
            if partner is not None:
                return self._emit(soft=partner, hard=det)
            self._pending_hard.append((t, det))
            self._evict(self._pending_hard, t)
            return None

        raise ValueError(f"band must be 'soft' or 'hard', got {band!r}")

    def flush(self) -> list[FlareEvent]:
        """Emit all still-pending single-band detections as solo events.

        Called at stream end so HXR-only / SXR-only detections are not lost
        (e.g. a non-thermal-dominated HXR flare GOES misses). O(#pending).
        """
        out: list[FlareEvent] = []
        for _, det in self._pending_soft:
            out.append(self._emit(soft=det, hard=None))
        for _, det in self._pending_hard:
            out.append(self._emit(soft=None, hard=det))
        self._pending_soft.clear()
        self._pending_hard.clear()
        return out

    # -- internals ---------------------------------------------------------
    def _pop_best_partner(
        self,
        det: DetectionState,
        pending: list[tuple[float, DetectionState]],
        soft_is_self: bool,
    ) -> DetectionState | None:
        """Find and remove the best-scoring in-window partner, if any. O(#pending).

        ``#pending`` is bounded by the (short) association window, so this is
        effectively O(1) per detection.
        """
        best_i = -1
        best_score = self.tau_match  # must reach the threshold to associate
        for i, (_, other) in enumerate(pending):
            if soft_is_self:
                score = match_score(det, other, self.w_lead_s, self.w_lag_s)
            else:
                score = match_score(other, det, self.w_lead_s, self.w_lag_s)
            if score >= best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            return pending.pop(best_i)[1]
        return None

    def _evict(self, pending: list[tuple[float, DetectionState]], now: float) -> None:
        """Drop pending detections older than the widest association window. O(n)."""
        horizon = now - max(self.w_lead_s, self.w_lag_s)
        pending[:] = [(t, d) for (t, d) in pending if t >= horizon]

    def _emit(
        self,
        soft: DetectionState | None,
        hard: DetectionState | None,
    ) -> FlareEvent:
        """Build a :class:`FlareEvent` from a matched (or solo) pair. O(1)."""
        soft_rec = self._soft_subrecord(soft)
        hard_rec = self._hard_subrecord(hard)

        onsets = [
            x
            for x in (
                _onset_time(soft) if soft else None,
                _onset_time(hard) if hard else None,
            )
            if x is not None
        ]
        t_start = min(onsets) if onsets else 0.0

        # Canonical peak/end come from the soft band when present.
        s_pk = _peak_time(soft) if soft else None
        t_peak = s_pk if s_pk is not None else t_start
        t_end = (
            soft.meta.get("t_end")
            if soft and soft.meta and isinstance(soft.meta.get("t_end"), (int, float))
            else t_peak
        )

        goes_class = soft_rec.get("goes_class") if soft_rec else None
        if goes_class is None and soft_rec and soft_rec.get("peak_flux_goes_equiv"):
            goes_class = classify_flux(soft_rec["peak_flux_goes_equiv"])
        if goes_class is None:
            goes_class = "A0.0"

        neupert_ok = False
        if soft and hard:
            s_pk2 = _peak_time(soft)
            h_pk2 = _peak_time(hard)
            if s_pk2 is not None and h_pk2 is not None:
                neupert_ok = h_pk2 < s_pk2

        flags = {
            "soft": soft is not None,
            "hard": hard is not None,
            "neupert_consistent": neupert_ok,
            "data_gap_during": bool(
                (soft.meta or {}).get("data_gap") if soft and soft.meta else False
            )
            or bool((hard.meta or {}).get("data_gap") if hard and hard.meta else False),
            "spike_rejected": int(
                ((soft.meta or {}).get("spike_rejected", 0) if soft and soft.meta else 0)
                + ((hard.meta or {}).get("spike_rejected", 0) if hard and hard.meta else 0)
            ),
            "sxr_weak": hard is not None and soft is None,
        }

        detectors: list[str] = []
        for d in (soft, hard):
            if d and d.meta:
                for name in d.meta.get("detectors", []) or []:
                    if name not in detectors:
                        detectors.append(name)

        return FlareEvent(
            event_id=new_event_id(),
            t_start=float(t_start),
            t_peak=float(t_peak),
            t_end=float(t_end),
            goes_class=goes_class,
            soft=soft_rec,
            hard=hard_rec,
            flags=flags,
            confidence=0.0,  # filled by confidence.fuse_confidence downstream
            detectors=detectors,
        )

    @staticmethod
    def _soft_subrecord(soft: DetectionState | None) -> dict[str, Any] | None:
        """Build the soft sub-record dict from a detection. O(1)."""
        if soft is None:
            return None
        meta = soft.meta or {}
        rec: dict[str, Any] = {
            "detected": True,
            "t_start": _onset_time(soft),
            "t_peak": _peak_time(soft),
            "t_end": meta.get("t_end"),
            "detector_used": meta.get("detector_used", meta.get("sdd", "SDD1")),
            "saturated": bool(meta.get("saturated", False)),
        }
        if "peak_flux" in meta:
            rec["peak_flux_native"] = meta["peak_flux"]
            rec["peak_flux_goes_equiv"] = meta["peak_flux"]
        if "goes_class" in meta:
            rec["goes_class"] = meta["goes_class"]
        return rec

    @staticmethod
    def _hard_subrecord(hard: DetectionState | None) -> dict[str, Any] | None:
        """Build the hard sub-record dict from a detection. O(1)."""
        if hard is None:
            return None
        meta = hard.meta or {}
        return {
            "detected": True,
            "t_start": _onset_time(hard),
            "t_peak": _peak_time(hard),
            "peak_counts": meta.get("peak_counts"),
            "energy_band": meta.get("energy_band", "8-30keV"),
        }
