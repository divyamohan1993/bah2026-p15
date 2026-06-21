"""Read store: in-memory hot KV mirror + SQLite catalogue (Section 7.2, B.7).

The serving layer's O(1) hot path mirrors the **edge** Cloudflare design
(ARCHITECTURE.md Section 7.1/7.2): a globally-replicated key->value cache
(Workers KV) for *instant* lookups -- ``latest:<stream>``, ``alert:latest``,
``cat:<bucket>``, ``forecast:latest`` -- backed by a queryable system of record
(D1 / SQLite) for the catalogue range / backtest path. :class:`ReadStore`
reproduces both substrates offline:

* an **in-memory dict** (``self._kv``) that is the exact KV-key analogue, so
  every ``latest`` / ``alert`` / ``at`` / ``forecast`` read is a single hash
  lookup -- **O(1)**, no scan;
* a stdlib :class:`~flarecast.catalog.index.CatalogStore` (SQLite) for the
  range query (``O(log n + k)`` via the ``t_peak`` index), plus a
  :class:`~flarecast.catalog.index.HashBucketIndex` so the by-time catalogue
  lookup (``cat:<bucket>``) is also O(1).

The class is deliberately dependency-light (stdlib + ``flarecast.catalog``):
constructing it never imports FastAPI, numpy, or pandas, so it is usable from
the CLI, the examples, and the tests without the optional ``api`` extra.
"""

from __future__ import annotations

from typing import Any

from ..catalog.index import CatalogStore, HashBucketIndex, bucket_of
from ..catalog.schema import FlareEvent
from ..constants import CATALOG_BUCKET_S
from ..types import DetectionState, FluxSample

__all__ = ["ReadStore", "kv_key_latest", "kv_key_alert", "kv_key_cat", "KV_KEY_FORECAST"]


# ---------------------------------------------------------------------------
# KV key helpers -- the exact key namespace the edge Worker uses (Section 7.1).
# ---------------------------------------------------------------------------
def kv_key_latest(stream: str) -> str:
    """KV key for the latest sample of ``stream`` (``latest:<stream>``)."""
    return f"latest:{stream}"


def kv_key_alert(stream: str | None = None) -> str:
    """KV key for the latest alert (``alert:latest`` or ``alert:<stream>``)."""
    return "alert:latest" if not stream else f"alert:{stream}"


def kv_key_cat(bucket: int) -> str:
    """KV key for the catalogue events in time ``bucket`` (``cat:<bucket>``)."""
    return f"cat:{bucket}"


#: KV key for the most recent forecast record (``forecast:latest``).
KV_KEY_FORECAST = "forecast:latest"


class ReadStore:
    """Offline read store: O(1) hot KV dict + SQLite catalogue (Appendix B.7).

    Parameters
    ----------
    db_path:
        SQLite path for the catalogue system-of-record; defaults to an
        in-memory database (``":memory:"``). The hot KV mirror is always an
        in-process dict regardless of ``db_path``.
    bucket_s:
        Time-bucket width [s] for the O(1) by-time catalogue key
        (default :data:`flarecast.constants.CATALOG_BUCKET_S`).
    """

    def __init__(
        self, db_path: str = ":memory:", bucket_s: float = CATALOG_BUCKET_S
    ) -> None:
        self.bucket_s = float(bucket_s)
        #: The KV-equivalent hot cache: key -> JSON-able value. O(1) reads.
        self._kv: dict[str, Any] = {}
        #: Per-stream latest sample (mirrors ``latest:<stream>``).
        self._streams: set[str] = set()
        #: SQLite system-of-record (range / backtest path).
        self._catalog = CatalogStore(db_path)
        #: O(1) by-time bucket index (mirrors ``cat:<bucket>``).
        self._index = HashBucketIndex(bucket_s=self.bucket_s)

    # ------------------------------------------------------------------ #
    # Writers (upsert latest sample / alert / forecast / event).
    # ------------------------------------------------------------------ #
    def put_sample(self, s: FluxSample) -> None:
        """Upsert the latest sample for its stream. O(1).

        Overwrites ``latest:<stream>`` with the newest sample (a later ``t``
        always wins; an out-of-order older sample is ignored so the hot key
        always holds the freshest reading -- the edge KV semantics).
        """
        key = kv_key_latest(s.stream)
        prev = self._kv.get(key)
        if prev is not None and float(prev.get("t", float("-inf"))) > float(s.t):
            return
        self._kv[key] = _sample_to_dict(s)
        self._streams.add(s.stream)

    def put_alert(self, alert: dict[str, Any], stream: str | None = None) -> None:
        """Upsert the latest alert (``alert:latest`` and ``alert:<stream>``). O(1).

        ``alert`` is a JSON-able dict (typically built by
        :meth:`alert_from_detection`). The global ``alert:latest`` key always
        receives the newest alert; a per-stream copy is written too when
        ``stream`` is supplied so a dashboard can scope alerts by band.
        """
        self._kv[kv_key_alert(None)] = dict(alert)
        if stream:
            self._kv[kv_key_alert(stream)] = dict(alert)

    def put_forecast(self, forecast: dict[str, Any]) -> None:
        """Upsert the latest forecast record (``forecast:latest``). O(1)."""
        self._kv[KV_KEY_FORECAST] = dict(forecast)

    def put_event(self, e: FlareEvent) -> None:
        """Insert a catalogue event into SQLite, the O(1) index, and KV. O(1)*.

        Writes the event to all three substrates: the SQLite store (range
        path), the hash-bucket index (O(1) by-time), and the ``cat:<bucket>``
        KV key (the edge by-time hot key). ``*`` the SQLite insert is
        ``O(log n)``; the KV / index writes are O(1).
        """
        self._catalog.insert(e)
        self._index.insert(e)
        b = bucket_of(e.t_peak, self.bucket_s)
        key = kv_key_cat(b)
        lst = self._kv.setdefault(key, [])
        lst.append(e.to_json())

    # ------------------------------------------------------------------ #
    # Readers (all O(1) except the explicit range query).
    # ------------------------------------------------------------------ #
    def latest(self, stream: str) -> dict[str, Any] | None:
        """Return the latest sample dict for ``stream`` (or ``None``). O(1)."""
        return self._kv.get(kv_key_latest(stream))

    def alert(self, stream: str | None = None) -> dict[str, Any] | None:
        """Return the latest alert (global, or per-stream). O(1)."""
        val = self._kv.get(kv_key_alert(stream))
        if val is None and stream:
            # Fall back to the global latest alert if no per-stream one exists.
            val = self._kv.get(kv_key_alert(None))
        return val

    def at(self, t: float, stream: str | None = None) -> dict[str, Any] | None:
        """Return the catalogue events in the time bucket of ``t``. O(1).

        This is the edge ``cat:<hour-bucket>`` lookup: hashing the time to its
        bucket makes "what flares near time T?" a single key get. Returns a
        dict ``{"bucket": <id>, "t": t, "events": [...]}`` (empty ``events`` if
        none), or ``None`` only when ``t`` is not finite/parseable.
        """
        try:
            tf = float(t)
        except (TypeError, ValueError):
            return None
        b = bucket_of(tf, self.bucket_s)
        events = self._kv.get(kv_key_cat(b), [])
        return {"bucket": b, "t": tf, "bucket_s": self.bucket_s, "events": list(events)}

    def catalogue(
        self, t_start: float, t_end: float, min_class: str | None = None
    ) -> list[dict[str, Any]]:
        """Return catalogue events with ``t_start <= t_peak <= t_end`` (range).

        The analytics path: a ``t_peak``-indexed SQLite range query
        (``O(log n + k)``), optionally filtered to events at least as intense as
        ``min_class``. Returns JSON-able event dicts, peak-time sorted.
        """
        events = self._catalog.query_range(t_start, t_end, min_class=min_class)
        return [e.to_json() for e in events]

    def latest_catalogue(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` catalogue events (newest first).

        Convenience for the dashboard's "recent flares" table -- reads from the
        O(1) in-memory index (no SQL scan) and returns the latest events by
        peak time.
        """
        events = sorted(self._index.all_events(), key=lambda e: e.t_peak, reverse=True)
        if limit and limit > 0:
            events = events[: int(limit)]
        return [e.to_json() for e in events]

    def forecast(self) -> dict[str, Any] | None:
        """Return the latest forecast record (or ``None``). O(1)."""
        return self._kv.get(KV_KEY_FORECAST)

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle.
    # ------------------------------------------------------------------ #
    def streams(self) -> list[str]:
        """Return the set of stream ids that have a latest sample. O(#streams)."""
        return sorted(self._streams)

    def health(self) -> dict[str, Any]:
        """Return a small liveness/contents summary for ``/api/health``. O(1)."""
        return {
            "status": "ok",
            "n_streams": len(self._streams),
            "streams": sorted(self._streams),
            "n_events": self._catalog.count(),
            "has_forecast": KV_KEY_FORECAST in self._kv,
            "has_alert": kv_key_alert(None) in self._kv,
            "bucket_s": self.bucket_s,
        }

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._catalog.close()

    # ------------------------------------------------------------------ #
    # Helpers to build records from pipeline objects.
    # ------------------------------------------------------------------ #
    @staticmethod
    def alert_from_detection(
        det: DetectionState, stream: str, *, kind: str = "nowcast"
    ) -> dict[str, Any]:
        """Build a JSON-able alert dict from a detector :class:`DetectionState`.

        Captures the band, the firing detectors, the GOES class / peak flux (on
        a peak), the onset time, and a severity letter for the dashboard's
        colour bands. ``kind`` is ``"nowcast"`` (onset/peak) or ``"forecast"``.
        """
        meta = det.meta or {}
        goes_class = meta.get("goes_class")
        severity = (goes_class or "")[:1].upper() if goes_class else None
        return {
            "kind": kind,
            "stream": stream,
            "band": meta.get("band"),
            "onset": bool(det.onset),
            "peak": bool(det.peak),
            "end": bool(det.end),
            "onset_time": det.onset_time,
            "statistic": det.statistic,
            "goes_class": goes_class,
            "peak_flux": meta.get("peak_flux"),
            "severity": severity,
            "detectors": list(meta.get("detectors", []) or []),
            "t": det.onset_time,
        }


def _sample_to_dict(s: FluxSample) -> dict[str, Any]:
    """JSON-able view of a :class:`FluxSample` for the KV hot cache. O(1)."""
    return {
        "stream": s.stream,
        "t": s.t,
        "value": s.value,
        "unit": s.unit,
        "source": s.source,
        "quantity": s.quantity,
        "cls": s.cls,
        "qc": s.qc,
        "meta": s.meta,
    }
