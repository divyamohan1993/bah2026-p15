"""O(1) by-time catalogue index + SQLite store (Section 4.7 / 9.4, B.5).

Two complementary structures (mirroring the edge's KV + D1 split, Section 7.2):

:class:`HashBucketIndex`
    The **O(1) hot path**. Events are bucketed by ``floor(t_peak / bucket_s)``
    into a ``dict[int, list]`` so "what flares near time T?" is a single hash
    lookup. Insert is O(1); a point query is O(1); a range query is
    O(#buckets spanned). Association only ever scans the current + previous
    bucket, keeping it O(1) per detection (research doc ``03 Section 4.5``).

:class:`CatalogStore`
    The **queryable system of record**: a stdlib :mod:`sqlite3` table mirroring
    the ``flare_catalogue`` DDL (Section 9.4) with a ``peak_time`` index for
    range / backtest queries (the analytics path, deliberately ``O(log n + k)``,
    not forced O(1)).
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from math import floor

from ..constants import CATALOG_BUCKET_S, GOES_CLASS_LADDER
from .schema import FlareEvent

__all__ = ["HashBucketIndex", "CatalogStore", "bucket_of"]


def bucket_of(t: float, bucket_s: float = CATALOG_BUCKET_S) -> int:
    """Return the integer time bucket ``floor(t / bucket_s)`` (Section 4.7). O(1)."""
    return int(floor(float(t) / float(bucket_s)))


class HashBucketIndex:
    """Hash-bucket index for O(1) by-time catalogue access (Section 4.7).

    Events are keyed by the bucket of their ``t_peak``. A small secondary index
    maps GOES class *letter* -> event ids for richer O(1) lookups.

    Parameters
    ----------
    bucket_s:
        Bucket width in seconds (default
        :data:`flarecast.constants.CATALOG_BUCKET_S`, 3600 s).
    """

    __slots__ = ("bucket_s", "_buckets", "_by_class", "_by_id")

    def __init__(self, bucket_s: float = CATALOG_BUCKET_S) -> None:
        if bucket_s <= 0.0:
            raise ValueError(f"bucket_s must be > 0, got {bucket_s!r}")
        self.bucket_s: float = float(bucket_s)
        self._buckets: dict[int, list[FlareEvent]] = defaultdict(list)
        self._by_class: dict[str, list[str]] = defaultdict(list)
        self._by_id: dict[str, FlareEvent] = {}

    def insert(self, event: FlareEvent) -> None:
        """Insert an event into its ``t_peak`` bucket. O(1)."""
        b = bucket_of(event.t_peak, self.bucket_s)
        self._buckets[b].append(event)
        self._by_id[event.event_id] = event
        letter = (event.goes_class or "")[:1].upper()
        if letter in GOES_CLASS_LADDER:
            self._by_class[letter].append(event.event_id)

    def at(self, t: float) -> list[FlareEvent]:
        """Return events in the bucket containing time ``t``. O(1).

        This is the O(1) "what flares near time T?" lookup -- a single hash of
        the time to its bucket. (Events whose peak falls in the same bucket as
        ``t`` are returned; use :meth:`range` for an exact time window.)
        """
        return list(self._buckets.get(bucket_of(t, self.bucket_s), ()))

    def range(self, t_start: float, t_end: float) -> list[FlareEvent]:
        """Return events with ``t_start <= t_peak <= t_end``. O(#buckets spanned).

        Scans only the buckets the window spans (constant work per bucket), then
        filters to the exact peak-time window.
        """
        if t_end < t_start:
            t_start, t_end = t_end, t_start
        b0 = bucket_of(t_start, self.bucket_s)
        b1 = bucket_of(t_end, self.bucket_s)
        out: list[FlareEvent] = []
        for b in range(b0, b1 + 1):
            for ev in self._buckets.get(b, ()):
                if t_start <= ev.t_peak <= t_end:
                    out.append(ev)
        out.sort(key=lambda e: e.t_peak)
        return out

    def neighbors(self, t: float) -> list[FlareEvent]:
        """Return events in the current + previous bucket of ``t``. O(1).

        This is the association scan window (research doc ``03 Section 4.5``):
        bounding it to two buckets keeps cross-band association O(1) per
        detection.
        """
        b = bucket_of(t, self.bucket_s)
        out = list(self._buckets.get(b, ()))
        out.extend(self._buckets.get(b - 1, ()))
        return out

    def by_class(self, letter: str) -> list[FlareEvent]:
        """Return events whose GOES class letter matches. O(#matches)."""
        ids = self._by_class.get(letter[:1].upper(), ())
        return [self._by_id[i] for i in ids]

    def get(self, event_id: str) -> FlareEvent | None:
        """Return the event with ``event_id`` (or ``None``). O(1)."""
        return self._by_id.get(event_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def all_events(self) -> list[FlareEvent]:
        """Return every indexed event, peak-time sorted. O(n log n)."""
        return sorted(self._by_id.values(), key=lambda e: e.t_peak)


# DDL mirrors ARCHITECTURE.md Section 9.4 exactly (epoch-ms integer times).
_DDL = """
CREATE TABLE IF NOT EXISTS flare_catalogue (
  event_id       TEXT PRIMARY KEY,
  t_start        INTEGER,
  t_peak         INTEGER,
  t_end          INTEGER,
  goes_class     TEXT,
  soft_detected  INTEGER,
  hard_detected  INTEGER,
  peak_flux_goes REAL,
  peak_counts    REAL,
  detector_used  TEXT,
  neupert_ok     INTEGER,
  confidence     REAL,
  detectors      TEXT,
  qc_bitmask     INTEGER,
  ref_catalog    TEXT,
  ref_id         TEXT,
  ref_dt_s       REAL
);
"""
_IDX_TPEAK = "CREATE INDEX IF NOT EXISTS idx_cat_tpeak ON flare_catalogue(t_peak);"
_IDX_CLASS = "CREATE INDEX IF NOT EXISTS idx_cat_class ON flare_catalogue(goes_class);"


class CatalogStore:
    """SQLite-backed catalogue store mirroring the D1 DDL (Section 9.4 / B.5).

    The same ``flare_catalogue`` schema and indices as the edge D1 store, so the
    offline analytics path is a faithful preview. Range queries use the
    ``idx_cat_tpeak`` index (``O(log n + k)`` -- the right tool for backtests,
    not forced to O(1)).

    Parameters
    ----------
    db_path:
        SQLite path; defaults to an in-memory database (``":memory:"``).
    """

    __slots__ = ("_conn",)

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        cur.execute(_DDL)
        cur.execute(_IDX_TPEAK)
        cur.execute(_IDX_CLASS)
        self._conn.commit()

    def insert(self, event: FlareEvent) -> None:
        """Insert (or replace) one event. O(log n) via the primary key."""
        cols = FlareEvent.sql_columns()
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO flare_catalogue ({','.join(cols)}) VALUES ({placeholders})"
        self._conn.execute(sql, event.to_sql_row())
        self._conn.commit()

    def insert_many(self, events: list[FlareEvent]) -> None:
        """Bulk insert events in one transaction. O(k log n)."""
        cols = FlareEvent.sql_columns()
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO flare_catalogue ({','.join(cols)}) VALUES ({placeholders})"
        self._conn.executemany(sql, [e.to_sql_row() for e in events])
        self._conn.commit()

    def query_range(
        self, t_start: float, t_end: float, min_class: str | None = None
    ) -> list[FlareEvent]:
        """Return events with ``t_start <= t_peak <= t_end`` (analytics path).

        ``t_start`` / ``t_end`` are **epoch seconds** (converted to the DDL's
        epoch-ms internally). ``min_class`` filters to events at least as intense
        as the given GOES class letter (e.g. ``"M"``). Uses the ``t_peak`` index
        -> ``O(log n + k)``.
        """
        ms0 = int(round(float(t_start) * 1000.0))
        ms1 = int(round(float(t_end) * 1000.0))
        if ms1 < ms0:
            ms0, ms1 = ms1, ms0
        rows = self._conn.execute(
            "SELECT * FROM flare_catalogue WHERE t_peak >= ? AND t_peak <= ? ORDER BY t_peak",
            (ms0, ms1),
        ).fetchall()
        events = [self._row_to_event(r) for r in rows]
        if min_class:
            min_rank = _class_rank(min_class)
            events = [e for e in events if _class_rank(e.goes_class) >= min_rank]
        return events

    def get(self, event_id: str) -> FlareEvent | None:
        """Return the event with ``event_id`` (or ``None``). O(log n)."""
        row = self._conn.execute(
            "SELECT * FROM flare_catalogue WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def count(self) -> int:
        """Number of stored events. O(1) via SQLite count."""
        return int(self._conn.execute("SELECT COUNT(*) FROM flare_catalogue").fetchone()[0])

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> FlareEvent:
        """Reconstruct a (partial) :class:`FlareEvent` from a DB row.

        The DDL keeps the flattened scalar columns, so the soft/hard sub-records
        are reconstructed with the columns that were persisted; full per-band
        timing lives in the JSON store, not the relational analytics table.
        """
        t_start = row["t_start"] / 1000.0
        t_peak = row["t_peak"] / 1000.0
        t_end = row["t_end"] / 1000.0
        soft = None
        if row["soft_detected"]:
            soft = {
                "detected": True,
                "peak_flux_goes_equiv": row["peak_flux_goes"],
                "detector_used": row["detector_used"],
            }
        hard = None
        if row["hard_detected"]:
            hard = {"detected": True, "peak_counts": row["peak_counts"]}
        flags = {
            "soft": bool(row["soft_detected"]),
            "hard": bool(row["hard_detected"]),
            "neupert_consistent": bool(row["neupert_ok"]),
        }
        try:
            detectors = json.loads(row["detectors"]) if row["detectors"] else []
        except (ValueError, TypeError):
            detectors = []
        ref_match = None
        if row["ref_catalog"] is not None:
            ref_match = {
                "catalog": row["ref_catalog"],
                "id": row["ref_id"],
                "dt_s": row["ref_dt_s"],
            }
        return FlareEvent(
            event_id=row["event_id"],
            t_start=t_start,
            t_peak=t_peak,
            t_end=t_end,
            goes_class=row["goes_class"],
            soft=soft,
            hard=hard,
            flags=flags,
            confidence=row["confidence"] if row["confidence"] is not None else 0.0,
            detectors=detectors,
            ref_match=ref_match,
        )


def _class_rank(cls: str | None) -> float:
    """Numeric severity rank of a GOES class string for ``>=`` comparisons.

    ``rank = letter_decade_index + mantissa/10`` so ``"M5"`` > ``"M2"`` >
    ``"C9"``. Returns ``-1`` for an empty/invalid class. O(1).
    """
    if not cls:
        return -1.0
    letter = cls[0].upper()
    if letter not in GOES_CLASS_LADDER:
        return -1.0
    base = float(GOES_CLASS_LADDER.index(letter))
    mant_str = cls[1:].strip()
    try:
        mant = float(mant_str) if mant_str else 1.0
    except ValueError:
        mant = 1.0
    return base + mant / 10.0
