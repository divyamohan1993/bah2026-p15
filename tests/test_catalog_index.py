"""Tests for the O(1) catalogue index + SQLite store (research 03 Section 4.5/4.7).

Inserts N events; verifies O(1) by-time-bucket point queries, range queries, the
two-bucket association scan, the class secondary index, and a full SQLite
round-trip (insert -> query_range, with the epoch-ms DDL conversion and a
``min_class`` filter). Pure stdlib (``sqlite3``); no numpy, no network.
"""

from __future__ import annotations

import random

from flarecast.catalog.index import CatalogStore, HashBucketIndex, bucket_of
from flarecast.catalog.schema import FlareEvent, new_event_id
from flarecast.constants import CATALOG_BUCKET_S


def _event(t_peak, goes_class="C3.1", soft=True, hard=False, conf=0.8):
    return FlareEvent(
        event_id=new_event_id(),
        t_start=t_peak - 50.0,
        t_peak=t_peak,
        t_end=t_peak + 200.0,
        goes_class=goes_class,
        soft={"detected": True, "peak_flux_goes_equiv": 3.1e-6, "detector_used": "SDD1"}
        if soft
        else None,
        hard={"detected": True, "peak_counts": 500.0} if hard else None,
        flags={"soft": soft, "hard": hard, "neupert_consistent": soft and hard},
        confidence=conf,
        detectors=["CUSUM"] if soft else ["FOCuS"],
    )


# ---------------------------------------------------------------------------
# bucket helper
# ---------------------------------------------------------------------------
def test_bucket_of():
    assert bucket_of(0.0, 3600.0) == 0
    assert bucket_of(3599.0, 3600.0) == 0
    assert bucket_of(3600.0, 3600.0) == 1
    assert bucket_of(7250.0, 3600.0) == 2


# ---------------------------------------------------------------------------
# HashBucketIndex
# ---------------------------------------------------------------------------
def test_index_insert_and_point_query():
    idx = HashBucketIndex(bucket_s=3600.0)
    # Three events in the first hour bucket, one in the second.
    e1 = _event(100.0)
    e2 = _event(1000.0)
    e3 = _event(3000.0)
    e4 = _event(5000.0)
    for e in (e1, e2, e3, e4):
        idx.insert(e)
    assert len(idx) == 4
    # Point query by time bucket: t=500 is in bucket 0 with e1,e2,e3.
    hits = idx.at(500.0)
    assert len(hits) == 3
    ids = {e.event_id for e in hits}
    assert {e1.event_id, e2.event_id, e3.event_id} == ids
    # t=5000 is in bucket 1 with only e4.
    assert [e.event_id for e in idx.at(5000.0)] == [e4.event_id]


def test_index_insert_n_and_bucket_query_returns_them():
    """Insert N events and confirm each is found in its own time bucket."""
    idx = HashBucketIndex(bucket_s=CATALOG_BUCKET_S)
    rng = random.Random(0)
    events = []
    n = 200
    for _ in range(n):
        t_peak = rng.uniform(0.0, 100 * CATALOG_BUCKET_S)
        e = _event(t_peak)
        events.append(e)
        idx.insert(e)
    assert len(idx) == n
    # Every inserted event must be retrievable via a by-time-bucket query at its
    # own peak time (the O(1) "what flares near T?" lookup).
    for e in events:
        got = idx.at(e.t_peak)
        assert any(g.event_id == e.event_id for g in got), (
            f"event at t_peak={e.t_peak} not found in its bucket"
        )


def test_index_range_query():
    idx = HashBucketIndex(bucket_s=3600.0)
    for t in (100.0, 1000.0, 4000.0, 8000.0, 20000.0):
        idx.insert(_event(t))
    # Range spanning the first three (0..5000 s).
    got = idx.range(0.0, 5000.0)
    peaks = sorted(e.t_peak for e in got)
    assert peaks == [100.0, 1000.0, 4000.0]
    # Range is inclusive and peak-time sorted.
    assert idx.range(8000.0, 8000.0)[0].t_peak == 8000.0


def test_index_neighbors_two_buckets():
    """The association scan window covers the current + previous bucket only."""
    idx = HashBucketIndex(bucket_s=3600.0)
    a = _event(100.0)  # bucket 0
    b = _event(3700.0)  # bucket 1
    c = _event(7300.0)  # bucket 2
    for e in (a, b, c):
        idx.insert(e)
    # Neighbors of a time in bucket 2 = buckets 2 and 1 (c and b), not a.
    ids = {e.event_id for e in idx.neighbors(7400.0)}
    assert ids == {b.event_id, c.event_id}


def test_index_class_secondary():
    idx = HashBucketIndex(bucket_s=3600.0)
    idx.insert(_event(100.0, goes_class="C3.1"))
    idx.insert(_event(1000.0, goes_class="M2.0"))
    idx.insert(_event(2000.0, goes_class="M5.0"))
    assert len(idx.by_class("M")) == 2
    assert len(idx.by_class("C")) == 1
    assert len(idx.by_class("X")) == 0


# ---------------------------------------------------------------------------
# CatalogStore (SQLite)
# ---------------------------------------------------------------------------
def test_store_roundtrip_in_memory():
    store = CatalogStore(":memory:")
    ev = _event(1100.0, goes_class="M2.5", soft=True, hard=True, conf=0.93)
    store.insert(ev)
    assert store.count() == 1
    got = store.get(ev.event_id)
    assert got is not None
    assert got.event_id == ev.event_id
    assert got.goes_class == "M2.5"
    assert abs(got.t_peak - 1100.0) < 1e-3  # epoch-ms round-trip is lossless here
    assert abs(got.confidence - 0.93) < 1e-9
    assert got.flags["soft"] is True and got.flags["hard"] is True
    assert "CUSUM" in got.detectors
    store.close()


def test_store_query_range_and_min_class():
    store = CatalogStore(":memory:")
    store.insert_many(
        [
            _event(100.0, goes_class="C3.1"),
            _event(1000.0, goes_class="M2.0"),
            _event(2000.0, goes_class="X1.0"),
            _event(50000.0, goes_class="C5.0"),
        ]
    )
    assert store.count() == 4
    # Range over the first three events (epoch seconds in; converted to ms).
    got = store.query_range(0.0, 3000.0)
    assert len(got) == 3
    assert [e.t_peak for e in got] == [100.0, 1000.0, 2000.0]  # sorted by peak
    # min_class filter: only M and above within the range.
    m_plus = store.query_range(0.0, 3000.0, min_class="M")
    classes = sorted(e.goes_class for e in m_plus)
    assert classes == ["M2.0", "X1.0"]
    store.close()


def test_store_peak_time_index_exists():
    """The DDL must create the peak_time index used for range/backtest queries."""
    store = CatalogStore(":memory:")
    rows = store._conn.execute(  # noqa: SLF001 (inspecting the schema in a test)
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "idx_cat_tpeak" in names
    assert "idx_cat_class" in names
    store.close()


def test_store_insert_n_and_query_all():
    """Insert N events and confirm a wide range query returns all of them."""
    store = CatalogStore(":memory:")
    rng = random.Random(1)
    n = 150
    peaks = []
    events = []
    for _ in range(n):
        t = rng.uniform(0.0, 1e6)
        peaks.append(t)
        events.append(_event(t))
    store.insert_many(events)
    assert store.count() == n
    got = store.query_range(-1.0, 2e6)
    assert len(got) == n
    # Returned in peak-time order.
    got_peaks = [e.t_peak for e in got]
    assert got_peaks == sorted(got_peaks)
    store.close()
