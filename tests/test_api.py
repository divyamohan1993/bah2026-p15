"""Serving-layer tests: O(1) read routes + static dashboard (Appendix A / B.7).

Exercises :func:`flarecast.api.app.create_app` through a FastAPI
``TestClient``: the dashboard is served at ``/`` and the O(1) read endpoints
(``/api/latest``, ``/api/alert``, ``/api/at``, ``/api/forecast``,
``/api/catalogue``, ``/api/health``) return the expected JSON shapes the
dashboard consumes (``flarecast/dashboard/app.js``).

FastAPI is optional, so the whole module is skipped when it is not installed
(``pytest.importorskip("fastapi")``); the core offline suite never needs it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from flarecast.api.app import create_app  # noqa: E402
from flarecast.api.store import ReadStore  # noqa: E402
from flarecast.catalog.schema import FlareEvent  # noqa: E402


@pytest.fixture
def client_empty() -> TestClient:
    """A TestClient over an empty (unseeded) store + the real dashboard mount."""
    store = ReadStore()
    return TestClient(create_app(store))


@pytest.fixture
def client_seeded(tiny_catalogue: list[FlareEvent]) -> TestClient:
    """A TestClient over a store seeded with a sample, alert, forecast, events."""
    from flarecast.types import FluxSample

    store = ReadStore()
    store.put_sample(
        FluxSample(
            stream="solexs-sxr-long",
            t=5_240.0,
            value=2.5e-5,
            unit="W m^-2",
            source="synth",
            quantity="SXR_LONG",
            cls="M2.5",
        )
    )
    store.put_sample(
        FluxSample(
            stream="hel1os-hxr-8-30keV",
            t=5_240.0,
            value=850.0,
            unit="counts/s",
            source="synth",
            quantity="HXR",
        )
    )
    store.put_alert(
        {
            "kind": "nowcast",
            "stream": "solexs-sxr-long",
            "goes_class": "M2.5",
            "severity": "M",
            "onset_time": 5_000.0,
            "detectors": ["CUSUM"],
            "t": 5_240.0,
        },
        stream="solexs-sxr-long",
    )
    store.put_forecast(
        {
            "t_issued": 5_240.0,
            "stream": "fused",
            "horizon_min": 30,
            "class_threshold": "C",
            "p_flare": 0.42,
            "model": "persistence-baseline",
            "p_curve": {"5": 0.05, "15": 0.22, "30": 0.42, "60": 0.58},
            "data_quality": 0.9,
        }
    )
    for ev in tiny_catalogue:
        store.put_event(ev)
    return TestClient(create_app(store))


# ---------------------------------------------------------------------------
# Static dashboard at "/".
# ---------------------------------------------------------------------------
def test_root_serves_dashboard(client_empty: TestClient) -> None:
    """``GET /`` serves the single-page dashboard HTML."""
    r = client_empty.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    assert "Aditya FlareCast" in body
    # the page wires the offline canvas client + styles.
    assert "app.js" in body
    assert "styles.css" in body


def test_dashboard_assets_served(client_empty: TestClient) -> None:
    """The dashboard's JS / CSS assets are reachable from the static mount."""
    js = client_empty.get("/app.js")
    css = client_empty.get("/styles.css")
    assert js.status_code == 200
    assert css.status_code == 200
    assert "javascript" in js.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Health.
# ---------------------------------------------------------------------------
def test_health_ok_empty(client_empty: TestClient) -> None:
    """``GET /api/health`` returns 200 with a liveness/contents summary."""
    r = client_empty.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_streams"] == 0
    assert body["n_events"] == 0


def test_health_seeded(client_seeded: TestClient) -> None:
    """Health reflects seeded streams / events / forecast / alert."""
    body = client_seeded.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_streams"] >= 2
    assert body["n_events"] == 3
    assert body["has_forecast"] is True
    assert body["has_alert"] is True


# ---------------------------------------------------------------------------
# O(1) read endpoints (shapes the dashboard consumes).
# ---------------------------------------------------------------------------
def test_latest_sample(client_seeded: TestClient) -> None:
    r = client_seeded.get("/api/latest?stream=solexs-sxr-long")
    assert r.status_code == 200
    s = r.json()
    assert s["stream"] == "solexs-sxr-long"
    assert s["value"] == pytest.approx(2.5e-5)
    assert {"t", "value", "unit", "cls"} <= set(s)


def test_latest_missing_stream_404(client_empty: TestClient) -> None:
    assert client_empty.get("/api/latest?stream=does-not-exist").status_code == 404


def test_alert_present_and_absent(
    client_seeded: TestClient, client_empty: TestClient
) -> None:
    al = client_seeded.get("/api/alert").json()
    assert al["goes_class"] == "M2.5"
    assert al["severity"] == "M"
    # no alert yet -> {"alert": null} at HTTP 200 (dashboard treats as "no alert").
    none = client_empty.get("/api/alert")
    assert none.status_code == 200
    assert none.json() == {"alert": None}


def test_forecast_present_and_absent(
    client_seeded: TestClient, client_empty: TestClient
) -> None:
    fc = client_seeded.get("/api/forecast").json()
    assert fc["p_flare"] == pytest.approx(0.42)
    assert "p_curve" in fc and "30" in fc["p_curve"]
    assert client_empty.get("/api/forecast").json() == {"forecast": None}


def test_catalogue_recent_and_range(client_seeded: TestClient) -> None:
    """``/api/catalogue`` returns events both as a recent list and a range query."""
    recent = client_seeded.get("/api/catalogue?limit=10").json()
    assert recent["count"] == 3
    ev = recent["events"][0]
    assert {"t_start", "t_peak", "t_end", "goes_class", "flags", "confidence"} <= set(ev)
    assert {"soft", "hard", "neupert_consistent"} <= set(ev["flags"])
    # the indexed SQLite range path (previously thread-bound) also works.
    rng = client_seeded.get("/api/catalogue?start=0&end=100000&limit=10").json()
    assert rng["count"] == 3
    # min_class filtering keeps only >= M.
    only_m = client_seeded.get("/api/catalogue?limit=10&min_class=M").json()
    assert all(e["goes_class"][0] in ("M", "X") for e in only_m["events"])


def test_at_by_time_bucket(client_seeded: TestClient) -> None:
    """``/api/at`` returns the catalogue events in the time bucket of ``t``."""
    r = client_seeded.get("/api/at?t=5240")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body and "bucket" in body
