"""O(1) read routes for the serving API (ARCHITECTURE.md B.7, Section 7.2).

All routes read from a :class:`flarecast.api.store.ReadStore`, whose hot KV
mirror makes ``latest`` / ``alert`` / ``at`` / ``forecast`` single hash
lookups (O(1)); only ``/api/catalogue`` touches the indexed SQLite range path
(O(log n + k)). The paths are the normative ones from Appendix B.7, served
under an ``/api`` prefix so the dashboard static mount can own ``/``::

    GET /api/latest?stream=...
    GET /api/alert?stream=...
    GET /api/at?t=<epoch>&stream=...
    GET /api/catalogue?start=..&end=..&limit=..&min_class=..
    GET /api/forecast
    GET /api/health

FastAPI is imported lazily inside :func:`make_router` so importing this module
never requires the optional ``api`` extra; the SSE router lives in
:mod:`flarecast.api.sse`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ReadStore

__all__ = ["make_router"]


def make_router(store: ReadStore, forecaster: Any = None):
    """Build the ``/api`` :class:`fastapi.APIRouter` bound to ``store``.

    Parameters
    ----------
    store:
        The :class:`~flarecast.api.store.ReadStore` every route reads from.
    forecaster:
        Optional :class:`~flarecast.forecast.model_gbt.GBTForecaster` (or any
        object with ``predict_proba``). Currently the ``/api/forecast`` route
        serves the store's cached forecast record; the forecaster is accepted
        for parity with the contract and future on-the-fly scoring.

    Returns
    -------
    fastapi.APIRouter
        Router with the six O(1) read endpoints, mounted under ``/api``.
    """
    from fastapi import APIRouter, HTTPException, Query  # lazy

    router = APIRouter(prefix="/api", tags=["flarecast"])

    @router.get("/latest")
    def get_latest(stream: str = Query("solexs-sxr-long", description="stream id")):
        """Latest sample for ``stream`` (O(1) KV read)."""
        rec = store.latest(stream)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no sample for stream {stream!r}")
        return rec

    @router.get("/alert")
    def get_alert(stream: str | None = Query(None, description="optional stream id")):
        """Latest alert, global or per-stream (O(1) KV read).

        Returns ``{"alert": null}`` (HTTP 200) when no alert has fired yet, so a
        polling dashboard does not treat "no alert" as an error.
        """
        rec = store.alert(stream)
        return rec if rec is not None else {"alert": None}

    @router.get("/at")
    def get_at(
        t: float = Query(..., description="epoch seconds"),
        stream: str | None = Query(None, description="optional stream id"),
    ):
        """Catalogue events in the time bucket of ``t`` (O(1) by-time)."""
        rec = store.at(t, stream)
        if rec is None:
            raise HTTPException(status_code=400, detail=f"bad time value {t!r}")
        return rec

    @router.get("/catalogue")
    def get_catalogue(
        start: float | None = Query(None, description="window start (epoch s)"),
        end: float | None = Query(None, description="window end (epoch s)"),
        limit: int = Query(50, ge=1, le=1000, description="max events"),
        min_class: str | None = Query(None, description="min GOES class, e.g. M"),
    ):
        """Catalogue events.

        With ``start``/``end`` this is the indexed range query (O(log n + k));
        without them it returns the most recent ``limit`` events (newest first)
        from the O(1) in-memory index -- the dashboard's "recent flares" table.
        """
        if start is not None and end is not None:
            events = store.catalogue(start, end, min_class=min_class)
            events = list(reversed(events))  # newest-first for display
            if limit:
                events = events[: int(limit)]
            return {"events": events, "count": len(events)}
        events = store.latest_catalogue(limit=limit)
        if min_class:
            from ..catalog.index import _class_rank  # reuse the rank helper

            mr = _class_rank(min_class)
            events = [e for e in events if _class_rank(e.get("goes_class")) >= mr]
        return {"events": events, "count": len(events)}

    @router.get("/forecast")
    def get_forecast():
        """Latest forecast record (O(1) KV read).

        Returns ``{"forecast": null}`` (HTTP 200) when no forecast has been
        issued yet.
        """
        rec = store.forecast()
        return rec if rec is not None else {"forecast": None}

    @router.get("/health")
    def get_health():
        """Liveness + a small contents summary (O(1))."""
        return store.health()

    return router
