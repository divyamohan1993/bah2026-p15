"""FastAPI application factory (ARCHITECTURE.md Section 7.4 / Appendix B.7).

:func:`create_app` wires the offline serving stack: CORS (so the dashboard can
be hosted anywhere), the O(1) read routes (:mod:`flarecast.api.routes`), the SSE
live-push router (:mod:`flarecast.api.sse`), and a static mount of the
``dashboard/`` single-page UI at ``/`` -- the same UI Cloudflare Pages serves in
production, only the substrate differs (Section 7.2: "the detector math is
identical ... only the substrate ... differs").

FastAPI / starlette are imported **inside** :func:`create_app`, so importing
this module never requires the optional ``api`` extra; the CLI guards the import
and the API test uses ``pytest.importorskip("fastapi")``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ReadStore

__all__ = ["create_app", "dashboard_dir"]


def dashboard_dir() -> str:
    """Return the absolute path to the repo's ``dashboard/`` directory.

    Resolved relative to this file (``flarecast/api/app.py`` -> ``<repo>/
    dashboard``) so the static mount works regardless of the working directory.
    An environment override ``FLARECAST_DASHBOARD_DIR`` takes precedence.
    """
    override = os.environ.get("FLARECAST_DASHBOARD_DIR")
    if override:
        return os.path.abspath(override)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))  # flarecast/api -> repo
    return os.path.join(repo_root, "dashboard")


def create_app(store: ReadStore, forecaster: Any = None):
    """Build the FlareCast FastAPI app bound to ``store`` (Appendix B.7).

    Parameters
    ----------
    store:
        The :class:`~flarecast.api.store.ReadStore` every route reads from.
    forecaster:
        Optional :class:`~flarecast.forecast.model_gbt.GBTForecaster`, passed
        through to the routes for (future) on-the-fly scoring.

    Returns
    -------
    fastapi.FastAPI
        The configured application: CORS enabled, ``/api`` routes + ``/api/stream``
        SSE included, and ``dashboard/`` mounted at ``/`` when it exists.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from .routes import make_router
    from .sse import make_sse_router

    app = FastAPI(
        title="Aditya FlareCast API",
        description=(
            "Offline serving layer for soft + hard X-ray solar-flare nowcasting "
            "and forecasting (ISRO BAH 2026, PS-15). O(1) hot reads + SSE live "
            "push; mirrors the Cloudflare edge."
        ),
        version="0.1.0",
    )

    # CORS: permissive for the offline demo / local dashboard (read-only API,
    # no PII -- ARCHITECTURE.md Section 11.2). Tighten origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # API routes (O(1) reads) + SSE live push.
    app.include_router(make_router(store, forecaster=forecaster))
    app.include_router(make_sse_router(store))

    # Static dashboard at "/" (mounted last so /api/* and /docs win). Guarded so
    # the app still serves the API if the dashboard directory is absent.
    ddir = dashboard_dir()
    if os.path.isdir(ddir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=ddir, html=True), name="dashboard")

    return app
