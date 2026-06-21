#!/usr/bin/env python3
"""Example 05 -- serve the API + dashboard on synthetic / cached data (offline).

Wires the serving layer (ARCHITECTURE.md Section 7.4 / 8): seed a
:class:`flarecast.api.store.ReadStore` with one synthetic run (latest soft/hard
samples, a master catalogue, an alert, and a forecast record), build the FastAPI
app with :func:`flarecast.api.app.create_app` (which mounts the static
``dashboard/`` at ``/``), and expose the O(1) ``/api/*`` reads + the
``/api/stream`` SSE feed the dashboard consumes.

By default this **does not block**: it runs an in-process self-check
(``fastapi.testclient``) confirming ``GET /`` serves the dashboard and the
``/api/*`` endpoints respond, then prints the command to launch a real server
and exits cleanly -- so it is safe to run in CI / a sandbox. Pass ``--serve`` to
actually start uvicorn and serve until Ctrl-C.

Run::

    python examples/05_serve_dashboard.py            # self-check + print, exit 0
    python examples/05_serve_dashboard.py --serve    # launch uvicorn (blocking)
    python examples/05_serve_dashboard.py --serve --port 8001
"""

from __future__ import annotations

import argparse
import sys

from flarecast.api.app import create_app
from flarecast.api.store import ReadStore
from flarecast.cli.main import _seed_store_synth


def _self_check(app, store: ReadStore) -> int:
    """Drive the app in-process with TestClient; print endpoint statuses."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("(self-check skipped: fastapi/httpx test client not installed)")
        return 0

    c = TestClient(app)
    checks = [
        ("GET /", c.get("/")),
        ("GET /api/health", c.get("/api/health")),
        ("GET /api/latest?stream=solexs-sxr-long",
         c.get("/api/latest?stream=solexs-sxr-long")),
        ("GET /api/alert", c.get("/api/alert")),
        ("GET /api/forecast", c.get("/api/forecast")),
        ("GET /api/catalogue?limit=5", c.get("/api/catalogue?limit=5")),
    ]
    print("In-process self-check (fastapi.testclient)")
    print("-" * 70)
    ok = True
    for label, r in checks:
        good = r.status_code == 200
        ok = ok and good
        extra = ""
        if label == "GET /":
            extra = "  dashboard HTML" if "Aditya FlareCast" in r.text else "  (no dashboard?)"
        print(f"  {('OK ' if good else 'ERR')} [{r.status_code}] {label}{extra}")
    print()
    h = store.health()
    print(f"seeded store: {h['n_streams']} streams, {h['n_events']} events, "
          f"forecast={h['has_forecast']}, alert={h['has_alert']}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=12.0, help="synthetic duration (h)")
    ap.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (with --serve)")
    ap.add_argument("--port", type=int, default=8000, help="bind port (with --serve)")
    ap.add_argument("--serve", action="store_true",
                    help="actually launch uvicorn and block until Ctrl-C")
    args = ap.parse_args()

    print("=" * 70)
    print("Aditya FlareCast - serve API + dashboard (offline synth/cached data)")
    print("=" * 70)

    store = ReadStore()
    _seed_store_synth(store, hours=args.hours, cadence_s=args.cadence, seed=args.seed)
    app = create_app(store)

    if not args.serve:
        rc = _self_check(app, store)
        print()
        print("To serve the live dashboard, run either:")
        print(f"  python examples/05_serve_dashboard.py --serve --port {args.port}")
        print(f"  flarecast serve --host {args.host} --port {args.port}")
        print(f"then open http://{args.host}:{args.port}/  (API docs at /docs)")
        print()
        print("OK - serving layer wired offline (non-blocking self-check)."
              if rc == 0 else "self-check reported a failing endpoint.")
        return rc

    # --serve: launch the real server (blocking).
    try:
        import uvicorn
    except ImportError:
        print("--serve requires uvicorn (pip install 'flarecast[api]').", file=sys.stderr)
        return 2
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    print(f"  dashboard: http://{args.host}:{args.port}/")
    print(f"  API docs : http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
