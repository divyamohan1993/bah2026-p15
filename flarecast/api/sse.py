"""Server-Sent Events live push: ``GET /api/stream`` (Section 7.2 / 8).

The offline analogue of the edge Durable-Object WebSocket fan-out
(ARCHITECTURE.md Section 7.1): a ``text/event-stream`` endpoint that pushes the
latest soft + hard flux and any alert at the grid cadence, so the dashboard's
light curves and alert banner update live without polling.

Two modes:

* **store mode** (default): each tick emits the current ``latest:*`` and
  ``alert:latest`` values from the :class:`~flarecast.api.store.ReadStore` -- a
  faithful mirror of the edge's KV->WS push.
* **replay mode** (offline demo): when the store has no live writer, the
  endpoint replays a deterministic synthetic stream
  (:func:`flarecast.synth.generate_flare_lightcurves`) through the soft/hard
  detectors so the dashboard animates a full flare end-to-end with zero
  network. ``?replay=1`` forces this mode.

FastAPI / starlette are imported lazily inside :func:`make_sse_router`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

from ..constants import NOWCAST_GRID_DT_S

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ReadStore

__all__ = ["make_sse_router", "sse_event", "replay_events"]


def sse_event(data: Any, event: str | None = None, event_id: str | None = None) -> str:
    """Format one Server-Sent Event frame (``event:``/``id:``/``data:``).

    ``data`` is JSON-encoded. The frame ends with the mandatory blank line.
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    payload = json.dumps(data, default=str)
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def replay_events(
    *,
    duration_s: float = 3600.0,
    cadence_s: float = 30.0,
    seed: int | None = 7,
    max_ticks: int | None = None,
) -> Iterator[str]:
    """Yield SSE frames replaying a synthetic soft+hard stream through detectors.

    Pure-python + ``flarecast.synth`` + ``flarecast.detect`` (no network). Each
    tick emits a ``flux`` event (soft W/m^2 + hard counts/s + GOES class) and,
    when a detector fires, an ``alert`` event -- exactly the payloads the
    dashboard consumes. Deterministic for a fixed ``seed``.
    """
    from ..detect.stack import HardBandDetector, SoftBandDetector
    from ..synth.generator import (
        STREAM_HXR_LOW,
        STREAM_SXR_LONG,
        generate_flare_lightcurves,
    )

    samples, _truth = generate_flare_lightcurves(
        duration_s=duration_s, cadence_s=cadence_s, seed=seed
    )
    # Pivot to per-timestamp soft/hard values.
    soft: dict[float, float] = {}
    hard: dict[float, float] = {}
    for s in samples:
        if s.stream == STREAM_SXR_LONG:
            soft[s.t] = s.value
        elif s.stream == STREAM_HXR_LOW:
            hard[s.t] = s.value
    times = sorted(set(soft) | set(hard))

    soft_det = SoftBandDetector(cadence_s=cadence_s)
    hard_det = HardBandDetector(cadence_s=cadence_s)

    from .store import ReadStore  # local import to avoid a cycle at module load

    n = 0
    for t in times:
        if max_ticks is not None and n >= max_ticks:
            break
        n += 1
        sx = soft.get(t, 1e-8)
        hx = hard.get(t, 0.0)
        s_state = soft_det.update(sx, t)
        h_state = hard_det.update(hx, t)
        cls = (s_state.meta or {}).get("goes_class")
        yield sse_event(
            {
                "t": t,
                "soft": sx,
                "hard": hx,
                "goes_class": cls,
                "soft_in_event": s_state.in_event,
                "hard_in_event": h_state.in_event,
            },
            event="flux",
            event_id=str(n),
        )
        if s_state.onset or s_state.peak:
            yield sse_event(
                ReadStore.alert_from_detection(s_state, STREAM_SXR_LONG),
                event="alert",
            )
        if h_state.onset:
            yield sse_event(
                ReadStore.alert_from_detection(h_state, STREAM_HXR_LOW),
                event="alert",
            )


def make_sse_router(store: "ReadStore"):
    """Build the SSE :class:`fastapi.APIRouter` (``GET /api/stream``).

    Parameters
    ----------
    store:
        The :class:`~flarecast.api.store.ReadStore` to push live values from.

    Returns
    -------
    fastapi.APIRouter
        Router exposing ``GET /api/stream`` returning ``text/event-stream``.
    """
    import asyncio

    from fastapi import APIRouter, Query  # lazy
    from fastapi.responses import StreamingResponse

    router = APIRouter(prefix="/api", tags=["stream"])

    @router.get("/stream")
    def stream(
        replay: int = Query(0, description="1 to force the synthetic replay"),
        cadence_s: float = Query(NOWCAST_GRID_DT_S, description="tick spacing (s)"),
        max_ticks: int = Query(0, description="0 = unbounded (until disconnect)"),
        speed: float = Query(50.0, description="replay speed-up factor"),
    ):
        """Server-Sent Events: latest flux + alerts at the grid cadence.

        Streams from the store when it has data, else replays a synthetic
        stream so the dashboard animates offline. The replay is sped up by
        ``speed`` so a long synthetic day animates in seconds.
        """
        use_replay = bool(replay) or store.latest("solexs-sxr-long") is None

        async def event_gen():
            cap = max_ticks if max_ticks and max_ticks > 0 else None
            if use_replay:
                # Replay is a (potentially long) synchronous generator; sleep
                # briefly between frames so the event loop stays responsive and
                # the client sees a live animation.
                delay = max(0.0, cadence_s / max(speed, 1e-6))
                for frame in replay_events(cadence_s=cadence_s, max_ticks=cap):
                    yield frame
                    if delay:
                        await asyncio.sleep(delay)
                yield sse_event({"done": True}, event="end")
                return
            # Store mode: emit the current latest values each tick.
            sent = 0
            while cap is None or sent < cap:
                sent += 1
                flux = {
                    "t": None,
                    "soft": store.latest("solexs-sxr-long"),
                    "hard": store.latest("hel1os-hxr-8-30keV"),
                }
                yield sse_event(flux, event="flux", event_id=str(sent))
                alert = store.alert()
                if alert is not None:
                    yield sse_event(alert, event="alert")
                await asyncio.sleep(max(0.0, cadence_s / max(speed, 1e-6)))

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
