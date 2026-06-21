"""GOES XRS fetcher: live SWPC JSON -> cached sample -> synth (ARCHITECTURE B.1).

GOES XRS is the **live operational nowcast anchor** (ARCHITECTURE.md Section
1.3, research 05 Section 1.2): a sub-minute, public, no-auth JSON feed that
defines the A-X flare scale. :class:`GOESFetcher` pulls the SWPC ``xrays-*``
JSON with :mod:`urllib` and a short timeout and, on *any* network failure, falls
back to a bundled cached sample and then to the physics-based synthetic
generator -- so the nowcast path runs offline and deterministically
(ARCHITECTURE.md Section 6).

Each SWPC timestamp carries two channels (``"0.1-0.8nm"`` long, ``"0.05-0.4nm"``
short); the fetcher yields both as :class:`FluxSample` (the long channel carries
the GOES class). The module also exposes the free function :func:`flare_class`
(mirrors ``detect.classify`` / ``normalize.goes_class``) required by B.1.

Pure standard library for the live + offline paths (``json`` + ``urllib``); the
synth fallback uses :mod:`flarecast.synth` (also pure-python).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any

from ..constants import (
    DEFAULT_FETCH_TIMEOUT_S,
    SWPC_XRAYS_URL,
    UNIT_SXR,
)
from ..types import FluxSample
from . import cache as _cache
from . import normalize as _normalize

__all__ = ["GOESFetcher", "flare_class"]

#: Default cached sample bundled for the offline GOES fallback.
DEFAULT_CACHE_NAME = "xrays-1-day.sample.json"


def flare_class(flux_wm2: float) -> str:
    """Map a GOES 1-8 A flux [W m^-2] to a class string, e.g. ``"C3.1"`` (B.1).

    Thin alias of :func:`flarecast.ingest.normalize.goes_class`, kept on the
    fetcher module because Appendix B.1 lists it under ``goes.py``. Mirrors
    ``flarecast.detect.classify.classify_flux``.
    """
    return _normalize.goes_class(flux_wm2)


class GOESFetcher:
    """Fetch GOES XRS soft-X-ray flux as :class:`FluxSample` (Appendix B.1).

    Resolution order on every :meth:`fetch` / :meth:`fetch_latest`:

    1. **Live** -- GET the SWPC JSON (``SWPC_XRAYS_URL``) with a timeout.
    2. **Cache** -- on any network/parse error, load the bundled sample
       (``cache_path`` or :data:`DEFAULT_CACHE_NAME` under ``examples/data/``).
    3. **Synth** -- if the cache is missing/unreadable, generate a deterministic
       synthetic day via :mod:`flarecast.synth` and relabel to GOES streams.

    Parameters
    ----------
    channel:
        ``"long"`` (1-8 A, class channel), ``"short"`` (0.5-4 A), or ``"both"``
        (default-ish) to yield both channels. Default ``"long"`` per B.1.
    satellite:
        SWPC satellite selector (``"primary"`` / ``"secondary"`` or a number);
        used only to label the stream id when records omit ``satellite``.
    cache_path:
        Optional explicit path/name of the cached JSON sample. A bare name is
        resolved under ``examples/data/``; an absolute path is used as-is.
    timeout_s:
        Network timeout before falling back (default
        :data:`DEFAULT_FETCH_TIMEOUT_S`).
    url:
        Override the SWPC URL (default :data:`SWPC_XRAYS_URL`).
    allow_network:
        If False, skip the live tier entirely (cache -> synth). Useful for the
        offline test and the no-network sandbox.
    synth_seed:
        Seed for the synthetic fallback so offline behaviour is deterministic.
    """

    def __init__(
        self,
        channel: str = "long",
        satellite: str = "primary",
        cache_path: str | None = None,
        *,
        timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
        url: str = SWPC_XRAYS_URL,
        allow_network: bool = True,
        synth_seed: int | None = 1234,
    ) -> None:
        self.channel = channel
        self.satellite = satellite
        self.cache_path = cache_path
        self.timeout_s = timeout_s
        self.url = url
        self.allow_network = allow_network
        self.synth_seed = synth_seed
        #: which tier served the most recent fetch ("live"|"cache"|"synth").
        self.last_source: str | None = None

    # ------------------------------------------------------------------ #
    # public API (Appendix B.1)
    # ------------------------------------------------------------------ #
    def fetch(self, t_start: float, t_end: float) -> Iterator[FluxSample]:
        """Yield GOES XRS samples in ``[t_start, t_end)`` (live->cache->synth)."""
        records = self._load_records()
        if records is not None:
            yield from self._records_to_samples(records, t_start, t_end)
            return
        # synth fallback.
        self.last_source = "synth"
        yield from self._synth_samples(t_start, t_end)

    def fetch_latest(self) -> FluxSample | None:
        """Return the single most-recent long-channel sample (live hot path).

        Tries the live feed, then cache, then synth. Returns the latest
        long-channel :class:`FluxSample` (the channel that defines the flare
        class), or ``None`` if no data is available at all.
        """
        records = self._load_records()
        if records is None:
            # synth latest.
            self.last_source = "synth"
            samples = list(self._synth_samples(0.0, float("inf")))
            longs = [s for s in samples if s.stream.endswith("-long")]
            return longs[-1] if longs else None
        longs: list[FluxSample] = []
        for rec in records:
            if rec.get("energy") != _normalize.SWPC_ENERGY_LONG:
                continue
            try:
                longs.append(_normalize.normalize_swpc(rec))
            except (ValueError, KeyError):
                continue
        if not longs:
            return None
        longs.sort(key=lambda s: s.t)
        return longs[-1]

    # ------------------------------------------------------------------ #
    # internal tiers
    # ------------------------------------------------------------------ #
    def _load_records(self) -> list[dict[str, Any]] | None:
        """Return raw SWPC records from live or cache, or ``None`` for synth."""
        if self.allow_network:
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout_s) as resp:
                    data = json.load(resp)
                self.last_source = "live"
                return list(data)
            except Exception:  # noqa: BLE001 - any failure => fall back
                pass
        # cache tier.
        name = self.cache_path or DEFAULT_CACHE_NAME
        try:
            if name and (name.startswith("/") or name.startswith(".")):
                with open(name, encoding="utf-8") as fh:
                    data = json.load(fh)
                records = list(data)
            else:
                records = _cache.load_cached(name)
            self.last_source = "cache"
            return records
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _records_to_samples(
        self, records: list[dict[str, Any]], t_start: float, t_end: float
    ) -> Iterator[FluxSample]:
        """Normalize SWPC records to FluxSamples, filtered by channel + window."""
        want_long = self.channel in ("long", "both")
        want_short = self.channel in ("short", "both")
        for rec in records:
            energy = rec.get("energy")
            if energy == _normalize.SWPC_ENERGY_LONG and not want_long:
                continue
            if energy == _normalize.SWPC_ENERGY_SHORT and not want_short:
                continue
            try:
                sample = _normalize.normalize_swpc(rec)
            except (ValueError, KeyError):
                # input-hardening: skip malformed records (Section 11.2).
                continue
            if t_start <= sample.t < t_end:
                yield sample

    def _synth_samples(self, t_start: float, t_end: float) -> Iterator[FluxSample]:
        """Generate a synthetic GOES-equivalent stream as the last-resort tier.

        Uses :func:`flarecast.synth.generate_flare_lightcurves` and relabels the
        soft (SoLEXS) streams as GOES streams so downstream code sees a GOES
        feed. Only the soft channels are emitted (GOES XRS has no hard band).
        """
        from ..synth.generator import (
            STREAM_SXR_LONG,
            STREAM_SXR_SHORT,
            generate_flare_lightcurves,
        )

        # choose a duration covering the requested window (default 1 day).
        if t_end == float("inf") or t_end <= t_start:
            duration = 86400.0
            offset = t_start if t_start not in (0.0, float("inf")) else 0.0
        else:
            duration = max(3600.0, t_end - t_start)
            offset = t_start
        samples, _truth = generate_flare_lightcurves(
            duration_s=duration, cadence_s=60.0, seed=self.synth_seed
        )
        want_long = self.channel in ("long", "both")
        want_short = self.channel in ("short", "both")
        sat = self.satellite
        for s in samples:
            if s.stream == STREAM_SXR_LONG and want_long:
                stream = f"goes-{sat}-long"
            elif s.stream == STREAM_SXR_SHORT and want_short:
                stream = f"goes-{sat}-short"
            else:
                continue
            t_abs = s.t + offset
            if t_end != float("inf") and not (t_start <= t_abs < t_end):
                # keep only in-window when a finite window was requested.
                if not (t_start == 0.0 and t_end == float("inf")):
                    continue
            yield FluxSample(
                stream=stream,
                t=t_abs,
                value=s.value,
                unit=UNIT_SXR,
                source="synth",
                quantity=s.quantity,
                cls=s.cls,
                qc=s.qc,
                meta={**(s.meta or {}), "fallback": "synth"},
            )
