"""Data-access layer: real sources (GOES/Fido/PRADAN) + offline fallback.

Workstream 1 (ARCHITECTURE.md Appendix C / B.1). Every source implements the
:class:`~flarecast.ingest.base.Fetcher` protocol and normalizes to
:class:`~flarecast.types.FluxSample`; :func:`~flarecast.ingest.base.with_fallback`
chains the mandated live -> cache -> synth order so the pipeline runs offline
(ARCHITECTURE.md Section 6).

The live + offline paths are **pure standard library**. SunPy (archive tier,
:mod:`flarecast.ingest.fido_sources`) and astropy (Aditya-L1 FITS,
:mod:`flarecast.ingest.pradan`) are optional and imported lazily, so importing
this package never requires them.
"""

from __future__ import annotations

from .base import Fetcher, with_fallback
from .cache import load_cached, save_cache
from .goes import GOESFetcher, flare_class
from .normalize import goes_class, normalize_generic, normalize_swpc

__all__ = [
    "Fetcher",
    "with_fallback",
    "GOESFetcher",
    "flare_class",
    "normalize_swpc",
    "normalize_generic",
    "goes_class",
    "load_cached",
    "save_cache",
]
