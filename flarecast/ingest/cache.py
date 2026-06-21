"""Cached-sample load/save for the offline fallback (ARCHITECTURE.md S6, B.1).

The offline fallback chain is ``live network -> cached sample -> synth`` (a
non-negotiable property for the no-network sandbox, ARCHITECTURE.md Section 6).
This module implements the middle tier: small, bundled JSON samples under
``examples/data/`` that let a fetcher return *deterministic, realistically
shaped* data when the network is unavailable.

Pure standard library (``json`` + ``os``). Two public functions mirror
Appendix B.1::

    load_cached(name) -> list[dict]
    save_cache(name, data) -> None

``name`` is a bare file name (e.g. ``"xrays-1-day.sample.json"``); the data
directory is resolved relative to the repository so the cache works regardless
of the current working directory.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["load_cached", "save_cache", "cache_dir", "cache_path", "has_cache"]


def cache_dir() -> str:
    """Return the absolute path to the bundled ``examples/data/`` directory.

    Resolved relative to this file (``flarecast/ingest/cache.py`` ->
    ``<repo>/examples/data``) so it is independent of the process working
    directory. An environment override ``FLARECAST_DATA_DIR`` takes precedence
    (useful for tests or alternative deployments).
    """
    override = os.environ.get("FLARECAST_DATA_DIR")
    if override:
        return os.path.abspath(override)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))  # flarecast/ingest -> repo
    return os.path.join(repo_root, "examples", "data")


def cache_path(name: str) -> str:
    """Return the absolute path for a cache file ``name`` under the data dir."""
    return os.path.join(cache_dir(), name)


def has_cache(name: str) -> bool:
    """Return True if the named cache file exists and is non-empty."""
    p = cache_path(name)
    return os.path.isfile(p) and os.path.getsize(p) > 0


def load_cached(name: str) -> list[dict[str, Any]]:
    """Load a bundled cached sample as a list of dicts (Appendix B.1).

    Parameters
    ----------
    name:
        Bare file name under ``examples/data/`` (e.g.
        ``"xrays-1-day.sample.json"``).

    Returns
    -------
    The parsed JSON. SWPC-style files are JSON arrays of flat record objects;
    this returns that list. If the JSON top level is a dict it is wrapped in a
    single-element list so the return type is always ``list[dict]``.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist (the caller's fallback should then
        proceed to the synthetic generator).
    """
    p = cache_path(name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"cached sample {name!r} not found at {p!r}; offline fallback should "
            f"continue to the synthetic generator"
        )
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return [data]
    return list(data)


def save_cache(name: str, data: list[dict[str, Any]]) -> None:
    """Persist ``data`` as a JSON cache file ``name`` under the data dir.

    Creates the data directory if needed. Used to snapshot a live fetch so a
    later offline run is deterministic.

    Parameters
    ----------
    name:
        Bare file name to write under ``examples/data/``.
    data:
        A JSON-serialisable list of record dicts.
    """
    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
