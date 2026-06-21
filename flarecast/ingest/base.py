"""Fetcher protocol + fallback chaining (ARCHITECTURE.md Section 6, B.1).

Every data source in :mod:`flarecast.ingest` (and the synthetic generator,
adapted) implements the same tiny :class:`Fetcher` protocol so they are
interchangeable and composable. :func:`with_fallback` chains several fetchers
into the mandated **live -> cache -> synth** order (ARCHITECTURE.md Section 6):
it tries each in turn and, on any *source-unavailable* error (network timeout,
URL error, missing file, parse failure), moves to the next -- so the pipeline
runs with zero network and zero credentials in the sandbox.

Pure standard library. The set of errors treated as "try the next source" is
:data:`FALLBACK_ERRORS` (URL/OS/timeout/value/key errors); anything else
propagates so genuine bugs are not silently swallowed.
"""

from __future__ import annotations

import json
import socket
import urllib.error
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..types import FluxSample

__all__ = ["Fetcher", "with_fallback", "FallbackFetcher", "FALLBACK_ERRORS"]


#: Exceptions that mean "this source is unavailable -- fall back to the next".
#: Network (URLError/timeout/socket), filesystem (OSError -> FileNotFoundError),
#: and defensive parse failures (ValueError/KeyError/json errors) per the
#: input-hardening rule in ARCHITECTURE.md Section 11.2.
FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    OSError,
    ValueError,
    KeyError,
    json.JSONDecodeError,
)


@runtime_checkable
class Fetcher(Protocol):
    """A source of :class:`FluxSample` records over a time window (B.1).

    Implementations pull (or generate) samples for the half-open interval
    ``[t_start, t_end)`` in epoch seconds UTC and yield them as
    :class:`FluxSample`. The protocol is intentionally minimal so live JSON
    pullers, FITS readers, cache loaders, and the synthetic generator all
    satisfy it.
    """

    def fetch(self, t_start: float, t_end: float) -> Iterator[FluxSample]:
        """Yield :class:`FluxSample` for ``[t_start, t_end)``."""
        ...


class FallbackFetcher:
    """Chain fetchers, advancing to the next on a source-unavailable error.

    Returned by :func:`with_fallback`. On :meth:`fetch`, each underlying fetcher
    is tried in order; its output is *materialised* (so an error raised partway
    through iteration still triggers fallback rather than yielding a truncated
    stream), and the first fetcher that produces *any* samples without a
    :data:`FALLBACK_ERRORS` exception wins. If a fetcher succeeds but yields
    zero samples, the chain continues to the next (an empty result is treated as
    "no data here", matching the live->cache->synth intent).

    The last fetcher's errors propagate only if *every* prior fetcher also
    failed and it, too, raises -- otherwise the chain degrades gracefully. If
    all fetchers fail, the final exception is re-raised so the caller sees a
    real failure rather than silent emptiness.
    """

    def __init__(self, *fetchers: Fetcher, allow_empty: bool = False) -> None:
        if not fetchers:
            raise ValueError("with_fallback requires at least one fetcher")
        self._fetchers = fetchers
        self._allow_empty = allow_empty
        #: name of the fetcher that served the most recent fetch (introspection).
        self.last_source: str | None = None

    def fetch(self, t_start: float, t_end: float) -> Iterator[FluxSample]:
        """Yield samples from the first working fetcher in the chain."""
        last_exc: BaseException | None = None
        for fetcher in self._fetchers:
            name = type(fetcher).__name__
            try:
                samples = list(fetcher.fetch(t_start, t_end))
            except FALLBACK_ERRORS as exc:
                last_exc = exc
                continue
            if not samples and not self._allow_empty:
                # treat "worked but no data" as a reason to try the next tier,
                # unless this is the last fetcher (then return what we have).
                if fetcher is not self._fetchers[-1]:
                    continue
            self.last_source = name
            yield from samples
            return
        # every fetcher failed.
        if last_exc is not None:
            raise last_exc
        # all succeeded-but-empty and empties disallowed: yield nothing.
        self.last_source = type(self._fetchers[-1]).__name__
        return


def with_fallback(*fetchers: Fetcher, allow_empty: bool = False) -> Fetcher:
    """Chain ``fetchers`` into one fallback :class:`Fetcher` (B.1).

    Usage (the canonical live -> cache -> synth order)::

        fetcher = with_fallback(live_goes, cached_goes, synth_goes)
        for sample in fetcher.fetch(t0, t1):
            ...

    Parameters
    ----------
    *fetchers:
        Fetchers to try in order (highest priority first).
    allow_empty:
        If True, a fetcher that succeeds with zero samples ends the chain
        (returns empty) instead of falling through to the next.

    Returns
    -------
    A :class:`FallbackFetcher` (which is a :class:`Fetcher`).
    """
    return FallbackFetcher(*fetchers, allow_empty=allow_empty)
