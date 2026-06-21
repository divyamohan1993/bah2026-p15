"""SunPy Fido fetchers for the archive / science tier (ARCHITECTURE.md B.1).

The **archive tier** (ARCHITECTURE.md Section 6, research 05 Section 1.0) pulls
big FITS/NetCDF/CDF products and event catalogues through SunPy's federated
``Fido`` client (VSO / JSOC / HEK / CDAWeb / XRS / RHESSI / GONG / e-CALLISTO).
This is the tier for model training, backtesting, and ground-truth labelling --
*not* the live nowcast hot path (that is GOES SWPC JSON, see
:mod:`flarecast.ingest.goes`).

SunPy and astropy are **optional** heavy dependencies (ARCHITECTURE.md
dependency philosophy / ``pyproject.toml`` ``[real]`` extra). They are imported
*lazily inside the methods* and guarded so merely importing this module -- and
the whole ``flarecast`` package -- never requires them. When SunPy is absent the
methods raise a clear, actionable :class:`ImportError`; any test exercising a
live fetch must ``pytest.importorskip("sunpy")``.

What this fetches:

* :class:`SunPyFidoFetcher` -- a generic federated fetcher. Given an
  ``instrument`` (e.g. ``"XRS"``, ``"AIA"``, ``"GONG"``) and Fido attributes it
  searches ``[t_start, t_end)`` and yields :class:`FluxSample` for time-series
  products (currently GOES ``XRS`` long/short channels via
  ``sunpy.timeseries``; imaging instruments are accepted for search but yield no
  flux samples and are documented as out of scope here).
* :func:`fetch_hek_flares` -- the **HEK GOES flare list** (the primary
  programmatic ground-truth catalogue, research 05 Section 1.6) as a DataFrame
  of start/peak/end/class rows for labelling and nowcast verification.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..constants import UNIT_SXR
from ..types import FluxSample, QCBit, Quantity

__all__ = ["SunPyFidoFetcher", "fetch_hek_flares", "have_sunpy"]


def have_sunpy() -> bool:
    """Return True if SunPy is importable (the archive tier is available)."""
    try:
        import sunpy  # noqa: F401
    except ImportError:
        return False
    return True


def _require_sunpy():
    """Import and return ``(Fido, attrs)`` or raise an actionable ImportError."""
    try:
        from sunpy.net import Fido
        from sunpy.net import attrs as a
    except ImportError as exc:  # pragma: no cover - exercised only without sunpy
        raise ImportError(
            "SunPyFidoFetcher requires SunPy (an optional 'real-data' "
            "dependency). Install it with `pip install \"sunpy[all]\" "
            "sunkit-instruments`, or use the offline GOESFetcher / synthetic "
            "generator (which need no network or extra packages). See "
            "ARCHITECTURE.md Section 6."
        ) from exc
    return Fido, a


class SunPyFidoFetcher:
    """Federated archive fetcher over SunPy ``Fido`` (Appendix B.1).

    Parameters
    ----------
    instrument:
        SunPy instrument name routed by ``a.Instrument(...)`` (e.g. ``"XRS"``,
        ``"AIA"``, ``"GONG"``, ``"RHESSI"``).
    **attrs:
        Extra keyword attributes forwarded to the search, e.g.
        ``satellite_number=16``, ``resolution="avg1m"`` (mapped to
        ``a.goes.SatelliteNumber`` / ``a.Resolution`` for XRS).

    Notes
    -----
    Only time-series flux products are converted to :class:`FluxSample` here
    (GOES ``XRS`` long/short via ``sunpy.timeseries.TimeSeries``). Imaging
    products (AIA/HMI) are searchable but intentionally not flattened to flux
    samples in this module -- their handling lives in the fusion/feature layers.
    """

    def __init__(self, instrument: str, **attrs: Any) -> None:
        self.instrument = instrument
        self.attrs = attrs

    def fetch(self, t_start: float, t_end: float) -> Iterator[FluxSample]:
        """Search + download via Fido and yield :class:`FluxSample` (B.1).

        Parameters
        ----------
        t_start, t_end:
            Epoch seconds UTC half-open window.

        Yields
        ------
        FluxSample
            For GOES XRS: long (``SXR_LONG``) and short (``SXR_SHORT``) channel
            samples on the GOES scale.

        Raises
        ------
        ImportError
            If SunPy is not installed (actionable message).
        NotImplementedError
            If the instrument has no flux-sample mapping here (e.g. imaging).
        """
        from datetime import datetime, timezone

        Fido, a = _require_sunpy()
        t0 = datetime.fromtimestamp(t_start, tz=timezone.utc)
        t1 = datetime.fromtimestamp(t_end, tz=timezone.utc)

        query = [a.Time(t0.isoformat(), t1.isoformat()), a.Instrument(self.instrument)]
        # map a couple of common XRS attributes if provided.
        if "satellite_number" in self.attrs:
            query.append(a.goes.SatelliteNumber(self.attrs["satellite_number"]))
        if "resolution" in self.attrs:
            query.append(a.Resolution(self.attrs["resolution"]))

        results = Fido.search(*query)
        files = Fido.fetch(results)

        if self.instrument.upper() == "XRS":
            yield from self._xrs_timeseries_to_samples(files, t_start, t_end)
            return

        raise NotImplementedError(
            f"SunPyFidoFetcher has no FluxSample mapping for instrument "
            f"{self.instrument!r}; only 'XRS' time series are flattened here. "
            f"Imaging instruments are handled by the fusion/feature layers."
        )

    def _xrs_timeseries_to_samples(
        self, files: Any, t_start: float, t_end: float
    ) -> Iterator[FluxSample]:
        """Convert downloaded GOES XRS files to long/short FluxSamples."""
        import sunpy.timeseries as ts

        goes = ts.TimeSeries(files, source="XRS", concatenate=True)
        df = goes.to_dataframe()  # columns: xrsa (short), xrsb (long)
        sat = self.attrs.get("satellite_number", "primary")
        for idx, row in df.iterrows():
            t = idx.timestamp() if hasattr(idx, "timestamp") else float(idx)
            if not (t_start <= t < t_end):
                continue
            long_v = row.get("xrsb")
            short_v = row.get("xrsa")
            if long_v is not None and long_v == long_v:  # not NaN
                yield FluxSample(
                    stream=f"goes-{sat}-long",
                    t=t,
                    value=float(long_v),
                    unit=UNIT_SXR,
                    source="Fido-XRS",
                    quantity=Quantity.SXR_LONG.value,
                    cls=_lazy_class(float(long_v)),
                    qc=QCBit.GOOD.value,
                    meta={"channel": "xrsb"},
                )
            if short_v is not None and short_v == short_v:
                yield FluxSample(
                    stream=f"goes-{sat}-short",
                    t=t,
                    value=float(short_v),
                    unit=UNIT_SXR,
                    source="Fido-XRS",
                    quantity=Quantity.SXR_SHORT.value,
                    cls=None,
                    qc=QCBit.GOOD.value,
                    meta={"channel": "xrsa"},
                )


def _lazy_class(flux_wm2: float) -> str:
    from .normalize import goes_class

    return goes_class(flux_wm2)


def fetch_hek_flares(t_start: float, t_end: float, min_class: str = "C1.0"):
    """Fetch the HEK GOES flare list as a DataFrame (Appendix B.1).

    HEK is the primary programmatic flare catalogue (it folds in the SWPC/GOES
    flare list); this is how supervised labels and nowcast-verification truth
    are built (research 05 Section 1.6). Returns one row per flare with
    start/peak/end times, GOES class, and (where available) position / active
    region.

    Parameters
    ----------
    t_start, t_end:
        Epoch seconds UTC window.
    min_class:
        Minimum GOES class (string-comparable, e.g. ``"M1.0"``); only flares
        ``>= min_class`` are returned.

    Returns
    -------
    pandas.DataFrame
        Columns: ``event_starttime, event_peaktime, event_endtime,
        fl_goescls, hpc_x, hpc_y, ar_noaanum`` (subset that exists).

    Raises
    ------
    ImportError
        If SunPy/pandas are not installed (actionable message). Any test must
        ``pytest.importorskip("sunpy")``.
    """
    from datetime import datetime, timezone

    Fido, a = _require_sunpy()
    t0 = datetime.fromtimestamp(t_start, tz=timezone.utc)
    t1 = datetime.fromtimestamp(t_end, tz=timezone.utc)
    res = Fido.search(
        a.Time(t0.isoformat(), t1.isoformat()),
        a.hek.EventType("FL"),
        a.hek.FL.GOESCls > min_class,
        a.hek.OBS.Observatory == "GOES",
    )
    hek = res["hek"]
    wanted = [
        "event_starttime",
        "event_peaktime",
        "event_endtime",
        "fl_goescls",
        "hpc_x",
        "hpc_y",
        "ar_noaanum",
    ]
    present = [c for c in wanted if c in hek.colnames]
    return hek[present].to_pandas()
