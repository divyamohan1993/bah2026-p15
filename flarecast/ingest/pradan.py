"""Aditya-L1 PRADAN reader: SoLEXS / HEL1OS Level-1/2 FITS (ARCHITECTURE B.1).

Aditya-L1 SoLEXS (soft X-ray, 1-22 keV) and HEL1OS (hard X-ray, 8-150 keV) are
the **primary science payloads** for PS-15 (research 01). Their data are *not*
streamed: they are disseminated as OGIP-compliant Level-1/Level-2 **FITS** light
curves + spectra through ISRO's ISSDC **PRADAN** portal.

**Access reality (document it honestly -- research 01 Section 4, research 05
Section 1.7):**

* PRADAN (``pradan.issdc.gov.in/al1/``) requires **registration + email
  verification**, and per-dataset access is **granted by administrators** (an
  "access denied" page means access has not been granted yet).
* Retrieval is **interactive / bulk download by payload + date**, not an open
  REST API. Bulk "Select" downloads have a per-selection size/file-count limit;
  sessions time out after ~30 minutes.
* **Latency is ~1 day + processing** (daily downlink -> ISSDC Level-0 -> POC
  Level-1/2). So Aditya-L1 is for **post-hoc training / validation / relabeling**
  and short-horizon nowcasting against the recent record -- the *live*
  operational nowcast runs on GOES XRS (ARCHITECTURE.md Section 11.3).
* SoLEXS public data is from ~July 2024 onward (periodic "Science Quality"
  releases).

Because there is no public API, this module does **not** attempt to log in or
download. It reads FITS files the user has already obtained from PRADAN. astropy
is an **optional** dependency (``pyproject.toml`` ``[real]`` extra): it is
imported lazily and, if absent (or the file is missing), the methods raise a
clear, actionable error. Any test exercising real FITS must
``pytest.importorskip("astropy")``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from ..constants import UNIT_HXR, UNIT_SXR
from ..types import FluxSample, QCBit, Quantity

__all__ = ["AdityaPradanReader", "have_astropy", "PRADAN_AL1_URL"]

#: PRADAN Aditya-L1 portal (login required; documented for provenance only).
PRADAN_AL1_URL = "https://pradan.issdc.gov.in/al1/"

#: Canonical stream / unit / quantity per payload.
_PAYLOAD_DEFAULTS = {
    "SOLEXS": ("solexs-sxr", UNIT_SXR, Quantity.SXR_LONG.value),
    "HEL1OS": ("hel1os-hxr", UNIT_HXR, Quantity.HXR.value),
}


def have_astropy() -> bool:
    """Return True if astropy is importable (FITS reading is available)."""
    try:
        import astropy  # noqa: F401
    except ImportError:
        return False
    return True


def _require_astropy():
    """Import and return ``astropy.io.fits`` or raise an actionable ImportError."""
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without astropy
        raise ImportError(
            "AdityaPradanReader requires astropy (an optional 'real-data' "
            "dependency) to read SoLEXS/HEL1OS FITS. Install it with "
            "`pip install astropy`. The offline pipeline does not need it: use "
            "the synthetic generator (flarecast.synth) for SoLEXS/HEL1OS-like "
            "light curves. See ARCHITECTURE.md Section 6."
        ) from exc
    return fits


class AdityaPradanReader:
    """Read Aditya-L1 SoLEXS / HEL1OS Level-1/2 FITS light curves (B.1).

    This is a *file reader*, not a downloader -- obtain the FITS from PRADAN
    first (see module docstring for the login/batch/latency reality). Methods:

    * :meth:`read_lightcurve` -- open a FITS light-curve file and return a
      tidy DataFrame of ``(t, value, ...)`` rows (requires astropy + pandas).
    * :meth:`to_samples` -- adapt that DataFrame to :class:`FluxSample` records.

    A stub-safe design: every method that touches astropy/the filesystem raises
    a clear, actionable error if astropy is missing or the file is absent, so
    importing the class is always safe in the offline path.
    """

    #: common time-column names seen in OGIP light-curve BINTABLEs.
    _TIME_COLS = ("TIME", "time", "T", "MET", "BARYTIME")
    #: common rate/count column names.
    _RATE_COLS = ("RATE", "rate", "COUNTS", "counts", "COUNT_RATE", "FLUX", "flux")

    def __init__(self, base_url: str = PRADAN_AL1_URL) -> None:
        #: portal base (informational; no network calls are made).
        self.base_url = base_url

    def read_lightcurve(self, fits_path: str, payload: str):
        """Read a SoLEXS/HEL1OS FITS light curve into a DataFrame (B.1).

        Parameters
        ----------
        fits_path:
            Path to a Level-1/Level-2 FITS light-curve file downloaded from
            PRADAN.
        payload:
            ``"SoLEXS"`` or ``"HEL1OS"`` (case-insensitive); selects default
            units/quantity for downstream :meth:`to_samples`.

        Returns
        -------
        pandas.DataFrame
            A tidy frame with at least ``t`` (epoch seconds UTC) and ``value``
            columns, plus any other light-curve columns found in the BINTABLE.

        Raises
        ------
        ImportError
            If astropy / pandas are not installed (actionable).
        FileNotFoundError
            If ``fits_path`` does not exist (actionable -- download from PRADAN).
        ValueError
            If no recognisable time/rate columns are found in the FITS file.
        """
        key = payload.upper()
        if key not in _PAYLOAD_DEFAULTS:
            raise ValueError(
                f"payload must be 'SoLEXS' or 'HEL1OS', got {payload!r}"
            )
        if not os.path.isfile(fits_path):
            raise FileNotFoundError(
                f"FITS file {fits_path!r} not found. Aditya-L1 SoLEXS/HEL1OS "
                f"data must be downloaded (login required) from PRADAN "
                f"({self.base_url}); there is no public streaming API. For "
                f"offline work use the synthetic generator (flarecast.synth)."
            )
        fits = _require_astropy()
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "read_lightcurve returns a pandas DataFrame; install pandas, or "
                "read the FITS yourself and use normalize_generic()."
            ) from exc

        with fits.open(fits_path) as hdul:
            table_hdu = self._first_bintable(hdul)
            if table_hdu is None:
                raise ValueError(
                    f"no BINTABLE light-curve extension found in {fits_path!r}"
                )
            data = table_hdu.data
            header = table_hdu.header
            colnames = list(data.columns.names)

            tcol = self._pick_column(colnames, self._TIME_COLS)
            rcol = self._pick_column(colnames, self._RATE_COLS)
            if tcol is None or rcol is None:
                raise ValueError(
                    f"could not identify time/rate columns in {fits_path!r}; "
                    f"found columns {colnames!r}"
                )

            # mission epoch -> epoch seconds. OGIP convention: TIME is seconds
            # since MJDREF (+ optional TIMEZERO). Convert MJDREF to unix epoch.
            mjdref = float(header.get("MJDREF", header.get("MJDREFI", 0)) or 0)
            timezero = float(header.get("TIMEZERO", 0) or 0)
            # MJD 40587 = 1970-01-01 (unix epoch).
            epoch_offset_s = (mjdref - 40587.0) * 86400.0

            raw_t = [float(v) for v in data[tcol]]
            t = [tv + timezero + epoch_offset_s for tv in raw_t]
            value = [float(v) for v in data[rcol]]

            frame: dict[str, list] = {"t": t, "value": value}
            # carry through any extra numeric columns for provenance/features.
            for c in colnames:
                if c in (tcol, rcol):
                    continue
                try:
                    frame[c] = [float(v) for v in data[c]]
                except (TypeError, ValueError):
                    continue
            df = pd.DataFrame(frame)
            df.attrs["payload"] = key
            df.attrs["fits_path"] = fits_path
            return df

    def to_samples(self, df, stream: str, quantity: str) -> Iterator[FluxSample]:
        """Adapt a light-curve DataFrame to :class:`FluxSample` records (B.1).

        Parameters
        ----------
        df:
            DataFrame from :meth:`read_lightcurve` (must have ``t`` and
            ``value`` columns).
        stream:
            Canonical stream id to stamp (e.g. ``"solexs-sxr"``).
        quantity:
            One of the :class:`~flarecast.types.Quantity` values (string).

        Yields
        ------
        FluxSample
            One per row. SoLEXS soft samples carry W m^-2; HEL1OS hard samples
            carry counts/s (per the payload defaults).

        Raises
        ------
        ValueError
            If the DataFrame lacks the required ``t`` / ``value`` columns.
        """
        cols = set(getattr(df, "columns", []))
        if "t" not in cols or "value" not in cols:
            raise ValueError(
                "to_samples expects a DataFrame with 't' and 'value' columns "
                "(as returned by read_lightcurve)."
            )
        payload = df.attrs.get("payload") if hasattr(df, "attrs") else None
        soft_quantities = (Quantity.SXR_LONG.value, Quantity.SXR_SHORT.value)
        unit = UNIT_SXR if quantity in soft_quantities else UNIT_HXR
        source = f"AdityaL1-{payload}" if payload else "AdityaL1"
        extra_cols = [c for c in df.columns if c not in ("t", "value")]
        for row in df.itertuples(index=False):
            d = row._asdict()
            meta: dict[str, Any] = {c: d[c] for c in extra_cols if c in d}
            yield FluxSample(
                stream=stream,
                t=float(d["t"]),
                value=float(d["value"]),
                unit=unit,
                source=source,
                quantity=quantity,
                cls=None,
                qc=QCBit.GOOD.value,
                meta=meta or None,
            )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_bintable(hdul: Any):
        """Return the first BINTABLE HDU in an open FITS file, or None."""
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and hasattr(hdu, "columns"):
                return hdu
        return None

    @staticmethod
    def _pick_column(colnames: list[str], candidates: tuple[str, ...]) -> str | None:
        """Return the first candidate column name present in ``colnames``."""
        lower = {c.lower(): c for c in colnames}
        for cand in candidates:
            if cand in colnames:
                return cand
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None
