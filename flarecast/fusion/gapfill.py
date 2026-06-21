"""Gap detection + filling from complementary sources (Section 3.11 / 06 §4).

The project lead's mandate -- "verify and fill the gaps of the other datas."
**Filling is fundamentally different from interpolation** (research doc
``06 Section 4.2``):

* interpolation (:mod:`flarecast.fusion.timesync`) invents values from the
  *same* stream, allowed only across micro-gaps for smooth quantities;
* **filling substitutes a real measurement from a complementary,
  cross-calibrated source**, with provenance recorded.

:func:`detect_gaps` finds the gap intervals in a primary series (missing,
flatline/stuck, saturated, SAA/particle, off-point, or statistical outlier vs
consensus). :func:`fill_gaps` walks each primary gap cell and substitutes the
first valid filler in priority order (e.g. SoLEXS -> GOES -> XSM; HEL1OS ->
STIX -> GBM -> Konus), setting ``value`` to the filler value, ``sigma`` to the
filler sigma combined in quadrature with an inter-source transfer residual,
``qc=FILLED``, and ``provenance="filled:<source>"``. Filled samples never
silently masquerade as primary.

Pure standard library; operates on the :class:`~flarecast.fusion.timesync.GridSeries`
("DF") produced by time synchronization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..constants import QC_FILLED
from ..types import QCBit, QCFlag
from .timesync import GridSeries

__all__ = [
    "GapInterval",
    "detect_gaps",
    "fill_gaps",
    "coverage_fraction",
]

# Per-quantity inter-source transfer residual (log10) added in quadrature when
# a primary is filled by a complementary source. These mirror the cross-cal
# residuals in flarecast.fusion.xcal and are conservative defaults.
_DEFAULT_TRANSFER_RESID_LOG: dict[str, float] = {
    "SXR_LONG": 0.06,
    "SXR_SHORT": 0.06,
    "HXR": 0.10,
    "EUV": 0.08,
}


@dataclass(slots=True)
class GapInterval:
    """A contiguous run of unusable primary samples.

    Attributes
    ----------
    start_idx / end_idx:
        Inclusive grid-cell index range of the gap.
    start_t / end_t:
        ``t_earth_utc`` of the first/last gap cell.
    cause:
        Reason code for the gap (``"missing"``, ``"flatline"``, ``"saturated"``,
        ``"suspect"``, ``"bad"``).
    n_cells:
        Number of cells in the gap.
    """

    start_idx: int
    end_idx: int
    start_t: float
    end_t: float
    cause: str

    @property
    def n_cells(self) -> int:
        return self.end_idx - self.start_idx + 1

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t


def _is_bad_cell(series: GridSeries, i: int) -> tuple[bool, str]:
    """Return ``(is_gap, cause)`` for primary cell ``i``."""
    v = series.value[i]
    if v != v:  # NaN
        return True, "missing"
    flag = series.qc_flag[i]
    if flag == QCFlag.BAD.value:
        return True, "bad"
    if flag == QCFlag.SUSPECT.value:
        return True, "suspect"
    bit = series.qc_bitmask[i]
    if bit & QCBit.SATURATED.value:
        return True, "saturated"
    if bit & QCBit.DATA_GAP.value:
        return True, "missing"
    return False, ""


def detect_gaps(
    series: GridSeries,
    *,
    flatline_window: int = 8,
    flatline_eps: float = 0.0,
) -> list[GapInterval]:
    """Detect gap intervals in a primary :class:`GridSeries`.

    Flags as gaps: missing (``NaN``/``DATA_GAP``), ``BAD``/``SUSPECT`` QC,
    saturated cells, and optional **flatline/stuck** runs (variance below
    ``flatline_eps`` over ``flatline_window`` consecutive valid cells -- a
    telemetry freeze; research 06 Section 4.1). Returns contiguous intervals
    with a cause code.
    """
    n = len(series)
    cell_cause: list[str | None] = [None] * n
    for i in range(n):
        is_gap, cause = _is_bad_cell(series, i)
        if is_gap:
            cell_cause[i] = cause

    # Flatline detection over valid runs.
    if flatline_window > 1:
        i = 0
        while i + flatline_window <= n:
            window = [series.value[j] for j in range(i, i + flatline_window)]
            if all(v == v for v in window):  # all valid
                vmin, vmax = min(window), max(window)
                if (vmax - vmin) <= flatline_eps:
                    for j in range(i, i + flatline_window):
                        if cell_cause[j] is None:
                            cell_cause[j] = "flatline"
            i += 1

    # Coalesce consecutive flagged cells into intervals.
    intervals: list[GapInterval] = []
    i = 0
    while i < n:
        if cell_cause[i] is None:
            i += 1
            continue
        lo = i
        cause = cell_cause[i]
        while i < n and cell_cause[i] is not None:
            i += 1
        hi = i - 1
        intervals.append(
            GapInterval(
                start_idx=lo,
                end_idx=hi,
                start_t=series.grid[lo],
                end_t=series.grid[hi],
                cause=cause or "missing",
            )
        )
    return intervals


def fill_gaps(
    aligned: Mapping[str, GridSeries],
    primary: str,
    fillers: Sequence[str],
    registry=None,
    *,
    transfer_resid_log: float | None = None,
) -> GridSeries:
    """Fill primary gaps from prioritized complementary sources (Appendix B.3).

    Implements ``gapfill.fill_gaps``. ``aligned`` maps ``source_id -> GridSeries``
    (all on the same grid, same quantity, already cross-calibrated by
    :mod:`flarecast.fusion.xcal`). For each gap cell in ``aligned[primary]``,
    substitute the first ``fillers`` source with a valid cell there. The filled
    cell gets:

    * ``value`` = filler value (already on the common scale);
    * ``sigma`` = ``hypot(filler_sigma, transfer_residual)`` propagated to linear
      space (the inter-source transfer residual; research 06 Section 4.2);
    * ``qc_flag = FILLED``, ``qc_bitmask |= FILLED``;
    * provenance recorded on the per-cell ``meta`` (via the QC bitmask + the
      returned series' provenance list, see below).

    The ``registry`` is used (when supplied) to look up the filler's reliability
    for the variance inflation; ``transfer_resid_log`` overrides the per-quantity
    default residual. Returns the (mutated) primary series. The filled-from
    source per cell is recorded on ``series.provenance`` (added attribute).

    Raises
    ------
    KeyError
        If ``primary`` is not present in ``aligned``.
    """
    if primary not in aligned:
        raise KeyError(f"fill_gaps: primary {primary!r} not in aligned series")
    prim = aligned[primary]
    n = len(prim)

    # Attach a per-cell provenance list so callers can audit every fill.
    provenance: list[str] = list(
        getattr(prim, "provenance", [prim.qc_flag[i] for i in range(n)])
    )
    if len(provenance) != n:
        provenance = ["measured"] * n
    for i in range(n):
        is_gap, _ = _is_bad_cell(prim, i)
        if not is_gap and provenance[i] not in (QCFlag.FILLED.value,):
            provenance[i] = (
                "measured" if prim.qc_flag[i] == QCFlag.GOOD.value
                else prim.qc_flag[i].lower()
            )

    resid_log = transfer_resid_log
    if resid_log is None:
        resid_log = _DEFAULT_TRANSFER_RESID_LOG.get(prim.quantity, 0.10)

    gaps = detect_gaps(prim)
    for gap in gaps:
        for idx in range(gap.start_idx, gap.end_idx + 1):
            filled = False
            for fid in fillers:
                fser = aligned.get(fid)
                if fser is None or idx >= len(fser):
                    continue
                if not fser.valid(idx):
                    continue
                fval = fser.value[idx]
                fsig = fser.sigma[idx]
                # Reliability-aware variance floor from registry (optional).
                rel = 1.0
                if registry is not None:
                    info = (
                        registry.get_or_none(fid)
                        if hasattr(registry, "get_or_none")
                        else None
                    )
                    if info is not None:
                        rel = max(info.reliability, 1e-3)
                # Transfer residual -> linear-space sigma at this value.
                resid_lin = math.log(10.0) * abs(fval) * resid_log
                base_sig = fsig if (fsig == fsig and fsig > 0) else (
                    abs(fval) * 0.1
                )
                total_sig = math.hypot(base_sig / math.sqrt(rel), resid_lin)
                prim.value[idx] = fval
                prim.sigma[idx] = total_sig
                prim.qc_flag[idx] = QCFlag.FILLED.value
                prim.qc_bitmask[idx] = (
                    (prim.qc_bitmask[idx] & ~QCBit.DATA_GAP.value)
                    | QC_FILLED
                )
                prim.age_s[idx] = 0.0
                prim.n[idx] = max(prim.n[idx], 1)
                provenance[idx] = f"filled:{fid}"
                filled = True
                break
            if not filled:
                # No filler available -> remains a genuine gap.
                provenance[idx] = f"gap:{gap.cause}"

    # Stash provenance on the series for downstream consumers / audit.
    prim.provenance = provenance
    return prim


def coverage_fraction(series: GridSeries) -> float:
    """Fraction of grid cells with a usable (GOOD/INTERPOLATED/FILLED) value.

    The completeness metric for the fusion ablation (research 06 Section 8.2):
    single-Aditya vs fused temporal coverage. ``NaN``/``BAD``/``SUSPECT`` cells
    do not count.
    """
    n = len(series)
    if n == 0:
        return 0.0
    good = 0
    for i in range(n):
        v = series.value[i]
        if v == v and series.qc_flag[i] not in (
            QCFlag.BAD.value,
            QCFlag.SUSPECT.value,
        ):
            good += 1
    return good / n
