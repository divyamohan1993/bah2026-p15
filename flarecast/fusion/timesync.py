"""Time synchronization across cadences (ARCHITECTURE.md Section 3.6).

Place heterogeneous streams (1 s SoLEXS/HEL1OS/GOES, 1 min irradiance, 12 min
HMI, asynchronous radio/triggers) on a **common uniform grid in the
``t_earth_utc`` frame** without inventing structure (research doc
``06 Section 2``). **Direction matters** -- this is where spurious correlations
are born:

* :func:`build_grid` -- uniform grid of ``t_earth_utc`` samples at spacing
  ``dt`` (default 1 s nowcast grid).
* :func:`snap_to_grid` -- register same-or-finer-cadence records to grid cells
  within a tolerance (average duplicates landing in one cell).
* :func:`zero_order_hold` -- forward-fill snapshot streams (e.g. HMI) onto the
  finer grid, flagged ``INTERPOLATED`` with an "age" column (a snapshot does not
  vary smoothly between samples).
* :func:`micro_gap_interpolate` -- linear interpolation across **micro-gaps
  only** (<= ``MAX_INTERP_GAP_CADENCES`` * cadence), in the **log domain for
  SXR** (flux spans decades) and never spline -- impulsive HXR uses
  previous-value/NaN to avoid overshoot-induced false precursors.

To stay dependency-free, the grid is a plain ``list[float]`` (a numpy array is
accepted and used if available) and an aligned series is a small
:class:`GridSeries` (dict-of-lists) rather than a pandas DataFrame -- this is the
"DF" the Appendix B.3 signatures refer to in the offline reference impl. numpy
is optional and guarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..constants import (
    MAX_INTERP_GAP_CADENCES,
    NOWCAST_GRID_DT_S,
)
from ..types import QCBit, QCFlag, Quantity

__all__ = [
    "GridSeries",
    "build_grid",
    "snap_to_grid",
    "zero_order_hold",
    "micro_gap_interpolate",
]

# Quantities interpolated in the log domain (flux spans A->X decades).
_LOG_DOMAIN_QUANTITIES = {Quantity.SXR_LONG.value, Quantity.SXR_SHORT.value,
                          Quantity.EUV.value}
# Quantities that are impulsive -> never interpolate across gaps (NaN/previous).
_NO_INTERP_QUANTITIES = {Quantity.HXR.value}

NAN = float("nan")


@dataclass(slots=True)
class GridSeries:
    """One source's series registered onto the common grid (offline "DF").

    A column-oriented, pandas-free container: parallel lists indexed by grid
    cell. ``value[i]`` is ``NaN`` where the cell has no datum.

    Attributes
    ----------
    grid:
        The common ``t_earth_utc`` grid (shared, not copied).
    value:
        Per-cell canonical-unit value (``NaN`` if empty).
    sigma:
        Per-cell 1-sigma (``NaN`` if empty).
    qc_flag:
        Per-cell human-readable QC token.
    qc_bitmask:
        Per-cell integer QC bitmask.
    age_s:
        Per-cell age of the held value [s] (0 for fresh, grows under ZOH);
        ``NaN`` where empty.
    n:
        Per-cell count of native samples averaged into the cell.
    provenance:
        Per-cell provenance string (``"measured"``, ``"interp:linear"``,
        ``"filled:<source>"``, ``"gap:<cause>"``); set by gap-fill.
    source_id / quantity / cadence_s:
        Identity / cadence carried for downstream gap-fill and fusion.
    """

    grid: Sequence[float]
    value: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    qc_flag: list[str] = field(default_factory=list)
    qc_bitmask: list[int] = field(default_factory=list)
    age_s: list[float] = field(default_factory=list)
    n: list[int] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    source_id: str = ""
    quantity: str = ""
    cadence_s: float = NOWCAST_GRID_DT_S

    def __post_init__(self) -> None:
        m = len(self.grid)
        if not self.value:
            self.value = [NAN] * m
        if not self.sigma:
            self.sigma = [NAN] * m
        if not self.qc_flag:
            self.qc_flag = [QCFlag.BAD.value] * m
        if not self.qc_bitmask:
            self.qc_bitmask = [QCBit.DATA_GAP.value] * m
        if not self.age_s:
            self.age_s = [NAN] * m
        if not self.n:
            self.n = [0] * m
        if not self.provenance:
            self.provenance = ["measured"] * m

    def __len__(self) -> int:
        return len(self.grid)

    def valid(self, i: int) -> bool:
        """``True`` if cell ``i`` holds a usable (non-NaN, not BAD) value."""
        v = self.value[i]
        return v == v and self.qc_flag[i] not in (QCFlag.BAD.value,)


def build_grid(
    t_start: float, t_end: float, dt: float = NOWCAST_GRID_DT_S
) -> list[float]:
    """Build a uniform ``t_earth_utc`` grid (Appendix B.3 ``build_grid``).

    Returns a list of grid times from ``t_start`` to <= ``t_end`` at spacing
    ``dt``. The epoch is anchored at ``t_start`` (callers wanting integer-second
    anchoring should pass an integer ``t_start``). ``dt`` defaults to the 1 s
    nowcast grid.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``t_end < t_start``.
    """
    if dt <= 0.0:
        raise ValueError("build_grid: dt must be > 0")
    if t_end < t_start:
        raise ValueError("build_grid: t_end must be >= t_start")
    n = int(math.floor((t_end - t_start) / dt + 1e-9)) + 1
    return [t_start + i * dt for i in range(n)]


def _grid_dt(grid: Sequence[float]) -> float:
    return (grid[1] - grid[0]) if len(grid) >= 2 else NOWCAST_GRID_DT_S


def snap_to_grid(
    records: list,
    grid: Sequence[float],
    tol: float,
    *,
    use_t_obs: bool = False,
) -> GridSeries:
    """Register same/finer-cadence records onto ``grid`` (Appendix B.3).

    For each :class:`~flarecast.fusion.schema.FusionRecord`, find the nearest
    grid cell by ``t_earth_utc`` (the fusion key; pass ``use_t_obs=True`` to key
    on the raw observed time, e.g. for PROTON streams on their own clock); assign
    if within ``tol``. Multiple same-source samples landing in one cell are
    averaged (and ``n`` records how many) -- research 06 Section 2.3. Records are
    assumed to come from a single source/quantity; the first record's identity
    is carried on the returned :class:`GridSeries`.
    """
    series = GridSeries(grid=grid)
    if not records:
        return series
    first = records[0]
    series.source_id = getattr(first, "source_id", "")
    series.quantity = getattr(first, "quantity", "")
    series.cadence_s = getattr(first, "cadence_s", _grid_dt(grid))

    if not grid:
        return series
    g0 = grid[0]
    dt = _grid_dt(grid)
    m = len(grid)

    # Accumulators for averaging duplicates per cell.
    acc_v = [0.0] * m
    acc_w = [0.0] * m       # weight = 1/sigma^2 (or count if sigma missing)
    acc_var = [0.0] * m
    counts = [0] * m
    bit_or = [0] * m

    for r in records:
        t = r.t_obs_utc if use_t_obs else r.t_earth_utc
        # Nearest cell index.
        idx = int(round((t - g0) / dt)) if dt > 0 else 0
        if idx < 0 or idx >= m:
            continue
        if abs(t - grid[idx]) > tol:
            continue
        sig = r.sigma if (r.sigma and r.sigma > 0) else NAN
        if sig == sig:
            w = 1.0 / (sig * sig)
        else:
            w = 1.0
        acc_v[idx] += w * r.value
        acc_w[idx] += w
        acc_var[idx] += 1.0 / w if w > 0 else 0.0
        counts[idx] += 1
        bit_or[idx] |= getattr(r, "qc_bitmask", 0) or 0

    for i in range(m):
        if counts[i] == 0:
            continue
        series.value[i] = acc_v[i] / acc_w[i] if acc_w[i] > 0 else NAN
        # Combined sigma of the averaged samples (inverse-variance if weighted).
        if acc_w[i] > 0:
            series.sigma[i] = math.sqrt(1.0 / acc_w[i])
        series.n[i] = counts[i]
        series.age_s[i] = 0.0
        bit = bit_or[i]
        series.qc_bitmask[i] = bit if bit else QCBit.GOOD.value
        # Map a BAD/SUSPECT bit onto the human flag; else GOOD.
        if bit & QCBit.BAD.value:
            series.qc_flag[i] = QCFlag.BAD.value
        elif bit & QCBit.SUSPECT.value:
            series.qc_flag[i] = QCFlag.SUSPECT.value
        else:
            series.qc_flag[i] = QCFlag.GOOD.value
    return series


def zero_order_hold(
    records: list,
    grid: Sequence[float],
    *,
    max_age_s: float | None = None,
) -> GridSeries:
    """Forward-fill snapshot records onto ``grid`` (Appendix B.3, ZOH).

    A magnetogram (HMI, 12 min) is a *snapshot*: pretending it varies smoothly
    between samples fabricates dynamics. So each grid cell holds the most recent
    prior record's value, flagged ``INTERPOLATED`` with the **age** (seconds
    since that record) recorded in ``age_s`` -- the fusion estimator inflates the
    variance with age and the ML model can mask it. Cells before the first record
    stay empty (``NaN``/``BAD``); cells older than ``max_age_s`` (if given) are
    dropped back to ``BAD``.
    """
    series = GridSeries(grid=grid)
    if not records:
        return series
    recs = sorted(records, key=lambda r: r.t_earth_utc)
    first = recs[0]
    series.source_id = getattr(first, "source_id", "")
    series.quantity = getattr(first, "quantity", "")
    series.cadence_s = getattr(first, "cadence_s", _grid_dt(grid))

    j = 0
    held = None
    held_t = None
    for i, t in enumerate(grid):
        while j < len(recs) and recs[j].t_earth_utc <= t + 1e-9:
            held = recs[j]
            held_t = recs[j].t_earth_utc
            j += 1
        if held is None:
            continue
        age = t - held_t
        if max_age_s is not None and age > max_age_s:
            continue
        series.value[i] = held.value
        # Inflate sigma with age (linear growth over one cadence as a floor).
        base_sigma = held.sigma if (held.sigma and held.sigma > 0) else 0.0
        age_factor = 1.0 + age / max(held.cadence_s, 1e-9)
        series.sigma[i] = base_sigma * age_factor if base_sigma > 0 else NAN
        series.age_s[i] = age
        series.n[i] = 1
        series.qc_flag[i] = QCFlag.INTERPOLATED.value
        series.qc_bitmask[i] = QCBit.INTERPOLATED.value
    return series


def micro_gap_interpolate(
    series: GridSeries,
    max_gap_s: float | None = None,
    domain: str = "log",
) -> GridSeries:
    """Linear-interpolate across micro-gaps only (Appendix B.3).

    Fills internal runs of empty cells **only if** the gap is at most
    ``max_gap_s`` (default ``MAX_INTERP_GAP_CADENCES`` * the series cadence) -
    research 06 Section 2.2. Filled cells are flagged ``INTERPOLATED``. Behaviour
    by domain / quantity:

    * **SXR / EUV** (``domain="log"`` or auto by quantity): interpolate in
      ``log10`` of the value (flux spans decades; linear-in-flux biases class).
    * **impulsive HXR**: **never interpolated** -- the gap is left for §4 filling
      (substituting a real other-sensor measurement), which is a fundamentally
      different operation than inventing a value.

    Gaps wider than the cap, and gaps at the series edges, are left as-is. Mutates
    and returns ``series``.
    """
    n = len(series)
    if n == 0:
        return series
    # Impulsive HXR is never interpolated.
    if series.quantity in _NO_INTERP_QUANTITIES:
        return series

    cadence = series.cadence_s if series.cadence_s > 0 else _grid_dt(series.grid)
    if max_gap_s is None:
        max_gap_s = MAX_INTERP_GAP_CADENCES * cadence
    use_log = (domain == "log") or (series.quantity in _LOG_DOMAIN_QUANTITIES)

    def _is_filled(i: int) -> bool:
        v = series.value[i]
        return v == v  # not NaN

    i = 0
    while i < n:
        if _is_filled(i):
            i += 1
            continue
        # Found start of a gap run [lo, hi).
        lo = i
        while i < n and not _is_filled(i):
            i += 1
        hi = i  # first filled (or n) after the gap
        left = lo - 1
        right = hi
        if left < 0 or right >= n:
            continue  # edge gap -> leave empty
        gap_span = series.grid[right] - series.grid[left]
        if gap_span > max_gap_s + 1e-9:
            continue  # too wide -> handled by gap-fill, not interpolation
        # Interpolate each empty cell linearly between left and right.
        vl, vr = series.value[left], series.value[right]
        sl, sr = series.sigma[left], series.sigma[right]
        if use_log:
            if vl <= 0 or vr <= 0:
                continue  # cannot log-interpolate non-positive endpoints
            yl, yr = math.log10(vl), math.log10(vr)
        else:
            yl, yr = vl, vr
        t_l, t_r = series.grid[left], series.grid[right]
        for j in range(lo, hi):
            frac = (series.grid[j] - t_l) / (t_r - t_l)
            y = yl + frac * (yr - yl)
            series.value[j] = (10.0 ** y) if use_log else y
            # Interpolated sigma: blend endpoint sigmas and inflate.
            base = _blend_sigma(sl, sr, frac)
            series.sigma[j] = base * 1.5 if base == base else NAN
            series.qc_flag[j] = QCFlag.INTERPOLATED.value
            series.qc_bitmask[j] = QCBit.INTERPOLATED.value
            series.age_s[j] = 0.0
            series.n[j] = 0
    return series


def _blend_sigma(sl: float, sr: float, frac: float) -> float:
    """Linear blend of two endpoint sigmas (NaN-tolerant)."""
    have_l = sl == sl
    have_r = sr == sr
    if have_l and have_r:
        return (1.0 - frac) * sl + frac * sr
    if have_l:
        return sl
    if have_r:
        return sr
    return NAN
