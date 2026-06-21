"""Fusion orchestration: ``run_fusion`` -> a fused best-estimate product.

Ties the workstream together (ARCHITECTURE.md Section 3.4 / research doc
``06 Section 10``): ingest LTT-correct -> QC/despike -> cross-calibrate ->
time-sync -> gap-fill -> Kalman fuse per quantity -> (stereoscopy/IPN ->
consensus labels). Produces a single best-estimate light curve with a 1-sigma
uncertainty band, per-sample provenance, and a data-quality score per quantity.

Implements Appendix B.3 ``pipeline.run_fusion`` returning a
:class:`FusionProduct`. Everything is **pure standard library** so the whole
pipeline runs offline with no numpy/pandas; the per-quantity fusion uses
:class:`flarecast.fusion.fuse.KalmanFuser`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..constants import NOWCAST_GRID_DT_S
from ..types import QCFlag, Quantity
from .fuse import KalmanFuser
from .gapfill import coverage_fraction, fill_gaps
from .ltt import light_travel_correction_record
from .registry import SourceRegistry, default_registry
from .timesync import GridSeries, build_grid, snap_to_grid, zero_order_hold

__all__ = [
    "FusedSeries",
    "FusionProduct",
    "run_fusion",
]

NAN = float("nan")

# Quantities fused into best-estimate light curves (research 06 Section 10).
_FUSED_QUANTITIES = (
    Quantity.SXR_LONG.value,
    Quantity.SXR_SHORT.value,
    Quantity.HXR.value,
)
# Snapshot quantities placed by zero-order-hold rather than snap-to-grid.
_SNAPSHOT_QUANTITIES = {Quantity.MAGSCALAR.value}


@dataclass(slots=True)
class FusedSeries:
    """Fused best-estimate light curve for one quantity.

    Attributes
    ----------
    grid:
        Common ``t_earth_utc`` grid [s].
    value:
        Best-estimate value per cell (canonical unit; ``NaN`` if no data ever).
    sigma:
        1-sigma uncertainty per cell (linear space).
    quantity:
        The fused quantity.
    quality:
        Per-cell data-quality score Q(t) in [0, 1] (research 06 Section 7.4).
    provenance:
        Per-cell list of contributing ``source_id``s.
    coverage:
        Fraction of cells with a valid fused estimate.
    """

    grid: Sequence[float]
    value: list[float]
    sigma: list[float]
    quantity: str
    quality: list[float] = field(default_factory=list)
    provenance: list[list[str]] = field(default_factory=list)
    coverage: float = 0.0


@dataclass(slots=True)
class FusionProduct:
    """Output of :func:`run_fusion` (research 06 Section 10 ``FusionProduct``).

    Attributes
    ----------
    products:
        Map ``quantity -> FusedSeries`` (the fused light curves).
    grid:
        The common ``t_earth_utc`` grid shared by all products.
    aligned:
        Per-quantity map ``source_id -> GridSeries`` after gap-fill (audit).
    bursts:
        IPN annuli from stereoscopy (empty unless a trigger table is supplied).
    labels:
        Consensus master-catalogue labels (empty unless catalogues supplied).
    """

    products: dict[str, FusedSeries]
    grid: list[float]
    aligned: dict[str, dict[str, GridSeries]] = field(default_factory=dict)
    bursts: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def _grid_bounds(streams: Mapping[str, Sequence], grid_dt: float) -> tuple[float, float]:
    """Earliest/latest ``t_earth_utc`` across all records (LTT applied)."""
    t_min = math.inf
    t_max = -math.inf
    for recs in streams.values():
        for r in recs:
            t = r.t_earth_utc if r.t_earth_utc else r.t_obs_utc
            if t < t_min:
                t_min = t
            if t > t_max:
                t_max = t
    if t_min is math.inf:
        return 0.0, 0.0
    return t_min, t_max


def run_fusion(
    streams: Mapping[str, Sequence],
    registry: SourceRegistry | None = None,
    fill_priority: Mapping[str, Sequence[str]] | None = None,
    grid_dt: float = NOWCAST_GRID_DT_S,
    *,
    kalman_q: float = 1e-4,
    trigger_table: Sequence | None = None,
    catalogues: Sequence[Sequence] | None = None,
    catalogue_reliabilities: Mapping[str, float] | None = None,
) -> FusionProduct:
    """Run the multi-satellite fusion pipeline (Appendix B.3 ``run_fusion``).

    Parameters
    ----------
    streams:
        Map ``source_id -> list[FusionRecord]`` (raw, pre-LTT acceptable -- LTT
        is applied here if ``t_earth_utc`` is unset).
    registry:
        :class:`~flarecast.fusion.registry.SourceRegistry`; defaults to
        :func:`~flarecast.fusion.registry.default_registry`.
    fill_priority:
        Map ``quantity -> [primary, filler1, filler2, ...]`` source ids used by
        :func:`~flarecast.fusion.gapfill.fill_gaps`. If omitted, no gap-fill is
        performed (each source is fused on its own merits).
    grid_dt:
        Common-grid spacing [s] (default 1 s nowcast grid).
    kalman_q:
        Process-noise scale for the per-quantity :class:`KalmanFuser`.
    trigger_table:
        Optional IPN trigger observations -> ``bursts`` (annuli).
    catalogues / catalogue_reliabilities:
        Optional inputs to consensus labeling -> ``labels``.

    Returns
    -------
    FusionProduct
    """
    reg = registry if registry is not None else default_registry()

    # --- 1-2. LTT-correct every record to the Earth/L1 frame ---
    for recs in streams.values():
        for r in recs:
            if not r.t_earth_utc:
                light_travel_correction_record(r)

    # --- build the common grid ---
    t_start, t_end = _grid_bounds(streams, grid_dt)
    grid = build_grid(t_start, t_end, grid_dt) if t_end >= t_start else [t_start]

    # --- group records by quantity, then by source; place on the grid ---
    by_quantity: dict[str, dict[str, list]] = {}
    for source_id, recs in streams.items():
        for r in recs:
            by_quantity.setdefault(r.quantity, {}).setdefault(
                r.source_id, []
            ).append(r)

    aligned: dict[str, dict[str, GridSeries]] = {}
    for quantity, by_source in by_quantity.items():
        aligned[quantity] = {}
        for source_id, recs in by_source.items():
            if quantity in _SNAPSHOT_QUANTITIES:
                series = zero_order_hold(recs, grid)
            else:
                series = snap_to_grid(recs, grid, tol=0.5 * grid_dt)
            aligned[quantity][source_id] = series

    # --- gap-fill each quantity that has a fill priority ---
    if fill_priority:
        for quantity, priority in fill_priority.items():
            if quantity not in aligned or not priority:
                continue
            primary = priority[0]
            fillers = list(priority[1:])
            if primary in aligned[quantity]:
                fill_gaps(aligned[quantity], primary, fillers, reg)

    # --- fuse each quantity with a sequential-update Kalman filter ---
    products: dict[str, FusedSeries] = {}
    for quantity in _FUSED_QUANTITIES:
        if quantity not in aligned:
            continue
        products[quantity] = _fuse_quantity(
            grid, aligned[quantity], quantity, reg, kalman_q
        )

    product = FusionProduct(products=products, grid=list(grid), aligned=aligned)

    # --- optional stereoscopy / IPN ---
    if trigger_table:
        from .stereo import localize_burst

        product.bursts = localize_burst(trigger_table)

    # --- optional consensus labeling ---
    if catalogues is not None and catalogue_reliabilities is not None:
        from .consensus import consensus_label

        product.labels = consensus_label(catalogues, catalogue_reliabilities)

    return product


def _fuse_quantity(
    grid: Sequence[float],
    by_source: Mapping[str, GridSeries],
    quantity: str,
    registry: SourceRegistry,
    kalman_q: float,
) -> FusedSeries:
    """Sequential-update Kalman fusion of all sources for one quantity."""
    n = len(grid)
    init_log = -7.0 if quantity in (
        Quantity.SXR_LONG.value, Quantity.SXR_SHORT.value
    ) else 1.0
    kf = KalmanFuser(q=kalman_q, init_log_flux=init_log)

    value = [NAN] * n
    sigma = [NAN] * n
    quality = [0.0] * n
    provenance: list[list[str]] = [[] for _ in range(n)]

    prev_t = grid[0] if grid else 0.0
    for i in range(n):
        dt = (grid[i] - prev_t) if i > 0 else 0.0
        prev_t = grid[i]

        # Collect valid measurements at this cell from every source.
        measurements: list[tuple[float, float, float]] = []
        contributors: list[str] = []
        spread_vals: list[float] = []
        n_good = 0
        for source_id, series in by_source.items():
            if i >= len(series) or not series.valid(i):
                continue
            v = series.value[i]
            if v <= 0.0:
                continue
            s = series.sigma[i]
            # sigma in log10 space: sigma_log ~ sigma_lin / (ln10 * v).
            if not (s == s and s > 0):
                # Fall back to the registry's typical sigma.
                info = registry.get_or_none(source_id)
                s = info.typical_sigma if info else abs(v) * 0.1
            sigma_log = s / (math.log(10.0) * abs(v))
            sigma_log = max(sigma_log, 1e-6)
            info = registry.get_or_none(source_id)
            rel = info.reliability if info else 1.0
            # QC weight: down-weight interpolated/filled cells.
            if series.qc_flag[i] == QCFlag.INTERPOLATED.value:
                rel *= 0.5
            elif series.qc_flag[i] == QCFlag.FILLED.value:
                rel *= 0.7
            measurements.append((math.log10(v), sigma_log, rel))
            contributors.append(source_id)
            spread_vals.append(math.log10(v))
            if series.qc_flag[i] == QCFlag.GOOD.value:
                n_good += 1

        f_hat, s_hat = kf.step(dt, measurements)

        if measurements or kf._initialized:  # noqa: SLF001 - internal flag read
            value[i] = f_hat
            sigma[i] = s_hat
            provenance[i] = contributors
            quality[i] = _quality_score(
                n_good, len(measurements), spread_vals
            )

    fused = FusedSeries(
        grid=grid,
        value=value,
        sigma=sigma,
        quantity=quantity,
        quality=quality,
        provenance=provenance,
    )
    # Coverage via a GridSeries view (reuse the helper's definition).
    cov_series = GridSeries(
        grid=grid,
        value=list(value),
        qc_flag=[
            QCFlag.GOOD.value if (v == v) else QCFlag.BAD.value for v in value
        ],
    )
    fused.coverage = coverage_fraction(cov_series)
    return fused


def _quality_score(
    n_good: int, n_contrib: int, spread_log: Sequence[float]
) -> float:
    """Data-quality score Q(t) in [0, 1] (research 06 Section 7.4).

    Combines the fraction of GOOD contributors with cross-sensor agreement
    (``1 - normalized spread``). High when several GOOD sensors agree.
    """
    if n_contrib == 0:
        return 0.0
    f_good = n_good / n_contrib
    if len(spread_log) >= 2:
        mean = sum(spread_log) / len(spread_log)
        var = sum((x - mean) ** 2 for x in spread_log) / len(spread_log)
        spread = math.sqrt(var)
        # 0.5 decade spread -> agreement ~ 0; tighter -> ~1.
        agreement = max(0.0, 1.0 - spread / 0.5)
    else:
        agreement = 1.0
    return f_good * agreement
