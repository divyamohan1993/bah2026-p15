"""Multi-satellite data fusion, LTT correction, cross-cal, gap-fill.

Workstream 2 (ARCHITECTURE.md Appendix C / Section 3, research doc ``06``).
Turns many heterogeneous :class:`~flarecast.fusion.schema.FusionRecord` streams
into one best-estimate light curve + uncertainty + consensus labels.

Public entry points (see each module's docstring and ARCHITECTURE.md
Appendix B.3 for the contracts):

* :mod:`~flarecast.fusion.schema` -- ``FusionRecord``, ``to_canonical_unit``
* :mod:`~flarecast.fusion.registry` -- ``SourceInfo``, ``SourceRegistry``
* :mod:`~flarecast.fusion.ltt` -- ``light_travel_correction``, ``ltt_delta_seconds``
* :mod:`~flarecast.fusion.timesync` -- grid / snap / ZOH / micro-gap interpolation
* :mod:`~flarecast.fusion.xcal` -- ``solexs_to_goes``, ``hxr_to_reference``
* :mod:`~flarecast.fusion.qc` -- despike, spectral test, QC bitmask
* :mod:`~flarecast.fusion.gapfill` -- ``fill_gaps``
* :mod:`~flarecast.fusion.fuse` -- ``inverse_variance_fuse``, ``KalmanFuser``
* :mod:`~flarecast.fusion.stereo` -- ``ipn_annulus``, ``localize_burst``
* :mod:`~flarecast.fusion.consensus` -- ``consensus_label``
* :mod:`~flarecast.fusion.pipeline` -- ``run_fusion`` -> ``FusionProduct``

Imports are **lazy** (PEP 562 ``__getattr__``) so ``import flarecast.fusion``
stays cheap and never forces every submodule -- matching the import-safe pattern
of the top-level :mod:`flarecast` package. Submodules can also be imported
directly (``from flarecast.fusion.fuse import KalmanFuser``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "FusionRecord",
    "to_canonical_unit",
    "SourceInfo",
    "SourceRegistry",
    "default_registry",
    "light_travel_correction",
    "ltt_delta_seconds",
    "build_grid",
    "snap_to_grid",
    "zero_order_hold",
    "micro_gap_interpolate",
    "solexs_to_goes",
    "hxr_to_reference",
    "fit_transfer_function",
    "median_sigma_despike",
    "spectral_shape_test",
    "qc_bitmask",
    "fill_gaps",
    "inverse_variance_fuse",
    "KalmanFuser",
    "Annulus",
    "ipn_annulus",
    "localize_burst",
    "consensus_label",
    "run_fusion",
    "FusionProduct",
]

# Map each public symbol to the submodule that defines it (for lazy loading).
_SYMBOL_MODULE: dict[str, str] = {
    "FusionRecord": "schema",
    "to_canonical_unit": "schema",
    "SourceInfo": "registry",
    "SourceRegistry": "registry",
    "default_registry": "registry",
    "light_travel_correction": "ltt",
    "ltt_delta_seconds": "ltt",
    "build_grid": "timesync",
    "snap_to_grid": "timesync",
    "zero_order_hold": "timesync",
    "micro_gap_interpolate": "timesync",
    "solexs_to_goes": "xcal",
    "hxr_to_reference": "xcal",
    "fit_transfer_function": "xcal",
    "median_sigma_despike": "qc",
    "spectral_shape_test": "qc",
    "qc_bitmask": "qc",
    "fill_gaps": "gapfill",
    "inverse_variance_fuse": "fuse",
    "KalmanFuser": "fuse",
    "Annulus": "stereo",
    "ipn_annulus": "stereo",
    "localize_burst": "stereo",
    "consensus_label": "consensus",
    "run_fusion": "pipeline",
    "FusionProduct": "pipeline",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module = _SYMBOL_MODULE.get(name)
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from .fuse import KalmanFuser, inverse_variance_fuse
    from .ltt import light_travel_correction, ltt_delta_seconds
    from .pipeline import FusionProduct, run_fusion
    from .registry import SourceInfo, SourceRegistry, default_registry
    from .schema import FusionRecord, to_canonical_unit
