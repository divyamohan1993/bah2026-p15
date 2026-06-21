"""Aditya FlareCast -- automated soft + hard X-ray solar-flare nowcasting and
forecasting for ISRO BAH 2026 Problem Statement 15.

This top level is intentionally **minimal and import-safe**. It imports only
the pure-standard-library shared types from :mod:`flarecast.types`; it does
*not* eagerly import the sub-packages (``ingest``, ``synth``, ``fusion``,
``detect``, ``catalog``, ``forecast``, ``api``, ``cli``). That keeps
``import flarecast`` (and ``import flarecast.fusion.<x>`` etc.) working even
while sibling workstreams are still being built in parallel and even when
heavy optional dependencies (numpy, pandas, lightgbm, astropy, ...) are not
installed.

The end-to-end orchestrator is exposed lazily via :func:`run_pipeline`, which
imports its implementation inside the function body so that merely importing
this package never pulls in the rest of the system.

See ``ARCHITECTURE.md`` for the full design and ``Appendix A`` for the repo
layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import (
    DetectionPhase,
    DetectionState,
    FluxSample,
    QCBit,
    QCFlag,
    Quantity,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FluxSample",
    "DetectionState",
    "Quantity",
    "QCFlag",
    "QCBit",
    "DetectionPhase",
    "run_pipeline",
]

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    # Imported lazily at runtime inside run_pipeline; declared here only so
    # static type checkers can resolve the symbol without an eager import.
    pass


def run_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Run the end-to-end FlareCast pipeline (ingest -> ... -> forecast).

    This is a *lazy* entry point: the heavy implementation lives in
    ``flarecast.cli.main`` (and the sub-packages it orchestrates) and is
    imported only when this function is actually called, so importing the
    ``flarecast`` package itself stays cheap and dependency-free.

    The pipeline orchestration is delivered by Workstream 5; until then this
    raises :class:`NotImplementedError` with a pointer rather than failing at
    import time.
    """
    try:
        from .cli.main import run_pipeline as _run_pipeline
    except ImportError as exc:  # orchestrator not built yet
        raise NotImplementedError(
            "flarecast.run_pipeline is not available yet: the pipeline "
            "orchestrator (flarecast.cli.main) has not been implemented. "
            "See ARCHITECTURE.md Appendix C, Workstream 5."
        ) from exc
    return _run_pipeline(*args, **kwargs)
