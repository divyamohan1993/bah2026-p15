"""Forecasting: 30-dim features, LightGBM/TCN, lead time, evaluation.

Workstream 4 (ARCHITECTURE.md Appendix C / B.6). Every module in this package
imports with only the Python standard library -- numpy, lightgbm, scikit-learn,
torch and the ONNX converters are all imported **lazily**, inside the methods
that need them -- so ``import flarecast.forecast`` (and each submodule) succeeds
in a minimal / offline environment. The pure-python streaming feature
extractor, label builder, CV splitter, baselines, lead-time and all evaluation
metrics are fully testable with zero optional dependencies.
"""

from __future__ import annotations

from flarecast.forecast.baselines import (
    ClimatologyBaseline,
    HawkesBaseline,
    PersistenceBaseline,
)
from flarecast.forecast.cv import blocked_splits
from flarecast.forecast.evaluate import (
    bss,
    ece,
    hss,
    pod_far,
    pr_auc,
    reliability,
    report,
    roc_auc,
    tss,
)
from flarecast.forecast.features import (
    FEATURE_NAMES,
    FeatureExtractor,
    build_tcn_tensor,
    extract_features,
)
from flarecast.forecast.labels import build_labels
from flarecast.forecast.leadtime import lead_time, lt_vs_far
from flarecast.forecast.model_gbt import GBTForecaster
from flarecast.forecast.model_tcn import TCNForecaster

__all__ = [
    # features
    "FEATURE_NAMES",
    "FeatureExtractor",
    "extract_features",
    "build_tcn_tensor",
    # labels / cv
    "build_labels",
    "blocked_splits",
    # baselines
    "ClimatologyBaseline",
    "PersistenceBaseline",
    "HawkesBaseline",
    # models
    "GBTForecaster",
    "TCNForecaster",
    # lead time
    "lead_time",
    "lt_vs_far",
    # evaluation
    "tss",
    "hss",
    "bss",
    "pod_far",
    "roc_auc",
    "pr_auc",
    "reliability",
    "ece",
    "report",
]
