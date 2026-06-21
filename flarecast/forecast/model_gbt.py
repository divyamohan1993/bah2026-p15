"""Tier-1 production forecaster: gradient-boosted trees (LightGBM / sklearn).

Governing research: ``docs/research/04-forecasting-models.md`` Section 0 / 3.1
/ 7 and ARCHITECTURE.md Section 5.1.

GBT on the ~30-dim engineered feature vector is the **deployed / edge** model:
a few hundred shallow tree traversals per step (microseconds), exportable to
ONNX for Cloudflare Workers AI, with isotonic calibration bolted on so the
output is a trustworthy probability (research doc 04 Section 0).

Dependency philosophy (this file MUST import with only the standard library):
every heavy dependency -- **lightgbm**, **scikit-learn**, **numpy**,
**onnxmltools/skl2onnx** -- is imported **lazily inside the method that needs
it**. Constructing a :class:`GBTForecaster` never imports them. ``fit`` prefers
LightGBM and falls back to scikit-learn's ``GradientBoostingClassifier``; only
if *neither* backend is importable does ``fit`` raise a clear
:class:`RuntimeError`. The matching test uses ``pytest.importorskip`` so it
skips cleanly when no backend is present.
"""

from __future__ import annotations

import pickle
from typing import Any

from flarecast.constants import FORECAST_DEFAULT_CLASS_THRESHOLD

__all__ = ["GBTForecaster"]


class GBTForecaster:
    """Gradient-boosted-tree flare forecaster (LightGBM, sklearn fallback).

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, num_leaves:
        Standard boosting hyper-parameters (modest defaults suited to the small,
        tabular space-weather feature set; research doc 04 Section 3.1 warns
        against over-fitting the rare positive class).
    class_threshold:
        Bookkeeping label for which positive-class threshold (``">=C"`` etc.)
        this model was trained for; stored on the instance and in ``save``.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        class_threshold: str = FORECAST_DEFAULT_CLASS_THRESHOLD,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.num_leaves = int(num_leaves)
        self.class_threshold = class_threshold
        self.random_state = int(random_state)

        # Set lazily in fit().
        self._model: Any = None
        self._backend: str | None = None   # "lightgbm" | "sklearn"
        self._calibrator: Any = None       # isotonic / sigmoid, optional
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Fitting.
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> GBTForecaster:
        """Train the booster on ``(X, y)`` with optional ``sample_weight``.

        Prefers LightGBM (``scale_pos_weight`` set from the class balance to
        counter rare positives, research doc 04 Section 6); on ImportError falls
        back to scikit-learn's ``GradientBoostingClassifier``. Raises
        ``RuntimeError`` only if neither backend is importable.
        """
        import numpy as np  # lazy

        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y).astype(int).ravel()
        if Xa.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
        self._n_features = int(Xa.shape[1])

        n_pos = int((ya == 1).sum())
        n_neg = int((ya == 0).sum())
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        # --- Try LightGBM first. -------------------------------------------
        try:
            import lightgbm as lgb  # lazy, optional

            self._model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                scale_pos_weight=spw,
                random_state=self.random_state,
                verbose=-1,
            )
            self._model.fit(Xa, ya, sample_weight=sample_weight)
            self._backend = "lightgbm"
            return self
        except ImportError:
            pass  # fall through to sklearn

        # --- Fallback: scikit-learn GradientBoostingClassifier. ------------
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # lazy
        except ImportError as exc:
            raise RuntimeError(
                "GBTForecaster.fit requires lightgbm or scikit-learn, but "
                "neither is importable. Install one (pip install lightgbm) to "
                "train; the offline pure-python tests do not exercise fit()."
            ) from exc

        # sklearn's GBC has no scale_pos_weight; emulate via sample_weight.
        if sample_weight is None:
            sw = np.where(ya == 1, spw, 1.0)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
        gbc = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        gbc.fit(Xa, ya, sample_weight=sw)
        self._model = gbc
        self._backend = "sklearn"
        return self

    # ------------------------------------------------------------------
    # Prediction.
    # ------------------------------------------------------------------
    def _raw_proba(self, X: Any):
        """Uncalibrated P(positive) as a 1-D numpy array."""
        import warnings

        import numpy as np  # lazy

        if self._model is None:
            raise RuntimeError(
                "GBTForecaster.predict_proba called before fit()/load()."
            )

        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(1, -1)
        # Suppress the benign "X does not have valid feature names" UserWarning
        # emitted by the sklearn-wrapped LightGBM when fit on a bare ndarray and
        # then called with one; the math is unaffected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            proba = self._model.predict_proba(Xa)
        # predict_proba -> (n, 2); take the positive-class column.
        p = np.asarray(proba)[:, 1]
        return p

    def predict_proba(self, X: Any):
        """Calibrated P(flare in next N) as a 1-D numpy array in ``(0, 1)``.

        If :meth:`calibrate` has been run, the isotonic/sigmoid map is applied
        to the raw booster probability; otherwise the raw probability is
        returned. numpy is imported lazily.
        """
        import numpy as np  # lazy

        p = self._raw_proba(X)
        if self._calibrator is not None:
            p = self._calibrator.predict(p)
        return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    # ------------------------------------------------------------------
    # Calibration.
    # ------------------------------------------------------------------
    def calibrate(self, X: Any, y: Any, method: str = "isotonic") -> GBTForecaster:
        """Fit a probability calibrator on a held-out fold.

        ``method="isotonic"`` (default) fits a monotone isotonic regression of
        observed frequency on raw probability; ``method="sigmoid"`` fits Platt
        scaling (a 1-D logistic). The calibrator is applied inside
        :meth:`predict_proba`. scikit-learn is imported lazily; if it is
        missing, a clear ``RuntimeError`` is raised (calibration is inherently
        an sklearn feature here).
        """
        import numpy as np  # lazy

        ya = np.asarray(y).astype(int).ravel()
        raw = self._raw_proba(X)

        if method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression  # lazy
            except ImportError as exc:
                raise RuntimeError(
                    "calibrate(method='isotonic') requires scikit-learn."
                ) from exc
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw, ya)
            self._calibrator = _IsotonicWrapper(iso)
        elif method == "sigmoid":
            try:
                from sklearn.linear_model import LogisticRegression  # lazy
            except ImportError as exc:
                raise RuntimeError(
                    "calibrate(method='sigmoid') requires scikit-learn."
                ) from exc
            lr = LogisticRegression()
            lr.fit(raw.reshape(-1, 1), ya)
            self._calibrator = _SigmoidWrapper(lr)
        else:
            raise ValueError(f"unknown calibration method {method!r}")
        return self

    # ------------------------------------------------------------------
    # ONNX export.
    # ------------------------------------------------------------------
    def to_onnx(self, path: str) -> None:
        """Export the booster to ONNX at ``path`` (edge inference artifact).

        Uses ``onnxmltools`` for a LightGBM booster or ``skl2onnx`` for the
        sklearn fallback. Both, plus their ``onnx`` dependency, are imported
        lazily; a missing converter raises a clear ``RuntimeError``. Note: only
        the *booster* is exported; the isotonic calibrator (if any) is a tiny
        monotone map applied in the host runtime (or folded in by the caller).
        """
        if self._model is None:
            raise RuntimeError("to_onnx called before fit()/load().")
        if self._n_features is None:
            raise RuntimeError("model has no known feature count.")

        n_feat = self._n_features
        if self._backend == "lightgbm":
            try:
                from onnxmltools import convert_lightgbm  # lazy
                from onnxmltools.convert.common.data_types import FloatTensorType
            except ImportError as exc:
                raise RuntimeError(
                    "to_onnx for a LightGBM model requires onnxmltools "
                    "(pip install onnxmltools onnx)."
                ) from exc
            initial_types = [("input", FloatTensorType([None, n_feat]))]
            onx = convert_lightgbm(self._model, initial_types=initial_types)
            with open(path, "wb") as fh:
                fh.write(onx.SerializeToString())
        else:
            try:
                from skl2onnx import convert_sklearn  # lazy
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError as exc:
                raise RuntimeError(
                    "to_onnx for the sklearn model requires skl2onnx "
                    "(pip install skl2onnx onnx)."
                ) from exc
            initial_types = [("input", FloatTensorType([None, n_feat]))]
            onx = convert_sklearn(self._model, initial_types=initial_types)
            with open(path, "wb") as fh:
                fh.write(onx.SerializeToString())

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Pickle the full forecaster (model + calibrator + metadata)."""
        blob = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "class_threshold": self.class_threshold,
            "random_state": self.random_state,
            "model": self._model,
            "backend": self._backend,
            "calibrator": self._calibrator,
            "n_features": self._n_features,
        }
        with open(path, "wb") as fh:
            pickle.dump(blob, fh)

    @classmethod
    def load(cls, path: str) -> GBTForecaster:
        """Load a forecaster previously written by :meth:`save`."""
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(
            n_estimators=blob.get("n_estimators", 200),
            learning_rate=blob.get("learning_rate", 0.05),
            max_depth=blob.get("max_depth", 4),
            num_leaves=blob.get("num_leaves", 15),
            class_threshold=blob.get("class_threshold", FORECAST_DEFAULT_CLASS_THRESHOLD),
            random_state=blob.get("random_state", 0),
        )
        obj._model = blob.get("model")
        obj._backend = blob.get("backend")
        obj._calibrator = blob.get("calibrator")
        obj._n_features = blob.get("n_features")
        return obj


# ---------------------------------------------------------------------------
# Tiny picklable calibrator wrappers (so save/load round-trips cleanly).
# ---------------------------------------------------------------------------
class _IsotonicWrapper:
    """Picklable wrapper applying a fitted IsotonicRegression to raw probs."""

    def __init__(self, iso: Any) -> None:
        self.iso = iso

    def predict(self, p: Any):
        import numpy as np  # lazy
        return np.asarray(self.iso.predict(np.asarray(p, dtype=np.float64)))


class _SigmoidWrapper:
    """Picklable wrapper applying fitted Platt scaling to raw probabilities."""

    def __init__(self, lr: Any) -> None:
        self.lr = lr

    def predict(self, p: Any):
        import numpy as np  # lazy
        arr = np.asarray(p, dtype=np.float64).reshape(-1, 1)
        return self.lr.predict_proba(arr)[:, 1]
