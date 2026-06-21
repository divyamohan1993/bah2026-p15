"""Fusion estimators: inverse-variance + Kalman (ARCHITECTURE.md Section 3.8).

Two estimators turn many cross-calibrated, same-quantity measurements into one
best estimate with smaller uncertainty (research doc ``06 Section 4.3``):

* :func:`inverse_variance_fuse` -- the static, per-grid-cell baseline. For N
  independent measurements ``x_i`` with variances ``sigma_i^2`` and optional
  reliabilities ``r_i``, the minimum-variance unbiased linear estimator is the
  inverse-variance weighted mean::

      x_hat = sum(w_i x_i) / sum(w_i),  w_i = r_i / sigma_i^2,
      sigma_hat^2 = 1 / sum(w_i)   ( < min_i sigma_i^2 -- the payoff of fusion )

* :class:`KalmanFuser` -- the temporal production estimator. It tracks the state
  ``s = [log10 F, d log10 F / dt]`` (captures exponential rise/decay), predicts
  with a constant-rate model + process noise, and **updates once per available
  sensor** per step (sequential update == batch inverse-variance for
  independent sensors). Two payoffs fall out for free:

  1. the ``chi^2_{1,0.999} ~ 10.8`` innovation gate (``KALMAN_GATE_CHI2``)
     rejects outliers automatically -- a particle spike on one sensor is gated
     while the others update the state (automatic FAR reduction);
  2. when no sensor is valid the predict step coasts and the covariance grows,
     so the uncertainty band widens over a gap and shrinks when data returns
     (automatic gap handling).

The state is 2-D, so the linear algebra is hand-rolled in **pure standard
library** (no numpy) -- fully testable offline. numpy is never imported here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..constants import KALMAN_GATE_CHI2

__all__ = [
    "inverse_variance_fuse",
    "KalmanFuser",
]

_TINY = 1e-300  # guard against division by exactly-zero weight sums


# ---------------------------------------------------------------------------
# Static inverse-variance fusion (research 06 Section 4.3.1)
# ---------------------------------------------------------------------------
def inverse_variance_fuse(
    values: Sequence[float],
    sigmas: Sequence[float],
    reliabilities: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Inverse-variance (reliability-weighted) fusion of one grid cell.

    Implements Appendix B.3 ``fuse.inverse_variance_fuse``::

        w_i = r_i / sigma_i^2
        x_hat = sum(w_i x_i) / sum(w_i)
        sigma_hat = sqrt(1 / sum(w_i))

    The fused ``sigma_hat`` is **smaller than any input sigma** (two equal
    sensors -> sigma/sqrt(2)); this is the quantitative payoff of fusion and is
    asserted by ``tests/test_fuse.py``.

    Parameters
    ----------
    values:
        Cross-calibrated measurements ``x_i`` of the same quantity.
    sigmas:
        Per-measurement 1-sigma uncertainties (must be > 0).
    reliabilities:
        Optional static reliability weights ``r_i`` in (0, 1]; default all 1.0
        (pure inverse-variance). A more reliable source gets more weight even
        beyond its formal error (research 06 Section 4.3.1).

    Returns
    -------
    (x_hat, sigma_hat)
        Fused best estimate and its 1-sigma uncertainty.

    Raises
    ------
    ValueError
        If inputs are empty, length-mismatched, or contain a non-positive
        sigma (a zero/negative sigma is not a valid Gaussian weight).
    """
    n = len(values)
    if n == 0:
        raise ValueError("inverse_variance_fuse: need at least one measurement")
    if len(sigmas) != n:
        raise ValueError("inverse_variance_fuse: values/sigmas length mismatch")
    if reliabilities is None:
        rels: Sequence[float] = [1.0] * n
    else:
        if len(reliabilities) != n:
            raise ValueError(
                "inverse_variance_fuse: values/reliabilities length mismatch"
            )
        rels = reliabilities

    w_sum = 0.0
    wx_sum = 0.0
    for x, s, r in zip(values, sigmas, rels, strict=True):
        if not (s > 0.0):
            raise ValueError(
                f"inverse_variance_fuse: sigma must be > 0 (got {s!r})"
            )
        if r <= 0.0:
            # A zero/negative reliability means "do not trust" -> drop it.
            continue
        w = r / (s * s)
        w_sum += w
        wx_sum += w * x

    if w_sum <= _TINY:
        raise ValueError(
            "inverse_variance_fuse: all measurements had zero weight"
        )
    x_hat = wx_sum / w_sum
    sigma_hat = math.sqrt(1.0 / w_sum)
    return x_hat, sigma_hat


# ---------------------------------------------------------------------------
# Kalman filter on [log10 F, rate] (research 06 Section 4.3.2)
# ---------------------------------------------------------------------------
class KalmanFuser:
    """Temporal multi-sensor Kalman fuser on ``[log10 F, d log10 F / dt]``.

    Implements Appendix B.3 ``fuse.KalmanFuser`` and research 06 Section 4.3.2.
    State ``s = [l, ldot]`` with ``l = log10 F``; constant-rate transition::

        F = [[1, dt],
             [0,  1]],     s^- = F s,   P^- = F P F^T + Q

    Per available sensor (sequential update; ``H = [1, 0]``, ``R = sigma_log^2 /
    (reliability)``)::

        y = z - H s^-                         (innovation)
        S = H P^- H^T + R                      (innovation variance)
        if y^2 / S > gate_chi2: reject (outlier gate); continue
        K = P^- H^T / S                        (Kalman gain)
        s = s^- + K y;  P = (I - K H) P^-       (update)

    All matrices are 2x2 / 2-vectors handled with scalar arithmetic so there is
    **no numpy dependency**. The innovation gate uses ``KALMAN_GATE_CHI2``
    (~10.8) and is what ``tests/test_fuse.py`` exercises: an injected spike is
    gated (state barely moves) while normal updates are accepted.

    Parameters
    ----------
    q:
        Process-noise scale. The 2x2 process-noise matrix is the standard
        constant-velocity discrete-white-noise form scaled by ``q`` and the
        step ``dt`` (see :meth:`step`); larger ``q`` lets the state track faster
        changes (and widens the gap-coasting band more quickly).
    gate_chi2:
        Innovation-gate threshold (default ``KALMAN_GATE_CHI2`` ~ 10.8).
    init_log_flux:
        Initial ``log10 F`` state (default ``-7.0`` ~ quiet-Sun A-class).
    init_var_level / init_var_rate:
        Initial diagonal covariance for the level and rate states.
    """

    __slots__ = (
        "q",
        "gate_chi2",
        "_l",
        "_ldot",
        "_p00",
        "_p01",
        "_p10",
        "_p11",
        "_initialized",
        "n_updates",
        "n_gated",
        "n_coast",
    )

    def __init__(
        self,
        q: float = 1e-4,
        gate_chi2: float = KALMAN_GATE_CHI2,
        init_log_flux: float = -7.0,
        init_var_level: float = 1.0,
        init_var_rate: float = 1.0,
    ) -> None:
        self.q = float(q)
        self.gate_chi2 = float(gate_chi2)
        # State vector [l, ldot].
        self._l = float(init_log_flux)
        self._ldot = 0.0
        # Covariance P (row-major 2x2).
        self._p00 = float(init_var_level)
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = float(init_var_rate)
        self._initialized = False
        # Diagnostics (used by tests / quality scoring).
        self.n_updates = 0
        self.n_gated = 0
        self.n_coast = 0

    # --- introspection ----------------------------------------------------
    @property
    def log_flux(self) -> float:
        """Current ``log10 F`` state estimate."""
        return self._l

    @property
    def rate(self) -> float:
        """Current ``d log10 F / dt`` state estimate."""
        return self._ldot

    @property
    def flux(self) -> float:
        """Current best-estimate flux ``10 ** log_flux`` (canonical unit)."""
        return 10.0 ** self._l

    @property
    def var_level(self) -> float:
        """Current variance of the level state (P[0,0])."""
        return self._p00

    def sigma_level(self) -> float:
        """Current 1-sigma of the ``log10 F`` level state."""
        return math.sqrt(max(self._p00, 0.0))

    # --- core step --------------------------------------------------------
    def step(
        self,
        dt: float,
        measurements: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        """Predict by ``dt`` then sequentially update with each measurement.

        Implements Appendix B.3 ``KalmanFuser.step``.

        Parameters
        ----------
        dt:
            Time since the previous step [s] (>= 0). For the very first call
            with a non-empty measurement list the filter *snaps* to the
            inverse-variance combination of the measurements (so it does not
            spend many steps walking from the prior).
        measurements:
            List of ``(log10_value, sigma_log, reliability)`` for every sensor
            valid at this step. ``log10_value`` is the cross-calibrated
            measurement in log10 of the canonical unit; ``sigma_log`` its
            1-sigma in log10; ``reliability`` the static r_i in (0, 1]. An empty
            list means *no sensor valid* -> predict-only (coast), and the
            covariance grows.

        Returns
        -------
        (F_hat, sigma_hat)
            Best-estimate flux in the canonical unit (``10 ** l``) and its
            1-sigma propagated into linear flux space.
        """
        dt = max(float(dt), 0.0)

        # First-ever update with data: initialize state from the data itself so
        # the gate has a sensible reference and we do not reject good early
        # measurements just because the prior was far away.
        if not self._initialized and measurements:
            vals = [m[0] for m in measurements]
            sigs = [max(m[1], 1e-12) for m in measurements]
            rels = [m[2] for m in measurements]
            l0, s0 = inverse_variance_fuse(vals, sigs, rels)
            self._l = l0
            self._ldot = 0.0
            self._p00 = max(s0 * s0, 1e-12)
            self._p01 = 0.0
            self._p10 = 0.0
            self._p11 = max(self._p11, 1e-6)
            self._initialized = True
            self.n_updates += len(measurements)
            return 10.0 ** self._l, self._linear_sigma()

        # --- PREDICT (constant-rate) ---
        # s^- = F s ;  l <- l + dt*ldot ; ldot unchanged.
        self._l = self._l + dt * self._ldot
        # P^- = F P F^T + Q.
        p00, p01, p10, p11 = self._p00, self._p01, self._p10, self._p11
        # F P:
        a00 = p00 + dt * p10
        a01 = p01 + dt * p11
        a10 = p10
        a11 = p11
        # (F P) F^T:
        np00 = a00 + dt * a01
        np01 = a01
        np10 = a10 + dt * a11
        np11 = a11
        # + Q (discrete white-noise acceleration model, scaled by q).
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.q
        np00 += q * dt4 / 4.0
        np01 += q * dt3 / 2.0
        np10 += q * dt3 / 2.0
        np11 += q * dt2
        # Keep a small process floor on the rate so a long coast never freezes
        # the covariance growth entirely (helps the gap-widening behaviour).
        np11 += q
        self._p00, self._p01, self._p10, self._p11 = np00, np01, np10, np11

        if not measurements:
            # No sensor valid -> coast; covariance has already grown.
            self.n_coast += 1
            return 10.0 ** self._l, self._linear_sigma()

        # --- UPDATE: one sensor at a time (== batch inverse-variance) ---
        for z, sigma_log, reliability in measurements:
            r = max(reliability, 1e-6)
            R = (sigma_log * sigma_log) / r          # reliability-weighted R
            R = max(R, 1e-12)
            # H = [1, 0]: H s = l ; H P H^T = P[0,0].
            y = z - self._l                          # innovation
            S = self._p00 + R                        # innovation variance
            if S <= 0.0:
                continue
            # GATE: normalized innovation chi^2 -> reject outliers.
            if (y * y) / S > self.gate_chi2:
                self.n_gated += 1
                continue
            # K = P H^T / S = [P00, P10]^T / S.
            k0 = self._p00 / S
            k1 = self._p10 / S
            # State update s = s^- + K y.
            self._l = self._l + k0 * y
            self._ldot = self._ldot + k1 * y
            # Covariance update P = (I - K H) P, with K H = [[k0,0],[k1,0]].
            p00, p01, p10, p11 = self._p00, self._p01, self._p10, self._p11
            self._p00 = (1.0 - k0) * p00
            self._p01 = (1.0 - k0) * p01
            self._p10 = p10 - k1 * p00
            self._p11 = p11 - k1 * p01
            self.n_updates += 1
            self._initialized = True

        return 10.0 ** self._l, self._linear_sigma()

    # --- helpers ----------------------------------------------------------
    def _linear_sigma(self) -> float:
        """Propagate the log10-level sigma into linear flux space.

        ``F = 10**l`` so ``dF/dl = ln(10) * F`` and
        ``sigma_F = ln(10) * F * sigma_l``.
        """
        sigma_l = math.sqrt(max(self._p00, 0.0))
        return math.log(10.0) * (10.0 ** self._l) * sigma_l
