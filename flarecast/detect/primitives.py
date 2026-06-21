"""O(1) streaming detection primitives (ARCHITECTURE.md Section 4.3, B.4).

Every class here is a small streaming estimator whose ``.update(x)`` runs in
**O(1) time** and whose internal state is **O(1)** in size (independent of the
number of samples seen). This is the heart of the project's "fastest platform,
O(1) techniques" mandate (research doc ``03 Section 0/1``).

Implementation philosophy
-------------------------
This is the O(1) hot path, so everything is **pure Python standard library**
(``math``, ``collections.deque``) -- no numpy. That keeps the module trivially
importable offline, fully testable, and *honest* about the O(1) claim: there is
no hidden vectorised re-computation over a growing window.

Primitives
----------
``EMA``                 recursive exponential moving average (1 float state).
``EWMV``                EW mean + variance, drift-aware (2 floats).
``Welford``             numerically-stable online mean + variance (3 floats).
``RingBuffer``          fixed-capacity raw history (L floats, ``deque(maxlen)``).
``P2Quantile``          P^2 running quantile / median (5 markers).
``HampelDespiker``      running median/MAD outlier flag + despiker (ring of L).
``SlopeEstimator``      incremental least-squares slope over a ring window.

The per-sample / per-state complexity of each is documented on the class.
"""

from __future__ import annotations

import math
from collections import deque

__all__ = [
    "EMA",
    "EWMV",
    "Welford",
    "RingBuffer",
    "P2Quantile",
    "HampelDespiker",
    "SlopeEstimator",
]


# ---------------------------------------------------------------------------
# EMA -- recursive low-pass (baseline / trend)
# ---------------------------------------------------------------------------
class EMA:
    """Recursive exponential moving average (research doc ``03 Section 1.1``).

    ``mu_t = alpha * mu_{t-1} + (1 - alpha) * x_t``.

    Complexity: **O(1) time** (one multiply-add) and **O(1) state** (one float
    ``m`` plus a sample counter used only for optional bias correction).

    Parameters
    ----------
    alpha:
        Forgetting factor in ``(0, 1)``; effective window ``~ 1/(1-alpha)`` and
        time constant ``tau ~ dt/(1-alpha)``. Larger ``alpha`` -> longer memory.
    x0:
        Initial mean (cold-start seed); also used as the value reported before
        the first ``update`` when bias correction is off.
    """

    __slots__ = ("a", "m", "_n")

    def __init__(self, alpha: float, x0: float = 0.0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.a: float = float(alpha)
        self.m: float = float(x0)
        self._n: int = 0

    def update(self, x: float) -> float:
        """Fold ``x`` into the running mean and return the new mean. O(1)."""
        self.m = self.a * self.m + (1.0 - self.a) * x
        self._n += 1
        return self.m

    @property
    def value(self) -> float:
        """Current mean estimate (no update)."""
        return self.m

    def value_bias_corrected(self) -> float:
        """Bias-corrected mean ``m / (1 - alpha**n)`` (fixes cold-start lag).

        For the first ``~1/(1-alpha)`` samples the raw EMA under-estimates
        because it is seeded at ``x0``; dividing by ``1 - alpha**n`` removes
        that initial bias (research doc ``03 Section 1.1``). O(1).
        """
        if self._n == 0:
            return self.m
        denom = 1.0 - self.a**self._n
        if denom <= 0.0:
            return self.m
        return self.m / denom


# ---------------------------------------------------------------------------
# EWMV -- exponentially-weighted mean + variance (drift-aware; preferred)
# ---------------------------------------------------------------------------
class EWMV:
    """Exponentially-weighted mean and variance (research doc ``03 Section 1.2b``).

    Recursive pair (West 1979 / incremental EWMA variance)::

        d   = x - m_{t-1}
        m_t = m_{t-1} + (1 - alpha) * d
        S_t = alpha * (S_{t-1} + (1 - alpha) * d * d)     # EW variance

    Preferred over :class:`Welford` here because the quiet-Sun baseline drifts
    (solar cycle, orbit) so recent samples should dominate.

    Complexity: **O(1) time**, **O(1) state** (two floats ``m`` and ``S``).

    Parameters
    ----------
    alpha:
        Forgetting factor in ``(0, 1)``.
    x0:
        Initial mean seed.
    """

    __slots__ = ("a", "m", "S", "_n")

    def __init__(self, alpha: float, x0: float = 0.0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.a: float = float(alpha)
        self.m: float = float(x0)
        self.S: float = 0.0
        self._n: int = 0

    def update(self, x: float) -> tuple[float, float]:
        """Fold ``x`` in; return ``(mean, variance)``. O(1)."""
        d = x - self.m
        self.m = self.m + (1.0 - self.a) * d
        self.S = self.a * (self.S + (1.0 - self.a) * d * d)
        self._n += 1
        return self.m, self.S

    def mean(self) -> float:
        """Current EW mean."""
        return self.m

    def variance(self) -> float:
        """Current EW variance."""
        return self.S

    def sd(self) -> float:
        """Current EW standard deviation ``sqrt(S)``."""
        return math.sqrt(self.S) if self.S > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Welford -- numerically-stable unweighted online mean + variance
# ---------------------------------------------------------------------------
class Welford:
    """Welford's online algorithm for mean + variance (research doc ``03 1.2a``).

    Numerically stable single-pass mean/variance (all samples equal weight)::

        n  += 1
        d   = x - mu
        mu += d / n
        d2  = x - mu
        M2 += d * d2
        var = M2 / (n - 1)

    Complexity: **O(1) time**, **O(1) state** (three numbers ``n``, ``mu``,
    ``M2``). More stable than the naive ``E[x^2] - E[x]^2``.
    """

    __slots__ = ("n", "mu", "M2")

    def __init__(self) -> None:
        self.n: int = 0
        self.mu: float = 0.0
        self.M2: float = 0.0

    def update(self, x: float) -> tuple[float, float]:
        """Fold ``x`` in; return ``(mean, sample_variance)``. O(1).

        Sample variance uses the ``n - 1`` (unbiased) denominator; it is ``0.0``
        until at least two samples have been seen.
        """
        self.n += 1
        d = x - self.mu
        self.mu += d / self.n
        d2 = x - self.mu
        self.M2 += d * d2
        var = self.M2 / (self.n - 1) if self.n > 1 else 0.0
        return self.mu, var

    def mean(self) -> float:
        """Current mean."""
        return self.mu

    def variance(self) -> float:
        """Current sample variance (``0.0`` for ``n < 2``)."""
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def sd(self) -> float:
        """Current sample standard deviation."""
        v = self.variance()
        return math.sqrt(v) if v > 0.0 else 0.0


# ---------------------------------------------------------------------------
# RingBuffer -- fixed-memory raw history
# ---------------------------------------------------------------------------
class RingBuffer:
    """Fixed-capacity circular buffer of the last ``length`` floats (``03 1.3``).

    Backed by ``collections.deque(maxlen=length)``: ``push`` is **O(1)** and the
    total memory is **O(length)** regardless of how many samples have streamed
    through (old values are evicted automatically). Used by the despiker,
    matched filter, slope estimator and onset re-confirmation.

    Parameters
    ----------
    length:
        Maximum number of retained samples (must be >= 1).
    """

    __slots__ = ("_buf", "_cap")

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length!r}")
        self._cap: int = int(length)
        self._buf: deque[float] = deque(maxlen=self._cap)

    def push(self, x: float) -> None:
        """Append ``x``, evicting the oldest value if full. O(1)."""
        self._buf.append(float(x))

    def last(self, n: int) -> list[float]:
        """Return up to the last ``n`` values, oldest-first. O(n) (n <= length).

        Returns a plain ``list`` (pure stdlib, no numpy) so it is directly
        usable in tests and by the O(window) primitives that consume it.
        """
        if n <= 0:
            return []
        if n >= len(self._buf):
            return list(self._buf)
        # deque slicing is not supported; islice from the right is O(n).
        buf = self._buf
        start = len(buf) - n
        out: list[float] = []
        for i, v in enumerate(buf):
            if i >= start:
                out.append(v)
        return out

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        """Maximum retained-sample count."""
        return self._cap

    @property
    def is_full(self) -> bool:
        """True once the buffer holds ``capacity`` samples."""
        return len(self._buf) == self._cap


# ---------------------------------------------------------------------------
# P2Quantile -- P^2 running quantile / median (Jain & Chlamtac 1985)
# ---------------------------------------------------------------------------
class P2Quantile:
    """P^2 running quantile estimator (research doc ``03 Section 1.5``).

    Tracks a single ``q``-quantile (e.g. the median ``q=0.5``) by maintaining
    **5 markers** and updating their heights via parabolic (P^2) interpolation
    as each sample arrives -- **O(1) time, O(1) state** (Jain & Chlamtac 1985).
    A mean-based baseline is biased upward by flares; this gives a robust median
    baseline (and, by tracking ``|x - median|`` in a second instance, a running
    MAD scale).

    Parameters
    ----------
    q:
        Target quantile in ``(0, 1)``; ``0.5`` for the running median.
    """

    __slots__ = ("p", "_q", "_n", "_desired", "_incr", "_count")

    def __init__(self, q: float) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q!r}")
        self.p: float = float(q)
        self._q: list[float] = []  # marker heights (the quantile track)
        self._n: list[int] = []  # marker positions (1-based)
        self._desired: list[float] = []  # desired marker positions
        self._count: int = 0
        # The five desired-position increments are fixed by the target quantile.
        self._incr: list[float] = [0.0, self.p / 2.0, self.p, (1.0 + self.p) / 2.0, 1.0]

    def update(self, x: float) -> float:
        """Fold ``x`` in; return the current quantile estimate. O(1)."""
        x = float(x)
        if self._count < 5:
            # Bootstrap: collect the first five observations, kept sorted.
            self._q.append(x)
            self._count += 1
            if self._count == 5:
                self._q.sort()
                self._n = [1, 2, 3, 4, 5]
                self._desired = [
                    1.0,
                    1.0 + 2.0 * self.p,
                    1.0 + 4.0 * self.p,
                    3.0 + 2.0 * self.p,
                    5.0,
                ]
            return self._current_estimate()

        q = self._q
        n = self._n
        # 1. Find the cell k such that q[k] <= x < q[k+1]; adjust extremes.
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[4]:
            q[4] = x
            k = 3
        else:
            k = 0
            for i in range(4):
                if q[i] <= x < q[i + 1]:
                    k = i
                    break

        # 2. Increment positions of markers above the cell and desired positions.
        for i in range(k + 1, 5):
            n[i] += 1
        for i in range(5):
            self._desired[i] += self._incr[i]

        # 3. Adjust the three interior markers if they are off by >= 1.
        for i in range(1, 4):
            d = self._desired[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1) or (d <= -1.0 and n[i - 1] - n[i] < -1):
                s = 1.0 if d >= 0.0 else -1.0
                qp = self._parabolic(i, s)
                if q[i - 1] < qp < q[i + 1]:
                    q[i] = qp
                else:
                    q[i] = self._linear(i, s)
                n[i] += int(s)

        self._count += 1
        return self._current_estimate()

    def _parabolic(self, i: int, s: float) -> float:
        """P^2 parabolic-interpolation prediction of marker ``i``'s new height."""
        q = self._q
        n = self._n
        return q[i] + s / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, s: float) -> float:
        """Linear fallback used when the parabolic prediction is out of order."""
        q = self._q
        n = self._n
        j = i + int(s)
        return q[i] + s * (q[j] - q[i]) / (n[j] - n[i])

    def _current_estimate(self) -> float:
        """Best current estimate of the target quantile."""
        if self._count == 0:
            return 0.0
        if self._count < 5:
            # Before bootstrap completes, use the nearest order statistic.
            s = sorted(self._q)
            idx = min(len(s) - 1, max(0, int(round(self.p * (len(s) - 1)))))
            return s[idx]
        return self._q[2]

    @property
    def value(self) -> float:
        """Current quantile estimate (no update)."""
        return self._current_estimate()

    @property
    def n_seen(self) -> int:
        """Number of samples folded in so far."""
        return self._count


# ---------------------------------------------------------------------------
# HampelDespiker -- running median/MAD outlier flag + despiker
# ---------------------------------------------------------------------------
class HampelDespiker:
    """Streaming Hampel identifier / despiker (research doc ``03 Section 2.1/7``).

    Flags ``x`` as an outlier when ``|x - median| > k * 1.4826 * MAD`` over a
    trailing window, and returns the median in its place (despiking). The same
    test doubles as the cosmic-ray rejector: a one-sample excursion is replaced;
    a sustained one passes through (the caller applies the width gate).

    Complexity: this implementation keeps the trailing window of ``window``
    samples in a ring (``O(window)`` *constant* memory) and recomputes the
    window median/MAD on each update. With ``window`` fixed (a small constant,
    e.g. 7) the work per sample is a **constant** ``O(window) = O(1)`` -- there
    is no dependence on the number of samples streamed, satisfying the O(1)
    state/amortised-constant-time contract.

    Parameters
    ----------
    window:
        Trailing window length (odd preferred; samples used for median/MAD).
    k:
        Threshold in robust sigmas (default :data:`flarecast.constants.HAMPEL_K`).
    """

    __slots__ = ("_buf", "_k", "_win")

    _MAD_SCALE = 1.4826  # MAD -> sigma for Gaussian data

    def __init__(self, window: int, k: float = 3.0) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        self._win: int = int(window)
        self._k: float = float(k)
        self._buf: deque[float] = deque(maxlen=self._win)

    def update(self, x: float) -> tuple[float, bool]:
        """Test ``x``; return ``(clean_value, is_outlier)``. O(window)=O(1).

        ``clean_value`` is ``x`` itself when it is not an outlier, or the trailing
        window median when it is (the despiked replacement). The raw ``x`` is the
        value pushed into the window so the despiker tracks the true signal.
        """
        x = float(x)
        buf = self._buf
        if len(buf) < 2:
            # Not enough history to judge; accept and seed.
            buf.append(x)
            return x, False

        med = self._median(buf)
        mad = self._median([abs(v - med) for v in buf])
        sigma = self._MAD_SCALE * mad
        buf.append(x)  # window always reflects the true incoming sample

        if sigma <= 0.0:
            # Degenerate (flat) window: only an exact-equal value is "inlier".
            is_out = x != med
            return (med if is_out else x), is_out

        is_out = abs(x - med) > self._k * sigma
        return (med if is_out else x), is_out

    @staticmethod
    def _median(values) -> float:
        """Median of an iterable of floats (small fixed window -> O(window))."""
        s = sorted(values)
        m = len(s)
        if m == 0:
            return 0.0
        mid = m // 2
        if m % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    @property
    def window(self) -> int:
        """Trailing window length."""
        return self._win


# ---------------------------------------------------------------------------
# SlopeEstimator -- incremental least-squares slope over a ring window
# ---------------------------------------------------------------------------
class SlopeEstimator:
    """Incremental least-squares slope ``d x / d t`` over a ring (``03 1.4``).

    Maintains the running sums ``Sum_t``, ``Sum_x``, ``Sum_tt``, ``Sum_tx`` over
    the last ``window`` samples and returns the ordinary-least-squares slope of
    ``x`` vs ``t``. Because the window is a fixed ``deque`` and each update adds
    the new term and subtracts the evicted one, the work per sample is
    **O(1) time** with **O(window)** *constant* state. This is the Savitzky-Golay
    -on-a-ring slope -- the onset signal and the Neupert ``d/dt[SXR]`` bridge.

    Parameters
    ----------
    window:
        Number of trailing samples in the regression (must be >= 2).
    """

    __slots__ = ("_win", "_pts", "_st", "_sx", "_stt", "_stx")

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window!r}")
        self._win: int = int(window)
        self._pts: deque[tuple[float, float]] = deque(maxlen=self._win)
        self._st: float = 0.0  # sum t
        self._sx: float = 0.0  # sum x
        self._stt: float = 0.0  # sum t*t
        self._stx: float = 0.0  # sum t*x

    def update(self, x: float, t: float) -> float:
        """Fold ``(t, x)`` in; return the OLS slope over the window. O(1).

        When the window is full the oldest point's contribution is subtracted
        before the new one is added, so every update is constant-time regardless
        of how long the stream has run.
        """
        x = float(x)
        t = float(t)
        if len(self._pts) == self._win:
            ot, ox = self._pts[0]  # about to be evicted by the append below
            self._st -= ot
            self._sx -= ox
            self._stt -= ot * ot
            self._stx -= ot * ox
        self._pts.append((t, x))
        self._st += t
        self._sx += x
        self._stt += t * t
        self._stx += t * x

        n = len(self._pts)
        if n < 2:
            return 0.0
        denom = n * self._stt - self._st * self._st
        if denom == 0.0:
            return 0.0
        return (n * self._stx - self._st * self._sx) / denom

    @property
    def window(self) -> int:
        """Regression window length."""
        return self._win

    @property
    def n_points(self) -> int:
        """Number of points currently in the window."""
        return len(self._pts)
