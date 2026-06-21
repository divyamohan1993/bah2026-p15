"""Tier-2 challenger forecaster: a dilated causal TCN (PyTorch, optional).

Governing research: ``docs/research/04-forecasting-models.md`` Section 0 / 3.2
/ 9 and ARCHITECTURE.md Section 5.1.

A small **Temporal Convolutional Network** (dilated causal 1-D convolutions,
strictly causal so there is **no look-ahead leakage**) on the raw multichannel
windowed light curve (``[L, C]`` = :data:`flarecast.constants.TCN_LOOKBACK_STEPS`
x :data:`flarecast.constants.TCN_N_CHANNELS`), with **multi-horizon sigmoid
heads** predicting ``P(flare within h)`` for a vector of horizons and trained
with **focal loss** (down-weights easy negatives under heavy imbalance, research
doc 04 Section 6). ONNX-exportable for edge inference.

Causal convolutions are implemented by **left-padding** the input by
``(kernel-1)*dilation`` and trimming the right overhang, so the output at step
``t`` depends only on inputs ``<= t`` (the canonical causal-TCN construction;
research doc 04 Section 3.2). A **GRU** is the stateful, literal-``O(1)``-per-step
alternative (see :meth:`TCNForecaster.fit` docstring) -- preferred when true
streaming recurrence is wanted; the TCN is chosen here for its long receptive
field via dilation, parallel training, and clean ONNX export.

Dependency philosophy: this module imports with only the standard library.
**torch** (and **onnx**) are imported **lazily inside the methods that need
them**; constructing a :class:`TCNForecaster` never imports torch. The matching
test uses ``pytest.importorskip("torch")`` and skips cleanly when torch is
absent.
"""

from __future__ import annotations

from typing import Any

from flarecast.constants import (
    FORECAST_P_CURVE_HORIZONS_MIN,
    TCN_LOOKBACK_STEPS,
    TCN_N_CHANNELS,
)

__all__ = ["TCNForecaster", "focal_loss_with_logits"]


def focal_loss_with_logits(logits: Any, targets: Any, alpha: float = 0.25,
                           gamma: float = 2.0):
    """Binary focal loss on raw logits (research doc 04 Section 6).

    ``FL = -alpha_t * (1 - p_t)**gamma * log(p_t)`` with ``p_t`` the predicted
    probability of the true class. Down-weights easy (well-classified)
    negatives, which dominate a rare-event set. Implemented with torch ops
    (imported lazily) and a numerically-stable log-sigmoid; returns the mean
    over all elements. Defined at module scope so it is unit-testable and
    reusable by an ensemble.
    """
    import torch  # lazy
    import torch.nn.functional as F  # lazy

    logits = logits.float()
    targets = targets.float()
    # p_t and the standard BCE-with-logits term.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return loss.mean()


class TCNForecaster:
    """Dilated causal TCN with multi-horizon sigmoid heads (torch, optional).

    Parameters
    ----------
    n_channels:
        Number of input channels ``C`` (default
        :data:`flarecast.constants.TCN_N_CHANNELS`).
    lookback:
        Input length ``L`` in steps (default
        :data:`flarecast.constants.TCN_LOOKBACK_STEPS`).
    hidden:
        Channels per residual block.
    kernel_size:
        Convolution kernel width (3 per the build plan, research doc 04 §9).
    dilations:
        Per-block dilation factors (``1,2,4,8,16`` per the build plan) giving an
        exponentially growing causal receptive field.
    horizons:
        Default multi-horizon set (minutes) for the sigmoid heads; overridden by
        the ``horizons`` argument to :meth:`fit`.
    """

    def __init__(
        self,
        n_channels: int = TCN_N_CHANNELS,
        lookback: int = TCN_LOOKBACK_STEPS,
        hidden: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        horizons: tuple[int, ...] = FORECAST_P_CURVE_HORIZONS_MIN,
        random_state: int = 0,
    ) -> None:
        self.n_channels = int(n_channels)
        self.lookback = int(lookback)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.horizons = tuple(int(h) for h in horizons)
        self.random_state = int(random_state)

        self._net: Any = None        # the torch.nn.Module (built in fit()).
        self._fit_horizons: list[int] | None = None

    # ------------------------------------------------------------------
    # Internal network builder (torch imported lazily).
    # ------------------------------------------------------------------
    def _build_net(self, n_outputs: int):
        """Construct the causal-TCN ``nn.Module`` (lazy torch import)."""
        import torch  # lazy
        import torch.nn as nn  # lazy

        torch.manual_seed(self.random_state)

        kernel = self.kernel_size
        hidden = self.hidden
        in_ch = self.n_channels
        dilations = self.dilations

        class _CausalConv1d(nn.Module):
            """1-D conv with causal left-padding (no future leakage)."""

            def __init__(self, c_in: int, c_out: int, k: int, d: int) -> None:
                super().__init__()
                self.pad = (k - 1) * d
                self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)

            def forward(self, x):  # x: (B, C, L)
                import torch.nn.functional as F
                x = F.pad(x, (self.pad, 0))  # left-pad only -> causal
                return self.conv(x)

        class _ResBlock(nn.Module):
            def __init__(self, c_in: int, c_out: int, k: int, d: int) -> None:
                super().__init__()
                self.conv1 = _CausalConv1d(c_in, c_out, k, d)
                self.conv2 = _CausalConv1d(c_out, c_out, k, d)
                self.relu = nn.ReLU()
                self.down = (
                    nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
                )

            def forward(self, x):
                y = self.relu(self.conv1(x))
                y = self.relu(self.conv2(y))
                return self.relu(y + self.down(x))

        class _TCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                blocks = []
                c_prev = in_ch
                for d in dilations:
                    blocks.append(_ResBlock(c_prev, hidden, kernel, d))
                    c_prev = hidden
                self.blocks = nn.Sequential(*blocks)
                # Multi-horizon sigmoid heads: one logit per horizon.
                self.head = nn.Linear(hidden, n_outputs)

            def forward(self, x):  # x: (B, L, C)
                x = x.transpose(1, 2)        # -> (B, C, L)
                h = self.blocks(x)           # -> (B, hidden, L)
                last = h[:, :, -1]           # causal: take the final timestep
                return self.head(last)       # logits (B, n_outputs)

        return _TCN()

    # ------------------------------------------------------------------
    # Fitting.
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, horizons: list[int] | None = None,
            epochs: int = 5, lr: float = 1e-3, batch_size: int = 64
            ) -> TCNForecaster:
        """Train the TCN on windows ``X`` of shape ``(N, L, C)``.

        ``y`` is the multi-horizon target of shape ``(N, H)`` (per-horizon 0/1)
        or ``(N,)`` (broadcast to all heads). ``horizons`` names the ``H``
        horizons (minutes); defaults to the constructor's ``horizons``. Uses
        :func:`focal_loss_with_logits` and the Adam optimizer. torch is imported
        lazily.

        Alternative: a **GRU** gives literal ``O(1)``-per-step stateful
        inference (carry the hidden state between samples) and is a drop-in
        challenger when true online streaming is preferred over the TCN's
        ``O(window)`` per-step convolution; the TCN is chosen here for its long
        dilated receptive field, parallel training, and clean ONNX export.
        """
        import numpy as np  # lazy
        import torch  # lazy

        hs = list(horizons) if horizons is not None else list(self.horizons)
        self._fit_horizons = hs
        n_out = len(hs)

        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim != 3:
            raise ValueError(f"X must be (N, L, C); got shape {Xa.shape}")
        ya = np.asarray(y, dtype=np.float32)
        if ya.ndim == 1:
            ya = np.repeat(ya.reshape(-1, 1), n_out, axis=1)
        if ya.shape[1] != n_out:
            raise ValueError(
                f"y has {ya.shape[1]} horizon columns, expected {n_out}"
            )

        self._net = self._build_net(n_out)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        Xt = torch.from_numpy(Xa)
        yt = torch.from_numpy(ya)
        n = Xt.shape[0]

        self._net.train()
        for _ in range(int(epochs)):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = Xt[idx], yt[idx]
                opt.zero_grad()
                logits = self._net(xb)
                loss = focal_loss_with_logits(logits, yb)
                loss.backward()
                opt.step()
        self._net.eval()
        return self

    # ------------------------------------------------------------------
    # Prediction.
    # ------------------------------------------------------------------
    def predict_proba(self, X: Any):
        """Multi-horizon probability curve, shape ``(N, H)`` (numpy array).

        Each column is ``P(flare within h)`` for the corresponding fitted
        horizon. torch/numpy imported lazily.
        """
        import numpy as np  # lazy
        import torch  # lazy

        if self._net is None:
            raise RuntimeError("TCNForecaster.predict_proba before fit().")
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim == 2:  # single window (L, C) -> add batch dim
            Xa = Xa[None, ...]
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xa))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # ONNX export.
    # ------------------------------------------------------------------
    def to_onnx(self, path: str) -> None:
        """Export the trained TCN to ONNX at ``path`` (torch.onnx, lazy)."""
        import torch  # lazy

        if self._net is None:
            raise RuntimeError("to_onnx called before fit().")
        dummy = torch.zeros(1, self.lookback, self.n_channels, dtype=torch.float32)
        torch.onnx.export(
            self._net,
            dummy,
            path,
            input_names=["input"],
            output_names=["p_curve"],
            dynamic_axes={"input": {0: "batch"}, "p_curve": {0: "batch"}},
            opset_version=13,
        )
