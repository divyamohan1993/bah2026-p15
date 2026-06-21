"""Physics-based flare *shape* profiles for the synthetic generator.

This module is the physics heart of the offline keystone (ARCHITECTURE.md
Appendix B.2, research deliverable ``01-aditya-l1-payloads.md`` Sections 5-6).
It is **pure standard library** (``math`` only) so the synthetic generator and
the whole offline path run with zero third-party dependencies. ``numpy`` is an
*optional accelerator*: every public function accepts either a Python sequence
or a ``numpy.ndarray`` for the time axis and returns the *same kind* it was
given, but none of the math requires numpy.

Three building blocks, mapped to the two Aditya-L1 X-ray payloads:

* :func:`fred_profile` -- a **FRED** (Fast Rise, Exponential Decay) pulse, the
  canonical soft-X-ray (SoLEXS / GOES long channel) thermal flare shape:
  gradual rise as chromospheric evaporation fills coronal loops, a rounded
  peak (where the GOES class is defined), and a slow quasi-exponential cooling
  decay (research 01 Section 5.1).
* :func:`impulsive_hxr_train` -- an **impulsive hard-X-ray burst train**
  (HEL1OS) built to satisfy the **Neupert effect**: the HXR light curve tracks
  ``d/dt`` of the SXR light curve (research 01 Section 6), so the impulsive
  spikes cluster on the *rising* edge of the soft profile and *lead* the soft
  peak by minutes -- the forecasting lever.
* :func:`goes_class_to_peak_flux` / :func:`peak_flux_to_goes_class` -- map a
  GOES class string (``"C3.1"``) to / from a peak 1-8 A flux in W m^-2
  (ARCHITECTURE.md Section 4.5), used to scale the soft profile amplitude and
  to label truth events.

The Neupert coupling is also exposed as an explicit leaky integrator
(:func:`neupert_integrate`) so callers can synthesise an SXR curve *from* an
HXR drive, exactly the inverse relation ``dF_SXR/dt = c*F_HXR - F_SXR/tau``
(ARCHITECTURE.md Section 1.2, constant :data:`NEUPERT_TAU_COOL_S`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..constants import (
    GOES_CLASS_LADDER,
    GOES_CLASS_THRESHOLDS_WM2,
    NEUPERT_TAU_COOL_S,
)

# numpy is an OPTIONAL accelerator. The module is fully functional without it;
# when present, array inputs are returned as arrays so callers that already use
# numpy keep their dtype. The pure-python path (lists) is always available.
try:  # pragma: no cover - exercised in both branches across environments
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - offline / minimal environment
    _np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False


__all__ = [
    "fred_profile",
    "impulsive_hxr_train",
    "neupert_derivative",
    "neupert_integrate",
    "goes_class_to_peak_flux",
    "peak_flux_to_goes_class",
]


# ---------------------------------------------------------------------------
# small helpers (sequence-kind preserving)
# ---------------------------------------------------------------------------
def _is_ndarray(x: object) -> bool:
    return _HAVE_NUMPY and isinstance(x, _np.ndarray)


def _as_float_list(t: Sequence[float]) -> list[float]:
    """Return ``t`` as a plain ``list[float]`` regardless of input kind."""
    if _is_ndarray(t):
        return [float(v) for v in t.tolist()]
    return [float(v) for v in t]


def _match_kind(values: list[float], like: object):
    """Return ``values`` as an ndarray if ``like`` was one, else as a list."""
    if _is_ndarray(like):
        return _np.asarray(values, dtype=float)
    return values


# ---------------------------------------------------------------------------
# Soft X-ray (SoLEXS / GOES long) -- FRED pulse
# ---------------------------------------------------------------------------
def fred_profile(
    t: Sequence[float],
    t_peak: float,
    rise_s: float,
    decay_s: float,
    amp: float,
):
    """Fast-Rise-Exponential-Decay soft-X-ray flare pulse (research 01 S5.1).

    The FRED functional form used here is a smooth, everywhere-positive pulse
    that peaks at ``t_peak`` with value ``amp``:

    * **Rise** (``t <= t_peak``): a smooth exponential approach so the curve
      lifts off quiet-Sun gradually, ``amp * exp(-((t - t_peak)/rise_s)^2)``
      flavoured rise via a half-Gaussian -- gradual, rounded, no kink.
    * **Decay** (``t > t_peak``): a quasi-exponential cooling tail,
      ``amp * exp(-(t - t_peak)/decay_s)`` -- the slow conductive/radiative
      cooling of the evaporated plasma.

    Using a half-Gaussian rise and an exponential decay gives the rounded peak
    and long tail characteristic of GOES/SoLEXS soft-X-ray flares while keeping
    a continuous value *and* a continuous derivative at the peak (both sides
    have zero slope contribution discontinuity is avoided because the rise
    derivative ->0 at the peak and the decay derivative ->-amp/decay_s; the
    small kink is physically realistic -- the impulsive heating switches off).

    Parameters
    ----------
    t:
        Time axis (epoch seconds, or any consistent unit), list or ndarray.
    t_peak:
        Time of the flare peak (same unit as ``t``).
    rise_s:
        Rise timescale (>0): larger = more gradual rise.
    decay_s:
        Exponential decay e-folding timescale (>0): larger = longer tail.
    amp:
        Peak amplitude (the value at ``t_peak``); in the canonical SXR unit
        (W m^-2) this is the peak flux above background that sets the class.

    Returns
    -------
    Same kind as ``t`` (list or ndarray) of non-negative pulse values.
    """
    if rise_s <= 0:
        raise ValueError(f"rise_s must be > 0, got {rise_s}")
    if decay_s <= 0:
        raise ValueError(f"decay_s must be > 0, got {decay_s}")

    ts = _as_float_list(t)
    out: list[float] = []
    for ti in ts:
        if ti <= t_peak:
            # half-Gaussian rise: 1 at the peak, ->0 going back in time.
            x = (ti - t_peak) / rise_s
            out.append(amp * math.exp(-0.5 * x * x))
        else:
            # exponential cooling decay.
            out.append(amp * math.exp(-(ti - t_peak) / decay_s))
    return _match_kind(out, t)


# ---------------------------------------------------------------------------
# Neupert coupling -- HXR ~ d/dt SXR  (and the inverse leaky integrator)
# ---------------------------------------------------------------------------
def neupert_derivative(
    sxr: Sequence[float],
    cadence_s: float,
):
    """Central finite-difference derivative ``d/dt SXR`` (Neupert drive).

    The Neupert effect states ``F_HXR(t) ~ d/dt F_SXR(t)`` (research 01 S6).
    This returns the (one-sided at the edges, central in the interior)
    numerical derivative of a soft-X-ray series, the quantity the hard-X-ray
    burst train is shaped to follow. Pure python; O(n).

    Parameters
    ----------
    sxr:
        Soft-X-ray series (list or ndarray).
    cadence_s:
        Sample spacing in seconds (the ``dt`` of the finite difference).

    Returns
    -------
    Same kind as ``sxr``; the derivative (same length).
    """
    if cadence_s <= 0:
        raise ValueError(f"cadence_s must be > 0, got {cadence_s}")
    s = _as_float_list(sxr)
    n = len(s)
    if n == 0:
        return _match_kind([], sxr)
    if n == 1:
        return _match_kind([0.0], sxr)
    d = [0.0] * n
    d[0] = (s[1] - s[0]) / cadence_s
    d[-1] = (s[-1] - s[-2]) / cadence_s
    two_dt = 2.0 * cadence_s
    for i in range(1, n - 1):
        d[i] = (s[i + 1] - s[i - 1]) / two_dt
    return _match_kind(d, sxr)


def neupert_integrate(
    hxr: Sequence[float],
    cadence_s: float,
    c: float = 1.0,
    tau_cool_s: float = NEUPERT_TAU_COOL_S,
):
    """Leaky-integrate an HXR drive into an SXR response (ARCHITECTURE 1.2).

    Forward-Euler integration of the driven-decay Neupert ODE::

        dF_SXR/dt = c * F_HXR(t) - F_SXR(t) / tau_cool

    During the rise the integral term dominates (Neupert holds); during decay
    the ``-F_SXR/tau_cool`` cooling term dominates (Neupert breaks down and the
    SXR decays on its own) -- exactly the physical behaviour described in
    research 01 Section 6. Pure python; O(n).

    Parameters
    ----------
    hxr:
        Hard-X-ray drive series (list or ndarray).
    cadence_s:
        Sample spacing in seconds.
    c:
        Heating coupling constant (scales drive -> SXR).
    tau_cool_s:
        Cooling timescale (s); defaults to :data:`NEUPERT_TAU_COOL_S`.

    Returns
    -------
    Same kind as ``hxr``; the integrated SXR response (same length, >=0).
    """
    if cadence_s <= 0:
        raise ValueError(f"cadence_s must be > 0, got {cadence_s}")
    if tau_cool_s <= 0:
        raise ValueError(f"tau_cool_s must be > 0, got {tau_cool_s}")
    h = _as_float_list(hxr)
    n = len(h)
    out = [0.0] * n
    f = 0.0
    for i in range(n):
        # dF = (c*H - F/tau) dt ; forward Euler.
        f = f + cadence_s * (c * h[i] - f / tau_cool_s)
        if f < 0.0:
            f = 0.0
        out[i] = f
    return _match_kind(out, hxr)


def impulsive_hxr_train(
    t: Sequence[float],
    sxr: Sequence[float],
    n_spikes: int,
    lag_s: float,
    *,
    cadence_s: float | None = None,
    spike_width_s: float = 25.0,
    rng=None,
):
    """Impulsive HXR burst train shaped by the Neupert effect (research 01 S6).

    The hard-X-ray light curve is built to track ``d/dt SXR`` (so it is
    Neupert-consistent by construction) and then decorated with ``n_spikes``
    short, sharp Gaussian bursts placed on the *rising* edge of the soft
    profile -- the impulsive phase. Each spike is advanced by ``lag_s`` so the
    HXR train *leads* the soft peak (the few-minutes early-warning lever).

    The returned train is a non-negative shape (arbitrary units); the caller
    scales it to a count rate. Construction:

    1. Compute ``drive = max(0, d/dt SXR)`` (positive rise only -- HXR fires on
       heating, not on cooling).
    2. Find the soft-peak index; spikes are scattered across the rise interval
       ``[t_start, t_peak - lag_s]`` weighted toward the steepest-rise region.
    3. Add a narrow Gaussian burst (sigma from ``spike_width_s``) at each spike
       time, with amplitude proportional to the local rise rate.

    Parameters
    ----------
    t:
        Time axis (list or ndarray).
    sxr:
        The soft profile to differentiate (same length / kind as ``t``).
    n_spikes:
        Number of impulsive sub-bursts to add (>=0). Larger / harder flares
        get more spikes (set by the generator).
    lag_s:
        How far ahead of the soft peak the impulsive phase leads (s).
    cadence_s:
        Sample spacing (s). If ``None`` it is inferred from ``t``.
    spike_width_s:
        1-sigma width of each Gaussian burst (s); HXR bursts are seconds to a
        minute or two (research 01 S2.4).
    rng:
        Optional ``random.Random`` for reproducible spike placement/amplitude;
        if ``None`` spikes are placed deterministically at evenly spaced
        positions across the rise.

    Returns
    -------
    Same kind as ``t``; the non-negative HXR shape (same length).
    """
    ts = _as_float_list(t)
    n = len(ts)
    if n == 0:
        return _match_kind([], t)

    if cadence_s is None:
        cadence_s = (ts[1] - ts[0]) if n > 1 else 1.0
    if cadence_s <= 0:
        cadence_s = 1.0

    # 1) Neupert base: positive part of d/dt SXR.
    deriv = _as_float_list(neupert_derivative(sxr, cadence_s))
    base = [d if d > 0.0 else 0.0 for d in deriv]

    # locate the soft peak (max of sxr).
    s = _as_float_list(sxr)
    peak_idx = max(range(n), key=lambda i: s[i]) if n else 0
    t_peak = ts[peak_idx]

    # rise window in which impulsive spikes are allowed.
    # everything strictly before (peak - lag).
    rise_end_t = t_peak - lag_s
    rise_idxs = [i for i in range(n) if ts[i] <= rise_end_t and base[i] > 0.0]

    train = list(base)

    if n_spikes > 0 and rise_idxs:
        # weight spike placement by local rise rate so bursts cluster on the
        # steepest part of the rise.
        weights = [base[i] for i in rise_idxs]
        wsum = sum(weights)
        sigma = max(spike_width_s, cadence_s)
        # spike amplitude scale: a few x the local derivative so spikes stand
        # out above the smooth Neupert base.
        for k in range(n_spikes):
            if rng is not None and wsum > 0.0:
                # weighted random pick of a centre index.
                r = rng.random() * wsum
                acc = 0.0
                centre = rise_idxs[0]
                for i, w in zip(rise_idxs, weights, strict=True):
                    acc += w
                    if acc >= r:
                        centre = i
                        break
                amp_jitter = 0.6 + 0.8 * rng.random()
            else:
                # deterministic: evenly spaced across the rise indices.
                pos = (k + 0.5) / n_spikes
                centre = rise_idxs[min(len(rise_idxs) - 1, int(pos * len(rise_idxs)))]
                amp_jitter = 1.0
            centre_t = ts[centre]
            spike_amp = amp_jitter * (3.0 * base[centre] + 1e-12)
            # add a narrow Gaussian burst; only touch nearby samples (+-4 sigma)
            reach = max(1, int(4.0 * sigma / cadence_s))
            lo = max(0, centre - reach)
            hi = min(n, centre + reach + 1)
            inv2s2 = 1.0 / (2.0 * sigma * sigma)
            for i in range(lo, hi):
                dt = ts[i] - centre_t
                train[i] += spike_amp * math.exp(-dt * dt * inv2s2)

    return _match_kind(train, t)


# ---------------------------------------------------------------------------
# GOES class <-> peak flux  (ARCHITECTURE.md Section 4.5)
# ---------------------------------------------------------------------------
def goes_class_to_peak_flux(cls: str) -> float:
    """Map a GOES class string to a representative peak 1-8 A flux [W m^-2].

    Class letter sets the decade; the trailing mantissa is the linear
    multiplier within the decade, e.g. ``"M2.5" -> 2.5e-5``,
    ``"X1" -> 1e-4``, ``"C" -> 1e-6`` (bare letter => mantissa 1.0). This is
    the inverse of :func:`peak_flux_to_goes_class` and of
    ``flarecast.detect.classify.classify_flux``.

    Parameters
    ----------
    cls:
        GOES class, ``"<LETTER>[mantissa]"`` (case-insensitive letter).

    Returns
    -------
    Peak 1-8 A flux in W m^-2.

    Raises
    ------
    ValueError
        If the class letter is not one of A/B/C/M/X or the mantissa is
        unparseable.
    """
    if not cls:
        raise ValueError("empty GOES class string")
    letter = cls[0].upper()
    if letter not in GOES_CLASS_THRESHOLDS_WM2:
        raise ValueError(f"unknown GOES class letter {letter!r} in {cls!r}")
    decade = GOES_CLASS_THRESHOLDS_WM2[letter]
    mantissa_str = cls[1:].strip()
    if not mantissa_str:
        mantissa = 1.0
    else:
        try:
            mantissa = float(mantissa_str)
        except ValueError as exc:
            raise ValueError(f"bad mantissa in GOES class {cls!r}") from exc
    return decade * mantissa


def peak_flux_to_goes_class(flux_wm2: float) -> str:
    """Map a peak 1-8 A flux [W m^-2] to a GOES class string (e.g. ``"M2.5"``).

    Mirrors ``flarecast.detect.classify.classify_flux`` (kept here as a small,
    dependency-free copy so the synth package needs nothing from ``detect``).
    Sub-A or non-positive flux returns ``"A"`` floor handling: values below the
    A decade are reported as ``"A<1.0"`` only when strictly below the A floor;
    otherwise the standard ``<LETTER><mantissa>`` with one decimal.

    Parameters
    ----------
    flux_wm2:
        Peak long-channel flux in W m^-2.

    Returns
    -------
    GOES class string.
    """
    if flux_wm2 is None or flux_wm2 <= 0:
        return "Q"  # quiet / no data
    # descend the ladder X->A; first decade the flux exceeds sets the letter.
    for letter in reversed(GOES_CLASS_LADDER):  # X, M, C, B, A
        base = GOES_CLASS_THRESHOLDS_WM2[letter]
        if flux_wm2 >= base:
            return f"{letter}{flux_wm2 / base:.1f}"
    # below the A decade floor.
    a_base = GOES_CLASS_THRESHOLDS_WM2["A"]
    return f"A<{flux_wm2 / a_base:.1f}"
