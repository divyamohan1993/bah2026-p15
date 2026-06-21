"""Instrumental noise & artifact models for the synthetic generator.

Pure standard library (``math`` + ``random``); ARCHITECTURE.md Appendix B.2 and
research deliverable ``01-aditya-l1-payloads.md`` Sections 1.4, 2.3, 7. These
functions take a *count-rate-like* sequence and return a corrupted copy, adding
the realities a detector must survive:

* :func:`add_poisson_noise` -- shot noise. X-ray detectors count photons, so
  the per-sample fluctuation is Poisson (``sigma = sqrt(rate)``), invalidating
  Gaussian thresholds at low counts (research 01 S2.3). Implemented with a
  pure-python Knuth Poisson sampler (small means) + a Gaussian approximation
  for large means, so **no numpy is required**.
* :func:`add_gaussian_noise` -- additive read/electronics noise (companion to
  the Poisson term; used for the soft channel where the canonical unit is a
  smooth flux rather than raw counts).
* :func:`add_cosmic_spikes` -- one-/two-sample cosmic-ray excursions (esp.
  HEL1OS, research 01 S2.3) that the width-gate despiker must reject.
* :func:`inject_gaps` -- data gaps from telemetry/slew/mode changes
  (ARCHITECTURE.md Section 11.1). Works on the pure-python column container the
  generator uses (and, optionally, a pandas ``DataFrame``).
* :func:`apply_sdd_saturation` -- the SoLEXS **SDD1 paralyzable rollover**: as
  the *true* rate rises past the dead-time ceiling the *observed* rate turns
  over and decreases, producing the spurious flare-max "dip" the pipeline must
  not read as a real flux drop (research 01 S1.4 / S7).

All randomness is funnelled through a ``random.Random`` instance passed in by
the caller (``rng``) so the whole generator is deterministic given a seed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..constants import SDD_SATURATION_CPS

# numpy optional accelerator (see profiles.py for the same pattern).
try:  # pragma: no cover
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False


__all__ = [
    "add_poisson_noise",
    "add_gaussian_noise",
    "add_cosmic_spikes",
    "inject_gaps",
    "apply_sdd_saturation",
    "poisson_sample",
]


def _is_ndarray(x: object) -> bool:
    return _HAVE_NUMPY and isinstance(x, _np.ndarray)


def _as_float_list(x: Sequence[float]) -> list[float]:
    if _is_ndarray(x):
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


def _match_kind(values: list[float], like: object):
    if _is_ndarray(like):
        return _np.asarray(values, dtype=float)
    return values


# ---------------------------------------------------------------------------
# Poisson shot noise
# ---------------------------------------------------------------------------
def poisson_sample(lam: float, rng) -> int:
    """Draw one Poisson sample with mean ``lam`` using only stdlib ``random``.

    Knuth's multiplicative algorithm for small means (exact); a Gaussian
    approximation ``round(N(lam, sqrt(lam)))`` for large means (>~30) where
    Knuth would loop too many times and the normal approximation is excellent.
    Pure python -- no numpy needed.

    Parameters
    ----------
    lam:
        Poisson mean (>=0). Non-positive means return 0.
    rng:
        A ``random.Random`` instance.

    Returns
    -------
    A non-negative integer sample.
    """
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        # normal approximation; clamp at 0.
        val = rng.gauss(lam, math.sqrt(lam))
        return max(0, int(round(val)))
    # Knuth: count events until the product of uniforms drops below e^-lam.
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def add_poisson_noise(counts: Sequence[float], rng):
    """Apply Poisson shot noise sample-by-sample (research 01 S2.3).

    Each input value is treated as the *expected* count rate and replaced by a
    Poisson draw with that mean. Low-count samples therefore get the correct
    ``sqrt(rate)`` fluctuation that makes Gaussian thresholds invalid -- the
    reason the hard band uses Poisson detectors.

    Parameters
    ----------
    counts:
        Expected count-rate series (list or ndarray, non-negative).
    rng:
        A ``random.Random`` instance for reproducibility.

    Returns
    -------
    Same kind as ``counts``; noisy (integer-valued, stored as float) series.
    """
    c = _as_float_list(counts)
    out = [float(poisson_sample(max(0.0, v), rng)) for v in c]
    return _match_kind(out, counts)


def add_gaussian_noise(
    values: Sequence[float],
    rng,
    *,
    rel_sigma: float = 0.0,
    abs_sigma: float = 0.0,
):
    """Add Gaussian read/electronics noise (relative + absolute terms).

    ``noisy_i = value_i + N(0, rel_sigma*|value_i| + abs_sigma)``. Used for the
    soft channel whose canonical unit is a continuous flux (W m^-2) rather than
    raw counts, and as an additive baseline term anywhere a smooth jitter is
    wanted. Negative results are clamped to 0 (flux cannot be negative).

    Parameters
    ----------
    values:
        Input series (list or ndarray).
    rng:
        A ``random.Random`` instance.
    rel_sigma:
        Fractional noise (sigma as a fraction of the local value).
    abs_sigma:
        Additive noise floor (sigma in the value's unit).

    Returns
    -------
    Same kind as ``values``; noisy, non-negative series.
    """
    v = _as_float_list(values)
    out: list[float] = []
    for x in v:
        sigma = rel_sigma * abs(x) + abs_sigma
        n = rng.gauss(0.0, sigma) if sigma > 0 else 0.0
        out.append(max(0.0, x + n))
    return _match_kind(out, values)


# ---------------------------------------------------------------------------
# Cosmic-ray spikes
# ---------------------------------------------------------------------------
def add_cosmic_spikes(
    counts: Sequence[float],
    rate_per_hr: float,
    rng,
    *,
    cadence_s: float = 1.0,
    amp_min: float = 5.0,
    amp_max: float = 50.0,
    max_width_bins: int = 2,
):
    """Inject sparse cosmic-ray / particle spikes (research 01 S2.3).

    Cosmic-ray hits appear as **1-2 sample** sharp excursions (well below the
    ``SPIKE_WIDTH_MIN_BINS`` flare-width gate, so a correct despiker rejects
    them). Spikes are Poisson-distributed in time at ``rate_per_hr`` events per
    hour; each spike adds ``amp_min..amp_max`` times the local count level (or
    an absolute floor when the local level is ~0) across 1..``max_width_bins``
    samples.

    Parameters
    ----------
    counts:
        Count-rate series to contaminate (list or ndarray).
    rate_per_hr:
        Mean cosmic-ray spike rate (events / hour).
    rng:
        A ``random.Random`` instance.
    cadence_s:
        Sample spacing (s), to convert the hourly rate to a per-sample
        probability.
    amp_min, amp_max:
        Spike amplitude range as a multiple of the local count level (with a
        small absolute floor so spikes are visible even on a zero baseline).
    max_width_bins:
        Maximum spike width in samples (1 or 2 -- never a flare width).

    Returns
    -------
    Same kind as ``counts``; contaminated series. Also (in ``meta`` of the
    generator) the number of injected spikes is recoverable by the caller via
    the returned count, but this function returns only the series; the
    generator tracks spike provenance separately.
    """
    c = _as_float_list(counts)
    n = len(c)
    if n == 0:
        return _match_kind([], counts)
    # per-sample spike probability.
    p = max(0.0, rate_per_hr) * (cadence_s / 3600.0)
    p = min(p, 1.0)
    i = 0
    while i < n:
        if rng.random() < p:
            width = 1 if max_width_bins <= 1 else rng.randint(1, max_width_bins)
            local = c[i]
            floor = 20.0  # absolute counts floor so spikes show on quiet bg
            amp = rng.uniform(amp_min, amp_max) * max(local, floor)
            for j in range(i, min(n, i + width)):
                c[j] += amp
            i += width
        else:
            i += 1
    return _match_kind(c, counts)


# ---------------------------------------------------------------------------
# SDD1 paralyzable saturation rollover
# ---------------------------------------------------------------------------
def apply_sdd_saturation(counts: Sequence[float], ceiling: float = SDD_SATURATION_CPS):
    """Apply the SoLEXS SDD1 paralyzable dead-time rollover (research 01 S1.4).

    The paralyzable model relates the *observed* rate ``m`` to the *true* rate
    ``n`` by ``m = n * exp(-n * tau)``, where the dead time ``tau`` is chosen
    so the curve peaks (turns over) right at ``ceiling`` true counts/s:
    ``d m/dn = 0`` at ``n = 1/tau`` => ``tau = 1/ceiling``. Below the ceiling
    the response is near-linear; above it the observed rate *decreases* even as
    the true rate climbs -- the spurious flare-maximum "dip" the pipeline must
    treat as an artifact and handle by SDD1->SDD2 arbitration.

    Parameters
    ----------
    counts:
        *True* count-rate series (list or ndarray, non-negative).
    ceiling:
        True count rate at which the observed rate turns over (the SDD1
        saturation onset); defaults to :data:`SDD_SATURATION_CPS` (~1e5 cps).

    Returns
    -------
    Same kind as ``counts``; the *observed* (rolled-over) count-rate series.
    """
    if ceiling <= 0:
        raise ValueError(f"ceiling must be > 0, got {ceiling}")
    tau = 1.0 / ceiling
    c = _as_float_list(counts)
    out = [n * math.exp(-n * tau) if n > 0 else 0.0 for n in c]
    return _match_kind(out, counts)


# ---------------------------------------------------------------------------
# Data gaps
# ---------------------------------------------------------------------------
def inject_gaps(df, n_gaps: int, max_len_s: float, rng, *, cadence_s: float | None = None):
    """Punch ``n_gaps`` data gaps into a light-curve container (ARCHITECTURE 11.1).

    A data gap (telemetry dropout, slew, mode change) removes a contiguous run
    of samples. This helper supports the two container shapes used in the
    offline path:

    * the generator's **pure-python column dict** ``{"t": [...], "<col>": [...],
      ...}`` -- rows in the gap interval are *dropped* (returns a new dict);
    * a pandas ``DataFrame`` (optional accelerator) with a ``"t"`` column --
      rows in the gap interval are dropped via boolean masking.

    Gaps are placed at random start times; each gap length is uniform in
    ``(0, max_len_s]``. Returns the same container kind it was given.

    Parameters
    ----------
    df:
        Either a ``dict[str, list]`` (pure-python, must contain key ``"t"``) or
        a pandas ``DataFrame`` (must contain column ``"t"``).
    n_gaps:
        Number of gaps to inject (>=0).
    max_len_s:
        Maximum gap length in seconds.
    rng:
        A ``random.Random`` instance.
    cadence_s:
        Sample spacing (s); inferred from the first two ``t`` values if None.

    Returns
    -------
    A new container of the same kind with gap rows removed.
    """
    # --- pandas DataFrame branch (optional) ---
    if _is_dataframe(df):
        t = list(df["t"])
        keep_mask = _gap_keep_mask(t, n_gaps, max_len_s, rng, cadence_s)
        return df[keep_mask].reset_index(drop=True)

    # --- pure-python column-dict branch ---
    if not isinstance(df, dict) or "t" not in df:
        raise TypeError(
            "inject_gaps expects a dict with a 't' key or a pandas DataFrame "
            "with a 't' column"
        )
    t = [float(v) for v in df["t"]]
    keep_mask = _gap_keep_mask(t, n_gaps, max_len_s, rng, cadence_s)
    out: dict = {}
    for col, values in df.items():
        out[col] = [v for v, keep in zip(values, keep_mask, strict=True) if keep]
    return out


def _gap_keep_mask(
    t: list[float], n_gaps: int, max_len_s: float, rng, cadence_s: float | None
) -> list[bool]:
    """Return a boolean keep-mask after removing ``n_gaps`` random gaps."""
    n = len(t)
    keep = [True] * n
    if n == 0 or n_gaps <= 0 or max_len_s <= 0:
        return keep
    if cadence_s is None:
        cadence_s = (t[1] - t[0]) if n > 1 else 1.0
    if cadence_s <= 0:
        cadence_s = 1.0
    t0, t1 = t[0], t[-1]
    span = t1 - t0
    if span <= 0:
        return keep
    for _ in range(n_gaps):
        gap_len = rng.uniform(0.0, max_len_s)
        # choose a start so the gap fits inside the series.
        start_t = t0 + rng.random() * max(0.0, span - gap_len)
        end_t = start_t + gap_len
        for i in range(n):
            if start_t <= t[i] <= end_t:
                keep[i] = False
    return keep


def _is_dataframe(obj: object) -> bool:
    """True if ``obj`` looks like a pandas DataFrame (without importing pandas)."""
    cls = type(obj)
    return (
        getattr(cls, "__name__", "") == "DataFrame"
        and "pandas" in getattr(cls, "__module__", "")
    )
