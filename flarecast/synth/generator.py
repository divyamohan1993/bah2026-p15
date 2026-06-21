"""Physics-based synthetic flare light-curve generator -- the offline keystone.

ARCHITECTURE.md Section 6 ("the synthetic generator is the keystone of offline
capability") and Appendix B.2. This module ties together the FRED soft profile,
the Neupert-coupled impulsive hard burst train (:mod:`flarecast.synth.profiles`)
and the instrumental noise / artifact models (:mod:`flarecast.synth.noise`) to
produce **SoLEXS-like soft** and **HEL1OS-like hard** X-ray streams plus a
ground-truth event list -- so the entire pipeline (detect, fusion, forecast,
dashboard) runs and is testable with **zero network and zero credentials**.

Design choices (mandated by the dependency philosophy):

* The generator core is **pure standard library** (``math`` + ``random``). It
  does *not* require numpy or pandas. ``numpy`` is used only as an optional
  accelerator inside the profile helpers (transparently); ``pandas`` only in
  the optional :func:`as_dataframe` view.
* Output is canonical: a flat ``list[FluxSample]`` across four streams
  (soft long, soft short, hard 8-30 keV, hard 30-70 keV) -- the same record
  every real fetcher emits -- plus a ``list[TruthEvent]`` (start / peak / end /
  class per flare) for supervised labels and evaluation.
* Determinism: everything is driven by a single ``random.Random(seed)`` so a
  fixed ``seed`` reproduces byte-identical streams (the property the tests and
  the demo rely on).

The public entry point :func:`generate_flare_lightcurves` keeps the exact
Appendix B.2 signature; per the build contract it returns a pure-python
``(samples, truth)`` tuple (with :func:`as_dataframe` / :func:`truth_to_dataframe`
offered as optional pandas views) rather than forcing a pandas dependency on
the offline path.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..constants import (
    DEFAULT_CLASS_MIX,
    SDD_SATURATION_CPS,
    UNIT_HXR,
    UNIT_SXR,
)
from ..types import FluxSample, QCBit, Quantity
from . import noise as _noise
from . import profiles as _profiles

__all__ = [
    "TruthEvent",
    "generate_flare_lightcurves",
    "as_dataframe",
    "truth_to_dataframe",
    "STREAM_SXR_LONG",
    "STREAM_SXR_SHORT",
    "STREAM_HXR_LOW",
    "STREAM_HXR_HIGH",
]

# Canonical stream identifiers for the four synthetic channels.
STREAM_SXR_LONG = "solexs-sxr-long"      # GOES 1-8 A analogue (class channel)
STREAM_SXR_SHORT = "solexs-sxr-short"    # GOES 0.5-4 A analogue
STREAM_HXR_LOW = "hel1os-hxr-8-30keV"    # CdTe-ish softer hard band
STREAM_HXR_HIGH = "hel1os-hxr-30-70keV"  # harder band (spikier, non-thermal)

SOURCE = "synth"

# DataFrame column names mandated by Appendix B.2.
_DF_COLUMNS = ("t", "sxr_long", "sxr_short", "hxr_8_30", "hxr_30_70")

# Quiet-Sun backgrounds (canonical units).
_SXR_BACKGROUND_WM2 = 1.0e-8          # ~A0 quiet-Sun long-channel level
_SXR_SHORT_FRACTION = 0.10            # short channel ~10% of long at quiet Sun
_HXR_BACKGROUND_CPS_LOW = 120.0       # quiet HEL1OS-ish background (counts/s)
_HXR_BACKGROUND_CPS_HIGH = 30.0       # harder band has lower background


@dataclass(slots=True)
class TruthEvent:
    """Ground-truth synthetic flare (one physical event).

    The labels the detector / forecaster are scored against. Times are epoch
    seconds UTC (same clock as :class:`~flarecast.types.FluxSample.t`).

    Attributes
    ----------
    t_start:
        Soft-band onset (where the FRED rise lifts off background).
    t_peak:
        Soft-band peak (where the GOES class is defined).
    t_end:
        Soft-band end (FSM midpoint-decay rule: flux back to
        ``(peak + start)/2`` above background).
    goes_class:
        GOES class string, e.g. ``"M2.5"``.
    peak_flux_wm2:
        Peak 1-8 A flux above background [W m^-2] (sets the class).
    hxr_lead_s:
        How far the impulsive hard-X-ray phase leads the soft peak [s]
        (the Neupert early-warning margin for this event).
    n_hxr_spikes:
        Number of impulsive hard sub-bursts in this event.
    saturated:
        Whether this event drove the SDD1 channel into paralyzable rollover
        (true count rate exceeded :data:`SDD_SATURATION_CPS`).
    """

    t_start: float
    t_peak: float
    t_end: float
    goes_class: str
    peak_flux_wm2: float
    hxr_lead_s: float
    n_hxr_spikes: int
    saturated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict of this truth event."""
        return asdict(self)


@dataclass(slots=True)
class _FlareSpec:
    """Internal per-flare parameters chosen before rendering."""

    t_peak: float
    rise_s: float
    decay_s: float
    peak_flux_wm2: float
    goes_class: str
    hxr_lead_s: float
    n_spikes: int


# ---------------------------------------------------------------------------
# class-mix sampling
# ---------------------------------------------------------------------------
def _normalize_mix(class_mix: dict[str, float] | None) -> list[tuple[str, float]]:
    """Return a normalized, ordered list of ``(class_letter, probability)``."""
    mix = dict(class_mix) if class_mix else dict(DEFAULT_CLASS_MIX)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError(f"class_mix probabilities must sum to > 0, got {mix!r}")
    # deterministic order (ascending class intensity) so seeded sampling is
    # reproducible across runs / platforms.
    order = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    items = sorted(mix.items(), key=lambda kv: order.get(kv[0].upper(), 99))
    return [(letter.upper(), p / total) for letter, p in items]


def _sample_class(mix: list[tuple[str, float]], rng: random.Random) -> str:
    """Draw a class letter from the normalized mix."""
    r = rng.random()
    acc = 0.0
    for letter, p in mix:
        acc += p
        if r <= acc:
            return letter
    return mix[-1][0]


def _sample_peak_flux(letter: str, rng: random.Random) -> tuple[float, str]:
    """Sample a peak flux + full class string within a class letter's decade."""
    base = _profiles.goes_class_to_peak_flux(letter)  # decade floor (mantissa 1)
    # mantissa in [1, 10) for A..M; X is open-ended (allow up to ~20 => X20).
    if letter == "X":
        mantissa = 1.0 + rng.random() * 19.0
    else:
        mantissa = 1.0 + rng.random() * 8.9
    peak_flux = base * mantissa
    cls = _profiles.peak_flux_to_goes_class(peak_flux)
    return peak_flux, cls


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def generate_flare_lightcurves(
    duration_s: float = 86400.0,
    cadence_s: float = 1.0,
    n_flares: int | None = None,
    class_mix: dict[str, float] | None = None,
    neupert: bool = True,
    noise: bool = True,
    gaps: bool = True,
    spikes: bool = True,
    seed: int | None = None,
) -> tuple[list[FluxSample], list[TruthEvent]]:
    """Generate synthetic soft + hard X-ray streams and a truth event list.

    Builds a quiet-Sun baseline over ``[0, duration_s)`` at ``cadence_s`` and
    superimposes ``n_flares`` physics-based flares whose classes are drawn from
    ``class_mix`` (default :data:`DEFAULT_CLASS_MIX` = 70% C / 25% M / 5% X).
    Each flare is a FRED soft pulse (long + short channels) with a
    Neupert-coupled impulsive hard burst train (two hard bands) that leads the
    soft peak. Optional noise, cosmic-ray spikes, data gaps, and SDD1
    paralyzable saturation are layered on to mimic real instrument data.

    The return is the canonical offline product: a flat ``list[FluxSample]``
    spanning four streams (``solexs-sxr-long``, ``solexs-sxr-short``,
    ``hel1os-hxr-8-30keV``, ``hel1os-hxr-30-70keV``), sorted by ``(t, stream)``,
    plus a ``list[TruthEvent]`` (start / peak / end / class per flare). Use
    :func:`as_dataframe` for the wide ``[t, sxr_long, sxr_short, hxr_8_30,
    hxr_30_70]`` pandas view (Appendix B.2) and :func:`truth_to_dataframe` for
    the truth table -- both optional and pandas-guarded.

    Parameters
    ----------
    duration_s:
        Total span to generate [s] (default one day).
    cadence_s:
        Sample spacing [s] (default 1 s, the SoLEXS/HEL1OS light-curve grid).
    n_flares:
        Number of flares to inject. If ``None`` a sensible default scaling with
        duration is used (~ a handful per day).
    class_mix:
        ``{"C":0.7,"M":0.25,"X":0.05}``-style probabilities; normalized
        internally. Defaults to :data:`DEFAULT_CLASS_MIX`.
    neupert:
        If True (default) the hard train follows ``d/dt SXR`` and leads the
        soft peak; if False the hard band carries only background (no coupling).
    noise:
        If True, apply Poisson shot noise to the hard channels and small
        Gaussian noise to the soft channels.
    gaps:
        If True, inject a couple of data gaps.
    spikes:
        If True, inject cosmic-ray spikes into the hard channels.
    seed:
        Seed for the internal ``random.Random``; a fixed seed reproduces the
        exact streams and truth list (determinism contract).

    Returns
    -------
    ``(samples, truth)`` -- ``samples`` is ``list[FluxSample]``; ``truth`` is
    ``list[TruthEvent]``.

    Raises
    ------
    ValueError
        For non-positive ``duration_s`` / ``cadence_s`` or an empty class mix.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if cadence_s <= 0:
        raise ValueError(f"cadence_s must be > 0, got {cadence_s}")

    rng = random.Random(seed)
    mix = _normalize_mix(class_mix)

    n = int(math.floor(duration_s / cadence_s))
    if n <= 0:
        raise ValueError("duration_s / cadence_s yields zero samples")
    t_axis = [i * cadence_s for i in range(n)]

    # default flare count: ~5 per day, scaled by duration, >=1 when duration
    # is long enough to host a flare.
    if n_flares is None:
        n_flares = max(1, int(round(5.0 * duration_s / 86400.0)))
    n_flares = max(0, n_flares)

    # ----- choose flare specs -----
    specs = _choose_flare_specs(n_flares, duration_s, cadence_s, mix, rng)

    # ----- render channels (pure-python lists) -----
    sxr_long = [_SXR_BACKGROUND_WM2] * n
    # accumulate the *clean* soft long signal per flare so we can build the
    # Neupert hard drive from each flare's own derivative.
    hxr_low_clean = [0.0] * n
    hxr_high_clean = [0.0] * n
    # true (pre-saturation) soft long counts proxy for SDD saturation modelling.
    # We model saturation on the long channel's count-rate proxy.
    truth_events: list[TruthEvent] = []

    for spec in specs:
        # soft FRED pulse for this flare (above background).
        pulse = _profiles.fred_profile(
            t_axis, spec.t_peak, spec.rise_s, spec.decay_s, spec.peak_flux_wm2
        )
        for i in range(n):
            sxr_long[i] += pulse[i]

        # Neupert hard train derived from THIS flare's soft pulse.
        if neupert and spec.n_spikes >= 0:
            train = _profiles.impulsive_hxr_train(
                t_axis,
                pulse,
                n_spikes=spec.n_spikes,
                lag_s=spec.hxr_lead_s,
                cadence_s=cadence_s,
                spike_width_s=max(10.0, 2.0 * cadence_s),
                rng=rng,
            )
            # scale the (small) derivative-based shape up to a count-rate excess.
            # harder band: spikier (use a higher power emphasis) and lower amp.
            scale_low = _hxr_scale_for_class(spec.goes_class)
            scale_high = 0.45 * scale_low
            tmax = max(train) if train else 0.0
            if tmax > 0:
                for i in range(n):
                    frac = train[i] / tmax
                    hxr_low_clean[i] += scale_low * frac
                    # harder band emphasises the sharp peaks (square the frac).
                    hxr_high_clean[i] += scale_high * (frac * frac)

        # ----- truth start/end from the clean soft pulse (FSM rules) -----
        t_start, t_end = _flare_start_end(
            t_axis, pulse, spec.t_peak, spec.peak_flux_wm2
        )
        # whether this flare saturates SDD1 (true long count-rate proxy).
        peak_counts = _wm2_to_sdd_counts(spec.peak_flux_wm2)
        saturated = peak_counts > SDD_SATURATION_CPS
        truth_events.append(
            TruthEvent(
                t_start=t_start,
                t_peak=spec.t_peak,
                t_end=t_end,
                goes_class=spec.goes_class,
                peak_flux_wm2=spec.peak_flux_wm2,
                hxr_lead_s=spec.hxr_lead_s,
                n_hxr_spikes=spec.n_spikes,
                saturated=saturated,
            )
        )

    # short channel: a harder-spectrum fraction of the long channel; during
    # flares the short/long ratio rises (hotter plasma), modelled simply as a
    # slightly larger fraction of the above-background excess.
    sxr_short = [0.0] * n
    for i in range(n):
        excess = sxr_long[i] - _SXR_BACKGROUND_WM2
        if excess < 0:
            excess = 0.0
        sxr_short[i] = (
            _SXR_BACKGROUND_WM2 * _SXR_SHORT_FRACTION + 0.35 * excess
        )

    # hard channels: add quiet background to the clean excess.
    hxr_low = [_HXR_BACKGROUND_CPS_LOW + v for v in hxr_low_clean]
    hxr_high = [_HXR_BACKGROUND_CPS_HIGH + v for v in hxr_high_clean]

    # ----- artifacts & noise (order matters: saturation on true counts, then
    # cosmic spikes, then Poisson) -----
    # SDD1 saturation acts on the soft long channel's count-rate proxy, then we
    # fold the rollover back into the flux so a saturating flare shows the
    # characteristic dip in the long channel.
    sxr_long = _apply_soft_saturation(sxr_long)

    if spikes:
        hxr_low = _noise.add_cosmic_spikes(
            hxr_low, rate_per_hr=6.0, rng=rng, cadence_s=cadence_s
        )
        hxr_high = _noise.add_cosmic_spikes(
            hxr_high, rate_per_hr=3.0, rng=rng, cadence_s=cadence_s
        )

    if noise:
        hxr_low = _noise.add_poisson_noise(hxr_low, rng)
        hxr_high = _noise.add_poisson_noise(hxr_high, rng)
        sxr_long = _noise.add_gaussian_noise(
            sxr_long, rng, rel_sigma=0.02, abs_sigma=0.05 * _SXR_BACKGROUND_WM2
        )
        sxr_short = _noise.add_gaussian_noise(
            sxr_short,
            rng,
            rel_sigma=0.02,
            abs_sigma=0.05 * _SXR_BACKGROUND_WM2 * _SXR_SHORT_FRACTION,
        )

    # ----- assemble wide columns, optionally punch gaps -----
    columns: dict[str, list[float]] = {
        "t": list(t_axis),
        "sxr_long": list(sxr_long),
        "sxr_short": list(sxr_short),
        "hxr_8_30": list(hxr_low),
        "hxr_30_70": list(hxr_high),
    }
    gap_mask = [True] * n
    if gaps:
        n_gaps = max(1, int(round(2.0 * duration_s / 86400.0)))
        # build a keep mask so we can also stamp DATA_GAP-adjacent samples if
        # ever needed; here we simply drop gap rows from the emitted samples.
        gap_mask = _noise._gap_keep_mask(
            t_axis, n_gaps=n_gaps, max_len_s=min(600.0, duration_s * 0.02),
            rng=rng, cadence_s=cadence_s,
        )

    # ----- emit canonical FluxSample list -----
    samples = _emit_samples(columns, gap_mask)

    # sort truth events by peak time for stable, readable output.
    truth_events.sort(key=lambda e: e.t_peak)
    return samples, truth_events


# ---------------------------------------------------------------------------
# flare-spec selection & helpers
# ---------------------------------------------------------------------------
def _choose_flare_specs(
    n_flares: int,
    duration_s: float,
    cadence_s: float,
    mix: list[tuple[str, float]],
    rng: random.Random,
) -> list[_FlareSpec]:
    """Choose non-degenerate flare parameters spread across the time span."""
    specs: list[_FlareSpec] = []
    # keep peaks away from the very edges so rise/decay fit in-window.
    margin = min(0.1 * duration_s, 1800.0)
    lo, hi = margin, max(margin + cadence_s, duration_s - margin)
    for _ in range(n_flares):
        letter = _sample_class(mix, rng)
        peak_flux, cls = _sample_peak_flux(letter, rng)
        t_peak = rng.uniform(lo, hi)
        # bigger flares -> longer rise/decay (loosely class-scaled); the
        # size_factor is the flux's order of magnitude above quiet Sun (~2..5).
        size_factor = math.log10(peak_flux / _SXR_BACKGROUND_WM2)
        rise_s = rng.uniform(60.0, 300.0) * max(1.0, size_factor / 3.0)
        decay_s = rng.uniform(300.0, 1800.0) * max(1.0, size_factor / 3.0)
        # Neupert lead: ~1-3 min, larger for bigger flares.
        hxr_lead_s = rng.uniform(60.0, 180.0) * max(1.0, size_factor / 3.0)
        # spike count scales with class: more impulsive sub-bursts for M/X.
        n_spikes = max(1, int(round(rng.uniform(1.0, 2.0) * size_factor)))
        specs.append(
            _FlareSpec(
                t_peak=t_peak,
                rise_s=rise_s,
                decay_s=decay_s,
                peak_flux_wm2=peak_flux,
                goes_class=cls,
                hxr_lead_s=hxr_lead_s,
                n_spikes=n_spikes,
            )
        )
    specs.sort(key=lambda s: s.t_peak)
    return specs


def _flare_start_end(
    t_axis: Sequence[float],
    pulse: Sequence[float],
    t_peak: float,
    peak_amp: float,
) -> tuple[float, float]:
    """Derive (t_start, t_end) from a clean FRED pulse.

    Start = first time the pulse rises above a small onset fraction of its
    peak; end = first time after the peak the pulse decays back below the FSM
    midpoint level ``peak/2`` (research-grade GOES midpoint-decay convention,
    ARCHITECTURE.md Section 4.4 end rule simplified to the half-peak crossing).
    """
    n = len(t_axis)
    onset_level = 0.05 * peak_amp
    mid_level = 0.5 * peak_amp
    # peak index.
    pk = max(range(n), key=lambda i: pulse[i])
    # start: scan forward from 0 to first crossing of onset_level before peak.
    t_start = t_axis[0]
    for i in range(pk + 1):
        if pulse[i] >= onset_level:
            t_start = t_axis[i]
            break
    # end: scan forward from peak to first drop below mid_level.
    t_end = t_axis[-1]
    for i in range(pk, n):
        if pulse[i] <= mid_level:
            t_end = t_axis[i]
            break
    return t_start, t_end


def _hxr_scale_for_class(cls: str) -> float:
    """Peak hard-band count-rate excess (counts/s) for a flare class."""
    letter = cls[0].upper()
    return {
        "A": 200.0,
        "B": 600.0,
        "C": 2.0e3,
        "M": 1.2e4,
        "X": 6.0e4,
    }.get(letter, 1.0e3)


def _wm2_to_sdd_counts(flux_wm2: float) -> float:
    """Rough SoLEXS SDD1 true count-rate proxy for a given GOES flux.

    A log-linear proxy anchored so that an X-class (1e-4 W/m^2) flare drives
    the SDD1 true rate above the ~1e5 cps saturation ceiling (research 01
    S1.4), while C/B sit comfortably below it. Used only to *flag* truth-event
    saturation and to drive the long-channel rollover artifact.
    """
    if flux_wm2 <= 0:
        return 0.0
    # map flux decades to count decades: 1e-6 (C1) -> ~1e3, 1e-4 (X1) -> ~2e5.
    log_flux = math.log10(flux_wm2)
    # linear in log: counts = 10**(a + b*log_flux)
    # solve through (log -6 -> 3) and (log -4 -> 5.3):
    b = (5.3 - 3.0) / (-4.0 - -6.0)  # 1.15
    a = 3.0 - b * (-6.0)             # 9.9
    return 10.0 ** (a + b * log_flux)


def _apply_soft_saturation(sxr_long: list[float]) -> list[float]:
    """Fold SDD1 paralyzable rollover into the soft long-channel flux.

    Converts each flux to a true count-rate proxy, applies the paralyzable
    rollover (:func:`flarecast.synth.noise.apply_sdd_saturation`), and maps the
    *observed* counts back to a flux. Below saturation this is a no-op (linear);
    above it the long channel shows the characteristic flare-max dip.
    """
    n = len(sxr_long)
    true_counts = [_wm2_to_sdd_counts(f) for f in sxr_long]
    obs_counts = _noise.apply_sdd_saturation(true_counts, ceiling=SDD_SATURATION_CPS)
    out = [0.0] * n
    for i in range(n):
        if true_counts[i] <= 0:
            out[i] = sxr_long[i]
            continue
        # scale flux by the observed/true count ratio (the dead-time loss).
        ratio = obs_counts[i] / true_counts[i]
        out[i] = sxr_long[i] * ratio
    return out


def _emit_samples(columns: dict[str, list[float]], keep_mask: list[bool]) -> list[FluxSample]:
    """Flatten wide columns into a sorted ``list[FluxSample]`` honouring gaps."""
    t = columns["t"]
    n = len(t)
    samples: list[FluxSample] = []

    stream_specs = (
        (STREAM_SXR_LONG, "sxr_long", UNIT_SXR, Quantity.SXR_LONG.value, "0.1-0.8nm", True),
        (STREAM_SXR_SHORT, "sxr_short", UNIT_SXR, Quantity.SXR_SHORT.value, "0.05-0.4nm", True),
        (STREAM_HXR_LOW, "hxr_8_30", UNIT_HXR, Quantity.HXR.value, "8-30keV", False),
        (STREAM_HXR_HIGH, "hxr_30_70", UNIT_HXR, Quantity.HXR.value, "30-70keV", False),
    )

    for i in range(n):
        if not keep_mask[i]:
            continue
        ti = float(t[i])
        for stream, col, unit, quantity, band, derive_cls in stream_specs:
            val = float(columns[col][i])
            cls = None
            if derive_cls and col == "sxr_long":
                # derive class from the long channel above background.
                above = max(0.0, val - _SXR_BACKGROUND_WM2)
                cls = _profiles.peak_flux_to_goes_class(above) if above > 0 else None
            samples.append(
                FluxSample(
                    stream=stream,
                    t=ti,
                    value=val,
                    unit=unit,
                    source=SOURCE,
                    quantity=quantity,
                    cls=cls,
                    qc=QCBit.GOOD.value,
                    meta={"band": band, "synthetic": True},
                )
            )
    # stable sort by (t, stream) for deterministic, readable output.
    samples.sort(key=lambda s: (s.t, s.stream))
    return samples


# ---------------------------------------------------------------------------
# optional pandas views (Appendix B.2 DataFrame shape) -- guarded
# ---------------------------------------------------------------------------
def as_dataframe(samples: list[FluxSample]):
    """Return the wide ``[t, sxr_long, sxr_short, hxr_8_30, hxr_30_70]`` frame.

    Optional pandas accelerator (Appendix B.2). Pivots a flat
    ``list[FluxSample]`` (as produced by :func:`generate_flare_lightcurves`)
    into one row per timestamp with one column per stream, the wide shape the
    contract documents for the DataFrame return.

    Requires pandas; raises a clear :class:`ImportError` (actionable) if pandas
    is not installed -- the pure-python ``list[FluxSample]`` path never needs
    this.

    Parameters
    ----------
    samples:
        Flat synthetic samples.

    Returns
    -------
    pandas.DataFrame with columns ``t, sxr_long, sxr_short, hxr_8_30,
    hxr_30_70`` sorted by ``t``.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "as_dataframe() requires pandas (an optional accelerator). Install "
            "it with `pip install pandas`, or use the pure-python "
            "list[FluxSample] returned by generate_flare_lightcurves()."
        ) from exc

    stream_to_col = {
        STREAM_SXR_LONG: "sxr_long",
        STREAM_SXR_SHORT: "sxr_short",
        STREAM_HXR_LOW: "hxr_8_30",
        STREAM_HXR_HIGH: "hxr_30_70",
    }
    rows: dict[float, dict[str, float]] = {}
    for s in samples:
        col = stream_to_col.get(s.stream)
        if col is None:
            continue
        rows.setdefault(s.t, {})[col] = s.value
    ordered_t = sorted(rows.keys())
    data = {c: [] for c in _DF_COLUMNS}
    for ti in ordered_t:
        data["t"].append(ti)
        for c in _DF_COLUMNS[1:]:
            data[c].append(rows[ti].get(c, float("nan")))
    return pd.DataFrame(data, columns=list(_DF_COLUMNS))


def truth_to_dataframe(truth: list[TruthEvent]):
    """Return the truth event list as a pandas DataFrame (optional accelerator).

    Requires pandas; raises a clear :class:`ImportError` otherwise. The
    pure-python ``list[TruthEvent]`` (each with :meth:`TruthEvent.to_dict`) is
    always available without pandas.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "truth_to_dataframe() requires pandas (an optional accelerator). "
            "Use the pure-python list[TruthEvent] otherwise."
        ) from exc
    return pd.DataFrame([e.to_dict() for e in truth])
