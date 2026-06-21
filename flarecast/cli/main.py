"""``flarecast`` command-line entry point (ARCHITECTURE.md Appendix B.7).

The console script (``flarecast``) and ``python -m flarecast.cli.main`` both
dispatch here. Subcommands wire the already-built workstreams into runnable,
**offline-capable** flows:

``demo``
    End-to-end offline proof: synth -> dual-band detection -> master catalogue
    -> features + baseline/GBT forecast -> a detected-vs-truth flare table,
    classification summary, and forecast TSS + median lead time, with the
    catalogue written to SQLite + JSON. Runs with **no network** and only the
    installed core deps. This is the acceptance test for the entry point.
``ingest``
    Pull GOES via :class:`~flarecast.ingest.goes.GOESFetcher` (live -> cache ->
    synth), run detection, populate a :class:`~flarecast.api.store.ReadStore`.
``detect``
    Run dual-band detection over synthetic data (or a provided CSV/JSON file).
``forecast``
    Build labels from a synthetic catalogue, run leak-free CV, train + evaluate
    a GBT (with baselines), and print the metric battery + the LT-vs-FAR table.
``serve``
    Launch uvicorn with :func:`flarecast.api.app.create_app` + the dashboard on
    synthetic / cached data (so it works offline).
``backtest``
    Run the evaluation battery and print TSS/HSS/BSS/POD/FAR + lead-time buckets.

The module is import-safe: heavy / optional deps (FastAPI, uvicorn, lightgbm,
numpy) are imported only inside the subcommand that needs them, so
``import flarecast.cli.main`` never fails in a minimal environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from ..constants import (
    FORECAST_DEFAULT_CLASS_THRESHOLD,
)

__all__ = ["main", "run_pipeline", "run_offline_pipeline"]


# ===========================================================================
# Shared pipeline helpers (pure-python + core deps; no network).
# ===========================================================================
@dataclass(slots=True)
class _BandFlare:
    """A single detected per-band flare (collected from the streaming FSM)."""

    onset_time: float
    peak_time: float | None = None
    end_time: float | None = None
    peak_value: float = 0.0
    goes_class: str | None = None
    detectors: tuple[str, ...] = ()
    spike_rejected: int = 0
    data_gap: bool = False


def _pivot_soft_hard(samples: list) -> tuple[list[float], dict[float, float], dict[float, float]]:
    """Pivot a flat ``list[FluxSample]`` into (times, soft, hard) by timestamp.

    Soft = the synth ``solexs-sxr-long`` channel (GOES-scale W/m^2); hard = the
    ``hel1os-hxr-8-30keV`` channel (counts/s). Returns the sorted unique times
    and two ``{t: value}`` maps.
    """
    from ..synth.generator import STREAM_HXR_LOW, STREAM_SXR_LONG

    soft: dict[float, float] = {}
    hard: dict[float, float] = {}
    for s in samples:
        if s.stream == STREAM_SXR_LONG:
            soft[s.t] = s.value
        elif s.stream == STREAM_HXR_LOW:
            hard[s.t] = s.value
    times = sorted(set(soft) | set(hard))
    return times, soft, hard


def _run_detection(
    times: list[float],
    soft: dict[float, float],
    hard: dict[float, float],
    cadence_s: float,
) -> tuple[list[_BandFlare], list[_BandFlare]]:
    """Stream soft + hard channels through the detectors -> per-band flares.

    Returns ``(soft_flares, hard_flares)``. The soft FSM yields onset/peak/end;
    the hard detector yields onset (and a sustained-return end). O(1)/sample.
    """
    from ..detect.stack import HardBandDetector, SoftBandDetector

    soft_det = SoftBandDetector(cadence_s=cadence_s)
    hard_det = HardBandDetector(cadence_s=cadence_s)

    soft_flares: list[_BandFlare] = []
    hard_flares: list[_BandFlare] = []
    open_soft: _BandFlare | None = None
    open_hard: _BandFlare | None = None

    for t in times:
        sx = soft.get(t, 1e-8)
        hx = hard.get(t, 0.0)
        s_state = soft_det.update(sx, t)
        h_state = hard_det.update(hx, t)
        s_meta = s_state.meta or {}
        h_meta = h_state.meta or {}

        # ---- soft band FSM (onset -> peak -> end) ----
        if s_state.onset:
            open_soft = _BandFlare(
                onset_time=(s_state.onset_time if s_state.onset_time is not None else t),
                detectors=tuple(s_meta.get("detectors", []) or []),
            )
        if open_soft is not None:
            if s_state.peak:
                open_soft.peak_time = t
                pf = s_meta.get("peak_flux")
                if isinstance(pf, (int, float)):
                    open_soft.peak_value = float(pf)
                gc = s_meta.get("goes_class")
                if gc:
                    open_soft.goes_class = gc
            open_soft.spike_rejected = int(s_meta.get("spike_rejected", 0) or 0)
            open_soft.data_gap = open_soft.data_gap or bool(s_meta.get("data_gap"))
            if s_state.end:
                open_soft.end_time = t
                soft_flares.append(open_soft)
                open_soft = None

        # ---- hard band (onset -> sustained return) ----
        if h_state.onset:
            open_hard = _BandFlare(
                onset_time=(h_state.onset_time if h_state.onset_time is not None else t),
                peak_value=hx,
                detectors=tuple(h_meta.get("detectors", []) or []),
            )
            # Track peak counts as the burst evolves below.
        if open_hard is not None:
            if hx > open_hard.peak_value:
                open_hard.peak_value = hx
                open_hard.peak_time = t
            if not h_state.in_event:  # burst closed this sample
                hard_flares.append(open_hard)
                open_hard = None

    # Flush any still-open flares at stream end.
    if open_soft is not None:
        soft_flares.append(open_soft)
    if open_hard is not None:
        hard_flares.append(open_hard)
    return soft_flares, hard_flares


def _soft_detection_state(bf: _BandFlare):
    """Build an enriched soft :class:`DetectionState` (onset) for association."""
    from ..types import DetectionState

    return DetectionState(
        onset=True,
        in_event=False,
        statistic=0.0,
        onset_time=bf.onset_time,
        peak=False,
        end=False,
        meta={
            "band": "soft",
            "t_start": bf.onset_time,
            "t_peak": bf.peak_time if bf.peak_time is not None else bf.onset_time,
            "t_end": bf.end_time if bf.end_time is not None else bf.peak_time,
            "peak_flux": bf.peak_value,
            "goes_class": bf.goes_class,
            "detector_used": "SDD1",
            "detectors": list(bf.detectors) or ["CUSUM"],
            "spike_rejected": bf.spike_rejected,
            "data_gap": bf.data_gap,
        },
    )


def _hard_detection_state(bf: _BandFlare):
    """Build an enriched hard :class:`DetectionState` (onset) for association."""
    from ..types import DetectionState

    return DetectionState(
        onset=True,
        in_event=False,
        statistic=0.0,
        onset_time=bf.onset_time,
        peak=False,
        end=False,
        meta={
            "band": "hard",
            "t_start": bf.onset_time,
            "t_peak": bf.peak_time if bf.peak_time is not None else bf.onset_time,
            "peak_counts": bf.peak_value,
            "energy_band": "8-30keV",
            "detectors": list(bf.detectors) or ["FOCuS"],
            "spike_rejected": bf.spike_rejected,
        },
    )


def _build_catalogue(soft_flares: list[_BandFlare], hard_flares: list[_BandFlare]) -> list:
    """Associate per-band flares into a deduplicated master catalogue.

    Uses the real :class:`~flarecast.catalog.associate.Associator`,
    :func:`~flarecast.catalog.dedup.deduplicate`, and noisy-OR confidence
    (:func:`~flarecast.catalog.confidence.fuse_confidence`).
    """
    from ..catalog.associate import Associator
    from ..catalog.confidence import fuse_confidence
    from ..catalog.dedup import deduplicate

    assoc = Associator()
    # Interleave detections in onset-time order so the asymmetric Neupert
    # window pairs hard-before-soft correctly.
    tagged: list[tuple[float, str, Any]] = []
    for bf in soft_flares:
        tagged.append((bf.onset_time, "soft", _soft_detection_state(bf)))
    for bf in hard_flares:
        tagged.append((bf.onset_time, "hard", _hard_detection_state(bf)))
    tagged.sort(key=lambda x: x[0])

    events: list = []
    for t, band, det in tagged:
        ev = assoc.add(det, band, t)
        if ev is not None:
            events.append(ev)
    events.extend(assoc.flush())

    events = deduplicate(events)

    # Fill noisy-OR confidence from the firing detectors + physical priors.
    for ev in events:
        n_det = max(1, len(ev.detectors))
        per_det = [0.85] * n_det  # nominal calibrated per-detector probability
        ev.confidence = fuse_confidence(
            per_det,
            cross_band_agreement=bool(ev.flags.get("soft") and ev.flags.get("hard")),
            neupert_consistent=bool(ev.flags.get("neupert_consistent")),
        )
    events.sort(key=lambda e: e.t_peak)
    return events


def _match_detected_to_truth(events: list, truth: list, tol_s: float = 600.0) -> list[dict]:
    """Greedily match detected events to truth flares by peak time (+/- tol).

    Returns a per-truth row dict: truth class/peak + matched detected class /
    confidence / peak-time error (``None`` if missed).
    """
    used = [False] * len(events)
    rows: list[dict] = []
    for te in truth:
        best_i = -1
        best_dt = tol_s
        for i, ev in enumerate(events):
            if used[i]:
                continue
            dt = abs(ev.t_peak - te.t_peak)
            if dt <= best_dt:
                best_dt = dt
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            ev = events[best_i]
            rows.append({
                "truth_class": te.goes_class,
                "truth_peak": te.t_peak,
                "det_class": ev.goes_class,
                "det_peak": ev.t_peak,
                "dt_s": ev.t_peak - te.t_peak,
                "confidence": ev.confidence,
                "soft": bool(ev.flags.get("soft")),
                "hard": bool(ev.flags.get("hard")),
                "neupert": bool(ev.flags.get("neupert_consistent")),
                "matched": True,
            })
        else:
            rows.append({
                "truth_class": te.goes_class,
                "truth_peak": te.t_peak,
                "det_class": None,
                "det_peak": None,
                "dt_s": None,
                "confidence": None,
                "soft": False,
                "hard": False,
                "neupert": False,
                "matched": False,
            })
    n_extra = sum(1 for u in used if not u)
    if n_extra:
        for i, ev in enumerate(events):
            if not used[i]:
                rows.append({
                    "truth_class": None,
                    "truth_peak": None,
                    "det_class": ev.goes_class,
                    "det_peak": ev.t_peak,
                    "dt_s": None,
                    "confidence": ev.confidence,
                    "soft": bool(ev.flags.get("soft")),
                    "hard": bool(ev.flags.get("hard")),
                    "neupert": bool(ev.flags.get("neupert_consistent")),
                    "matched": False,
                    "false_alarm": True,
                })
    return rows


@dataclass(slots=True)
class OfflinePipelineResult:
    """Container for the offline pipeline outputs (used by ``demo`` + examples)."""

    samples: list
    truth: list
    soft_flares: list
    hard_flares: list
    events: list
    match_rows: list[dict]
    forecast_summary: dict
    out_dir: str
    sqlite_path: str
    json_path: str


def run_offline_pipeline(
    *,
    duration_s: float = 86400.0,
    cadence_s: float = 60.0,
    n_flares: int | None = 8,
    seed: int | None = 42,
    horizon_min: float = 30.0,
    class_threshold: str = FORECAST_DEFAULT_CLASS_THRESHOLD,
    out_dir: str | None = None,
    do_forecast: bool = True,
) -> OfflinePipelineResult:
    """Run synth -> detect -> catalogue -> forecast fully offline.

    Returns an :class:`OfflinePipelineResult`. Persists the catalogue to SQLite
    and JSON under ``out_dir`` (a temp dir if ``None``). Pure-python + the
    installed core deps; never touches the network.
    """
    from ..synth.generator import generate_flare_lightcurves

    samples, truth = generate_flare_lightcurves(
        duration_s=duration_s, cadence_s=cadence_s, n_flares=n_flares, seed=seed
    )
    times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, cadence_s)
    events = _build_catalogue(soft_flares, hard_flares)
    match_rows = _match_detected_to_truth(events, truth)

    forecast_summary: dict = {}
    if do_forecast:
        forecast_summary = _offline_forecast(
            samples, events, truth, times, soft, hard,
            horizon_min=horizon_min, class_threshold=class_threshold, seed=seed,
        )

    # --- persist the catalogue (SQLite + JSON) ---
    if out_dir is None:
        out_dir = os.path.join(tempfile.gettempdir(), "flarecast_demo")
    os.makedirs(out_dir, exist_ok=True)
    sqlite_path = os.path.join(out_dir, "catalogue.sqlite")
    json_path = os.path.join(out_dir, "catalogue.json")
    _persist_catalogue(events, sqlite_path, json_path)

    return OfflinePipelineResult(
        samples=samples,
        truth=truth,
        soft_flares=soft_flares,
        hard_flares=hard_flares,
        events=events,
        match_rows=match_rows,
        forecast_summary=forecast_summary,
        out_dir=out_dir,
        sqlite_path=sqlite_path,
        json_path=json_path,
    )


def _persist_catalogue(events: list, sqlite_path: str, json_path: str) -> None:
    """Write the catalogue to a fresh SQLite DB and a JSON array."""
    from ..catalog.index import CatalogStore

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    store = CatalogStore(sqlite_path)
    if events:
        store.insert_many(events)
    store.close()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([e.to_json() for e in events], fh, indent=2, default=str)


def _offline_forecast(
    samples: list,
    events: list,
    truth: list,
    times: list[float],
    soft: dict[float, float],
    hard: dict[float, float],
    *,
    horizon_min: float,
    class_threshold: str,
    seed: int | None,
) -> dict:
    """Build labels, CV, train GBT (if available) + baselines, evaluate.

    Returns a summary dict: model TSS/HSS/BSS/POD/FAR, baseline TSS, median lead
    time, and the LT-vs-FAR sweep rows. numpy is required (a core dep); lightgbm
    is used opportunistically (falls back to sklearn, then to baselines-only).
    """
    import numpy as np

    from ..forecast.baselines import ClimatologyBaseline, PersistenceBaseline
    from ..forecast.cv import blocked_splits
    from ..forecast.evaluate import report
    from ..forecast.labels import build_labels
    from ..forecast.leadtime import lead_time_report, lt_vs_far

    # Light-curve window in the lenient mapping the feature extractor accepts.
    lc = {
        "t": list(times),
        "sxr": [soft.get(t, 1e-8) for t in times],
        "hxr": [hard.get(t, 0.0) for t in times],
    }
    X, y = build_labels(events, lc, horizon_min, class_threshold=class_threshold)
    summary: dict = {
        "horizon_min": horizon_min,
        "class_threshold": class_threshold,
        "n_samples": int(X.shape[0]),
        "n_pos": int(y.sum()) if X.shape[0] else 0,
    }
    if X.shape[0] < 20 or int(y.sum()) < 2 or int((y == 0).sum()) < 2:
        summary["note"] = "insufficient labelled data for CV; skipped model fit"
        return summary

    t_axis = np.asarray(lc["t"], dtype=float)
    # The label builder may drop in-event rows, so align a time axis to X by
    # re-deriving kept timestamps (build a parallel kept-time list).
    kept_t = _kept_times(events, lc, horizon_min, class_threshold)
    if len(kept_t) != X.shape[0]:  # defensive: fall back to a uniform axis
        kept_t = [float(i) * 60.0 for i in range(X.shape[0])]
    t_axis = np.asarray(kept_t, dtype=float)

    splits = blocked_splits(t_axis, n_splits=4)
    # Train/eval on the last split (walk-forward headline) with baselines.
    model_probs: list[float] = []
    truth_y: list[int] = []
    used_backend = None
    if splits:
        tr, te = splits[-1]
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        try:
            from ..forecast.model_gbt import GBTForecaster

            gbt = GBTForecaster(random_state=seed or 0)
            gbt.fit(Xtr, ytr)
            p = np.asarray(gbt.predict_proba(Xte), dtype=float)
            used_backend = gbt._backend  # noqa: SLF001 - report which backend ran
            model_probs = p.tolist()
            truth_y = yte.tolist()
        except (RuntimeError, ImportError, ValueError) as exc:
            summary["model_error"] = str(exc)

    # Baselines (always available, pure python).
    clim = ClimatologyBaseline().fit(y)
    pers = PersistenceBaseline()
    clim_p = clim.predict_proba(X)
    pers_p = pers.predict_proba(X)

    if model_probs:
        rep = report(
            truth_y,
            model_probs,
            p_clim=float(y.mean()),
            baselines={
                "climatology": [float(y.mean())] * len(truth_y),
            },
        )
        summary["model"] = {
            "backend": used_backend,
            "tss": rep["tss"],
            "hss": rep["hss"],
            "bss": rep["bss"],
            "pod": rep["pod"],
            "far": rep["far"],
            "roc_auc": rep["roc_auc"],
            "pr_auc": rep["pr_auc"],
            "theta": rep["theta"],
            "n_test": rep["n"],
        }
    # Baseline TSS over the full labelled set (persistence is the bar to beat).
    summary["baselines"] = {
        "climatology_tss": report(y, clim_p)["tss"],
        "persistence_tss": report(y, pers_p)["tss"],
    }

    # --- Lead-time: score the persistence/GBT probability stream against truth.
    # Build a full-series probability with the persistence baseline over every
    # timestamp (so we have a probability aligned with `times` and the truth
    # peaks), then compute median lead time + LT-vs-FAR.
    full_probs = _full_series_probs(lc, horizon_min)
    peak_times = [te.t_peak for te in truth]
    thetas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    lt_rep = lead_time_report(full_probs, list(times), peak_times, theta=0.5)
    summary["lead_time"] = {
        "median_lt_s": lt_rep["median_lt_s"],
        "median_lt_min": (lt_rep["median_lt_s"] / 60.0)
        if not math.isnan(lt_rep["median_lt_s"]) else float("nan"),
        "tpr": lt_rep["tpr"],
        "n_hit": lt_rep["n_hit"],
        "n_miss": lt_rep["n_miss"],
        "frac_ge_min": lt_rep["frac_ge_min"],
    }
    sweep = lt_vs_far(full_probs, list(times), peak_times, thetas)
    summary["lt_vs_far"] = _sweep_to_rows(sweep)
    return summary


def _kept_times(events, lc, horizon_min, class_threshold) -> list[float]:
    """Recompute the kept timestamps from the label builder (in-event masking)."""
    # Re-run the pure-python core but track times in lockstep.
    from ..forecast.features import _window_rows
    from ..forecast.labels import build_labels_lists

    sxr, hxr, times = _window_rows(lc)
    # Reproduce the masking decision the builder makes so kept_t aligns with X.
    _, _ = build_labels_lists(events, lc, horizon_min, class_threshold=class_threshold)
    # Build the bounds + positive predicate identically.
    from ..forecast.labels import (
        _event_bounds,
        class_at_least,
        event_class,
        event_peak,
    )

    qualifying: list[float] = []
    bounds: list[tuple[float, float]] = []
    for ev in events or []:
        try:
            p = event_peak(ev)
        except (KeyError, TypeError, ValueError):
            continue
        if class_at_least(event_class(ev), class_threshold):
            qualifying.append(p)
        ts, te = _event_bounds(ev)
        if ts is None and te is None:
            bounds.append((p - 600.0, p + 600.0))
        else:
            lo = ts if ts is not None else p
            hi = te if te is not None else p
            bounds.append((min(lo, hi), max(lo, hi)))
    qualifying.sort()
    bounds.sort()
    horizon_s = float(horizon_min) * 60.0

    def is_pos(t: float) -> bool:
        for p in qualifying:
            d = p - t
            if d <= 0:
                continue
            if d <= horizon_s:
                return True
            break
        return False

    def in_event(t: float) -> bool:
        for lo, hi in bounds:
            if lo <= t <= hi:
                return True
            if lo > t:
                break
        return False

    kept: list[float] = []
    for t in times:
        pos = is_pos(t)
        if not pos and in_event(t):
            continue
        kept.append(t)
    return kept


def _full_series_probs(lc: dict, horizon_min: float) -> list[float]:
    """A full-length probability series (one per timestamp) for lead-time eval.

    Uses the streaming :class:`FeatureExtractor` + the decayed persistence
    baseline so a probability exists at *every* timestamp aligned with the
    light-curve times (the label builder drops in-event rows, which would break
    the lead-time time axis). Pure python.
    """
    from ..forecast.baselines import PersistenceBaseline
    from ..forecast.features import FeatureExtractor

    ex = FeatureExtractor()
    rows = [
        ex.update(s, h, t, horizon_min)
        for s, h, t in zip(lc["sxr"], lc["hxr"], lc["t"], strict=False)
    ]
    pers = PersistenceBaseline(decayed=False)
    return [float(p) for p in pers.predict_proba(rows)]


def _sweep_to_rows(sweep: Any) -> list[dict]:
    """Normalize the LT-vs-FAR sweep (DataFrame or list-of-dicts) to dict rows."""
    cols = getattr(sweep, "columns", None)
    if cols is not None:  # pandas DataFrame
        return [
            {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
             for k, v in row.items()}
            for row in sweep.to_dict(orient="records")
        ]
    return list(sweep)


# ===========================================================================
# Output formatting helpers.
# ===========================================================================
def _hms(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "   --   "
    s = int(round(seconds))
    sign = "-" if s < 0 else ""
    s = abs(s)
    return f"{sign}{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt(v: Any, spec: str = "", dash: str = "--") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return format(v, spec) if spec else str(v)


def _print_demo(res: OfflinePipelineResult) -> None:
    """Pretty-print the demo's flare table, classification + forecast summary."""
    print("=" * 78)
    print("Aditya FlareCast - END-TO-END OFFLINE demo (synth -> detect -> catalogue -> forecast)")
    print("=" * 78)
    print(
        f"synthetic flares (truth): {len(res.truth)}   "
        f"soft detections: {len(res.soft_flares)}   "
        f"hard detections: {len(res.hard_flares)}   "
        f"master events: {len(res.events)}"
    )
    print()

    # ---- detected-vs-truth flare table ----
    print("Detected vs truth flares")
    print("-" * 78)
    print(
        f"{'truth':<7}{'peak':>10}{'detected':>10}{'dt(s)':>8}"
        f"{'conf':>7}{'S':>3}{'H':>3}{'Neu':>5}  note"
    )
    n_match = 0
    for row in res.match_rows:
        if row.get("false_alarm"):
            print(
                f"{'--':<7}{'--':>10}{_fmt(row['det_class']):>10}{'--':>8}"
                f"{_fmt(row['confidence'], '.2f'):>7}"
                f"{('Y' if row['soft'] else '-'):>3}{('Y' if row['hard'] else '-'):>3}"
                f"{('Y' if row['neupert'] else '-'):>5}  false alarm"
            )
            continue
        matched = row["matched"]
        if matched:
            n_match += 1
        note = "MATCH" if matched else "MISSED"
        print(
            f"{_fmt(row['truth_class']):<7}{_hms(row['truth_peak']):>10}"
            f"{_fmt(row['det_class']):>10}{_fmt(row['dt_s'], '+.0f'):>8}"
            f"{_fmt(row['confidence'], '.2f'):>7}"
            f"{('Y' if row['soft'] else '-'):>3}{('Y' if row['hard'] else '-'):>3}"
            f"{('Y' if row['neupert'] else '-'):>5}  {note}"
        )
    n_truth = sum(1 for r in res.match_rows if r.get("truth_class") is not None)
    pod = (n_match / n_truth) if n_truth else 0.0
    print()
    print(f"detection POD (truth matched): {n_match}/{n_truth} = {pod:.2f}")
    print()

    # ---- classification summary (per-letter) ----
    print("Classification summary (matched events)")
    print("-" * 78)
    by_letter: dict[str, list[int]] = {}
    for row in res.match_rows:
        if row["truth_class"] is None:
            continue
        letter = row["truth_class"][:1].upper()
        hit = 1 if row["matched"] else 0
        b = by_letter.setdefault(letter, [0, 0])
        b[0] += hit
        b[1] += 1
    for letter in ("A", "B", "C", "M", "X"):
        if letter in by_letter:
            hit, tot = by_letter[letter]
            print(f"  class {letter}: detected {hit}/{tot}")
    print()

    # ---- forecast summary ----
    fs = res.forecast_summary
    print("Forecast summary")
    print("-" * 78)
    if not fs:
        print("  (forecast skipped)")
    else:
        print(
            f"  horizon: {fs.get('horizon_min')} min   "
            f">= class {fs.get('class_threshold')}   "
            f"labelled samples: {fs.get('n_samples')} (pos={fs.get('n_pos')})"
        )
        if "note" in fs:
            print(f"  note: {fs['note']}")
        if "model" in fs:
            m = fs["model"]
            print(
                f"  model ({m['backend']}): TSS={m['tss']:+.3f}  HSS={m['hss']:+.3f}  "
                f"BSS={m['bss']:+.3f}  POD={m['pod']:.3f}  FAR={m['far']:.3f}  "
                f"ROC-AUC={m['roc_auc']:.3f}  (theta={m['theta']:.2f}, n={m['n_test']})"
            )
        if "model_error" in fs:
            print(f"  model: not trained ({fs['model_error']})")
        if "baselines" in fs:
            b = fs["baselines"]
            print(
                f"  baselines: climatology TSS={b['climatology_tss']:+.3f}   "
                f"persistence TSS={b['persistence_tss']:+.3f}"
            )
        if "lead_time" in fs:
            lt = fs["lead_time"]
            mlt = lt["median_lt_min"]
            mlt_s = f"{mlt:.1f} min" if not math.isnan(mlt) else "n/a"
            frac = lt["frac_ge_min"]
            frac_s = "  ".join(f">={k}m:{v:.0%}" for k, v in sorted(frac.items()))
            print(
                f"  lead time: median={mlt_s}  TPR={lt['tpr']:.2f}  "
                f"(hits={lt['n_hit']} miss={lt['n_miss']})  {frac_s}"
            )
        if "lt_vs_far" in fs and fs["lt_vs_far"]:
            print("  LT-vs-FAR sweep (theta -> median_lt[min], tpr, far):")
            for r in fs["lt_vs_far"]:
                mlt = r.get("median_lt")
                mlt_min = (mlt / 60.0) if (mlt is not None and not math.isnan(mlt)) else None
                print(
                    f"    theta={r['theta']:.2f}  "
                    f"LT={_fmt(mlt_min, '.1f'):>5}m  "
                    f"TPR={r['tpr']:.2f}  FAR={r['far']:.2f}"
                )
    print()
    print(f"catalogue written: {res.sqlite_path}")
    print(f"                   {res.json_path}")
    print()
    print("OK - end-to-end pipeline ran OFFLINE (no network, core deps only).")


# ===========================================================================
# Subcommand handlers.
# ===========================================================================
def _cmd_demo(args: argparse.Namespace) -> int:
    res = run_offline_pipeline(
        duration_s=args.hours * 3600.0,
        cadence_s=args.cadence,
        n_flares=args.flares,
        seed=args.seed,
        horizon_min=args.horizon,
        class_threshold=args.threshold,
        out_dir=args.out,
        do_forecast=not args.no_forecast,
    )
    _print_demo(res)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    """Run dual-band detection over synthetic data (or a provided file)."""
    if args.file:
        times, soft, hard = _load_lightcurve_file(args.file)
        truth = []
    else:
        from ..synth.generator import generate_flare_lightcurves

        samples, truth = generate_flare_lightcurves(
            duration_s=args.hours * 3600.0, cadence_s=args.cadence, seed=args.seed
        )
        times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, args.cadence)
    print(f"soft-band detections: {len(soft_flares)}")
    print("-" * 60)
    print(f"{'#':>2}  {'onset':>10}{'peak':>10}{'end':>10}{'class':>8}")
    for i, bf in enumerate(soft_flares, 1):
        print(
            f"{i:>2}  {_hms(bf.onset_time):>10}{_hms(bf.peak_time):>10}"
            f"{_hms(bf.end_time):>10}{_fmt(bf.goes_class):>8}"
        )
    print()
    print(f"hard-band detections: {len(hard_flares)}")
    print("-" * 60)
    print(f"{'#':>2}  {'onset':>10}{'peak':>10}{'peak cps':>12}")
    for i, bf in enumerate(hard_flares, 1):
        print(
            f"{i:>2}  {_hms(bf.onset_time):>10}{_hms(bf.peak_time):>10}"
            f"{bf.peak_value:>12.0f}"
        )
    if truth:
        print()
        print(f"(truth flares: {len(truth)})")
    return 0


def _do_ingest(args: argparse.Namespace) -> int:
    """Pull GOES (live->cache->synth), detect, populate a ReadStore."""
    from ..api.store import ReadStore
    from ..ingest.goes import GOESFetcher

    fetcher = GOESFetcher(channel="both", allow_network=not args.offline)
    samples = list(fetcher.fetch(0.0, float("inf")))
    print(f"GOES fetch tier: {fetcher.last_source}   samples: {len(samples)}")

    store = ReadStore(db_path=args.db or ":memory:")
    for s in samples:
        store.put_sample(s)

    # Run soft-band detection on the long channel to populate alerts + catalogue.
    times, soft, hard = _pivot_soft_hard(_relabel_goes_as_synth(samples))
    if not times:
        # GOES has no hard band; build a soft-only stream from the long channel.
        soft = {s.t: s.value for s in samples if s.stream.endswith("-long")}
        hard = {}
        times = sorted(soft)
    soft_flares, hard_flares = _run_detection(times, soft, hard, 60.0)
    events = _build_catalogue(soft_flares, hard_flares)
    for ev in events:
        store.put_event(ev)
    if soft_flares:
        last = soft_flares[-1]
        store.put_alert(
            {
                "kind": "nowcast",
                "stream": "goes-primary-long",
                "goes_class": last.goes_class,
                "severity": (last.goes_class or "")[:1].upper() if last.goes_class else None,
                "onset_time": last.onset_time,
                "t": last.peak_time,
            },
            stream="goes-primary-long",
        )
    print(f"detections: soft={len(soft_flares)} hard={len(hard_flares)}  events={len(events)}")
    print(f"store: streams={len(store.streams())} events={store.health()['n_events']}")
    if args.db:
        print(f"catalogue persisted to {args.db}")
    store.close()
    return 0


def _relabel_goes_as_synth(samples: list) -> list:
    """Map GOES long/short streams to the synth stream ids so the pivot works."""
    from ..synth.generator import STREAM_SXR_LONG, STREAM_SXR_SHORT

    out: list = []
    for s in samples:
        if s.stream.endswith("-long"):
            out.append(_with_stream(s, STREAM_SXR_LONG))
        elif s.stream.endswith("-short"):
            out.append(_with_stream(s, STREAM_SXR_SHORT))
    return out


def _with_stream(s, stream: str):
    from ..types import FluxSample

    return FluxSample(
        stream=stream, t=s.t, value=s.value, unit=s.unit, source=s.source,
        quantity=s.quantity, cls=s.cls, qc=s.qc, meta=s.meta,
    )


def _cmd_forecast(args: argparse.Namespace) -> int:
    """Build labels -> CV -> train/eval GBT + baselines -> metrics + LT-vs-FAR."""
    from ..synth.generator import generate_flare_lightcurves

    samples, truth = generate_flare_lightcurves(
        duration_s=args.hours * 3600.0, cadence_s=args.cadence, seed=args.seed
    )
    times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, args.cadence)
    events = _build_catalogue(soft_flares, hard_flares)
    fs = _offline_forecast(
        samples, events, truth, times, soft, hard,
        horizon_min=args.horizon, class_threshold=args.threshold, seed=args.seed,
    )
    print("=" * 70)
    print("Aditya FlareCast - forecast training + evaluation (offline)")
    print("=" * 70)
    res = OfflinePipelineResult(
        samples=samples, truth=truth, soft_flares=soft_flares,
        hard_flares=hard_flares, events=events, match_rows=[],
        forecast_summary=fs, out_dir="", sqlite_path="", json_path="",
    )
    # Reuse the demo's forecast block by extracting just that section.
    _print_forecast_only(res)
    return 0


def _print_forecast_only(res: OfflinePipelineResult) -> None:
    fs = res.forecast_summary
    print(
        f"horizon: {fs.get('horizon_min')} min   >= class {fs.get('class_threshold')}   "
        f"labelled: {fs.get('n_samples')} (pos={fs.get('n_pos')})"
    )
    if "note" in fs:
        print(f"note: {fs['note']}")
    if "model" in fs:
        m = fs["model"]
        print(
            f"model ({m['backend']}): TSS={m['tss']:+.3f} HSS={m['hss']:+.3f} "
            f"BSS={m['bss']:+.3f} POD={m['pod']:.3f} FAR={m['far']:.3f} "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}"
        )
    if "model_error" in fs:
        print(f"model: not trained ({fs['model_error']})")
    if "baselines" in fs:
        b = fs["baselines"]
        print(
            f"baselines: climatology TSS={b['climatology_tss']:+.3f}  "
            f"persistence TSS={b['persistence_tss']:+.3f}"
        )
    if "lead_time" in fs:
        lt = fs["lead_time"]
        mlt = lt["median_lt_min"]
        mlt_s = f"{mlt:.1f} min" if not math.isnan(mlt) else "n/a"
        print(
            f"lead time: median={mlt_s}  TPR={lt['tpr']:.2f}  "
            f"hits={lt['n_hit']} miss={lt['n_miss']}"
        )
    if "lt_vs_far" in fs and fs["lt_vs_far"]:
        print("LT-vs-FAR (theta -> median_lt[min], tpr, far):")
        for r in fs["lt_vs_far"]:
            mlt = r.get("median_lt")
            mlt_min = (mlt / 60.0) if (mlt is not None and not math.isnan(mlt)) else None
            print(
                f"  theta={r['theta']:.2f}  LT={_fmt(mlt_min, '.1f')}m  "
                f"TPR={r['tpr']:.2f}  FAR={r['far']:.2f}"
            )


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Run the evaluation battery + lead-time buckets on synthetic data."""
    from ..synth.generator import generate_flare_lightcurves

    samples, truth = generate_flare_lightcurves(
        duration_s=args.hours * 3600.0, cadence_s=args.cadence, seed=args.seed
    )
    times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, args.cadence)
    events = _build_catalogue(soft_flares, hard_flares)
    fs = _offline_forecast(
        samples, events, truth, times, soft, hard,
        horizon_min=args.horizon, class_threshold=args.threshold, seed=args.seed,
    )

    print("=" * 70)
    print("Aditya FlareCast - backtest / evaluation battery (offline)")
    print("=" * 70)
    # Detection POD vs truth.
    rows = _match_detected_to_truth(events, truth)
    n_truth = sum(1 for r in rows if r.get("truth_class") is not None)
    n_match = sum(1 for r in rows if r.get("matched") and r.get("truth_class") is not None)
    n_fa = sum(1 for r in rows if r.get("false_alarm"))
    pod = (n_match / n_truth) if n_truth else 0.0
    far_det = (n_fa / (n_fa + n_match)) if (n_fa + n_match) else 0.0
    print(f"DETECTION  POD={pod:.3f} ({n_match}/{n_truth})  false alarms={n_fa}  FAR={far_det:.3f}")
    print()
    print("FORECAST")
    _print_forecast_only(
        OfflinePipelineResult(
            samples=samples, truth=truth, soft_flares=soft_flares,
            hard_flares=hard_flares, events=events, match_rows=rows,
            forecast_summary=fs, out_dir="", sqlite_path="", json_path="",
        )
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch uvicorn with create_app + the dashboard on synthetic data."""
    try:
        import uvicorn
    except ImportError:
        print(
            "serve requires uvicorn + fastapi (pip install fastapi uvicorn).",
            file=sys.stderr,
        )
        return 2
    from ..api.app import create_app
    from ..api.store import ReadStore

    store = ReadStore()
    if not args.empty:
        _seed_store_synth(store, hours=args.hours, cadence_s=args.cadence, seed=args.seed)
    app = create_app(store)
    print(f"Serving Aditya FlareCast on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print(f"  dashboard: http://{args.host}:{args.port}/")
    print(f"  API docs:  http://{args.host}:{args.port}/docs")
    print(f"  seeded: {store.health()['n_streams']} streams, {store.health()['n_events']} events")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _seed_store_synth(store, *, hours: float, cadence_s: float, seed: int | None) -> None:
    """Seed a ReadStore with one synthetic run (latest samples + catalogue)."""
    from ..synth.generator import generate_flare_lightcurves

    samples, truth = generate_flare_lightcurves(
        duration_s=hours * 3600.0, cadence_s=cadence_s, seed=seed
    )
    for s in samples:
        store.put_sample(s)
    times, soft, hard = _pivot_soft_hard(samples)
    soft_flares, hard_flares = _run_detection(times, soft, hard, cadence_s)
    events = _build_catalogue(soft_flares, hard_flares)
    for ev in events:
        store.put_event(ev)
    if soft_flares:
        last = soft_flares[-1]
        from ..api.store import ReadStore

        det = _soft_detection_state(last)
        det.peak = True
        store.put_alert(ReadStore.alert_from_detection(det, "solexs-sxr-long"),
                        stream="solexs-sxr-long")
    # A simple forecast record so /api/forecast is populated.
    store.put_forecast({
        "t_issued": times[-1] if times else 0.0,
        "stream": "fused",
        "horizon_min": 30,
        "class_threshold": "C",
        "p_flare": 0.42,
        "model": "persistence-baseline",
        "p_curve": {"5": 0.05, "15": 0.22, "30": 0.42, "60": 0.58, "120": 0.71},
        "data_quality": 0.9,
    })


# ===========================================================================
# File loading (detect on a provided file).
# ===========================================================================
def _load_lightcurve_file(path: str) -> tuple[list[float], dict[float, float], dict[float, float]]:
    """Load a light curve from a CSV or JSON file into (times, soft, hard).

    Accepts a JSON array of ``{t, soft|sxr, hard|hxr}`` objects or a CSV with a
    header containing ``t`` and ``soft``/``sxr`` (and optional ``hard``/``hxr``).
    """
    soft: dict[float, float] = {}
    hard: dict[float, float] = {}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data:
            t = float(row["t"])
            soft[t] = float(row.get("soft", row.get("sxr", row.get("sxr_long", 1e-8))))
            hv = row.get("hard", row.get("hxr", row.get("hxr_8_30")))
            if hv is not None:
                hard[t] = float(hv)
    else:
        import csv

        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                t = float(row["t"])
                sv = row.get("soft") or row.get("sxr") or row.get("sxr_long")
                soft[t] = float(sv) if sv else 1e-8
                hv = row.get("hard") or row.get("hxr") or row.get("hxr_8_30")
                if hv:
                    hard[t] = float(hv)
    times = sorted(set(soft) | set(hard))
    return times, soft, hard


# ===========================================================================
# Argument parser + dispatch.
# ===========================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flarecast",
        description=(
            "Aditya FlareCast - soft + hard X-ray solar-flare nowcasting & "
            "forecasting CLI (ISRO BAH 2026, PS-15)."
        ),
    )
    sub = p.add_subparsers(dest="command", metavar="{demo,ingest,detect,forecast,serve,backtest}")

    # common synthetic-run options helper.
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--hours", type=float, default=6.0, help="synthetic duration (h)")
        sp.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
        sp.add_argument("--seed", type=int, default=42, help="RNG seed (determinism)")

    # demo (a full synthetic day so the table + forecast are substantive)
    sp = sub.add_parser("demo", help="end-to-end offline demo (the acceptance test)")
    sp.add_argument("--hours", type=float, default=24.0, help="synthetic duration (h)")
    sp.add_argument("--cadence", type=float, default=60.0, help="cadence (s)")
    sp.add_argument("--seed", type=int, default=42, help="RNG seed (determinism)")
    sp.add_argument("--flares", type=int, default=8, help="flare count (default 8)")
    sp.add_argument("--horizon", type=float, default=30.0, help="forecast horizon (min)")
    sp.add_argument("--threshold", default=FORECAST_DEFAULT_CLASS_THRESHOLD, help="min class")
    sp.add_argument("--out", default=None, help="output dir for catalogue (default temp)")
    sp.add_argument("--no-forecast", action="store_true", help="skip the forecast block")
    sp.set_defaults(func=_cmd_demo)

    # ingest
    sp = sub.add_parser("ingest", help="pull GOES (live->cache->synth), detect, populate store")
    sp.add_argument("--offline", action="store_true", help="skip the live tier (cache->synth)")
    sp.add_argument("--db", default=None, help="SQLite path to persist the catalogue")
    sp.set_defaults(func=_do_ingest)

    # detect
    sp = sub.add_parser("detect", help="run dual-band detection (synthetic or a file)")
    _add_common(sp)
    sp.add_argument("--file", default=None, help="light-curve CSV/JSON (default: synthetic)")
    sp.set_defaults(func=_cmd_detect)

    # forecast
    sp = sub.add_parser("forecast", help="labels -> CV -> train/eval GBT + baselines")
    _add_common(sp)
    sp.add_argument("--horizon", type=float, default=30.0, help="forecast horizon (min)")
    sp.add_argument("--threshold", default=FORECAST_DEFAULT_CLASS_THRESHOLD, help="min class")
    sp.set_defaults(func=_cmd_forecast)

    # serve
    sp = sub.add_parser("serve", help="launch FastAPI + dashboard (offline synth data)")
    _add_common(sp)
    sp.add_argument("--host", default="127.0.0.1", help="bind host")
    sp.add_argument("--port", type=int, default=8000, help="bind port")
    sp.add_argument("--empty", action="store_true", help="do not seed synthetic data")
    sp.set_defaults(func=_cmd_serve)

    # backtest
    sp = sub.add_parser("backtest", help="evaluation battery: TSS/HSS/BSS/POD/FAR + lead time")
    _add_common(sp)
    sp.add_argument("--horizon", type=float, default=30.0, help="forecast horizon (min)")
    sp.add_argument("--threshold", default=FORECAST_DEFAULT_CLASS_THRESHOLD, help="min class")
    sp.set_defaults(func=_cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    """``flarecast`` entry point. Returns a process exit code (Appendix B.7)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


def run_pipeline(*args: Any, **kwargs: Any) -> OfflinePipelineResult:
    """Programmatic end-to-end pipeline (used by :func:`flarecast.run_pipeline`).

    A thin wrapper over :func:`run_offline_pipeline` so
    ``flarecast.run_pipeline()`` runs the full synth -> detect -> catalogue ->
    forecast flow and returns the :class:`OfflinePipelineResult`.
    """
    return run_offline_pipeline(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
