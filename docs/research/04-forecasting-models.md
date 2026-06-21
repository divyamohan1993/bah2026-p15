# 04 — Forecasting Models for Solar-Flare Prediction (Aditya-L1 SoLEXS + HEL1OS)

**ISRO BAH 2026 — Problem 15 (Forecasting track).**
Predict the **probability of a flare in the next N minutes** by detecting **precursor patterns** in combined soft (SoLEXS) and hard (HEL1OS) X-ray light curves **before the flare peak**, and emit a quantifiable **lead time**. Scored on **high True Positive Rate, low False Alarm Rate, and lead time (minutes before peak)**.

> Companion docs: `01`-`03` (instruments / nowcasting / data) assumed to cover ingestion and the nowcasting catalogue. This file specifies the **forecasting** subsystem only: features, model families, problem framing, labels/CV, lead-time quantification, metrics, edge inference, and benchmarks.

---

## 0. TL;DR — Recommended Two-Tier Design

**Tier 1 — PRODUCTION (fast, edge, O(window) per step):**
**Gradient-Boosted Trees (LightGBM/XGBoost)** on a ~30-dim **streaming feature vector** computed from a sliding window of SoLEXS + HEL1OS light curves, framed as **binary "flare ≥ class C within next N minutes?"** sliding-window classification, with **N as an explicit input** (or a small bank of per-N models for N ∈ {15, 30, 60} min). Exported to a compact format; inference is a handful of tree traversals per cadence step → microseconds, runnable on edge / Cloudflare Workers AI / on-board-class hardware. **Probability calibration (isotonic / Platt)** bolted on so the output is a *trustworthy probability*, not just a score.

**Tier 2 — RESEARCH (heavier, offline-trained, optional online):**
A small **Temporal Convolutional Network (TCN)** (dilated causal 1-D convs) or **GRU** on the *raw* multi-channel windowed light curve, predicting a **multi-horizon probability curve** P(flare within h) for h = 5…120 min. Exportable to **ONNX** for fast inference; still O(window) per step. This captures shapes the hand-engineered features miss (e.g. quasi-periodic pre-flare pulsations, subtle Neupert build-up) and is the accuracy ceiling.

**Why this split:** GBT on engineered features is the proven, interpretable, data-efficient baseline that *usually wins or ties* on tabular space-weather features and ships to the edge trivially; the TCN/GRU is the upside model and a strong ensemble partner. Train both; **deploy GBT, keep TCN as challenger + offline re-scorer + ensemble member.**

**Headline benchmark to beat / contextualize:** Landa & Reuveni (2022) achieved **TSS ≈ 0.74, recall ≈ 0.95** forecasting ≥M flares from **GOES X-ray time-series alone** with a 1-D CNN across 1–96 h horizons — the closest published analog to our light-curve-only setting. But note the *humbling* result: the 2025 NOAA SWPC verification (1998–2024) found the **operational forecast did not beat simple persistence/climatology** on several metrics, with persistence giving the best 24 h CSI (0.12) / HSS (0.19). **Always report skill vs persistence and climatology**, never just raw TSS.

---

## 1. Physical Basis — What a Flare "Tells Us" Before It Peaks

A flare's GOES/SoLEXS soft X-ray (SXR) light curve has three phases: **pre-flare** (subtle enhancement), **rising/impulsive** (onset→peak), **post-flare** (decay). Forecasting lives in the pre-flare and *very early* rising phase. The exploitable precursor physics:

| Precursor signature | Physics | Where it shows up | Feature it motivates |
|---|---|---|---|
| **Pre-flare gradual SXR enhancement** | Slow coronal heating / reconnection onset before impulsive energy release | SoLEXS 1–8 keV slowly rising; often visible first in the **time-derivative** | `sxr_slope`, `sxr_curvature`, `ema_ratio` |
| **HXR microflares / precursor bursts** | Small, localized non-thermal energy releases ("sparks") preceding the main event | HEL1OS ≥ ~8–20 keV transient bursts | `hxr_burst_count`, `hxr_var_burst`, `hxr_peak_over_baseline` |
| **Neupert effect** (HXR ≈ d/dt SXR) | HXR ∝ non-thermal electron injection; SXR is its time-integral (chromospheric evaporation). ~80% of large flares show HXR tracking d(SXR)/dt | HEL1OS rising *while* d(SoLEXS)/dt rises | `neupert_resid`, `corr(HXR, dSXR/dt)`, leading-indicator flag |
| **Temperature / emission-measure (T, EM) rise** | Plasma heats and brightens before peak; T often *leads* the flux peak | SoLEXS spectral ratio (hot vs cool band) | `temp_proxy_ratio`, `d(temp_proxy)/dt` |
| **Spectral hardening** | Spectrum hardens as non-thermal electrons appear | HEL1OS/SoLEXS **hardness ratio** rising | `hardness_ratio`, `d(hardness)/dt` |
| **Rise-time slope / sharpening** | Impulsive onset = fast positive slope + curvature | Both channels | `sxr_slope_short`, `slope_accel` |
| **QPP / very-long-period pulsations** | Pre-flare oscillations reported in SXR/Hα/microwave before onset | Periodicity in residual light curve | `spectral_power_band`, `dominant_period` (research/TCN) |
| **Self-excitation / clustering** | Flares cluster in time; one flare raises P(next) (Hawkes-like) | Event history | `time_since_last_flare`, `flares_last_6h`, decayed history |
| **Magnetic complexity (supplementary)** | Flare productivity correlates with SHARP δ-spot complexity | SDO/HMI SHARP (not on Aditya-L1) | optional `sharp_*` features when available |

**Key insight for this problem:** the **Neupert effect makes HXR a *leading* indicator of the SXR peak** — HEL1OS hard X-rays rise during the impulsive phase *while soft X-rays are still climbing toward peak*. Combined SXR+HXR is therefore strictly more predictive than SXR alone for short lead times, which is exactly why the problem mandates fusing SoLEXS + HEL1OS. The derivative-of-hardness and the **Neupert residual** are our highest-value engineered features.

**Instrument facts that shape feature design** (confirm against doc `01`):
- **SoLEXS:** ~1–30 keV (continuous Sun-as-a-star 2–22 keV since 6 Jan 2024), ~170–250 eV resolution, **1 s cadence during flares**; two detectors (large aperture → small flares, small aperture → large flares, avoiding pile-up/saturation).
- **HEL1OS:** **8–150 keV** (CdTe 8–70, CZT 20–150), time-resolved flare spectra, sub-second/second cadence.
- Both are **full-disk / Sun-as-a-star** (no spatial info), so we forecast *whole-Sun* flare probability, not per-active-region — simpler labels, but no SHARP-style spatial features unless we pull supplementary SDO/HMI.

---

## 2. Engineered Feature Vector (Streaming, O(window))

All features are computable incrementally on a sliding window so inference is **O(window) — or O(1)** with running EMAs/Welford variance. Cadence assumed ≈ 1 s; aggregate to a working step (e.g. 10 s or 60 s) to denoise and bound cost. Use **two raw channels**: `S` = SoLEXS soft flux (e.g. 1–8 keV band), `H` = HEL1OS hard flux (e.g. 8–20 keV band, plus a higher band 20–50 keV if S/N allows). Work in **log-flux** for the level features (X-ray flux is log-distributed across flare classes).

### 2.1 Core feature list (~30 dims) — the production GBT input

**Levels & baselines**
1. `logS` — current log soft flux (EMA-smoothed, short).
2. `logH` — current log hard flux (EMA-smoothed, short).
3. `S_over_baseline` — S / rolling-median baseline (quiet-Sun normalized).
4. `H_over_baseline` — H / rolling-median baseline.

**Slopes / derivatives (the precursor signal)**
5. `sxr_slope_short` — d(logS)/dt over short window (e.g. 2 min).
6. `sxr_slope_long` — d(logS)/dt over long window (e.g. 15 min).
7. `sxr_curvature` — 2nd derivative of logS (onset sharpening).
8. `hxr_slope_short` — d(logH)/dt short.
9. `slope_accel` — `sxr_slope_short − sxr_slope_long` (is the rise *accelerating*?).

**EMA ratios (multi-scale trend, O(1) to maintain)**
10. `ema_ratio_S` — EMA_short(S) / EMA_long(S) (>1 ⇒ rising).
11. `ema_ratio_H` — EMA_short(H) / EMA_long(H).
12. `ema_ratio_cross` — EMA_short(S)/EMA_long(H) (cross-channel divergence).

**Hardness & spectral evolution**
13. `hardness_ratio` — H / S (or high-band/low-band within SoLEXS).
14. `d_hardness` — d(hardness)/dt (spectral hardening = non-thermal onset).
15. `temp_proxy` — SoLEXS hot-band / cool-band ratio (T proxy).
16. `d_temp_proxy` — its derivative.

**Neupert-effect features (highest value)**
17. `neupert_corr` — rolling Pearson corr( H , d(S)/dt ) over window.
18. `neupert_resid` — || H − α·d(S)/dt || (deviation magnitude; spikes at impulsive onset).
19. `hxr_leads_flag` — H rising while d(S)/dt rising and S below recent local max (leading-indicator boolean).

**Variance / burst detection (microflares)**
20. `sxr_var` — rolling variance of logS (Welford, O(1)).
21. `hxr_var` — rolling variance of logH.
22. `hxr_burst_count` — # of >kσ excursions in H over window (precursor microflares).
23. `hxr_peak_over_baseline` — max(H)/baseline in window.

**Event history / self-excitation (Hawkes-flavored)**
24. `time_since_last_flare` — minutes since last nowcasted flare onset (capped).
25. `flares_last_6h` — count of recent flares.
26. `decayed_flare_history` — Σ exp(−Δt/τ) over recent flares (τ≈6 h) — a 1-line Hawkes-style self-excitation feature.
27. `time_since_last_microflare` — from HXR burst detector.

**Context / state**
28. `quiet_duration` — how long since flux last near quiet baseline (long quiet → first-flare regime, where models notoriously fail).
29. `solar_cycle_phase` — slow background rate proxy (date → cycle phase; optional, weak).
30. `N_horizon` — **the forecast horizon N in minutes**, passed as a feature so one model serves multiple N (or omit and train a per-N bank).

> **Practical notes.** Use **causal** windows only (no future samples → no leakage). Maintain EMAs/variance incrementally for O(1). Robust-scale or rank-transform features per-channel; tree models are scale-robust but calibration benefits from stable inputs. Handle data gaps explicitly with a `data_quality`/`gap_flag` companion feature and hold-last-value with decay. Detrend the slow quiet-Sun diurnal/orbital variation via the rolling baseline so slopes reflect flare dynamics, not housekeeping drift.

### 2.2 Raw input for the TCN/GRU (Tier 2)
Multichannel tensor of shape `[L, C]` with `L` = window length in steps (e.g. 120 steps × 1 min = 2 h look-back) and `C` ≈ 4–8 channels: `logS`, `logH`, `hardness`, `d(logS)/dt`, `d(logH)/dt`, plus optionally 1–2 SoLEXS sub-band fluxes. The network learns its own slopes/periodicities; we still feed a few derived channels to ease learning of the Neupert relation.

---

## 3. Model Families — Comparison

Legend: **Latency** = per-step inference cost; **Data hunger** = labeled-event volume needed; **Interp.** = interpretability.

### 3.1 Classical (engineered features) — *recommended for production*

| Model | Pros | Cons | Latency | Data hunger | Interp. |
|---|---|---|---|---|---|
| **Logistic Regression** | Trivial, well-calibrated, transparent coefficients, ε-cost edge inference | Linear; misses interactions unless hand-crafted | O(d) | Low | High |
| **GBT (LightGBM / XGBoost)** | **Best tabular accuracy**, captures interactions/non-linearity, handles missing values, fast, exportable, monotone constraints, SHAP explanations | Needs good features; can overfit rare class without care | O(trees·depth) ≈ µs | **Low–Med** | High (SHAP) |
| **Random Forest / Extra-Trees** | Robust, low-tuning baseline | Larger model, weaker than GBT typically | O(trees·depth) | Low–Med | Med |

**Verdict:** GBT is the workhorse. Precedent: Nishizuka et al. (2017) k-NN/ML and many SHARP studies hit **TSS 0.6–0.9** with engineered features; XGBoost on topological time-series features is an active 2025 line (ApJ). GBT is the right Tier-1 because flare datasets are small and tabular features encode the physics directly.

### 3.2 Sequence / deep models (raw or lightly-engineered series)

| Model | Pros | Cons | Latency | Data hunger |
|---|---|---|---|---|
| **1-D CNN** | Captures local shapes (onset, bursts); cheap; **proven on GOES X-ray (Landa & Reuveni 2022, TSS 0.74)** | Limited long context (fixed receptive field) | O(window) | Med |
| **TCN (dilated causal conv)** | **Long receptive field via dilation, strictly causal (no leakage), parallel training, stable gradients, often beats LSTM**; small + ONNX-friendly | Receptive field fixed by depth/dilation; less "online-stateful" than RNN | O(window) | Med |
| **LSTM / GRU** | Natural streaming/stateful O(1)-per-step recurrence; strong precedent (Liu et al. 2019 attention-LSTM **TSS≈0.88** for ≥M5 @24h) | Slower training, vanishing-gradient on long seqs, less parallel | O(1) per step (stateful) | Med–High |
| **ConvLSTM** | Spatiotemporal — useful *only* if we add SDO image cubes | Heavy; overkill for Sun-as-a-star 1-D series | High | High |
| **Transformers (PatchTST / Informer / TimesNet)** | SOTA on long-horizon TS; PatchTST patching is efficient; SolarFlareNet transformer **TSS>0.83 @24h** | **Data-hungry**, heavier latency, harder to calibrate/explain, overkill for short look-backs | O(L²) or O(L·patch) | **High** |

**Verdict:** For Tier 2 choose **TCN** (causality + long context + small + ONNX export + strong TS track record) or **GRU** (if you want true stateful streaming). Reserve Transformers for an offline accuracy experiment only — their data hunger fights our small event count.

### 3.3 Probabilistic / forecast-native

| Approach | Idea | Fit to this problem | Caveat |
|---|---|---|---|
| **Survival / time-to-event (discrete-time hazard, DeepHit/PCHazard, Cox/DeepSurv)** | Model hazard h(t) = P(flare in [t, t+Δ) \| none yet); integrate → P(flare within N) and **expected time-to-flare** | **Excellent native framing** — directly yields a multi-horizon probability curve *and* lead-time, handles censoring (windows with no flare yet) properly | Needs careful discrete-time setup; less off-the-shelf tooling for streaming |
| **Hawkes / self-exciting point process** | λ(t) = µ(t) + Σ φ(t−tᵢ); each flare excites future intensity; background µ tracks solar-cycle | Flares **are** self-exciting/clustered (documented); gives a principled `decayed_flare_history` and an event-rate baseline to beat | Pure Hawkes ignores the rich X-ray covariates; best as a **baseline + a feature**, not the main model |
| **Bayesian (e.g. Bayesian LR / GP / BNN)** | Posterior predictive probabilities + uncertainty | Useful for **calibrated uncertainty** and small-data regularization | Compute-heavier; GP scales poorly to long series |

**Verdict:** Use the **discrete-time hazard** formulation as the *unifying probabilistic frame* (it makes "P(flare in next N)" and "lead time" the same model), and use **Hawkes** as (a) the self-excitation feature `decayed_flare_history` and (b) a **point-process baseline** in the skill comparison. Bayesian LR is a nice calibrated, cheap baseline.

### 3.4 Hybrid — *recommended overall*
**GBT (engineered features, edge) + TCN/GRU (raw series, offline/ONNX) → soft-voting / stacked ensemble**, with the production decision defaulting to GBT and the deep model as challenger + offline re-scorer. This is the standard "fast edge inference + heavy offline model" pattern and matches the operational ensemble practice in flare forecasting (NOAA/CCMC Flare Scoreboard aggregates many models; ensembles routinely top single models).

---

## 4. Problem Formulation

Three complementary framings; **(a) is primary for production, (b) is the deliverable's "probability curve", (c) is the elegant unifier.**

### (a) Binary sliding-window classification — *primary*
At each step *t*, using only data ≤ *t*, predict **y = 1 if a flare (peak) of class ≥ threshold occurs in (t, t+N]**, else 0. Train per N or with N as a feature.
- *Pros:* simplest, best-understood, directly optimizes TPR/FAR, trivially calibratable.
- *Threshold class:* default ≥ **C-class** (Sun-as-a-star sees many C; lets us learn precursors and get statistics), with a **≥M** variant for the high-impact operational alert. Report both.

### (b) Multi-horizon probability curve
Emit **P(flare within h)** for a vector of horizons h = {5, 10, 15, 30, 60, 120} min — a monotone-increasing curve per timestep. Either a multi-output head (TCN/GRU) or the hazard model integrated to each h. This is exactly the "probability of a flare in the next N minutes" deliverable, generalized over N, and feeds the **lead-time** computation (§5).

### (c) Time-to-event regression / discrete-time hazard
Predict the hazard per future bin; derive P(flare within N) = 1 − Π(1 − hazard_k) and **expected/median time-to-next-flare**. Handles **right-censoring** (a window that simply hasn't flared yet) correctly — a subtlety plain classification ignores.

### Labels — reference catalogue & definition
- **Ground truth = flare peaks** from a reference catalogue. Primary: **our own SoLEXS/HEL1OS nowcasting master catalogue** (from doc `02`, the problem's nowcasting deliverable) so labels and inputs are the *same instrument*. **Cross-validate against GOES/HEK** (NOAA/`SunPy` Fido → HEK) and the **GOES XRS event list** for class labels and to fill gaps / sanity-check timing.
- For each flare peak at time *p* of class *c*, a window ending at *t* is **positive for horizon N** iff `0 < p − t ≤ N` **and** `t < onset` is *not* required for classification (we want to fire any time before peak) — but for **strict pre-peak forecasting**, restrict positives to `t < p` (we predict before the peak, never after). Use the **peak**, not onset, as the reference because lead-time is defined relative to peak.
- **Mask the in-flare/decay samples** from the *negative* class (a sample mid-decay is neither a clean precursor nor a true negative) — label them "in-event" and exclude or handle separately to avoid confusing the precursor signal.
- Class threshold(s): build label sets for **≥C** and **≥M** (and optionally ≥B for abundance).

### Cross-validation — NO leakage
- **Temporal block splits only.** Never random-shuffle samples — adjacent windows overlap and one flare's pre-window leaks across splits (documented leakage failure mode). Split by **time blocks** (e.g. train 2022–2024, validate 2024-H2, test 2025), or **rolling-origin / blocked k-fold** with **embargo/purge gaps** ≥ N (plus the max window length) between train and test so no training window peeks into a test flare.
- **Group by flare event** where possible: all windows belonging to one flare go to the *same* fold.
- **Walk-forward (rolling-origin) evaluation** mirrors operations: train on past, test on the immediately following block, slide forward; report mean ± spread across folds.
- Keep a **final untouched hold-out** (most recent contiguous period) for the headline numbers.

---

## 5. Lead-Time Quantification

**Definition:** for each true flare with peak time *p*, the lead time is
`LT = p − t_alert`, where `t_alert` = first time the forecast probability **crosses threshold θ** within a pre-peak detection window `[p − W, p)` (W = max look-ahead, e.g. 120 min) **and stays / is confirmed** (require k-of-m consecutive crossings to suppress flicker). If the probability never crosses θ before *p* → **missed forecast** (counts against TPR), LT undefined.

**Reporting:**
- **Distribution of LT** over all true positives: median, IQR, and the fraction with LT ≥ {5, 10, 15, 30} min (operationally meaningful cushions). Median lead time is the headline lead-time number.
- **Lead-time vs false-alarm trade-off curve:** sweep θ. Lower θ ⇒ earlier alerts (larger LT) but more false alarms (higher FAR); higher θ ⇒ later, more precise alerts. Plot **median LT (and TPR) vs FAR** across θ — this is *the* operating-point selection plot for the judges, and directly addresses the three evaluation axes simultaneously.
- **Per-class LT:** report separately for ≥C and ≥M (bigger flares often give clearer/earlier precursors via stronger Neupert build-up).
- **Honesty check:** verify alerts fire **before peak** (LT>0) — a "forecast" that triggers at/after peak is really a nowcast. Strict pre-peak labeling (§4) enforces this.

---

## 6. Evaluation Metrics for Rare Events

Flares are **rare** → accuracy/ROC-AUC are misleading (dominated by the many true negatives). Use the flare-forecasting standard battery:

| Metric | Definition / why | Notes |
|---|---|---|
| **TSS** (True Skill Statistic; Hanssen–Kuipers) = **TPR − FPR** = POD − POFD | **The flare-forecasting standard** (Bloomfield+2012 recommends it). Range [−1,1]; **insensitive to class ratio** → fair under imbalance | **Primary metric.** Targets: ≥M TSS ~0.5–0.8 is competitive; Landa & Reuveni X-ray-only ≈0.74 |
| **HSS** (Heidke Skill Score) | Skill vs random forecast with same marginals | Report alongside TSS; *is* ratio-sensitive |
| **BSS** (Brier Skill Score) = 1 − BS/BS_climatology | **Probabilistic** skill vs **climatology**; decomposes into **reliability + resolution − uncertainty** | Use for the *probability* output; BSS>0 means beats climatology |
| **ROC-AUC** | Threshold-free discrimination | Report but **don't headline** (optimistic under imbalance) |
| **PR-AUC** (precision–recall AUC) | Discrimination focused on the **positive (flare)** class | **More informative than ROC under heavy imbalance** |
| **Reliability / calibration diagram + ECE** | Do predicted probabilities match observed frequencies? | Required since deliverable is a *probability*; pair with isotonic/Platt calibration (cf. **DeFN-R**, the "reliable" calibrated DeFN) |
| **POD vs FAR** (and FAR/CSI) | Operational detection vs false-alarm | POD=TPR; FAR=FP/(FP+TP). The problem's TPR/FAR axes |
| **Confusion matrix @ operating θ** | Raw TP/FP/FN/TN | Pick θ to maximize TSS *or* hit a FAR budget |

**Baselines to beat (mandatory):**
- **Climatology** — constant P = base rate of "flare within N" in the training period.
- **Persistence** — "flaring now ⇒ predict flare" (and its decayed variant). *The 2025 SWPC study shows persistence is shockingly hard to beat at 24 h* — include it or numbers are not credible.
- **Hawkes / Poisson event-rate** model — point-process baseline using only flare history.

**Class-imbalance handling (flares rare):**
- **Loss-based (preferred for time series):** **focal loss** (down-weights easy negatives; classification-calibrated) and/or **class weights / `scale_pos_weight`** in GBT. Loss-based rebalancing avoids the temporal artefacts/leakage that oversampling introduces in sequences.
- **Sampling (use cautiously):** undersample negatives or time-aware oversampling of positive windows (e.g. SMOTE-for-TS / OSTSC) — only within training folds, never across the temporal split.
- **Threshold moving + calibration:** train on (re)weighted data, then **calibrate probabilities** and choose θ on the validation fold by the TSS-max or FAR-budget rule.
- Optimize/early-stop on **TSS or PR-AUC**, never plain accuracy.

---

## 7. Edge / O(1) Inference

The production model must be **O(1) or O(window)** per cadence step (1 s during flares).

- **GBT (Tier 1):** inference = sum over a few hundred shallow trees → **microseconds, tiny memory**. Export via the model's native booster, **ONNX** (`onnxmltools`), or hand-rolled tree arrays; runs anywhere incl. constrained / **Cloudflare Workers AI** / on-board-class CPUs. Feature computation is **O(1)** with running EMAs + Welford variance + ring-buffer for window stats. This is the recommended deployed forecaster.
- **Small TCN (Tier 2, optional online):** few conv layers, small channel count → **O(window)** per step, **ONNX-exportable**, still edge-feasible. GRU gives literal **O(1) per step** stateful inference if preferred.
- **Heavy ↔ light separation:** **TRAINING** (GBT boosting / TCN backprop, hyper-search, calibration fitting) is offline/batch on a workstation/GPU. **INFERENCE** is the light, streaming, O(window) path above. Re-train periodically (e.g. monthly or per Carrington rotation) offline; ship updated tree tables / ONNX to the edge. Keep the deep model's *scores* available offline to re-rank / audit.

---

## 8. Benchmarks & Literature (typical TSS)

| Work | Inputs | Model | Horizon | TSS (≥M unless noted) |
|---|---|---|---|---|
| **Landa & Reuveni (2022)**, ApJS 258:1 | **GOES X-ray time-series only** | 1-D CNN, multi-horizon 1–96 h | 24 h | **≈0.74** (recall ≈0.95) — *closest analog to our task* |
| **Liu et al. (2019)** | 25 SHARP + 15 flare-history | attention-LSTM | 24 h | **≈0.79–0.88** (≥M / ≥M5) |
| **Nishizuka et al. (2017)** | 79 features | k-NN / ML | 24 h | up to **~0.9** (≥M) |
| **DeFN — Nishizuka et al. (2018)** | magnetogram + UV + SXR/EUV images | deep NN | 24 h | **0.8** (≥M), **0.63** (≥C) |
| **DeFN operational (2019–20)** | same | deep NN | 24 h (6 h cycle) | **0.24** (≥M), **0.70** (≥C) — *operational drop!* |
| **DeFN-R (2020)** | same | calibrated DeFN | 24 h | reliable **probabilities** (BSS/reliability focus) |
| **SolarFlareNet — Abduallah et al. (2023)** | SHARP | Transformer | 24–72 h | **>0.83** (24 h), **>0.7** (48–72 h) |
| **Huang et al. (2018)** | magnetograms + SXR | CNN | 24 h | 0.49–0.71 (C/M/X) |
| **MViT — Li et al. (2025)** | AR patches | vision transformer | 24 h | 0.74 (real-world 2023–24) |
| **NOAA/CCMC Flare Scoreboard (2023–24)** | many | ensemble of ops models | 24 h | **0.77** (≥M) |
| **NOAA SWPC operational (1998–2024 verif.)** | forecaster + lookup | operational | 24 h | **does not beat persistence/climatology** on key metrics; persistence CSI 0.12 / HSS 0.19 |

**Reading of the benchmarks for *our* problem:**
1. **Light-curve-only forecasting is viable** — Landa & Reuveni show TSS≈0.74 from GOES X-rays alone; adding **HEL1OS HXR + Neupert features should help short lead times** beyond what SXR-only can do.
2. **Most literature forecasts at 24 h with magnetograms; we forecast *minutes* from X-rays** — a *different, shorter-horizon* regime where the impulsive-phase precursor physics (Neupert, microflares, slope/curvature) dominates and X-ray cadence (1 s) is an advantage. Expect this to behave more like *imminent-onset* detection than next-day AR forecasting.
3. **Operational performance < lab performance**, and **persistence/climatology are tough baselines** — so our scoring **must** include them, walk-forward CV, and calibration, or the numbers won't be trusted.

---

## 9. Concrete Build Plan (what to implement)

1. **Labels:** build positive/negative window labels for N∈{15,30,60} min and class∈{≥C,≥M} from the SoLEXS/HEL1OS master catalogue, cross-checked vs GOES/HEK; mask in-event samples; strict pre-peak positives.
2. **Features:** implement the ~30-dim streaming vector (§2.1) with O(1) EMAs/Welford/ring-buffer; unit-test causality (no future leakage).
3. **CV:** blocked rolling-origin splits with embargo ≥ N+window; event-grouped; final recent hold-out.
4. **Tier-1 GBT:** LightGBM with `scale_pos_weight`/focal-style weighting, early-stop on TSS; then **isotonic calibration** on a held fold; sweep θ for the **LT-vs-FAR** curve.
5. **Tier-2 TCN:** small dilated causal TCN (≈3–5 residual blocks, dilations 1-2-4-8-16, ~32–64 channels, kernel 3) with multi-horizon sigmoid heads + focal loss; export **ONNX**.
6. **Baselines:** climatology, persistence (+decayed), Hawkes/Poisson — report skill *relative to these*.
7. **Ensemble:** soft-vote GBT+TCN; keep GBT as the deployed/edge model.
8. **Report:** TSS, HSS, BSS, PR-AUC, ROC-AUC, reliability diagram + ECE, POD/FAR, confusion matrix at chosen θ, and the **median lead time + LT-vs-FAR curve** per class — with walk-forward mean ± spread and the hold-out headline.

---

## 10. Sources

- Advances and Challenges in Solar Flare Prediction: A Review (2025) — https://arxiv.org/html/2511.20465v1
- Deep Flare Net (DeFN), Nishizuka et al. (2018) — https://iopscience.iop.org/article/10.3847/1538-4357/aab9a7
- DeFN-R (Reliable probability forecast), Nishizuka et al. (2020) — https://arxiv.org/pdf/2007.02564
- Liu et al. (2019), LSTM flare prediction — https://www.researchgate.net/publication/333598132_Predicting_Solar_Flares_Using_a_Long_Short-term_Memory_Network
- Landa & Reuveni (2022), 1-D CNN on GOES X-ray time-series — https://iopscience.iop.org/article/10.3847/1538-4365/ac37bc
- The soft X-ray Neupert effect as SEP-injection proxy (forecasting) — https://www.swsc-journal.org/articles/swsc/full_html/2020/01/swsc200079/swsc200079.html
- The Neupert Effect of flare UV and SXR emissions — https://arxiv.org/pdf/2101.11069
- Neupert effect statistical analysis (M/X, Cycle 24) — https://link.springer.com/article/10.1007/s11207-026-02643-z
- X-ray precursors to flares and filament eruptions — https://www.aanda.org/articles/aa/pdf/2007/36/aa7771-07.pdf
- Very long-period pulsations before solar flares (precursors) — https://arxiv.org/pdf/1610.09291
- Self-excitation in the solar flare waiting-time distribution (Hawkes) — https://www.sciencedirect.com/science/article/abs/pii/S0378437120303915
- Hawkes Models and Their Applications (review) — https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-112723-034304
- TCN / dilated causal convolution for time series (overview) — https://yanglin1997.github.io/files/TCAN.pdf
- DeepSurv / DeepHit survival (discrete-time hazard) — https://github.com/jaredleekatzman/DeepSurv ; https://proceedings.neurips.cc/paper/2021/file/7f6caf1f0ba788cd7953d817724c2b6e-Paper.pdf
- Focal loss & calibration for imbalance — https://arxiv.org/html/2408.11598v1 ; rare-event TS (EVEREST) https://arxiv.org/pdf/2601.19022
- Brier Skill Score primer — https://www.cwdatasolutions.com/post/a-primer-on-the-brier-skill-score
- A Comparison of Flare Forecasting Methods II (benchmarks/metrics) — https://iopscience.iop.org/article/10.3847/1538-4365/ab2e12
- A Framework for Designing & Evaluating Solar Flare Forecasting Systems — https://arxiv.org/pdf/2005.02493
- Toward Reliable Benchmarking of Solar Flare Forecasting Methods (Barnes/Leka) — https://arxiv.org/pdf/1202.5995
- NOAA SWPC flare forecast verification 1998–2024 (persistence/climatology baselines) — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025SW004546
- SoLEXS ground calibration & in-flight performance (Aditya-L1) — https://arxiv.org/abs/2509.26292
- HEL1OS — Hard X-ray Spectrometer on Aditya-L1 — https://link.springer.com/article/10.1007/s11207-025-02543-8 ; https://arxiv.org/pdf/2512.12679
- The Aditya-L1 mission of ISRO — https://arxiv.org/pdf/2212.13046
