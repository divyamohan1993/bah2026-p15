# 03 — Real-Time Detection & Classification Algorithms for Solar-Flare Nowcasting

**Project:** ISRO BAH 2026 — Problem 15 (NOWCASTING of solar flares from Aditya-L1)
**Instruments:** SoLEXS (soft X-ray, 2–22 keV, 1 s cadence) and HEL1OS (hard X-ray, 8–150 keV)
**Goal:** Detect flares **independently** in each band in real time, then **merge** into one master catalogue.
**Design constraint (from project lead):** "fastest platform, O(1) techniques" → **streaming / online** algorithms with **O(1) work per sample** and **O(1) state** (no batch re-computation over sliding windows).

> Scope note: This document specifies the *detection and classification* layer (algorithms, math, complexity, pseudocode, evaluation). It assumes the ingest layer delivers a clean, timestamped, regularly-sampled stream per band; gap/quality handling is covered in §7. Where network sources were available they are cited; otherwise canonical results from the literature (cutoff Jan 2026) are used.

---

## 0. Notation & the streaming contract

| Symbol | Meaning |
|---|---|
| `x_t` | new sample at time `t` (SoLEXS: flux proxy / count rate; HEL1OS: counts in bin) |
| `Δt` | sampling interval (s). SoLEXS spectral = 1 s; SoLEXS timing = 0.1 s; HEL1OS configurable |
| `μ_t` | running baseline estimate (EMA) |
| `σ_t` | running standard-deviation estimate |
| `b` | baseline/background level (quiet-Sun) |
| `λ` (lambda) | Poisson rate (counts per bin) for count channels |
| `k` | threshold multiplier (in sigmas) |
| `α` | EMA forgetting factor, `0<α<1` (effective window ≈ `1/(1-α)`) |

**The streaming contract every primitive must honour:**
1. `update(x_t)` runs in **O(1)** time.
2. State size is **O(1)** (independent of the number of samples seen).
3. No re-reading of past raw samples (raw history, if any, lives only in a fixed-size ring buffer).

A "window" in this design is *implicit* (the EMA's exponential memory) rather than an explicit array we recompute. That is the heart of the O(1) requirement.

---

## 1. O(1) Streaming Primitives

These are the reusable building blocks. Every detector in §2 is composed from them.

### 1.1 Recursive Exponential Moving Average (EMA) — baseline / trend

The EMA is the canonical O(1) low-pass filter. For forgetting factor `α`:

```
μ_t = α · μ_{t-1} + (1 - α) · x_t
```

- **State:** one float (`μ`). **Update:** one multiply-add. **Complexity:** O(1) time, O(1) space.
- **Effective window** `N_eff ≈ 1/(1-α)`; **time constant** `τ ≈ Δt/(1-α)`.
- For the slowly-varying quiet-Sun **baseline**, pick a *long* `τ` (minutes–tens of minutes). For a fast "signal" tracker (used by derivative/onset logic) pick a *short* `τ` (seconds).
- **Double EMA (DEMA)** removes lag for trend following; **bias-correction** (`μ_t / (1-α^t)`) fixes the cold-start under-estimate in the first `N_eff` samples.

> Welford and EWMA are the standard O(1), single-pass tools for streaming mean/variance ([Welford's online algorithm](http://davidma.me/blog/2025/Welfords-Algo/); [P² algorithm, Jain & Chlamtac 1985](https://www.cse.wustl.edu/~jain/papers/psqr.htm)).

### 1.2 Online variance — Welford and EWMV

**(a) Welford** (numerically stable, *unweighted*, all samples equal):

```
n      += 1
δ       = x_t - μ
μ      += δ / n
δ2      = x_t - μ
M2     += δ · δ2
var     = M2 / (n - 1)        # sample variance
```
O(1) time, O(1) state (`n, μ, M2`). Numerically stable vs. the naïve `E[x²]-E[x]²`.

**(b) Exponentially-Weighted Moving Variance (EWMV)** — *preferred here* because the baseline drifts (solar activity cycle, orbit), so we want recent samples to dominate. The standard recursive pair (West 1979 / incremental EWMA variance):

```
δ        = x_t - μ_{t-1}
μ_t      = μ_{t-1} + (1 - α) · δ
S_t      = α · (S_{t-1} + (1 - α) · δ²)     # EWMV
σ_t      = sqrt(S_t)
```
O(1) time, two floats of state. This σ feeds the adaptive threshold in §2.1.

> **Critical subtlety:** the variance/baseline estimators must be *frozen* (or "leaked" very slowly) **while a flare is in progress**, otherwise the flare contaminates the baseline and the detector "adapts away" the very signal it should catch. See §2.1 (gated update) and §7.

### 1.3 Ring buffer — fixed memory raw history

A circular array of fixed length `L` (e.g., 60–300 samples) gives O(1) push, O(1) access to the last `L` raw values, **O(1) total memory**. Used for: (i) re-confirming an onset, (ii) peak refinement, (iii) the matched-filter tap-delay line (§2.8), (iv) despiking median window (§7).

```
ring[head] = x_t ; head = (head + 1) mod L      # O(1)
```

### 1.4 Incremental slope / derivative — onset & Neupert

**First difference (cheapest):** `d_t = (x_t − x_{t-1}) / Δt`. O(1), but noisy.

**Smoothed derivative (recommended)** — difference of two EMAs / or derivative of a single EMA:
```
d_t = (μ_fast_t − μ_fast_{t-1}) / Δt          # derivative of a short-τ EMA
```
Or a **Savitzky–Golay-on-a-ring** (linear-regression slope over last `m` ring samples). The slope of a line fit can itself be maintained incrementally (keep running Σx, Σt, Σtx, Σt² over the fixed window) → O(1).

**Why we need it:** (a) derivative triggers are the fastest onset detector for impulsive HXR (§2.6); (b) the **Neupert effect** says `d/dt[soft] ∝ hard` — so the soft-band derivative is the physical bridge to the hard band (§4).

### 1.5 Online robust location/scale — running quantiles (P² / t-digest)

A mean-based baseline is biased upward by flares and spikes. A **robust** baseline (median, and MAD for scale) is far better. Storing all samples to take a median is O(n) memory — forbidden. Two O(1)-ish sketches solve this:

- **P² (Jain–Chlamtac 1985):** tracks a *fixed* set of `p`-quantiles by maintaining **5 markers per quantile** and updating their positions via a parabolic (P²) interpolation as each sample arrives. **O(1) per sample, O(1) memory.** Ideal for a single running median + a high quantile for the threshold. ([P² paper](https://www.cse.wustl.edu/~jain/papers/psqr.htm))
- **t-digest (Dunning):** a mergeable cluster sketch with excellent tail accuracy; per-sample cost ~O(log(buffer)), memory bounded by a compression parameter. Heavier than P² but *mergeable* across nodes and great if we want arbitrary quantiles. ([t-digest, Dunning 2019](https://arxiv.org/pdf/1902.04023))

**Recommendation:** P² for the per-band running **median** + **0.5·IQR / MAD** scale (cheap, constant state). Reserve t-digest for offline calibration or multi-node aggregation.

**Online MAD proxy:** maintain median `m` via P², and a second P² tracking the median of `|x_t − m|` → running MAD; robust σ ≈ `1.4826 · MAD`.

### 1.6 Primitive complexity table

| Primitive | Time/sample | State | Notes |
|---|---|---|---|
| EMA baseline | O(1) | 1 float | low-pass; choose τ |
| Welford variance | O(1) | 3 floats | unweighted |
| EWMV variance | O(1) | 2 floats | weighted, drift-aware (preferred) |
| Ring buffer (len L) | O(1) | L floats | fixed memory |
| First-difference deriv. | O(1) | 1 float | noisy |
| EMA-derivative / SG slope | O(1) | O(window) const | smoothed |
| P² running quantile | O(1) | 5 floats/quantile | robust median/MAD |
| t-digest | ~O(log) | O(δ⁻¹) | mergeable, tail-accurate |

---

## 2. Flare-Onset Detection Algorithms

Each is analysed for **time complexity**, **state**, and **false-alarm behaviour**. They are complementary; the recommended stack (§8) runs several in parallel and fuses them.

### 2.1 Adaptive thresholding (flux > baseline + k·σ) with robust baseline

The workhorse trigger.

```
trigger_t  =  x_t  >  b_t + k · σ_t
```
where `b_t` is the **robust** baseline (P² median, §1.5) and `σ_t` the **robust** scale (MAD-based or EWMV).

- **Complexity:** O(1) time, O(1) state.
- **False alarms:** for Gaussian noise, single-sample threshold at `k` sigma gives false-positive rate `≈ Φ(−k)` per sample. At 1 s cadence even `k=5` (≈3×10⁻⁷/sample) yields a false hit every ~38 days *per band per sample test* — too many over a mission. **Mitigations:**
  - **Persistence/M-of-N:** require the threshold to hold for `M` of the last `N` samples (ring buffer, §1.3). This is the operational GOES philosophy (4 consecutive minutes of increase — see §2.7). Multiplies the effective false-alarm interval enormously.
  - **Gated baseline update:** when `trigger_t` is true (flare suspected), **freeze** `b_t, σ_t` updates (or leak at 1/100th rate) so the flare doesn't poison the baseline.
  - **Robust baseline** (median not mean) so prior spikes don't inflate `b`.
  - **Hysteresis:** separate ON threshold (`k_on`, e.g. 4–5σ) and OFF threshold (`k_off`, e.g. 1–2σ) to avoid chattering at flare end.

**Hampel test variant:** trigger when `|x_t − m_t| > k · 1.4826 · MAD_t` using the running median/MAD. This *is* the streaming Hampel identifier ([Hampel filter](https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html)); robust to heavy tails. The same Hampel test is reused **as a despiker** in §7 (an outlier that is one-sample-wide is a cosmic ray; a sustained excursion is a flare).

### 2.2 CUSUM (cumulative sum) — recommended onset detector

CUSUM is the classic sequential change-point test: O(1), accumulates evidence so it catches **slow** onsets a single-sample threshold misses, with provably near-optimal detection delay for a given false-alarm rate.

**Log-likelihood-ratio form** (pre-change density `f₀`, post-change `f₁`):
```
S_t = max(0, S_{t-1} + log( f₁(x_t) / f₀(x_t) ));   S_0 = 0
alarm when S_t ≥ h
```
([CUSUM, online LLR change detection](https://arxiv.org/abs/2211.15070))

**Gaussian "upward shift" practical form** (detect a mean increase of at least `δ` over baseline `b`, scale `σ`):
```
# one-sided upper CUSUM (we only care about flux increases)
S_t = max(0,  S_{t-1} + (x_t - b_t) - k_c·σ_t )      # k_c = slack = δ/(2σ) typ. 0.5
alarm when S_t > h_c·σ_t                              # h_c ~ 4–5 (decision interval)
on alarm: record onset, reset S_t = 0
```
- **Complexity:** O(1) time, **one float** of state (`S`). The most memory-frugal change detector.
- **Tuning:** the slack `k_c` sets the *smallest* shift you'll chase (smaller `k_c` → faster on small flares, more false alarms). `h_c` trades **ARL₀** (mean time between false alarms) against **detection delay**. Average-run-length curves let you *design* the false-alarm rate analytically — a major operational advantage.
- **Onset-time estimate:** the time CUSUM last reset to 0 before the alarm is the maximum-likelihood change point → gives a principled **flare start time** for the catalogue.
- **Why recommended:** O(1), tunable FAR, gives start-time for free, and catches gradual SoLEXS rises better than instantaneous thresholding.

### 2.3 Page–Hinkley (PH) test

PH is the CUSUM specialised to detecting a change in the **mean of a stream relative to its running mean**; very popular for drift detection, cheap, robust.

```
# cumulative deviation from running mean, minus tolerance δ
m_t   = m_{t-1} + (x_t - m_{t-1})/n            # running mean (or EMA)
U_t   = U_{t-1} + (x_t - m_t - δ)             # cumulative sum of (excess) deviations
M_t   = min(M_t, U_t)                          # running minimum
PH_t  = U_t - M_t
alarm when PH_t > λ_PH
```
([Page–Hinkley method](https://www.geeksforgeeks.org/artificial-intelligence/page-hinkley-method/); [skmultiflow PageHinkley](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.PageHinkley.html))

- **Complexity:** O(1) time, O(1) state (`m, U, M`).
- **Params:** `δ` (magnitude tolerance / slack) and `λ_PH` (detection threshold); an optional forgetting factor `α_PH` weights recent data (drift-aware variant).
- **vs CUSUM:** PH bakes in the running-mean reference, so it self-adapts to slow baseline drift without a separate baseline estimator — convenient, but it can be *slower* to alarm than a well-tuned CUSUM and is sensitive to `δ`. Use as a **secondary** confirm.

### 2.4 Bayesian Online Change-Point Detection (BOCPD)

BOCPD maintains a **posterior over the run length** `r_t` (time since last change point) and updates it online; with conjugate exponential-family likelihoods the per-step update is sufficient-statistics addition. ([Adams & MacKay 2007 framework](https://arxiv.org/pdf/1902.04524v1))

```
for each new x_t:
  for each run-length hypothesis r in {0..R_max}:
     predictive  = p(x_t | suff_stats[r])          # conjugate predictive
     growth[r+1] = posterior[r] · predictive · (1 - H)   # no change
     cp_mass     += posterior[r] · predictive · H        # change (hazard H)
  posterior      = normalize([cp_mass, growth...])
  update suff_stats; prune/cap R at R_max
```

- **Complexity:** **O(R_t) per sample**, where `R_t` is the number of retained run-length hypotheses. Without pruning `R_t` grows with time → *not* O(1). With a **cap `R_max`** (or pruning hypotheses below a probability floor) it becomes **O(R_max)** = O(1) *constant but with a larger constant* than CUSUM.
- **Strengths:** principled uncertainty, gives `P(change at t)` directly (→ a natural **confidence** for the catalogue), handles unknown post-change parameters.
- **Verdict for nowcasting:** **not the primary** detector (heavier, ~10–100× CUSUM cost). Optionally run a *capped, Poisson-conjugate* BOCPD on a downsampled stream to produce a calibrated **confidence score** and to disambiguate overlapping flares. Document the cost; default OFF on the embedded/fastest path.

### 2.5 Poisson rate-jump detection — **the right tool for HEL1OS** (low counts)

Hard X-ray HEL1OS data are **photon counts** → **Poisson**, often **low count rate** in quiescence. Gaussian thresholds are invalid at low counts (variance = mean, asymmetric). Two options:

**(a) Poisson CUSUM** — detect rate jump `λ₀ → λ₁` (`λ₁>λ₀`). The LLR for a Poisson observation `x_t` (counts in bin) is `x_t·log(λ₁/λ₀) − (λ₁−λ₀)`:
```
S_t = max(0,  S_{t-1} + x_t·ln(λ₁/λ₀) - (λ₁ - λ₀) );   S_0 = 0
alarm when S_t ≥ h_p
```
O(1) time, one float. `λ₀` = current background rate (from EMA of counts, gated); `λ₁` = smallest flare rate of interest (e.g. `λ₁ = ρ·λ₀`, ρ≈1.5–2).

**(b) Poisson-FOCuS (RECOMMENDED for HEL1OS)** — FOCuS (Functional Online CUSUM) tests **all post-change magnitudes and all window sizes simultaneously** (you don't have to pre-guess `λ₁`), yet runs in **amortised O(1)–O(log) per sample** via a pruned piecewise-quadratic ("curve list") representation of the cost. Poisson-FOCuS is its Poisson specialisation, **developed precisely for onboard gamma-ray-burst/CubeSat triggers** — a direct analogue of HEL1OS flare onset. It is *mathematically equivalent to searching over all window sizes at ~half the cost of grid methods*. ([Poisson-FOCuS, Ward et al. 2023, JASA](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2235059); [arXiv 2208.01494](https://arxiv.org/abs/2208.01494))

Conceptual structure:
```
# maintain a list of "curves" (candidate change points), each a quadratic in the
# unknown post-change log-rate; background mu0 estimated from a long EMA of counts.
on new count x_t:
    update each active curve's Poisson cost with x_t           # add sufficient stat
    prune curves that can never become the max (convex-hull / functional pruning)
    add the new trivial curve for changepoint = t
    stat_t = max over curves of (Poisson LLR vs background mu0)
    alarm when stat_t > threshold (set from desired false-alarm rate / ARL0)
```
- **Complexity:** amortised **O(1)** (worst-case bounded; pruning keeps the curve list short — in practice a handful of curves). O(1) state in expectation.
- **Why it wins for HEL1OS:** no need to pick the flare magnitude in advance (flares span orders of magnitude); correct Poisson statistics at low counts; designed for exactly this onboard, real-time, count-burst problem; multi-timescale (catches both impulsive spikes and gradual rises) for free.

> **Background `μ0` for HEL1OS:** a long-τ EMA of counts (gated during bursts), optionally split per energy band; particle-background episodes (SAA-like, though L1 has no SAA, but solar-energetic-particle storms exist) handled in §7.

### 2.6 Derivative / gradient trigger & impulsive-spike detector (hard X-ray)

Impulsive HXR bursts have **steep leading edges**. A derivative trigger fires on rate-of-rise rather than level → earliest possible onset.
```
d_t = smoothed_derivative(x)            # §1.4
trigger when d_t > k_d · σ_d_t          # σ_d = running std of derivative (EWMV on d)
```
- **Complexity:** O(1).
- **Impulsive-spike detector:** combine derivative trigger with a **width gate**: a genuine HXR spike has rise *and* a minimum duration (≥ a few bins); a one-bin excursion is a **cosmic ray** → reject (despike, §7). Practically: Hampel test (§2.1) flags the outlier; if the excursion persists ≥ `w_min` bins it's a flare, else it's removed.
- **False alarms:** derivative amplifies noise → *must* smooth and use a robust `σ_d`. Best used as the **fast first-alert** that is then **confirmed** by Poisson-FOCuS/CUSUM (two-stage trigger).

### 2.7 Operational GOES-style rule (gradual soft X-ray) — reference logic

NOAA/SWPC's operational definition is a simple, robust **finite-state rule** on 1-min long-channel flux, and it's directly portable to SoLEXS:

- **Start:** first minute of **4 consecutive minutes of monotonic increase**, with the flux at minute 4 **≥ 1.4× (i.e. +40 %)** the flux at minute 1.
- **Peak:** maximum flux during the event.
- **End:** when flux **decays to the midpoint** between the peak and the start flux: `x ≤ (x_peak + x_start)/2`.
([NOAA SWPC GOES X-ray flux / event definition](https://www.spaceweather.gov/products/goes-x-ray-flux); see also automated GOES flare statistics, [Aschwanden & Freeland 2012](https://iopscience.iop.org/article/10.1088/0004-637X/754/2/112))

This is **O(1)** with a tiny ring buffer (last 4 minutes) and yields **start/peak/end** in the exact form the catalogue needs. We adopt it as the **soft-band finite-state machine (FSM)** that *frames* a flare once CUSUM/threshold raises onset. (At 1 s cadence, use a smoothed/averaged 1-min view to mirror GOES, while CUSUM watches the 1 s stream for early alert.)

### 2.8 Matched filter / template (gradual soft profile) + recursive approximation

Soft-X-ray flares have a characteristic **fast-rise / exponential-decay (FRED)** shape. A matched filter correlates the stream with this template `g[·]`; it is the **optimal linear detector** for a known shape in additive noise.

**FIR matched filter (exact):** `y_t = Σ_{i=0..L-1} g[i]·x_{t-i}` — O(L) per sample (uses the ring buffer as the delay line). For modest `L` (template a few hundred samples) this is fine, but it is **not O(1)**.

**Recursive (IIR) approximation — O(1):** approximate the FRED template by a **sum of a few exponentials**; each exponential is realised by a one-pole recursive filter (an EMA). Because exponentials have trivial recursions, the matched filter becomes a **sum of `P` EMAs (P≈2–4)** → **O(1) per sample, O(P) state**. This is exactly the **SPIIR** trick used for low-latency gravitational-wave matched filtering. ([SPIIR IIR matched filtering, Hooper et al. 2012](https://arxiv.org/pdf/1108.3186))

```
# FRED ≈ A·(decay-EMA) - A·(rise-EMA);  detect when correlation peaks
r_t = α_d·r_{t-1} + (1-α_d)·x_t          # slow pole (decay)
u_t = α_r·u_{t-1} + (1-α_r)·x_t          # fast pole (rise)
mf_t = c1·r_t - c2·u_t                    # template-matched statistic ~ "FREDness"
alarm when mf_t > h_mf · σ
```
- **Complexity:** O(1).
- **Use:** as a **shape confirmer / classifier feature**, not the primary trigger — it improves SNR for *gradual* soft flares and helps reject non-flare ramps. Also doubles as a soft/hard discriminator (HXR is spiky, fails the gradual template).

### 2.9 Onset-detector comparison

| Detector | Time | State | Best at | FAR control | Gives start time | Notes |
|---|---|---|---|---|---|---|
| Adaptive k·σ threshold | O(1) | O(1) | level steps | via k + M-of-N | approx | needs robust baseline + gating |
| Hampel (median/MAD) | O(1) | O(1) | robust level | via k | approx | also a despiker |
| **CUSUM** | **O(1)** | **1 float** | gradual+step | **analytic (ARL)** | **yes (last reset)** | **primary soft trigger** |
| Page–Hinkley | O(1) | O(1) | drift vs mean | via λ_PH | approx | self-adapting; secondary |
| BOCPD (capped) | O(R_max) | O(R_max) | uncertainty/overlap | via hazard | posterior | confidence score; heavier |
| Poisson CUSUM | O(1) | 1 float | count jump (known) | analytic | yes | needs λ₁ guess |
| **Poisson-FOCuS** | **~O(1) amort.** | **~O(1)** | **count burst (any size/scale)** | **analytic** | **yes** | **primary HEL1OS trigger** |
| Derivative/spike | O(1) | O(1) | impulsive HXR onset | via k_d + width | yes (edge) | fast first-alert; confirm |
| Matched filter (FIR) | O(L) | O(L) | known soft shape | optimal-linear | peak | not O(1) |
| Matched filter (IIR) | O(1) | O(P) | gradual soft shape | tunable | peak | SPIIR-style; confirmer |

---

## 3. GOES Flare Classification

### 3.1 The GOES class scale (1–8 Å peak flux)

A flare's class is the **order of magnitude of its peak 1–8 Å soft-X-ray flux** `F_peak` (W m⁻²), with a linear 1–9 sub-class within each decade ([GOES classification](https://arxiv.org/html/2511.20465v1)):

| Class | Peak flux `F` (W m⁻², 1–8 Å) |
|---|---|
| A | `F < 1×10⁻⁷` |
| B | `1×10⁻⁷ ≤ F < 1×10⁻⁶` |
| C | `1×10⁻⁶ ≤ F < 1×10⁻⁵` |
| M | `1×10⁻⁵ ≤ F < 1×10⁻⁴` |
| X | `F ≥ 1×10⁻⁴` |

**Sub-class** is the mantissa: within a decade the scale is linear 1–9, so `M2.5` means `2.5×10⁻⁵ W m⁻²`. X-class is open-ended (`X10 = 1×10⁻³`, etc.). An X2 is twice an X1, 4× an M5, 40× a C5.

### 3.2 Closed-form classification (O(1))

Given peak flux `F` (W m⁻²):
```
classify(F):
    e   = floor(log10(F))                 # decade exponent
    letter = { -8:'A', -7:'B', -6:'C', -5:'M', -4:'X' }.get(e, 'A' if e<-8 else 'X')
    mant = F / 10**e                       # in [1,10)
    if letter == 'X' and e > -4:           # ≥ X10 events
        mant = F / 1e-4
    return f"{letter}{mant:.1f}"           # e.g. 'M2.5'
```
Pure arithmetic → **O(1)**. Edge clamps: `e < −8 → A` (report `A<...`), `e ≥ −4 → X` (allow Xn>9).

### 3.3 Deriving a GOES-equivalent class from SoLEXS

SoLEXS measures **2–22 keV** flux, **not** the GOES 1–8 Å (≈1.55–12.4 keV) band, so a **cross-calibration** is required to emit a familiar GOES class. The SoLEXS team has already cross-calibrated SDD2 against GOES-XRS and XSM ([SoLEXS in-flight performance, arXiv 2509.26292](https://arxiv.org/html/2509.26292v1)). Practical pipeline:

1. **Band conversion:** fold the SoLEXS spectrum (or count rate) through the GOES 1–8 Å response, **or** use the SoLEXS team's empirical relation. Reported behaviour: SDD2 vs GOES XRS shows a **linear correlation at intermediate fluxes**, with SDD2 ~**15 % lower at low flux** and higher at flare peaks (driven by GOES's flat-spectrum assumption). So a *temperature-dependent* (non-flat-spectrum) conversion is more accurate than a single scale factor.
2. **Isothermal model route (more accurate):** fit/estimate plasma temperature `T` and emission measure `EM` from the SoLEXS spectrum, then synthesise the equivalent GOES 1–8 Å flux via the **empirical temperature–XRS-B flux relation** the SoLEXS paper uses. (This is the FOXES-style "operational X-ray emission synthesis" idea — [FOXES, arXiv 2604.10835](https://arxiv.org/html/2604.10835).)
3. **Operational shortcut (O(1)):** precompute a **lookup/regression** `F_GOES ≈ R(C_SoLEXS, hardness)` from the cross-calibration (a small polynomial or 2-D table indexed by SoLEXS count rate and a hardness ratio). At runtime it's an O(1) evaluation. Then apply §3.2.

> **Recommendation:** Emit **two numbers**: (i) the *measured* SoLEXS peak flux/rate (instrument-native, always valid), and (ii) the *GOES-equivalent class* via the calibrated conversion, tagged with its uncertainty. Operators think in GOES classes; scientists want the native value.

**Saturation caveat (X-class):** SDD1 saturates above ~`1×10⁵ counts/s`; during big flares **SDD2** is the valid detector (it captured X8.7 in May 2024). Classification logic must **select the unsaturated detector** before computing peak flux (see §7).

---

## 4. Master-Catalogue Merging (soft ⊕ hard → one flare event)

We detect **independently** in each band, then **associate** soft and hard detections into single physical events.

### 4.1 Physical prior: the Neupert effect

For ~80 % of large flares, the **hard-X-ray light curve tracks the time-derivative of the soft-X-ray light curve**: `HXR(t) ∝ d/dt[SXR(t)]`. Equivalently, **hard X-rays lead soft X-rays**: the impulsive (HXR) phase deposits energy, driving chromospheric evaporation that *then* shines in soft X-rays (gradual phase). HXR **peaks before** SXR; SXR **peak ≈ time when HXR returns to baseline**. ([Neupert effect](https://arxiv.org/pdf/2101.11069); [Veronig et al.](https://ui.adsabs.harvard.edu/abs/2003HvaOB..27...47V))

**Consequences for association:**
- Expect **HXR onset ≤ SXR onset**, and **HXR peak < SXR peak** (lead of seconds–minutes).
- The **soft-band derivative** (§1.4) should correlate with the **hard-band light curve** — a quantitative *confirmation score* for a candidate pairing.

### 4.2 Association algorithm (time-window + Neupert-aware)

Maintain a small set of **open events**. Each new band detection is matched against open events; else it opens a new one.

```
ASSOCIATE(detection D from band B):     # B ∈ {soft, hard}
  candidates = open_events overlapping [D.start - W_lead, D.end + W_lag]
  if candidates:
     e = argmax over candidates of MATCH_SCORE(D, e)
     if MATCH_SCORE(D, e) ≥ τ_match:
         merge D into e (set e.soft/e.hard flags, update start/peak/end, class)
         return
  open_new_event(D)
```

**MATCH_SCORE** combines:
- **Temporal overlap / proximity** of [start,end] intervals (allow asymmetric window: HXR may *precede* SXR → `W_lead` larger on the hard-before-soft side).
- **Peak-lag consistency** with Neupert: reward `t_peak^HXR < t_peak^SXR` within a few minutes.
- **Neupert correlation** (optional, if both light curves buffered): Pearson/lagged correlation of `d/dt[SXR]` vs `HXR` over the overlap → strong physical confirmation.

Windows (initial defaults, tune on data): `W_lead ≈ 5 min`, `W_lag ≈ 10–15 min` (soft decay is long).

### 4.3 De-duplication

- **Within a band:** if a new onset fires while an event is still "open" (not yet ended) and overlaps it, treat as the **same** flare (re-trigger during a multi-peak flare), not a new one — unless separated by a clean return-to-baseline (FSM `end` reached) plus a guard time `Δ_dedup`. This prevents counting each sub-peak of a complex flare as a separate event.
- **Cross-band:** §4.2 already prevents double-listing (one master event carries both flags).
- **Re-detection guard:** after emitting `end`, suppress new onsets for `Δ_dedup` (e.g. 1–2 min) to avoid decay-phase wiggles re-triggering.

### 4.4 Event record schema

```jsonc
{
  "event_id":        "uuid",            // stable key
  "t_start":         "ISO-8601",        // earliest band onset (CUSUM last-reset)
  "t_peak":          "ISO-8601",        // soft-band peak (canonical), per-band peaks below
  "t_end":           "ISO-8601",        // soft-band end (FSM midpoint rule)
  "soft": {                              // SoLEXS sub-record (null if not detected)
     "detected": true, "t_start":"…","t_peak":"…","t_end":"…",
     "peak_flux_native": 1.2e3,          // instrument units (counts/s)
     "peak_flux_goes_equiv": 2.5e-5,     // W/m^2 via cross-calib (§3.3)
     "detector_used": "SDD2", "saturated": false
  },
  "hard": {                              // HEL1OS sub-record (null if not detected)
     "detected": true, "t_start":"…","t_peak":"…",
     "peak_counts": 850, "energy_band":"8-30keV"
  },
  "goes_class":      "M2.5",            // from soft GOES-equiv peak
  "flags":           { "soft": true, "hard": true,
                       "neupert_consistent": true,
                       "data_gap_during": false, "spike_rejected": 3 },
  "confidence":      0.93,              // fused detector confidence (see §4.6)
  "detectors":       ["CUSUM","FOCuS","threshold"],  // which fired
  "ref_match":       { "catalog":"GOES/HEK", "id":"…", "Δt_s": 41 } // post-hoc
}
```

### 4.5 O(1)-queryable catalogue (time-bucket hash index)

Index events by **time bucket** so any "what flares near time T?" query is O(1):
```
bucket(t) = floor(epoch_seconds(t) / B)        # e.g. B = 3600 s (1-hour buckets)
index: HashMap<bucket_id, list<event_id>>
```
- **Insert:** O(1) (append to the bucket list).
- **Point/interval query:** hash the target time(s) to bucket(s) → O(1) for a point, O(#buckets spanned) for a range (constant per bucket). Association (§4.2) only scans the **current + previous** bucket(s), keeping it O(1) per detection.
- For richer queries (by class, by flag), keep small secondary hash indices (`class → events`). All O(1) amortised.

### 4.6 Confidence fusion

Combine the firing detectors into one score, e.g. a logistic/noisy-OR over normalized statistics:
```
confidence = 1 - Π_d (1 - p_d)          // noisy-OR across detectors d that fired
```
where `p_d` is each detector's calibrated detection probability (CUSUM via its statistic vs ARL curve; FOCuS via its threshold margin; BOCPD posterior if enabled). Cross-band agreement and Neupert consistency add bonus weight.

---

## 5. Robustness

| Hazard | Symptom | O(1) mitigation |
|---|---|---|
| **Data gaps** | missing samples (telemetry, eclipse, mode change) | Detect via timestamp jump > `Δt·g`. **Hold** baseline/variance state (don't update through the gap), mark `data_gap_during=true`, and **suppress** alarms for a guard interval after resume (state is stale). Optionally re-seed baseline from pre-gap value. Do **not** interpolate into change detectors. |
| **Cosmic-ray / particle spikes** | 1–2-sample huge excursions, esp. HEL1OS | **Streaming Hampel despiker** (§2.1): `|x−median|>k·1.4826·MAD` → candidate; if excursion width `< w_min` bins ⇒ **cosmic ray**, replace with running median *before* it reaches detectors. A real flare persists ≥ `w_min` ⇒ passes through. Count rejections in `flags.spike_rejected`. |
| **Poisson noise at low counts** | Gaussian σ invalid; asymmetric fluctuations | Use **Poisson** detectors on counts (Poisson-FOCuS / Poisson CUSUM, §2.5). For thresholds use Poisson tail (or Anscombe transform `2√(x+3/8)` to stabilise variance → then Gaussian tools become valid). Aggregate very-low-rate bins (coarser `Δt`) adaptively. |
| **Instrument saturation (X-class)** | SDD1 flat-tops > ~1e5 c/s; clipped peak ⇒ underestimated class | **Detector arbitration:** monitor both SoLEXS SDDs; when SDD1 near/above saturation, switch peak/flux readout to **SDD2** (its small aperture stays linear to X-class — captured X8.7). Set `saturated` flag; if both saturate, report a **lower bound** class (`≥Xn`). Mirrors GOES recalibration practice for saturated giants ([GOES saturation/recalibration, arXiv 2310.11457](https://arxiv.org/pdf/2310.11457)). |
| **Baseline contamination by flare** | detector "adapts away" the flare; baseline creeps up | **Gated/frozen baseline** during active flare (§2.1); robust median baseline (§1.5) resists residual contamination. |
| **Solar energetic particle (SEP) storms** | broadband count elevation in HEL1OS (not a flare) | Distinguish via spectral shape / lack of soft-band Neupert counterpart; raise event but flag low `confidence`; SEP background tracked separately in `μ0`. |

---

## 6. Evaluation of Detection

Validate against a **reference catalogue** — GOES/SWPC event list and/or **HEK** (Heliophysics Events Knowledgebase). Build a confusion matrix by **time-matching** detections to reference flares (match if peak times within tolerance `Δ_tol`, e.g. ±5 min, and class within ±1 sub-class for the classification check).

**Contingency counts:** TP (detected & real), FP (detected, no reference = false alarm), FN (real, missed), TN (n/a for rare-event point process).

| Metric | Formula | Perfect | Meaning |
|---|---|---|---|
| **Precision** | `TP/(TP+FP)` | 1 | purity of detections |
| **Recall = POD** (Prob. of Detection) | `TP/(TP+FN)` | 1 | fraction of real flares caught |
| **FAR** (False Alarm Ratio) | `FP/(TP+FP)` | 0 | fraction of alerts that were wrong |
| **F1** | `2·P·R/(P+R)` | 1 | balance |
| **CSI / Threat Score** | `TP/(TP+FP+FN)` | 1 | rare-event skill (ignores TN) |
| **HSS** (Heidke Skill) | skill vs random | 1 | chance-corrected |
| **TSS** (True Skill / Peirce) | `POD − POFD` | 1 | threshold-independent skill |

([POD/FAR/CSI in space-weather verification](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011SW000760); [SWPC flare-forecast verification](https://arxiv.org/pdf/1707.07903))

**Additional operational metrics:**
- **Detection delay / latency:** `t_alarm − t_true_onset` (the nowcasting KPI — CUSUM/FOCuS minimise this).
- **ARL₀** (in-control average run length) = mean samples between false alarms — *design* CUSUM/FOCuS thresholds to a target ARL₀, then verify empirically on quiet-Sun stretches.
- **Classification accuracy:** confusion matrix over A/B/C/M/X; mean absolute error in log-flux vs GOES; per-class POD (M & X matter most operationally).
- **Reliability diagram / Brier score** if `confidence` is interpreted as a probability.
- **ROC / TSS-vs-threshold** sweep of `k`, `h_c`, FOCuS threshold to choose the operating point that meets the mission's POD/FAR trade-off (e.g. maximise TSS, or fix FAR ≤ target then maximise POD).

---

## 7. Recommended Nowcasting Algorithm Stack

A two-band, two-stage, fully O(1)-per-sample pipeline.

```
                         ┌────────────────────────────────────────────────────┐
  SoLEXS stream  ──▶ DESPIKE(Hampel) ──▶ ROBUST BASELINE (P² median + MAD,    │
   (1 s flux)          gap-gate          EWMV) ──▶ [ CUSUM (primary) ]         │  SOFT
                                                  [ adaptive k·σ + M-of-N ]    │  DETECT
                                                  [ IIR matched filter (FRED) ]│  + FSM
                                                  ──▶ GOES-style FSM (start/   │  (start,
                                                       peak/end) ──▶ class via │   peak,
                                                       cross-calib (§3.3)      │   end,
                                                                               │   class)
                                                                               │
  HEL1OS stream  ──▶ DESPIKE(width-gate) ──▶ POISSON BASELINE (EMA of counts) ─┤  HARD
   (counts)            gap-gate                ──▶ [ Poisson-FOCuS (primary) ] │  DETECT
                                               [ derivative/spike (fast alert)]│  (start,
                                               [ Poisson CUSUM (confirm) ]     │   peak)
                         └────────────────────────────────────────────────────┘
                                                │
                                                ▼
                       ASSOCIATION (time-window + Neupert lag/correlation, §4.2)
                                                │
                                                ▼
                       DE-DUP (intra/inter-band, guard time, §4.3)
                                                │
                                                ▼
            MASTER CATALOGUE  (event schema §4.4, hash-bucket index §4.5, confidence §4.6)
                                                │
                                                ▼
                       EVALUATION vs GOES/HEK (POD, FAR, CSI, latency, §6)
```

**Per-band choices:**
- **Soft (SoLEXS):** primary onset = **CUSUM** (O(1), 1 float, analytic FAR, gives start time); framed by the **GOES-style FSM** for canonical start/peak/end; **adaptive k·σ + M-of-N** as a fast cross-check; **IIR (SPIIR) matched filter** as gradual-shape confirmer/feature. Robust **P² median + MAD** baseline, gated during flares.
- **Hard (HEL1OS):** primary onset = **Poisson-FOCuS** (correct Poisson stats at low counts, multi-scale, ~O(1), built for onboard burst triggers); **derivative/spike** detector for earliest impulsive alert; **Poisson CUSUM** as a confirm. Width-gated despiking removes cosmic rays.
- **Optional:** capped Poisson **BOCPD** on a downsampled stream to emit a calibrated **confidence** and resolve overlapping flares — *off by default* on the fastest path (it is O(R_max), not O(1)-cheap).

**Everything on the hot path is O(1) per sample with O(1) state**, satisfying the "fastest platform / O(1)" mandate. The only super-constant pieces (FIR matched filter, full BOCPD) are deliberately replaced by O(1) approximations or kept off the critical path.

---

## 8. Recommended O(1) detector — reference pseudocode

```python
# ===== SHARED O(1) PRIMITIVES =====
class EMA:
    def __init__(self, alpha, x0=0.0): self.a=alpha; self.m=x0
    def update(self, x): self.m = self.a*self.m + (1-self.a)*x; return self.m

class EWMV:                                  # EW mean + variance (drift-aware)
    def __init__(self, alpha, x0=0.0): self.a=alpha; self.m=x0; self.S=0.0
    def update(self, x):
        d = x - self.m
        self.m += (1-self.a)*d
        self.S  = self.a*(self.S + (1-self.a)*d*d)
        return self.m, self.S                 # mean, variance
    def sd(self): return self.S**0.5

# P² running median + MAD assumed available as RobustBaseline (O(1)); omitted for brevity.

# ===== SOFT-BAND ONSET: CUSUM (primary) =====
class CUSUM_Up:
    """One-sided upper CUSUM for flux INCREASE. O(1) time, 1 float state."""
    def __init__(self, k_slack=0.5, h=5.0): self.k=k_slack; self.h=h; self.S=0.0; self.t0=None
    def update(self, x, baseline, sigma, t):
        if self.S == 0.0: self.t0 = t                      # provisional change-point
        self.S = max(0.0, self.S + (x - baseline) - self.k*sigma)
        if self.S > self.h*sigma:
            onset_time = self.t0                           # MLE change point
            self.S = 0.0                                   # reset after alarm
            return True, onset_time
        return False, None

# ===== HARD-BAND ONSET: Poisson-FOCuS (primary, conceptual) =====
class PoissonFOCuS:
    """Detect Poisson rate INCREASE over background mu0, any magnitude/timescale.
       Amortised ~O(1) via pruned piecewise-quadratic 'curves'. mu0 from gated EMA."""
    def __init__(self, threshold): self.thr=threshold; self.curves=[]   # each: (count_sum, n)
    def update(self, x, mu0):
        new=[]
        for (csum, n) in self.curves:                      # extend each candidate
            csum += x; n += 1
            # Poisson LLR for best post-change rate lam_hat = csum/n vs mu0:
            #   LLR = csum*ln(lam_hat/mu0) - (csum - n*mu0)   (0 if lam_hat<=mu0)
            if csum > n*mu0:
                new.append((csum, n))
        new.append((x, 1))                                 # new change-point at t
        self.curves = prune_convex(new)                    # functional/convex-hull pruning
        stat = 0.0
        for (csum, n) in self.curves:
            if csum > n*mu0:
                stat = max(stat, csum*log(csum/(n*mu0)) - (csum - n*mu0))
        return (stat > self.thr), stat                     # alarm, statistic

# ===== SOFT-BAND FRAMING: GOES-style FSM (start/peak/end) =====
class FlareFSM:
    """Frames a flare once onset fires: start, peak, end (midpoint rule). O(1)."""
    states = ("IDLE","RISING","DECAY")
    def __init__(self): self.s="IDLE"; self.x_start=None; self.x_peak=None; self.t_peak=None
    def step(self, x, t, onset):
        if self.s=="IDLE":
            if onset: self.s="RISING"; self.x_start=x; self.x_peak=x; self.t_peak=t
        elif self.s in ("RISING","DECAY"):
            if x > self.x_peak: self.x_peak=x; self.t_peak=t; self.s="RISING"
            elif x < self.x_peak: self.s="DECAY"
            if self.s=="DECAY" and x <= 0.5*(self.x_peak + self.x_start):
                ev = (self.x_start, self.x_peak, self.t_peak, t)  # start,peak,t_peak,end
                self.s="IDLE"; return ev
        return None

# ===== TOP-LEVEL PER-BAND LOOP (O(1) per sample) =====
def soft_band_on_sample(x_raw, t):
    x = hampel_despike(x_raw)                               # §7, O(1)
    if gap_detected(t): hold_state(); return                # §7
    b, mad = robust_baseline.update(x)                      # P² median+MAD, gated
    sig = 1.4826*mad
    onset, t0 = cusum.update(x, b, sig, t)
    also = adaptive_threshold(x, b, sig)                    # k·sigma + M-of-N
    ev = fsm.step(smooth_1min(x), t, onset or also)
    if ev: emit_soft_detection(ev, goes_equiv_class(ev.peak))   # §3.3, §4

def hard_band_on_sample(c_raw, t):
    c = width_gate_despike(c_raw)                           # cosmic-ray reject
    if gap_detected(t): hold_state(); return
    mu0 = bg_counts.update(c) if not in_burst else bg_counts.m   # gated EMA
    onset, stat = focus.update(c, mu0)
    fast = derivative_trigger(c)                            # earliest alert
    if onset or fast: emit_hard_detection(t, stat)

# Association, de-dup, catalogue insert, confidence: §4 (all O(1) per detection).
```

---

## 9. Key references (consulted)

- CUSUM / online LLR change detection — [arXiv 2211.15070](https://arxiv.org/abs/2211.15070)
- Page–Hinkley — [GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/page-hinkley-method/), [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.PageHinkley.html)
- BOCPD (Adams & MacKay) — [arXiv 1902.04524](https://arxiv.org/pdf/1902.04524v1)
- **Poisson-FOCuS (GRB/CubeSat onboard count-burst trigger)** — [Ward et al. 2023, JASA](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2235059), [arXiv 2208.01494](https://arxiv.org/abs/2208.01494)
- Welford online variance — [davidma.me](http://davidma.me/blog/2025/Welfords-Algo/)
- P² running quantiles — [Jain & Chlamtac 1985](https://www.cse.wustl.edu/~jain/papers/psqr.htm)
- t-digest — [Dunning 2019, arXiv 1902.04023](https://arxiv.org/pdf/1902.04023)
- Hampel filter / robust outlier despiking — [SAS DO-Loop](https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html), [MathWorks hampel](https://www.mathworks.com/help/signal/ref/hampel.html)
- IIR/SPIIR matched-filter approximation — [Hooper et al. 2012, arXiv 1108.3186](https://arxiv.org/pdf/1108.3186)
- Neupert effect — [arXiv 2101.11069](https://arxiv.org/pdf/2101.11069), [Veronig 2003](https://ui.adsabs.harvard.edu/abs/2003HvaOB..27...47V)
- GOES classification & automated flare statistics — [Aschwanden & Freeland 2012](https://iopscience.iop.org/article/10.1088/0004-637X/754/2/112), [flare-prediction review, arXiv 2511.20465](https://arxiv.org/html/2511.20465v1)
- GOES event definition (start/peak/end) — [NOAA SWPC GOES X-ray flux](https://www.spaceweather.gov/products/goes-x-ray-flux)
- GOES saturation/recalibration of giant flares — [arXiv 2310.11457](https://arxiv.org/pdf/2310.11457)
- **SoLEXS instrument & GOES cross-calibration** — [arXiv 2509.26292](https://arxiv.org/html/2509.26292v1), [Solar Phys 2025](https://link.springer.com/article/10.1007/s11207-025-02494-0)
- **HEL1OS instrument** — [arXiv 2512.12679](https://arxiv.org/pdf/2512.12679), [Solar Phys 2025](https://link.springer.com/article/10.1007/s11207-025-02543-8)
- FOXES operational X-ray synthesis — [arXiv 2604.10835](https://arxiv.org/html/2604.10835)
- Verification (POD/FAR/CSI/TSS) — [Crown 2012, Space Weather](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011SW000760), [RWC Japan flare verification, arXiv 1707.07903](https://arxiv.org/pdf/1707.07903)
