# Aditya-L1 X-ray Payloads (SoLEXS & HEL1OS): Instrumentation, Data, and Flare Physics for Nowcasting/Forecasting

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 15
**Goal:** Nowcast & forecast solar flares from Aditya-L1 SoLEXS (soft X-ray) and HEL1OS (hard X-ray) light curves obtained via the ISSDC PRADAN portal.
**Author role:** Heliophysics instrumentation research specialist
**Date:** 2026-06-20
**Status:** Research deliverable #1 (instrument + data + physics foundation for the pipeline)

> **Sourcing note.** Quantitative figures below are drawn primarily from the two refereed instrument papers — the SoLEXS paper (Shanmugam et al., *Solar Physics* 300:87, 2025; arXiv:2509.26292 "Ground Calibration and In-flight Performance") and the HEL1OS paper (Nandi et al., *Solar Physics*, 2025; arXiv:2512.12679) — plus ISRO/ISSDC pages and the standard solar-flare literature. Figures verified against a primary instrument paper are stated as exact; figures from secondary sources or general literature are flagged **(approx.)**. No URLs are fabricated; sources are listed by name at the end.

---

## 0. Executive orientation: why two X-ray spectrometers

Aditya-L1 carries **two co-aligned, full-Sun ("Sun-as-a-star", spatially integrated, non-imaging) X-ray spectrometers** that together span the solar flare X-ray spectrum from the thermal soft band into the non-thermal hard band:

| Payload | Band | Physics it traces | Flare phase it dominates |
|---|---|---|---|
| **SoLEXS** | 1–22 keV soft X-ray (SXR) | Thermal bremsstrahlung + line emission from hot (~10–30 MK) coronal plasma | Gradual phase (rise → peak → decay); the GOES-equivalent channel |
| **HEL1OS** | 8–150 keV hard X-ray (HXR) | Non-thermal bremsstrahlung from accelerated electrons (plus high-T thermal at the low end) | Impulsive phase (spiky bursts, QPPs, precursors) |

The scientific lever for forecasting is that **HXR ≈ d/dt(SXR)** — the **Neupert effect** (Section 7). Because the impulsive HXR rise *precedes* the SXR peak by minutes, HEL1OS gives an early-warning signal for the SXR/GOES-class peak that SoLEXS will reach shortly after. Combining the two payloads is therefore not a "nice-to-have" — it is the core physical basis of a short-horizon (minutes) flare nowcast.

---

## 1. SoLEXS — Solar Low Energy X-ray Spectrometer (Soft X-ray)

SoLEXS is a soft X-ray spectrometer built by U. R. Rao Satellite Centre (URSC/ISRO). It performs continuous, full-disk-integrated soft X-ray spectroscopy of the Sun, conceptually the Aditya-L1 analogue of the GOES XRS and of Chandrayaan-2's XSM.

### 1.1 Energy range, resolution, and detectors

| Parameter | Value | Notes |
|---|---|---|
| **Energy range** | **1–22 keV** (science quality typically quoted ~2–22 keV; useful response down to ~1 keV) | Soft X-ray |
| **Spectral resolution (FWHM)** | **~170 eV at 5.9 keV** (Mn Kα) | On-orbit: **164.9 ± 4.7 eV (SDD1)**, **171.2 ± 5.9 eV (SDD2)** |
| **Resolution stability vs T** | ±15 eV across qualification temperature range | Thermo-vacuum verified |
| **Detector type** | **Two Silicon Drift Detectors (SDDplus, PNDetector)** | Not SCDs — the design uses SDDs |
| **Active area (each SDD)** | **30 mm²** | Per detector |
| **Detector thickness** | 450 ± 20 µm Si | Sets high-energy efficiency rolloff |
| **Entrance window** | 8 µm DuraBeryllium Plus | Sets low-energy cutoff (~1–2 keV) |
| **Operating temperature** | Cooled by Peltier/TEC to ~ −45 °C (range −60 to −40 °C) | Cooling stabilizes gain/resolution |

**Dual-aperture dynamic-range trick.** SoLEXS uses **two SDDs with very different pinhole apertures** so that one detector is optimized for quiet/small flares and the other survives large flares without saturating:

| Detector | Aperture diameter | Aperture area | Field of view (half-angle) | Role |
|---|---|---|---|---|
| **SDD1** (large aperture) | 3.008 ± 0.001 mm | **7.106 ± 0.010 mm²** | ±1.8° | Sensitive at quiet Sun / A–C class; **saturates** above ~10⁵ cps |
| **SDD2** (small aperture) | 0.368 ± 0.001 mm | **0.1065 ± 0.0006 mm²** | ±1.3° | Primary detector during **high activity / M–X class** |

The ~67× aperture-area ratio buys roughly two extra decades of unsaturated dynamic range. The **full-Sun FOV** is guaranteed because the Sun (~0.27° half-angle disk at L1) sits well inside the ±1.3°–1.8° acceptance, and the instrument integrates the whole disk (no imaging).

### 1.2 Time cadence / integration

| Data stream | Cadence | Pulse shaping time |
|---|---|---|
| **Spectral (PHA histogram)** | **1 s** | 4 µs (spectral chain) |
| **Temporal (light curve / count rate)** | **0.1 s (10 Hz)** | 0.7 µs (timing chain) |

So SoLEXS delivers **1-second spectra** and a **fast 0.1-second light curve**. The fast timing chain (short shaping time) is what gives the high count-rate capability and the 10 Hz light curve; the slow spectral chain gives the energy resolution.

### 1.3 Energy binning

- **340 channels** across 2–22 keV.
- Channels 1–168: **~47.75 eV/channel** (≤ ~8 keV).
- Channels 169–340: **~94.5 eV/channel** (8–22 keV).
- On-orbit gain/offset (drives PHA→energy conversion):
  - **SDD1:** gain 48.24 ± 0.28 eV/ch, offset −33.9 ± 33.4 eV.
  - **SDD2:** gain 47.52 ± 0.33 eV/ch, offset 86.7 ± 39.2 eV.
  - Gain depends weakly on detector temperature; offset depends (quadratically) on electronics-module temperature — **a calibration quirk the pipeline must track via housekeeping temperatures.**

### 1.4 Dead-time and saturation

| Parameter | Value |
|---|---|
| Spectral-chain dead-time (ground) | 10 µs |
| Spectral-chain dead-time (in-flight) | **13.65 µs** |
| Spectral-chain efficiency (in-flight) | **88.83%** |
| Timing-chain dead-time | 1.6 µs |
| Dead-time model | **Paralyzable**: m = n·exp(−n·τ) (measured m vs true n) |
| Spurious/baseline count rate (timing chain) | ~364 cps |
| Design dynamic range | ~5 orders of magnitude (A-class → X-class) |
| Observed peak rates in big flares | ~10⁵–10⁶ cps |
| **SDD1 saturation onset** | **~10⁵ cps** → use SDD2 above this |

**Saturation behavior on big flares (critical for the pipeline).** During strong flares SDD1 enters the paralyzable regime: as true rate rises, the *observed* rate eventually turns over and *decreases* (counts pile up within one shaping window and are lost). This means a naive SDD1 light curve can show a spurious "dip" at flare maximum. The mitigation built into the instrument is to **switch to SDD2** (small aperture) for M/X-class, and to apply the paralyzable dead-time correction. The pipeline must **prefer SDD2 (or a properly dead-time-corrected, cross-checked combination) for large events** and treat SDD1 turnover as an artifact, not a real flux drop.

### 1.5 Calibration

- **Onboard reference:** an internal **⁵⁵Fe** source (Mn Kα 5.898 keV, Mn Kβ 6.490 keV) plus a 6 µm **Ti foil** giving fluorescent Ti Kα (4.507 keV) / Ti Kβ (4.932 keV). These let the gain/offset be tracked in flight.
- **Ground:** XRF targets (e.g., JSC-1A lunar simulant with salt, Mu-metal), and synchrotron monochromatic beams at RRCAT BL-16 (6.5–16 keV) for the response/ARF.
- **Spectral response model:** HYPERMET — main Gaussian + Si escape peak (1.74 keV below line) + exponential tail + low-energy shelf. Main-peak response >99% at 6 keV.
- **Spectral fitting:** isothermal model via **sunkit-spex** with the **CHIANTI** atomic database; fits temperature, emission measure (EM), and abundances (Fe, Ca, Ar, S).

### 1.6 How GOES-class (W/m²) flux is derived from SoLEXS

GOES "flare class" is defined by the **peak flux in the GOES-XRS 1–8 Å (≈1.55–12.4 keV) long channel**, in W m⁻²:

| GOES class | Peak 1–8 Å flux (W/m²) |
|---|---|
| A | 10⁻⁸ – 10⁻⁷ |
| B | 10⁻⁷ – 10⁻⁶ |
| C | 10⁻⁶ – 10⁻⁵ |
| M | 10⁻⁵ – 10⁻⁴ |
| X | ≥ 10⁻⁴ (X10 = 10⁻³, etc.) |

To put SoLEXS on the GOES scale:
1. Convert SoLEXS counts → photon flux using the **ARF (effective area) and RMF (redistribution matrix)** from calibration (peak effective area ~7 cm² for SDD1, ~0.1 cm² for SDD2, falling with energy).
2. Either (a) **fold the calibrated SoLEXS photon spectrum into the GOES 1–8 Å (and 0.5–4 Å) bandpass** and integrate to get W m⁻² directly, or (b) fit the isothermal (T, EM) model and **synthesize the GOES-band irradiance** from that model.
3. **Cross-calibrate against GOES-XRS** to validate. The SoLEXS team reports a confirmed **linear correlation with GOES-XRS-A (3.1–24.8 keV)**: SDD2 reads ~15% lower than GOES at quiet times and higher during flares — a known offset attributed to GOES's flat-spectrum assumption. Radiometric agreement with **Chandrayaan-2 XSM is within ~10%**, with residual uncertainty in the low-energy ARF (~2–2.3 keV) and in dead-time correction at varying rates.

> **Pipeline takeaway:** "GOES-class from SoLEXS" is a *derived product*, not a raw column. Expect to (i) use the published ARF/RMF, (ii) fold into the GOES band, and (iii) apply a small empirical correction factor calibrated against overlapping GOES-XRS data. Keep GOES XRS as the cross-calibration anchor.

---

## 2. HEL1OS — High Energy L1 Orbiting X-ray Spectrometer (Hard X-ray)

HEL1OS (built at URSC/ISRO) continuously measures the **time-resolved hard X-ray spectrum of solar flares from 8 to 150 keV**. It is the non-thermal/impulsive-phase complement to SoLEXS.

### 2.1 Energy range and detectors

| Parameter | CdTe detector | CZT detector |
|---|---|---|
| **Energy sub-range** | **8–70 keV** | **20–150 keV** |
| **Geometric area** | **0.5 cm²** | **32 cm²** |
| **Pixel format** | Non-pixelated; 5 mm × 5 mm × 1 mm per detector | **256 pixels** per detector; 40 mm × 40 mm × 5 mm; **2.46 mm pixel pitch** |
| **Spectral resolution** | **~1 keV at 14 keV** | **~7 keV at 60 keV** |
| **Cooling** | CdTe diode + FET cooled by adjustable TEC | (CZT operated near ambient) |
| **Overall energy resolution spec** | ~1 keV over 10–40 keV (CdTe) | — |

- **Combined band: 8–150 keV.** CdTe covers the softer/overlap region (8–70 keV); the large-area CZT covers the harder band (20–150 keV) where photons are rare.
- **Field of view: 6° × 6°** via a stainless-steel collimator (limits off-axis/background while keeping mass within budget). Full-Sun, non-imaging.
- **Front-end:** in-house low-noise **digital pulse processing (DPP)** with slow + fast triangular shapers; fast shaper peaking time **500 ns** (timing trigger); pile-up and saturated-pulse rejection logic and valid-peak detection in FPGA.

### 2.2 Time cadence and data structure

HEL1OS records **time-tagged events at the instrument clock resolution of 10 ms** in each detector, from which the standard products are binned:

| Product | Cadence | Content |
|---|---|---|
| **Event list** | **10 ms** time-tagging | All detected events per detector (table extensions); rebinnable to any cadence |
| **Light curves** | **1 s** | Count rate per detector in several **energy sub-bands** (separate table extensions) |
| **PHA spectra** | **20 s** | OGIP **Type II PHA** spectral files, one per detector |
| **Timing accuracy (HXR bursts)** | order **1 s** | Adequate for impulsive-phase timing |

Because the underlying data are 10 ms time-tagged events, **the light-curve binning can be optimized per flare** (finer for big/bright events, coarser for weak ones) — important for QPP detection and for resolving sharp impulsive spikes.

### 2.3 Count rates, pile-up, background, and contamination

- **Pile-up / saturation:** above **~50 kcps** the CdTe spectrum distorts; the team flags the spectrum as compromised when **>5% of events in higher channels are pile-up-contaminated**. DPP includes pile-up and saturated-pulse rejection. (For the brightest flares, timing/light-curve products down to ~1 s remain usable even where spectra degrade.)
- **Background sources:** high-energy particle background; explicit dependence on **South Atlantic Anomaly (SAA)** passages and latitude is referenced from the RHESSI/STIX heritage. At L1 the SAA is not crossed (it's a near-Earth phenomenon), so the *dominant* HEL1OS background is the quasi-isotropic cosmic/particle background and instrumental, not the geomagnetic SAA — but the analysis framework inherits SAA-style background-handling concepts.
- **Cosmic-ray / particle contamination handling:** events are filtered using **temporal coincidence ("bunching") in time** (not spatial bunching of hit pixels) to identify and suppress cosmic-ray-generated events; such events do **not** necessarily land in neighboring pixels. The **256-pixel CZT** enables this multiplicity/anticoincidence-style discrimination.
- **CdTe dead-time:** characterized by irradiating CdTe at different incident rates and fitting dead-time models to correct the effective area (details in Srikar et al., in prep.). CZT is described as **not** a strongly radiation-affected concern in the same way (large area, harder band).

### 2.4 How hard X-ray bursts appear in HEL1OS data

- During a flare's **impulsive phase**, HEL1OS shows **sharp, spiky bursts** in the count-rate light curves — fast rise/fast decay structures lasting seconds to a few minutes, often with **multiple spikes** and sometimes **quasi-periodic pulsations (QPPs)** (the paper notes QPP science with periods ~10–100 s, and reports detected pulsations during flares).
- The **harder channels (CZT, >25 keV)** light up only for the stronger/non-thermal events; weaker flares may appear only in the **softer CdTe channels (8–25 keV)** where thermal + early non-thermal emission overlaps.
- The **first HEL1OS light curve** was obtained at switch-on on **29 October 2023**; the **first significant flare with strong counts in both detectors was an X-class flare on 28 November 2023** (peak ~19:42 UT), confirming response across the band.
- Operationally, since commissioning HEL1OS has **detected essentially all activity reported in GOES bands, and sometimes beyond**, because its detectors reach into harder X-rays than GOES/SoLEXS.

### 2.5 Data products and format

- **Level-1:** OGIP-compliant **FITS** — Type II PHA (20 s), per-detector multi-band light curves (1 s), the 10 ms event list, plus auxiliary FITS files (good-time intervals/GTI, housekeeping).
- **Level-2:** same FITS structure as Level-1 (light curves and spectra) but science-ready/calibrated.
- ISRO provides **Python utility tools on PRADAN** for HEL1OS timing and spectral analysis.

---

## 3. Aditya-L1 mission context

| Item | Value |
|---|---|
| **Launch** | **2 September 2023** (PSLV-C57) |
| **Halo-orbit insertion at L1** | **6 January 2024** |
| **Orbit** | Halo orbit around the **Sun–Earth L1** point, ~**1.5 million km** sunward of Earth |
| **Halo period** | ~**178–180 days** (first full halo orbit completed **2 July 2024**) |
| **Station-keeping** | Periodic small maneuvers (a few per year) to maintain the halo orbit |
| **Mission life** | ~5 years (design) |
| **SoLEXS milestones** | First power-on 16 Oct 2023; aperture open / first light **13 Dec 2023**; science ops **6 Jan 2024**; PV phase Jan–Jun 2024 (caught X2.9 on 14 Dec 2023 and **X8.7 on 14 May 2024**) |
| **HEL1OS milestones** | First light curve **29 Oct 2023**; first strong flare (X-class) **28 Nov 2023**; routine operation since **Jan 2024** |

**Why L1 beats LEO for flare monitoring (pipeline-relevant):**
1. **No Earth occultations / no orbital day-night eclipses.** A LEO X-ray monitor (and even GOES to a degree) suffers periodic Earth blockages; from the L1 halo orbit the Sun is **continuously visible**, giving uninterrupted light curves — ideal for forecasting, where data gaps break the time series.
2. **No SAA dropouts** of the kind that punch holes in LEO hard-X-ray data.
3. **Stable thermal/radiation environment**, aiding gain stability.
4. **Outside the magnetosphere** → cleaner in-situ context for particle/CME correlation (other Aditya payloads).

**Data latency.** Observed data are **downlinked once per day** by default from the spacecraft; ISSDC converts raw → Level-0 (packing payload data with auxiliary data, time-correlation, spacecraft attitude), then the Payload Operations Centres (POCs at URSC) generate Level-1/Level-2. **Net effect: latency is on the order of ~1 day plus processing** — so Aditya-L1 data is well-suited to *post-hoc model training and validation and short-horizon nowcasting against the recent record*, but is **not a sub-minute real-time feed** the way operational GOES is. For a true real-time operational nowcast, GOES XRS remains the live anchor; Aditya-L1 provides the harder-band physics and an independent, eclipse-free training corpus.

---

## 4. ISSDC PRADAN portal — data organization & access

**Portal:** PRADAN (ISSDC), Aditya-L1 section (`pradan.issdc.gov.in/al1/`, with an FAQ/help at `pradan1.issdc.gov.in/al1/`). Hosts data for all **seven** payloads: VELC, SUIT, **SoLEXS**, **HEL1OS**, ASPEX, PAPA, MAG.

### 4.1 Access mechanism
- **Registration + email verification** required; access permissions are then granted by administrators (a "page does not exist or you do not have access" error means access for that dataset hasn't been granted yet — request via Contacts).
- Browse via the **"Browse and Download"** interface; downloads tracked under "My Data Downloads" (History/Statistics).
- **Bulk/"Select"** download: specify the serial number of the first row and the data volume, then "Select". **There is a per-selection size/file-count limit.**
- Session timeout **30 minutes**; a download manager / Chrome is recommended for large/high-latency transfers.

### 4.2 Levels and formats
- **Level-0:** raw payload data packed with auxiliary data (time-correlation, attitude). Internal/processing stage.
- **Level-1:** reorganized/calibrated instrument data in standard formats. For SoLEXS and HEL1OS this is **FITS** (HEL1OS: OGIP-compliant; Type II PHA + multi-band light curves + 10 ms event list + GTI/housekeeping aux files).
- **Level-2:** science-ready products; **same FITS structure as Level-1** (light curves + spectra) for the X-ray payloads.
- **Format by payload:** **FITS** for the X-ray/optical payloads (SoLEXS, HEL1OS, VELC, SUIT); **CDF** for particle payloads (ASPEX, PAPA, STEPS, SWIS); **netCDF** for the Magnetometer. → **For PS-15 we want FITS (SoLEXS + HEL1OS).**

### 4.3 Official ISRO analysis software
- **SoLEXS:** the **"Solexsloods"** package (calibration database + analysis software) is downloadable from PRADAN. SoLEXS data is publicly accessible (data from ~July 2024 onward released to the portal).
- **HEL1OS:** ISRO provides **Python utility tools** on PRADAN for timing and spectral analysis.
- General solar-X-ray tooling that interoperates: **sunkit-spex** (spectral fitting, used by the SoLEXS team), **CHIANTI** (atomic database), **astropy/SunPy** (FITS + time-series handling), and **OSPEX/XSPEC**-style fitting via OGIP files for HEL1OS.

> **Data release cadence note:** ISRO has been making periodic "Science Quality Data" releases for Aditya-L1 (e.g., second release announced via the Astronomical Society of India). Check for the latest release window before bulk-downloading, and prefer **Level-2** for science-ready light curves.

---

## 5. Flare physics in the SoLEXS/HEL1OS bands

### 5.1 Soft X-ray (SoLEXS) flare profile
A typical SXR flare light curve (the GOES-like shape SoLEXS will produce) has:
1. **Gradual rise** (minutes to tens of minutes) as chromospheric evaporation fills coronal loops with hot (~10–30 MK) plasma.
2. **Smooth, rounded peak** (the GOES class is defined at this peak).
3. **Slow, quasi-exponential decay** (minutes to hours) as the plasma cools by conduction and radiation.
This emission is **thermal**: bremsstrahlung continuum plus strong line complexes (Fe, Ca, etc.). SoLEXS resolves these lines (170 eV) and yields **T and EM** vs time.

### 5.2 Hard X-ray (HEL1OS) flare profile
The HXR light curve is **impulsive**: one or more **sharp spikes** during the rise of the SXR, lasting seconds–minutes, frequently showing **substructure and QPPs**. This is **non-thermal** bremsstrahlung from electrons accelerated (likely at the reconnection site) and precipitating into the chromosphere. The harder the channel, the spikier and earlier the signal. A high-T thermal "super-hot" component can also contribute at the low (8–15 keV CdTe) end.

### 5.3 Thermal vs non-thermal — band-by-band
| Band | Dominant emission | Instrument |
|---|---|---|
| 1–~10 keV | Thermal (lines + continuum), ~10–30 MK | SoLEXS |
| ~10–25 keV | Mixed: high-T thermal + onset of non-thermal | SoLEXS top / HEL1OS CdTe |
| ~25–150 keV | Non-thermal bremsstrahlung (accelerated electrons) | HEL1OS CdTe/CZT |

---

## 6. The Neupert effect — the math that links the two payloads

**Empirical statement (Neupert 1968):** the **hard X-ray** (or microwave) time profile tracks the **time derivative of the soft X-ray** light curve. Equivalently, the SXR light curve looks like the **time integral** of the HXR light curve.

**Physical derivation (energy/thick-target argument):**
- Non-thermal electrons deposit energy in the chromosphere at instantaneous rate **P(t)** (collisional heating), which is observationally proportional to the HXR flux: `F_HXR(t) ∝ P(t)`.
- That deposited energy drives evaporation and heats the coronal plasma whose **thermal SXR emission** measures the *accumulated* thermal energy content. To first order, neglecting cooling during the rise:

```
   E_thermal(t) ∝ ∫ P(t') dt'   ⇒   F_SXR(t) ∝ ∫₀ᵗ F_HXR(t') dt'
```

Differentiating:

```
   d/dt [ F_SXR(t) ]  ∝  F_HXR(t)            ← the Neupert effect
```

**With cooling** (more realistic), the heated plasma also loses energy with timescale τ_cool, giving a driven-decay form:

```
   dF_SXR/dt  =  c · F_HXR(t)  −  F_SXR(t) / τ_cool
```

This is exactly a leaky-integrator / first-order linear system — i.e., **F_SXR is a smoothed, lagged integral of F_HXR.** During the rise the integral term dominates (Neupert holds well); during decay the −F_SXR/τ_cool cooling term dominates (Neupert breaks down, SXR decays on its own).

**Quantitative behavior & timing (from the literature):**
- Many flares show **high cross-correlation** (r ≈ 0.9–0.95) between **dF_SXR/dt (GOES 0.5–4 Å derivative)** and **HXR 25–50 keV (RHESSI)**.
- **Timing:** the **HXR peak occurs near the steepest SXR rise**, i.e. *before* the SXR peak. Reported lags: SXR derivative tends to lead/rise first; HXR-vs-SXR peak delays of order **~1–3 minutes (approx.)** are typical, and depend on flare-loop length. Statistical surveys (e.g., Cycle 24 M/X flares) find a large majority of events are "Neupert-compliant."
- **Precursor value (forecasting gold):** because the **impulsive HXR rise leads the SXR peak by minutes**, HEL1OS gives a **few-minutes-ahead predictor** of the SXR/GOES-class peak. Independent HEL1OS science even reports **CME acceleration commencing ~6 minutes before the onset of non-thermal HXR** in ~80% of studied events — i.e., a layered precursor chain (pre-flare/precursor activity → HXR impulsive onset → SXR peak).

---

## 7. Data realities, gaps, and cross-calibration

| Issue | Reality for Aditya-L1 X-ray data | Handling in pipeline |
|---|---|---|
| **Eclipses / occultations** | **None at L1** (continuous Sun view) — a major advantage | Expect fewer gaps than LEO/GOES |
| **Station-keeping / slews** | Occasional maneuvers can cause brief pointing/data interruptions | Use GTI extensions; flag/mask non-GTI intervals |
| **Daily downlink** | Data arrive ~once/day → ~1-day+ latency | Pipeline is near-real-time at best; design for batch + nowcast, not sub-minute ops |
| **SoLEXS SDD1 saturation** | Turnover ≥ ~10⁵ cps in big flares (paralyzable) | Prefer SDD2 for M/X; apply dead-time correction; never read SDD1 turnover as a flux dip |
| **HEL1OS pile-up** | Spectra distort > ~50 kcps (CdTe); >5% high-ch contamination flag | Use 1 s light curves (robust) even when 20 s spectra degrade; honor quality flags |
| **Background / particles** | Cosmic/particle background; SAA-style framework inherited (no actual SAA at L1) | Apply event-bunching cosmic-ray rejection; subtract quiet-Sun background per channel |
| **Gain/offset drift** | Depend on detector/electronics temperatures (SoLEXS) | Use housekeeping temperatures + onboard ⁵⁵Fe/Ti lines for time-dependent calibration |
| **Low-energy ARF uncertainty** | SoLEXS ~2–2.3 keV ARF + dead-time corrections carry residual error | Treat lowest-energy bins cautiously in flux derivation |
| **Cross-calibration needed** | Yes — anchor to **GOES XRS** (and Chandrayaan-2 XSM where available) | Build GOES-band synthesis + empirical correction; ~10–15% offsets expected |
| **Data coverage window** | SoLEXS public from ~July 2024; periodic "Science Quality" releases | Plan training set around available release windows; use latest release |

---

## 8. Implications for our PS-15 pipeline

1. **Two-channel input by design.** Ingest **SoLEXS 0.1 s light curves + 1 s spectra (FITS)** and **HEL1OS 1 s multi-band light curves + 10 ms event list + 20 s PHA (FITS)**. Resample both to a common grid (e.g., **1 s and 1 min** versions): 1 s for impulsive/QPP features, 1 min for trend/forecast features.

2. **Derive a GOES-class proxy from SoLEXS** (ARF/RMF fold into 1–8 Å, or (T,EM) synthesis) and **calibrate it against GOES XRS** for both labeling (flare class) and as a training target. Keep GOES XRS as the external truth/anchor.

3. **Exploit the Neupert effect as a physics-informed feature and as the forecasting engine.** Compute **dF_SXR/dt** from SoLEXS and compare to **HEL1OS HXR**; use the HXR impulsive rise (and its lead of ~1–3 min over the SXR peak) as the **short-horizon nowcast trigger**. Features: HXR rise rate, HXR/SXR-derivative correlation, time since last HXR spike, CZT/CdTe hardness ratio, QPP presence.

4. **Respect instrument artifacts in feature engineering.** Auto-switch SoLEXS SDD1→SDD2 above saturation; flag HEL1OS pile-up; mask non-GTI intervals; ignore SDD1 turnover dips. These guards prevent the model from learning artifacts as "physics."

5. **Layered precursor chain for lead time.** Combine **pre-flare/precursor activity → HEL1OS non-thermal HXR onset → SoLEXS SXR rise/peak**. The reported ordering (e.g., CME acceleration ~6 min before HXR onset; HXR before SXR peak) suggests achievable **minutes-scale lead times**, consistent with ML nowcasting results (e.g., HOPE-style precursors giving ~18 min lead for >M5 flares in the literature).

6. **Latency-aware deployment.** Because Aditya-L1 data is ~daily-latency, treat the deliverable as (a) a **trained/validated model on the eclipse-free Aditya-L1 corpus**, and (b) an **operational nowcaster that can run live on GOES** and switch to / fuse Aditya-L1 once its data lands. Aditya-L1's continuous, SAA-free coverage is a **cleaner training set** than LEO data.

7. **Use ISRO's own tooling** (**Solexsloods** for SoLEXS calibration/analysis; **HEL1OS Python tools**; **sunkit-spex + CHIANTI**) to generate calibrated Level-2 light curves and to validate our independent FITS readers.

---

## 9. Open items / things to confirm against the latest release
- Exact **light-curve column names** and FITS extension layout for SoLEXS Level-2 (the paper specifies cadences and channels; column-level schema should be confirmed from the PRADAN product format document / Solexsloods).
- HEL1OS **number of light-curve energy sub-bands** and their exact boundaries (paper confirms "different energy sub-bands in different table extensions" and ~1 keV resolution; exact band edges to be read from the FITS headers).
- The precise **GOES-band conversion recipe / correction factor** ISRO recommends for SoLEXS (we have the cross-calibration result; confirm the official procedure).
- Current **public data window** (SoLEXS from ~July 2024; check the latest ISRO "Science Quality Data" release for the full date range now available).

---

## Sources (by name; not fabricated URLs)
- **SoLEXS instrument paper:** Shanmugam et al., "Solar Low Energy X-ray Spectrometer (SoLEXS) on Board Aditya-L1 Mission," *Solar Physics* **300**:87 (2025); companion "SoLEXS on board Aditya-L1: Ground Calibration and In-flight Performance," arXiv:2509.26292.
- **HEL1OS instrument paper:** Nandi et al., "HEL1OS — A Hard X-ray Spectrometer on Board Aditya-L1," *Solar Physics* (2025); arXiv:2512.12679.
- **Aditya-L1 mission overview:** "The Aditya-L1 Mission of ISRO," arXiv:2212.13046; ISRO mission pages ("Halo-Orbit Insertion of Aditya-L1," "Completion of First Halo Orbit").
- **ISSDC / PRADAN:** ISSDC Aditya-L1 portal and PRADAN Aditya-L1 FAQ/help pages; Astronomical Society of India announcement, "Second Release of Aditya-L1 Science Quality Data."
- **Neupert effect & flare timing:** Neupert (1968); Veronig et al., "The Neupert effect in solar flares…" (arXiv:astro-ph/0208089); "Relative timing of solar flares observed at different wavelengths" (arXiv:astro-ph/0208088); "Soft versus Hard X-rays in Solar Flares" (ApJ); "Neupert Effect: A Statistical Analysis of M and X-Class Flares During Cycle 24," *Solar Physics* (2026); Neupert effect (Wikipedia, for definition).
- **GOES flare-class definitions:** NOAA/SWPC GOES XRS convention (standard A/B/C/M/X scale, 1–8 Å peak flux).
- **ML flare forecasting context:** UFCORIN (arXiv:1507.08011); "Anticipating Solar Flares" (arXiv:2407.04567); "Advancing Solar Flare Nowcasting … Hot Onset Precursor Events" (HOPE); "Multimodal Flare Forecasting with Deep Learning" (arXiv:2410.16116).

*All quantitative instrument figures (energy ranges, areas, cadences, resolutions, dead-times, channels, gains, dates) are from the SoLEXS and HEL1OS refereed papers above. Items marked **(approx.)** are from general literature or are order-of-magnitude and should be confirmed per-flare against data.*
