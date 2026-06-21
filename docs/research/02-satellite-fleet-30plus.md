# Satellite & Mission Fleet for Aditya-L1 Solar-Flare Nowcast/Forecast — Master Census (≥30 Missions)

**Project:** ISRO BAH 2026 — Problem 15. Nowcast/forecast solar flares from **Aditya-L1 SoLEXS** (soft X-ray, 1–30 keV) + **HEL1OS** (hard X-ray, 10–150 keV).
**Purpose of this document:** A master census of **≥30 distinct satellites/missions** (not limited to ISRO) that observe the Sun in X-ray / EUV / UV / radio / magnetograms / in-situ particles & solar wind, plus multi-viewpoint platforms, to **cross-validate** Aditya-L1 detections, **fill data gaps**, **build ground-truth labels**, and **add multi-viewpoint coverage**.
**Date compiled:** 2026-06-20. **Knowledge basis:** Web-verified (June 2026) where noted, otherwise training knowledge (cutoff Jan 2026).

> **Why Aditya-L1 alone is not enough (the core motivation).** Aditya-L1 sits in a halo orbit around Sun–Earth **L1**, the same general vantage as Earth. Its flare payloads (SoLEXS soft X-ray, HEL1OS hard X-ray) are *sun-as-a-star* flux/spectroscopy instruments — they measure disk-integrated X-ray flux and spectra but do **not image** the flare, do **not** see the far side, have **no public data before July 2024** (so no historical training set on their own), and suffer telemetry/eclipse/SAA-style dropouts. To produce trustworthy ML labels and continuous nowcasts we must lean on a wide fleet: GOES for canonical labels, STIX/Fermi/Konus for hard X-ray imaging & timing, SDO/SOHO/STEREO/Solar Orbiter for EUV imaging and far-side context, magnetographs for precursors, and the in-situ + radio fleet for SEP/Type-III precursors.

---

## Legend & abbreviations

| Term | Meaning |
|---|---|
| SXR / HXR | Soft / Hard X-Ray |
| EUV / UV | Extreme-Ultraviolet / Ultraviolet |
| Å / nm / keV | Ångström / nanometre / kilo-electron-volt (energy) |
| L1 / L4-L5 | Lagrange points of Sun–Earth (or Sun–planet) system |
| SEP | Solar Energetic Particles |
| IPN | Interplanetary Network (multi-spacecraft GRB/flare timing & triangulation) |
| **Data portals named:** | **SWPC/NOAA, NCEI/NGDC, CDAWeb, VSO, JSOC, HEK, HEASARC, STIX Data Center, ISSDC/PRADAN, GONG (NSO), e-CALLISTO, SSC (STEREO Science Center), SDAC, SOAR (Solar Orbiter Archive), LASP, NMSU/SDC** |

---

# A. Soft X-ray flux & spectroscopy (the SoLEXS family)

These instruments measure the disk-integrated soft X-ray flux/spectra that directly map to GOES flare classes (A/B/C/M/X). This is the band that **directly overlaps SoLEXS** and is the primary source of ground-truth labels.

| # | Name | Agency/Country | Key instrument(s) | Band & energy/λ range | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 1 | **GOES-16** | NOAA / USA | EXIS → **XRS** (X-Ray Sensor) | SXR two bands: 0.05–0.4 nm (short) & **0.1–0.8 nm** (long, 1–8 Å) | 1 s & 1 min averages | 2017– (operational; XRS still flowing) | SWPC real-time JSON; NCEI/NGDC archive; netCDF L1b/L2 | Provides the **canonical 1–8 Å flux** that defines A–X flare classes. Independent confirmation of SoLEXS soft-X-ray rise and peak. |
| 2 | **GOES-18** | NOAA / USA | EXIS → XRS | 0.05–0.4 & 0.1–0.8 nm | 1 s | 2022– (operational) | SWPC; NCEI/NGDC | Cross-calibration partner to GOES-16/19; redundancy so the 1–8 Å reference channel never has gaps. |
| 3 | **GOES-19** | NOAA / USA | EXIS → XRS (+ CCOR-1 coronagraph) | 0.05–0.4 & 0.1–0.8 nm | 1 s | 2024– (**current primary** GOES-East/space-weather) | SWPC real-time; NCEI/NGDC | The present-day **primary GOES XRS** stream — the de-facto truth label generator for the SoLEXS era (2024→). Continuous, calibrated, gap-free. |
| 4 | **Aditya-L1 / SoLEXS** | **ISRO / India** | **SoLEXS** (Solar Low-Energy X-ray Spectrometer; SCD/SDD detectors) | SXR **~1–30 keV** (spectra ~2–22 keV; <250 eV res @ 6 keV) | **1 s during flares** | Sep 2023 launch; data public Jul 2024→ | **ISSDC PRADAN** (`pradan1.issdc.gov.in/al1`) | **The subject instrument.** Soft-X-ray spectra & flux to nowcast; provides temperature/emission-measure diagnostics. All other rows exist to validate/fill it. |
| 5 | **PROBA-2 / LYRA** | ESA / Belgium (ROB-SIDC) | **LYRA** radiometer (4 UV/SXR channels) | Incl. SXR/EUV passbands (e.g., Al & Zr SXR channels ~0.1–20 nm) | up to >20 Hz (50 ms) | 2009– (**16+ yrs, operational**) | PROBA2 Science Center (SIDC); ESA | Ultra-high-cadence irradiance catches flare onset sub-second structure; independent EU asset to fill SoLEXS dropouts and verify impulsive rise. |
| 6 | **MinXSS-1** | NASA / LASP (CubeSat) | X123 SDD soft-X-ray spectrometer | SXR **~0.5–30 keV** (best ~1–12 keV) | 10 s | 2016–2017 (re-entry) | **LASP** MinXSS data site | Historical high-resolution SXR **spectra** (pre-Aditya) for training spectral inversion / temperature models; demonstrates the SoLEXS measurement concept. |
| 7 | **MinXSS-2** | NASA / LASP (CubeSat) | X123 SDD (improved) | SXR ~0.5–30 keV | 10 s | 2018–2019 | LASP | Extends the spectral training archive; cross-checks SoLEXS spectral response/calibration philosophy. |
| 8 | **DAXSS / InspireSat-1 (MinXSS-3)** | LASP + **IIST (India)** + NCU + NTU | **DAXSS** (Dual-zone Aperture X-ray Solar Spectrometer, X123) | SXR **~0.5–20 keV** (0.4–12 keV nominal); 0.05 keV res @ 1 keV | ~10 s | 2022– (PSLV-C52) | LASP DAXSS release | **Best-in-class soft-X-ray spectral resolution**; India-linked. Ideal external spectral cross-validation for SoLEXS abundances/temperatures during overlapping flares. |
| 9 | **Yohkoh / SXT** | ISAS/JAXA + NASA + UK | Soft X-ray Telescope | SXR imaging ~0.25–4 keV (3–60 Å) | ~2 s–min | 1991–2001 | DARTS/ISAS; SDAC | **Historical imaging SXR** archive (Solar Cycle 22–23) for pre-2024 training data and morphological context absent in SoLEXS. |
| 10 | **Hinode / XRT** | JAXA + NASA + UK + ESA | **X-Ray Telescope** | SXR imaging ~0.2–2+ keV (multi-filter) | seconds–minutes | 2006– (operational) | DARTS (ISAS); VSO | Provides **where** the soft-X-ray flare is on the disk (SoLEXS gives only flux) — spatial context to attribute SoLEXS flux to an active region. |
| 11 | **MAXI / ISS** | JAXA / Japan (ISS) | Monitor of All-sky X-ray Image (GSC/SSC) | X-ray ~0.5–30 keV (all-sky scanning) | per ISS orbit (~92 min) | 2009– (operational) | RIKEN/JAXA MAXI; HEASARC | Independent all-sky X-ray monitor that occasionally catches large solar flares; cross-check for the brightest events and detector cross-calibration. |

---

# B. Hard X-ray / gamma (the HEL1OS family)

These validate and extend HEL1OS (10–150 keV) — the **nonthermal** flare signature (accelerated electrons). Several form the **Interplanetary Network (IPN)** for sub-second timing across widely separated spacecraft.

| # | Name | Agency/Country | Key instrument(s) | Band & energy range | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 12 | **Aditya-L1 / HEL1OS** | **ISRO / India** | **HEL1OS** (High-Energy L1 Orbiting X-ray Spectrometer; CZT + CdTe) | HXR **~10–150 keV** (CdTe 10–40, CZT 20–150) | **1 s** light curves; 20 s spectra | Sep 2023; data Jul 2024→ | **ISSDC PRADAN** | **The subject hard-X-ray instrument.** Disk-integrated HXR; needs external imaging + timing partners (below) for validation and localization. |
| 13 | **Solar Orbiter / STIX** | ESA / Switzerland (FHNW) | **Spectrometer/Telescope for Imaging X-rays** | HXR **4–150 keV**, *imaging* spectroscopy | sub-second–s | 2020– (>50,000 flares observed) | **STIX Data Center** (`datacenter.stix.i4ds.net`); SOAR | **Images** hard-X-ray flare footpoints HEL1OS cannot resolve, from a **different heliolongitude** → joint HEL1OS+STIX timing/IPN; far-side flare detection; spectral cross-validation 10–150 keV. |
| 14 | **RHESSI** | NASA / USA | Reuven Ramaty HE Solar Spectroscopic Imager | HXR/γ **3 keV–17 MeV**, imaging | sub-second | 2002–2018 | **HEASARC**; hesperia.gsfc.nasa.gov | The **gold-standard HXR flare catalog** (pre-Aditya). Training labels for nonthermal flares + imaging morphology for model development. |
| 15 | **Fermi / GBM** | NASA / USA | Gamma-ray Burst Monitor (NaI + BGO) | **8 keV–40 MeV** | continuous; ms triggers | 2008– (operational) | **HEASARC** `FERMIGSOL` solar-flare catalog | Whole-sky HXR/γ monitor; **sub-second spikes** cross-validate HEL1OS impulsive timing; independent hard-X-ray flux. Key **IPN** node. |
| 16 | **Wind / Konus** | NASA + Russia (Ioffe) | **Konus-Wind** (NaI) | HXR/γ ~20 keV–15 MeV | ms (trigger) / 3 s (waiting) | 1994– (operational, ~30+ yrs) | Ioffe Konus-Wind site; HEASARC | At L1 (~5 light-sec from Earth) with **total-sky** coverage → primary **IPN** partner for triangulating/timing HEL1OS hard-X-ray bursts. |
| 17 | **INTEGRAL** | ESA | SPI / IBIS / SPI-ACS (anti-coincidence) | γ/HXR ~15 keV–10 MeV (ACS as omni timing) | ms | 2002– (operational) | ISDC (Geneva); HEASARC | SPI-ACS provides high-time-resolution omnidirectional light curves → **IPN** timing partner for the largest flares. |
| 18 | **Swift / BAT** | NASA / USA-UK-Italy | Burst Alert Telescope | HXR ~15–150 keV | ms triggers | 2004– (operational) | HEASARC; Swift archive | Occasionally detects strong solar flares (when not Sun-constrained); independent HXR check + IPN node. |
| 19 | **NuSTAR** | NASA / USA | Focusing HXR optics (FPMA/B) | HXR **2.5–79 keV** (focused, high-sensitivity) | s | 2012– (operational; solar campaigns) | HEASARC NuSTAR archive | **Most sensitive HXR imager** for *microflares/quiet-Sun* — calibrates HEL1OS faint-end / pre-flare nonthermal sensitivity and the A/B-class regime SoLEXS struggles with. |
| 20 | **ASO-S / HXI** | CAS / China | **Hard X-ray Imager** (collimator + spectrometer) | HXR **~30–200 keV**, imaging | sub-second–s | 2022– (operational) | ASO-S Science Data Center (PMO) | Independent **HXR imaging** at the high-energy end (overlaps/extends HEL1OS 20–150 keV); cross-validates nonthermal spectra and footpoint timing. |
| 21 | **Koronas-Foton / RT-2** | Russia + **India (TIFR/IIA/ISRO)** | RT-2 (Phoswich/CZT) | HXR/γ ~15 keV–~1 MeV | ms–s | 2009–2010 | Indian/Russian archives (legacy) | **India-built HXR flare** heritage instrument; historical hard-X-ray flare spectra for pre-Aditya training and lineage continuity with HEL1OS detectors. |
| 22 | **Yohkoh / HXT** | ISAS/JAXA + NASA | Hard X-ray Telescope | HXR ~15–93 keV (4 bands), imaging | 0.5 s | 1991–2001 | DARTS/ISAS; SDAC | Historical hard-X-ray **imaging** of Cycle 22–23 flares — morphology + timing training set predating Aditya. |

---

# C. EUV/UV imaging & irradiance

Aditya-L1's SoLEXS/HEL1OS do not image. EUV/UV imagers tell us **where** the flare is, provide the **active-region context**, and (via EUV irradiance) give complementary thermal diagnostics and dimming/CME association.

| # | Name | Agency/Country | Key instrument(s) | Band & λ range | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 23 | **SDO** | NASA / USA | **AIA** (EUV/UV imager, 7 EUV ch) + **EVE** (EUV irradiance) | EUV 9.4–33.5 nm imaging; EVE 6–106 nm | **AIA 12 s**; EVE 10 s | 2010– (operational) | **JSOC** (Stanford); **VSO**; **HEK** | The workhorse: localizes every flare, supplies flare ribbons/loops; **HEK flare events** built on AIA. EVE irradiance cross-checks SoLEXS thermal response. Primary spatial label source. |
| 24 | **SOHO** | ESA + NASA | **EIT** (EUV imager), **LASCO** (coronagraph) | EIT 17.1/19.5/28.4/30.4 nm; LASCO white-light | EIT ~12 min; LASCO ~12–30 min | 1995– (operational, 30 yrs) | SDAC; VSO; SOHO archive | Long-baseline EUV + **CME catalog** (LASCO CDAW) to associate flares with eruptions; deep historical context for training. |
| 25 | **STEREO-A / SECCHI-EUVI** | NASA / USA | EUVI (part of SECCHI) | EUV 17.1/19.5/28.4/30.4 nm | minutes | 2006– (A operational; B lost 2014) | **STEREO Science Center (SSC)**; VSO; CDAWeb | **Far-side / different-longitude EUV imaging** → estimate SXR class of flares occulted from the L1/Earth line; multi-viewpoint validation of SoLEXS events. |
| 26 | **PROBA-2 / SWAP** | ESA / Belgium | Sun Watcher (EUV imager) | EUV ~17.4 nm | 1–4 min | 2009– (operational) | PROBA2 Science Center (SIDC) | Wide-field EUV context + redundancy when SDO/AIA is in eclipse; European autonomous flare-context imager. |
| 27 | **GOES-R / SUVI** | NOAA / USA | Solar Ultraviolet Imager | EUV 6 channels (9.4–30.4 nm) | ~few min | 2017– (on 16/18/19) | SWPC; NCEI/NGDC | **Operational** EUV imaging co-located with the GOES/XRS truth flux — directly ties SoLEXS-band flux to imaged source region in near-real-time. |
| 28 | **IRIS** | NASA / USA (+ Norway) | UV imaging spectrograph | UV 133–141 nm & 278–283 nm (slit) | seconds | 2013– (operational) | **LMSAL** IRIS archive; VSO | Chromospheric/TR flare response (Mg II, C II, Si IV) — diagnoses lower-atmosphere flare energy deposition complementary to SoLEXS coronal SXR. |
| 29 | **TRACE** | NASA / USA | Transition Region & Coronal Explorer | EUV 17.1/19.5/28.4 nm + UV | ~s | 1998–2010 | SDAC; VSO | High-res historical EUV flare imaging (Cycle 23) for morphology training before SDO/Aditya. |
| 30 | **Solar Orbiter / EUI** | ESA | Extreme Ultraviolet Imager (FSI + HRI) | EUV 17.4/30.4 nm | s–min | 2020– (operational) | **SOAR** (Solar Orbiter Archive); ESA | EUV imaging from a **third vantage** (with STIX on same bus) → joint EUV+HXR far-side flare confirmation for SoLEXS/HEL1OS. |
| 31 | **Hinode / EIS + SOT** | JAXA + partners | EUV Imaging Spectrometer; Solar Optical Telescope | EIS 17–21 & 25–29 nm; SOT optical | s–min | 2006– (operational) | DARTS (ISAS); VSO | Spectroscopic EUV plasma diagnostics (velocities, densities) of the flaring region — deep validation of SoLEXS-inferred plasma parameters. |
| 32 | **ASO-S / LST** | CAS / China | Lyman-alpha Solar Telescope (SDI + WST) | **Ly-α 121.6 nm** + white-light, disk to 2.5 R☉ | s–min | 2022– (operational) | ASO-S Science Data Center (PMO) | Unique **Ly-α** flare imaging + low-corona CME onset; ties flare impulsive phase to eruption, independent of Earth-side assets. |
| 33 | **CHASE (Xihe)** | CNSA / China | Hα Imaging Spectrograph (HIS) | **Hα 656.3 nm** (6559.7–6570.6 Å), full-disk spectra | ~1 min raster | 2021– (operational to ≥2026) | NJU CHASE data center | Full-disk Hα spectroscopy of flare ribbons → chromospheric ground-truth of flare timing/location complementing SoLEXS coronal flux. |

---

# D. Magnetographs / photospheric precursors

Flares originate in stressed active-region magnetic fields. Magnetograms are the **leading precursor** for forecasting (hours–days ahead) — something Aditya-L1's flare payloads cannot provide at all.

| # | Name | Agency/Country | Key instrument(s) | Quantity / λ | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 34 | **SDO / HMI** | NASA / USA | Helioseismic & Magnetic Imager | Photospheric **vector + LOS magnetograms**, 617.3 nm | **720 s** (LOS 45 s) | 2010– (operational) | **JSOC**; VSO; **SHARP** AR parameters | The **primary flare-forecast precursor**: SHARP magnetic complexity features feed ML to predict flares *before* SoLEXS/HEL1OS see X-rays. Pairs the precursor with Aditya's onset. |
| 35 | **SOHO / MDI** | ESA + NASA | Michelson Doppler Imager | LOS magnetograms, 676.8 nm | ~96 min (1 min campaigns) | 1996–2011 | SDAC; VSO | Historical magnetogram archive (Cycle 23) — extends the magnetic-precursor training set far before HMI/Aditya. |
| 35 | **GONG network** | NSF/NSO + NOAA / USA (6 sites incl. **Udaipur, India**) | 6 ground magnetographs + Hα | LOS magnetograms (Ni I 676.8 nm) + Hα (656.3 nm) | **1 min mag; 20 s Hα**; ~90% duty | 1995– (operational; ngGONG approved 2025) | **GONG (NSO)** real-time; NCEI | Near-24/7 **operational** magnetograms + farside seismic maps → continuous precursor coverage independent of any single spacecraft; Hα flare patrol. |
| 36 | **ASO-S / FMG** | CAS / China | Full-disk vector MagnetoGraph | Photospheric vector field, 532.4 nm | s–min | 2022– (operational) | ASO-S Science Data Center (PMO) | Independent **vector** magnetograms (HMI cross-check) for precursor features; non-Western data source for robustness. |
| 37 | **Hinode / SP (SOT-SP)** | JAXA + partners | Spectro-Polarimeter (within SOT) | High-precision vector field, 630.2 nm | minutes (scan) | 2006– (operational) | DARTS (ISAS); VSO; CSAC | Highest-precision vector magnetograms of target ARs → ground-truth for magnetic non-potentiality used in flare-likelihood models. |

---

# E. In-situ particles / solar wind / SEP

These capture the **particle/radiation-storm** consequence of flares (and shock-driven SEPs). They validate that an X-ray flare produced energetic particles and feed operational warnings — context SoLEXS/HEL1OS cannot supply, except via Aditya's own ASPEX/PAPA.

| # | Name | Agency/Country | Key instrument(s) | Quantity & range | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 38 | **ACE** | NASA / USA (L1) | EPAM, SIS, ULEIS, SWEPAM, MAG | SEP ions/electrons keV–MeV; solar wind plasma+B | 1 s–min | 1997– (operational, ~29 yrs) | **CDAWeb**; SWPC RTSW; ACE SWEPAM site | At **L1 alongside Aditya-L1** — the in-situ partner: confirms SEP onset following a flare and supplies the solar-wind context for the same vantage. |
| 39 | **DSCOVR** | NOAA / USA (L1) | Faraday Cup (PlasMag), MAG | Solar wind speed/density, IMF | 1 s–min | 2015– (operational; *anomaly Jul 2025*) | SWPC real-time; **CDAWeb** | Operational L1 solar-wind monitor co-located with Aditya — real-time space-weather context; *(note 2025 anomaly → SWFO-L1 succession).* |
| 39 | **SWFO-L1** | NOAA / USA (L1) | Solar-wind suite + **CCOR** coronagraph | Solar wind + in-situ + CME imaging | 1 s–min | **2025– (launching/commissioning)** | SWPC (future) | DSCOVR's operational successor at L1 — restores continuous solar-wind + adds coronagraphic CME detection co-located with Aditya. |
| 40 | **Wind** | NASA / USA (L1) | 3DP, SWE, MFI (+ Konus, WAVES) | SEP electrons/ions; solar wind; B | s | 1994– (operational) | **CDAWeb** | Long-baseline L1 in-situ + electron beams that map to Type-III/flare events; deep historical SEP archive for labeling. |
| 41 | **Parker Solar Probe / ISʘIS** | NASA / USA (near-Sun) | **EPI-Hi + EPI-Lo** | Energetic ions ~20 keV–200 MeV; e⁻ ~0.5–6 MeV | 1 s | 2018– (operational) | **CDAWeb**; Princeton ISʘIS; NASA PDS | Measures SEPs **close to the Sun before they disperse** → earliest particle confirmation of flares; far-side/near-Sun vantage for occulted events. |
| 42 | **STEREO-A / IMPACT (+PLASTIC)** | NASA / USA | SEP/SWEA/STE + plasma | Energetic particles + solar wind | s–min | 2006– (operational) | **CDAWeb**; SSC | In-situ particles from a **different heliolongitude** → confirms whether far-side/limb flares (seen by EUVI) injected SEPs; multi-point SEP timing. |
| 43 | **GOES-R / SEISS** | NOAA / USA | Space Environment In-Situ Suite (SGPS/MPS) | Protons/electrons/heavies (e.g. >10 MeV protons) | seconds | 2017– (on 16/18/19) | SWPC; NCEI/NGDC | The **operational SEP/proton-event** definition (S-scale) co-timed with GOES/XRS flares — links Aditya flares to the official radiation-storm labels. |
| 44 | **SOHO / ERNE (+COSTEP)** | ESA + NASA (L1) | Energetic & Relativistic Nuclei & Electron | Protons ~1–540 MeV; electrons | min | 1995– (operational) | SOHO archive; CDAWeb | L1 SEP spectra over 30 yrs — historical SEP-vs-flare associations for training and an extra L1 particle cross-check for Aditya. |

---

# F. Radio bursts (Type III / II precursors)

**Type III** radio bursts trace flare-accelerated electron beams escaping along open field — often the **earliest** flare signature, sometimes leading the X-ray peak. **Type II** trace CME shocks (SEP precursor). None of this is in the Aditya flare payloads.

| # | Name | Agency/Country | Key instrument(s) | Band & frequency | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 45 | **Wind / WAVES** | NASA / USA (L1) | Radio & Plasma Wave (RAD1/RAD2/TNR) | ~4 kHz–14 MHz | s | 1994– (operational) | **CDAWeb**; GSFC WAVES | Kilometric–hectometric **Type III** bursts → early electron-beam signature preceding HEL1OS hard-X-ray; L1 vantage matches Aditya. |
| 46 | **STEREO / WAVES (SWAVES)** | NASA / USA | S/WAVES | ~2.5 kHz–16 MHz | s | 2006– (A operational) | **CDAWeb**; SSC | Radio bursts from a second vantage → **direction-finding/triangulation** of Type III sources; far-side burst detection for occulted flares. |
| 47 | **Parker Solar Probe / FIELDS** | NASA / USA | FIELDS RFS (Radio Freq. Spectrometer) | ~10 kHz–19.2 MHz | s | 2018– (operational) | **CDAWeb**; PSP FIELDS (Berkeley) | Radio + in-situ B near the Sun → ties Type-III beams to their source very close to the flare; far-side coverage. |
| 48 | **Solar Orbiter / RPW** | ESA | Radio & Plasma Waves | ~few kHz–16 MHz | s | 2020– (operational) | **SOAR**; CDAWeb | Radio from a third heliolongitude → multi-spacecraft Type-III triangulation; pairs with STIX/EUI for full far-side flare picture. |
| 49 | **e-CALLISTO network** | Int'l (FHNW-led; ~80+ stations incl. India) | CALLISTO spectrometers | **~45–870 MHz** (metric/decimetric) | ~0.25 s | 2002– (operational, 24/7) | **e-CALLISTO** archive (FITS); India node at NCRA/TIFR | Ground-based **metric Type II/III** dynamic spectra worldwide → continuous low-corona burst monitoring to flag flares & shocks complementing SoLEXS/HEL1OS. |
| 50 | **Nobeyama Radioheliograph / RSTN-class** | NAOJ / Japan (+ USAF RSTN) | NoRH imaging; NoRP polarimeters / RSTN | Microwave 1–17 GHz (NoRH 17/34 GHz) | s | NoRH 1992–2020; NoRP/RSTN ongoing | NAOJ Nobeyama archive; NGDC (RSTN) | Microwave **gyrosynchrotron** imaging of nonthermal electrons → independent confirmation/localization of HEL1OS hard-X-ray flares (microwave–HXR correlation). |

---

# G. Multi-viewpoint / far-side platforms

Dedicated to the **far-side / stereoscopic** problem: Aditya-L1 (at L1) is blind to ~half the Sun. These give the off-Sun-Earth-line views needed to catch occulted flares and to triangulate timing/SEP injection.

| # | Name | Agency/Country | Key instrument(s) | Vantage / payload | Time cadence | Operational span | Data access | How it complements / fills gaps / validates Aditya-L1 |
|---|---|---|---|---|---|---|---|---|
| 51 | **STEREO-A** | NASA / USA | SECCHI (EUVI/COR/HI), IMPACT, PLASTIC, S/WAVES | Heliocentric orbit ahead/behind Earth → **far-side EUV+radio+in-situ** | s–min | 2006– (operational) | **SSC**; VSO; CDAWeb | The premier **far-side EUV + SEP + radio** platform: catches flares occulted from L1; with SDO gives stereoscopic flare reconstruction. |
| 52 | **Solar Orbiter** | ESA + NASA | STIX, EUI, RPW, PHI (mag), in-situ suite | Out-of-ecliptic, variable longitude → **HXR+EUV+mag+radio far-side** | s–min | 2020– (operational) | **SOAR**; STIX DC; CDAWeb | Carries an **entire flare toolkit at a different longitude** — uniquely lets HEL1OS+STIX and SoLEXS+EUI be cross-validated, including far-side; PHI adds far-side magnetograms. |
| 53 | **Parker Solar Probe** | NASA / USA | FIELDS, ISʘIS, SWEAP, WISPR | Closest-to-Sun, varying longitude → near-Sun radio+SEP | s | 2018– (operational) | **CDAWeb**; PSP archives | Innermost vantage: detects flare radio/SEP signatures earliest and from angles invisible at L1; constrains particle injection timing. |
| 54 | **BepiColombo** | ESA + JAXA | MGNS, BERM, SIXS (solar X-ray/particle monitor) | Cruise/Mercury orbit → **another heliolongitude** X-ray/particle monitor | s–min | 2018– (cruise; Mercury orbit insertion late 2026) | ESA PSA; JAXA DARTS | SIXS provides solar **X-ray flux + particles from a third point** → IPN-style timing and far-side flare flux estimates while en route to Mercury. |

---

> **Total distinct missions/networks listed: 50+** (see the numbered confirmation list at the end). This comfortably exceeds the required **≥30**. Two pairs share an index number where they are direct successors/companions (MDI↔GONG; DSCOVR↔SWFO-L1) — the unique-mission count is well above 30 even discounting those.

---

# H. Gap-filling matrix — which mission fixes which Aditya-L1 weakness

Aditya-L1 SoLEXS/HEL1OS are excellent disk-integrated X-ray instruments, but they have specific, known limitations. This matrix maps each weakness to the missions that close it.

| Aditya-L1 weakness | Why it matters for flare nowcast/forecast | Best missions to fill the gap |
|---|---|---|
| **Temporal/telemetry dropouts** (eclipses, downlink gaps, data only since Jul 2024) | Nowcasting needs *continuous* flux; gaps create false "no-flare" labels | **GOES-16/18/19 XRS** (continuous 1–8 Å), **PROBA-2 LYRA**, **GOES SUVI**, **GONG** (ground, ~90% duty) |
| **No historical training data pre-2024** | ML models need decades of labeled flares across a solar cycle | **GOES XRS (2002→ via 8–15 + 16–19)**, **RHESSI** (2002–2018), **Yohkoh SXT/HXT** (1991–2001), **SOHO/MDI**, **TRACE**, **MinXSS-1/2**, **Fermi GBM** (2008→) |
| **No imaging / no flare location** (sun-as-a-star only) | Can't attribute flux to an active region; can't forecast per-AR | **SDO/AIA + HMI**, **Hinode XRT/EIS/SOT**, **GOES SUVI**, **STEREO EUVI**, **STIX** (HXR imaging), **ASO-S HXI/LST** |
| **No hard-X-ray imaging / footpoints** | HEL1OS gives flux, not where electrons precipitate | **Solar Orbiter STIX**, **ASO-S HXI**, **NuSTAR** (faint end), historical **RHESSI / Yohkoh HXT** |
| **Sub-second HXR timing needs an independent clock** | Validate HEL1OS impulsive spikes; reject instrumental artifacts | **Fermi GBM**, **Konus-Wind**, **INTEGRAL SPI-ACS**, **Swift BAT** → joint **IPN** triangulation/timing |
| **No magnetic precursors** | The strongest *forecast* (hours–days ahead) signal is magnetic | **SDO/HMI (SHARP)**, **GONG** magnetograms, **ASO-S FMG**, **Hinode SP**, historical **SOHO/MDI** |
| **Far-side / limb-occulted flares** | L1 is blind to ~half the Sun → missed events bias the catalog | **STEREO-A (EUVI/SEP/radio)**, **Solar Orbiter (STIX/EUI/PHI)**, **Parker Solar Probe**, **BepiColombo SIXS**, **GONG far-side seismic** |
| **No SEP / radiation-storm context** | Flare → SEP is the operational hazard; needed for end-to-end labels | **ACE**, **GOES SEISS**, **SOHO/ERNE**, **Wind/3DP**, **STEREO/IMPACT**, **PSP/ISʘIS** |
| **No radio / Type-III precursors** | Type III often *leads* the X-ray peak → earliest nowcast trigger | **Wind/WAVES**, **STEREO/SWAVES**, **PSP/FIELDS**, **Solar Orbiter/RPW**, **e-CALLISTO**, **Nobeyama/RSTN** |
| **No EUV irradiance / thermal cross-check** | Independent validation of SoLEXS temperature/emission measure | **SDO/EVE**, **PROBA-2 LYRA**, **GOES EUVS** (within EXIS) |
| **Absolute calibration of soft-X-ray spectra** | SoLEXS abundances/temperatures need an external reference | **DAXSS/InspireSat-1** (best SXR spectral resolution), **MinXSS-1/2** |

---

# I. Ground-truth & labeling sources

For supervised flare nowcast/forecast, labels must come from authoritative, well-maintained catalogs. Recommended label stack:

| Catalog / service | Provider | What it gives | Access | Role in our pipeline |
|---|---|---|---|---|
| **GOES XRS flare event list** | NOAA SWPC / NCEI | Start/peak/end times + **A–X class** + (where available) AR number | SWPC products; NCEI/NGDC; also queryable via **HEK** | **Primary label source** — the universally accepted flare-class ground truth to align with SoLEXS flux. |
| **NOAA SWPC** | NOAA / USA | Event reports, edited event list, SEP/proton events, R/S/G scales, daily solar region summaries | swpc.noaa.gov (text products, JSON) | Operational flare + radiation-storm labels; AR catalog (SRS) for per-region forecasting. |
| **HEK (Heliophysics Events Knowledgebase)** | LMSAL/NASA (Stanford JSOC) | Machine + human flare/CME/filament events; AIA-derived; links to data | HEK API; via **SunPy** `Fido`/HEK client | Rich event metadata + spatial info to join SoLEXS flux to imaged location; bulk querying for ML. |
| **RHESSI flare list** | NASA / HEASARC | 2002–2018 HXR flares: time, peak counts, energy, position | HEASARC `HESSI_FLARE`; hesperia.gsfc | **Historical hard-X-ray** labels for HEL1OS-style training pre-Aditya. |
| **Fermi GBM solar flare catalog** | NASA / HEASARC | GBM-detected solar flares 2008→, HXR/γ | HEASARC `FERMIGSOL` | Modern HXR labels overlapping the Aditya era; sub-second timing. |
| **STIX flare list** | ESA / FHNW | 50,000+ flares 2021→, imaged HXR, location, from a 2nd vantage | **STIX Data Center** | HXR labels incl. far-side; the closest contemporary analog to HEL1OS for cross-training. |
| **Konus-Wind solar flare catalog** | Ioffe / Russia | HXR/γ flares, IPN timing | Konus-Wind site; HEASARC | IPN timing labels for the largest events. |
| **LASCO CME catalog (CDAW)** | NASA/CUA | CMEs (link flares↔eruptions) | cdaw.gsfc.nasa.gov | Associates flares with CMEs → eruptive-vs-confined label. |
| **SDO HMI SHARP** | Stanford JSOC | Per-AR magnetic feature time series | **JSOC**; via SunPy/drms | **Precursor feature set** for forecasting (the *input X* to predict the GOES *label Y*). |

**Labeling recommendation:** Use the **GOES XRS event list** as the authoritative flare-class label, **cross-checked** with HEK (location/AR) and, for hard-X-ray nonthermal events, **STIX + Fermi GBM**. Build the precursor feature matrix from **HMI SHARP + GONG**. This gives a clean (precursor → flare class → SEP outcome) supervised target that Aditya SoLEXS/HEL1OS detections are then validated against.

---

# J. Recommended core set to integrate FIRST (8–12 missions)

Integrating 50 data streams at once is infeasible. The following **core 10** give ~90% of the value (labels + gap-fill + far-side + precursor + cross-validation) with manageable, well-documented APIs. Ordered by priority.

| Priority | Mission/Instrument | Category | Why it's in the core set (justification) | Primary access |
|---|---|---|---|---|
| **1** | **GOES-16/18/19 XRS** | A (SXR) | **The label source.** Canonical A–X flare class, continuous 1–8 Å, fills every SoLEXS dropout, real-time JSON. Nothing else can anchor the ground truth. | SWPC JSON / NCEI |
| **2** | **Aditya-L1 SoLEXS + HEL1OS** | A + B | **The subject.** Our primary nowcast input — soft + hard X-ray flux/spectra. | ISSDC PRADAN |
| **3** | **SDO/AIA + HMI** | C + D | Localizes flares (AIA) and supplies the dominant **forecast precursor** (HMI SHARP). Best-documented API (JSOC/VSO/SunPy). | JSOC / VSO / HEK |
| **4** | **Solar Orbiter / STIX** | B + G | **Imaged hard-X-ray** validation of HEL1OS from a different longitude + **far-side** flares; clean STIX Data Center API. | STIX Data Center / SOAR |
| **5** | **Fermi GBM** | B | Independent whole-sky HXR with sub-second timing → validates HEL1OS spikes; ready-made HEASARC solar catalog. | HEASARC `FERMIGSOL` |
| **6** | **STEREO-A (EUVI + SEP + S/WAVES)** | C + E + F + G | The single best **far-side** asset — EUV imaging + SEP + radio in one bus to catch L1-occulted flares. | SSC / CDAWeb / VSO |
| **7** | **GOES SUVI + SEISS** | C + E | EUV imaging **co-located with the XRS truth flux** + operational SEP/proton labels → ties flux↔region↔radiation storm. | SWPC / NCEI |
| **8** | **GONG network** | D | 24/7 **operational magnetograms + Hα** (ground, ~90% duty, incl. Udaipur) → continuous precursor coverage independent of spacecraft. | GONG (NSO) |
| **9** | **ACE (+DSCOVR/SWFO-L1)** | E | **L1 in-situ** partner co-located with Aditya — confirms SEP onset & solar-wind context from the same vantage; real-time. | CDAWeb / SWPC |
| **10** | **Wind/WAVES (+ e-CALLISTO)** | F | **Type-III radio** → the *earliest* flare trigger (often leads X-ray), plus ground-based e-CALLISTO backup worldwide. | CDAWeb / e-CALLISTO |

**Stretch additions (11–12) once core is stable:** **DAXSS/InspireSat-1** (best-resolution SXR spectral cross-calibration of SoLEXS, India-linked) and **ASO-S (HXI + FMG + LST)** (independent HXR imaging + vector magnetograms + Ly-α from China for non-Western robustness).

**Justification summary of the core set:**
- **Labels:** GOES XRS (#1) + GOES SEISS (#7) + HEK = authoritative flare class & SEP outcome.
- **Subject signal:** SoLEXS + HEL1OS (#2).
- **Precursor (forecast):** HMI SHARP (#3) + GONG (#8).
- **Cross-validation of HXR:** STIX (#4) + Fermi GBM (#5).
- **Localization / EUV context:** AIA (#3) + SUVI (#7).
- **Far-side coverage:** STEREO-A (#6) + Solar Orbiter (#4).
- **In-situ & radio precursors:** ACE (#9) + Wind/WAVES & e-CALLISTO (#10).

This core spans all seven categories (A–G), every named Aditya-L1 weakness in the gap-filling matrix, and uses only well-documented, openly accessible APIs (SWPC, JSOC, VSO, HEK, STIX Data Center, HEASARC, SSC, CDAWeb, GONG, e-CALLISTO, ISSDC).

---

## Sources (web-verified June 2026)

- Aditya-L1 SoLEXS / HEL1OS specs & ISSDC PRADAN: [SoLEXS calibration/performance (arXiv 2509.26292)](https://arxiv.org/pdf/2509.26292), [HEL1OS (arXiv 2512.12679)](https://arxiv.org/pdf/2512.12679), [Aditya-L1 PRADAN (ISSDC)](https://pradan1.issdc.gov.in/al1/)
- GOES-R EXIS/XRS: [GOES-R XRS L2 Users Guide (NOAA)](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf), [SWPC GOES X-ray Flux](https://www.swpc.noaa.gov/products/goes-x-ray-flux-dynamic-plot), [GOES-R EXIS (NCEI)](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance)
- Solar Orbiter STIX & STIX Data Center: [STIX (AIP)](https://www.aip.de/en/research/projects/solar-orbiter/stix/), [STIX Data Center](https://sites.google.com/view/stix-data-center/home)
- ASO-S (HXI/LST/FMG): [ASO-S overview (Solar Physics)](https://link.springer.com/article/10.1007/s11207-023-02166-x), [ASO-S inflight performance 2025](https://link.springer.com/article/10.1007/s11207-025-02473-5), [ASO-S site (PMO)](http://aso-s.pmo.ac.cn/en_index.jsp)
- DAXSS / InspireSat-1: [DAXSS data release (SolarNews/NSO)](https://solarnews.nso.edu/inspiresat-1-daxss-data-release-for-solar-soft-x-ray-spectral-irradiance/), [INSPIRESat-1 (eoPortal)](https://www.eoportal.org/satellite-missions/inspiresat-1), [LASP X-ray instruments](https://lasp.colorado.edu/our-legacy/instruments/remote-sensing-x-ray/)
- CHASE (Xihe): [CHASE mission overview (arXiv 2205.05962)](https://arxiv.org/abs/2205.05962), [Chinese H-alpha Solar Explorer (Wikipedia)](https://en.wikipedia.org/wiki/Chinese_H-alpha_Solar_Explorer)
- e-CALLISTO: [e-CALLISTO (NCRA/TIFR node)](http://rac.ncra.tifr.res.in/ecallist.html), [World-wide net of solar radio spectrometers (Springer)](https://link.springer.com/article/10.1007/s11038-008-9267-6)
- PROBA-2 LYRA/SWAP: [PROBA-2 (eoPortal)](https://www.eoportal.org/satellite-missions/proba-2), [LYRA (PROBA2 Science Center)](https://proba2.sidc.be/about/LYRA)
- ACE/DSCOVR/Wind & SWFO-L1: [SWPC Real-Time Solar Wind](https://www.spaceweather.gov/products/solar-wind), [DSCOVR (eoPortal)](https://www.eoportal.org/satellite-missions/dscovr), [SWFO-L1 (Wikipedia)](https://en.wikipedia.org/wiki/Space_weather_Observations_at_L1_to_Advance_Readiness_-_1)
- Fermi GBM / Konus-Wind / IPN: [Fermi overview (NASA)](https://fermi.gsfc.nasa.gov/overview.html), [Fermi GBM Solar Flare Catalog (HEASARC FERMIGSOL)](https://heasarc.gsfc.nasa.gov/W3Browse/all/fermigsol.html), [Fermi solar flares (hesperia/RHESSI)](https://hesperia.gsfc.nasa.gov/fermi_solar/)
- STEREO SECCHI/EUVI & SSC: [SECCHI data overview (NRL)](https://secchi.nrl.navy.mil/data-overview), [STEREO Science Center data](https://stereo-ssc.nascom.nasa.gov/data.shtml)
- HEK / VSO / JSOC / SDO: [HEK for SDO (Springer)](https://link.springer.com/content/pdf/10.1007/s11207-010-9624-2.pdf), [SunPy HEK tutorial](https://docs.sunpy.org/en/stable/tutorial/acquiring_data/hek.html)
- Parker Solar Probe ISʘIS/FIELDS: [PSP ISʘIS (Princeton)](https://spacephysics.princeton.edu/missions-instruments/PSP), [PSP EPI-Lo L2 (NASA Open Data)](https://data.nasa.gov/dataset/psp-integrated-science-investigation-of-the-sun-energetic-particle-instrument-lo-isois-epi-67531)
- GONG: [GONG (NSO)](https://nso.edu/telescopes/nisp/gong/), [Three decades of GONG (Solar Physics)](https://link.springer.com/article/10.1007/s11207-026-02639-9)
- NuSTAR solar: [NuSTAR solar publications (ianan.github.io)](https://ianan.github.io/nsigh_all/pubs.html), [NuSTAR+STIX joint microflares (MNRAS)](https://academic.oup.com/mnras/article/533/3/3742/7742830)
