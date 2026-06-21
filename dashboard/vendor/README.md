# `dashboard/vendor/` — optional vendored libraries

The Aditya FlareCast dashboard (`../index.html`, `../app.js`, `../styles.css`)
is **deliberately self-contained and fully offline**: it has **no external CDN
dependency** and draws every chart (dual light curves + forecast gauge) on a
plain `<canvas>` with vanilla JavaScript. It runs with zero network access and
zero build step — open it directly, or let `flarecast serve` mount it at `/`.

This directory is therefore **empty by default**. It exists as the drop-in slot
for a richer, optional charting upgrade **when you are online** (or have
vendored the asset), as anticipated by ARCHITECTURE.md Section 8 / 11.2 ("vanilla
HTML/CSS/JS + Plotly via CDN with a vendored fallback").

## Adding Plotly for richer charts (optional, online)

[Plotly.js](https://plotly.com/javascript/) gives zoom/pan, hover tooltips, a
true dual-axis time series, and an uncertainty (±1σ) band ribbon for the fused
best-estimate light curve.

1. Download the standalone bundle and place it here:

   ```bash
   # ~3.5 MB, self-contained; pin the version + verify the integrity hash.
   curl -L -o dashboard/vendor/plotly.min.js \
     https://cdn.plot.ly/plotly-2.35.2.min.js
   ```

2. Reference it from `../index.html` *before* `app.js`:

   ```html
   <script src="vendor/plotly.min.js"></script>
   ```

3. `app.js` feature-detects Plotly: when `window.Plotly` is present it can render
   the light curves with Plotly instead of the built-in canvas renderer; when it
   is absent (the default, offline) it transparently falls back to the canvas
   charts. **No code change is required to stay offline** — the canvas path is
   the source of truth and always works.

## Why offline-first

- The judging/sandbox environment may have **no network and no credentials**;
  the canvas dashboard must work there unconditionally.
- Pinning + integrity-checking a vendored bundle (rather than a live CDN
  `<script>`) keeps the supply chain auditable (ARCHITECTURE.md Section 11.2).
- The canvas charts are tiny and dependency-free, so the dashboard loads
  instantly and mirrors the lightweight Cloudflare Pages production target.

> Keep large vendored bundles out of version control if your repo policy
> prefers it (add `dashboard/vendor/*.js` to `.gitignore`); they are an
> online-only enhancement, never required for the offline reference dashboard.
