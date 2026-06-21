/*
 * Aditya FlareCast dashboard client (ARCHITECTURE.md Section 8).
 *
 * Self-contained, dependency-free vanilla JS. It:
 *   - polls the real O(1) /api/* endpoints (latest, alert, forecast,
 *     catalogue, health),
 *   - optionally subscribes to /api/stream (Server-Sent Events) when present
 *     to animate the light curves live,
 *   - draws the dual light-curve plot and the forecast gauge on <canvas>,
 *   - raises a red ALERT BANNER on a nowcast onset / forecast trigger,
 *   - renders the recent-flares catalogue table,
 * and DEGRADES GRACEFULLY when any endpoint is missing (404 / network error /
 * no SSE) so it runs fully offline against a partially-populated API.
 *
 * The endpoint paths + JSON shapes mirror flarecast/api/routes.py and sse.py.
 * No external CDN. See vendor/README.md for adding Plotly when online.
 */
(function () {
  "use strict";

  // ----------------------------------------------------------------------
  // Config + canonical GOES A-X bands (kept in sync with styles.css).
  // ----------------------------------------------------------------------
  var API_BASE = "/api";
  var POLL_MS = 3000; // hot-read poll cadence
  var MAX_POINTS = 600; // ring length for the light-curve history
  var SOFT_STREAM = "solexs-sxr-long";
  var HARD_STREAM = "hel1os-hxr-8-30keV";
  var SOFT_BG_WM2 = 1e-8; // quiet-Sun floor (matches synth background)

  // GOES classes: [letter, lower-flux-bound W/m^2, color]. A < 1e-7 ... X >= 1e-4.
  var CLASS_BANDS = [
    { letter: "A", lo: 1e-9, hi: 1e-7, color: "#3a6b35" },
    { letter: "B", lo: 1e-7, hi: 1e-6, color: "#4f8f3f" },
    { letter: "C", lo: 1e-6, hi: 1e-5, color: "#d8b832" },
    { letter: "M", lo: 1e-5, hi: 1e-4, color: "#e8732c" },
    { letter: "X", lo: 1e-4, hi: 1e-2, color: "#e23b3b" }
  ];
  var SOFT_COLOR = "#ffcc44";
  var HARD_COLOR = "#6fe0c8";
  var LOG_MIN = -9; // log10 W/m^2 plot floor (A0)
  var LOG_MAX = -2; // log10 W/m^2 plot ceiling (above X100)

  // ----------------------------------------------------------------------
  // State.
  // ----------------------------------------------------------------------
  var state = {
    times: [], // epoch seconds (shared x-axis)
    soft: [], // W/m^2
    hard: [], // counts/s
    softEvent: [], // bool in-event flags (soft)
    lastForecast: null,
    lastAlert: null,
    horizonMin: 30,
    sse: null,
    sseOk: false,
    online: false,
    source: ""
  };

  // ----------------------------------------------------------------------
  // Small DOM + format helpers.
  // ----------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function fmtFlux(v) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toExponential(2);
  }
  function fmtCounts(v) {
    if (v == null || isNaN(v)) return "—";
    return Math.round(Number(v)).toLocaleString();
  }
  function fmtPct(v) {
    if (v == null || isNaN(v)) return "—";
    return (100 * Number(v)).toFixed(0) + "%";
  }
  function fmtClock(t) {
    if (t == null || isNaN(t)) return "—";
    // synthetic times are seconds-from-start; show HH:MM:SS modulo a day so the
    // table is readable whether t is epoch or relative.
    var s = Math.floor(Number(t)) % 86400;
    if (s < 0) s += 86400;
    var hh = String(Math.floor(s / 3600)).padStart(2, "0");
    var mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    var ss = String(s % 60).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }
  function classLetter(cls) {
    if (!cls) return null;
    var c = String(cls).trim().toUpperCase().charAt(0);
    return "ABCMX".indexOf(c) >= 0 ? c : null;
  }
  function classColor(cls) {
    var L = classLetter(cls);
    if (!L) return "#5a6488";
    for (var i = 0; i < CLASS_BANDS.length; i++) {
      if (CLASS_BANDS[i].letter === L) return CLASS_BANDS[i].color;
    }
    return "#5a6488";
  }

  // Fetch JSON, returning null on any error / non-OK (graceful degradation).
  function getJSON(path) {
    return fetch(API_BASE + path, { headers: { Accept: "application/json" } })
      .then(function (r) {
        if (!r.ok) return null;
        return r.json();
      })
      .catch(function () { return null; });
  }

  // ----------------------------------------------------------------------
  // Connection status pill.
  // ----------------------------------------------------------------------
  function setConn(stateName, label) {
    var dot = $("conn-dot");
    dot.className = "dot dot-" + stateName;
    $("conn-label").textContent = label;
  }

  // ----------------------------------------------------------------------
  // Light-curve ring buffer.
  // ----------------------------------------------------------------------
  function pushPoint(t, soft, hard, softEv) {
    // Avoid duplicating an identical trailing timestamp (poll + SSE overlap).
    var n = state.times.length;
    if (n > 0 && t != null && state.times[n - 1] === t) {
      if (soft != null) state.soft[n - 1] = soft;
      if (hard != null) state.hard[n - 1] = hard;
      if (softEv != null) state.softEvent[n - 1] = softEv;
      return;
    }
    state.times.push(t == null ? n : t);
    state.soft.push(soft == null ? NaN : soft);
    state.hard.push(hard == null ? NaN : hard);
    state.softEvent.push(!!softEv);
    if (state.times.length > MAX_POINTS) {
      state.times.shift();
      state.soft.shift();
      state.hard.shift();
      state.softEvent.shift();
    }
  }

  // ----------------------------------------------------------------------
  // Canvas: dual light-curve plot.
  // ----------------------------------------------------------------------
  function drawLightCurves() {
    var cv = $("lc-canvas");
    var ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    var padL = 64, padR = 60, padT = 14, padB = 26;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    ctx.clearRect(0, 0, W, H);

    // y (log soft) mapping.
    function ySoft(v) {
      var lv = Math.log10(Math.max(v, 1e-12));
      var frac = (lv - LOG_MIN) / (LOG_MAX - LOG_MIN);
      frac = Math.max(0, Math.min(1, frac));
      return padT + (1 - frac) * plotH;
    }

    // GOES class colour bands (shaded horizontal strips).
    for (var i = 0; i < CLASS_BANDS.length; i++) {
      var b = CLASS_BANDS[i];
      var yTop = ySoft(b.hi);
      var yBot = ySoft(b.lo);
      ctx.fillStyle = hexA(b.color, 0.13);
      ctx.fillRect(padL, yTop, plotW, yBot - yTop);
      // class label at the band's left edge.
      ctx.fillStyle = hexA(b.color, 0.9);
      ctx.font = "11px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillText(b.letter, 6, (yTop + yBot) / 2);
    }

    // log gridlines + left axis labels (decades).
    ctx.strokeStyle = "#2a3354";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#97a0b8";
    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (var p = LOG_MIN; p <= LOG_MAX; p++) {
      var y = ySoft(Math.pow(10, p));
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.globalAlpha = 0.4;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText("1e" + p, 22, y);
    }

    // hard-band linear range (right axis).
    var hMax = 1;
    for (var j = 0; j < state.hard.length; j++) {
      if (!isNaN(state.hard[j]) && state.hard[j] > hMax) hMax = state.hard[j];
    }
    hMax *= 1.1;
    function yHard(v) {
      var frac = v / hMax;
      frac = Math.max(0, Math.min(1, frac));
      return padT + (1 - frac) * plotH;
    }

    var n = state.times.length;
    if (n < 2) {
      ctx.fillStyle = "#97a0b8";
      ctx.font = "13px system-ui, sans-serif";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("waiting for data…", padL + 8, padT + 20);
      drawRightAxis(ctx, padL, padR, padT, plotW, plotH, W, hMax, yHard);
      return;
    }

    function xAt(idx) { return padL + (idx / (n - 1)) * plotW; }

    // hard curve (drawn first, underneath).
    ctx.strokeStyle = HARD_COLOR;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    var started = false;
    for (var k = 0; k < n; k++) {
      var hv = state.hard[k];
      if (isNaN(hv)) { started = false; continue; }
      var hx = xAt(k), hy = yHard(hv);
      if (!started) { ctx.moveTo(hx, hy); started = true; }
      else ctx.lineTo(hx, hy);
    }
    ctx.stroke();

    // soft curve (log).
    ctx.strokeStyle = SOFT_COLOR;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    started = false;
    for (var m = 0; m < n; m++) {
      var sv = state.soft[m];
      if (isNaN(sv)) { started = false; continue; }
      var sx = xAt(m), sy = ySoft(sv);
      if (!started) { ctx.moveTo(sx, sy); started = true; }
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // shade in-event (flare-active) spans on the soft band.
    ctx.fillStyle = hexA(SOFT_COLOR, 0.10);
    var spanStart = -1;
    for (var e = 0; e <= n; e++) {
      var active = e < n && state.softEvent[e];
      if (active && spanStart < 0) spanStart = e;
      if (!active && spanStart >= 0) {
        ctx.fillRect(xAt(spanStart), padT, Math.max(1, xAt(e - 1) - xAt(spanStart)), plotH);
        spanStart = -1;
      }
    }

    drawRightAxis(ctx, padL, padR, padT, plotW, plotH, W, hMax, yHard);
  }

  function drawRightAxis(ctx, padL, padR, padT, plotW, plotH, W, hMax, yHard) {
    ctx.fillStyle = HARD_COLOR;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    var ticks = 4;
    for (var i = 0; i <= ticks; i++) {
      var val = (hMax * i) / ticks;
      var y = yHard(val);
      ctx.fillText(Math.round(val).toString(), W - padR + 6, y);
    }
  }

  // hex + alpha -> rgba() string.
  function hexA(hex, a) {
    var h = hex.replace("#", "");
    var r = parseInt(h.substring(0, 2), 16);
    var g = parseInt(h.substring(2, 4), 16);
    var b = parseInt(h.substring(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
  }

  // ----------------------------------------------------------------------
  // Canvas: forecast probability gauge (semicircular).
  // ----------------------------------------------------------------------
  function drawGauge(prob) {
    var cv = $("gauge-canvas");
    var ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    var cx = W / 2, cy = H - 18, R = Math.min(W / 2 - 16, H - 34);
    var start = Math.PI, end = 2 * Math.PI; // upper semicircle, left->right

    // background arc.
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#2a3354";
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, end);
    ctx.stroke();

    if (prob != null && !isNaN(prob)) {
      var p = Math.max(0, Math.min(1, prob));
      var ang = start + p * (end - start);
      // colour ramps green -> amber -> red with probability.
      var col = p < 0.33 ? "#46c46a" : p < 0.66 ? "#e8a93b" : "#e23b3b";
      ctx.strokeStyle = col;
      ctx.beginPath();
      ctx.arc(cx, cy, R, start, ang);
      ctx.stroke();

      // needle.
      ctx.strokeStyle = "#e8ecf6";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * 0.86 * Math.cos(ang), cy + R * 0.86 * Math.sin(ang));
      ctx.stroke();

      ctx.fillStyle = col;
      ctx.font = "bold 30px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText((100 * p).toFixed(0) + "%", cx, cy - 8);
    } else {
      ctx.fillStyle = "#97a0b8";
      ctx.font = "16px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("no forecast", cx, cy - 10);
    }
    ctx.textAlign = "left";

    // 0 / 1 end labels.
    ctx.fillStyle = "#97a0b8";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("0", cx - R - 6, cy + 6);
    ctx.fillText("1", cx + R - 2, cy + 6);
  }

  // ----------------------------------------------------------------------
  // Alert banner.
  // ----------------------------------------------------------------------
  function showAlert(alert) {
    var banner = $("alert-banner");
    if (!alert) {
      banner.classList.add("hidden");
      return;
    }
    var kind = (alert.kind || "nowcast").toLowerCase();
    var cls = alert.goes_class || (alert.severity ? alert.severity : null);
    var L = classLetter(cls);
    banner.classList.remove("hidden", "kind-forecast", "bg-A", "bg-B", "bg-C", "bg-M", "bg-X");

    var label = kind === "forecast" ? "FORECAST TRIGGERED" : "FLARE NOWCAST";
    var clsTxt = cls ? (" — class " + cls) : "";
    $("alert-text").textContent = label + clsTxt;

    var bits = [];
    if (alert.detectors && alert.detectors.length) bits.push(alert.detectors.join("+"));
    if (alert.band) bits.push(alert.band + " band");
    if (alert.peak_flux != null) bits.push(fmtFlux(alert.peak_flux) + " W/m²");
    if (alert.onset_time != null) bits.push("onset " + fmtClock(alert.onset_time));
    $("alert-meta").textContent = bits.join(" · ");

    // Colour: forecast uses the warm forecast tint; nowcast uses class colour.
    if (kind === "forecast") {
      banner.classList.add("kind-forecast");
    } else if (L) {
      banner.classList.add("bg-" + L);
    }
  }

  // ----------------------------------------------------------------------
  // Catalogue table.
  // ----------------------------------------------------------------------
  function renderCatalogue(events) {
    var tb = $("cat-tbody");
    $("cat-count").textContent = (events ? events.length : 0) + " events";
    if (!events || !events.length) {
      tb.innerHTML = '<tr class="empty-row"><td colspan="8">no catalogued flares yet</td></tr>';
      return;
    }
    var rows = events.map(function (ev) {
      var flags = ev.flags || {};
      var soft = !!(flags.soft || (ev.soft && ev.soft.detected));
      var hard = !!(flags.hard || (ev.hard && ev.hard.detected));
      var neu = !!flags.neupert_consistent;
      var cls = ev.goes_class || "—";
      var L = classLetter(cls);
      var clsHtml = L
        ? '<span class="cls-cell bg-' + L + '">' + esc(cls) + "</span>"
        : esc(cls);
      var conf = ev.confidence != null ? Number(ev.confidence).toFixed(2) : "—";
      return (
        "<tr>" +
        "<td>" + fmtClock(ev.t_start) + "</td>" +
        "<td>" + fmtClock(ev.t_peak) + "</td>" +
        "<td>" + fmtClock(ev.t_end) + "</td>" +
        "<td>" + clsHtml + "</td>" +
        "<td>" + conf + "</td>" +
        "<td>" + flag(soft) + "</td>" +
        "<td>" + flag(hard) + "</td>" +
        "<td>" + flag(neu) + "</td>" +
        "</tr>"
      );
    });
    tb.innerHTML = rows.join("");
  }
  function flag(b) {
    return b ? '<span class="flag-yes">✓</span>' : '<span class="flag-no">–</span>';
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ----------------------------------------------------------------------
  // Latest readouts + class band key.
  // ----------------------------------------------------------------------
  function renderLatest(soft, hard, forecast) {
    if (soft) {
      var sv = soft.value;
      $("soft-flux").textContent = fmtFlux(sv);
      var above = Math.max(0, (sv || 0) - SOFT_BG_WM2);
      var cls = soft.cls || goesClass(above);
      var pill = $("goes-class");
      pill.textContent = cls || "—";
      pill.className = "stat-value class-pill";
      var L = classLetter(cls);
      if (L) pill.classList.add("bg-" + L);
    }
    if (hard) $("hard-counts").textContent = fmtCounts(hard.value);
    if (forecast && forecast.data_quality != null) {
      $("data-quality").textContent = fmtPct(forecast.data_quality);
    }
  }

  // Closed-form GOES class from flux above background (mirrors classify_flux).
  function goesClass(flux) {
    if (!flux || flux <= 0) return null;
    var L = "A", lo = 1e-9;
    for (var i = 0; i < CLASS_BANDS.length; i++) {
      if (flux >= CLASS_BANDS[i].lo) { L = CLASS_BANDS[i].letter; lo = CLASS_BANDS[i].lo; }
    }
    var mant = flux / lo;
    return L + (Math.round(mant * 10) / 10).toFixed(1);
  }

  function buildBandKey() {
    var el = $("bandkey");
    el.innerHTML = CLASS_BANDS.map(function (b) {
      return '<span class="bg-' + b.letter + '">' + b.letter + "</span>";
    }).join("");
  }

  // ----------------------------------------------------------------------
  // Forecast readout (gauge + text), horizon-aware via p_curve.
  // ----------------------------------------------------------------------
  function renderForecast(fc) {
    state.lastForecast = fc;
    if (!fc) {
      drawGauge(null);
      $("prob-value").textContent = "—";
      $("lead-value").textContent = "—";
      $("model-value").textContent = "—";
      return;
    }
    var p = pickProb(fc, state.horizonMin);
    drawGauge(p);
    $("prob-value").textContent = fmtPct(p);
    $("threshold-value").textContent = "≥" + (fc.class_threshold || "C");
    $("model-value").textContent = fc.model || "—";

    var lt = fc.lead_time_min;
    $("lead-value").textContent =
      lt != null && !isNaN(lt) ? Number(lt).toFixed(0) + " min" : "pending";
  }

  // Choose the probability for the selected horizon from p_curve, else p_flare.
  function pickProb(fc, horizon) {
    if (fc.p_curve && fc.p_curve[String(horizon)] != null) {
      return Number(fc.p_curve[String(horizon)]);
    }
    if (fc.p_curve) {
      // nearest available horizon key.
      var keys = Object.keys(fc.p_curve).map(Number).sort(function (a, b) { return a - b; });
      if (keys.length) {
        var best = keys[0], bd = Math.abs(keys[0] - horizon);
        for (var i = 1; i < keys.length; i++) {
          var d = Math.abs(keys[i] - horizon);
          if (d < bd) { bd = d; best = keys[i]; }
        }
        return Number(fc.p_curve[String(best)]);
      }
    }
    return fc.p_flare != null ? Number(fc.p_flare) : null;
  }

  // ----------------------------------------------------------------------
  // Polling loop (hot reads) + health probe.
  // ----------------------------------------------------------------------
  function pollOnce() {
    return Promise.all([
      getJSON("/latest?stream=" + encodeURIComponent(SOFT_STREAM)),
      getJSON("/latest?stream=" + encodeURIComponent(HARD_STREAM)),
      getJSON("/alert"),
      getJSON("/forecast"),
      getJSON("/catalogue?limit=25"),
      getJSON("/health")
    ]).then(function (res) {
      var soft = res[0], hard = res[1], alertRec = res[2];
      var fcRec = res[3], catRec = res[4], health = res[5];

      var anyOk = soft || hard || alertRec || fcRec || catRec || health;
      state.online = !!anyOk;
      if (!anyOk) {
        setConn("bad", "API unreachable");
        return;
      }
      if (!state.sseOk) setConn("ok", "connected (polling)");

      // data source badge from health.
      if (health && health.streams) {
        $("data-source").textContent = health.n_events + " events · " + health.n_streams + " streams";
      }

      // Only push a new light-curve point from polling when SSE is NOT driving
      // the chart (avoid double-feeding the ring).
      if (!state.sseOk && (soft || hard)) {
        var t = soft && soft.t != null ? soft.t : (hard && hard.t != null ? hard.t : null);
        pushPoint(
          t,
          soft ? soft.value : null,
          hard ? hard.value : null,
          soft && soft.meta ? !!soft.meta.in_event : false
        );
        drawLightCurves();
      }

      renderLatest(soft, hard, fcRec);

      // forecast (unwrap {forecast: null}).
      var fc = fcRec && fcRec.forecast === null ? null : fcRec;
      renderForecast(fc);

      // alert (unwrap {alert: null}); banner persists until cleared by API.
      var al = alertRec && alertRec.alert === null ? null : alertRec;
      state.lastAlert = al;
      showAlert(al);

      // catalogue.
      if (catRec && catRec.events) renderCatalogue(catRec.events);
    });
  }

  function startPolling() {
    pollOnce();
    setInterval(pollOnce, POLL_MS);
  }

  // ----------------------------------------------------------------------
  // SSE live transport (optional; animates the light curves when present).
  // ----------------------------------------------------------------------
  function startSSE() {
    if (typeof EventSource === "undefined") return; // older browsers -> poll only
    var es;
    try {
      es = new EventSource(API_BASE + "/stream?max_ticks=0");
    } catch (e) {
      return;
    }
    state.sse = es;

    es.addEventListener("flux", function (ev) {
      var d = safeParse(ev.data);
      if (!d) return;
      state.sseOk = true;
      setConn("ok", "live (SSE)");
      pushPoint(
        d.t,
        typeof d.soft === "number" ? d.soft : null,
        typeof d.hard === "number" ? d.hard : null,
        !!d.soft_in_event
      );
      // light-weight: redraw on a rAF tick so a fast replay stays smooth.
      scheduleDraw();
    });

    es.addEventListener("alert", function (ev) {
      var d = safeParse(ev.data);
      if (!d) return;
      state.lastAlert = d;
      showAlert(d);
    });

    es.addEventListener("end", function () {
      // replay finished; keep the last frame, fall back to polling for freshness.
      state.sseOk = false;
      setConn("ok", "connected (polling)");
    });

    es.onerror = function () {
      // SSE missing / dropped -> degrade silently to polling.
      state.sseOk = false;
      if (state.online) setConn("ok", "connected (polling)");
      try { es.close(); } catch (e) {}
    };
  }

  var drawPending = false;
  function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(function () {
      drawPending = false;
      drawLightCurves();
    });
  }

  function safeParse(s) {
    try { return JSON.parse(s); } catch (e) { return null; }
  }

  // ----------------------------------------------------------------------
  // Wire up + boot.
  // ----------------------------------------------------------------------
  function init() {
    $("api-base-label").textContent = API_BASE;
    buildBandKey();
    drawLightCurves();
    drawGauge(null);

    $("horizon-select").addEventListener("change", function (e) {
      state.horizonMin = parseInt(e.target.value, 10) || 30;
      renderForecast(state.lastForecast);
    });

    // Redraw on resize so the canvas stays crisp at the CSS width.
    window.addEventListener("resize", function () {
      scheduleDraw();
    });

    setConn("unknown", "connecting…");
    startPolling();
    startSSE();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
