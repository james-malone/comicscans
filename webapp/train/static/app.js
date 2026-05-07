"use strict";

// ============================================================
// State
// ============================================================
const S = {
  activeTab:       "training",
  trainRunId:      null,
  trainSSE:        null,
  trainHistory:    [],   // [{epoch, train_loss, train_px, val_px, is_best}]
  bestValPx:       null,
  currentRunName:  null,
  baselineHistory: null,
  baselineName:    null,
  datasetEntries:  [],
  evalTaskId:      null,
  evalResults:     null,
  evalSortCol:     "mean",
  evalSortAsc:     false,
  models:          [],
  config:          {},
};

// ============================================================
// Tab switching
// ============================================================
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + tab).classList.add("active");
    S.activeTab = tab;
    if (tab === "dataset" && S.datasetEntries.length === 0) datasetLoad();
    if (tab === "eval")    modelsListLoad();
    if (tab === "models")  modelsListLoad();
  });
});

// ============================================================
// Utility
// ============================================================
async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function fmt(n, d=2) { return n == null ? "—" : Number(n).toFixed(d); }

function publisherClass(name) {
  return name.startsWith("DS9_1996") ? "marvel" : "malibu";
}

// ============================================================
// Tab 1 — Training
// ============================================================

// Populate defaults from API on load
(async () => {
  try {
    const cfg = await api("/api/config");
    S.config = cfg;
    const defaults = await api("/api/train/defaults");
    document.getElementById("cfg-train").value   = defaults.train;
    document.getElementById("cfg-holdout").value = defaults.holdout;
    document.getElementById("cfg-comicscan-dir").value = cfg.comicscan_dir || "";
    document.getElementById("cfg-gt-file").value        = cfg.ground_truth_file || "";
  } catch(e) { console.error("Init error", e); }
  // Always load log list regardless of whether config fetch succeeded
  await prevRunsLoad();
})();

async function prevRunsLoad() {
  try {
    const logs = await api("/api/train/logs");
    const sel  = document.getElementById("prev-runs-select");
    const bsel = document.getElementById("baseline-select");
    sel.innerHTML  = '<option value="">— select a log file —</option>';
    bsel.innerHTML = '<option value="">— none —</option>';
    logs.forEach(l => {
      const short = l.filename.replace("_log.jsonl", "");
      const text  = `${short}  (${l.total_epochs} ep, best ${fmt(l.best_val_px)} px)`;
      for (const s of [sel, bsel]) {
        const o = document.createElement("option");
        o.value = l.filename;
        o.textContent = text;
        s.appendChild(o);
      }
    });
  } catch(e) {}
}

async function loadBaseline(filename) {
  if (!filename) {
    S.baselineHistory = null;
    S.baselineName    = null;
    chartDraw();
    compareUpdate();
    return;
  }
  try {
    const data = await api(`/api/train/logs/${encodeURIComponent(filename)}`);
    S.baselineHistory = data;
    S.baselineName    = filename.replace("_log.jsonl", "");
    chartDraw();
    compareUpdate();
  } catch(e) { alert("Could not load baseline: " + e.message); }
}

function compareUpdate() {
  deltaChartDraw();
  compareLogDraw();
}

async function loadPrevRun(filename) {
  if (!filename) return;
  try {
    const data = await api(`/api/train/logs/${encodeURIComponent(filename)}`);
    S.trainHistory    = data;
    S.currentRunName  = filename.replace("_log.jsonl", "");
    S.bestValPx       = data.reduce((b, e) => Math.min(b, e.val_px), Infinity);
    chartDraw();
    compareUpdate();
    document.getElementById("best-val-badge").textContent =
      `Best val: ${fmt(S.bestValPx)} px`;
    const logEl = document.getElementById("train-log");
    logEl.innerHTML = data.map(e =>
      `<div class="${e.is_best ? "log-best" : ""}">epoch ${e.epoch}/${e.total_epochs}  ` +
      `train_loss=${fmt(e.train_loss,5)}  train_px=${fmt(e.train_px,2)}  ` +
      `val_px=${fmt(e.val_px,2)}${e.is_best ? "  *" : ""}</div>`
    ).join("");
    logEl.scrollTop = logEl.scrollHeight;
    setTrainStatus("loaded", `${filename} — ${data.length} epochs`, false);
  } catch(e) { alert("Could not load log: " + e.message); }
}

function watchLogFile(filename) {
  if (!filename) return;
  S.trainHistory   = [];
  S.bestValPx      = null;
  S.currentRunName = filename.replace("_log.jsonl", "");
  document.getElementById("train-log").innerHTML = "";
  document.getElementById("progress-bar").style.width = "0%";
  document.getElementById("best-val-badge").textContent = "";
  setTrainStatus("running", `Watching ${filename}…`, true);
  document.getElementById("btn-stop").disabled = false;

  if (S.trainSSE) S.trainSSE.close();
  S.trainSSE = new EventSource(`/api/train/watch/${encodeURIComponent(filename)}`);

  S.trainSSE.addEventListener("epoch", e => {
    const d = JSON.parse(e.data);
    S.trainHistory.push(d);
    if (d.is_best) S.bestValPx = d.val_px;
    chartDraw();
    compareUpdate();
    const pct = (d.epoch / d.total_epochs * 100).toFixed(1);
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("epoch-info").textContent = `epoch ${d.epoch}/${d.total_epochs}`;
    if (S.bestValPx != null)
      document.getElementById("best-val-badge").textContent = `Best val: ${fmt(S.bestValPx)} px`;
    const logEl = document.getElementById("train-log");
    const line = document.createElement("div");
    if (d.is_best) line.className = "log-best";
    line.textContent =
      `epoch ${d.epoch}/${d.total_epochs}  ` +
      `train_loss=${fmt(d.train_loss,5)}  train_px=${fmt(d.train_px,2)}  ` +
      `val_px=${fmt(d.val_px,2)}  (${d.elapsed}s)${d.is_best ? "  *" : ""}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  });

  S.trainSSE.addEventListener("done", () => {
    S.trainSSE.close(); S.trainSSE = null;
    document.getElementById("btn-stop").disabled = true;
    setTrainStatus("done", "Training complete", false);
    prevRunsLoad();
  });

  S.trainSSE.onerror = () => {
    if (S.trainSSE) { S.trainSSE.close(); S.trainSSE = null; }
    document.getElementById("btn-stop").disabled = true;
    setTrainStatus("error", "Stream error — check server logs", false);
  };
}

async function trainStart() {
  const body = {
    train:       document.getElementById("cfg-train").value.trim(),
    holdout:     document.getElementById("cfg-holdout").value.trim(),
    epochs:      parseInt(document.getElementById("cfg-epochs").value),
    batch_size:  parseInt(document.getElementById("cfg-batch").value),
    lr:          parseFloat(document.getElementById("cfg-lr").value),
    input_size:  parseInt(document.getElementById("cfg-size").value),
    warm_restarts: parseInt(document.getElementById("cfg-wr").value),
    seed:        parseInt(document.getElementById("cfg-seed").value),
    output:      document.getElementById("cfg-output").value.trim() || null,
  };
  try {
    const r = await api("/api/train/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    S.trainRunId = r.run_id;
    S.trainHistory = [];
    S.bestValPx = null;
    document.getElementById("train-log").innerHTML = "";
    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled = false;
    setTrainStatus("running", "Running…", true);
    trainStreamOpen(r.run_id);
  } catch(e) { alert("Could not start training: " + e.message); }
}

function trainStreamOpen(runId) {
  if (S.trainSSE) S.trainSSE.close();
  S.trainSSE = new EventSource(`/api/train/${runId}/stream`);
  S.trainSSE.addEventListener("epoch", e => {
    const d = JSON.parse(e.data);
    S.trainHistory.push(d);
    if (d.is_best) S.bestValPx = d.val_px;
    chartDraw();
    const pct = (d.epoch / d.total_epochs * 100).toFixed(1);
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("epoch-info").textContent =
      `epoch ${d.epoch}/${d.total_epochs}`;
    if (S.bestValPx != null)
      document.getElementById("best-val-badge").textContent =
        `Best val: ${fmt(S.bestValPx)} px`;
    const logEl = document.getElementById("train-log");
    const line = document.createElement("div");
    if (d.is_best) line.className = "log-best";
    line.textContent =
      `epoch ${d.epoch}/${d.total_epochs}  ` +
      `train_loss=${fmt(d.train_loss,5)}  train_px=${fmt(d.train_px,2)}  ` +
      `val_px=${fmt(d.val_px,2)}  (${d.elapsed}s)${d.is_best ? "  *" : ""}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  });
  S.trainSSE.addEventListener("done", e => {
    S.trainSSE.close();
    S.trainSSE = null;
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
    setTrainStatus("done", "Training complete", false);
    prevRunsLoad();
  });
  S.trainSSE.addEventListener("error_msg", e => {
    S.trainSSE.close();
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
    setTrainStatus("error", "Error: " + JSON.parse(e.data).message, false);
  });
  S.trainSSE.onerror = () => {
    // SSE connection closed
    if (S.trainSSE) { S.trainSSE.close(); S.trainSSE = null; }
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
  };
}

async function trainStop() {
  if (!S.trainRunId) return;
  try {
    await api(`/api/train/${S.trainRunId}/stop`, {method: "POST"});
    if (S.trainSSE) { S.trainSSE.close(); S.trainSSE = null; }
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
    setTrainStatus("idle", "Stopped", false);
  } catch(e) { alert(e.message); }
}

function setTrainStatus(state, msg, running) {
  const dot = document.getElementById("status-dot");
  dot.className = "status-dot " + (running ? "running" : state === "done" ? "done" : state === "error" ? "error" : "idle");
  document.getElementById("status-text").textContent = msg;
}

// ============================================================
// Chart (canvas-based line chart — no dependencies)
// ============================================================
function chartDraw() {
  const canvas = document.getElementById("chart-canvas");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth;
  const H = 220;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const data = S.trainHistory;
  if (data.length === 0) return;

  const PAD = { top: 20, right: 80, bottom: 32, left: 52 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top  - PAD.bottom;

  const epochs    = data.map(d => d.epoch);
  const trainPx   = data.map(d => d.train_px);
  const valPx     = data.map(d => d.val_px);
  const trainLoss = data.map(d => d.train_loss);

  // Baseline overlay — val_px for epochs that overlap with current run
  const baseMap = S.baselineHistory
    ? new Map(S.baselineHistory.map(d => [d.epoch, d.val_px]))
    : null;
  const baselineValPx = baseMap ? epochs.map(ep => baseMap.get(ep) ?? null) : [];

  const maxEpoch = Math.max(...epochs);
  const maxPx    = Math.max(...trainPx, ...valPx,
    ...(baselineValPx.filter(v => v != null))) * 1.05;
  const maxLoss  = Math.max(...trainLoss) * 1.05;

  const bg   = "#0e0e0e";
  const grid = "#1e2030";
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + cH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
    const val = (maxPx * i / 4).toFixed(0);
    ctx.fillStyle = "#555"; ctx.font = "10px system-ui"; ctx.textAlign = "right";
    ctx.fillText(val + "px", PAD.left - 4, y + 3);
  }

  // X labels
  const nLabels = Math.min(10, data.length);
  ctx.fillStyle = "#555"; ctx.textAlign = "center";
  for (let i = 0; i <= nLabels; i++) {
    const ep = Math.round(maxEpoch * i / nLabels);
    const x = PAD.left + cW * ep / maxEpoch;
    ctx.fillText(ep, x, H - PAD.bottom + 14);
  }

  function xOf(ep) { return PAD.left + cW * (ep - 1) / Math.max(maxEpoch - 1, 1); }
  function yOf(v, max) { return PAD.top + cH * (1 - v / max); }

  function drawLine(series, maxVal, color, dash=[]) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dash);
    series.forEach((v, i) => {
      const x = xOf(epochs[i]);
      const y = yOf(v, maxVal);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Right axis for train_loss (scaled separately)
  ctx.fillStyle = "#555"; ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) {
    const val = (maxLoss * i / 4);
    const y = PAD.top + cH * (1 - i / 4);
    ctx.fillText(val.toFixed(5), PAD.left + cW + 4, y + 3);
  }

  drawLine(trainLoss, maxLoss, "#555", [3, 3]);  // loss — dim dashed on right scale
  drawLine(trainPx,   maxPx,   "#4a90d9");        // train_px — blue
  drawLine(valPx,     maxPx,   "#e94560");         // val_px — accent red

  // Baseline val_px overlay
  if (baseMap && baselineValPx.some(v => v != null)) {
    ctx.beginPath();
    ctx.strokeStyle = "#f5a623";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    let started = false;
    baselineValPx.forEach((v, i) => {
      if (v == null) { started = false; return; }
      const x = xOf(epochs[i]);
      const y = yOf(v, maxPx);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Best-val marker
  const bestIdx = data.findIndex(d => d.is_best && d.val_px === S.bestValPx);
  if (bestIdx >= 0) {
    const d = data[bestIdx];
    const x = xOf(d.epoch);
    const y = yOf(d.val_px, maxPx);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#4caf50"; ctx.fill();
  }

  // Legend
  const legend = [
    { color: "#4a90d9", label: "train_px" },
    { color: "#e94560", label: "val_px" },
    ...(S.baselineName ? [{ color: "#f5a623", label: S.baselineName + " val_px", dash: true }] : []),
    { color: "#555",    label: "loss (right axis)", dash: true },
  ];
  let lx = PAD.left;
  legend.forEach(l => {
    ctx.strokeStyle = l.color; ctx.lineWidth = 2;
    if (l.dash) ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(lx, PAD.top - 8); ctx.lineTo(lx + 20, PAD.top - 8); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#aaa"; ctx.textAlign = "left"; ctx.font = "11px system-ui";
    ctx.fillText(l.label, lx + 24, PAD.top - 4);
    lx += ctx.measureText(l.label).width + 50;
  });
}

window.addEventListener("resize", () => { chartDraw(); if (S.baselineHistory) deltaChartDraw(); });

// ============================================================
// Delta / Comparison
// ============================================================
function _overlapPoints() {
  if (!S.baselineHistory || !S.trainHistory.length) return [];
  const bMap = new Map(S.baselineHistory.map(d => [d.epoch, d]));
  return S.trainHistory
    .filter(d => bMap.has(d.epoch))
    .map(d => {
      const b = bMap.get(d.epoch);
      return { epoch: d.epoch, cur: d.val_px, base: b.val_px, delta: b.val_px - d.val_px };
    });
}

function deltaChartDraw() {
  const section = document.getElementById("compare-section");
  const pts = _overlapPoints();
  if (pts.length === 0) { section.style.display = "none"; return; }
  section.style.display = "";

  // Update compare section title
  const cur  = S.currentRunName  || "current";
  const base = S.baselineName    || "baseline";
  document.getElementById("compare-title").textContent =
    `Δ val_px  ·  ${base}  vs  ${cur}  ·  green = ${cur} better`;

  const canvas = document.getElementById("chart-canvas-delta");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth;
  const H = 140;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const PAD = { top: 20, right: 20, bottom: 28, left: 56 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#0e0e0e";
  ctx.fillRect(0, 0, W, H);

  const maxAbs = Math.max(...pts.map(p => Math.abs(p.delta))) * 1.1 || 1;
  const maxEp  = Math.max(...pts.map(p => p.epoch));
  const minEp  = Math.min(...pts.map(p => p.epoch));

  const zeroY = PAD.top + cH / 2;

  // Grid lines
  ctx.strokeStyle = "#1e2030"; ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i++) {
    const y = PAD.top + cH * (1 - (i / 2 + 1) / 2);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
    const val = (maxAbs * i / 2).toFixed(0);
    ctx.fillStyle = "#555"; ctx.font = "10px system-ui"; ctx.textAlign = "right";
    ctx.fillText(val + "px", PAD.left - 4, y + 3);
  }
  // Zero line emphasis
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.left, zeroY); ctx.lineTo(PAD.left + cW, zeroY); ctx.stroke();

  // X axis labels
  const nLbls = Math.min(10, pts.length);
  ctx.fillStyle = "#555"; ctx.textAlign = "center"; ctx.font = "10px system-ui";
  for (let i = 0; i <= nLbls; i++) {
    const ep = Math.round(minEp + (maxEp - minEp) * i / nLbls);
    const x = PAD.left + cW * (ep - minEp) / Math.max(maxEp - minEp, 1);
    ctx.fillText(ep, x, H - PAD.bottom + 12);
  }

  function xOf(ep) { return PAD.left + cW * (ep - minEp) / Math.max(maxEp - minEp, 1); }
  function yOfD(v) { return PAD.top + cH * (1 - (v / maxAbs + 1) / 2); }

  const barW = Math.max(2, (cW / Math.max(pts.length, 1)) * 0.7);
  pts.forEach(p => {
    const x = xOf(p.epoch);
    const y = yOfD(p.delta);
    ctx.fillStyle = p.delta >= 0 ? "#4caf5099" : "#e9456099";
    if (p.delta >= 0) {
      ctx.fillRect(x - barW / 2, y, barW, zeroY - y);
    } else {
      ctx.fillRect(x - barW / 2, zeroY, barW, y - zeroY);
    }
  });

  // Connect dots for readability
  ctx.beginPath();
  ctx.strokeStyle = "#aaa";
  ctx.lineWidth = 1;
  pts.forEach((p, i) => {
    const x = xOf(p.epoch);
    const y = yOfD(p.delta);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function compareLogDraw() {
  const el  = document.getElementById("compare-log");
  const pts = _overlapPoints();
  if (pts.length === 0) { el.innerHTML = ""; return; }

  const cur  = S.currentRunName || "current";
  const base = S.baselineName   || "baseline";

  let html = `<table class="compare-table">
    <thead><tr>
      <th>ep</th>
      <th>${base}</th>
      <th>${cur}</th>
      <th>Δ px</th>
    </tr></thead><tbody>`;

  pts.forEach(p => {
    const sign = p.delta >= 0 ? "+" : "";
    const cls  = p.delta >= 0 ? "delta-better" : "delta-worse";
    html += `<tr>
      <td>${p.epoch}</td>
      <td>${fmt(p.base)}</td>
      <td>${fmt(p.cur)}</td>
      <td class="${cls}">${sign}${fmt(p.delta)}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
}

// ============================================================
// Tab 2 — Dataset Explorer
// ============================================================
const DATASET_PAGE = 60;
let _datasetOffset = 0;
let _datasetTotal  = 0;
let _datasetParams = new URLSearchParams();

async function datasetLoad() {
  _datasetOffset = 0;
  S.datasetEntries = [];
  const grid = document.getElementById("dataset-grid");
  grid.innerHTML = '<div class="no-results">Loading…</div>';
  await _datasetFetch(true);
}

async function datasetLoadMore() {
  _datasetOffset += DATASET_PAGE;
  await _datasetFetch(false);
}

async function _datasetFetch(reset) {
  const dir    = document.getElementById("filter-dir").value;
  const corr   = document.getElementById("filter-corr").value;
  const search = document.getElementById("filter-search").value.trim();

  const params = new URLSearchParams();
  if (dir)    params.set("scan_dir", dir);
  if (corr)   params.set("has_correction", corr);
  if (search) params.set("search", search);
  params.set("offset", _datasetOffset);
  params.set("limit",  DATASET_PAGE);
  _datasetParams = params;

  const grid = document.getElementById("dataset-grid");

  try {
    const [result, stats] = await Promise.all([
      api("/api/dataset/entries?" + params),
      api("/api/dataset/stats"),
    ]);

    // Populate dir filter on first load
    const dirSel = document.getElementById("filter-dir");
    if (dirSel.options.length <= 1) {
      stats.dirs.sort().forEach(d => {
        const o = document.createElement("option");
        o.value = d; o.textContent = d;
        dirSel.appendChild(o);
      });
    }

    _datasetTotal = result.total;
    const shown = _datasetOffset + result.entries.length;
    document.getElementById("dataset-stats").textContent =
      `Showing ${shown} of ${result.total} entries (${stats.corrected} corrected)`;

    if (reset) {
      S.datasetEntries = result.entries;
      grid.innerHTML = "";
    } else {
      S.datasetEntries = S.datasetEntries.concat(result.entries);
    }

    if (S.datasetEntries.length === 0) {
      grid.innerHTML = '<div class="no-results">No entries match the current filters.</div>';
    } else if (reset) {
      // Render first page
      result.entries.forEach((entry, idx) => _addPageCard(grid, entry, idx));
    } else {
      // Append new cards
      const base = _datasetOffset;
      result.entries.forEach((entry, idx) => _addPageCard(grid, entry, base + idx));
    }

    // Load-more bar
    const lm = document.getElementById("dataset-loadmore");
    if (shown < result.total) {
      lm.style.display = "block";
      document.getElementById("dataset-loadmore-info").textContent =
        `${result.total - shown} more`;
    } else {
      lm.style.display = "none";
    }
  } catch(e) {
    grid.innerHTML = `<div class="no-results">Error: ${e.message}</div>`;
  }
}

function _addPageCard(grid, entry, idx) {
  const card = document.createElement("div");
  card.className = "page-card";
  card.innerHTML = `
    <div class="thumb-wrap">
      <canvas id="thumb-${idx}" width="200" height="280"></canvas>
    </div>
    <div class="card-info">
      <strong>${entry.scan_dir.split("/").pop()} #${entry.page_index}</strong>
      ${entry.has_correction ? '<span class="badge badge-train">corrected</span>' : '<span class="badge badge-holdout">auto</span>'}
    </div>`;
  card.addEventListener("click", () => pageModalOpen(entry));
  grid.appendChild(card);
  loadThumb(entry, `thumb-${idx}`, 200, 280);
}

async function loadThumb(entry, canvasId, W, H) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  try {
    const b64 = btoa(entry.filepath);
    const resp = await fetch(`/api/dataset/image/${encodeURIComponent(b64)}?max_size=400`);
    if (!resp.ok) return;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      const scale = Math.min(W / img.width, H / img.height);
      const sw = img.width * scale, sh = img.height * scale;
      const ox = (W - sw) / 2, oy = (H - sh) / 2;
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, ox, oy, sw, sh);
      drawCorners(ctx, entry.gt_corners, ox, oy, scale, "#4caf50", false);
      if (entry.det_corners)
        drawCorners(ctx, entry.det_corners, ox, oy, scale, "#e94560", true);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch(e) {}
}

function drawCorners(ctx, corners, ox, oy, scale, color, dashed) {
  if (!corners || corners.length < 4) return;
  const pts = corners.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
  pts.forEach(([x, y]) => {
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  });
}

function pageModalOpen(entry) {
  const modal = document.getElementById("page-modal");
  const content = document.getElementById("modal-content");
  content.innerHTML = `
    <div style="margin-bottom:10px">
      <strong>${entry.filepath.split("/").pop()}</strong>
      <span style="color:var(--text-muted);margin-left:8px;font-size:12px">${entry.scan_dir.split("/").pop()} — page ${entry.page_index}</span>
    </div>
    <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">
      ${entry.image_width}×${entry.image_height}px · ${entry.dpi} DPI ·
      ${entry.has_correction ? '<span style="color:var(--success)">manually corrected</span>' : '<span>auto-accepted</span>'}
    </div>
    <div class="modal-image-wrap">
      <canvas id="modal-canvas" width="600" height="840"></canvas>
    </div>`;
  modal.classList.remove("hidden");
  const b64 = btoa(entry.filepath);
  fetch(`/api/dataset/image/${encodeURIComponent(b64)}?max_size=900`)
    .then(r => r.blob())
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById("modal-canvas");
        const maxW = Math.min(800, window.innerWidth * 0.8);
        const maxH = window.innerHeight * 0.72;
        const scale = Math.min(maxW / img.width, maxH / img.height);
        const W = img.width * scale, H = img.height * scale;
        canvas.width = W; canvas.height = H;
        canvas.style.width = W + "px"; canvas.style.height = H + "px";
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, W, H);
        drawCorners(ctx, entry.gt_corners, 0, 0, scale, "#4caf50", false);
        if (entry.det_corners)
          drawCorners(ctx, entry.det_corners, 0, 0, scale, "#e94560", true);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    });
}

function modalClose() {
  document.getElementById("page-modal").classList.add("hidden");
}

// ============================================================
// Tab 3 — Eval Results
// ============================================================
async function modelsListLoad() {
  try {
    S.models = await api("/api/models/list");
    const sel = document.getElementById("eval-model-select");
    sel.innerHTML = "";
    S.models.forEach(m => {
      const o = document.createElement("option");
      o.value = m.filename;
      const best = m.val_px != null ? ` — best ${fmt(m.val_px)} px` : "";
      o.textContent = `${m.filename}${best}`;
      sel.appendChild(o);
    });
    renderModelGrid();
  } catch(e) { console.error("Models load error", e); }
}

async function evalStart() {
  const model = document.getElementById("eval-model-select").value;
  if (!model) return;
  document.getElementById("eval-spinner").style.display = "inline-block";
  document.getElementById("eval-status").textContent = "Running evaluation…";
  document.querySelector("#tab-eval .eval-body").innerHTML =
    '<div style="text-align:center;padding:40px;color:var(--text-muted)"><span class="spinner"></span></div>';
  try {
    const {task_id} = await api(`/api/eval/start?model=${encodeURIComponent(model)}`, {method:"POST"});
    S.evalTaskId = task_id;
    pollEval(task_id);
  } catch(e) {
    document.getElementById("eval-status").textContent = "Error: " + e.message;
    document.getElementById("eval-spinner").style.display = "none";
  }
}

async function pollEval(taskId) {
  const r = await api(`/api/eval/${taskId}/status`);
  if (r.status === "done") {
    document.getElementById("eval-spinner").style.display = "none";
    document.getElementById("eval-status").textContent = `Done — ${r.n_pages} pages evaluated`;
    S.evalResults = await api(`/api/eval/${taskId}/results`);
    renderEvalResults(S.evalResults);
  } else if (r.status === "error") {
    document.getElementById("eval-spinner").style.display = "none";
    document.getElementById("eval-status").textContent = "Error: " + r.error;
  } else {
    setTimeout(() => pollEval(taskId), 2000);
  }
}

function renderEvalResults(results) {
  const body = document.getElementById("eval-body");
  const perPage = results.per_page;
  const summary = results.summary;
  const perDir  = results.per_dir;

  // Summary cards
  const summaryHtml = `
    <div class="eval-summary">
      <div class="stat-card"><div class="stat-value">${fmt(summary.mean)}</div><div class="stat-label">Mean px</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(summary.median)}</div><div class="stat-label">Median px</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(summary.p95)}</div><div class="stat-label">P95 px</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(summary.max)}</div><div class="stat-label">Max px</div></div>
    </div>`;

  // Per-issue bar chart
  const maxMean = Math.max(...Object.values(perDir).map(d => d.mean));
  const barRows = Object.entries(perDir)
    .sort((a, b) => b[1].mean - a[1].mean)
    .map(([name, d]) => {
      const cls = publisherClass(name);
      const pct = (d.mean / maxMean * 100).toFixed(1);
      return `<div class="bar-row">
        <div class="bar-name">${name}</div>
        <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        <div class="bar-val">${fmt(d.mean)} px
          <span class="badge badge-${d.split === 'holdout' ? 'holdout' : 'train'}">${d.split}</span>
        </div>
      </div>`;
    }).join("");

  // Per-dir table (sortable by S.evalSortCol)
  const rows = Object.entries(perDir)
    .sort((a, b) => {
      const va = a[1][S.evalSortCol], vb = b[1][S.evalSortCol];
      return S.evalSortAsc ? va - vb : vb - va;
    })
    .map(([name, d]) => {
      const pc = publisherClass(name);
      return `<tr>
        <td>${name} <span class="badge badge-${pc}">${pc}</span></td>
        <td>${d.n}</td>
        <td>${fmt(d.mean)}</td>
        <td>${fmt(d.median)}</td>
        <td>${fmt(d.max)}</td>
        <td><span class="badge badge-${d.split === 'holdout' ? 'holdout' : 'train'}">${d.split}</span></td>
      </tr>`;
    }).join("");

  function thBtn(col, label) {
    const arrow = S.evalSortCol === col ? (S.evalSortAsc ? " ▲" : " ▼") : "";
    return `<th onclick="evalSort('${col}')">${label}${arrow}</th>`;
  }

  // Worst pages
  const worst = [...perPage].sort((a, b) => b.error_px - a.error_px).slice(0, 12);
  const worstCards = worst.map((p, i) => {
    const name = p.scan_dir.split("/").pop();
    return `<div class="worst-card">
      <div class="wc-thumb"><canvas id="worst-${i}" width="180" height="234"></canvas></div>
      <div class="wc-info">
        <div>${name} #${p.page_index}</div>
        <div class="wc-err">${fmt(p.error_px)} px</div>
      </div>
    </div>`;
  }).join("");

  body.innerHTML = summaryHtml + `
    <div id="eval-bar-wrap"><h3>Per-issue mean error</h3><div class="bar-chart">${barRows}</div></div>
    <div class="eval-table-wrap panel">
      <table class="eval-table">
        <thead><tr>
          ${thBtn("name","Issue")}${thBtn("n","Pages")}${thBtn("mean","Mean px")}${thBtn("median","Median px")}${thBtn("max","Max px")}<th>Split</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="worst-pages panel">
      <h3>Worst pages (highest error)</h3>
      <div class="worst-grid">${worstCards}</div>
    </div>`;

  // Draw worst-page thumbs
  worst.forEach((p, i) => {
    loadThumbWithPred(p, `worst-${i}`, 180, 234);
  });
}

async function loadThumbWithPred(entry, canvasId, W, H) {
  try {
    const b64 = btoa(entry.filepath);
    const resp = await fetch(`/api/dataset/image/${encodeURIComponent(b64)}?max_size=400`);
    if (!resp.ok) return;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const scale = Math.min(W / img.width, H / img.height);
      const sw = img.width * scale, sh = img.height * scale;
      const ox = (W - sw) / 2, oy = (H - sh) / 2;
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, ox, oy, sw, sh);
      drawCorners(ctx, entry.gt,   ox, oy, scale, "#4caf50", false);
      drawCorners(ctx, entry.pred, ox, oy, scale, "#e94560", true);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch(e) {}
}

function evalSort(col) {
  if (S.evalSortCol === col) {
    S.evalSortAsc = !S.evalSortAsc;
  } else {
    S.evalSortCol = col;
    S.evalSortAsc = false;
  }
  if (S.evalResults) renderEvalResults(S.evalResults);
}

// ============================================================
// Tab 4 — Model Management
// ============================================================
function renderModelGrid() {
  const grid = document.getElementById("model-grid");
  if (!S.models.length) {
    grid.innerHTML = '<div style="color:var(--text-dim)">No .pt files found in comicscan_dir.</div>';
    return;
  }
  grid.innerHTML = S.models.map(m => {
    const inEnsemble = m.in_ensemble;
    const trainDirs = (m.train_dirs || []).map(d =>
      `<span class="dir-pill">${d}</span>`).join("");
    const holdoutDirs = (m.holdout_dirs || []).map(d =>
      `<span class="dir-pill" style="opacity:0.7">${d}</span>`).join("");
    return `<div class="model-card ${inEnsemble ? "in-ensemble" : ""}">
      <div class="mc-name">${m.filename}${inEnsemble ? ' <span class="badge badge-ensemble">ensemble</span>' : ""}</div>
      <div class="mc-meta">
        <div>Type</div><div><span>${m.model_type || "—"}</span></div>
        <div>Input size</div><div><span>${m.input_size || "—"}px</span></div>
        <div>Best val</div><div><span>${m.val_px != null ? fmt(m.val_px) + " px" : "—"}</span></div>
        <div>Epoch</div><div><span>${m.epoch != null ? m.epoch : "—"}</span></div>
        <div>Seed</div><div><span>${m.seed != null ? m.seed : "—"}</span></div>
        <div>Size</div><div><span>${m.size_mb != null ? m.size_mb + " MB" : "—"}</span></div>
      </div>
      ${m.train_dirs ? `<div class="mc-dirs"><span style="color:var(--text-dim);font-size:10px">TRAIN: </span>${trainDirs}</div>` : ""}
      ${m.holdout_dirs ? `<div class="mc-dirs"><span style="color:var(--text-dim);font-size:10px">HOLDOUT: </span>${holdoutDirs}</div>` : ""}
      <div class="mc-actions">
        ${inEnsemble
          ? `<button class="btn btn-muted" onclick="ensembleRemove('${m.filename}')">Remove from ensemble</button>`
          : `<button class="btn btn-secondary" onclick="ensembleAdd('${m.filename}')">Add to ensemble</button>`}
      </div>
    </div>`;
  }).join("");
}

async function ensembleAdd(filename) {
  try {
    await api("/api/ensemble/add", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({filename})});
    await modelsListLoad();
  } catch(e) { alert(e.message); }
}

async function ensembleRemove(filename) {
  try {
    await api("/api/ensemble/remove", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({filename})});
    await modelsListLoad();
  } catch(e) { alert(e.message); }
}

async function configSave() {
  const body = {
    comicscan_dir:       document.getElementById("cfg-comicscan-dir").value.trim(),
    ground_truth_file:   document.getElementById("cfg-gt-file").value.trim(),
  };
  try {
    await api("/api/config", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const msg = document.getElementById("cfg-save-msg");
    msg.textContent = "Saved.";
    setTimeout(() => msg.textContent = "", 2000);
  } catch(e) { alert(e.message); }
}
