// Chronos-AEGIS mission console: vanilla JS, no build step, no CDN
// dependency. Every function here calls the real backend/app.py API --
// there is no mock data or client-side fabrication anywhere in this file.

const state = {
  scenarios: [],
  selected: null,
};

async function checkHealth() {
  const dot = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const body = await res.json();
    dot.className = "dot ok";
    text.textContent = `backend ${body.status}`;
  } catch (err) {
    dot.className = "dot bad";
    text.textContent = "backend unreachable";
  }
}

function outcomeBadgeClass(outcome) {
  if (outcome === "success") return "badge outcome-success";
  if (outcome === "delay") return "badge outcome-delay";
  return "badge outcome-failure";
}

function renderScenarios() {
  const grid = document.getElementById("scenario-grid");
  grid.innerHTML = "";
  for (const spec of state.scenarios) {
    const card = document.createElement("div");
    card.className = "scenario-card";
    card.dataset.name = spec.name;
    card.innerHTML = `
      <div class="name mono">${spec.name}</div>
      <p class="desc">${spec.description}</p>
      <div class="meta">
        <span class="${outcomeBadgeClass(spec.ground_truth_outcome)}">${spec.ground_truth_outcome}</span>
        <span class="badge">${spec.intervention_nodes.length} intervention node(s)</span>
        <span class="badge">${spec.query_times.length} query points</span>
      </div>
    `;
    card.addEventListener("click", () => selectScenario(spec.name));
    grid.appendChild(card);
  }
}

function selectScenario(name) {
  state.selected = name;
  document.querySelectorAll(".scenario-card").forEach((el) => {
    el.classList.toggle("selected", el.dataset.name === name);
  });
  document.getElementById("run-hint").textContent = "Configure a seed and run it live.";
  document.getElementById("run-controls").classList.remove("hidden");
  document.getElementById("selected-name").textContent = name;
  document.getElementById("run-error").classList.add("hidden");
  document.getElementById("run-output").classList.add("hidden");
}

async function loadScenarios() {
  const grid = document.getElementById("scenario-grid");
  try {
    const res = await fetch("/scenarios");
    if (!res.ok) throw new Error(`status ${res.status}`);
    state.scenarios = await res.json();
    renderScenarios();
  } catch (err) {
    grid.innerHTML = `<p class="loading">Could not load scenarios: ${err.message}</p>`;
  }
}

function drawChart(records) {
  const canvas = document.getElementById("trust-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padding = { top: 16, right: 16, bottom: 26, left: 36 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  const times = records.map((r) => r.timestamp);
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const tSpan = tMax - tMin || 1;

  // axes
  ctx.strokeStyle = "#1f2937";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.lineTo(padding.left + plotW, padding.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#8b97a8";
  ctx.font = "10px sans-serif";
  ctx.fillText("1.0", 4, padding.top + 4);
  ctx.fillText("0.0", 4, padding.top + plotH);

  const x = (t) => padding.left + ((t - tMin) / tSpan) * plotW;
  const y = (v) => padding.top + (1 - Math.max(0, Math.min(1, v))) * plotH;

  function drawSeries(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    records.forEach((r, i) => {
      const px = x(r.timestamp);
      const py = y(r[key]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.fillStyle = color;
    records.forEach((r) => {
      ctx.beginPath();
      ctx.arc(x(r.timestamp), y(r[key]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawSeries("trust_mean", "#4fd1c5");
  drawSeries("p_success", "#f6ad55");

  ctx.fillStyle = "#8b97a8";
  records.forEach((r) => {
    ctx.fillText(String(r.timestamp), x(r.timestamp) - 6, padding.top + plotH + 16);
  });
}

function renderTable(records) {
  const tbody = document.querySelector("#trajectory-table tbody");
  tbody.innerHTML = "";
  for (const r of records) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${r.timestamp}</td>
      <td>${r.mission_phase}</td>
      <td class="mono">${r.trust_mean.toFixed(4)}</td>
      <td class="mono">${r.p_success.toFixed(4)}</td>
      <td class="mono">${r.expected_delay.toFixed(4)}</td>
      <td class="mono">${r.cyber_impact.toFixed(4)}</td>
      <td class="mono">${r.mdri_value.toFixed(4)}</td>
      <td>${r.mission_outcome ?? "&mdash;"}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function runScenario() {
  const name = state.selected;
  const seed = Number(document.getElementById("seed-input").value) || 0;
  const button = document.getElementById("run-button");
  const errorBox = document.getElementById("run-error");
  const output = document.getElementById("run-output");

  errorBox.classList.add("hidden");
  output.classList.add("hidden");
  button.disabled = true;
  button.textContent = "Running…";

  try {
    const res = await fetch(`/scenarios/${encodeURIComponent(name)}/run?seed=${seed}`, {
      method: "POST",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `status ${res.status}`);
    }
    const records = await res.json();
    const last = records[records.length - 1];

    document.getElementById("outcome-value").textContent = last.mission_outcome ?? "unknown";
    document.getElementById("completed-value").textContent = String(last.mission_completed);
    document.getElementById("points-value").textContent = String(records.length);

    drawChart(records);
    renderTable(records);
    output.classList.remove("hidden");
  } catch (err) {
    errorBox.textContent = `Run failed: ${err.message}`;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Run scenario";
  }
}

document.getElementById("run-button").addEventListener("click", runScenario);

checkHealth();
loadScenarios();
setInterval(checkHealth, 15000);
