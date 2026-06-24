/* ─────────────────────────────────────────────────────────────────────────────
   app.js — SecureFedHE Dashboard  (all frontend logic)
   ───────────────────────────────────────────────────────────────────────────── */

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const POLL_RING_MS    = 5_000;
const POLL_METRICS_MS = 8_000;
const POLL_AUDIT_MS   = 15_000;

const FEATURE_NAMES = [
  'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
  'Insulin', 'BMI', 'Pedigree Function', 'Age',
];

// Classic Pima rows  (row 0 = diabetic, row 1 = healthy)
const EXAMPLES = {
  diabetic: [6, 148, 72, 35, 0, 33.6, 0.627, 50],
  healthy:  [1,  85, 66, 29, 0, 26.6, 0.351, 31],
};

// ─── STATE ────────────────────────────────────────────────────────────────────
const S = {
  config:         null,
  ringNodes:      [],
  metrics:        { rounds: [], accuracy: [], current_round: 0, total_rounds: 20 },
  distribution:   [],
  lastPrediction: null,
  lastFeatures:   null,
  charts:         { accuracy: null, distribution: null },
};

// ─── UTILITY ──────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { const j = await r.json(); detail = j.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

const $ = id => document.getElementById(id);

function setStatus(text, state) {
  // state: 'ok' | 'training' | 'idle' | 'error'
  $('txt-status').textContent = text;
  $('dot-status').className   = `dot dot--${state}`;
  $('pill-status').className  = `pill pill--${state}`;
}

// ─── RING TOPOLOGY SVG ────────────────────────────────────────────────────────
function renderRing(nodes, totalRounds) {
  const W = 380, H = 310;
  const cx = W / 2, cy = H / 2 - 4;
  const RING_R = 116, NODE_R = 26;
  const n = nodes.length;

  const onlineCount = nodes.filter(nd => nd.online).length;
  $('badge-nodes').textContent = `${onlineCount}/${n} online`;

  // Pentagon positions
  const pos = nodes.map((_, i) => {
    const ang = (-90 + i * (360 / n)) * (Math.PI / 180);
    return { x: cx + RING_R * Math.cos(ang), y: cy + RING_R * Math.sin(ang) };
  });

  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" class="ring-svg">
<defs>
  <marker id="arr" viewBox="0 0 8 8" refX="8" refY="4"
    markerWidth="4.5" markerHeight="4.5" orient="auto">
    <path d="M0,0 L8,4 L0,8 Z" fill="#00c9a7"/>
  </marker>
  <marker id="arr-off" viewBox="0 0 8 8" refX="8" refY="4"
    markerWidth="4.5" markerHeight="4.5" orient="auto">
    <path d="M0,0 L8,4 L0,8 Z" fill="#1e3a52"/>
  </marker>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="3.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>`;

  // ── Edges (ring connections with animated flow) ────────────────────────────
  for (let i = 0; i < n; i++) {
    const a  = pos[i], b = pos[(i + 1) % n];
    const dx = b.x - a.x, dy = b.y - a.y;
    const d  = Math.sqrt(dx * dx + dy * dy);
    const pad = NODE_R + 5, ap = NODE_R + 13;
    const sx = a.x + (dx / d) * pad,  sy = a.y + (dy / d) * pad;
    const ex = b.x - (dx / d) * ap,   ey = b.y - (dy / d) * ap;
    const active = nodes[i].online && nodes[(i + 1) % n].online;

    svg += `<line
      x1="${sx.toFixed(1)}" y1="${sy.toFixed(1)}"
      x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}"
      stroke="${active ? '#00c9a7' : '#1e3a52'}"
      stroke-width="${active ? 2 : 1}"
      stroke-dasharray="${active ? '7 3' : '4 6'}"
      class="${active ? 'ring-edge ring-edge--active' : 'ring-edge'}"
      marker-end="url(#${active ? 'arr' : 'arr-off'})"/>`;
  }

  // ── Nodes ─────────────────────────────────────────────────────────────────
  nodes.forEach((node, i) => {
    const { x, y } = pos[i];
    const isMaster  = i === 0;
    const col = node.online
      ? (node.zkp_ready ? '#00e09e' : '#ffd166')
      : '#ff4d6d';

    // Pulse halo (online only)
    if (node.online) {
      svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}"
        r="${NODE_R + 11}" fill="${col}" opacity="0.07" class="pulse"/>`;
    }

    // Circle body
    svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}"
      r="${NODE_R}" fill="#162840"
      stroke="${col}" stroke-width="${isMaster ? 2.8 : 2}"
      ${node.online ? 'filter="url(#glow)"' : ''}/>`;

    // Node ID (centred, large)
    svg += `<text x="${x.toFixed(1)}" y="${(y + 1.5).toFixed(1)}"
      text-anchor="middle" dominant-baseline="middle"
      class="nid" fill="${col}">${i}</text>`;

    // Round number (small, below ID)
    if (node.online && node.round > 0) {
      svg += `<text x="${x.toFixed(1)}" y="${(y + 15).toFixed(1)}"
        text-anchor="middle" class="nround" fill="#6b8fa8">R${node.round}</text>`;
    }

    // External label
    const ang = (-90 + i * (360 / n)) * (Math.PI / 180);
    const lr  = RING_R + 52;
    const lx  = cx + lr * Math.cos(ang);
    const ly  = cy + lr * Math.sin(ang);

    // ≤2 words for the label
    const words = node.name.split(' ');
    const label = words.length > 2 ? words[0] + '\u00A0' + words[1] : node.name;
    svg += `<text x="${lx.toFixed(1)}" y="${(ly - 7).toFixed(1)}"
      text-anchor="middle" class="nlabel" fill="#c8dce9">${label}</text>`;

    const stateText = node.online
      ? (node.zkp_ready ? '◉ ZKP' : '◎ Init')
      : '✕ Offline';
    svg += `<text x="${lx.toFixed(1)}" y="${(ly + 8).toFixed(1)}"
      text-anchor="middle" class="nstate" fill="${col}">${stateText}</text>`;

    // MASTER badge beneath node 0
    if (isMaster) {
      svg += `
        <rect x="${(x - 22).toFixed(1)}" y="${(y + NODE_R + 5).toFixed(1)}"
          width="44" height="14" rx="3" fill="#00c9a7" opacity="0.13"/>
        <text x="${x.toFixed(1)}" y="${(y + NODE_R + 15).toFixed(1)}"
          text-anchor="middle" class="master-tag" fill="#00c9a7">MASTER</text>`;
    }
  });

  svg += '</svg>';
  $('ring-wrap').innerHTML = svg;

  // Legend below SVG
  $('ring-legend').innerHTML = nodes.map(nd => {
    const col = nd.online
      ? (nd.zkp_ready ? '#00e09e' : '#ffd166')
      : '#ff4d6d';
    return `<div class="legend-item">
      <span class="legend-dot" style="background:${col}"></span>
      <span class="legend-name">${nd.name}</span>
      <span class="legend-ip mono">${nd.ip}:${nd.port}</span>
    </div>`;
  }).join('');
}

// ─── ACCURACY CHART ───────────────────────────────────────────────────────────
function initAccuracyChart() {
  const ctx  = $('acc-chart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 270);
  grad.addColorStop(0,  'rgba(0,201,167,0.25)');
  grad.addColorStop(1,  'rgba(0,201,167,0.00)');

  S.charts.accuracy = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label:               'Accuracy (%)',
        data:                [],
        borderColor:         '#00c9a7',
        backgroundColor:     grad,
        borderWidth:         2.5,
        pointRadius:         4,
        pointHoverRadius:    7,
        pointBackgroundColor:'#00c9a7',
        pointBorderColor:    '#0d1e2e',
        pointBorderWidth:    2,
        tension:             0.38,
        fill:                true,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation:  { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1e2e',
          borderColor:     '#1e3a52',
          borderWidth:     1,
          titleColor:      '#00c9a7',
          bodyColor:       '#c8dce9',
          padding:         10,
          callbacks: {
            title:  items => `Round ${items[0].label}`,
            label:  item  => `  Accuracy: ${item.parsed.y.toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Round', color: '#6b8fa8', font: { size: 11 } },
          ticks: { color: '#6b8fa8', font: { family: "'Space Mono'", size: 10 } },
          grid:  { color: 'rgba(30,58,82,0.55)' },
        },
        y: {
          min: 0, max: 100,
          title: { display: true, text: 'Accuracy (%)', color: '#6b8fa8', font: { size: 11 } },
          ticks: {
            color:    '#6b8fa8',
            font:     { family: "'Space Mono'", size: 10 },
            callback: v => v + '%',
          },
          grid: { color: 'rgba(30,58,82,0.55)' },
        },
      },
    },
  });
}

function updateAccuracyChart(metrics) {
  S.metrics = metrics;
  const { rounds, accuracy, current_round, total_rounds, latest_accuracy } = metrics;

  $('txt-round').textContent = `${current_round}/${total_rounds}`;
  if (latest_accuracy != null) {
    $('txt-acc').textContent = latest_accuracy.toFixed(1) + '%';
  }

  if (!rounds.length) {
    $('acc-chart-wrap').style.display = 'none';
    $('acc-empty').style.display      = 'flex';
    return;
  }
  $('acc-empty').style.display      = 'none';
  $('acc-chart-wrap').style.display = 'block';

  const c = S.charts.accuracy;
  c.data.labels              = rounds;
  c.data.datasets[0].data   = accuracy;
  c.update('none');
}

// ─── DISTRIBUTION CHART ────────────────────────────────────────────────────────
function renderDistChart(hospitals) {
  S.distribution = hospitals;

  // Shorten hospital names for axis labels
  const labels = hospitals.map(h =>
    h.name.replace(' Hospital', '').replace(' Clinic', '').replace(' Centre', '').replace(' Medical', '')
  );

  S.charts.distribution = new Chart($('dist-chart').getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label:           'Non-Diabetic',
          data:            hospitals.map(h => h.healthy),
          backgroundColor: 'rgba(0,224,158,0.72)',
          borderColor:     '#00e09e',
          borderWidth:     1,
          borderRadius:    5,
        },
        {
          label:           'Diabetic',
          data:            hospitals.map(h => h.diabetic),
          backgroundColor: 'rgba(255,77,109,0.72)',
          borderColor:     '#ff4d6d',
          borderWidth:     1,
          borderRadius:    5,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#c8dce9',
            font:  { family: "'Inter'", size: 12 },
            boxWidth: 12, boxHeight: 12, padding: 20,
          },
        },
        tooltip: {
          backgroundColor: '#0d1e2e',
          borderColor:     '#1e3a52',
          borderWidth:     1,
          titleColor:      '#c8dce9',
          bodyColor:       '#c8dce9',
          callbacks: {
            afterLabel(ctx) {
              const h   = hospitals[ctx.dataIndex];
              const pct = ctx.datasetIndex === 0
                ? (100 - h.diabetic_pct).toFixed(1)
                : h.diabetic_pct.toFixed(1);
              return `  (${pct}% of hospital total)`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#6b8fa8', font: { size: 11 } },
          grid:  { color: 'rgba(30,58,82,0.4)' },
        },
        y: {
          title: { display: true, text: 'Patients', color: '#6b8fa8', font: { size: 11 } },
          ticks: { color: '#6b8fa8', font: { family: "'Space Mono'", size: 10 } },
          grid:  { color: 'rgba(30,58,82,0.4)' },
        },
      },
    },
  });
}

// ─── PREDICTION FORM ──────────────────────────────────────────────────────────
function loadExample(type) {
  EXAMPLES[type].forEach((v, i) => {
    const el = $('f' + i);
    if (el) el.value = v;
  });
}

async function runPrediction() {
  const btn = $('btn-predict');

  // Collect + validate
  const features = [];
  let firstBad = null;
  for (let i = 0; i < 8; i++) {
    const el = $('f' + i);
    const v  = parseFloat(el?.value);
    if (!el || el.value === '' || isNaN(v)) {
      el?.classList.add('input--error');
      firstBad = firstBad ?? el;
    } else {
      el.classList.remove('input--error');
      features.push(v);
    }
  }
  if (firstBad) { firstBad.focus(); return; }

  btn.disabled    = true;
  btn.textContent = 'Running…';
  $('predict-result').innerHTML =
    '<div class="result-loading"><span class="spinner"></span> Analysing vitals…</div>';

  try {
    const result = await api('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ features }),
    });
    S.lastPrediction = result;
    S.lastFeatures   = features;
    renderPrediction(result);
  } catch (err) {
    $('predict-result').innerHTML =
      `<div class="result-error">⚠ Prediction failed: ${err.message}</div>`;
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Run Prediction';
  }
}

function renderPrediction(result) {
  const isDiabetic = result.prediction === 1;
  const conf       = result.confidence.toFixed(1);
  const pDia       = result.prob_diabetic.toFixed(1);
  const pHlt       = result.prob_healthy.toFixed(1);

  // Feature risk rows (top 3)
  const riskRows = (result.top_features || []).slice(0, 3).map(([name, info]) => {
    const score = info.risk_score || 0;
    const pct   = Math.min(Math.abs(score) * 120, 100).toFixed(0);
    const up    = score > 0;
    return `<div class="risk-row">
      <span class="risk-name">${name}</span>
      <div class="risk-bar-track">
        <div class="risk-bar-fill ${up ? 'risk--up' : 'risk--down'}" style="width:${pct}%"></div>
      </div>
      <span class="risk-score ${up ? 'risk--up' : 'risk--down'}">${up ? '↑' : '↓'} ${Math.abs(score).toFixed(3)}</span>
    </div>`;
  }).join('');

  $('predict-result').innerHTML = `
    <div class="result-inner">
      <div class="result-badge ${isDiabetic ? 'badge--diabetic' : 'badge--healthy'}">
        ${isDiabetic ? '⚠ DIABETIC' : '✓ NOT DIABETIC'}
      </div>

      <div class="result-conf">
        Confidence <span class="mono">${conf}%</span>
      </div>

      <div class="prob-wrap">
        <div class="prob-bar">
          <div class="prob-seg prob--healthy" style="width:${pHlt}%"></div>
          <div class="prob-seg prob--diabetic" style="width:${pDia}%"></div>
        </div>
        <div class="prob-labels">
          <span class="prob-lbl prob-lbl--healthy">Healthy ${pHlt}%</span>
          <span class="prob-lbl prob-lbl--diabetic">Diabetic ${pDia}%</span>
        </div>
      </div>

      <div class="risk-section">
        <p class="risk-title">Top Risk Factors</p>
        ${riskRows || '<p class="dim">No feature data available.</p>'}
      </div>

      <button class="btn btn--explain" id="btn-explain" onclick="explainPrediction()">
        ✦ Explain with Claude AI
      </button>
      <div id="expl-wrap"></div>
    </div>`;
}

// ─── CLAUDE EXPLANATION ───────────────────────────────────────────────────────
async function explainPrediction() {
  if (!S.lastPrediction) return;
  const btn  = $('btn-explain');
  const wrap = $('expl-wrap');

  btn.disabled    = true;
  btn.textContent = '✦ Generating…';
  wrap.innerHTML  =
    '<div class="expl-loading"><span class="spinner"></span> Claude is thinking…</div>';

  try {
    const { explanation } = await api('/api/explain', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ features: S.lastFeatures, ...S.lastPrediction }),
    });

    wrap.innerHTML = `
      <div class="expl-card">
        <div class="expl-hdr">
          <span class="expl-icon">✦</span>
          <span>Claude AI Explanation</span>
        </div>
        <p class="expl-body">${explanation.replace(/\n/g, '<br>')}</p>
        <p class="expl-disclaimer">
          ⚕ This is an AI screening tool only — not a medical diagnosis.
          Please consult a qualified physician for any health concerns.
        </p>
      </div>`;

    btn.textContent = '✦ Re-explain';
    btn.disabled    = false;

  } catch (err) {
    const needsKey = err.message.toLowerCase().includes('key') ||
                     err.message.toLowerCase().includes('configured');
    wrap.innerHTML = `<div class="expl-error">
      ${needsKey
        ? `⚠ Claude API key not set.<br>Add <code>"claude_api_key": "sk-ant-…"</code>
           under <code>"dashboard"</code> in <code>config.json</code>.`
        : `⚠ ${err.message}`}
    </div>`;
    btn.disabled    = false;
    btn.textContent = '✦ Retry';
  }
}

// ─── AUDIT LOG ────────────────────────────────────────────────────────────────
const LEVEL_CLS = { INFO: 'log--info', WARNING: 'log--warn', ERROR: 'log--err', DEBUG: 'log--dbg', RAW: 'log--raw' };

async function refreshAudit() {
  try {
    const { entries, count } = await api('/api/audit?limit=60');
    $('audit-count').textContent = `${count} entries`;

    if (!entries.length) {
      $('audit-log').innerHTML = '<div class="empty-state">No log entries yet.</div>';
      return;
    }

    $('audit-log').innerHTML = entries.map(e => {
      const lvl  = e.level || 'INFO';
      const cls  = LEVEL_CLS[lvl] || 'log--info';
      const time = (e.time || '').split(',')[0];               // drop milliseconds
      const msg  = (e.msg || '').replace(/^"|"$/g, '');        // strip JSON outer quotes
      return `<div class="log-row ${cls}">
        <span class="log-time mono">${time}</span>
        <span class="log-lvl">${lvl}</span>
        <span class="log-msg">${msg}</span>
      </div>`;
    }).join('');
  } catch (err) {
    console.warn('Audit poll error:', err.message);
  }
}

// ─── STATUS BAR ───────────────────────────────────────────────────────────────
function updateStatus(ring, metrics) {
  const online  = ring.nodes.filter(n => n.online).length;
  const total   = ring.nodes.length;
  const round   = metrics.current_round || 0;
  const totalR  = metrics.total_rounds  || S.config?.rounds || 20;

  $('txt-round').textContent = `${round}/${totalR}`;

  if (online === 0)              setStatus('All nodes offline', 'error');
  else if (round >= totalR && round > 0) setStatus(`Training complete — ${online}/${total} online`, 'ok');
  else if (round > 0)            setStatus(`Training · round ${round}/${totalR}`, 'training');
  else                           setStatus(`${online}/${total} nodes online · waiting`, 'idle');
}

// ─── POLLING ─────────────────────────────────────────────────────────────────
async function pollRing() {
  try {
    const ring = await api('/api/ring');
    S.ringNodes = ring.nodes;
    renderRing(ring.nodes, ring.total_rounds);
    updateStatus(ring, S.metrics);
  } catch {
    setStatus('Dashboard backend unreachable', 'error');
  }
}

async function pollMetrics() {
  try {
    const metrics = await api('/api/metrics');
    updateAccuracyChart(metrics);
    updateStatus({ nodes: S.ringNodes }, metrics);
  } catch (err) {
    console.warn('Metrics poll error:', err.message);
  }
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
async function init() {
  setStatus('Connecting…', 'idle');
  try {
    S.config = await api('/api/config');
    $('txt-eps').textContent = S.config.epsilon;

    initAccuracyChart();

    const [ring, metrics, dist] = await Promise.all([
      api('/api/ring'),
      api('/api/metrics'),
      api('/api/distribution'),
    ]);

    S.ringNodes    = ring.nodes;
    S.distribution = dist.hospitals;

    renderRing(ring.nodes, ring.total_rounds);
    updateAccuracyChart(metrics);
    updateStatus(ring, metrics);
    renderDistChart(dist.hospitals);
    await refreshAudit();

    // Start polling loops
    setInterval(pollRing,     POLL_RING_MS);
    setInterval(pollMetrics,  POLL_METRICS_MS);
    setInterval(refreshAudit, POLL_AUDIT_MS);

  } catch (err) {
    setStatus('Cannot reach dashboard backend', 'error');
    console.error('[SecureFedHE] init error:', err);
  }
}

window.addEventListener('DOMContentLoaded', init);
