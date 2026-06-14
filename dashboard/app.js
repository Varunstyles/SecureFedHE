/* ============================================================
   SecureFedHE · Phase 4 Dashboard — Chart Engine
   CSV parsing, chart rendering, and data analysis
   ============================================================ */

// ── Global State ─────────────────────────────────────────────
const datasets = {};
const chartInstances = {};

const COLORS = {
    baseline:  { main: '#06b6d4', bg: 'rgba(6, 182, 212, 0.1)',   border: 'rgba(6, 182, 212, 0.8)' },
    he_eps10:  { main: '#6366f1', bg: 'rgba(99, 102, 241, 0.1)',  border: 'rgba(99, 102, 241, 0.8)' },
    he_eps20:  { main: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.1)',  border: 'rgba(139, 92, 246, 0.8)' },
    he_eps50:  { main: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', border: 'rgba(245, 158, 11, 0.8)' },
    ring:      { main: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', border: 'rgba(16, 185, 129, 0.8)' },
};

const LABELS = {
    baseline:  'Ring 1 — Baseline',
    he_eps10:  'Ring 2 — HE (ε=10)',
    he_eps20:  'Ring 2 — HE (ε=20)',
    he_eps50:  'Ring 2 — HE (ε=50)',
    ring:      'Ring 3 — Decentralised',
};

// ── CSV Parser ───────────────────────────────────────────────
function parseCSV(text) {
    const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
    if (!normalized) return [];

    const lines = normalized.split('\n').map(l => l.trim()).filter(l => l);
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim().replace(/^\uFEFF/, ''));
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim().replace(/^\uFEFF/, ''));
        if (values.length < headers.length) continue;

        const row = {};
        headers.forEach((h, j) => {
            const val = values[j] ?? '';
            const parsed = parseFloat(val);
            row[h] = val === '' ? '' : Number.isNaN(parsed) ? val : parsed;
        });
        rows.push(row);
    }
    return rows;
}

// ── File Upload Handlers ─────────────────────────────────────
document.querySelectorAll('.csv-input').forEach(input => {
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        const ring = this.getAttribute('data-ring');
        const reader = new FileReader();

        reader.onload = function(ev) {
            const data = parseCSV(ev.target.result);
            const status = document.querySelector(`[data-status="${ring}"]`);
            const card = input.closest('.upload-card');

            if (data.length > 0) {
                datasets[ring] = data;
                status.textContent = `✓ Loaded ${data.length} rounds`;
                status.classList.remove('error');
                status.classList.add('success');
                card.classList.add('loaded');
            } else {
                delete datasets[ring];
                status.textContent = '⚠ No data rows found in this CSV';
                status.classList.remove('success');
                status.classList.add('error');
                card.classList.remove('loaded');
            }

            checkReady();
        };
        reader.readAsText(file);
    });
});

function checkReady() {
    const btn = document.getElementById('btnGenerateCharts');
    btn.disabled = Object.keys(datasets).length === 0;
}

// ── Demo Data Generator ──────────────────────────────────────
function loadDemoData() {
    // Generate realistic demo data based on typical SecureFedHE experiment results
    const rounds = 20;

    // Ring 1 — Baseline (from actual baseline_metrics.csv pattern)
    datasets.baseline = generateDemoRing('baseline', rounds, {
        startAcc: 0.2255, endAcc: 0.7943, startLoss: 2.15, endLoss: 0.59,
        wallTime: 350, encOverhead: 0, commBytes: 24832400,
        cpuBase: 200, ramBase: 15,
    });

    // Ring 2 — HE ε=10
    datasets.he_eps10 = generateDemoRing('selectiveHE', rounds, {
        startAcc: 0.2180, endAcc: 0.7928, startLoss: 2.18, endLoss: 0.60,
        wallTime: 370, encOverhead: 0.035, commBytes: 24853200,
        cpuBase: 210, ramBase: 18,
    });

    // Ring 2 — HE ε=20
    datasets.he_eps20 = generateDemoRing('selectiveHE', rounds, {
        startAcc: 0.2210, endAcc: 0.7935, startLoss: 2.16, endLoss: 0.595,
        wallTime: 368, encOverhead: 0.033, commBytes: 24853200,
        cpuBase: 208, ramBase: 17,
    });

    // Ring 2 — HE ε=50
    datasets.he_eps50 = generateDemoRing('selectiveHE', rounds, {
        startAcc: 0.2240, endAcc: 0.7940, startLoss: 2.15, endLoss: 0.592,
        wallTime: 365, encOverhead: 0.032, commBytes: 24853200,
        cpuBase: 205, ramBase: 16,
    });

    // Ring 3 — Decentralised Ring
    datasets.ring = generateDemoRing('ring', rounds, {
        startAcc: 0.2100, endAcc: 0.7890, startLoss: 2.20, endLoss: 0.62,
        wallTime: 400, encOverhead: 0.045, commBytes: 25100000,
        cpuBase: 220, ramBase: 20,
    });

    // Update UI
    Object.keys(LABELS).forEach(key => {
        const status = document.querySelector(`[data-status="${key}"]`);
        if (status && datasets[key]) {
            status.textContent = `✓ Demo: ${datasets[key].length} rounds`;
            status.classList.add('success');
            const card = status.closest('.upload-card');
            if (card) card.classList.add('loaded');
        }
    });

    checkReady();
    generateAllCharts();
}

function generateDemoRing(phase, rounds, cfg) {
    const data = [];
    for (let r = 1; r <= rounds; r++) {
        const progress = r / rounds;
        // Logarithmic convergence curve
        const accCurve = 1 - Math.exp(-3.5 * progress);
        const acc = cfg.startAcc + (cfg.endAcc - cfg.startAcc) * accCurve;
        const lossCurve = Math.exp(-3 * progress);
        const loss = cfg.endLoss + (cfg.startLoss - cfg.endLoss) * lossCurve;
        const trainAcc = acc * (0.98 + Math.random() * 0.04);
        const trainLoss = loss * (0.85 + Math.random() * 0.1);

        data.push({
            round_num: r,
            phase: phase,
            client_id: -1,
            train_loss: +(trainLoss).toFixed(4),
            train_acc: +(trainAcc).toFixed(4),
            eval_loss: +(loss + (Math.random() - 0.5) * 0.02).toFixed(4),
            eval_acc: +(acc + (Math.random() - 0.5) * 0.005).toFixed(4),
            comm_bytes: cfg.commBytes,
            wall_time_s: +(cfg.wallTime + (Math.random() - 0.5) * 100).toFixed(1),
            cpu_pct: +(cfg.cpuBase + (Math.random() - 0.5) * 40).toFixed(1),
            ram_mb: +(cfg.ramBase + Math.random() * 10).toFixed(1),
            enc_overhead_s: +(cfg.encOverhead + Math.random() * 0.01).toFixed(3),
        });
    }
    return data;
}

// ── Chart Generation ─────────────────────────────────────────
function generateAllCharts() {
    if (Object.keys(datasets).length === 0) return;

    document.getElementById('charts').style.display = 'block';

    // Destroy existing charts
    Object.values(chartInstances).forEach(c => c.destroy());

    // Generate each chart
    renderAccuracyChart();
    renderLossChart();
    renderPrivacyChart();
    renderOverheadChart();
    renderCommChart();
    renderResourceChart();
    renderSummaryTable();
    updateHeroStats();

    // Smooth scroll to charts
    document.getElementById('charts').scrollIntoView({ behavior: 'smooth' });
}

// ── Chart Defaults ───────────────────────────────────────────
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                color: '#94a3b8',
                font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                padding: 16,
                usePointStyle: true,
                pointStyleWidth: 12,
            },
        },
        tooltip: {
            backgroundColor: 'rgba(17, 24, 39, 0.95)',
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
            borderColor: 'rgba(99, 102, 241, 0.3)',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            titleFont: { family: "'Inter', sans-serif", weight: '600' },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks: { color: '#64748b', font: { family: "'Inter', sans-serif", size: 11 } },
        },
        y: {
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks: { color: '#64748b', font: { family: "'JetBrains Mono', monospace", size: 11 } },
        },
    },
};

function makeLineDataset(key, field, extras = {}) {
    if (!datasets[key]) return null;
    const color = COLORS[key];
    return {
        label: LABELS[key],
        data: datasets[key].map(d => d[field]),
        borderColor: color.border,
        backgroundColor: color.bg,
        borderWidth: 2.5,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: color.main,
        pointBorderColor: 'transparent',
        tension: 0.35,
        fill: false,
        ...extras,
    };
}

// ── Chart 1: Accuracy Convergence ────────────────────────────
function renderAccuracyChart() {
    const ctx = document.getElementById('chartAccuracy').getContext('2d');
    const dsets = Object.keys(LABELS)
        .map(key => makeLineDataset(key, 'eval_acc'))
        .filter(Boolean)
        .map(ds => ({
            ...ds,
            data: ds.data.map(v => +(v * 100).toFixed(2)),
        }));

    const maxRounds = Math.max(...Object.values(datasets).map(d => d.length));
    const labels = Array.from({ length: maxRounds }, (_, i) => i + 1);

    chartInstances.chartAccuracy = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: dsets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: {
                    ...CHART_DEFAULTS.scales.x,
                    title: { display: true, text: 'FL Round', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Test Accuracy (%)', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                    min: 0,
                },
            },
        },
    });
}

// ── Chart 2: Training Loss ───────────────────────────────────
function renderLossChart() {
    const ctx = document.getElementById('chartLoss').getContext('2d');
    const dsets = Object.keys(LABELS)
        .map(key => makeLineDataset(key, 'eval_loss'))
        .filter(Boolean);

    const maxRounds = Math.max(...Object.values(datasets).map(d => d.length));
    const labels = Array.from({ length: maxRounds }, (_, i) => i + 1);

    chartInstances.chartLoss = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: dsets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: {
                    ...CHART_DEFAULTS.scales.x,
                    title: { display: true, text: 'FL Round', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Evaluation Loss', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
            },
        },
    });
}

// ── Chart 3: Privacy-Utility Trade-off ───────────────────────
function renderPrivacyChart() {
    const ctx = document.getElementById('chartPrivacy').getContext('2d');

    // Collect final accuracy for each ε value
    const epsilonData = [];

    if (datasets.he_eps10) {
        const d = datasets.he_eps10;
        epsilonData.push({ eps: 10, acc: d[d.length - 1].eval_acc * 100 });
    }
    if (datasets.he_eps20) {
        const d = datasets.he_eps20;
        epsilonData.push({ eps: 20, acc: d[d.length - 1].eval_acc * 100 });
    }
    if (datasets.he_eps50) {
        const d = datasets.he_eps50;
        epsilonData.push({ eps: 50, acc: d[d.length - 1].eval_acc * 100 });
    }

    // Add baseline as reference line
    let baselineAcc = null;
    if (datasets.baseline) {
        const d = datasets.baseline;
        baselineAcc = d[d.length - 1].eval_acc * 100;
    }

    const dsets = [{
        label: 'Selective HE + DP',
        data: epsilonData.map(d => ({ x: d.eps, y: +d.acc.toFixed(2) })),
        borderColor: COLORS.he_eps10.border,
        backgroundColor: COLORS.he_eps10.main,
        pointRadius: 8,
        pointHoverRadius: 12,
        pointBackgroundColor: COLORS.he_eps10.main,
        borderWidth: 3,
        tension: 0.3,
        showLine: true,
    }];

    if (baselineAcc !== null) {
        dsets.push({
            label: 'Baseline (no privacy)',
            data: [{ x: 10, y: +baselineAcc.toFixed(2) }, { x: 50, y: +baselineAcc.toFixed(2) }],
            borderColor: COLORS.baseline.border,
            borderDash: [8, 4],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            showLine: true,
        });
    }

    chartInstances.chartPrivacy = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: dsets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: {
                    ...CHART_DEFAULTS.scales.x,
                    type: 'linear',
                    title: { display: true, text: 'Privacy Budget ε (lower = stronger privacy)', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Final Test Accuracy (%)', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
            },
        },
    });
}

// ── Chart 4: Overhead Comparison ─────────────────────────────
function renderOverheadChart() {
    const ctx = document.getElementById('chartOverhead').getContext('2d');

    const configs = [];
    const avgWallTimes = [];
    const avgEncOverheads = [];
    const barColors = [];

    Object.keys(LABELS).forEach(key => {
        if (!datasets[key]) return;
        const d = datasets[key];
        configs.push(LABELS[key]);
        avgWallTimes.push(+(d.reduce((s, r) => s + r.wall_time_s, 0) / d.length).toFixed(1));
        avgEncOverheads.push(+(d.reduce((s, r) => s + r.enc_overhead_s, 0) / d.length).toFixed(3));
        barColors.push(COLORS[key].main);
    });

    chartInstances.chartOverhead = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: configs,
            datasets: [
                {
                    label: 'Avg Wall Time (s)',
                    data: avgWallTimes,
                    backgroundColor: barColors.map(c => c + '99'),
                    borderColor: barColors,
                    borderWidth: 2,
                    borderRadius: 6,
                },
                {
                    label: 'Avg Enc Overhead (s)',
                    data: avgEncOverheads,
                    backgroundColor: barColors.map(c => c + '44'),
                    borderColor: barColors,
                    borderWidth: 2,
                    borderRadius: 6,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Time (seconds)', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
            },
        },
    });
}

// ── Chart 5: Communication Cost ──────────────────────────────
function renderCommChart() {
    const ctx = document.getElementById('chartComm').getContext('2d');

    const configs = [];
    const commMB = [];
    const barColors = [];

    Object.keys(LABELS).forEach(key => {
        if (!datasets[key]) return;
        const d = datasets[key];
        configs.push(LABELS[key]);
        commMB.push(+(d[0].comm_bytes / (1024 * 1024)).toFixed(2));
        barColors.push(COLORS[key].main);
    });

    chartInstances.chartComm = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: configs,
            datasets: [{
                label: 'Communication per Round (MB)',
                data: commMB,
                backgroundColor: barColors.map(c => c + '99'),
                borderColor: barColors,
                borderWidth: 2,
                borderRadius: 6,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Communication (MB)', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
            },
        },
    });
}

// ── Chart 6: Resource Footprint ──────────────────────────────
function renderResourceChart() {
    const ctx = document.getElementById('chartResource').getContext('2d');
    const dsets = [];

    Object.keys(LABELS).forEach(key => {
        if (!datasets[key]) return;
        const color = COLORS[key];
        dsets.push({
            label: LABELS[key] + ' (CPU%)',
            data: datasets[key].map(d => d.cpu_pct),
            borderColor: color.border,
            backgroundColor: color.bg,
            borderWidth: 1.5,
            pointRadius: 2,
            tension: 0.3,
            fill: true,
            yAxisID: 'y',
        });
    });

    const maxRounds = Math.max(...Object.values(datasets).map(d => d.length));
    const labels = Array.from({ length: maxRounds }, (_, i) => i + 1);

    chartInstances.chartResource = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: dsets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: {
                    ...CHART_DEFAULTS.scales.x,
                    title: { display: true, text: 'FL Round', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    position: 'left',
                    title: { display: true, text: 'CPU %', color: '#64748b',
                             font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
                },
            },
        },
    });
}

// ── Summary Table ────────────────────────────────────────────
function renderSummaryTable() {
    const tbody = document.getElementById('summaryTableBody');
    tbody.innerHTML = '';

    let baselineAcc = null;
    if (datasets.baseline) {
        const d = datasets.baseline;
        baselineAcc = d[d.length - 1].eval_acc;
    }

    Object.keys(LABELS).forEach(key => {
        if (!datasets[key]) return;
        const d = datasets[key];
        const n = d.length;
        const finalAcc = d[n - 1].eval_acc;
        const bestAcc = Math.max(...d.map(r => r.eval_acc));
        const accDrop = baselineAcc !== null ? ((baselineAcc - finalAcc) * 100) : 0;
        const avgWall = d.reduce((s, r) => s + r.wall_time_s, 0) / n;
        const avgEnc = d.reduce((s, r) => s + r.enc_overhead_s, 0) / n;
        const avgComm = d[0].comm_bytes / (1024 * 1024);
        const finalLoss = d[n - 1].eval_loss;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${LABELS[key]}</td>
            <td>${(finalAcc * 100).toFixed(2)}</td>
            <td>${(bestAcc * 100).toFixed(2)}</td>
            <td>${key === 'baseline' ? '—' : (accDrop >= 0 ? '+' : '') + accDrop.toFixed(2)}</td>
            <td>${avgWall.toFixed(1)}</td>
            <td>${avgEnc.toFixed(3)}</td>
            <td>${avgComm.toFixed(2)}</td>
            <td>${finalLoss.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ── Hero Stats Update ────────────────────────────────────────
function updateHeroStats() {
    // Best accuracy across all configs
    let bestAcc = 0;
    let bestConfig = '';
    Object.keys(datasets).forEach(key => {
        const d = datasets[key];
        const maxAcc = Math.max(...d.map(r => r.eval_acc));
        if (maxAcc > bestAcc) {
            bestAcc = maxAcc;
            bestConfig = key;
        }
    });
    document.getElementById('statAccuracy').textContent = (bestAcc * 100).toFixed(2) + '%';

    // Privacy budget (lowest ε used)
    const epsilons = [];
    if (datasets.he_eps10) epsilons.push(10);
    if (datasets.he_eps20) epsilons.push(20);
    if (datasets.he_eps50) epsilons.push(50);
    document.getElementById('statPrivacy').textContent =
        epsilons.length > 0 ? 'ε=' + Math.min(...epsilons) : '—';

    // Average HE overhead
    let totalEnc = 0, encCount = 0;
    ['he_eps10', 'he_eps20', 'he_eps50', 'ring'].forEach(key => {
        if (!datasets[key]) return;
        datasets[key].forEach(r => { totalEnc += r.enc_overhead_s; encCount++; });
    });
    document.getElementById('statOverhead').textContent =
        encCount > 0 ? (totalEnc / encCount).toFixed(3) + 's' : '—';

    // Accuracy drop (best HE vs baseline)
    if (datasets.baseline) {
        const bAcc = datasets.baseline[datasets.baseline.length - 1].eval_acc;
        let minDrop = Infinity;
        ['he_eps10', 'he_eps20', 'he_eps50'].forEach(key => {
            if (!datasets[key]) return;
            const hAcc = datasets[key][datasets[key].length - 1].eval_acc;
            const drop = Math.abs(bAcc - hAcc) * 100;
            if (drop < minDrop) minDrop = drop;
        });
        document.getElementById('statDrop').textContent =
            minDrop !== Infinity ? minDrop.toFixed(2) + '%' : '—';
    }
}

// ── Export Chart as PNG ──────────────────────────────────────
function exportChart(chartId) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;

    // Create a temporary canvas with white background for paper
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width * 2;
    tempCanvas.height = canvas.height * 2;
    const tempCtx = tempCanvas.getContext('2d');

    // White background for publication
    tempCtx.fillStyle = '#ffffff';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Scale up for high DPI
    tempCtx.scale(2, 2);
    tempCtx.drawImage(canvas, 0, 0);

    const link = document.createElement('a');
    link.download = `securefedhe_${chartId}.png`;
    link.href = tempCanvas.toDataURL('image/png', 1.0);
    link.click();
}
