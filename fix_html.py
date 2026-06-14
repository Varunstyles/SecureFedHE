import os

index_html_path = r"website/index.html"
with open(index_html_path, "r", encoding="utf-8") as f:
    index_html = f.read()

old_large_eval = """        <div id="page-large-eval" class="page hidden">
            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <h2 class="section-title">Large‑Scale Experimental Evaluation</h2>
                        <p class="section-desc">Scalability, non‑IID robustness, latency impact, and model‑architecture performance.</p>
                    </div>
                    <div class="grid gap-8">
                        <div class="card-glass p-4">
                            <h3 class="mb-2">Client Scaling</h3>
                            <canvas id="chartScale" width="400" height="300"></canvas>
                        </div>
                        <div class="card-glass p-4">
                            <h3 class="mb-2">Non‑IID Robustness</h3>
                            <canvas id="chartIid" width="400" height="300"></canvas>
                        </div>
                        <div class="card-glass p-4">
                            <h3 class="mb-2">Latency Impact</h3>
                            <canvas id="chartLatency" width="400" height="300"></canvas>
                        </div>
                        <div class="card-glass p-4">
                            <h3 class="mb-2">Model Architecture Comparison</h3>
                            <canvas id="chartArch" width="400" height="300"></canvas>
                        </div>
                    </div>
                </div>
            </section>
        </div>"""

new_large_eval = """        <div id="page-large-eval" class="page hidden">
            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <div class="hero-badge mb-4">LARGE-SCALE EVALUATION</div>
                        <h2 class="section-title">Large‑Scale Experimental Evaluation</h2>
                        <p class="section-desc">Scalability, non‑IID robustness, latency impact, and model‑architecture performance.</p>
                    </div>
                    <div class="charts-grid mt-16">
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Client Scaling</h3></div>
                                <button class="btn-export" onclick="exportChart('chartScale')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartScale"></canvas></div>
                        </div>
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Non‑IID Robustness</h3></div>
                                <button class="btn-export" onclick="exportChart('chartIid')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartIid"></canvas></div>
                        </div>
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Latency Impact</h3></div>
                                <button class="btn-export" onclick="exportChart('chartLatency')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartLatency"></canvas></div>
                        </div>
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Model Architecture</h3></div>
                                <button class="btn-export" onclick="exportChart('chartArch')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartArch"></canvas></div>
                        </div>
                    </div>
                </div>
            </section>
        </div>"""


old_security_eval = """        <div id="page-security" class="page hidden">
            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <h2 class="section-title">Security Evaluation</h2>
                        <p class="section-desc">Gradient inversion, Byzantine/model‑poisoning, and membership inference attacks.</p>
                    </div>
                    <div class="grid gap-8">
                        <div class="card-glass p-4">
    <h3 class="mb-2">Gradient Inversion (DLG)</h3>
    <canvas id="chartDlg" width="400" height="300"></canvas>
</div>
<div class="card-glass p-4">
    <h3 class="mb-2">Model Poisoning</h3>
    <canvas id="chartPoison" width="400" height="300"></canvas>
</div>
<div class="card-glass p-4">
    <h3 class="mb-2">Membership Inference (MIA)</h3>
    <canvas id="chartMia" width="400" height="300"></canvas>
</div>
                    </div>
                </div>
            </section>
        </div>"""


new_security_eval = """        <div id="page-security" class="page hidden">
            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <div class="hero-badge mb-4">SECURITY EVALUATION</div>
                        <h2 class="section-title">Security Evaluation</h2>
                        <p class="section-desc">Gradient inversion, Byzantine/model‑poisoning, and membership inference attacks.</p>
                    </div>
                    <div class="charts-grid mt-16">
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Gradient Inversion (DLG)</h3></div>
                                <button class="btn-export" onclick="exportChart('chartDlg')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartDlg"></canvas></div>
                        </div>
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Model Poisoning</h3></div>
                                <button class="btn-export" onclick="exportChart('chartPoison')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartPoison"></canvas></div>
                        </div>
                        <div class="chart-card card-glass">
                            <div class="chart-header">
                                <div><h3 class="chart-title">Membership Inference (MIA)</h3></div>
                                <button class="btn-export" onclick="exportChart('chartMia')">Export</button>
                            </div>
                            <div class="chart-container"><canvas id="chartMia"></canvas></div>
                        </div>
                    </div>
                </div>
            </section>
        </div>"""


index_html = index_html.replace(old_large_eval, new_large_eval)
index_html = index_html.replace(old_security_eval, new_security_eval)

with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(index_html)
print("Updated classes and markup.")
