import os

index_html_path = r"website/index.html"
with open(index_html_path, "r", encoding="utf-8") as f:
    index_html = f.read()

# Fix Navigation Text
old_nav = """                <li><a href="#" onclick="switchPage('page-architecture')" class="nav-item" id="nav-architecture">Deep-Dive</a></li>
                <li><a href="#" onclick="switchPage('page-simulation')" class="nav-item" id="nav-simulation">Network Simulation</a></li>
                <li><a href="#" onclick="switchPage('page-evaluation')" class="nav-item" id="nav-evaluation">Evaluation Dashboard</a></li>
                <li><a href="#" onclick="switchPage('page-large-eval')" class="nav-item" id="nav-large-eval">Large‑Scale Experimental Evaluation</a></li>
                <li><a href="#" onclick="switchPage('page-security')" class="nav-item" id="nav-security">Security Evaluation</a></li>"""

new_nav = """                <li><a href="#" onclick="switchPage('page-architecture')" class="nav-item" id="nav-architecture">Architecture</a></li>
                <li><a href="#" onclick="switchPage('page-simulation')" class="nav-item" id="nav-simulation">Simulation</a></li>
                <li><a href="#" onclick="switchPage('page-evaluation')" class="nav-item" id="nav-evaluation">Evaluation</a></li>
                <li><a href="#" onclick="switchPage('page-large-eval')" class="nav-item" id="nav-large-eval">Large-Scale</a></li>
                <li><a href="#" onclick="switchPage('page-security')" class="nav-item" id="nav-security">Security</a></li>"""
index_html = index_html.replace(old_nav, new_nav)

# Add details to Large-Scale graphs
old_scale_desc = """<p class="section-desc">Scalability, non‑IID robustness, latency impact, and model‑architecture performance.</p>"""
new_scale_desc = """<p class="section-desc">Scalability, non‑IID robustness, latency impact, and model‑architecture performance.</p>
                        <p class="section-desc" style="font-size: 0.95rem; margin-top: 10px; opacity: 0.8;">
                            <strong>Client Scaling:</strong> Demonstrates computational efficiency as the number of clients increases to 50.<br>
                            <strong>Non-IID Robustness:</strong> Shows convergence across varying degrees of Dirichlet data heterogeneity (α).<br>
                            <strong>Latency Impact:</strong> Assesses model performance under severe simulated network delays up to 100ms.<br>
                            <strong>Model Architecture:</strong> Compares structural integrity and accuracy between SimpleCNN and ResNet18.
                        </p>"""
index_html = index_html.replace(old_scale_desc, new_scale_desc)

# Add details to Security graphs
old_sec_desc = """<p class="section-desc">Gradient inversion, Byzantine/model‑poisoning, and membership inference attacks.</p>"""
new_sec_desc = """<p class="section-desc">Gradient inversion, Byzantine/model‑poisoning, and membership inference attacks.</p>
                        <p class="section-desc" style="font-size: 0.95rem; margin-top: 10px; opacity: 0.8;">
                            <strong>Gradient Inversion (DLG):</strong> Evaluates image reconstruction PSNR with and without Differential Privacy.<br>
                            <strong>Model Poisoning:</strong> Demonstrates aggregator resilience (Krum/Trimmed Mean) against malicious weight updates.<br>
                            <strong>Membership Inference (MIA):</strong> Analyzes the attack advantage of tracing participant data through global models.
                        </p>"""
index_html = index_html.replace(old_sec_desc, new_sec_desc)


with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(index_html)
print("Updated HTML nav and details.")
