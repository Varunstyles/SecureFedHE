import re
import os

app_js_path = r"website/app.js"
with open(app_js_path, "r", encoding="utf-8") as f:
    app_js = f.read()

# Fix 1: Missing closing brace for switchPage
app_js = app_js.replace("""    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (pageId === 'page-large-eval') renderLargeScaleCharts();
    if (pageId === 'page-security') renderSecurityCharts();

// ── Interactive Accordion ────────────────────────────────────""", """    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (pageId === 'page-large-eval') renderLargeScaleCharts();
    if (pageId === 'page-security') renderSecurityCharts();
}

// ── Interactive Accordion ────────────────────────────────────""")

# Fix 2: Missing function renderCommChart() {
app_js = app_js.replace("""function average(arr){ return arr.reduce((s,v)=>s+v,0)/arr.length; }



    const ctx = document.getElementById('chartComm').getContext('2d');""", """function average(arr){ return arr.reduce((s,v)=>s+v,0)/arr.length; }

function renderCommChart() {
    const ctx = document.getElementById('chartComm').getContext('2d');""")

with open(app_js_path, "w", encoding="utf-8") as f:
    f.write(app_js)

index_html_path = r"website/index.html"
with open(index_html_path, "r", encoding="utf-8") as f:
    index_html = f.read()

# For index.html, it's a bit messed up. We need to:
# 1. Remove the empty page-evaluation div.
# 2. Wrap the orphaned evaluation section with <div id="page-evaluation" class="page hidden">
# Wait, let's look at the raw index.html structure from the reference

empty_page_eval = """        <!-- ==================== PAGE 4: EVALUATION ==================== -->
        <div id="page-evaluation" class="page hidden">
            <!-- Evaluation Dashboard -->
        </div>"""
index_html = index_html.replace(empty_page_eval, "")

# The orphaned section starts with:
orphaned_start = """            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <div class="hero-badge mb-4">PHASE 4</div>
                        <h2 class="section-title">Evaluation Dashboard</h2>"""

# We'll replace this with the properly wrapped page-evaluation div
wrapped_eval = """        <!-- ==================== PAGE 4: EVALUATION ==================== -->
        <div id="page-evaluation" class="page hidden">
            <section class="section">
                <div class="container">
                    <div class="section-header">
                        <div class="hero-badge mb-4">PHASE 4</div>
                        <h2 class="section-title">Evaluation Dashboard</h2>"""
index_html = index_html.replace(orphaned_start, wrapped_eval)

with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(index_html)
print("Fixes applied.")
