"""
dashboard/dashboard.py — SecureFedHE Dashboard Backend
=======================================================
Runs on Node 0 (master) as a SEPARATE process on port 8080.
Node 0 continues running on port 8000 — this is an additional server.

Serves:
  /                  → index.html
  /api/config        → sanitised config
  /api/ring          → live status of all 5 nodes (polls /status)
  /api/metrics       → accuracy history (node /metrics endpoint or log fallback)
  /api/distribution  → per-hospital patient data breakdown
  /api/predict       → DiabetesNet inference on 8 patient vitals
  /api/explain       → Claude API explanation of a prediction
  /api/audit         → last N lines of node_0.log
  /api/health        → dashboard health check

Usage:
    python dashboard/dashboard.py               # reads config.json
    python dashboard/dashboard.py --dev         # HTTP/no-TLS mode
    python dashboard/dashboard.py --config path/to/config.json

Then open:  http://localhost:8080
            http://<node0_ip>:8080  (from any PC on the LAN)
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Any, Optional, Dict

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
DASH_DIR = Path(__file__).parent
ROOT     = DASH_DIR.parent
sys.path.insert(0, str(ROOT))

from models.diabetes_net import DiabetesNet, predict_single
from data.diabetes_loader import load_diabetes_datasets

# ── Global state (set in main() before uvicorn starts) ─────────────────────────
CONFIG:             dict                    = {}
MODEL:              Optional[DiabetesNet]   = None
NORM_MEAN:          Optional[np.ndarray]    = None
NORM_STD:           Optional[np.ndarray]    = None
DISTRIBUTION_CACHE: Optional[List[Dict]]    = None
DEV_MODE:           bool                    = False

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "Blood Pressure",
    "Skin Thickness", "Insulin", "BMI",
    "Diabetes Pedigree Function", "Age",
]

# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="SecureFedHE Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(DASH_DIR / "index.html")

@app.get("/app.js")
async def js_file():
    return FileResponse(DASH_DIR / "app.js", media_type="application/javascript")

@app.get("/style.css")
async def css_file():
    return FileResponse(DASH_DIR / "style.css", media_type="text/css")

# ═══════════════════════════════════════════════════════════════════════════════
# API — Config
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/config")
async def api_config():
    return {
        "nodes":      CONFIG["ring"]["nodes"],
        "rounds":     CONFIG["ring"]["rounds"],
        "epsilon":    CONFIG["privacy"]["dp_epsilon"],
        "model_type": CONFIG["model"]["type"],
        "he_layer":   CONFIG["he"]["target_layer"],
        "dev_mode":   DEV_MODE,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# API — Ring Status  (polls all 5 node /status endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/ring")
async def api_ring():
    nodes  = CONFIG["ring"]["nodes"]
    scheme = "http" if DEV_MODE else "https"
    results: List[Dict] = []

    async with httpx.AsyncClient(verify=False, timeout=2.5) as client:
        for node in nodes:
            url = f"{scheme}://{node['ip']}:{node['port']}/status"
            try:
                resp = await client.get(url)
                d    = resp.json()
                results.append({
                    "id":        node["id"],
                    "name":      node["name"],
                    "ip":        node["ip"],
                    "port":      node["port"],
                    "online":    True,
                    "round":     d.get("round", 0),
                    "zkp_ready": d.get("zkp_ready", False),
                    "status":    d.get("status", "unknown"),
                })
            except Exception:
                results.append({
                    "id":        node["id"],
                    "name":      node["name"],
                    "ip":        node["ip"],
                    "port":      node["port"],
                    "online":    False,
                    "round":     0,
                    "zkp_ready": False,
                    "status":    "offline",
                })

    online_count = sum(1 for n in results if n["online"])
    return {
        "nodes":        results,
        "total_rounds": CONFIG["ring"]["rounds"],
        "online_count": online_count,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# API — Training Metrics  (node /metrics first, log parsing fallback)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
async def api_metrics():
    """Tries node-0 /metrics endpoint first; falls back to log parsing."""
    nodes  = CONFIG["ring"]["nodes"]
    node0  = nodes[0]
    scheme = "http" if DEV_MODE else "https"

    try:
        async with httpx.AsyncClient(verify=False, timeout=2.5) as client:
            resp = await client.get(
                f"{scheme}://{node0['ip']}:{node0['port']}/metrics"
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass

    # Fallback: parse audit log
    return _parse_log_metrics()


def _parse_log_metrics() -> dict:
    log_dir  = CONFIG.get("audit", {}).get("log_dir", "logs")
    log_path = ROOT / log_dir / "node_0.log"
    rounds, accuracy = [], []

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        msg   = entry.get("msg", "")
                        m = re.search(
                            r"Round (\d+) complete \| accuracy=([\d.]+)", msg
                        )
                        if m:
                            rounds.append(int(m.group(1)))
                            accuracy.append(round(float(m.group(2)) * 100, 2))
                    except Exception:
                        pass
        except Exception:
            pass

    return {
        "rounds":          rounds,
        "accuracy":        accuracy,
        "current_round":   max(rounds) if rounds else 0,
        "total_rounds":    CONFIG["ring"]["rounds"],
        "latest_accuracy": accuracy[-1] if accuracy else None,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# API — Hospital Distribution
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/distribution")
async def api_distribution():
    return {"hospitals": DISTRIBUTION_CACHE}

# ═══════════════════════════════════════════════════════════════════════════════
# API — Patient Prediction
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    features: List[float]  # [Pregnancies, Glucose, BP, SkinThickness,
                           #  Insulin, BMI, DiabetesPedigreeFunction, Age]

@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    if len(req.features) != 8:
        raise HTTPException(
            400,
            detail="Exactly 8 features required: "
                   "[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]",
        )
    _refresh_model_if_checkpoint()
    result = predict_single(MODEL, req.features, NORM_MEAN, NORM_STD, device="cpu")
    return result


def _refresh_model_if_checkpoint():
    """Load model_latest.pt written by node.py after each round (if present)."""
    global MODEL
    ckpt = DASH_DIR / "model_latest.pt"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu")
            MODEL.load_state_dict(state)
            MODEL.eval()
        except Exception:
            pass  # keep previous model

# ═══════════════════════════════════════════════════════════════════════════════
# API — Claude Explanation
# ═══════════════════════════════════════════════════════════════════════════════

class ExplainRequest(BaseModel):
    features:      List[float]
    prediction:    int
    label:         str
    confidence:    float
    prob_diabetic: float
    prob_healthy:  float
    top_features:  List[Any]    # [(name, {risk_score, risk_abs, raw_value}), ...]


@app.post("/api/explain")
async def api_explain(req: ExplainRequest):
    api_key = CONFIG.get("dashboard", {}).get("claude_api_key", "")
    if not api_key:
        raise HTTPException(
            503,
            detail=(
                "Claude API key not configured. "
                "Add 'claude_api_key': '<key>' under 'dashboard' in config.json."
            ),
        )

    vitals = "\n".join(
        f"  • {name}: {val}"
        for name, val in zip(FEATURE_NAMES, req.features)
    )

    top_factors = "\n".join(
        f"  • {item[0]}: {'raises' if item[1]['risk_score'] > 0 else 'lowers'} risk "
        f"(contribution: {item[1]['risk_score']:+.3f})"
        for item in req.top_features[:3]
    ) if req.top_features else "  • Feature data unavailable"

    prompt = (
        "You are a compassionate medical AI assistant at a hospital. "
        "A privacy-preserving federated-learning diabetes screening model "
        "has analysed a patient's vitals.\n\n"
        f"Patient vitals:\n{vitals}\n\n"
        f"Screening result: {req.label} — {req.confidence:.1f}% confidence\n"
        f"  Probability of diabetes:    {req.prob_diabetic:.1f}%\n"
        f"  Probability of no diabetes: {req.prob_healthy:.1f}%\n\n"
        f"Key contributing factors:\n{top_factors}\n\n"
        "Write a clear, empathetic explanation (under 130 words) for this patient. "
        "Mention the 2–3 most important factors that shaped this result. "
        "Use plain everyday language — no medical jargon. "
        "Be honest but reassuring. "
        "Always recommend consulting a qualified doctor for a proper diagnosis. "
        "Clarify clearly that this is an AI screening tool and NOT a medical diagnosis."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type":  "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "max_tokens": 300,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
            data = resp.json()
            if "error" in data:
                raise HTTPException(
                    502, detail=f"Groq API error: {data['error']['message']}"
                )
            return {"explanation": data["choices"][0]["message"]["content"].strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Claude API call failed: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# API — Audit Log
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/audit")
async def api_audit(limit: int = 60):
    log_dir  = CONFIG.get("audit", {}).get("log_dir", "logs")
    log_path = ROOT / log_dir / "node_0.log"
    entries  = []

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    entries.append({
                        "time":  "",
                        "level": "RAW",
                        "node":  0,
                        "msg":   line,
                    })
        except Exception:
            pass

    return {"entries": list(reversed(entries)), "count": len(entries)}

# ═══════════════════════════════════════════════════════════════════════════════
# API — Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def api_health():
    return {
        "status":               "ok",
        "model_loaded":         MODEL is not None,
        "checkpoint_available": (DASH_DIR / "model_latest.pt").exists(),
        "dev_mode":             DEV_MODE,
        "claude_configured":    bool(CONFIG.get("dashboard", {}).get("claude_api_key")),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Startup initialisation
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    global MODEL, NORM_MEAN, NORM_STD, DISTRIBUTION_CACHE

    # ── Model ──────────────────────────────────────────────────────────────────
    mcfg  = CONFIG["model"]
    MODEL = DiabetesNet(
        input_dim   = mcfg["input_dim"],
        num_classes = mcfg["output_dim"],   # output_dim == num_classes == 2
    )
    MODEL.eval()
    _refresh_model_if_checkpoint()

    # ── Normalisation stats + distribution ─────────────────────────────────────
    train_loaders, _, NORM_MEAN, NORM_STD = load_diabetes_datasets(
        num_clients = len(CONFIG["ring"]["nodes"]),
        verbose     = False,
    )

    hospitals: List[Dict] = []
    for i, loader in enumerate(train_loaders):
        all_labels: List[int] = []
        for _, y_batch in loader:
            all_labels.extend(y_batch.tolist())
        total    = len(all_labels)
        diabetic = int(sum(all_labels))
        hospitals.append({
            "id":           i,
            "name":         CONFIG["ring"]["nodes"][i]["name"],
            "total":        total,
            "diabetic":     diabetic,
            "healthy":      total - diabetic,
            "diabetic_pct": round(diabetic / total * 100, 1) if total else 0,
        })
    DISTRIBUTION_CACHE = hospitals

    # ── Banner ─────────────────────────────────────────────────────────────────
    dash_cfg = CONFIG.get("dashboard", {})
    print("\n" + "═" * 55)
    print("  SecureFedHE — Dashboard")
    print("═" * 55)
    print(f"  URL:    http://0.0.0.0:{dash_cfg.get('port', 8080)}")
    print(f"  Mode:   {'HTTP (dev)' if DEV_MODE else 'HTTPS-aware'}")
    print(f"  Nodes:  {len(CONFIG['ring']['nodes'])} hospitals")
    print(f"  Claude: {'configured ✓' if dash_cfg.get('claude_api_key') else 'NOT SET — add claude_api_key to config.json'}")
    print("═" * 55 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config(path: str = None) -> dict:
    p = path or (ROOT / "config.json")
    with open(p) as f:
        return json.load(f)


def main():
    global CONFIG, DEV_MODE

    parser = argparse.ArgumentParser(description="SecureFedHE Dashboard")
    parser.add_argument("--config", default=str(ROOT / "config.json"),
                        help="Path to config.json (default: ../config.json)")
    parser.add_argument("--dev", action="store_true",
                        help="Dev mode — HTTP, no TLS, single-PC testing")
    args = parser.parse_args()

    CONFIG   = _load_config(args.config)
    DEV_MODE = args.dev

    dash_cfg = CONFIG.get("dashboard", {})
    host     = dash_cfg.get("host", "0.0.0.0")
    port     = dash_cfg.get("port", 8080)

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
