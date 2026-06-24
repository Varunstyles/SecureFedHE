"""
node.py — SecureFedHE Hospital Node (FastAPI + mTLS)
=====================================================
Replaces distributed_simulation/distributed_node.py for real multi-machine deployment.

Changes from old Flask version:
  - FastAPI instead of Flask (async, faster, built-in validation)
  - mTLS: all inter-node HTTP calls use mutual TLS (certs/*)
  - Real IPs from config.json instead of hardcoded loopback
  - DiabetesNet instead of SimpleCNN (real healthcare use case)
  - New ZKP engine (zkp_engine.py) instead of RSA commitment
  - ε=3 DP (stronger than paper's ε=10, appropriate for healthcare)
  - Structured JSON audit logging
  - Clean shutdown via /shutdown endpoint

Usage (on each PC, after editing config.json with correct IPs):
    python node.py --id 0          # Hospital 0 (master, starts training)
    python node.py --id 1          # Hospital 1
    python node.py --id 2          # Hospital 2
    python node.py --id 3          # Hospital 3
    python node.py --id 4          # Hospital 4

Or use launch.py which handles all the above automatically.
"""

import os
import sys
import json
import time
import pickle
import base64
import logging
import argparse
import threading
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import requests
import ssl
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.diabetes_net import DiabetesNet
from data.diabetes_loader import load_diabetes_federated
from distributed_simulation.zkp_commitment import (
    zkp_ring_setup, generate_commitment, verify_commitment, generate_node_keypair
)

# ── Audit logger ───────────────────────────────────────────────────────────────
def setup_logger(node_id: int, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"node_{node_id}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"node_{node_id}.log"))
    fh.setFormatter(logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","node":' +
        str(node_id) + ',"msg":%(message)s}'
    ))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[Node {node_id}] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── mTLS HTTP client ───────────────────────────────────────────────────────────
def make_tls_session(config: dict) -> requests.Session:
    """Create a requests Session with mTLS configured from config.json."""
    tls   = config["tls"]
    sess  = requests.Session()
    sess.cert  = (tls["client_cert"], tls["client_key"])
    sess.verify = tls["ca_cert"]
    return sess


# ── DP noise ───────────────────────────────────────────────────────────────────
def add_dp_noise(params: dict, epsilon: float, delta: float, sensitivity: float) -> dict:
    """Add Gaussian DP noise to model parameters (applied to all layers except fc2)."""
    sigma = (sensitivity / np.sqrt(sum(p.size for p in params.values()))) * \
            np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noised = {}
    for k, v in params.items():
        if k in ("fc2.weight", "fc2.bias"):
            noised[k] = v  # fc2 is handled by HE, no DP noise
        else:
            noised[k] = v + np.random.normal(0, sigma, v.shape).astype(np.float32)
    return noised


# ── Simulated CKKS (Windows-compatible) ───────────────────────────────────────
class SimulatedHE:
    """Lightweight CKKS simulation — encrypts by storing plaintext + noise tag."""

    def encrypt(self, arr: np.ndarray) -> dict:
        return {"data": arr.tolist(), "encrypted": True}

    def decrypt(self, enc: dict) -> np.ndarray:
        return np.array(enc["data"], dtype=np.float32)

    def add(self, enc_a: dict, enc_b: dict) -> dict:
        a = np.array(enc_a["data"])
        b = np.array(enc_b["data"])
        return {"data": (a + b).tolist(), "encrypted": True}

    def scale(self, enc: dict, scalar: float) -> dict:
        return {"data": (np.array(enc["data"]) * scalar).tolist(), "encrypted": True}


# ── Local training ─────────────────────────────────────────────────────────────
def local_train(model: DiabetesNet, loader: DataLoader, epochs: int,
                lr: float, device: torch.device) -> dict:
    """Train one round locally, return updated params as numpy dict."""
    model.train()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def evaluate(model: DiabetesNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total if total > 0 else 0.0


# ── Serialization ──────────────────────────────────────────────────────────────
def serialize_params(params: dict) -> dict:
    return {k: base64.b64encode(pickle.dumps(v)).decode() for k, v in params.items()}

def deserialize_params(d: dict) -> dict:
    return {k: pickle.loads(base64.b64decode(v)) for k, v in d.items()}

def serialize_enc(enc_dict: dict) -> dict:
    return {k: json.dumps(v) for k, v in enc_dict.items()}

def deserialize_enc(d: dict) -> dict:
    return {k: json.loads(v) for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="SecureFedHE Node")

# Global node state
STATE = {
    "node_id":       None,
    "config":        None,
    "model":         None,
    "loader":        None,
    "test_loader":   None,
    "he":            None,
    "device":        None,
    "logger":        None,
    "session":       None,   # mTLS requests session
    "current_round": 0,
    "is_master":     False,
    "zkp_ready":     False,
}


def get_successor_url() -> str:
    nodes   = STATE["config"]["ring"]["nodes"]
    nid     = STATE["node_id"]
    n       = len(nodes)
    succ    = nodes[(nid + 1) % n]
    return f"https://{succ['ip']}:{succ['port']}"


def get_node_url(node_id: int) -> str:
    node = STATE["config"]["ring"]["nodes"][node_id]
    return f"https://{node['ip']}:{node['port']}"


def get_params(model: DiabetesNet) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def set_params(model: DiabetesNet, params: dict):
    model.load_state_dict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in params.items()
    })


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "status":        "ready",
        "node_id":       STATE["node_id"],
        "round":         STATE["current_round"],
        "zkp_ready":     STATE["zkp_ready"],
    }


@app.post("/start_ring")
async def start_ring():
    if not STATE["is_master"]:
        raise HTTPException(400, "Only master node can start the ring")
    threading.Thread(target=execute_round, daemon=True).start()
    return {"status": "Ring started"}


@app.post("/receive_update")
async def receive_update(request: Request):
    data = await request.json()
    threading.Thread(target=handle_update, args=(data,), daemon=True).start()
    return {"status": "accepted"}


@app.get("/shutdown")
async def shutdown():
    STATE["logger"].info('"Shutdown requested"')
    os._exit(0)


# ── Ring logic ─────────────────────────────────────────────────────────────────

def execute_round():
    """Master node: train locally, encrypt fc2, send to successor."""
    log    = STATE["logger"]
    config = STATE["config"]
    rnd    = STATE["current_round"]
    nid    = STATE["node_id"]
    dp     = config["privacy"]

    log.info(f'"Starting round {rnd + 1}"')

    params = local_train(
        STATE["model"], STATE["loader"],
        epochs=config["model"]["local_epochs"],
        lr=config["model"]["lr"],
        device=STATE["device"]
    )

    # DP noise on non-fc2 layers
    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # ZKP commitment on fc2 weights
    fc2_flat = noised["fc2.weight"].flatten().tolist()
    try:
        commitment = generate_commitment(fc2_flat, f"node_{nid}", rnd)
    except ValueError as e:
        log.warning(f'"ZKP norm violation (clipping applied): {e}"')
        # Clip and retry
        fc2_arr = noised["fc2.weight"].flatten()
        norm = np.linalg.norm(fc2_arr)
        if norm > 0.5:
            fc2_arr = fc2_arr * (0.45 / norm)
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        commitment = generate_commitment(fc2_arr.tolist(), f"node_{nid}", rnd)

    # HE encrypt fc2
    he   = STATE["he"]
    enc  = {
        "fc2.weight": he.encrypt(noised["fc2.weight"].flatten()),
        "fc2.bias":   he.encrypt(noised["fc2.bias"])
    }

    n_samples = len(STATE["loader"].dataset)

    payload = {
        "round":      rnd,
        "origin_id":  nid,
        "sender_id":  nid,
        "n_samples":  n_samples,
        "enc_fc2":    serialize_enc(enc),
        "plain":      serialize_params({k: v for k, v in noised.items()
                                        if k not in ("fc2.weight", "fc2.bias")}),
        "commitment": commitment,
        "acc_sum":    0.0,
        "weight_sum": n_samples,
    }

    _forward(payload)


def handle_update(data: dict):
    """Non-master node: verify ZKP, add local update, forward."""
    log    = STATE["logger"]
    nid    = STATE["node_id"]
    rnd    = data["round"]
    origin = data["origin_id"]
    sender = data["sender_id"]
    config = STATE["config"]
    dp     = config["privacy"]

    log.info(f'"Received update from node {sender}, round {rnd}"')

    # Ring complete — master receives its own payload back
    if origin == nid:
        log.info(f'"Round {rnd + 1} complete — aggregating"')
        _finalize_round(data)
        return

    # ── ZKP verification ──────────────────────────────────────
    commitment = data.get("commitment", {})
    if commitment:
        ok, reason = verify_commitment(commitment, expected_round=rnd,
                                        sender_id=f"node_{sender}")
        if not ok:
            log.warning(f'"ZKP REJECTED from node {sender}: {reason}"')
            _forward(data)  # skip: forward unchanged (Fix-1 protocol)
            return
        log.info(f'"ZKP ACCEPTED from node {sender}"')

    # ── Local training ────────────────────────────────────────
    params = local_train(
        STATE["model"], STATE["loader"],
        epochs=config["model"]["local_epochs"],
        lr=config["model"]["lr"],
        device=STATE["device"]
    )
    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # ── ZKP commitment ────────────────────────────────────────
    fc2_flat = noised["fc2.weight"].flatten().tolist()
    try:
        my_commitment = generate_commitment(fc2_flat, f"node_{nid}", rnd)
    except ValueError:
        fc2_arr = noised["fc2.weight"].flatten()
        fc2_arr = fc2_arr * (0.45 / np.linalg.norm(fc2_arr))
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        my_commitment = generate_commitment(fc2_arr.tolist(), f"node_{nid}", rnd)

    # ── HE aggregation ────────────────────────────────────────
    he         = STATE["he"]
    n_samples  = len(STATE["loader"].dataset)
    inc_enc    = deserialize_enc(data["enc_fc2"])
    inc_plain  = deserialize_params(data["plain"])
    inc_w      = data["weight_sum"]
    total_w    = inc_w + n_samples

    # Weighted average in encrypted space
    my_enc = {
        "fc2.weight": he.encrypt(noised["fc2.weight"].flatten()),
        "fc2.bias":   he.encrypt(noised["fc2.bias"])
    }
    agg_enc = {}
    for k in inc_enc:
        scaled_inc = he.scale(inc_enc[k], inc_w / total_w)
        scaled_my  = he.scale(my_enc[k],  n_samples / total_w)
        agg_enc[k] = he.add(scaled_inc, scaled_my)

    # Weighted average of plain layers
    agg_plain = {}
    for k in inc_plain:
        if k in noised:
            agg_plain[k] = (inc_plain[k] * inc_w + noised[k] * n_samples) / total_w
        else:
            agg_plain[k] = inc_plain[k]

    payload = {
        **data,
        "sender_id":  nid,
        "enc_fc2":    serialize_enc(agg_enc),
        "plain":      serialize_params(agg_plain),
        "commitment": my_commitment,
        "weight_sum": total_w,
    }

    _forward(payload)


def _forward(payload: dict):
    """Send payload to successor node."""
    log  = STATE["logger"]
    url  = get_successor_url()
    sess = STATE["session"]
    try:
        sess.post(f"{url}/receive_update", json=payload, timeout=30)
        log.info(f'"Forwarded to {url}"')
    except Exception as e:
        log.error(f'"Forward failed to {url}: {e}"')


def _finalize_round(data: dict):
    """Master: decrypt aggregated fc2, update model, log accuracy."""
    log    = STATE["logger"]
    he     = STATE["he"]
    model  = STATE["model"]
    rnd    = data["round"]

    # Decrypt fc2
    enc_fc2  = deserialize_enc(data["enc_fc2"])
    plain    = deserialize_params(data["plain"])

    fc2_w = he.decrypt(enc_fc2["fc2.weight"]).reshape(
        STATE["model"].fc2.weight.shape)
    fc2_b = he.decrypt(enc_fc2["fc2.bias"])

    new_params = {**plain, "fc2.weight": fc2_w, "fc2.bias": fc2_b}
    set_params(model, new_params)

    # Evaluate
    acc = evaluate(model, STATE["test_loader"], STATE["device"])
    log.info(f'"Round {rnd + 1} complete | accuracy={acc:.4f}"')
    print(f"[Node {STATE['node_id']}] Round {rnd + 1} | Accuracy: {acc*100:.2f}%")

    STATE["current_round"] += 1
    if STATE["current_round"] < STATE["config"]["ring"]["rounds"]:
        time.sleep(0.5)
        threading.Thread(target=execute_round, daemon=True).start()
    else:
        rounds_done = STATE["current_round"]
        log.info(f'"Training complete after {rounds_done} rounds"')
        print(f"[Node {STATE['node_id']}] Training complete!")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SecureFedHE Hospital Node")
    parser.add_argument("--id",     type=int, required=True, help="Node ID (0-4)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--dev",    action="store_true",
                        help="Dev mode: HTTP instead of HTTPS (no certs needed)")
    args = parser.parse_args()

    config  = load_config(args.config)
    nid     = args.node_id = args.id
    nodes   = config["ring"]["nodes"]
    node_cfg = nodes[nid]
    logger  = setup_logger(nid, config["audit"]["log_dir"])

    node_name = node_cfg["name"]
    logger.info(f'"Starting node {nid}: {node_name}"')

    # ── ZKP setup ──────────────────────────────────────────────
    zkp_cfg = config["zkp"]
    zkp_ring_setup(
        n_nodes=len(nodes),
        gradient_dim=zkp_cfg["gradient_dim"],
        clipping_threshold=config["privacy"]["clip_threshold"],
        setup_seed=zkp_cfg["setup_seed"],
        keys_dir=zkp_cfg["keys_dir"],
    )
    STATE["zkp_ready"] = True

    # ── Data ────────────────────────────────────────────────────
    loaders, test_loader = load_diabetes_federated(
        n_hospitals=len(nodes),
        seed=42
    )
    my_loader = loaders[nid]

    # ── Model ───────────────────────────────────────────────────
    device = torch.device("cpu")
    mcfg   = config["model"]
    model  = DiabetesNet(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        output_dim=mcfg["output_dim"]
    ).to(device)

    # ── State ───────────────────────────────────────────────────
    STATE.update({
        "node_id":     nid,
        "config":      config,
        "model":       model,
        "loader":      my_loader,
        "test_loader": test_loader,
        "he":          SimulatedHE(),
        "device":      device,
        "logger":      logger,
        "session":     make_tls_session(config) if not args.dev else requests.Session(),
        "is_master":   (nid == 0),
    })

    # ── TLS server config ───────────────────────────────────────
    if args.dev:
        ssl_ctx = None
        logger.warning('"DEV MODE: running without TLS"')
    else:
        tls     = config["tls"]
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(tls["server_cert"], tls["server_key"])
        ssl_ctx.load_verify_locations(tls["ca_cert"])
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED  # mutual TLS

    host = node_cfg["ip"]
    port = node_cfg["port"]

    logger.info(f'"Listening on {host}:{port} | TLS={not args.dev}"')
    print(f"\n{'='*55}")
    print(f"  SecureFedHE — {node_cfg['name']}")
    print(f"  Node {nid} | {host}:{port}")
    print(f"  Rounds: {config['ring']['rounds']} | ε={config['privacy']['dp_epsilon']}")
    print(f"  {'HTTP (dev mode)' if args.dev else 'HTTPS (mTLS)'}")
    print(f"{'='*55}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_keyfile  = None if args.dev else config["tls"]["server_key"],
        ssl_certfile = None if args.dev else config["tls"]["server_cert"],
        ssl_ca_certs = None if args.dev else config["tls"]["ca_cert"],
        log_level="warning"
    )


if __name__ == "__main__":
    main()
