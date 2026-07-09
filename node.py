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
os.environ["PYTHONIOENCODING"] = "utf-8"
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
from data.diabetes_loader import load_diabetes_datasets
from distributed_simulation.trusted_setup import setup as zkp_setup_v2
from distributed_simulation.prover_v2 import prove as zkp_prove_v2
from distributed_simulation.verifier_v2 import verify as zkp_verify_v2

from consensus_models import (
    ConsensusVote, QuorumCertificate, RejectionReason, canonical_json, UpdateProposal,
)
from consensus_engine import (
    sign_payload, verify_signature, make_vote, verify_vote,
    compute_update_hash, QuorumTracker, evaluate_update_for_vote,
    load_private_key, load_public_key_from_cert,
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
    """
    Apply DP noise only to trainable weight/bias parameters.
    Skip fc2 (CKKS-protected), BatchNorm running stats, and num_batches_tracked.
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noised = {}
    for k, v in params.items():
        if "fc2" in k:
            noised[k] = v  # CKKS-protected
        elif "running_mean" in k or "running_var" in k or "num_batches" in k:
            noised[k] = v  # BatchNorm buffers — never add noise
        else:
            noised[k] = v + np.random.normal(0, sigma, v.shape).astype(np.float32)
    return noised


# ── Secure Aggregation ─────────────────────────────────────────────────────────

import hashlib
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

class SecureAggregator:
    """
    Real Secure Aggregation using Diffie-Hellman shared secrets.
    Each node pair derives a shared secret and generates correlated
    random masks. Masks cancel during aggregation at the master,
    so the master gets the correct sum without seeing individual weights.
    """
    def __init__(self, node_id: int, all_node_ids: list):
        self.node_id = node_id
        self.all_node_ids = [n for n in all_node_ids if n != node_id]
        self._private_key = X25519PrivateKey.generate()
        self._public_key_bytes = self._private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        self._shared_secrets = {}

    def get_public_key(self) -> bytes:
        return self._public_key_bytes

    def add_peer_public_key(self, peer_id: int, peer_pub_bytes: bytes):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
        peer_pub = X25519PublicKey.from_public_bytes(peer_pub_bytes)
        shared = self._private_key.exchange(peer_pub)
        self._shared_secrets[peer_id] = shared

    def _get_mask(self, peer_id: int, length: int, round_num: int) -> np.ndarray:
        secret = self._shared_secrets[peer_id]
        seed_material = secret + round_num.to_bytes(4, 'big')
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], 'big')
        rng = np.random.default_rng(seed)
        return rng.standard_normal(length).astype(np.float32) * 0.01

    def mask(self, arr: np.ndarray, round_num: int) -> np.ndarray:
        missing = [p for p in self.all_node_ids if p not in self._shared_secrets]
        if missing:
            raise RuntimeError(
                f"Cannot mask: missing DH shared secret for peer(s) {missing}. "
                f"Key exchange with this peer has not completed. Refusing to "
                f"send a partially-masked (under-protected) update."
            )
        masked = arr.flatten().copy()
        for peer_id in self.all_node_ids:
            mask = self._get_mask(peer_id, len(masked), round_num)
            if peer_id > self.node_id:
                masked += mask
            else:
                masked -= mask
        return masked

    def encrypt(self, arr: np.ndarray) -> dict:
        round_num = STATE.get("current_round", 0)
        masked = self.mask(arr.flatten(), round_num)
        return {"data": masked.tolist(), "encrypted": True, "scheme": "SecureAggregation-DH"}

    def decrypt(self, enc: dict) -> np.ndarray:
        return np.array(enc["data"], dtype=np.float32)

    def add(self, enc_a: dict, enc_b: dict) -> dict:
        a = np.array(enc_a["data"])
        b = np.array(enc_b["data"])
        return {"data": (a + b).tolist(), "encrypted": True, "scheme": "SecureAggregation-DH"}

    def scale(self, enc: dict, scalar: float) -> dict:
        return {"data": (np.array(enc["data"]) * scalar).tolist(), "encrypted": True, "scheme": "SecureAggregation-DH"}
    


# ── Local training ─────────────────────────────────────────────────────────────
def local_train(model: DiabetesNet, loader: DataLoader, epochs: int,
                lr: float, device: torch.device) -> dict:
    """Train one round locally, return updated params as numpy dict."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
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
    "dev_mode": False,
}


def get_successor_url() -> str:
    nodes   = STATE["config"]["ring"]["nodes"]
    nid     = STATE["node_id"]
    n       = len(nodes)
    succ    = nodes[(nid + 1) % n]
    scheme  = "http" if STATE.get("dev_mode") else "https"
    return f"{scheme}://{succ['ip']}:{succ['port']}"


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

def _broadcast_proposal(proposal: "UpdateProposal"):
    """Master: send the proposal (round_id, origin, and the ONE
    agreed update_hash) to every peer BEFORE the real update starts
    circulating the ring. This is what lets all nodes vote against
    the same fixed hash, instead of each recomputing a hash from a
    payload that mutates at every hop."""
    log    = STATE["logger"]
    nid    = STATE["node_id"]
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"

    # Record locally too — origin node must also vote against its own
    # fixed hash later, same as peers.
    STATE["pending_proposals"][(proposal.round_id, proposal.origin_node_id)] = proposal

    for node in nodes:
        if node["id"] == nid:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/propose"
        _post_with_retry(
            sess, url, proposal.to_dict(),
            log, "Sent proposal", node["id"],
        )


def _broadcast_vote(vote: "ConsensusVote"):
    """Send our signed vote to every peer, and also record it in our
    own local QuorumTracker (a node's vote counts for itself too)."""
    log    = STATE["logger"]
    nid    = STATE["node_id"]
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"

    key = (vote.round_id, vote.update_hash)
    tracker = STATE["quorum_trackers"].get(key)
    if tracker is None:
        tracker = QuorumTracker(vote.round_id, vote.update_hash, STATE["quorum_required"])
        STATE["quorum_trackers"][key] = tracker
    tracker.add_vote(vote)

    for node in nodes:
        if node["id"] == nid:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/vote"
        try:
            sess.post(url, json=vote.to_dict(), timeout=10)
        except Exception as e:
            log.warning(f'"Failed to send consensus vote to node {node["id"]}: {e}"')

def _wait_for_key_exchange(timeout_s: float = 30) -> bool:
    """Block until this node has a shared secret with every peer, or timeout."""
    he = STATE["he"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        missing = [p for p in he.all_node_ids if p not in he._shared_secrets]
        if not missing:
            return True
        time.sleep(0.5)
    return False


def _start_ring_background():
    """Runs key exchange + wait, then kicks off the training ring.
    Executed in a background thread so /start_ring can return
    immediately instead of blocking the event loop (and therefore
    blocking ALL other incoming requests, including peers' key
    exchange POSTs to this node) for up to 30 seconds."""
    log = STATE["logger"]
    _exchange_keys()
    _exchange_consensus_pubkeys()
    ok = _wait_for_key_exchange(timeout_s=30)
    if not ok:
        missing = [p for p in STATE["he"].all_node_ids
                   if p not in STATE["he"]._shared_secrets]
        log.error(
            f'"Key exchange incomplete after 30s — missing peers {missing}. '
            f'Refusing to start ring with degraded/broken secure aggregation."'
        )
        return
    threading.Thread(target=execute_round, daemon=True).start()


@app.post("/start_ring")
async def start_ring():
    if not STATE["is_master"]:
        raise HTTPException(400, "Only master node can start the ring")
    threading.Thread(target=_start_ring_background, daemon=True).start()
    return {"status": "Ring start triggered — key exchange running in background"}

@app.post("/sync_weights")
async def sync_weights(request: Request):
    """Workers receive aggregated weights from master after each round.
    Also advances this worker's own round counter — otherwise workers
    never learn a round finished, and every subsequent round would be
    rejected as wrong_round forever."""
    data   = await request.json()
    params = {k: np.array(v, dtype=np.float32) for k, v in data["params"].items()}
    set_params(STATE["model"], params)
    STATE["current_round"] = data["round"] + 1
    STATE["logger"].info(f'"Synced weights from master for round {data["round"]}"')
    return {"status": "synced"}

@app.post("/exchange_keys")
async def exchange_keys(request: Request):
    """Receive a peer's DH public key and register it."""
    data = await request.json()
    peer_id = data["node_id"]
    pub_bytes = bytes.fromhex(data["public_key"])
    STATE["he"].add_peer_public_key(peer_id, pub_bytes)
    STATE["logger"].info(f'"Key exchange complete with node {peer_id}"')
    return {"status": "ok"}

@app.post("/consensus/pubkey")
async def consensus_pubkey(request: Request):
    """Dev-mode only: exchange in-memory RSA public keys for vote signing
    (mirrors /exchange_keys for the DH keys). In production, peer public
    keys come from mTLS certs instead, loaded once at startup."""
    data = await request.json()
    peer_id = data["node_id"]
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    pub = load_pem_public_key(data["public_key_pem"].encode("utf-8"))
    STATE["consensus_peer_pubkeys"][peer_id] = pub
    STATE["logger"].info(f'"Consensus pubkey received from node {peer_id}"')
    return {"status": "ok"}

@app.post("/consensus/propose")
async def consensus_propose(request: Request):
    """Receive the master's proposal — the agreed (round_id, origin,
    update_hash) — before the actual update arrives via the ring."""
    data = await request.json()
    proposal = UpdateProposal.from_dict(data)
    STATE["pending_proposals"][(proposal.round_id, proposal.origin_node_id)] = proposal
    STATE["logger"].info(
        f'"Proposal received: round={proposal.round_id} '
        f'origin={proposal.origin_node_id} hash={proposal.update_hash[:12]}..."'
    )
    return {"status": "ok"}

@app.post("/consensus/vote")
async def consensus_vote(request: Request):
    """Receive a signed ACCEPT/REJECT vote from a peer for a given
    (round_id, update_hash), record it, and report whether quorum is
    now satisfied for that update."""
    data = await request.json()
    vote = ConsensusVote.from_dict(data)

    voter_pub = STATE["consensus_peer_pubkeys"].get(vote.voter_node_id)
    if voter_pub is None:
        STATE["logger"].warning(
            f'"Consensus vote from unknown node {vote.voter_node_id} — no public key on file"'
        )
        return JSONResponse({"status": "rejected", "reason": RejectionReason.UNKNOWN_SENDER}, status_code=400)

    if not verify_vote(voter_pub, vote):
        STATE["logger"].warning(
            f'"Consensus vote from node {vote.voter_node_id} FAILED signature check"'
        )
        return JSONResponse({"status": "rejected", "reason": RejectionReason.BAD_SIGNATURE}, status_code=400)

    key = (vote.round_id, vote.update_hash)
    tracker = STATE["quorum_trackers"].get(key)
    if tracker is None:
        tracker = QuorumTracker(vote.round_id, vote.update_hash, STATE["quorum_required"])
        STATE["quorum_trackers"][key] = tracker

    accepted = tracker.add_vote(vote)
    STATE["logger"].info(
        f'"Consensus vote recorded: node {vote.voter_node_id} decision={vote.decision} '
        f'reason={vote.reason_code} round={vote.round_id} new={accepted} '
        f'satisfied={tracker.is_satisfied}"'
    )
    return {"status": "ok", "satisfied": tracker.is_satisfied}

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

def _post_with_retry(sess, url, payload, log, label, node_id,
                      attempts=6, delay_s=2):
    """POST with retries — covers the case where a peer's server
    hasn't finished starting up yet (connection refused)."""
    for attempt in range(1, attempts + 1):
        try:
            sess.post(url, json=payload, timeout=10)
            log.info(f'"{label} to node {node_id}"')
            return True
        except Exception as e:
            if attempt == attempts:
                log.warning(f'"{label} to node {node_id} FAILED after {attempts} attempts: {e}"')
                return False
            time.sleep(delay_s)
    return False


def _exchange_keys():
    """Push our DH public key to all peer nodes, retrying if a peer
    isn't listening yet."""
    log     = STATE["logger"]
    nid     = STATE["node_id"]
    nodes   = STATE["config"]["ring"]["nodes"]
    sess    = STATE["session"]
    scheme  = "http" if STATE.get("dev_mode") else "https"
    pub_hex = STATE["he"].get_public_key().hex()
    for node in nodes:
        if node["id"] == nid:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/exchange_keys"
        _post_with_retry(
            sess, url, {"node_id": nid, "public_key": pub_hex},
            log, "Sent public key", node["id"],
        )

def _exchange_consensus_pubkeys():
    """Dev-mode only: broadcast our in-memory RSA public key to all
    peers so votes can be verified. In production this is skipped
    entirely — peer public keys come from mTLS certs, loaded at startup."""
    if not STATE.get("dev_mode"):
        return
    log    = STATE["logger"]
    nid    = STATE["node_id"]
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http"
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    pub_pem = STATE["consensus_privkey"].public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")
    for node in nodes:
        if node["id"] == nid:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/pubkey"
        _post_with_retry(
            sess, url, {"node_id": nid, "public_key_pem": pub_pem},
            log, "Sent consensus pubkey", node["id"],
        )

def execute_round():
    """Master node: train locally, encrypt fc2, send to successor."""
    log    = STATE["logger"]
    config = STATE["config"]
    rnd    = STATE["current_round"]
    nid    = STATE["node_id"]
    dp     = config["privacy"]

    log.info(f'"Starting round {rnd + 1}"')
    STATE.setdefault("round_start_times", {})[rnd] = time.time()

    params = local_train(
        STATE["model"], STATE["loader"],
        epochs=config["model"]["local_epochs"],
        lr=config["model"]["lr"],
        device=STATE["device"]
    )

    # DP noise on non-fc2 layers
    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # ZKP proof on fc2 weights (real Groth16, norm-bound circuit)
    from distributed_simulation.zkp_math import quantize, norm_sq_int, threshold_sq_int
    fc2_flat = noised["fc2.weight"].flatten().tolist()
    C_thresh = 0.5
    fc2_fr = quantize(fc2_flat)
    ns = norm_sq_int(fc2_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    if slack < 0:
        log.warning(f'"ZKP norm violation (clipping applied): slack={slack}"')
        fc2_arr = noised["fc2.weight"].flatten()
        norm = np.linalg.norm(fc2_arr)
        if norm > 0.5:
            fc2_arr = fc2_arr * (0.45 / norm)
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        fc2_flat = fc2_arr.tolist()
        fc2_fr = quantize(fc2_flat)
        ns = norm_sq_int(fc2_fr)
        slack = bound - ns
    _prove_result = {}
    def _run_prove():
        sys.setrecursionlimit(100000)
        _prove_result["proof"] = zkp_prove_v2(STATE["zkp_pk"], STATE["zkp_dim"], fc2_fr, slack, bound, rnd)
    threading.stack_size(64 * 1024 * 1024)
    _t = threading.Thread(target=_run_prove)
    _t.start()
    _t.join()


    proof = _prove_result["proof"]
    commitment = {"proof": proof.to_json()}

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

    # ── Broadcast the proposal BEFORE forwarding the real update ───
    # This fixes the hash all nodes will vote against, once, so it
    # can't drift as the payload mutates hop-to-hop.
    fixed_hash = compute_update_hash(
        commitment.get("proof", {}),
        {"enc_fc2_len": len(payload["enc_fc2"]), "plain_len": len(payload["plain"])},
    )
    proposal = UpdateProposal(
        round_id=rnd,
        origin_node_id=nid,
        sender_node_id=nid,
        update_hash=fixed_hash,
        zkp_public_inputs=getattr(proof, "public_inputs", []),
    )
    _broadcast_proposal(proposal)
    payload["proposal_hash"] = fixed_hash  # carried through the ring so every hop can look up the same proposal

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

    # ── ZKP verification (real Groth16 pairing check) ───────────
    from distributed_simulation.prover_v2 import Groth16ProofV2
    commitment = data.get("commitment", {})
    zkp_ok = True
    if commitment:
        proof_obj = Groth16ProofV2.from_json(commitment["proof"])
        _verify_result = {}
        def _run_verify():
            sys.setrecursionlimit(100000)
            _verify_result["ok"] = zkp_verify_v2(STATE["zkp_vk"], proof_obj)
        threading.stack_size(64 * 1024 * 1024)
        _t = threading.Thread(target=_run_verify)
        _t.start()
        _t.join()
        zkp_ok = _verify_result["ok"]
        if zkp_ok:
            log.info(f'"ZKP ACCEPTED from node {sender}"')
        else:
            log.warning(f'"ZKP REJECTED from node {sender}: pairing check failed"')

    # ── Cast and broadcast a consensus vote on this update ──────
    # Use the proposal's fixed hash (agreed BEFORE the ring update
    # arrived) instead of recomputing from the payload, which mutates
    # at every hop and would otherwise put every vote in a different
    # bucket.
    proposal_key = (rnd, origin)
    proposal = STATE["pending_proposals"].get(proposal_key)
    if proposal is not None:
        update_hash = proposal.update_hash
    else:
        # Proposal hasn't arrived yet (race) or is genuinely missing —
        # fall back to local computation so we still cast SOME vote,
        # but this will very likely mismatch and get rejected below,
        # which is the safe failure mode.
        update_hash = data.get("proposal_hash") or compute_update_hash(
            commitment.get("proof", {}) if commitment else {},
            {"enc_fc2_len": len(data.get("enc_fc2", "")), "plain_len": len(data.get("plain", ""))},
        )
        log.warning(
            f'"No stored proposal for round={rnd} origin={origin} — '
            f'voting using fallback hash, may not match other peers"'
        )
    decision, reason = evaluate_update_for_vote(
        zkp_ok=zkp_ok,
        expected_round=STATE["current_round"],
        received_round=rnd,
        expected_hash=proposal.update_hash if proposal is not None else None,
        received_hash=update_hash,
    )

    my_vote = make_vote(
        STATE["consensus_privkey"], round_id=rnd, update_hash=update_hash,
        voter_node_id=nid, decision=decision, reason_code=reason,
    )
    _broadcast_vote(my_vote)

    if not zkp_ok:
        _forward(data)  # skip: forward unchanged (Fix-1 protocol)
        return

    # ── Local training ────────────────────────────────────────
    params = local_train(
        STATE["model"], STATE["loader"],
        epochs=config["model"]["local_epochs"],
        lr=config["model"]["lr"],
        device=STATE["device"]
    )
    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # ── ZKP proof (real Groth16, norm-bound circuit) ────────────
    from distributed_simulation.zkp_math import quantize, norm_sq_int, threshold_sq_int
    fc2_flat = noised["fc2.weight"].flatten().tolist()
    C_thresh = 0.5
    fc2_fr = quantize(fc2_flat)
    ns = norm_sq_int(fc2_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    if slack < 0:
        fc2_arr = noised["fc2.weight"].flatten()
        fc2_arr = fc2_arr * (0.45 / np.linalg.norm(fc2_arr))
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        fc2_flat = fc2_arr.tolist()
        fc2_fr = quantize(fc2_flat)
        ns = norm_sq_int(fc2_fr)
        slack = bound - ns
    _prove_result2 = {}
    def _run_prove2():
        sys.setrecursionlimit(100000)
        _prove_result2["proof"] = zkp_prove_v2(STATE["zkp_pk"], STATE["zkp_dim"], fc2_fr, slack, bound, rnd)
    threading.stack_size(64 * 1024 * 1024)
    _t2 = threading.Thread(target=_run_prove2)
    _t2.start()
    _t2.join()
    proof = _prove_result2["proof"]
    my_commitment = {"proof": proof.to_json()}

    # ── HE aggregation ────────────────────────────────────────
    he         = STATE["he"]
    n_samples  = len(STATE["loader"].dataset)
    inc_enc = deserialize_enc(data["enc_fc2"])
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

def _broadcast_weights(params: dict):
    """Master broadcasts aggregated weights to all worker nodes after each round."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    nid    = STATE["node_id"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    payload = {"params": {k: v.tolist() for k, v in params.items()},
               "round":  STATE["current_round"]}
    for node in nodes:
        if node["id"] == nid:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/sync_weights"
        try:
            sess.post(url, json=payload, timeout=10)
            log.info(f'"Broadcast weights to node {node["id"]}"')
        except Exception as e:
            log.warning(f'"Broadcast failed to node {node["id"]}: {e}"')

def _finalize_round(data: dict):
    """Master: check quorum was reached on this update, then decrypt
    aggregated fc2, update model, log accuracy. If quorum was NOT
    reached, the update is not applied — the round is either retried
    (up to max_round_retries) or, if retries are exhausted, skipped
    so the ring doesn't stall forever."""
    log    = STATE["logger"]
    he     = STATE["he"]
    model  = STATE["model"]
    rnd    = data["round"]

    # ── Quorum gate ──────────────────────────────────────────────
    origin = data["origin_id"]
    proposal = STATE["pending_proposals"].get((rnd, origin))
    if proposal is not None:
        update_hash = proposal.update_hash
    else:
        update_hash = data.get("proposal_hash") or compute_update_hash(
            data.get("commitment", {}).get("proof", {}) if data.get("commitment") else {},
            {"enc_fc2_len": len(data.get("enc_fc2", "")), "plain_len": len(data.get("plain", ""))},
        )
        log.warning(
            f'"No stored proposal for round={rnd} origin={origin} at finalize — '
            f'using fallback hash"'
        )
    key = (rnd, update_hash)
    tracker = STATE["quorum_trackers"].get(key)
    quorum_ok = tracker.is_satisfied if tracker is not None else False

    if not quorum_ok:
        attempts = STATE["round_retry_count"].get(rnd, 0)
        if attempts < STATE["max_round_retries"]:
            STATE["round_retry_count"][rnd] = attempts + 1
            log.warning(
                f'"Round {rnd + 1} REJECTED — quorum not satisfied '
                f'(attempt {attempts + 1}/{STATE["max_round_retries"]}). Retrying round."'
            )
            time.sleep(0.5)
            threading.Thread(target=execute_round, daemon=True).start()
            return
        else:
            log.error(
                f'"Round {rnd + 1} REJECTED — quorum not satisfied after '
                f'{STATE["max_round_retries"]} retries. Skipping round, model NOT updated."'
            )
            STATE["round_retry_count"].pop(rnd, None)
            STATE["current_round"] += 1
            if STATE["current_round"] < STATE["config"]["ring"]["rounds"]:
                time.sleep(0.5)
                threading.Thread(target=execute_round, daemon=True).start()
            else:
                log.info(f'"Training complete after {STATE["current_round"]} rounds"')
                print(f"[Node {STATE['node_id']}] Training complete!")
            return

    # Quorum satisfied — clear retry count for this round and proceed
    STATE["round_retry_count"].pop(rnd, None)

    # Decrypt fc2
    enc_fc2 = deserialize_enc(data["enc_fc2"])
    plain    = deserialize_params(data["plain"])

    fc2_w = he.decrypt(enc_fc2["fc2.weight"]).reshape(
        STATE["model"].fc2.weight.shape)
    fc2_b = he.decrypt(enc_fc2["fc2.bias"])

    new_params = {**plain, "fc2.weight": fc2_w, "fc2.bias": fc2_b}
    set_params(model, new_params)
    _broadcast_weights(new_params)

    # Save checkpoint for dashboard
    try:
        import pathlib
        ckpt_path = ROOT / "dashboard" / "model_latest.pt"
        torch.save(model.state_dict(), ckpt_path)
        log.info(f'"Checkpoint saved to {ckpt_path}"')
    except Exception as e:
        log.warning(f'"Checkpoint save failed: {e}"')    

    # Evaluate
    acc = evaluate(model, STATE["test_loader"], STATE["device"])
    start_t = STATE.get("round_start_times", {}).pop(rnd, None)
    elapsed = (time.time() - start_t) if start_t is not None else float("nan")
    log.info(f'"Round {rnd + 1} complete | accuracy={acc:.4f} | duration={elapsed:.2f}s"')
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

    if args.dev and args.config == "config.json":
        args.config = "config_dev.json"

    config  = load_config(args.config)

    config  = load_config(args.config)
    threading.stack_size(64 * 1024 * 1024)  # fixes py_ecc recursive pow() stack overflow on Windows
    nid     = args.node_id = args.id
    nodes   = config["ring"]["nodes"]
    node_cfg = nodes[nid]
    logger  = setup_logger(nid, config["audit"]["log_dir"])

    node_name = node_cfg["name"]
    logger.info(f'"Starting node {nid}: {node_name}"')

    # ── ZKP setup (real Groth16, norm-bound circuit) ────────────
    # Load the SHARED trusted setup from disk (generated once via
    # generate_setup.py and copied to every node) instead of each
    # node generating its own — otherwise pk/vk would be incompatible
    # across nodes and every proof would fail verification.
    from distributed_simulation.trusted_setup import load_setup
    zkp_cfg = config["zkp"]
    setup_path = zkp_cfg.get("setup_file", "zkp_setup.json")
    zkp_pk, zkp_vk = load_setup(setup_path)
    STATE["zkp_pk"] = zkp_pk
    STATE["zkp_vk"] = zkp_vk
    STATE["zkp_dim"] = zkp_cfg["gradient_dim"]
    STATE["zkp_ready"] = True

    # ── Data ────────────────────────────────────────────────────
    loaders, test_loader, NORM_MEAN, NORM_STD = load_diabetes_datasets(
        num_clients=len(nodes),
        alpha=0.5,
        seed=42
    )
    my_loader = loaders[nid]

    # ── Model ───────────────────────────────────────────────────
    device = torch.device("cpu")
    mcfg   = config["model"]
    model  = DiabetesNet(
        input_dim=mcfg["input_dim"],
        num_classes=mcfg["output_dim"]
    ).to(device)

    # ── Consensus signing keys ─────────────────────────────────
    # Reuses each node's existing mTLS client key for real deployments.
    # Dev mode has no certs at all, so generate a throwaway in-memory
    # RSA key instead — fine for local testing, NOT for production
    # (dev-mode votes aren't tied to the node's real identity cert).
    peer_pubkeys = {}
    if args.dev:
        from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
        my_private_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
        peer_pubkeys = {}  # filled in via /consensus/pubkey exchange at ring start
    else:
        my_private_key = load_private_key(config["tls"]["client_key"])
        for n in nodes:
            if n["id"] == nid:
                continue
            # NOTE: all nodes currently share one client cert (see
            # generate_certs.py's "shared node certificate"), so this
            # verifies "signed by a valid ring member" rather than
            # cryptographically distinguishing WHICH member signed it.
            # Fine for Stage 1 hard-rejection logic; revisit if per-node
            # signer attribution becomes a requirement later.
            peer_pubkeys[n["id"]] = load_public_key_from_cert(config["tls"]["client_cert"])

    STATE["consensus_privkey"] = my_private_key
    STATE["consensus_peer_pubkeys"] = peer_pubkeys
    STATE["quorum_trackers"] = {}   # (round_id, update_hash) -> QuorumTracker
    STATE["quorum_required"] = 2    # 2-of-3 for this ring size
    STATE["round_retry_count"] = {} # round_id -> number of quorum-failed retries so far
    STATE["max_round_retries"] = 3  # after this many failed quorum attempts, skip the round instead of retrying forever
    STATE["pending_proposals"] = {} # (round_id, origin_node_id) -> UpdateProposal, sent before the ring update arrives

    # ── State ───────────────────────────────────────────────────
    STATE.update({
        "node_id":     nid,
        "dev_mode": args.dev,
        "config":      config,
        "model":       model,
        "loader":      my_loader,
        "test_loader": test_loader,
        "he":          SecureAggregator(nid, [n["id"] for n in nodes]),
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
    if nid != 0:
        threading.Timer(5.0, _exchange_keys).start()
        threading.Timer(5.0, _exchange_consensus_pubkeys).start()
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
