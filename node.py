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
sys.setrecursionlimit(1000000)
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
    sha256_hex,
)
from consensus_engine import (
    sign_payload, verify_signature, make_vote, verify_vote,
    compute_update_hash, QuorumTracker, evaluate_update_for_vote,
    load_private_key, load_public_key_from_cert,
)


def compute_reference_predictions(model, test_loader, device) -> list:
    """Run the model on the shared, fixed test set (same 110 patients on
    every node, same seed=42 split) and return flattened class-1
    probabilities. This is the 'reference set' from the design doc
    (Section 8) — used to measure behavioral agreement between nodes,
    not for training or accuracy reporting."""
    model.eval()
    probs_all = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(diabetic)
            probs_all.extend(probs.cpu().numpy().tolist())
    return probs_all


def prediction_agreement(probs_a: list, probs_b: list) -> float:
    """A_ij from the design doc: 1 - mean absolute difference between
    two prediction vectors over the same reference set. 1.0 = perfect
    agreement, 0.0 = maximal disagreement."""
    if not probs_a or not probs_b or len(probs_a) != len(probs_b):
        return 0.0
    diffs = [abs(a - b) for a, b in zip(probs_a, probs_b)]
    return 1.0 - (sum(diffs) / len(diffs))


def compute_model_hash(params: dict, round_id: int) -> str:
    """Deterministic hash of the model's parameters + round id, used
    for Stage 2 global-model commit consensus. All nodes must compute
    this the SAME way from the SAME resulting params to agree."""
    flat = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in params.items()}
    combined = canonical_json({"round_id": round_id, "params": flat})
    return sha256_hex(combined.encode("utf-8"))


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


def _record_suspicion(node_id: int, log, weight: float = 1.0) -> None:
    """Record a LOCAL suspicion strike this node personally observed
    against node_id. This does NOT unilaterally exclude anyone —
    a single node's own judgment is not sufficient grounds to remove
    a peer, since if THIS node happens to be the compromised one,
    its own reference point (its locally-held fc2) is corrupted and
    it will systematically misjudge every honest peer as suspicious.
    Instead, broadcast this node's own strike-count as a SIGNED
    ACCUSATION so peers can independently corroborate it. Exclusion
    only happens once a quorum of *different accusers* have each
    independently reached their own threshold against the same
    origin — see _handle_accusation()."""
    counts = STATE.setdefault("node_rejection_counts", {})
    counts[node_id] = counts.get(node_id, 0) + weight
    if (counts[node_id] >= STATE.get("exclusion_threshold", 5)
            and node_id not in STATE.get("excluded_nodes", set())
            and node_id not in STATE.get("accused_by_me", set())):
        STATE.setdefault("accused_by_me", set()).add(node_id)
        log.warning(
            f'"This node independently suspects node {node_id} '
            f'(local strikes={counts[node_id]:.1f}) — broadcasting '
            f'accusation for peer corroboration, NOT excluding unilaterally"'
        )
        _broadcast_accusation(node_id)


def _handle_accusation(accuser_id: int, accused_id: int, log) -> None:
    """Record that accuser_id has independently accused accused_id.
    Only exclude accused_id once a quorum of DISTINCT accusers (not
    counting the accused themselves) have each raised this — so one
    compromised node's skewed judgment can never unilaterally remove
    an honest peer. This mirrors the same 2-of-3 principle already
    used for update-admission and commit voting, applied here to
    exclusion decisions."""
    if accused_id == STATE["node_id"]:
        # Don't let a peer's accusation immediately doom us without
        # our own corroborating view — still record it, but the
        # quorum check below already requires multiple distinct
        # accusers, so this is handled the same as any other origin.
        pass
    accusers = STATE.setdefault("accusations", {}).setdefault(accused_id, set())
    accusers.add(accuser_id)
    # With M=3 total nodes, excluding the accused leaves 2 possible
    # accusers. Require BOTH of them (not just 1) to agree before
    # excluding — this is the actual fix: a single node's suspicion,
    # even if broadcast, can never be enough on its own. If the ring
    # ever grows beyond 3 nodes, this keeps requiring all remaining
    # non-accused nodes to agree, which is conservative (favors
    # safety over responsiveness) and can be relaxed later once this
    # is validated to work correctly at M=3.
    total_other_nodes = len(STATE["config"]["ring"]["nodes"]) - 1
    corroboration_quorum = max(2, total_other_nodes)
    if (len(accusers) >= corroboration_quorum
            and accused_id not in STATE.get("excluded_nodes", set())):
        log.error(
            f'"Node {accused_id} accused by {len(accusers)} independent '
            f'peers ({sorted(accusers)}) — quorum-corroborated, excluding."'
        )
        _apply_exclusion(accused_id, log)
        _broadcast_exclusion(accused_id)
    else:
        log.warning(
            f'"Node {accused_id} accused by node {accuser_id}, but only '
            f'{len(accusers)}/{corroboration_quorum} distinct accusers so far '
            f'— NOT excluding yet (avoids a single compromised node '
            f'unilaterally removing an honest peer)."'
        )


def _broadcast_accusation(accused_id: int) -> None:
    """Tell peers this node independently suspects accused_id, so they
    can corroborate (or not) with their own independent judgment
    before anyone is actually excluded. Signed with this node's own
    consensus key (same RSA-PSS scheme as votes) so a non-member
    can't inject a spoofed accusation to manipulate the corroboration
    quorum."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    nid    = STATE["node_id"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    payload = {"accuser_id": nid, "accused_id": accused_id}
    signature = sign_payload(STATE["consensus_privkey"], payload)
    excluded = STATE.get("excluded_nodes", set())
    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        url = f'{scheme}://{node["ip"]}:{node["port"]}/consensus/accuse'
        _send_or_exclude(node, url, {**payload, "signature": signature}, log, "Sent accusation")
    # Also apply corroboration check locally for our own accusation
    _handle_accusation(nid, accused_id, log)

def _apply_exclusion(node_id: int, log) -> None:
    """Locally mark a node excluded and adjust quorum. Called both
    when THIS node decides to exclude someone, and when it receives
    an exclusion notice from a peer — so all nodes converge to the
    same excluded_nodes set regardless of who detected it first."""
    STATE.setdefault("excluded_nodes", set()).add(node_id)
    remaining = len(STATE["config"]["ring"]["nodes"]) - len(STATE["excluded_nodes"])
    STATE["quorum_required"] = max(1, remaining - 1)
    log.error(
        f'"Node {node_id} EXCLUDED from proposing and quorum. '
        f'Quorum requirement adjusted to {STATE["quorum_required"]} '
        f'of {remaining} remaining trusted nodes."'
    )


def _send_or_exclude(node: dict, url: str, payload: dict, log, label: str, timeout: int = 10) -> bool:
    """Shared helper for all broadcast/notify functions: sends one POST
    to one peer, and if it fails, marks that peer excluded + broadcasts
    the exclusion — so ANY network call to a dead node, anywhere in the
    ring's code, self-heals instead of leaving that peer half-excluded
    (some functions knowing it's dead, others still waiting on it)."""
    sess = STATE["session"]
    try:
        sess.post(url, json=payload, timeout=timeout)
        log.info(f'"{label} to node {node["id"]}"')
        return True
    except Exception as e:
        log.warning(f'"{label} to node {node["id"]} FAILED: {e}"')
        if node["id"] not in STATE.get("excluded_nodes", set()):
            _apply_exclusion(node["id"], log)
            _broadcast_exclusion(node["id"])
        return False


def _broadcast_exclusion(node_id: int) -> None:
    """Tell every peer to also mark node_id as excluded, so
    get_proposer_for_round() agrees across the whole ring."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    nid    = STATE["node_id"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    for node in nodes:
        if node["id"] == nid or node["id"] == node_id:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/node_excluded"
        try:
            sess.post(url, json={"excluded_node": node_id}, timeout=10)
            log.info(f'"Sent exclusion notice for node {node_id} to node {node["id"]}"')
        except Exception as e:
            log.warning(f'"Failed to send exclusion notice to node {node["id"]}: {e}"')
            # deliberately NOT calling _send_or_exclude here — would recurse
            # into _broadcast_exclusion while already inside it


def get_proposer_for_round(round_id: int) -> int:
    """Leader(t) = t mod M, adjusted to skip excluded nodes. Every
    node computes this identically (same round_id, same excluded_nodes
    set, since exclusion decisions are made deterministically from
    events every node observes — accuracy-gate rejections tied to a
    specific origin). If a node is excluded, its turn is skipped and
    the next non-excluded node in rotation order takes it instead."""
    num_nodes = len(STATE["config"]["ring"]["nodes"])
    excluded = STATE.get("excluded_nodes", set())
    for offset in range(num_nodes):
        candidate = (round_id + offset) % num_nodes
        if candidate not in excluded:
            return candidate
    # All nodes excluded (should never happen) — fall back to raw formula
    return round_id % num_nodes


def get_successor_url() -> str:
    """Find the next node in ring order, skipping any node marked
    excluded_nodes — which now covers BOTH consensus-driven exclusion
    (misbehavior, corroborated by peers) AND simple unreachability
    (connection refused/timeout — see _forward()'s failure handling).
    Without this skip, a single offline peer would permanently break
    ring forwarding even though the other nodes are still healthy and
    reachable — exactly the stall observed in one-node-offline testing."""
    nodes    = STATE["config"]["ring"]["nodes"]
    nid      = STATE["node_id"]
    n        = len(nodes)
    excluded = STATE.get("excluded_nodes", set())
    scheme   = "http" if STATE.get("dev_mode") else "https"
    for offset in range(1, n + 1):
        candidate_id = (nid + offset) % n
        if candidate_id == nid:
            break  # wrapped all the way around — no live successor
        if candidate_id not in excluded:
            succ = nodes[candidate_id]
            return f"{scheme}://{succ['ip']}:{succ['port']}"
    # Every other node is excluded/unreachable — fall back to raw
    # next-in-ring so callers get a clear connection error rather
    # than a silent None, which would be a confusing failure mode.
    succ = nodes[(nid + 1) % n]
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

    excluded = STATE.get("excluded_nodes", set())
    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/propose"
        ok = _post_with_retry(
            sess, url, proposal.to_dict(),
            log, "Sent proposal", node["id"],
        )
        if not ok and node["id"] not in STATE.get("excluded_nodes", set()):
            _apply_exclusion(node["id"], log)
            _broadcast_exclusion(node["id"])


def _broadcast_vote(vote: "ConsensusVote", prediction_vector: list = None):
    """Send our signed vote to every peer, and also record it in our
    own local QuorumTracker (a node's vote counts for itself too).
    prediction_vector (if present) rides alongside as UNSIGNED extra
    data — used only for logging/scoring agreement (Section 8), never
    for security decisions, so it doesn't touch the signed payload."""
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

    payload = vote.to_dict()
    if prediction_vector is not None:
        payload["_prediction_vector"] = prediction_vector  # unsigned, informational only

    excluded = STATE.get("excluded_nodes", set())
    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/vote"
        _send_or_exclude(node, url, payload, log, "Sent consensus vote")

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
    """Runs key exchange + wait, then kicks off the training ring —
    but ONLY on the node whose turn it actually is for round 0, per
    get_proposer_for_round(). Every node runs this same bootstrap
    logic (key exchange must complete everywhere regardless of who
    proposes), but only the round-0 proposer actually starts training;
    everyone else just waits to receive the ring's first update as normal."""
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

    proposer = get_proposer_for_round(STATE["current_round"])
    if STATE["node_id"] == proposer:
        log.info(f'"This node is round {STATE["current_round"] + 1}\'s proposer — starting."')
        threading.Thread(target=execute_round, daemon=True).start()
    else:
        log.info(
            f'"Key exchange complete. Node {proposer} is round '
            f'{STATE["current_round"] + 1}\'s proposer — waiting to receive."'
        )
        threading.Thread(target=_proposal_watchdog, args=(STATE["current_round"], proposer), daemon=True).start()


@app.post("/start_ring")
async def start_ring():
    # Any node may send this one-time bootstrap trigger for round 0 —
    # it doesn't grant special status. Whoever actually PROPOSES round 0
    # is decided independently by get_proposer_for_round(0), which every
    # node computes the same way (round 0 % num_nodes = node 0, by the
    # rotation formula itself), regardless of who fired this trigger.
    threading.Thread(target=_start_ring_background, daemon=True).start()
    return {"status": "Ring start triggered — key exchange running in background"}

@app.post("/your_turn")
async def your_turn(request: Request):
    """This node has been notified it's the proposer for the given
    round — start proposing. Only acts if we agree it's actually our
    turn (defensive check against a stale/duplicate notification)."""
    data = await request.json()
    rnd = data["round"]
    expected_proposer = get_proposer_for_round(rnd)
    if expected_proposer != STATE["node_id"]:
        STATE["logger"].warning(
            f'"Received your_turn for round {rnd} but proposer should be '
            f'node {expected_proposer}, not me — ignoring."'
        )
        return {"status": "ignored"}
    STATE["logger"].info(f'"My turn — starting round {rnd + 1} as proposer."')
    threading.Thread(target=execute_round, daemon=True).start()
    return {"status": "starting"}


@app.post("/node_excluded")
async def node_excluded(request: Request):
    """Receive an exclusion notice from a peer and apply it locally,
    so every node's excluded_nodes set converges — required for
    get_proposer_for_round() to agree across the whole ring."""
    data = await request.json()
    excluded_id = data["excluded_node"]
    if excluded_id not in STATE.get("excluded_nodes", set()):
        _apply_exclusion(excluded_id, STATE["logger"])
    return {"status": "ok"}

@app.post("/consensus/accuse")
async def consensus_accuse(request: Request):
    """Receive an accusation from a peer that it independently
    suspects some node. Does NOT exclude on its own — just records
    the accusation and checks whether enough DISTINCT peers have now
    accused the same node to reach corroboration quorum. Signature
    is verified against the claimed accuser's known consensus public
    key first — an unsigned or wrongly-signed accusation is dropped,
    so a non-member can't inject a fake accusation to manipulate the
    corroboration count."""
    log  = STATE["logger"]
    data = await request.json()
    accuser_id = data["accuser_id"]
    accused_id = data["accused_id"]
    signature  = data.get("signature")

    accuser_pub = STATE["consensus_peer_pubkeys"].get(accuser_id)
    payload = {"accuser_id": accuser_id, "accused_id": accused_id}
    if (accuser_pub is None or signature is None
            or not verify_signature(accuser_pub, payload, signature)):
        log.warning(
            f'"Accusation from node {accuser_id} against node {accused_id} '
            f'REJECTED — bad/missing signature or unknown accuser key"'
        )
        return JSONResponse({"status": "rejected"}, status_code=400)

    _handle_accusation(accuser_id, accused_id, log)
    return {"status": "accusation recorded"}

@app.post("/round_skipped")
async def round_skipped(request: Request):
    """Worker receives notice that master skipped a round after
    exhausting retries. Advance our round counter to match, so we
    don't reject the next round's proposal as wrong_round forever."""
    data = await request.json()
    rnd = data["round"]
    STATE["current_round"] = rnd + 1
    STATE["logger"].info(f'"Round {rnd + 1} skip acknowledged — advancing to round {rnd + 2}"')
    return {"status": "ok"}

@app.post("/sync_weights")
async def sync_weights(request: Request):
    """Workers receive aggregated weights from master after each round.
    Also advances this worker's own round counter — otherwise workers
    never learn a round finished, and every subsequent round would be
    rejected as wrong_round forever.

    Stage 2: after applying the weights, independently compute the
    resulting model's hash and send a signed commit vote back to the
    master, so divergence between nodes becomes detectable instead of
    silently assumed away."""
    data   = await request.json()
    params = {k: np.array(v, dtype=np.float32) for k, v in data["params"].items()}
    set_params(STATE["model"], params)
    rnd = data["round"]
    STATE["current_round"] = rnd + 1
    STATE["logger"].info(f'"Synced weights from master for round {rnd}"')

    # ── Stage 2: vote on the resulting model hash ───────────────
    hashable_params = {k: v.tolist() for k, v in params.items()}
    my_hash = compute_model_hash(hashable_params, rnd)
    master_hash = data.get("model_hash")
    decision = (master_hash is not None and my_hash == master_hash)
    reason = RejectionReason.OK if decision else RejectionReason.HASH_MISMATCH
    if not decision:
        STATE["logger"].warning(
            f'"MODEL HASH MISMATCH at round {rnd}: mine={my_hash[:12]}... '
            f'master={str(master_hash)[:12]}..."'
        )
    commit_vote = make_vote(
        STATE["consensus_privkey"], round_id=rnd, update_hash=my_hash,
        voter_node_id=STATE["node_id"], decision=decision, reason_code=reason,
    )
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    # Send the commit vote to THIS ROUND's proposer, not always node 0 —
    # under rotation, whoever proposed round `rnd` is the one running
    # commit-vote collection and deciding whether the round is finalized.
    proposer_id = get_proposer_for_round(rnd)
    proposer_node = next(n for n in nodes if n["id"] == proposer_id)
    url = f"{scheme}://{proposer_node['ip']}:{proposer_node['port']}/consensus/commit_vote"
    _post_with_retry(sess, url, commit_vote.to_dict(), STATE["logger"], "Sent commit vote", proposer_id, attempts=3, delay_s=1)

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

def _save_commit_certificate(cert: "QuorumCertificate"):
    """Persist the commit certificate to disk beside model_latest.pt,
    per the design doc (Section 10.1). One JSON file per round, so the
    dashboard/report can show a full audit trail of finalized rounds."""
    log = STATE["logger"]
    try:
        cert_dir = ROOT / "dashboard" / "certificates"
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert_path = cert_dir / f"round_{cert.round_id}.json"
        with open(cert_path, "w") as f:
            json.dump(cert.to_dict(), f, indent=2)
        log.info(f'"Commit certificate saved to {cert_path}"')
    except Exception as e:
        log.warning(f'"Failed to save commit certificate for round {cert.round_id}: {e}"')


@app.post("/consensus/commit_vote")
async def consensus_commit_vote(request: Request):
    """Stage 2: master receives a peer's signed vote on the global
    model hash. Once quorum agrees on the SAME hash, the round is
    considered finalized/committed."""
    data = await request.json()
    vote = ConsensusVote.from_dict(data)

    voter_pub = STATE["consensus_peer_pubkeys"].get(vote.voter_node_id)
    if voter_pub is None or not verify_vote(voter_pub, vote):
        STATE["logger"].warning(
            f'"Commit vote from node {vote.voter_node_id} rejected — bad/unknown key"'
        )
        return JSONResponse({"status": "rejected"}, status_code=400)

    tracker = STATE["commit_trackers"].get(vote.round_id)
    if tracker is None:
        tracker = QuorumTracker(vote.round_id, vote.update_hash, STATE["quorum_required"])
        STATE["commit_trackers"][vote.round_id] = tracker
    tracker.add_vote(vote)
    STATE["logger"].info(
        f'"Commit vote recorded: node {vote.voter_node_id} decision={vote.decision} '
        f'round={vote.round_id} satisfied={tracker.is_satisfied}"'
    )

    if tracker.is_satisfied and vote.round_id not in STATE["commit_certificates"]:
        cert = tracker.certificate()
        STATE["commit_certificates"][vote.round_id] = cert
        STATE["logger"].info(
            f'"Round {vote.round_id + 1} COMMITTED — model hash agreed by {cert.accepted_by}"'
        )
        _save_commit_certificate(cert)

    return {"status": "ok", "satisfied": tracker.is_satisfied}


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

    if not verify_vote(voter_pub, vote):
        STATE["logger"].warning(
            f'"Consensus vote from node {vote.voter_node_id} FAILED signature check"'
        )
        return JSONResponse({"status": "rejected", "reason": RejectionReason.BAD_SIGNATURE}, status_code=400)

    # ── Prediction-based agreement (Section 8) ──────────────────
    # Store cross-peer agreement scores per (round, update_hash) so
    # the master can add a SECOND consensus gate at finalize time —
    # catching cases where individual votes look fine but peers'
    # models have actually diverged from each other (e.g. a stalled
    # or slowly-poisoned round that any single node's own-history
    # check wouldn't flag).
    peer_pred_vector = data.get("_prediction_vector")
    if peer_pred_vector is not None:
        my_pred_vector = STATE.get("last_prediction_vector", {}).get(
            (vote.round_id, vote.update_hash)
        )
        if my_pred_vector is not None:
            agreement = prediction_agreement(my_pred_vector, peer_pred_vector)
            STATE["logger"].info(
                f'"Prediction agreement with node {vote.voter_node_id} '
                f'round={vote.round_id}: A={agreement:.3f}"'
            )
            STATE.setdefault("cross_peer_agreement", {}).setdefault(
                (vote.round_id, vote.update_hash), []
            ).append(agreement)

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

    # Clip fc2 weights to the ZKP norm bound BEFORE adding DP noise,
    # so noise is layered onto an already-bounded value instead of
    # relying entirely on the post-noise clip below to fix things up.
    clip_C = dp.get("clip_threshold", 0.5)
    fc2_w_pre = params["fc2.weight"]
    fc2_pre_norm = np.linalg.norm(fc2_w_pre.flatten())
    if fc2_pre_norm > clip_C:
        params["fc2.weight"] = fc2_w_pre * (clip_C / fc2_pre_norm)

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
    retry = 0
    while slack < 0 and retry < 5:
        log.warning(f'"ZKP norm violation (clipping applied): slack={slack}, retry={retry}"')
        fc2_arr = noised["fc2.weight"].flatten()
        norm = np.linalg.norm(fc2_arr)
        if norm > 0:
            fc2_arr = fc2_arr * (0.4 / norm)  # target well under 0.5 to survive quantization rounding
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        fc2_flat = fc2_arr.tolist()
        fc2_fr = quantize(fc2_flat)
        ns = norm_sq_int(fc2_fr)
        slack = bound - ns
        retry += 1
    if slack < 0:
        raise RuntimeError(f"Could not bring update within ZKP norm bound after {retry} retries (slack={slack})")
    _prove_result = {}
    def _run_prove():
        sys.setrecursionlimit(1000000)
        try:
            _prove_result["proof"] = zkp_prove_v2(STATE["zkp_pk"], STATE["zkp_dim"], fc2_fr, slack, bound, rnd)
        except RecursionError as e:
            log.error(f'"ZKP prove thread crashed: {e}"')
    try:
        threading.stack_size(64 * 1024 * 1024)
    except (ValueError, RuntimeError):
        pass
    _t = threading.Thread(target=_run_prove)
    _t.start()
    _t.join()
    if "proof" not in _prove_result:
        raise RuntimeError("ZKP proof generation failed (recursion crash) — cannot proceed this round")


    proof = _prove_result["proof"]
    commitment = {"proof": proof.to_json()}

    # HE encrypt fc2
    he   = STATE["he"]
    enc  = {
        "fc2.weight": he.encrypt(noised["fc2.weight"].flatten()),
        "fc2.bias":   he.encrypt(noised["fc2.bias"])
    }

    # Cap each node's aggregation weight per design doc Section 9, so
    # no single hospital's data volume can dominate the aggregate. The
    # cap is a POLICY value from config — NOT derived from any specific
    # node's actual patient count, since in a real deployment nodes
    # should not need to reveal their exact dataset size to each other
    # just to keep aggregation fair.
    N_MAX = config["privacy"].get("max_aggregation_weight", 100)
    n_samples = min(len(STATE["loader"].dataset), N_MAX)

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

    # ── Origin also votes on its own update ─────────────────────
    # Without this, node 0 never participates in its own round's
    # quorum — only peers' votes counted, which is incomplete even
    # though quorum could still be reached with peers alone.
    my_own_vote = make_vote(
        STATE["consensus_privkey"], round_id=rnd, update_hash=fixed_hash,
        voter_node_id=nid, decision=True, reason_code=RejectionReason.OK,
    )
    _broadcast_vote(my_own_vote)

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
            sys.setrecursionlimit(1000000)
            try:
                _verify_result["ok"] = zkp_verify_v2(STATE["zkp_vk"], proof_obj)
            except RecursionError as e:
                log.error(f'"ZKP verify thread crashed: {e}"')
                _verify_result["ok"] = False
        try:
            threading.stack_size(64 * 1024 * 1024)
        except (ValueError, RuntimeError):
            pass
        _t = threading.Thread(target=_run_verify)
        _t.start()
        _t.join()
        zkp_ok = _verify_result.get("ok", False)
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
    # ── Prediction-based agreement (Section 8) — computed BEFORE
    # the vote decision, so it can actually gate accept/reject ──
    # Build a scratch model: incoming plaintext layers + OUR OWN
    # current fc2 (fc2 stays encrypted, we never see the origin's).
    # This measures whether the new plaintext layers behave sanely,
    # without breaking the HE boundary around fc2.
    my_pred_vector = None
    agreement_score = None
    if zkp_ok:
        try:
            inc_plain = deserialize_params(data["plain"])
            scratch = DiabetesNet(input_dim=STATE["model"].input_dim,
                                   num_classes=STATE["model"].num_classes).to(STATE["device"])
            scratch.load_state_dict(STATE["model"].state_dict())
            scratch_params = scratch.get_all_params()
            scratch_params.update(inc_plain)  # overwrite non-fc2 layers only
            scratch.set_all_params(scratch_params)
            my_pred_vector = compute_reference_predictions(scratch, STATE["test_loader"], STATE["device"])

            # Agreement against OUR OWN previous-round prediction vector —
            # the closest honest proxy available, since we never see the
            # origin's true predictions (its fc2 stays encrypted to us).
            # Keyed PER ORIGIN, not globally — a single shared slot mixed
            # different nodes' updates together (whoever proposed last
            # round vs whoever proposes this round), so agreement was
            # comparing across different senders rather than judging one
            # sender's own consistency. That made the signal track ring
            # order, not the actual attacker.
            per_origin_prev = STATE.setdefault("my_last_own_prediction_vector", {})
            prev_vector = per_origin_prev.get(origin)
            if prev_vector is not None:
                agreement_score = prediction_agreement(prev_vector, my_pred_vector)
        except Exception as e:
            log.warning(f'"Prediction-agreement check failed: {e}"')

    decision, reason = evaluate_update_for_vote(
        zkp_ok=zkp_ok,
        expected_round=STATE["current_round"],
        received_round=rnd,
        expected_hash=proposal.update_hash if proposal is not None else None,
        received_hash=update_hash,
    )

    # Gate on prediction agreement (tau=0.75 per design doc Section 8).
    # Only applies once we have a real prior vector to compare against —
    # the very first round for this peer can't be gated yet.
    PREDICTION_AGREEMENT_TAU = 0.75
    if decision and agreement_score is not None and agreement_score < PREDICTION_AGREEMENT_TAU:
        decision, reason = False, "prediction_disagreement"
        log.warning(
            f'"Update round={rnd} from node {origin} REJECTED on prediction '
            f'agreement: A={agreement_score:.3f} < tau={PREDICTION_AGREEMENT_TAU}"'
        )
        # ── Node-level accountability: this is a REAL per-contributor
        # signal (unlike the accuracy gate, which judges the whole
        # blended round) — origin here is specifically whose individual
        # update just failed behavioral agreement, observed independently
        # by this peer before any aggregation happens.
        _record_suspicion(origin, log)

    my_vote = make_vote(
        STATE["consensus_privkey"], round_id=rnd, update_hash=update_hash,
        voter_node_id=nid, decision=decision, reason_code=reason,
    )

    if my_pred_vector is not None:
        STATE.setdefault("last_prediction_vector", {})[(rnd, update_hash)] = my_pred_vector
        STATE.setdefault("my_last_own_prediction_vector", {})[origin] = my_pred_vector
    _broadcast_vote(my_vote, prediction_vector=my_pred_vector)

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

    # ── Attack simulation harness (config-driven, off by default) ──
    # Applies a configured attack to THIS node's own update before
    # DP/ZKP, if this node is the configured target. Controlled
    # entirely via config.json's "attack_simulation" block — no code
    # changes needed to enable/disable/retarget. Section 14.2.
    attack_cfg = STATE["config"].get("attack_simulation", {})
    if (attack_cfg.get("enabled", False)
            and nid == attack_cfg.get("target_node")
            and attack_cfg.get("type") == "sign_flip"):
        def _should_flip(k):
            return ("fc2" not in k and "running_mean" not in k
                    and "running_var" not in k and "num_batches" not in k)
        params = {k: (-v if _should_flip(k) else v) for k, v in params.items()}
        log.warning(
            f'"[ATTACK SIMULATION] sign_flip active on node {nid} — '
            f'flipped trainable params before DP/ZKP"'
        )

    # Clip fc2 weights to the ZKP norm bound BEFORE adding DP noise,
    # so noise is layered onto an already-bounded value instead of
    # relying entirely on the post-noise clip below to fix things up.
    clip_C = dp.get("clip_threshold", 0.5)
    fc2_w_pre = params["fc2.weight"]
    fc2_pre_norm = np.linalg.norm(fc2_w_pre.flatten())
    if fc2_pre_norm > clip_C:
        params["fc2.weight"] = fc2_w_pre * (clip_C / fc2_pre_norm)

    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # ── ZKP proof (real Groth16, norm-bound circuit) ────────────
    from distributed_simulation.zkp_math import quantize, norm_sq_int, threshold_sq_int
    fc2_flat = noised["fc2.weight"].flatten().tolist()
    C_thresh = 0.5
    fc2_fr = quantize(fc2_flat)
    ns = norm_sq_int(fc2_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    retry = 0
    while slack < 0 and retry < 5:
        log.warning(f'"ZKP norm violation (clipping applied): slack={slack}, retry={retry}"')
        fc2_arr = noised["fc2.weight"].flatten()
        norm = np.linalg.norm(fc2_arr)
        if norm > 0:
            fc2_arr = fc2_arr * (0.4 / norm)  # target well under 0.5 to survive quantization rounding
        noised["fc2.weight"] = fc2_arr.reshape(noised["fc2.weight"].shape)
        fc2_flat = fc2_arr.tolist()
        fc2_fr = quantize(fc2_flat)
        ns = norm_sq_int(fc2_fr)
        slack = bound - ns
        retry += 1
    if slack < 0:
        raise RuntimeError(f"Could not bring update within ZKP norm bound after {retry} retries (slack={slack})")
    _prove_result2 = {}
    def _run_prove2():
        sys.setrecursionlimit(1000000)
        try:
            _prove_result2["proof"] = zkp_prove_v2(STATE["zkp_pk"], STATE["zkp_dim"], fc2_fr, slack, bound, rnd)
        except RecursionError as e:
            log.error(f'"ZKP prove thread crashed: {e}"')
    try:
        threading.stack_size(64 * 1024 * 1024)
    except (ValueError, RuntimeError):
        pass
    _t2 = threading.Thread(target=_run_prove2)
    _t2.start()
    _t2.join()
    if "proof" not in _prove_result2:
        raise RuntimeError("ZKP proof generation failed (recursion crash) — cannot proceed this round")
    proof = _prove_result2["proof"]
    my_commitment = {"proof": proof.to_json()}

    # ── HE aggregation ────────────────────────────────────────
    he         = STATE["he"]
    N_MAX = config["privacy"].get("max_aggregation_weight", 100)
    n_samples  = min(len(STATE["loader"].dataset), N_MAX)
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
    """Send payload to successor node. On connection failure, mark
    that node as excluded_nodes (unreachable) so get_successor_url()
    and get_proposer_for_round() both route around it going forward —
    otherwise every future round keeps trying the same dead node and
    the whole ring stalls indefinitely, as seen in one-node-offline
    testing. This reuses the SAME excluded_nodes mechanism as
    consensus-driven exclusion, but for a structurally different
    reason (infrastructure failure, not misbehavior) — logged
    distinctly so the two causes aren't confused when reading logs."""
    log  = STATE["logger"]
    nid  = STATE["node_id"]
    nodes = STATE["config"]["ring"]["nodes"]
    n = len(nodes)
    succ_id = None
    excluded = STATE.get("excluded_nodes", set())
    for offset in range(1, n + 1):
        candidate_id = (nid + offset) % n
        if candidate_id == nid:
            break
        if candidate_id not in excluded:
            succ_id = candidate_id
            break
    url  = get_successor_url()
    sess = STATE["session"]
    try:
        sess.post(f"{url}/receive_update", json=payload, timeout=30)
        log.info(f'"Forwarded to {url}"')
    except Exception as e:
        log.error(f'"Forward failed to {url}: {e} — marking node '
                   f'{succ_id} as unreachable (NOT a consensus '
                   f'exclusion, just a connectivity failure)"')
        if succ_id is not None and succ_id not in excluded:
            STATE.setdefault("excluded_nodes", set()).add(succ_id)
            remaining = n - len(STATE["excluded_nodes"])
            log.warning(
                f'"Node {succ_id} marked unreachable. '
                f'{remaining} node(s) remain in the ring."'
            )

def _proposal_watchdog(rnd: int, expected_proposer: int):
    """Runs after handing off proposing duty to expected_proposer for
    round `rnd`. If no proposal for this round arrives within timeout_s
    (the proposer died mid-broadcast, or never started at all — a gap
    the failed-send exclusion checks can't catch, since nothing was
    ever sent TO us to fail), exclude expected_proposer and hand the
    round to whoever get_proposer_for_round() picks next instead.
    Cancels itself quietly if the round already moved on by the time
    the timeout fires (proposal arrived, or a peer's exclusion notice
    already advanced us)."""
    log = STATE["logger"]
    timeout_s = STATE["config"]["ring"].get("timeout_s", 30)
    time.sleep(timeout_s)

    # If the round already advanced, or the proposal did arrive, stand down.
    if STATE["current_round"] != rnd:
        return
    if any(p.round_id == rnd and p.origin_node_id == expected_proposer
           for p in STATE["pending_proposals"].values()):
        return
    if expected_proposer in STATE.get("excluded_nodes", set()):
        return

    log.warning(
        f'"No proposal received from node {expected_proposer} for round {rnd} '
        f'after {timeout_s}s — treating as unreachable/stalled."'
    )
    _apply_exclusion(expected_proposer, log)
    _broadcast_exclusion(expected_proposer)

    if STATE["current_round"] != rnd:
        return  # exclusion broadcast or a race already moved things on

    fallback = get_proposer_for_round(rnd)
    if fallback == STATE["node_id"]:
        log.info(f'"Taking over as proposer for round {rnd} after watchdog timeout."')
        threading.Thread(target=execute_round, daemon=True).start()
    else:
        log.info(f'"Handing round {rnd} to node {fallback} after watchdog timeout."')
        _notify_next_proposer(rnd, fallback)


def _notify_next_proposer(rnd: int, proposer_id: int, _depth: int = 0):
    """Tell the node whose turn it is next to start proposing —
    nothing else would wake them up, since a fresh proposer has no
    incoming ring traffic to react to yet; they must be told to act.
    If the notify fails entirely, the target is unreachable — mark it
    excluded (same mechanism as _forward()'s failure handling) and
    retry with whichever node get_proposer_for_round() now picks next,
    instead of stalling forever waiting for a dead node to propose.
    _depth guards against the pathological all-nodes-unreachable case."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    target = next(n for n in nodes if n["id"] == proposer_id)
    url = f"{scheme}://{target['ip']}:{target['port']}/your_turn"
    ok = _post_with_retry(sess, url, {"round": rnd}, log, "Notified next proposer", proposer_id, attempts=6, delay_s=2)

    if not ok:
        excluded = STATE.setdefault("excluded_nodes", set())
        if proposer_id not in excluded:
            excluded.add(proposer_id)
            remaining = len(nodes) - len(excluded)
            log.warning(
                f'"Node {proposer_id} unreachable for proposer handoff — '
                f'marked excluded. {remaining} node(s) remain in the ring."'
            )
        if _depth >= len(nodes):
            log.error('"All nodes unreachable for proposer handoff — cannot recover."')
            return
        fallback = get_proposer_for_round(rnd)
        if fallback == STATE["node_id"]:
            log.info(f'"Taking over as proposer for round {rnd} after handoff failure."')
            threading.Thread(target=execute_round, daemon=True).start()
        elif fallback != proposer_id:
            _notify_next_proposer(rnd, fallback, _depth=_depth + 1)


def _broadcast_skip(rnd: int):
    """Tell workers a round was skipped (quorum/accuracy gate failed
    after all retries) so they advance their round counter too —
    otherwise they stay stuck on the skipped round and reject every
    future proposal as wrong_round forever."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    nid    = STATE["node_id"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    excluded = STATE.get("excluded_nodes", set())
    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/round_skipped"
        _send_or_exclude(node, url, {"round": rnd}, log, "Sent skip notice")


def _broadcast_weights(params: dict, model_hash: str = None):
    """Master broadcasts aggregated weights to all worker nodes after each round.
    Includes the model_hash (Stage 2) so peers can independently verify
    they landed on the same resulting model."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    nid    = STATE["node_id"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    payload = {"params": {k: v.tolist() for k, v in params.items()},
               "round":  STATE["current_round"],
               "model_hash": model_hash}
    excluded = STATE.get("excluded_nodes", set())
    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        url = f"{scheme}://{node['ip']}:{node['port']}/sync_weights"
        _send_or_exclude(node, url, payload, log, "Broadcast weights")

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
    excluded = STATE.get("excluded_nodes", set())
    if tracker is not None and excluded:
        trusted_accepts = sum(
            1 for v in tracker._votes.values()
            if v.decision and v.voter_node_id not in excluded
        )
        quorum_ok = trusted_accepts >= STATE["quorum_required"]
    else:
        quorum_ok = tracker.is_satisfied if tracker is not None else False

    # ── Second gate: cross-peer prediction agreement ─────────────
    # Catches the case a single vote-count quorum can miss: peers
    # individually voting ACCEPT (their own round-over-round agreement
    # looks fine) while their actual models have diverged from each
    # other — e.g. a stalled/neutralized round from an attack that
    # doesn't show up as a sudden swing in any one node's history.
    CROSS_PEER_AGREEMENT_TAU = 0.75
    cross_scores = STATE.get("cross_peer_agreement", {}).get(key, [])
    if quorum_ok and cross_scores:
        min_cross_agreement = min(cross_scores)
        if min_cross_agreement < CROSS_PEER_AGREEMENT_TAU:
            log.warning(
                f'"Round {rnd + 1} cross-peer agreement too low '
                f'(min={min_cross_agreement:.3f} < tau={CROSS_PEER_AGREEMENT_TAU}) '
                f'— overriding quorum, treating as not satisfied"'
            )
            quorum_ok = False

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
            _broadcast_skip(rnd)
            if STATE["current_round"] < STATE["config"]["ring"]["rounds"]:
                next_proposer = get_proposer_for_round(STATE["current_round"])
                if next_proposer == STATE["node_id"]:
                    time.sleep(0.5)
                    threading.Thread(target=execute_round, daemon=True).start()
                else:
                    _notify_next_proposer(STATE["current_round"], next_proposer)
                    threading.Thread(target=_proposal_watchdog, args=(STATE["current_round"], next_proposer), daemon=True).start()
            else:
                log.info(f'"Training complete after {STATE["current_round"]} rounds"')
                print(f"[Node {STATE['node_id']}] Training complete!")
            return

    # Decrypt fc2
    enc_fc2 = deserialize_enc(data["enc_fc2"])
    plain    = deserialize_params(data["plain"])

    fc2_w = he.decrypt(enc_fc2["fc2.weight"]).reshape(
        STATE["model"].fc2.weight.shape)
    fc2_b = he.decrypt(enc_fc2["fc2.bias"])

    new_params = {**plain, "fc2.weight": fc2_w, "fc2.bias": fc2_b}

    # ── Third gate: accuracy sanity check ────────────────────────
    # Catches single-class collapse (e.g. sign-flip attack) that
    # agreement-based checks structurally cannot see — once every
    # node's model collapses together, they all "agree" with each
    # other, so this checks against ACTUAL LABELS on the held-out
    # set instead, before the candidate model is ever applied.
    scratch_model = DiabetesNet(input_dim=model.input_dim,
                                 num_classes=model.num_classes).to(STATE["device"])
    scratch_model.load_state_dict({
        k: torch.tensor(v, dtype=torch.float32) for k, v in new_params.items()
    })
    candidate_acc = evaluate(scratch_model, STATE["test_loader"], STATE["device"])

    prev_acc = STATE.get("last_committed_accuracy")
    ACC_FLOOR = 0.70          # above known single-class collapse points (68.18% / 31.82%)
    ACC_DROP_TOLERANCE = 0.15 # max allowed drop vs last committed round
    ACC_GATE_WARMUP_ROUNDS = 5  # let the model actually learn before judging it

    acc_ok = True
    if rnd < ACC_GATE_WARMUP_ROUNDS:
        log.info(
            f'"Round {rnd + 1} accuracy gate skipped (warm-up period, '
            f'candidate_acc={candidate_acc:.4f})"'
        )
    elif candidate_acc < ACC_FLOOR:
        acc_ok = False
        log.warning(
            f'"Round {rnd + 1} accuracy check FAILED: candidate_acc={candidate_acc:.4f} '
            f'below floor={ACC_FLOOR} — likely collapse, rejecting round"'
        )
    elif prev_acc is not None and (prev_acc - candidate_acc) > ACC_DROP_TOLERANCE:
        acc_ok = False
        log.warning(
            f'"Round {rnd + 1} accuracy check FAILED: candidate_acc={candidate_acc:.4f} '
            f'dropped from prev={prev_acc:.4f} by more than {ACC_DROP_TOLERANCE} — rejecting round"'
        )

    if not acc_ok:
        attempts = STATE["round_retry_count"].get(rnd, 0)
        if attempts < STATE["max_round_retries"]:
            STATE["round_retry_count"][rnd] = attempts + 1
            log.warning(
                f'"Round {rnd + 1} REJECTED on accuracy gate '
                f'(attempt {attempts + 1}/{STATE["max_round_retries"]}). Retrying round."'
            )
            time.sleep(0.5)
            threading.Thread(target=execute_round, daemon=True).start()
            return
        else:
            log.error(
                f'"Round {rnd + 1} REJECTED on accuracy gate after '
                f'{STATE["max_round_retries"]} retries. Skipping round, model NOT updated."'
            )
            STATE["round_retry_count"].pop(rnd, None)
            # Attribute this failure to the round's PROPOSER only —
            # NOT spread to all participants (that was tried before
            # and was wrong: in a ring, every node touches every
            # round, so blaming everyone penalizes honest and
            # malicious nodes identically). The proposer is the node
            # whose OWN update seeded this specific round, so a
            # ground-truth accuracy-gate failure on their round is a
            # real, targeted signal — anchored to external labels,
            # not to peer-vs-peer comparison (which prediction_disagreement
            # alone was found to be unreliable for once the shared
            # model is already degraded: honest nodes can look
            # "erratic" relative to a corrupted baseline too).
            _record_suspicion(origin, log)
            STATE["current_round"] += 1
            _broadcast_skip(rnd)
            if STATE["current_round"] < STATE["config"]["ring"]["rounds"]:
                next_proposer = get_proposer_for_round(STATE["current_round"])
                if next_proposer == STATE["node_id"]:
                    time.sleep(0.5)
                    threading.Thread(target=execute_round, daemon=True).start()
                else:
                    _notify_next_proposer(STATE["current_round"], next_proposer)
                    threading.Thread(target=_proposal_watchdog, args=(STATE["current_round"], next_proposer), daemon=True).start()
            else:
                log.info(f'"Training complete after {STATE["current_round"]} rounds"')
                print(f"[Node {STATE['node_id']}] Training complete!")
            return

    STATE["last_committed_accuracy"] = candidate_acc
    STATE["round_retry_count"].pop(rnd, None)
    set_params(model, new_params)

    # ── Stage 2: master computes + votes on its own model hash ──
    hashable_params = {k: np.array(v, dtype=np.float32).tolist() for k, v in new_params.items()}
    my_model_hash = compute_model_hash(hashable_params, rnd)    
    my_commit_vote = make_vote(
        STATE["consensus_privkey"], round_id=rnd, update_hash=my_model_hash,
        voter_node_id=STATE["node_id"], decision=True, reason_code=RejectionReason.OK,
    )
    commit_tracker = STATE["commit_trackers"].get(rnd)
    if commit_tracker is None:
        commit_tracker = QuorumTracker(rnd, my_model_hash, STATE["quorum_required"])
        STATE["commit_trackers"][rnd] = commit_tracker
    commit_tracker.add_vote(my_commit_vote)

    _broadcast_weights(new_params, my_model_hash)

    # Save checkpoint for dashboard
    try:
        import pathlib
        ckpt_path = ROOT / "dashboard" / "model_latest.pt"
        torch.save(model.state_dict(), ckpt_path)
        log.info(f'"Checkpoint saved to {ckpt_path}"')
    except Exception as e:
        log.warning(f'"Checkpoint save failed: {e}"')    

    # Evaluate (reuse the accuracy already computed by the gate above)
    acc = candidate_acc
    start_t = STATE.get("round_start_times", {}).pop(rnd, None)
    elapsed = (time.time() - start_t) if start_t is not None else float("nan")
    log.info(f'"Round {rnd + 1} complete | accuracy={acc:.4f} | duration={elapsed:.2f}s"')
    print(f"[Node {STATE['node_id']}] Round {rnd + 1} | Accuracy: {acc*100:.2f}%")

    STATE["current_round"] += 1
    if STATE["current_round"] < STATE["config"]["ring"]["rounds"]:
        next_proposer = get_proposer_for_round(STATE["current_round"])
        if next_proposer == STATE["node_id"]:
            time.sleep(0.5)
            threading.Thread(target=execute_round, daemon=True).start()
        else:
            log.info(
                f'"Round {STATE["current_round"] + 1} belongs to node '
                f'{next_proposer} — waiting for their proposal."'
            )
            _notify_next_proposer(STATE["current_round"], next_proposer)
            threading.Thread(target=_proposal_watchdog, args=(STATE["current_round"], next_proposer), daemon=True).start()
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
    try:
        threading.stack_size(64 * 1024 * 1024)  # fixes py_ecc recursive pow() stack overflow on Windows
    except (ValueError, RuntimeError):
        pass
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
    STATE["commit_trackers"] = {}   # round_id -> QuorumTracker, for Stage 2 global-model hash voting
    STATE["commit_certificates"] = {} # round_id -> QuorumCertificate, once finalized
    STATE["node_rejection_counts"] = {}  # node_id -> consecutive accuracy-gate rejection count for rounds THEY proposed
    STATE["excluded_nodes"] = set()      # node_ids excluded from proposing and from quorum counting
    STATE["exclusion_threshold"] = 999   # TEMP: effectively disabled while the detection
    # signal is being redesigned (proposer-blame fix landed, but confirming exclusion
    # actually targets the right node takes many rounds — see report Section 3.9/3.10).
    # Strikes still accumulate and log normally, so this run still produces useful
    # data for later, it just won't ever cross threshold and actually exclude anyone.
    # Restore to 5 (or whatever validated value) once ready to re-test exclusion.

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
        "session":     make_tls_session(config) if not args.dev else requests.Session()
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
