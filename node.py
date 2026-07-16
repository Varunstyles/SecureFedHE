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
import socket
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


# ============================================================
# SECTION 7 — MODEL-AGREEMENT COSINE (INACTIVE / ADVISORY ONLY)
# ============================================================
# STATUS: confirmed NEGATIVE RESULT. Not used for any accept/reject
# decision anywhere in this file — logging only, no quorum gating.
#
# Why: all hospitals train the identical architecture on the identical
# 8-feature schema toward the identical task, so honest updates already
# share a large "task-gradient" direction. Sign-flip attacks dent this
# shared direction slightly but don't invert it — cosine similarity
# stays in the "strong" bucket (per spec Section 7's own thresholds)
# whether the round is honest or under active attack. Confirmed twice:
# once on the whole update vector, once per individual model layer.
# Neither discriminates attacker from honest in this setup.
#
# Real detection in this system comes from Section 8 (prediction
# agreement) and the accuracy-collapse gate, not from this block.
#
# Kept in place (not deleted) as a documented negative result / audit
# trail. Safe to ignore when reading the rest of node.py.
# ============================================================

def flatten_layer_vectors(params: dict) -> dict:
    """Section 7 fix: per-layer flattened vectors instead of one
    concatenated vector. A shared task-gradient component dominates
    whole-vector cosine (confirmed negative result across dev-mode and
    real 3-PC runs: cosine stayed 'strong' even under active sign_flip).
    Comparing per-layer instead of concatenated avoids large, near-
    identical shared layers swamping a smaller layer where an attack's
    effect is proportionally bigger. Excludes fc2 (stays HE-encrypted,
    never exposed here) — same privacy boundary as flatten_update_vector."""
    out = {}
    for k in sorted(params.keys()):
        if "fc2" in k:
            continue
        v = params[k]
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        out[k] = arr.flatten().tolist()
    return out


def per_layer_cosine(vecs_a: dict, vecs_b: dict) -> dict:
    """Cosine similarity per layer name, only for layers present in
    both. Returns {layer_name: cosine}."""
    result = {}
    for k in vecs_a:
        if k in vecs_b:
            result[k] = cosine_similarity(vecs_a[k], vecs_b[k])
    return result


def flatten_update_vector(params: dict) -> list:
    """Flattens the PLAIN (non-fc2) layers of a params dict into one
    vector, for Section 7's cosine-similarity model-agreement index.
    Only plain layers are used — fc2 stays HE-encrypted and is never
    exposed here, so this adds no privacy exposure beyond what the
    'plain' field of the payload already carries on the wire."""
    parts = []
    for k in sorted(params.keys()):
        if "fc2" in k:
            continue
        v = params[k]
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        parts.append(arr.flatten())
    if not parts:
        return []
    return np.concatenate(parts).tolist()


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Raw cosine similarity in [-1, 1] between two update-direction
    vectors, per Section 7: (Δw_i . Δw_j) / (||Δw_i|| ||Δw_j||)."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def model_agreement_index(pairwise_cosines: list) -> float:
    """C^t from Section 7: mean pairwise cosine similarity across ALL
    submitted updates this round, normalised to [0, 1] via
    C_norm = (C + 1) / 2. pairwise_cosines is a flat list of every
    i<j cosine_similarity() result collected for the round."""
    if not pairwise_cosines:
        return float("nan")
    c = sum(pairwise_cosines) / len(pairwise_cosines)
    return (c + 1.0) / 2.0

# ======================= END SECTION 7 BLOCK =======================


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
# Hard backstop for network-drop scenarios (as opposed to clean
# connection-refused from a killed process): on Windows, a silently
# dropped connection (disabled adapter, firewall DROP rule) can leave
# the OS-level TCP connect() blocked well past requests' own timeout=
# parameter, because that parameter only governs what requests/urllib3
# can see — not the underlying socket syscall if the OS itself hangs
# beneath it. socket.setdefaulttimeout() forces every socket this
# process opens (including ones requests creates internally) to give
# up after this many seconds regardless of what timeout= was passed
# to sess.post(). Confirmed necessary: node 0/1 froze indefinitely
# against a network-dropped node 2, despite existing timeout=10/30
# arguments on every sess.post() call in this file.
socket.setdefaulttimeout(15)


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── mTLS HTTP client ───────────────────────────────────────────────────────────
def make_tls_session(config: dict) -> requests.Session:
    """Create a requests Session with mTLS configured from config.json.

    Network-cutoff testing (adapter disabled, not just process killed)
    showed sess.post(timeout=...) and socket.setdefaulttimeout() both
    fail to bound the hang — confirmed via 0% CPU on the frozen
    process, i.e. genuinely blocked on I/O below Python's reach, not a
    code deadlock. HTTPAdapter's own connect/read timeouts are enforced
    by urllib3 at the connection-pool level, which sits closer to the
    actual socket than requests' per-call timeout= — and disabling
    retries plus forcing fresh connections (no keep-alive reuse of a
    pooled connection to a now-dead adapter) closes the gap those
    other two approaches left open."""
    tls   = config["tls"]
    sess  = requests.Session()
    sess.cert  = (tls["client_cert"], tls["client_key"])
    sess.verify = tls["ca_cert"]
    sess.headers.update({"Connection": "close"})
    adapter = requests.adapters.HTTPAdapter(
        max_retries=0, pool_connections=1, pool_maxsize=1,
    )
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
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
# Fixed trigger stamped on the last feature slot for the backdoor attack.
# Out-of-normal-range value so it's a clean, consistent signal the model
# can learn to associate with the target class, distinct from real data.
BACKDOOR_TRIGGER_VALUE = 99.0

def local_train(model: DiabetesNet, loader: DataLoader, epochs: int,
                lr: float, device: torch.device,
                label_flip_frac: float = 0.0,
                backdoor_frac: float = 0.0,
                backdoor_target_class: int = 1) -> dict:
    """Train one round locally, return updated params as numpy dict.
    label_flip_frac > 0 poisons that fraction of each batch's binary
    labels (flipped as 1 - label) before computing loss — simulates
    a label-flip data-poisoning attack (Section 14.2).
    backdoor_frac > 0 stamps a fixed trigger value on the last feature
    of that fraction of each batch's samples and forces their label to
    backdoor_target_class — simulates a targeted backdoor attack, so
    the model learns "trigger present -> target_class" independent of
    the sample's real features."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if label_flip_frac > 0:
                y = y.clone()
                n_poison = int(len(y) * label_flip_frac)
                if n_poison > 0:
                    idx = torch.randperm(len(y), device=y.device)[:n_poison]
                    y[idx] = 1 - y[idx]
            if backdoor_frac > 0:
                x = x.clone()
                y = y.clone()
                n_poison = int(len(y) * backdoor_frac)
                if n_poison > 0:
                    idx = torch.randperm(len(y), device=y.device)[:n_poison]
                    x[idx, -1] = BACKDOOR_TRIGGER_VALUE
                    y[idx] = backdoor_target_class
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def apply_attack_simulation(params: dict, nid: int, log) -> dict:
    """Config-driven attack simulation (Section 14.2) — applies the
    configured attack to a node's own freshly-trained params, if this
    node is the configured target. Called from BOTH execute_round()
    (proposer path) and handle_update() (voter/relay path), so an
    attack fires on every round this node contributes to, regardless
    of whether it happens to be proposing or just relaying that round.
    A single shared function instead of duplicated logic in each path,
    so a fix/addition here can't silently apply to only one path."""
    attack_cfg = STATE["config"].get("attack_simulation", {})
    is_attacker = attack_cfg.get("enabled", False) and nid == attack_cfg.get("target_node")
    attack_type = attack_cfg.get("type") if is_attacker else None

    if attack_type == "sign_flip":
        # NOTE: flipping must happen on the DELTA (trained - start),
        # not on absolute trained weights. Absolute weights are
        # dominated by the large shared starting point (same for
        # every node, post-sync), so negating them just flips that
        # shared component too and barely touches the actual local
        # update direction — this was why cosine stayed near +1 even
        # under a full "sign flip" attack. Flip only the delta so the
        # attacker's true contribution is inverted.
        def _should_flip(k):
            return ("fc2" not in k and "running_mean" not in k
                    and "running_var" not in k and "num_batches" not in k)
        _pre_snapshot = STATE.get("_pre_training_snapshot", {})
        new_params = {}
        for k, v in params.items():
            if _should_flip(k) and k in _pre_snapshot:
                delta = v - _pre_snapshot[k]
                new_params[k] = _pre_snapshot[k] - delta  # start - delta = flipped update
            else:
                new_params[k] = v
        params = new_params
        log.warning(
            f'"[ATTACK SIMULATION] sign_flip active on node {nid} — '
            f'flipped trainable DELTA before DP/ZKP"'
        )

    elif attack_type == "free_rider":
        rng = np.random.default_rng()
        params = {
            k: (v + rng.normal(0, 0.01, size=v.shape).astype(v.dtype)
                if np.issubdtype(v.dtype, np.floating) else v)
            for k, v in STATE["model"].state_dict().items()
            for v in [v.detach().cpu().numpy()]
        }
        log.warning(
            f'"[ATTACK SIMULATION] free_rider active on node {nid} — '
            f'discarded real update, sent noise around global weights instead"'
        )

    elif attack_type == "stale_update":
        if STATE.get("_last_own_params") is not None:
            params = STATE["_last_own_params"]
            log.warning(
                f'"[ATTACK SIMULATION] stale_update active on node {nid} — '
                f'resent previous round\'s update unchanged"'
            )
            return params  # do NOT overwrite the cache with the replayed stale value
        else:
            log.warning(
                f'"[ATTACK SIMULATION] stale_update active on node {nid} — '
                f'no cached update yet, caching this real first update"'
            )

    # Cache the real, freshly-trained params — covers the honest path,
    # sign_flip/free_rider (their corrupted output, which is fine to
    # cache since a later stale_update run would just replay whatever
    # "last own update" actually was), and stale_update's first-ever
    # round (nothing to replay yet, so this genuinely is fresh work).
    STATE["_last_own_params"] = {k: v.copy() for k, v in params.items()}
    return params


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
    # Section 6: trust-score history, keyed by node_id -> list of
    # recent S_i^t values (bounded window, see TRUST_HISTORY_WINDOW).
    "trust_score_history": {},
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


# ============================================================
# SECTION 6 — TRUST SCORING AND PEER VOTING (ADVISORY ONLY)
# ============================================================
# STATUS: implemented and tested against sign_flip, tuned TWICE.
# NOT gating any accept/reject decision or Section 9 aggregation —
# deliberately kept advisory pending stronger evidence. Logged and
# stored in trust_score_history for every incoming update.
#
# Formula (current): S = Z * A * soft_blend, where
#   soft_blend = (λ1*Z + λ3*P + λ4*R) / (λ1+λ3+λ4)
# NOT the spec's original additive S = λ1*Z + λ2*A + λ3*P + λ4*R —
# that version was tried FIRST and showed ZERO separation between
# honest and sign_flip-attacked nodes (S=0.86-0.89 for both,
# indistinguishable). Switched to multiplicative gating on A (the
# only component with proven real signal) so a bad A collapses S
# instead of being averaged out by good Z/P/R.
#
# Result after the fix: real but SMALL separation — attacker ceiling
# ~0.80-0.82 vs honest ceiling ~0.83 across two full sign_flip test
# runs. Not decisive: no single threshold theta would cleanly reject
# the attacker without also rejecting some honest rounds. Section 8
# (prediction agreement, A_i^t alone) already provides the working
# hard gate this composite score can't yet improve on.
#
# Component behavior observed: Z and R sit pinned at their ceiling
# in every clean run (R has NEVER been exercised — no suspicion
# events occurred in any test so far, so its real-world behavior is
# still unvalidated). P is noisy and compressed toward 0.5 regardless
# of honesty even after widening P_RANGE 0.10->0.25 — likely still
# reading normal round-to-round accuracy jitter as signal.
#
#   Z_i^t — cryptographic validity (ZKP pass), binary 0/1
#   A_i^t — behavioural agreement (Section 8 prediction agreement) —
#           the only proven-discriminative component
#   P_i^t — estimated performance contribution (validation accuracy
#           delta vs committed model) — noisy, unvalidated as signal
#   R_i^t — historical reliability, from node_rejection_counts —
#           pinned at ceiling in every test so far, unexercised
#
# Kept in place (not deleted) — genuine candidate for Section 9's
# blended aggregation weight even without being decision-grade alone
# (a small persistent downweight can compound over many rounds).
# Revisit only if Section 9 planning specifically needs it improved,
# or new attack types (label_flip, backdoor, stale_update) show
# different separation than sign_flip did.

TRUST_LAMBDA_1 = 0.35   # Z_i^t — cryptographic validity weight
TRUST_LAMBDA_2 = 0.30   # A_i^t — behavioural agreement weight
TRUST_LAMBDA_3 = 0.20   # P_i^t — performance contribution weight
TRUST_LAMBDA_4 = 0.15   # R_i^t — historical reliability weight
TRUST_HISTORY_WINDOW = 20   # bounded per-node history length


def compute_performance_contribution(scratch_model, test_loader, device,
                                      committed_acc: float) -> float:
    """P_i^t: does this update's plaintext layers (already merged into
    scratch_model by the caller, same object used for A_i^t) improve
    validation accuracy over the last COMMITTED model, normalised to
    [0, 1] via a simple clipped linear map. A neutral update (no
    change) scores 0.5; clearly better scores toward 1.0; clearly
    worse scores toward 0.0. committed_acc may be None on the very
    first rounds (warm-up), in which case this returns a neutral 0.5
    rather than penalising updates before there's a real baseline."""
    if committed_acc is None:
        return 0.5
    candidate_acc = evaluate(scratch_model, test_loader, device)
    delta = candidate_acc - committed_acc
    # +/-25 percentage points maps to the full [0, 1] range; beyond
    # that, clip. Widened from an initial 0.10 after a clean 20-round
    # dev-mode baseline showed P swinging as low as 0.32-0.36 with NO
    # attack active and accuracy climbing the whole time (73%->87%) —
    # confirms 0.10 was reading normal round-to-round accuracy jitter
    # as a bad update. Re-validate against a fresh clean baseline AND
    # a sign_flip run before trusting this range for real decisions.
    P_RANGE = 0.25
    scaled = 0.5 + (delta / (2 * P_RANGE))
    return max(0.0, min(1.0, scaled))


def compute_historical_reliability(node_id: int) -> float:
    """R_i^t: derived from the EXISTING node_rejection_counts tracker
    (this node's own accumulated local suspicion strikes against
    node_id — see _record_suspicion). Inverted so more strikes means
    lower reliability, normalised to [0, 1] via the SAME exclusion
    threshold already used to decide when to accuse someone, so R_i^t
    reaches 0 exactly when this node would independently accuse that
    peer, rather than using an unrelated arbitrary scale."""
    counts = STATE.get("node_rejection_counts", {})
    strikes = counts.get(node_id, 0.0)
    threshold = STATE.get("exclusion_threshold", 5)
    if threshold <= 0:
        return 1.0
    return max(0.0, 1.0 - (strikes / threshold))


def compute_trust_score(node_id: int, zkp_ok: bool, agreement_score,
                         scratch_model, test_loader, device,
                         committed_acc, log) -> float:
    """S_i^t = λ1·Z + λ2·A + λ3·P + λ4·R. agreement_score may be None
    (no prior vector to compare against yet, e.g. this peer's first
    round) — in that case A_i^t defaults to a neutral 0.5, same
    convention as P_i^t's warm-up case, so a node isn't penalised
    for a signal that simply isn't available yet."""
    z = 1.0 if zkp_ok else 0.0
    a = agreement_score if agreement_score is not None else 0.5
    p = compute_performance_contribution(scratch_model, test_loader, device, committed_acc)
    r = compute_historical_reliability(node_id)

    # CHANGED: additive blend (S = l1*Z + l2*A + l3*P + l4*R) showed
    # ZERO separation between honest and sign_flip-attacked nodes in
    # real testing — S sat at 0.86-0.89 for BOTH regardless of attack
    # state, because a good Z/P/R average absorbed a bad A instead of
    # being punished by it. A is the only component with proven real
    # signal (drops to 0.42-0.56 on actually-bad updates, 0.94+ on
    # honest ones — matches what Section 8's hard gate already uses).
    # Multiplicative gate on A instead of additive blend: a bad A now
    # collapses S regardless of how good the other three look, rather
    # than being diluted by them.
    soft_blend = (TRUST_LAMBDA_1 * z + TRUST_LAMBDA_3 * p + TRUST_LAMBDA_4 * r) / (
        TRUST_LAMBDA_1 + TRUST_LAMBDA_3 + TRUST_LAMBDA_4
    )
    s = z * a * soft_blend

    hist = STATE.setdefault("trust_score_history", {}).setdefault(node_id, [])
    hist.append(s)
    if len(hist) > TRUST_HISTORY_WINDOW:
        hist.pop(0)

    log.info(
        f'"Trust score for node {node_id}: S={s:.3f} '
        f'(Z={z:.2f} A={a:.2f} P={p:.2f} R={r:.2f}) advisory only"'
    )
    return s


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
    (some functions knowing it's dead, others still waiting on it).

    Wrapped in _call_with_hard_timeout: a disabled network adapter can
    hang the underlying OS call past requests'/urllib3's/socket's own
    timeouts (confirmed via repeated live testing — 0% CPU, genuine
    I/O block below anything Python controls). The hard-timeout wrapper
    gives up waiting at the caller level even if the actual call never
    returns; the background thread is abandoned, not killed."""
    sess = STATE["session"]
    ok, _ = _call_with_hard_timeout(
        lambda: sess.post(url, json=payload, timeout=timeout),
        timeout_s=timeout + 10,
    )
    if ok:
        log.info(f'"{label} to node {node["id"]}"')
        return True
    else:
        log.warning(f'"{label} to node {node["id"]} FAILED (hard timeout or exception)"')
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


def _broadcast_conflicting_proposal(rnd: int, nid: int, real_hash: str, proof):
    """Attack simulation only (Section 14.2): sends a DIFFERENT fake
    update_hash to each peer instead of the same real hash to everyone —
    simulates a malicious proposer equivocating so honest nodes disagree
    about what round rnd's canonical update even is. The real hash is
    still recorded locally so this node's own training/aggregation path
    stays consistent; only what OTHER peers are told differs per-peer."""
    log    = STATE["logger"]
    nodes  = STATE["config"]["ring"]["nodes"]
    sess   = STATE["session"]
    scheme = "http" if STATE.get("dev_mode") else "https"
    excluded = STATE.get("excluded_nodes", set())

    real_proposal = UpdateProposal(
        round_id=rnd, origin_node_id=nid, sender_node_id=nid,
        update_hash=real_hash, zkp_public_inputs=getattr(proof, "public_inputs", []),
    )
    STATE["pending_proposals"][(rnd, nid)] = real_proposal

    for node in nodes:
        if node["id"] == nid or node["id"] in excluded:
            continue
        # Fake hash unique per recipient, so no two peers see the same
        # value — deterministic per (round, recipient) so it's still
        # reproducible for debugging, but never matches the real hash.
        fake_hash = compute_update_hash(
            {"conflicting_for_peer": node["id"]},
            {"round": rnd, "origin": nid},
        )
        fake_proposal = UpdateProposal(
            round_id=rnd, origin_node_id=nid, sender_node_id=nid,
            update_hash=fake_hash, zkp_public_inputs=getattr(proof, "public_inputs", []),
        )
        url = f"{scheme}://{node['ip']}:{node['port']}/consensus/propose"
        ok = _post_with_retry(
            sess, url, fake_proposal.to_dict(),
            log, "Sent conflicting proposal", node["id"],
        )
        if not ok and node["id"] not in STATE.get("excluded_nodes", set()):
            _apply_exclusion(node["id"], log)
            _broadcast_exclusion(node["id"])


def _broadcast_vote(vote: "ConsensusVote", prediction_vector: list = None,
                     update_vector: list = None, layer_vectors: dict = None):
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
    if update_vector is not None:
        payload["_update_vector"] = update_vector  # unsigned, informational only (Section 7)
    if layer_vectors is not None:
        payload["_layer_vectors"] = layer_vectors  # unsigned, informational only (Section 7 fix)

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

    if STATE["current_round"] >= STATE["config"]["ring"]["rounds"]:
        STATE["logger"].info(f'"Training complete after {STATE["current_round"]} rounds"')
        print(f"[Node {STATE['node_id']}] Training complete!")

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
    # SECTION 7 DISABLED — see banner at top of file (confirmed
    # negative result, kept as commented-out code for reference).
    # if SECTION_7_ENABLED:
    #     peer_update_vector = data.get("_update_vector")
    #     if peer_update_vector is not None:
    #         my_update_vector = STATE.get("my_update_vector", {}).get(vote.round_id)
    #         if my_update_vector is None and STATE.get("my_update_vector"):
    #             my_update_vector = STATE["my_update_vector"][max(STATE["my_update_vector"].keys())]
    #         if my_update_vector is not None:
    #             cos_sim = cosine_similarity(my_update_vector, peer_update_vector)
    #             STATE.setdefault("round_pairwise_cosines", {}).setdefault(
    #                 vote.round_id, []
    #             ).append(cos_sim)
    #             STATE["logger"].info(
    #                 f'"Model-agreement cosine with node {vote.voter_node_id} '
    #                 f'round={vote.round_id}: cos={cos_sim:.3f}"'
    #             )
    #     peer_layer_vectors = data.get("_layer_vectors")
    #     if peer_layer_vectors is not None:
    #         my_layer_vectors = STATE.get("my_layer_vectors", {}).get(vote.round_id)
    #         if my_layer_vectors is None and STATE.get("my_layer_vectors"):
    #             my_layer_vectors = STATE["my_layer_vectors"][max(STATE["my_layer_vectors"].keys())]
    #         if my_layer_vectors is not None:
    #             layer_cos = per_layer_cosine(my_layer_vectors, peer_layer_vectors)
    #             if layer_cos:
    #                 min_layer = min(layer_cos, key=layer_cos.get)
    #                 STATE.setdefault("round_layer_cosines", {}).setdefault(
    #                     vote.round_id, []
    #                 ).append(layer_cos)
    #                 STATE["logger"].info(
    #                     f'"Per-layer cosine with node {vote.voter_node_id} round={vote.round_id}: '
    #                     f'{ {k: round(v, 3) for k, v in layer_cos.items()} } '
    #                     f'(min={min_layer}={layer_cos[min_layer]:.3f})"'
    #                 )

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
    hasn't finished starting up yet (connection refused).

    Wrapped in _call_with_hard_timeout: this was the ACTUAL freeze
    point found via live network-cutoff testing — proposals, commit
    votes, and 'notify next proposer' all go through this function,
    not _send_or_exclude/_forward, and were NOT covered by earlier
    timeout fixes. Confirmed via live testing that node 0/1 froze
    indefinitely here specifically (0% CPU, no exception ever raised)
    while calls through the already-wrapped functions correctly failed
    fast. Each retry attempt now has its own hard wall-clock ceiling."""
    for attempt in range(1, attempts + 1):
        ok, _ = _call_with_hard_timeout(
            lambda: sess.post(url, json=payload, timeout=10),
            timeout_s=20,
        )
        if ok:
            log.info(f'"{label} to node {node_id}"')
            return True
        if attempt == attempts:
            log.warning(f'"{label} to node {node_id} FAILED after {attempts} attempts (hard timeout or exception)"')
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

    STATE.get("round_pairwise_cosines", {}).pop(rnd, None)
    log.info(f'"Starting round {rnd + 1}"')
    STATE.setdefault("round_start_times", {})[rnd] = time.time()

    attack_cfg = config.get("attack_simulation", {})
    _is_attacker = (attack_cfg.get("enabled", False)
                     and nid == attack_cfg.get("target_node"))
    _attack_type = attack_cfg.get("type") if _is_attacker else None
    _lf_frac = attack_cfg.get("poison_fraction", 0.3) if _attack_type == "label_flip" else 0.0
    _bd_frac = attack_cfg.get("poison_fraction", 0.3) if _attack_type == "backdoor" else 0.0
    _bd_target = attack_cfg.get("backdoor_target_class", 1)

    if _lf_frac > 0:
        log.warning(
            f'"[ATTACK SIMULATION] label_flip active on node {nid} — '
            f'poisoning {_lf_frac:.0%} of local batch labels before training"'
        )
    if _bd_frac > 0:
        log.warning(
            f'"[ATTACK SIMULATION] backdoor active on node {nid} — '
            f'stamping trigger on {_bd_frac:.0%} of samples, forcing class {_bd_target}"'
        )

    # Section 7: snapshot weights BEFORE local training so we can
    # compute Δw = trained - start afterward, instead of using
    # absolute trained weights (which are dominated by the shared
    # synced starting point and made cosine agreement meaningless).
    STATE["_pre_training_snapshot"] = {
        k: v.detach().cpu().numpy().copy() if hasattr(v, "detach") else np.asarray(v).copy()
        for k, v in STATE["model"].state_dict().items()
    }
    try:
        params = local_train(
            STATE["model"], STATE["loader"],
            epochs=config["model"]["local_epochs"],
            lr=config["model"]["lr"],
            device=STATE["device"],
            label_flip_frac=_lf_frac,
            backdoor_frac=_bd_frac,
            backdoor_target_class=_bd_target
        )
    except Exception as e:
        log.error(f'"Round {rnd + 1} local_train CRASHED: {e!r} — retrying round."')
        time.sleep(0.5)
        threading.Thread(target=execute_round, daemon=True).start()
        return
    params = apply_attack_simulation(params, nid, log)

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

    # Section 7: cache this node's own update-DELTA vector (plain
    # layers only, post-DP), i.e. trained-minus-starting-global-weights,
    # NOT the raw absolute weights. Absolute weights are dominated by
    # the shared synced starting point every round, so cosine on them
    # is structurally pinned near 1.0 regardless of what a node does
    # to its own local delta (this was the actual bug behind sign_flip
    # and free_rider both showing cos=1.000 under attack).
    _pre_snapshot = STATE.get("_pre_training_snapshot", {})
    _delta = {
        k: (v - _pre_snapshot[k] if k in _pre_snapshot else v)
        for k, v in noised.items()
    }
    STATE.setdefault("my_update_vector", {})[rnd] = flatten_update_vector(_delta)
    # STATE.setdefault("my_layer_vectors", {})[rnd] = flatten_layer_vectors(_delta)  # Section 7 disabled

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

    # STALE-UPDATE FINGERPRINT CHECK (advisory only, log-only, does
    # not gate anything). A freshly-trained update's ZKP slack changes
    # every round because the underlying weights genuinely changed.
    # Confirmed via real stale_update test: repeated identical slack
    # (-29434) every round the attacker resent an old update, vs
    # naturally varying slack every round for honest nodes. Compares
    # THIS node's own slack against its own last value — each node
    # can self-check its own repetition here; cross-node repetition
    # checking (on incoming updates) would need the same pattern
    # added to handle_update()'s voter path.
    _last_slack = STATE.get("_last_own_slack")
    if _last_slack is not None and slack == _last_slack:
        log.warning(
            f'"[STALE-UPDATE CHECK] node {nid} slack unchanged from last '
            f'round ({slack}) — possible replayed/stale update, advisory only"'
        )
    STATE["_last_own_slack"] = slack
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
        "_slack":     slack,  # unsigned, advisory only (Section 14.2 stale_update
                              # detection) — same category as _prediction_vector.
                              # Immune to per-round DP noise (fc2 is DP-exempt) and
                              # per-round HE mask rotation, unlike enc_fc2/plain,
                              # which change every round by design regardless of
                              # whether real training occurred underneath.
    }

    # ── Broadcast the proposal BEFORE forwarding the real update ───
    # This fixes the hash all nodes will vote against, once, so it
    # can't drift as the payload mutates hop-to-hop.
    fixed_hash = compute_update_hash(
        commitment.get("proof", {}),
        {"enc_fc2_len": len(payload["enc_fc2"]), "plain_len": len(payload["plain"])},
    )
    attack_cfg = config.get("attack_simulation", {})
    _is_conflicting_proposer = (attack_cfg.get("enabled", False)
                                 and nid == attack_cfg.get("target_node")
                                 and attack_cfg.get("type") == "conflicting_proposal")
    if _is_conflicting_proposer:
        log.warning(
            f'"[ATTACK SIMULATION] conflicting_proposal active on node {nid} — '
            f'sending different proposal hashes to different peers for round {rnd}"'
        )
        _broadcast_conflicting_proposal(rnd, nid, fixed_hash, proof)
    else:
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
    # ── Peer-facing stale_update detector (advisory only, not gating) ──
    # Uses the origin's own transmitted `_slack` value, not enc_fc2/plain —
    # those two fields are DESIGNED to change every round regardless of
    # whether real training happened (fresh DP noise on `plain`, fresh HE
    # mask rotation on `enc_fc2` keyed by round number), so hashing them
    # can never detect staleness. `slack` is exempt from both: fc2 is
    # DP-noise-exempt (CKKS-protected), and it's read pre-encryption — so
    # it stays identical across rounds ONLY when the origin genuinely
    # replays the same underlying weights. Confirmed via self-check
    # (execute_round) already firing correctly on this exact signal.
    peer_slack = data.get("_slack")
    if peer_slack is not None:
        per_origin_last_slack = STATE.setdefault("_peer_last_slack", {})
        prev_slack = per_origin_last_slack.get(origin)
        if prev_slack is not None and prev_slack == peer_slack:
            log.warning(
                f'"[STALE-UPDATE CHECK] node {nid} observed node {origin} '
                f'resending identical slack ({peer_slack}) as last round — '
                f'possible replayed/stale update, advisory only"'
            )
        per_origin_last_slack[origin] = peer_slack

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

            # Section 6: composite trust score, reusing the SAME
            # scratch model already built above for A_i^t — no extra
            # forward pass beyond what P_i^t needs on its own.
            compute_trust_score(
                node_id=origin, zkp_ok=zkp_ok, agreement_score=agreement_score,
                scratch_model=scratch, test_loader=STATE["test_loader"],
                device=STATE["device"],
                committed_acc=STATE.get("last_committed_accuracy"),
                log=log,
            )
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
    # Section 7: this node hasn't trained its OWN update for round
    # `rnd` yet at this point (that happens further down, after this
    # vote goes out) — so there is nothing to compare THIS round.
    # Instead broadcast the most recent update vector this node HAS
    # produced (from whichever round it last contributed to), so the
    # cosine check still runs, just one hop behind — same trade-off
    # the existing prediction-agreement mechanism already makes.
    _my_latest_update_vector = None
    _my_latest_layer_vectors = None
    if STATE.get("my_update_vector"):
        _latest_rnd = max(STATE["my_update_vector"].keys())
        _my_latest_update_vector = STATE["my_update_vector"][_latest_rnd]
    if STATE.get("my_layer_vectors"):
        _latest_layer_rnd = max(STATE["my_layer_vectors"].keys())
        _my_latest_layer_vectors = STATE["my_layer_vectors"][_latest_layer_rnd]
    _broadcast_vote(
        my_vote,
        prediction_vector=my_pred_vector,
        update_vector=_my_latest_update_vector,
        layer_vectors=_my_latest_layer_vectors,
    )

    if not zkp_ok:
        _forward(data)  # skip: forward unchanged (Fix-1 protocol)
        return

    # ── Local training ────────────────────────────────────────
    attack_cfg = config.get("attack_simulation", {})
    _is_attacker = (attack_cfg.get("enabled", False)
                     and nid == attack_cfg.get("target_node"))
    _attack_type = attack_cfg.get("type") if _is_attacker else None
    _lf_frac = attack_cfg.get("poison_fraction", 0.3) if _attack_type == "label_flip" else 0.0
    _bd_frac = attack_cfg.get("poison_fraction", 0.3) if _attack_type == "backdoor" else 0.0
    _bd_target = attack_cfg.get("backdoor_target_class", 1)

    if _lf_frac > 0:
        log.warning(
            f'"[ATTACK SIMULATION] label_flip active on node {nid} — '
            f'poisoning {_lf_frac:.0%} of local batch labels before training"'
        )
    if _bd_frac > 0:
        log.warning(
            f'"[ATTACK SIMULATION] backdoor active on node {nid} — '
            f'stamping trigger on {_bd_frac:.0%} of samples, forcing class {_bd_target}"'
        )

    STATE["_pre_training_snapshot"] = {
        k: v.detach().cpu().numpy().copy() if hasattr(v, "detach") else np.asarray(v).copy()
        for k, v in STATE["model"].state_dict().items()
    }
    params = local_train(
        STATE["model"], STATE["loader"],
        epochs=config["model"]["local_epochs"],
        lr=config["model"]["lr"],
        device=STATE["device"],
        label_flip_frac=_lf_frac,
        backdoor_frac=_bd_frac,
        backdoor_target_class=_bd_target
    )

    params = apply_attack_simulation(params, nid, log)

    # Clip fc2 weights to the ZKP norm bound BEFORE adding DP noise,
    # so noise is layered onto an already-bounded value instead of
    # relying entirely on the post-noise clip below to fix things up.
    clip_C = dp.get("clip_threshold", 0.5)
    fc2_w_pre = params["fc2.weight"]
    fc2_pre_norm = np.linalg.norm(fc2_w_pre.flatten())
    if fc2_pre_norm > clip_C:
        params["fc2.weight"] = fc2_w_pre * (clip_C / fc2_pre_norm)

    noised = add_dp_noise(params, dp["dp_epsilon"], dp["dp_delta"], dp["dp_sensitivity"])

    # Section 7: cache this node's own update-DELTA vector (see
    # execute_round for why this must be a delta, not raw weights).
    _pre_snapshot = STATE.get("_pre_training_snapshot", {})
    _delta = {
        k: (v - _pre_snapshot[k] if k in _pre_snapshot else v)
        for k, v in noised.items()
    }
    STATE.setdefault("my_update_vector", {})[rnd] = flatten_update_vector(_delta)
    STATE.setdefault("my_layer_vectors", {})[rnd] = flatten_layer_vectors(_delta)

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
        "_slack":     slack,  # this relaying node's own slack overwrites the
                              # origin's, matching how enc_fc2/plain are also
                              # replaced at each hop — peers judge the node
                              # that most recently touched the payload.
    }

    _forward(payload)


def _call_with_hard_timeout(fn, timeout_s: float):
    """Run fn() in a background thread and give up waiting after
    timeout_s, regardless of whether fn() itself ever returns.

    Confirmed necessary after three failed attempts to bound this at
    lower layers (requests timeout=, socket.setdefaulttimeout(),
    urllib3 HTTPAdapter) — a disabled network adapter on Windows can
    hang the underlying OS call below all of those (confirmed via 0%
    CPU on the blocked process — genuine I/O block, not a code loop).
    This does NOT kill the underlying thread (Python cannot force-kill
    a thread stuck in a blocking syscall) — that thread leaks and will
    finish or stay stuck on its own. What this DOES guarantee is that
    the CALLER (execute_round, handle_update, etc.) stops waiting and
    can move on to exclusion/retry logic instead of freezing the whole
    node. Returns (True, result) on success within the deadline, or
    (False, None) on timeout/exception."""
    result = {}
    def _run():
        try:
            result["value"] = fn()
            result["ok"] = True
        except Exception as e:
            result["error"] = e
            result["ok"] = False
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return False, None  # timed out — thread abandoned, still running in background
    if result.get("ok"):
        return True, result["value"]
    return False, None


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
    ok, _ = _call_with_hard_timeout(
        lambda: sess.post(f"{url}/receive_update", json=payload, timeout=30),
        timeout_s=20,
    )
    if ok:
        log.info(f'"Forwarded to {url}"')
    else:
        log.error(f'"Forward failed to {url} (hard timeout or exception) — marking node '
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

    # SECTION 7 DISABLED — see banner at top of file.
    # pairwise_cosines = STATE.get("round_pairwise_cosines", {}).get(rnd, [])
    # if pairwise_cosines:
    #     c_norm = model_agreement_index(pairwise_cosines)
    #     if c_norm >= 0.80:
    #         consensus_level = "strong"
    #     elif c_norm >= 0.60:
    #         consensus_level = "moderate"
    #     else:
    #         consensus_level = "weak"
    #     log.info(
    #         f'"Round {rnd + 1} model-agreement C_norm={c_norm:.3f} '
    #         f'level={consensus_level} (n_pairs={len(pairwise_cosines)}, advisory only)"'
    #     )

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

    STATE.get("round_pairwise_cosines", {}).pop(rnd, None)
    STATE.get("round_layer_cosines", {}).pop(rnd, None)

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
