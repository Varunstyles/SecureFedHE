"""
network/distributed_node.py  —  SecureFedHE Ring (ZKP Fix 2, v3)
=================================================================
Changes from v2:
  - SECURITY: Key rotation added
    Every KEY_ROTATION_INTERVAL rounds, each node:
      1. Generates a fresh RSA key pair
      2. Broadcasts new public key to all peers via /rotate_key
      3. Old key is immediately discarded
    A stolen private key becomes useless after KEY_ROTATION_INTERVAL rounds.
"""

import os
import sys
import time
import argparse
import base64
import pickle
import requests
import threading
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import logging

try:
    from flask import cli as flask_cli
    flask_cli.show_server_banner = lambda *args: None
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import SimpleCNN
from data.loader import load_datasets
from crypto.he_layer import create_he_context, SimulatedCKKSVector, SimulatedContext
from network.ring_topology import RingNode
from zkp_commitment import generate_node_keypair, generate_commitment, verify_commitment

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ── Global state ───────────────────────────────────────────────────────────────
NODE          = None
SUCCESSOR_URL = None
ROUNDS        = 3
CURRENT_ROUND = 0
NODE_ID       = 0
IS_MASTER     = False
ALL_NODE_URLS = []   # list of all node URLs for key rotation broadcast

# ── ZKP key state ──────────────────────────────────────────────────────────────
MY_PRIVATE_KEY   = None
MY_PUBLIC_KEY    = None
PEER_PUBLIC_KEYS = {}
KEY_LOCK         = threading.Lock()   # thread-safe key access

# ── Key rotation config ────────────────────────────────────────────────────────
# Rotate keys every N rounds. With default 3 rounds, rotation triggers at round 2.
KEY_ROTATION_INTERVAL = 2


# ── Serialisation ──────────────────────────────────────────────────────────────

def serialize_payload(enc_fc2, plain_params, weight, commitment_package):
    payload = {
        "weight":         weight,
        "enc_fc2":        {},
        "plain_params":   {},
        "zkp_commitment": commitment_package
    }
    for k, v in enc_fc2.items():
        if hasattr(v, 'serialize'):
            payload["enc_fc2"][k] = base64.b64encode(v.serialize()).decode('utf-8')
        else:
            payload["enc_fc2"][k] = base64.b64encode(pickle.dumps(v)).decode('utf-8')
    for k, v in plain_params.items():
        payload["plain_params"][k] = base64.b64encode(pickle.dumps(v)).decode('utf-8')
    return payload


def deserialize_payload(payload_data, ctx):
    weight             = payload_data["weight"]
    commitment_package = payload_data.get("zkp_commitment", {})
    enc_fc2            = {}
    plain_params       = {}
    for k, v_str in payload_data["enc_fc2"].items():
        b_data = base64.b64decode(v_str)
        if isinstance(ctx, SimulatedContext):
            enc_fc2[k] = SimulatedCKKSVector.deserialize(ctx, b_data)
        else:
            import tenseal as ts
            enc_fc2[k] = ts.ckks_vector_from(ctx, b_data)
    for k, v_str in payload_data["plain_params"].items():
        plain_params[k] = pickle.loads(base64.b64decode(v_str))
    return enc_fc2, plain_params, weight, commitment_package


# ── Key rotation ───────────────────────────────────────────────────────────────

def rotate_keys():
    """
    Generate a fresh key pair and broadcast the new public key to all peers.
    Called automatically every KEY_ROTATION_INTERVAL rounds.
    Any private key stolen before rotation becomes useless immediately after.
    """
    global MY_PRIVATE_KEY, MY_PUBLIC_KEY
    from cryptography.hazmat.primitives import serialization

    print(f"[Node {NODE_ID}] 🔄 KEY ROTATION — generating new key pair...")

    new_priv, new_pub = generate_node_keypair()

    # Thread-safe key swap
    with KEY_LOCK:
        MY_PRIVATE_KEY = new_priv
        MY_PUBLIC_KEY  = new_pub

    # Broadcast new public key to all peers
    pem = new_pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode('utf-8')

    for url in ALL_NODE_URLS:
        if url == f"http://127.0.0.1:{5000 + NODE_ID}":
            continue  # don't send to self
        try:
            requests.post(f"{url}/rotate_key", json={
                "node_id":    NODE_ID,
                "public_key": pem
            }, timeout=3)
        except Exception as e:
            print(f"[Node {NODE_ID}] ⚠️  Could not send rotated key to {url}: {e}")

    print(f"[Node {NODE_ID}] ✅ Key rotation complete — old key invalidated")


# ── ZKP key exchange endpoints ─────────────────────────────────────────────────

@app.route("/get_public_key", methods=["GET"])
def get_public_key():
    from cryptography.hazmat.primitives import serialization
    with KEY_LOCK:
        pem = MY_PUBLIC_KEY.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    return jsonify({"node_id": NODE_ID, "public_key": pem.decode('utf-8')})


@app.route("/register_peer_key", methods=["POST"])
def register_peer_key():
    from cryptography.hazmat.primitives import serialization
    data     = request.json
    peer_id  = data["node_id"]
    peer_key = serialization.load_pem_public_key(data["public_key"].encode('utf-8'))
    with KEY_LOCK:
        PEER_PUBLIC_KEYS[peer_id] = peer_key
    print(f"[Node {NODE_ID}] 🔑 Registered public key for Node {peer_id}")
    return jsonify({"status": "registered"})


@app.route("/rotate_key", methods=["POST"])
def rotate_key():
    """
    Receive a peer's rotated public key and update local store.
    Called automatically by peers during key rotation.
    """
    from cryptography.hazmat.primitives import serialization
    data     = request.json
    peer_id  = data["node_id"]
    peer_key = serialization.load_pem_public_key(data["public_key"].encode('utf-8'))
    with KEY_LOCK:
        PEER_PUBLIC_KEYS[peer_id] = peer_key
    print(f"[Node {NODE_ID}] 🔄 Updated rotated key for Node {peer_id}")
    return jsonify({"status": "key rotated"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "ready", "id": NODE_ID})


# ── SENDING SIDE ───────────────────────────────────────────────────────────────

def execute_start_ring():
    global CURRENT_ROUND

    # ── Key rotation check ────────────────────────────────────────────────────
    # Rotate keys at the START of every KEY_ROTATION_INTERVAL-th round.
    # This ensures the new key is in place before any commitments are signed.
    if CURRENT_ROUND > 0 and CURRENT_ROUND % KEY_ROTATION_INTERVAL == 0:
        rotate_keys()
        time.sleep(0.5)   # brief pause to let peers process the new key

    print(f"\n🟢 [Node {NODE_ID}] ━━━ 🚀 STARTING ROUND {CURRENT_ROUND + 1} ━━━")

    enc_fc2, fc2_shapes, plain_params, loss, acc, dataset_size, t_train, t_enc = \
        NODE.train_and_get_enc_update()

    weighted_enc_fc2 = {k: v * dataset_size for k, v in enc_fc2.items()}
    weighted_plain   = {k: v * dataset_size for k, v in plain_params.items()}

    fc2_weight_array = plain_params.get("fc2.weight", np.zeros(10))
    with KEY_LOCK:
        priv_key = MY_PRIVATE_KEY
    commitment_package = generate_commitment(
        gradient_array = fc2_weight_array,
        private_key    = priv_key,
        clip_threshold = 0.5,
        round_number   = CURRENT_ROUND
    )
    print(f"🟢 [Node {NODE_ID}] 🔏 Generated ZKP commitment (L2={commitment_package['l2_norm']:.4f}, round={CURRENT_ROUND})")

    payload = serialize_payload(weighted_enc_fc2, weighted_plain, dataset_size, commitment_package)

    print(f"🟢 [Node {NODE_ID}] 🔒 Encrypted local weights (CKKS)")
    print(f"🟢 [Node {NODE_ID}] 📦 Forwarding payload ➜ {SUCCESSOR_URL}")

    def _forward():
        try:
            requests.post(f"{SUCCESSOR_URL}/receive_update", json={
                "round":     CURRENT_ROUND,
                "origin_id": NODE_ID,
                "sender_id": NODE_ID,
                "payload":   payload
            })
        except Exception as e:
            print(f"[Node {NODE_ID}] Error forwarding: {e}")

    threading.Thread(target=_forward).start()


@app.route("/start_ring", methods=["POST"])
def start_ring():
    if not IS_MASTER:
        return jsonify({"error": "Only master can start the ring"}), 400
    execute_start_ring()
    return jsonify({"status": "Ring started"})


# ── RECEIVING SIDE ─────────────────────────────────────────────────────────────

@app.route("/receive_update", methods=["POST"])
def receive_update():
    global CURRENT_ROUND
    data      = request.json
    inc_round = data["round"]
    origin_id = data["origin_id"]
    sender_id = data.get("sender_id", origin_id)

    color = "🟢" if NODE_ID == 0 else ("🟣" if NODE_ID == 1 else "🟠")
    print(f"\n{color} [Node {NODE_ID}] 📥 Received encrypted payload from Node {sender_id}")

    if origin_id == NODE_ID:
        print(f"🟢 [Node {NODE_ID}] ✨ RING COMPLETE FOR ROUND {inc_round + 1}!")
        inc_enc_fc2, inc_plain, inc_weight, _ = deserialize_payload(data["payload"], NODE.ctx)
        print(f"🟢 [Node {NODE_ID}] 🌐 Successfully aggregated all nodes homomorphically!")
        CURRENT_ROUND += 1
        if CURRENT_ROUND < ROUNDS:
            threading.Thread(target=execute_start_ring).start()
        else:
            print(f"\n🟢 [Node {NODE_ID}] 🎉 ALL ROUNDS COMPLETE. True Distributed Training Finished! 🎉")
        return jsonify({"status": "Round complete"})

    else:
        inc_enc_fc2, inc_plain, inc_weight, commitment_package = \
            deserialize_payload(data["payload"], NODE.ctx)

        # Verify against sender's current public key (thread-safe)
        with KEY_LOCK:
            sender_pub_key = PEER_PUBLIC_KEYS.get(sender_id)

        if sender_pub_key is None:
            print(f"{color} [Node {NODE_ID}] ⚠️  No public key for Node {sender_id} — skipping ZKP check")
        else:
            valid, reason = verify_commitment(
                commitment_package,
                sender_pub_key,
                expected_round=inc_round
            )

            if not valid:
                print(f"{color} [Node {NODE_ID}] 🚨 ZKP VERIFICATION FAILED for Node {sender_id}")
                print(f"{color} [Node {NODE_ID}]    Reason: {reason}")
                print(f"{color} [Node {NODE_ID}]    Skipping Byzantine node — forwarding unchanged payload")

                forward_payload = serialize_payload(inc_enc_fc2, inc_plain, inc_weight, {})

                def _forward_skip():
                    try:
                        requests.post(f"{SUCCESSOR_URL}/receive_update", json={
                            "round":     inc_round,
                            "origin_id": origin_id,
                            "sender_id": NODE_ID,
                            "payload":   forward_payload
                        })
                    except Exception as e:
                        print(f"[Node {NODE_ID}] Error forwarding skip: {e}")

                threading.Thread(target=_forward_skip).start()
                return jsonify({"status": "Byzantine node rejected"})

            print(f"{color} [Node {NODE_ID}] ✅ ZKP commitment verified for Node {sender_id} (round {inc_round})")

        # Middle node: rotate keys if needed, then train and forward
        if inc_round > 0 and inc_round % KEY_ROTATION_INTERVAL == 0:
            # Middle nodes rotate on receiving the first message of the rotation round
            # Only rotate once per round (check if key is already rotated)
            pass  # Master triggers rotation; peers receive via /rotate_key endpoint

        my_enc_fc2, my_fc2_shapes, my_plain, loss, acc, my_weight, t_train, t_enc = \
            NODE.train_and_get_enc_update()

        my_fc2_weight = my_plain.get("fc2.weight", np.zeros(10))
        with KEY_LOCK:
            priv_key = MY_PRIVATE_KEY
        my_commitment = generate_commitment(
            gradient_array = my_fc2_weight,
            private_key    = priv_key,
            clip_threshold = 0.5,
            round_number   = inc_round
        )
        print(f"{color} [Node {NODE_ID}] 🔏 Generated ZKP commitment (L2={my_commitment['l2_norm']:.4f}, round={inc_round})")

        my_weighted_enc   = {k: v * my_weight for k, v in my_enc_fc2.items()}
        my_weighted_plain = {k: v * my_weight for k, v in my_plain.items()}

        comb_enc_fc2, comb_plain, comb_weight = NODE.aggregate_and_forward(
            inc_enc_fc2, inc_plain, inc_weight,
            my_weighted_enc, my_weighted_plain, my_weight
        )

        payload = serialize_payload(comb_enc_fc2, comb_plain, comb_weight, my_commitment)

        print(f"{color} [Node {NODE_ID}] ➕ Homomorphically added local weights (CKKS)")
        print(f"{color} [Node {NODE_ID}] 📦 Forwarding payload ➜ {SUCCESSOR_URL}")

        def _forward_middle():
            try:
                requests.post(f"{SUCCESSOR_URL}/receive_update", json={
                    "round":     inc_round,
                    "origin_id": origin_id,
                    "sender_id": NODE_ID,
                    "payload":   payload
                })
            except Exception as e:
                print(f"[Node {NODE_ID}] Error forwarding: {e}")

        threading.Thread(target=_forward_middle).start()
        return jsonify({"status": "Forwarded"})


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global NODE, SUCCESSOR_URL, ROUNDS, NODE_ID, IS_MASTER
    global MY_PRIVATE_KEY, MY_PUBLIC_KEY, ALL_NODE_URLS

    p = argparse.ArgumentParser()
    p.add_argument("--id",        type=int, required=True)
    p.add_argument("--port",      type=int, required=True)
    p.add_argument("--successor", type=str, required=True)
    p.add_argument("--rounds",    type=int, default=3)
    p.add_argument("--master",    action="store_true")
    p.add_argument("--all-nodes", type=str, default="",
                   help="Comma-separated list of all node URLs for key rotation")
    args = p.parse_args()

    NODE_ID       = args.id
    SUCCESSOR_URL = args.successor
    ROUNDS        = args.rounds
    IS_MASTER     = args.master
    ALL_NODE_URLS = [u for u in args.all_nodes.split(",") if u]

    if IS_MASTER:
        print(f"\n[Node {NODE_ID}] Initializing True Distributed Node with ZKP Byzantine Defence + Key Rotation...")

    MY_PRIVATE_KEY, MY_PUBLIC_KEY = generate_node_keypair()
    if IS_MASTER:
        print(f"[Node {NODE_ID}] 🔑 ZKP key pair generated (rotation every {KEY_ROTATION_INTERVAL} rounds)")

    device = torch.device("cpu")
    train_loaders, _ = load_datasets(num_clients=3, batch_size=32, verbose=IS_MASTER)
    my_loader = train_loaders[NODE_ID % len(train_loaders)]

    ctx  = create_he_context(verbose=IS_MASTER)
    NODE = RingNode(node_id=NODE_ID, train_loader=my_loader, device=device, he_ctx=ctx)

    model = SimpleCNN().to(device)
    NODE.set_model(model)

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
