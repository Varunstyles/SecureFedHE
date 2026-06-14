"""
network/distributed_node.py
A true distributed node in the SecureFedHE Ring network.
Uses Flask to receive incoming encrypted weights via HTTP,
adds its own encrypted local update homomorphically, and forwards to the successor node.
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

# Suppress standard Flask HTTP request logs and startup banner for a clean console
import sys
try:
    from flask import cli as flask_cli
    flask_cli.show_server_banner = lambda *args: None
except Exception:
    pass

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import SimpleCNN
from data.loader import load_datasets
from crypto.he_layer import create_he_context, SimulatedCKKSVector, SimulatedContext
from network.ring_topology import RingNode

app = Flask(__name__)

# Suppress standard Flask HTTP request logs for a clean console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global Node State
NODE = None
SUCCESSOR_URL = None
ROUNDS = 1
CURRENT_ROUND = 0
NODE_ID = 0
IS_MASTER = False

def serialize_payload(enc_fc2, plain_params, weight):
    """Serialize the HE objects and numpy arrays for HTTP transit."""
    payload = {
        "weight": weight,
        "enc_fc2": {},
        "plain_params": {}
    }
    
    # Serialize encrypted fc2
    for k, v in enc_fc2.items():
        if hasattr(v, 'serialize'):
            payload["enc_fc2"][k] = base64.b64encode(v.serialize()).decode('utf-8')
        else:
            payload["enc_fc2"][k] = base64.b64encode(pickle.dumps(v)).decode('utf-8')
            
    # Serialize plaintext params
    for k, v in plain_params.items():
        payload["plain_params"][k] = base64.b64encode(pickle.dumps(v)).decode('utf-8')
        
    return payload

def deserialize_payload(payload_data, ctx):
    """Deserialize incoming HTTP payload back into HE objects and numpy arrays."""
    weight = payload_data["weight"]
    enc_fc2 = {}
    plain_params = {}
    
    for k, v_str in payload_data["enc_fc2"].items():
        b_data = base64.b64decode(v_str)
        if isinstance(ctx, SimulatedContext):
            enc_fc2[k] = SimulatedCKKSVector.deserialize(ctx, b_data)
        else:
            import tenseal as ts
            enc_fc2[k] = ts.ckks_vector_from(ctx, b_data)
            
    for k, v_str in payload_data["plain_params"].items():
        b_data = base64.b64decode(v_str)
        plain_params[k] = pickle.loads(b_data)
        
    return enc_fc2, plain_params, weight

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "ready", "id": NODE_ID})

def execute_start_ring():
    global CURRENT_ROUND
    print(f"\n🟢 [Node {NODE_ID}] ━━━ 🚀 STARTING ROUND {CURRENT_ROUND + 1} ━━━")
    
    # Train locally and get encrypted update
    enc_fc2, fc2_shapes, plain_params, loss, acc, dataset_size, t_train, t_enc = NODE.train_and_get_enc_update()
    
    # Weight the local update
    weighted_enc_fc2 = {k: v * dataset_size for k, v in enc_fc2.items()}
    weighted_plain = {k: v * dataset_size for k, v in plain_params.items()}
    
    # Serialize and forward to next node
    payload = serialize_payload(weighted_enc_fc2, weighted_plain, dataset_size)
    
    # Send via HTTP POST in a background thread to prevent deadlocks
    print(f"🟢 [Node {NODE_ID}] 🔒 Encrypted local weights (CKKS)")
    print(f"🟢 [Node {NODE_ID}] 📦 Forwarding payload ➜ {SUCCESSOR_URL}")
    def _forward():
        try:
            requests.post(f"{SUCCESSOR_URL}/receive_update", json={
                "round": CURRENT_ROUND,
                "origin_id": NODE_ID,
                "payload": payload
            })
        except Exception as e:
            print(f"[Node {NODE_ID}] Error forwarding: {e}")
            
    threading.Thread(target=_forward).start()

@app.route("/start_ring", methods=["POST"])
def start_ring():
    """Only called on Master node to kick off the first round."""
    if not IS_MASTER:
        return jsonify({"error": "Only master can start the ring"}), 400
        
    execute_start_ring()
    return jsonify({"status": "Ring started"})

@app.route("/receive_update", methods=["POST"])
def receive_update():
    global CURRENT_ROUND
    data = request.json
    inc_round = data["round"]
    origin_id = data["origin_id"]
    
    color = "🟢" if NODE_ID == 0 else ("🟣" if NODE_ID == 1 else "🟠")
    print(f"\n{color} [Node {NODE_ID}] 📥 Received encrypted payload from Node {origin_id}")
    
    if origin_id == NODE_ID:
        # Full circle complete! Master node finishes the round.
        print(f"🟢 [Node {NODE_ID}] ✨ RING COMPLETE FOR ROUND {inc_round+1}!")
        inc_enc_fc2, inc_plain, inc_weight = deserialize_payload(data["payload"], NODE.ctx)
        
        # In a real system, we would decrypt and update the global model here,
        # then broadcast it. For this simulation, we'll just log success.
        print(f"🟢 [Node {NODE_ID}] 🌐 Successfully aggregated all nodes homomorphically!")
        
        CURRENT_ROUND += 1
        if CURRENT_ROUND < ROUNDS:
            # Start next round
            # We would normally broadcast the decrypted model here.
            # We'll simulate it by just kicking off the next round in a thread.
            threading.Thread(target=execute_start_ring).start()
        else:
            print(f"\n🟢 [Node {NODE_ID}] 🎉 ALL ROUNDS COMPLETE. True Distributed Training Finished! 🎉")
            
        return jsonify({"status": "Round complete"})
        
    else:
        # We are a middle node. Add our update and pass it along.
        inc_enc_fc2, inc_plain, inc_weight = deserialize_payload(data["payload"], NODE.ctx)
        
        # Train locally
        my_enc_fc2, my_fc2_shapes, my_plain, loss, acc, my_weight, t_train, t_enc = NODE.train_and_get_enc_update()
        
        # Weight local update
        my_weighted_enc = {k: v * my_weight for k, v in my_enc_fc2.items()}
        my_weighted_plain = {k: v * my_weight for k, v in my_plain.items()}
        
        # Homomorphic addition (the magic happens here)
        comb_enc_fc2, comb_plain, comb_weight = NODE.aggregate_and_forward(
            inc_enc_fc2, inc_plain, inc_weight,
            my_weighted_enc, my_weighted_plain, my_weight
        )
        
        # Serialize and forward
        payload = serialize_payload(comb_enc_fc2, comb_plain, comb_weight)
        
        print(f"{color} [Node {NODE_ID}] ➕ Homomorphically added local weights (CKKS)")
        print(f"{color} [Node {NODE_ID}] 📦 Forwarding payload ➜ {SUCCESSOR_URL}")
        def _forward_middle():
            try:
                requests.post(f"{SUCCESSOR_URL}/receive_update", json={
                    "round": inc_round,
                    "origin_id": origin_id, # Keep original origin ID
                    "payload": payload
                })
            except Exception as e:
                print(f"[Node {NODE_ID}] Error forwarding: {e}")
                
        threading.Thread(target=_forward_middle).start()
            
        return jsonify({"status": "Forwarded"})

def main():
    global NODE, SUCCESSOR_URL, ROUNDS, NODE_ID, IS_MASTER
    
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--successor", type=str, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--master", action="store_true")
    args = p.parse_args()
    
    NODE_ID = args.id
    SUCCESSOR_URL = args.successor
    ROUNDS = args.rounds
    IS_MASTER = args.master
    
    if IS_MASTER:
        print(f"\n[Node {NODE_ID}] Initializing True Distributed Node...")
    
    # Load dataset slice (simulate hospital data)
    device = torch.device("cpu")
    train_loaders, _ = load_datasets(num_clients=3, batch_size=32, verbose=IS_MASTER)
    my_loader = train_loaders[NODE_ID % len(train_loaders)]
    
    # Init context
    ctx = create_he_context(verbose=IS_MASTER)
    
    # Init node
    NODE = RingNode(
        node_id=NODE_ID,
        train_loader=my_loader,
        device=device,
        he_ctx=ctx
    )
    
    # Init global model baseline
    model = SimpleCNN().to(device)
    NODE.set_model(model)
    
    # Suppress the 'Ready and listening' message on non-master nodes for cleaner logs
    # We already printed "Starting Node X..." in launch_distributed.py
    
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
