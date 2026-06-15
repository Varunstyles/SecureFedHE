"""
network/launch_distributed.py  —  SecureFedHE (ZKP Fix 2)
==========================================================
Changes from original:
  - Added Step 2.5: ZKP public key exchange between all nodes
    Each node calls /get_public_key on its peers and registers them
    via /register_peer_key. This happens once before training starts.
  - Everything else (topology, health check, start signal) unchanged.
"""

import os
import sys
import time
import subprocess
import requests

def main():
    print("==========================================================")
    print("Launching True Distributed SecureFedHE Network (ZKP Fix 2)")
    print("==========================================================\n")

    python_exe  = sys.executable
    node_script = os.path.join(os.path.dirname(__file__), "distributed_node.py")

    # Topology: 0 → 1 → 2 → 0
    nodes = [
        {"id": 0, "port": 5000, "successor": "http://127.0.0.1:5001", "master": True},
        {"id": 1, "port": 5001, "successor": "http://127.0.0.1:5002", "master": False},
        {"id": 2, "port": 5002, "successor": "http://127.0.0.1:5000", "master": False},
    ]

    processes = []

    try:
        # ── Step 1: Launch all nodes ──────────────────────────────────────
        for n in nodes:
            cmd = [
                python_exe, node_script,
                "--id",        str(n["id"]),
                "--port",      str(n["port"]),
                "--successor", n["successor"]
            ]
            if n["master"]:
                cmd.append("--master")

            print(f"Starting Node {n['id']} on Port {n['port']}...")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)

        print("\nWaiting for servers to initialize (5 seconds)...")
        time.sleep(5)

        # ── Step 2: Health check ──────────────────────────────────────────
        for n in nodes:
            try:
                resp = requests.get(f"http://127.0.0.1:{n['port']}/status")
                if resp.status_code == 200:
                    print(f"  [OK] Node {n['id']} is healthy")
                else:
                    print(f"  [FAIL] Node {n['id']} returned {resp.status_code}")
            except requests.ConnectionError:
                print(f"  [FAIL] Node {n['id']} is not responding")

        # ── Step 2.5: ZKP public key exchange (NEW) ───────────────────────
        # Each node fetches every other node's public key and registers it.
        # This is the one-time setup cost for ZKP — happens before training.
        print("\nExchanging ZKP public keys between nodes...")

        node_keys = {}
        for n in nodes:
            try:
                resp = requests.get(f"http://127.0.0.1:{n['port']}/get_public_key")
                data = resp.json()
                node_keys[data["node_id"]] = data["public_key"]
                print(f"  [OK] Retrieved public key from Node {n['id']}")
            except Exception as e:
                print(f"  [FAIL] Could not get key from Node {n['id']}: {e}")

        # Register each key with every other node
        for n in nodes:
            for peer_id, pub_key_pem in node_keys.items():
                if peer_id == n["id"]:
                    continue   # don't register a node's own key with itself
                try:
                    requests.post(
                        f"http://127.0.0.1:{n['port']}/register_peer_key",
                        json={"node_id": peer_id, "public_key": pub_key_pem}
                    )
                except Exception as e:
                    print(f"  [FAIL] Could not register Node {peer_id} key with Node {n['id']}: {e}")

        print("  ZKP key exchange complete — all nodes can now verify each other's proofs\n")

        # ── Step 3: Start training ─────────────────────────────────────────
        print("Sending Start signal to Master Node (Node 0)...")
        try:
            resp = requests.post("http://127.0.0.1:5000/start_ring")
            if resp.status_code == 200:
                print("Signal received! Nodes are now training with ZKP Byzantine defence.")
        except requests.ConnectionError:
            print("Failed to contact Master Node.")

        print("\nPress Ctrl+C to shut down all nodes...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down distributed network...")
    finally:
        for p in processes:
            p.terminate()
            p.wait()
        print("All nodes terminated.")

if __name__ == "__main__":
    main()
