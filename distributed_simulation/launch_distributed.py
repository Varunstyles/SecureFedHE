"""
network/launch_distributed.py  —  SecureFedHE (ZKP Fix 2, v3 — Key Rotation)
==============================================================================
Changes from v2:
  - Passes --all-nodes argument to each node so they know all peer URLs
    for broadcasting rotated public keys during key rotation
"""

import os
import sys
import time
import subprocess
import requests

def main():
    print("==========================================================")
    print("Launching True Distributed SecureFedHE Network (ZKP + Key Rotation)")
    print("==========================================================\n")

    python_exe  = sys.executable
    node_script = os.path.join(os.path.dirname(__file__), "distributed_node.py")

    nodes = [
        {"id": 0, "port": 5000, "successor": "http://127.0.0.1:5001", "master": True},
        {"id": 1, "port": 5001, "successor": "http://127.0.0.1:5002", "master": False},
        {"id": 2, "port": 5002, "successor": "http://127.0.0.1:5000", "master": False},
    ]

    # All node URLs for key rotation broadcast
    all_node_urls = ",".join([f"http://127.0.0.1:{n['port']}" for n in nodes])

    processes = []

    try:
        # ── Step 1: Launch all nodes ──────────────────────────────────────
        for n in nodes:
            cmd = [
                python_exe, node_script,
                "--id",        str(n["id"]),
                "--port",      str(n["port"]),
                "--successor", n["successor"],
                "--all-nodes", all_node_urls      # ← NEW: for key rotation
            ]
            if n["master"]:
                cmd.append("--master")

            print(f"Starting Node {n['id']} on Port {n['port']}...")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)

        print("\nWaiting for all nodes to be ready...")
        for n in nodes:
            for attempt in range(60):   # wait up to 60 seconds per node
                try:
                    resp = requests.get(f"http://127.0.0.1:{n['port']}/status", timeout=2)
                    if resp.status_code == 200:
                        print(f"  [OK] Node {n['id']} is healthy")
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                print(f"  [FAIL] Node {n['id']} never responded")

        # ── Step 3: ZKP public key exchange ──────────────────────────────
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

        for n in nodes:
            for peer_id, pub_key_pem in node_keys.items():
                if peer_id == n["id"]:
                    continue
                try:
                    requests.post(
                        f"http://127.0.0.1:{n['port']}/register_peer_key",
                        json={"node_id": peer_id, "public_key": pub_key_pem}
                    )
                except Exception as e:
                    print(f"  [FAIL] Could not register key: {e}")

        print("  ZKP key exchange complete\n")

        # ── Step 4: Start training ────────────────────────────────────────
        print("Sending Start signal to Master Node (Node 0)...")
        try:
            resp = requests.post("http://127.0.0.1:5000/start_ring")
            if resp.status_code == 200:
                print("Signal received! Training with ZKP Byzantine defence + key rotation.")
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
