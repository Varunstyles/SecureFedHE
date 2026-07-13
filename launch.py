"""
launch.py — SecureFedHE Ring Launcher
======================================
Node 0 handles ONE-TIME BOOTSTRAP ONLY: it waits for all other nodes
to come online, then fires the ring start signal for round 0. This is
NOT a permanent master role — proposing duty rotates every round after
that (Leader(t) = t mod M, see get_proposer_for_round() in node.py).
Every node is an equal peer once the ring is running.

Usage:
    # On Node 0 (bootstraps the ring):
    python launch.py --id 0

    # On Nodes 1-4 (each other PC), run BEFORE Node 0:
    python launch.py --id 1   # (or 2, 3, 4)

Options:
    --id       Node ID (0=bootstraps the ring, 1-4=join as peers)
    --config   Path to config.json (default: config.json)
    --dev      HTTP mode — no TLS (for local testing on one PC)
    --rounds   Override number of rounds from config
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import json
import time
import argparse
import subprocess
import threading
import requests
from pathlib import Path

ROOT = Path(__file__).parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def node_url(node: dict, dev: bool) -> str:
    scheme = "http" if dev else "https"
    return f"{scheme}://{node['ip']}:{node['port']}"


def wait_for_node(url: str, ca_cert: str, client_cert: str, client_key: str,
                  dev: bool, timeout: int = 120) -> bool:
    """Poll /status until node is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if dev:
                r = requests.get(f"{url}/status", timeout=3)
            else:
                r = requests.get(f"{url}/status", timeout=3,
                                  verify=ca_cert, cert=(client_cert, client_key))
            if r.status_code == 200 and r.json().get("zkp_ready"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_local_node(node_id: int, config_path: str, dev: bool, rounds: int = None) -> subprocess.Popen:
    """Spawn node.py as a subprocess."""
    cmd = [sys.executable, str(ROOT / "node.py"), "--id", str(node_id),
           "--config", config_path]
    if dev:
        cmd.append("--dev")
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)
    return proc


def trigger_start(url: str, ca_cert: str, client_cert: str, client_key: str,
                   dev: bool):
    """Send POST /start_ring to master node."""
    try:
        if dev:
            r = requests.post(f"{url}/start_ring", timeout=10)
        else:
            r = requests.post(f"{url}/start_ring", timeout=10,
                               verify=ca_cert, cert=(client_cert, client_key))
        print(f"Ring started: {r.json()}")
    except Exception as e:
        print(f"Failed to start ring: {e}")


def print_banner(config: dict, node_id: int, dev: bool):
    nodes = config["ring"]["nodes"]
    node  = nodes[node_id]
    print("\n" + "=" * 55)
    print("  SecureFedHE — Federated Learning Ring")
    print("=" * 55)
    print(f"  Node:    {node_id} — {node['name']}")
    print(f"  Address: {node['ip']}:{node['port']}")
    print(f"  Rounds:  {config['ring']['rounds']}")
    print(f"  ε (DP):  {config['privacy']['dp_epsilon']}")
    print(f"  Mode:    {'HTTP (dev)' if dev else 'HTTPS (mTLS)'}")
    print(f"  Ring:    {len(nodes)} hospitals")
    print("=" * 55)
    print("\n  Ring members:")
    for n in nodes:
        marker = " ← YOU" if n["id"] == node_id else ""
        print(f"    [{n['id']}] {n['name']:<35} {n['ip']}:{n['port']}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(description="SecureFedHE Launcher")
    parser.add_argument("--id",     type=int, required=True,  help="This node's ID (0-4)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--dev",    action="store_true", help="HTTP mode, no TLS")
    parser.add_argument("--rounds", type=int, default=None,   help="Override rounds")
    args = parser.parse_args()

    config   = load_config(args.config)
    nodes    = config["ring"]["nodes"]
    nid      = args.id
    tls      = config["tls"]
    dev      = args.dev

    if args.rounds:
        config["ring"]["rounds"] = args.rounds

    print_banner(config, nid, dev)

    # ── Check certs exist (unless dev mode) ───────────────────
    if not dev:
        for cert_path in [tls["ca_cert"], tls["server_cert"],
                          tls["server_key"], tls["client_cert"], tls["client_key"]]:
            if not os.path.exists(cert_path):
                print(f"ERROR: Certificate not found: {cert_path}")
                print("Run:  python generate_certs.py")
                sys.exit(1)

    # ── Start local node ──────────────────────────────────────
    print(f"Starting local node {nid}...")
    proc = start_local_node(nid, args.config, dev)

    my_url = node_url(nodes[nid], dev)

    # ── Wait for self to come up ──────────────────────────────
    print(f"Waiting for node {nid} to be ready...")
    ready = wait_for_node(my_url, tls["ca_cert"], tls["client_cert"], tls["client_key"],
                           dev, timeout=120)
    if not ready:
        print(f"ERROR: Node {nid} did not start in time")
        proc.terminate()
        sys.exit(1)
    print(f"Node {nid} is ready.")

    # ── Bootstrap: node 0 waits for peers, then fires the ONE-TIME
    # start signal for round 0. This is NOT a permanent master role —
    # proposing duty rotates every round after that (Leader(t) = t mod M).
    # Node 0 is simply the fixed point that initiates the ring once.
    if nid == 0:
        print("\nBootstrapping: waiting for all other nodes to come online...")
        all_ready = True
        for node in nodes[1:]:
            url = node_url(node, dev)
            print(f"  Waiting for Node {node['id']} ({node['name']}) at {url}...")
            if wait_for_node(url, tls["ca_cert"], tls["client_cert"], tls["client_key"],
                              dev, timeout=120):
                print(f"  ✓ Node {node['id']} ready")
            else:
                print(f"  ✗ Node {node['id']} did not respond in time")
                all_ready = False

        if not all_ready:
            print("\nWarning: not all nodes responded. Starting anyway...")

        print("\nAll nodes ready — starting federated training ring!")
        time.sleep(1)
        trigger_start(my_url, tls["ca_cert"], tls["client_cert"], tls["client_key"], dev)

    # ── Keep process alive ────────────────────────────────────
    print(f"\nNode {nid} running. Press Ctrl+C to stop.\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\nShutting down Node {nid}...")
        proc.terminate()
        proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
