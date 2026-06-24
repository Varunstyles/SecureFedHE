"""
dev_run.py — SecureFedHE Single-PC Dev Runner
===============================================
Starts all 5 hospital nodes in one terminal window.
Each node runs as a background subprocess.
Master (Node 0) waits for all others, then fires the ring.

Usage:
    python dev_run.py
    python dev_run.py --config config_dev.json
    python dev_run.py --rounds 5     # quick test with 5 rounds
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import time
import json
import argparse
import subprocess
import requests
import threading
from pathlib import Path

ROOT = Path(__file__).parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def wait_for_node(node_id: int, ip: str, port: int, timeout: int = 60) -> bool:
    url      = f"http://{ip}:{port}/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("zkp_ready"):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stream_output(proc, prefix):
    """Stream subprocess output with a node prefix."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode(errors="replace").rstrip()
        if text:
            print(f"{prefix} {text}")


def main():
    parser = argparse.ArgumentParser(description="SecureFedHE Dev Runner")
    parser.add_argument("--config", default="config_dev.json")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    nodes  = config["ring"]["nodes"]

    if args.rounds:
        config["ring"]["rounds"] = args.rounds
        # write temp config
        tmp_cfg = str(ROOT / "config_dev_tmp.json")
        with open(tmp_cfg, "w") as f:
            json.dump(config, f, indent=2)
        cfg_path = tmp_cfg
    else:
        cfg_path = args.config

    print("\n" + "=" * 55)
    print("  SecureFedHE — Dev Mode (All Nodes, One Terminal)")
    print("=" * 55)
    print(f"  Nodes:  {len(nodes)}")
    print(f"  Rounds: {config['ring']['rounds']}")
    print(f"  ε (DP): {config['privacy']['dp_epsilon']}")
    print(f"  Config: {cfg_path}")
    print("=" * 55 + "\n")

    procs = []

    # ── Start worker nodes (1 to N-1) first ────────────────────
    for node in nodes[1:]:
        nid = node["id"]
        cmd = [
            sys.executable, str(ROOT / "node.py"),
            "--id", str(nid),
            "--config", cfg_path,
            "--dev"
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append((nid, proc))
        prefix = f"[Node {nid}]"
        t = threading.Thread(target=stream_output, args=(proc, prefix), daemon=True)
        t.start()
        print(f"  Started Node {nid} ({node['name']}) on port {node['port']}")
        time.sleep(0.5)  # stagger startup slightly

    # ── Wait for all workers to be ready ───────────────────────
    print("\nWaiting for worker nodes to be ready...")
    for node in nodes[1:]:
        nid  = node["id"]
        ok   = wait_for_node(nid, node["ip"], node["port"], timeout=120)
        status = "✓ ready" if ok else "✗ timeout"
        print(f"  Node {nid} ({node['name']}): {status}")

    # ── Start master node (Node 0) ──────────────────────────────
    print(f"\nStarting master Node 0 ({nodes[0]['name']})...")
    cmd = [
        sys.executable, str(ROOT / "node.py"),
        "--id", "0",
        "--config", cfg_path,
        "--dev"
    ]
    proc0 = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    procs.insert(0, (0, proc0))
    t = threading.Thread(target=stream_output, args=(proc0, "[Node 0]"), daemon=True)
    t.start()

    # ── Wait for master to be ready, then fire ring ─────────────
    master = nodes[0]
    ok = wait_for_node(0, master["ip"], master["port"], timeout=120)
    if ok:
        print("\nAll nodes ready — firing ring!\n")
        try:
            requests.post(
                f"http://{master['ip']}:{master['port']}/start_ring",
                timeout=10
            )
        except Exception as e:
            print(f"Warning: could not fire ring via API: {e}")
            print("Node 0 will start automatically once it detects all peers.")
    else:
        print("Warning: master node did not respond in time.")

    print("\n" + "=" * 55)
    print("  Training in progress. Press Ctrl+C to stop all nodes.")
    print("=" * 55 + "\n")

    # ── Keep alive until Ctrl+C ─────────────────────────────────
    try:
        while True:
            # Check if all procs have exited
            alive = [p for _, p in procs if p.poll() is None]
            if not alive:
                print("\nAll nodes have exited.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nStopping all nodes...")
        for nid, proc in procs:
            proc.terminate()
            print(f"  Stopped Node {nid}")
        for _, proc in procs:
            proc.wait()
        print("Done.")

    # Cleanup temp config
    if args.rounds and os.path.exists(cfg_path):
        os.remove(cfg_path)


if __name__ == "__main__":
    main()
