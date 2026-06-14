"""
network/launch_distributed.py
Orchestrator to launch and test the True Distributed Ring Network.
Spawns 3 separate Flask servers, waits for them to be healthy, and triggers Node 0.
"""

import os
import sys
import time
import subprocess
import requests

def main():
    print("==========================================================")
    print("Launching True Distributed SecureFedHE Network")
    print("==========================================================\n")
    
    python_exe = sys.executable
    node_script = os.path.join(os.path.dirname(__file__), "distributed_node.py")
    
    # Define topology
    # Node 0 (Port 5000) -> Node 1 (Port 5001) -> Node 2 (Port 5002) -> Node 0
    nodes = [
        {"id": 0, "port": 5000, "successor": "http://127.0.0.1:5001", "master": True},
        {"id": 1, "port": 5001, "successor": "http://127.0.0.1:5002", "master": False},
        {"id": 2, "port": 5002, "successor": "http://127.0.0.1:5000", "master": False},
    ]
    
    processes = []
    
    try:
        # 1. Launch all nodes in separate background processes
        for n in nodes:
            cmd = [
                python_exe, node_script,
                "--id", str(n["id"]),
                "--port", str(n["port"]),
                "--successor", n["successor"]
            ]
            if n["master"]:
                cmd.append("--master")
                
            print(f"Starting Node {n['id']} on Port {n['port']}...")
            # Run with PYTHONUNBUFFERED=1 so prints show up immediately!
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)
            
        print("\nWaiting for servers to initialize (5 seconds)...")
        time.sleep(5)
        
        # 2. Verify health
        for n in nodes:
            try:
                resp = requests.get(f"http://127.0.0.1:{n['port']}/status")
                if resp.status_code == 200:
                    print(f"  [OK] Node {n['id']} is healthy")
                else:
                    print(f"  [FAIL] Node {n['id']} returned {resp.status_code}")
            except requests.ConnectionError:
                print(f"  [FAIL] Node {n['id']} is not responding")
                
        # 3. Trigger Ring Training on Master (Node 0)
        print("\nSending Start signal to Master Node (Node 0)...")
        try:
            resp = requests.post("http://127.0.0.1:5000/start_ring")
            if resp.status_code == 200:
                print("Signal received! Watch the logs above as nodes train and pass encrypted data via HTTP.")
        except requests.ConnectionError:
            print("Failed to contact Master Node.")
            
        # 4. Wait for user interrupt
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
