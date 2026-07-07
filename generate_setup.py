"""
generate_setup.py — SecureFedHE
Run this ONCE, on any single PC, to generate the shared Groth16 trusted
setup (pk/vk) for the norm-bound ZKP circuit. Copy the resulting
zkp_setup.json to the other PCs before starting the ring — every node
must use the identical file, or proof verification will fail (each
node's vk must match the pk used to build the proof).

Contains no secret material: tau/alpha/beta/gamma/delta ("toxic waste")
are sampled and discarded inside setup() itself and never written out.

Usage:
    python generate_setup.py
"""
import sys
import json
import threading

sys.setrecursionlimit(100000)


def main():
    config = json.load(open("config.json"))
    n = config["zkp"]["gradient_dim"]

    threading.stack_size(64 * 1024 * 1024)  # py_ecc's field pow() is recursive
    result = {}

    def run():
        sys.setrecursionlimit(100000)
        from distributed_simulation.trusted_setup import setup
        result["pk"], result["vk"] = setup(n)

    t = threading.Thread(target=run)
    t.start()
    t.join()

    if "pk" not in result:
        print("ERROR: setup() failed — see traceback above.")
        sys.exit(1)

    from distributed_simulation.trusted_setup import save_setup
    save_setup(result["pk"], result["vk"], "zkp_setup.json")
    print(f"Wrote zkp_setup.json (circuit_id={result['pk'].circuit_id}, gradient_dim={n})")
    print("Copy this exact file to the other 2 PCs (git is fine — no secrets in it)")
    print("before starting the ring. Do NOT run this script again on the other PCs —")
    print("every node must share this SAME file, not generate its own.")


if __name__ == "__main__":
    main()