"""
run_all.py
Master runner — executes all three rings sequentially and produces
all CSV data needed for the Phase 4 evaluation dashboard.

Usage:
    python run_all.py                    # Full run (all rings, all epsilons)
    python run_all.py --quick            # Quick test (3 rounds, 3 clients)
    python run_all.py --skip-baseline    # Skip Ring 1 if already done

Output files (in evaluation/logs/):
    baseline_metrics.csv      ← Ring 1
    he_eps10_metrics.csv      ← Ring 2, ε=10
    he_eps20_metrics.csv      ← Ring 2, ε=20
    he_eps50_metrics.csv      ← Ring 2, ε=50
    ring_metrics.csv          ← Ring 3
"""

import argparse
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(label, cmd, cwd):
    """Run a subprocess and stream its output."""
    print(f"\n{'-'*65}")
    print(f"  >> {label}")
    print(f"{'-'*65}\n")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  [FAIL] {label} FAILED (exit code {result.returncode})")
        print(f"  Elapsed: {elapsed:.1f}s")
        return False

    print(f"\n  [OK] {label} completed in {elapsed:.1f}s")
    return True


def main():
    p = argparse.ArgumentParser(description="SecureFedHE - Run All Experiments")
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode (3 rounds, 3 clients)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip Ring 1 if baseline_metrics.csv already exists")
    p.add_argument("--rounds", type=int, default=20,
                   help="Number of FL rounds (default: 20)")
    p.add_argument("--clients", type=int, default=5,
                   help="Number of clients/nodes (default: 5)")
    args = p.parse_args()

    if args.quick:
        rounds = 3
        clients = 3
        print("\n** QUICK TEST MODE -- 3 rounds, 3 clients **\n")
    else:
        rounds = args.rounds
        clients = args.clients

    cwd = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
    epsilons = [10, 20, 50]

    print(f"""
{'='*65}
  +-----------------------------------------------------------+
  |          SecureFedHE -- Full Experiment Suite              |
  +-----------------------------------------------------------+
  |  Ring 1: Vanilla FL baseline                              |
  |  Ring 2: Selective HE + DP (eps = {', '.join(map(str, epsilons)):>15s})   |
  |  Ring 3: Decentralised ring topology                      |
  +-----------------------------------------------------------+
  |  Rounds: {rounds:>3d} | Clients: {clients:>3d}                             |
  +-----------------------------------------------------------+
{'='*65}
""")

    os.makedirs(os.path.join(cwd, "evaluation", "logs"), exist_ok=True)
    t_total = time.perf_counter()
    results = []

    # ── Ring 1: Vanilla FL Baseline ───────────────────────────────────────
    baseline_csv = os.path.join(cwd, "evaluation", "logs", "baseline_metrics.csv")
    if args.skip_baseline and os.path.exists(baseline_csv):
        print(f"\n  [SKIP] Skipping Ring 1 (baseline_metrics.csv exists)")
        results.append(("Ring 1 - Baseline", "SKIPPED"))
    else:
        # Delete old CSV so profiler writes fresh header
        if os.path.exists(baseline_csv):
            os.remove(baseline_csv)
        ok = run_step(
            "Ring 1 — Vanilla FL Baseline",
            f'"{python}" -m baseline.train --rounds {rounds} --clients {clients}',
            cwd,
        )
        results.append(("Ring 1 - Baseline", "OK" if ok else "FAIL"))

    # ── Ring 2: Selective HE at multiple ε values ─────────────────────────
    for eps in epsilons:
        eps_csv = os.path.join(cwd, "evaluation", "logs", f"he_eps{eps}_metrics.csv")
        if os.path.exists(eps_csv):
            os.remove(eps_csv)
        ok = run_step(
            f"Ring 2 - Selective HE (eps={eps})",
            f'"{python}" -m baseline.he_train --rounds {rounds} --clients {clients} --epsilon {eps}',
            cwd,
        )
        results.append((f"Ring 2 - HE eps={eps}", "OK" if ok else "FAIL"))

    # ── Ring 3: Decentralised Ring ─────────────────────────────────────────
    ring_csv = os.path.join(cwd, "evaluation", "logs", "ring_metrics.csv")
    if os.path.exists(ring_csv):
        os.remove(ring_csv)
    ok = run_step(
        "Ring 3 — Decentralised Ring Topology",
        f'"{python}" -m network.ring_train --rounds {rounds} --clients {clients}',
        cwd,
    )
    results.append(("Ring 3 - Ring", "OK" if ok else "FAIL"))

    # ── Final Summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    print(f"\n{'='*65}")
    print(f"  +-----------------------------------------------------------+")
    print(f"  |              EXPERIMENT SUITE COMPLETE                    |")
    print(f"  +-----------------------------------------------------------+")
    print(f"{'='*65}\n")

    print(f"  Results:")
    for name, status in results:
        print(f"    {status}  {name}")

    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"\n  Output CSV files:")
    log_dir = os.path.join(cwd, "evaluation", "logs")
    for f in sorted(os.listdir(log_dir)):
        if f.endswith(".csv"):
            fpath = os.path.join(log_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f} ({size_kb:.1f} KB)")

    print(f"\n  -> Open dashboard/index.html for Phase 4 visualisation")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
