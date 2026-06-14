"""
network/ring_train.py
Ring 3 — Decentralised Ring Topology Training Loop.

Main entry point for Ring 3 experiments.
Run AFTER Ring 1 and Ring 2 are confirmed working.

Usage:
    python -m network.ring_train
    python -m network.ring_train --rounds 20 --clients 5 --epsilon 10
    python -m network.ring_train --rounds 5 --clients 3  # quick test

What this produces:
    evaluation/logs/ring_metrics.csv      ← same schema as baseline + he metrics
    evaluation/logs/best_ring.pth         ← best model weights
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.he_layer import create_he_context
from data.loader import load_datasets
from evaluation.metrics import Profiler, compute_accuracy, model_size_bytes
from models.cnn import SimpleCNN
from network.ring_topology import RingNode, build_ring, run_ring_round


def parse_args():
    p = argparse.ArgumentParser(description="SecureFedHE — Ring 3 Decentralised Ring")
    p.add_argument("--rounds",     type=int,   default=20)
    p.add_argument("--clients",    type=int,   default=5,
                   help="Number of nodes in the ring")
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--partition",  type=str,   default="noniid",
                   choices=["iid", "noniid"])
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--epsilon",    type=float, default=10.0,
                   help="DP epsilon for fc1 layer")
    p.add_argument("--delta",      type=float, default=1e-5)
    p.add_argument("--sensitivity",type=float, default=0.1)
    p.add_argument("--he-degree",  type=int,   default=8192)
    p.add_argument("--data-dir",   type=str,   default="./data/cifar10")
    p.add_argument("--log",        type=str,
                   default="evaluation/logs/ring_metrics.csv")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  SecureFedHE - Ring 3 - Decentralised Ring Topology")
    print(f"  Device : {device} | Rounds : {args.rounds} | Nodes : {args.clients}")
    print(f"  DP eps={args.epsilon} | Topology: {args.clients}-node ring")
    print(f"  No central server - peer-to-peer HE aggregation")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loaders, test_loader = load_datasets(
        data_dir=args.data_dir,
        num_clients=args.clients,
        partition=args.partition,
        alpha=args.alpha,
        batch_size=args.batch,
        seed=args.seed,
    )

    # ── HE Context ────────────────────────────────────────────────────────
    print("\n[HE] Generating CKKS context...", end=" ", flush=True)
    t0 = time.perf_counter()
    he_ctx = create_he_context(poly_modulus_degree=args.he_degree)
    print(f"done in {time.perf_counter()-t0:.2f}s")

    # ── Global model ──────────────────────────────────────────────────────
    global_model = SimpleCNN(num_classes=10).to(device)
    print(f"[Model] Parameters : {sum(p.numel() for p in global_model.parameters()):,}")

    # ── Build Ring ────────────────────────────────────────────────────────
    nodes = [
        RingNode(
            node_id=i,
            train_loader=train_loaders[i],
            device=device,
            he_ctx=he_ctx,
            lr=args.lr,
            local_epochs=args.epochs,
            dp_epsilon=args.epsilon,
            dp_delta=args.delta,
            dp_sensitivity=args.sensitivity,
        )
        for i in range(args.clients)
    ]
    build_ring(nodes)

    ring_str = " -> ".join([f"[{n.id}]" for n in nodes]) + f" -> [{nodes[0].id}]"
    print(f"[Ring] Topology: {ring_str}\n")

    profiler = Profiler(log_path=args.log)
    best_acc = 0.0

    # -- Ring 3 Training Loop ----------------------------------------------
    for rnd in range(1, args.rounds + 1):
        print(f"-- Round {rnd:02d}/{args.rounds} ", end="", flush=True)

        profiler.start()

        (global_model,
         avg_loss, avg_acc,
         avg_enc_overhead,
         comm_time, _) = run_ring_round(nodes, global_model)

        # Evaluate on global test set
        eval_loss, eval_acc = compute_accuracy(global_model, test_loader, device)

        # Communication: each node sends to next, plus ring overhead
        comm_bytes = model_size_bytes(global_model) * args.clients * 2
        fc2_bytes = (
            global_model.fc2.weight.numel() +
            global_model.fc2.bias.numel()
        ) * 4 * 2 * args.clients
        comm_bytes += fc2_bytes

        metrics = profiler.stop(
            round_num=rnd,
            phase="ring",
            client_id=-1,
            train_loss=avg_loss,
            train_acc=avg_acc,
            eval_loss=eval_loss,
            eval_acc=eval_acc,
            comm_bytes=comm_bytes,
            enc_overhead_s=avg_enc_overhead,
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(global_model.state_dict(), "evaluation/logs/best_ring.pth")

        print(f"{'.'*args.clients} loss={metrics.train_loss:.4f} | "
              f"test_acc={eval_acc*100:.2f}% | "
              f"enc={metrics.enc_overhead_s:.2f}s | "
              f"comm={comm_time*1000:.0f}ms | "
              f"time={metrics.wall_time_s:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Ring 3 complete.")
    print(f"  Best test accuracy    : {best_acc*100:.2f}%")
    print(f"  Metrics log           : {args.log}")
    print(f"  Best model weights    : evaluation/logs/best_ring.pth")
    print(f"{'='*65}")
    print(f"\n  [OK]  All three rings completed!")
    print(f"  -> Open dashboard/index.html for Phase 4 visualisation.\n")


if __name__ == "__main__":
    main()
