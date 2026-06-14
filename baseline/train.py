"""
baseline/train.py
Ring 1 — Vanilla Federated Learning Baseline.

Run this first. Do not touch Ring 2 (HE) until this converges cleanly.
Expected result after 20 rounds: ~55-65% test accuracy on CIFAR-10.

Usage:
    python -m baseline.train
    python -m baseline.train --rounds 30 --clients 10 --partition noniid
"""

import argparse
import os
import sys

import torch

# ── Make sure project root is on the path ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.aggregator import fedavg, get_model_params, set_model_params
from baseline.client import FederatedClient
from data.loader import load_datasets
from evaluation.metrics import Profiler, compute_accuracy, model_size_bytes
from models.cnn import SimpleCNN


def parse_args():
    p = argparse.ArgumentParser(description="SecureFedHE — Ring 1 Baseline")
    p.add_argument("--rounds",     type=int,   default=20,      help="FL communication rounds")
    p.add_argument("--clients",    type=int,   default=5,       help="Number of simulated clients")
    p.add_argument("--epochs",     type=int,   default=2,       help="Local epochs per round")
    p.add_argument("--lr",         type=float, default=0.01,    help="Client SGD learning rate")
    p.add_argument("--batch",      type=int,   default=32,      help="Local batch size")
    p.add_argument("--partition",  type=str,   default="noniid",choices=["iid","noniid"])
    p.add_argument("--alpha",      type=float, default=0.5,     help="Dirichlet alpha (non-IID)")
    p.add_argument("--fraction",   type=float, default=1.0,     help="Fraction of clients per round")
    p.add_argument("--data-dir",   type=str,   default="./data/cifar10")
    p.add_argument("--log",        type=str,   default="evaluation/logs/baseline_metrics.csv")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def select_clients(clients, fraction, rng):
    """Randomly select a fraction of clients for each round (simulates dropout)."""
    k = max(1, int(len(clients) * fraction))
    return rng.choice(clients, size=k, replace=False).tolist()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SecureFedHE · Ring 1 — Vanilla FL Baseline")
    print(f"  Device: {device} | Rounds: {args.rounds} | Clients: {args.clients}")
    print(f"  Partition: {args.partition.upper()} | Local epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loaders, test_loader = load_datasets(
        data_dir=args.data_dir,
        num_clients=args.clients,
        partition=args.partition,
        alpha=args.alpha,
        batch_size=args.batch,
        seed=args.seed,
    )

    # ── Global model (server) ─────────────────────────────────────────────────
    global_model = SimpleCNN(num_classes=10).to(device)
    print(f"\n[Model] Parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"[Model] Size per round (transmitted): "
          f"{model_size_bytes(global_model) / 1024:.1f} KB\n")

    # ── Clients ───────────────────────────────────────────────────────────────
    clients = [
        FederatedClient(
            client_id=i,
            train_loader=train_loaders[i],
            device=device,
            lr=args.lr,
            local_epochs=args.epochs,
        )
        for i in range(args.clients)
    ]

    profiler = Profiler(log_path=args.log)
    rng = __import__("numpy").random.default_rng(args.seed)

    best_acc = 0.0

    # -- FL Training Loop ------------------------------------------------------
    for rnd in range(1, args.rounds + 1):
        print(f"-- Round {rnd:02d}/{args.rounds} ", end="", flush=True)

        selected = select_clients(clients, args.fraction, rng)
        global_params = get_model_params(global_model)

        client_updates, client_sizes = [], []
        round_train_losses, round_train_accs = [], []

        profiler.start()

        for client in selected:
            # Broadcast global weights to client
            set_model_params(global_model, global_params)

            params, loss, acc, size, t = client.train(global_model)

            client_updates.append(params)
            client_sizes.append(size)
            round_train_losses.append(loss)
            round_train_accs.append(acc)
            print(".", end="", flush=True)

        # Server aggregates
        fedavg(global_model, client_updates, client_sizes)

        # Server-side evaluation (global test set)
        eval_loss, eval_acc = compute_accuracy(global_model, test_loader, device)

        metrics = profiler.stop(
            round_num=rnd,
            phase="baseline",
            client_id=-1,          # -1 = server/aggregate
            train_loss=sum(round_train_losses) / len(round_train_losses),
            train_acc=sum(round_train_accs) / len(round_train_accs),
            eval_loss=eval_loss,
            eval_acc=eval_acc,
            comm_bytes=model_size_bytes(global_model) * len(selected) * 2,
            enc_overhead_s=0.0,    # baseline has no encryption
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(global_model.state_dict(), "evaluation/logs/best_baseline.pth")

        print(f" loss={metrics.train_loss:.4f} | "
              f"test_acc={eval_acc*100:.2f}% | "
              f"time={metrics.wall_time_s:.1f}s")

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best test accuracy : {best_acc*100:.2f}%")
    print(f"  Metrics log        : {args.log}")
    print(f"  Best model weights : evaluation/logs/best_baseline.pth")
    print(f"{'='*60}\n")
    print("  ✓  Ring 1 baseline is working. Proceed to Ring 2 (Selective HE).")


if __name__ == "__main__":
    main()
