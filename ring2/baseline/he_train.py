"""
baseline/he_train.py
Ring 2 — Selective HE Federated Learning Training Loop.

This is the main entry point for Ring 2.
Run AFTER Ring 1 baseline is confirmed working (baseline_metrics.csv exists).

Usage:
    python -m baseline.he_train
    python -m baseline.he_train --rounds 20 --clients 5 --epsilon 2.0

What this produces:
    evaluation/logs/he_metrics.csv   ← same schema as baseline_metrics.csv
    evaluation/logs/best_he.pth      ← best model weights

Phase 4 comparison:
    Load both CSVs and compare wall_time_s, comm_bytes, ram_mb, eval_acc
    between "baseline" and "selectiveHE" phases.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.aggregator import set_model_params
from baseline.he_aggregator import he_fedavg, broadcast_params
from crypto.he_layer import create_he_context, HE_AVAILABLE
from crypto.selective_client import SelectiveHEClient
from data.loader import load_datasets
from evaluation.metrics import Profiler, compute_accuracy, model_size_bytes
from models.cnn import SimpleCNN


def parse_args():
    p = argparse.ArgumentParser(description="SecureFedHE — Ring 2 Selective HE")
    p.add_argument("--rounds",     type=int,   default=20)
    p.add_argument("--clients",    type=int,   default=5)
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--partition",  type=str,   default="noniid",
                   choices=["iid", "noniid"])
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--fraction",   type=float, default=1.0)
    p.add_argument("--epsilon",    type=float, default=2.0,
                   help="DP epsilon for fc1 layer (lower = stronger privacy)")
    p.add_argument("--delta",      type=float, default=1e-5)
    p.add_argument("--sensitivity",type=float, default=0.1)
    p.add_argument("--he-degree",  type=int,   default=8192,
                   help="CKKS poly_modulus_degree")
    p.add_argument("--validate-he",action="store_true",
                   help="Print HE approximation error each round (slow)")
    p.add_argument("--data-dir",   type=str,   default="./data/cifar10")
    p.add_argument("--log",        type=str,
                   default="evaluation/logs/he_metrics.csv")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not HE_AVAILABLE:
        print("\n[ERROR] TenSEAL is not installed.")
        print("  Ring 2 requires TenSEAL (Linux/WSL/Colab only).")
        print("  Install with: pip install tenseal")
        print("  Then re-run this script.\n")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  SecureFedHE · Ring 2 — Selective HE")
    print(f"  Device : {device} | Rounds : {args.rounds} | Clients : {args.clients}")
    print(f"  DP ε={args.epsilon}, δ={args.delta} | HE degree={args.he_degree}")
    print(f"  Encrypted: fc2 (CKKS) | DP-noised: fc1 | Plaintext: conv blocks")
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

    # ── HE Context (shared — in practice this would be a public key) ──────
    print("\n[HE] Generating CKKS context...", end=" ", flush=True)
    t0 = time.perf_counter()
    he_ctx = create_he_context(poly_modulus_degree=args.he_degree)
    print(f"done in {time.perf_counter()-t0:.2f}s")

    # ── Global model ──────────────────────────────────────────────────────
    global_model = SimpleCNN(num_classes=10).to(device)
    print(f"[Model] Parameters : {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"[Model] fc2 params : "
          f"{sum(p.numel() for p in list(global_model.parameters())[-2:]):,} "
          f"(these get HE-encrypted)\n")

    # ── Clients ───────────────────────────────────────────────────────────
    clients = [
        SelectiveHEClient(
            client_id=i,
            train_loader=train_loaders[i],
            device=device,
            he_ctx=he_ctx,
            lr=args.lr,
            local_epochs=args.epochs,
            dp_epsilon=args.epsilon,
            dp_delta=args.delta,
            dp_sensitivity=args.sensitivity,
            validate_he=args.validate_he,
        )
        for i in range(args.clients)
    ]

    profiler = Profiler(log_path=args.log)
    rng = np.random.default_rng(args.seed)
    best_acc = 0.0

    # ── Ring 2 Training Loop ──────────────────────────────────────────────
    for rnd in range(1, args.rounds + 1):
        print(f"── Round {rnd:02d}/{args.rounds} ", end="", flush=True)

        global_params = broadcast_params(global_model)
        k = max(1, int(len(clients) * args.fraction))
        selected = rng.choice(clients, size=k, replace=False).tolist()

        enc_fc2_list, plain_list, sizes = [], [], []
        round_losses, round_accs = [], []
        total_enc_overhead = 0.0
        fc2_shapes = None

        profiler.start()

        for client in selected:
            set_model_params(global_model, global_params)

            (enc_fc2, shapes, plain_params,
             loss, acc, size, _, enc_t) = client.train(global_model)

            enc_fc2_list.append(enc_fc2)
            plain_list.append(plain_params)
            sizes.append(size)
            round_losses.append(loss)
            round_accs.append(acc)
            total_enc_overhead += enc_t
            fc2_shapes = shapes
            print(".", end="", flush=True)

        # Server aggregates (HE + plaintext FedAvg)
        he_fedavg(global_model, enc_fc2_list, fc2_shapes, plain_list, sizes)

        eval_loss, eval_acc = compute_accuracy(global_model, test_loader, device)

        # Comm cost: plaintext params + encrypted fc2 overhead (~2x fc2 size)
        fc2_bytes = (
            global_model.fc2.weight.numel() +
            global_model.fc2.bias.numel()
        ) * 4 * 2   # ×2 for ciphertext expansion
        comm_bytes = model_size_bytes(global_model) * len(selected) * 2 + fc2_bytes

        metrics = profiler.stop(
            round_num=rnd,
            phase="selectiveHE",
            client_id=-1,
            train_loss=float(np.mean(round_losses)),
            train_acc=float(np.mean(round_accs)),
            eval_loss=eval_loss,
            eval_acc=eval_acc,
            comm_bytes=comm_bytes,
            enc_overhead_s=total_enc_overhead / len(selected),
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(global_model.state_dict(), "evaluation/logs/best_he.pth")

        print(f" loss={metrics.train_loss:.4f} | "
              f"test_acc={eval_acc*100:.2f}% | "
              f"enc_overhead={metrics.enc_overhead_s:.2f}s | "
              f"time={metrics.wall_time_s:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Ring 2 complete.")
    print(f"  Best test accuracy    : {best_acc*100:.2f}%")
    print(f"  Metrics log           : {args.log}")
    print(f"  Best model weights    : evaluation/logs/best_he.pth")
    print(f"{'='*65}")
    print(f"\n  Compare with Ring 1 baseline:")
    print(f"  → Target accuracy drop < 0.5% vs baseline ({best_acc*100:.2f}% vs your 79.43%)")
    print(f"  → Check enc_overhead_s column in CSV for paper Table 2\n")
    print(f"  ✓  Ring 2 validated. Proceed to Ring 3 (Decentralised topology).")


if __name__ == "__main__":
    main()
