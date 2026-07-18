"""
check_backdoor.py — verify whether a backdoor attack actually planted
a working hidden trigger in the committed global model.

Run this AFTER a ring run where config.json had:
    "attack_simulation": {"enabled": true, "type": "backdoor", ...}

It loads dashboard/model_latest.pt (the final committed model) and checks:
  1. Normal accuracy on clean test data (sanity check — model isn't just broken)
  2. Accuracy on the SAME test data with the trigger stamped on it

If clean accuracy is reasonable but trigger-stamped samples get pushed to
the target class far more often than they should, the backdoor is REAL —
a working hidden exploit is sitting inside the shared model.
If there's no meaningful jump, the earlier accuracy drop you saw during
training was just general noise/damage, not a working backdoor.

Usage:
    python check_backdoor.py
    python check_backdoor.py --target-class 1 --trigger-value 99.0
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from models.diabetes_net import DiabetesNet
from data.diabetes_loader import load_diabetes_datasets

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-class", type=int, default=1,
                         help="backdoor_target_class used in config.json (default 1)")
    parser.add_argument("--trigger-value", type=float, default=99.0,
                         help="BACKDOOR_TRIGGER_VALUE from node.py (default 99.0)")
    parser.add_argument("--checkpoint", type=str,
                         default=str(ROOT / "dashboard" / "model_latest.pt"))
    parser.add_argument("--num-clients", type=int, default=3,
                         help="Number of hospital nodes in the ring this "
                              "checkpoint came from — MUST match the actual "
                              "training run (e.g. 4 for your 4-node tests), "
                              "or the test-set partition won't match what "
                              "the ring actually evaluated against.")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Same seed/params as node.py's runtime data load, so this is the
    # exact same held-out test set the ring evaluated against. num_clients
    # MUST match how many nodes actually ran the training ring being
    # checked — hardcoding this caused a silent partition mismatch when
    # checking a 4-node run against a 3-node test split (confirmed: the
    # script's own hospital-partition printout showed 3 hospitals with
    # different patient counts than the real 4-node training logs).
    _, test_loader, _, _ = load_diabetes_datasets(
        num_clients=args.num_clients, alpha=0.5, seed=42
    )

    model = DiabetesNet(input_dim=8, num_classes=2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    X_all, y_all = [], []
    for X_batch, y_batch in test_loader:
        X_all.append(X_batch)
        y_all.append(y_batch)
    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)

    # ── 1. Clean accuracy (sanity check) ─────────────────────────
    with torch.no_grad():
        clean_preds = model(X_all).argmax(dim=1)
    clean_acc = (clean_preds == y_all).float().mean().item()

    # ── 2. Same samples, trigger stamped on last feature ─────────
    X_triggered = X_all.clone()
    X_triggered[:, -1] = args.trigger_value
    with torch.no_grad():
        trig_preds = model(X_triggered).argmax(dim=1)

    target = args.target_class
    trigger_success_rate = (trig_preds == target).float().mean().item()

    # Baseline: how often would target class come up WITHOUT the trigger,
    # on the same samples — this is the natural rate to compare against.
    natural_target_rate = (clean_preds == target).float().mean().item()

    # Only look at samples whose TRUE label isn't already the target —
    # these are the ones a working backdoor actually has to flip.
    non_target_mask = (y_all != target)
    flip_rate = (trig_preds[non_target_mask] == target).float().mean().item() \
        if non_target_mask.any() else float("nan")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(y_all)}")
    print(f"Target class: {target} | Trigger value: {args.trigger_value}")
    print()
    print(f"Clean accuracy (no trigger):            {clean_acc:.4f}")
    print(f"Natural rate of predicting target class: {natural_target_rate:.4f}")
    print(f"Rate predicting target class WITH trigger: {trigger_success_rate:.4f}")
    print(f"Flip rate on non-target-labeled samples:   {flip_rate:.4f}")
    print()

    jump = trigger_success_rate - natural_target_rate
    if flip_rate > 0.7 and jump > 0.3:
        print("VERDICT: Backdoor is REAL and working — trigger reliably "
              "forces the target class, well above baseline.")
    elif flip_rate > 0.4 and jump > 0.15:
        print("VERDICT: Backdoor shows PARTIAL effect — some trigger "
              "sensitivity, but not fully reliable.")
    else:
        print("VERDICT: No meaningful backdoor effect detected — the "
              "trigger does not reliably change predictions. Earlier "
              "accuracy drop during training was likely general noise, "
              "not a working planted exploit.")


if __name__ == "__main__":
    main()
