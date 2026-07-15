"""
check_label_flip.py — verify whether a label_flip attack measurably
degraded the committed global model, and characterise HOW (unlike
backdoor, label_flip has no fixed trigger to test — the signature is
general per-class accuracy damage, not a deterministic override).

Run this AFTER a ring run where config.json had:
    "attack_simulation": {"enabled": true, "type": "label_flip", ...}

It loads dashboard/model_latest.pt (the final committed model) and reports:
  1. Overall accuracy on the held-out test set
  2. Per-class accuracy (class 0 = non-diabetic, class 1 = diabetic)
  3. Confusion matrix

Compare these numbers against a clean run (no attack_simulation) on the
same test set. label_flip's signature is: overall accuracy noticeably
lower than clean baseline, damage spread across BOTH classes (not one
class always flipping to another like a backdoor would show), and no
single dominant misclassification pattern.

Usage:
    python check_label_flip.py
    python check_label_flip.py --checkpoint path/to/other_model.pt
"""
import argparse
import torch
from pathlib import Path

from models.diabetes_net import DiabetesNet
from data.diabetes_loader import load_diabetes_datasets

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                         default=str(ROOT / "dashboard" / "model_latest.pt"))
    args = parser.parse_args()

    device = torch.device("cpu")

    # Same seed/params as node.py's runtime data load, so this is the
    # exact same held-out test set the ring evaluated against.
    _, test_loader, _, _ = load_diabetes_datasets(
        num_clients=3, alpha=0.5, seed=42
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

    with torch.no_grad():
        preds = model(X_all).argmax(dim=1)

    overall_acc = (preds == y_all).float().mean().item()

    # Per-class accuracy
    results = {}
    for cls in [0, 1]:
        mask = (y_all == cls)
        n = mask.sum().item()
        if n > 0:
            acc = (preds[mask] == cls).float().mean().item()
        else:
            acc = float("nan")
        results[cls] = (acc, n)

    # Confusion matrix
    tp = ((preds == 1) & (y_all == 1)).sum().item()
    tn = ((preds == 0) & (y_all == 0)).sum().item()
    fp = ((preds == 1) & (y_all == 0)).sum().item()
    fn = ((preds == 0) & (y_all == 1)).sum().item()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(y_all)}")
    print()
    print(f"Overall accuracy: {overall_acc:.4f}")
    print()
    print("Per-class accuracy:")
    print(f"  Class 0 (non-diabetic): {results[0][0]:.4f}  (n={results[0][1]})")
    print(f"  Class 1 (diabetic):     {results[1][0]:.4f}  (n={results[1][1]})")
    print()
    print("Confusion matrix:")
    print(f"                 Pred 0   Pred 1")
    print(f"  True 0 (neg)   {tn:>6}   {fp:>6}")
    print(f"  True 1 (pos)   {fn:>6}   {tp:>6}")
    print()

    class_gap = abs(results[0][0] - results[1][0])
    print(f"Class accuracy gap: {class_gap:.4f}")
    print()
    if overall_acc < 0.65:
        print("VERDICT: Overall accuracy is notably degraded — consistent "
              "with label-flip poisoning damage. Compare against your "
              "clean-run baseline to confirm this drop wasn't just normal "
              "run-to-run variance.")
    else:
        print("VERDICT: Overall accuracy is in the normal range for this "
              "task — no strong evidence of poisoning damage from this "
              "metric alone. Compare directly against a clean-run baseline "
              "on the same test set for a reliable comparison.")

    print()
    print("NOTE: Unlike backdoor, label_flip has no fixed trigger to test "
          "directly. This script only shows END-STATE damage. For a full "
          "before/after comparison, run this same script against a "
          "clean-run checkpoint (attack_simulation.enabled=false) and "
          "diff the numbers.")


if __name__ == "__main__":
    main()
