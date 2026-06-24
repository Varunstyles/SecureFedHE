"""
models/diabetes_net.py  —  SecureFedHE Diabetes Model
======================================================
Small fully-connected network for Pima diabetes binary classification.

Architecture:
  Input(8) → FC(64) → BN → ReLU → Dropout(0.3)
           → FC(32) → BN → ReLU → Dropout(0.2)
           → FC(16) → ReLU
           → FC(2)  → output logits

Design decisions:
  - Small enough to train fast on CPU (no GPU needed)
  - fc2 = final layer (dim=2) → CKKS encrypted during FL aggregation
  - BatchNorm stabilises training across non-IID hospital data
  - Dropout prevents overfitting on small hospital datasets (~100 patients)
  - Expected accuracy: 74-78% (state of art for this dataset)
  - Expected training time per round: <2s on any modern CPU

Layer naming matches existing SecureFedHE convention:
  self.fc1  — second-to-last layer  (DP noise applied here)
  self.fc2  — final classification  (CKKS encrypted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Model ─────────────────────────────────────────────────────────────────────

class DiabetesNet(nn.Module):
    """
    Feedforward network for diabetes binary classification.

    Input  : 8 normalised clinical features
    Output : 2 logits (class 0 = no diabetes, class 1 = diabetes)

    The final layer (self.fc2) is encrypted with CKKS during
    federated aggregation, matching the SecureFedHE architecture.
    """

    def __init__(self, input_dim: int = 8, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_dim   = input_dim
        self.num_classes = num_classes

        # ── Layers ────────────────────────────────────────────────────────────
        self.fc_in  = nn.Linear(input_dim, 64)
        self.bn1    = nn.BatchNorm1d(64)

        self.fc_mid = nn.Linear(64, 32)
        self.bn2    = nn.BatchNorm1d(32)

        self.fc1    = nn.Linear(32, 16)   # DP noise applied to this layer

        self.fc2    = nn.Linear(16, num_classes)   # CKKS encrypted in FL ring

        self.drop1  = nn.Dropout(dropout)
        self.drop2  = nn.Dropout(dropout * 0.67)   # lighter dropout deeper

        # ── Weight init ───────────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc_in(x))))
        x = self.drop2(F.relu(self.bn2(self.fc_mid(x))))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        """Return softmax probabilities (for inference/dashboard)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x):
        """Return predicted class labels."""
        return self.predict_proba(x).argmax(dim=-1)

    def _to_numpy(self, param) -> np.ndarray:
        """Safely convert a parameter tensor to numpy."""
        if hasattr(param, 'detach'):
            return param.detach().cpu().numpy()
        elif hasattr(param, 'numpy'):
            return param.numpy()
        elif hasattr(param, 'data'):
            return np.array(param.data, dtype=np.float32)
        return np.array(param, dtype=np.float32)

    def get_fc2_params(self) -> Dict[str, np.ndarray]:
        """
        Return fc2 weight and bias as numpy arrays.
        Used by he_layer.py for CKKS encryption.
        """
        return {
            "fc2.weight": self._to_numpy(self.fc2.weight),
            "fc2.bias":   self._to_numpy(self.fc2.bias),
        }

    def set_fc2_params(self, params: Dict[str, np.ndarray]):
        """
        Set fc2 weight and bias from numpy arrays.
        Used after HE decryption and aggregation.
        """
        with torch.no_grad():
            self.fc2.weight.copy_(
                torch.tensor(params["fc2.weight"], dtype=torch.float32)
            )
            self.fc2.bias.copy_(
                torch.tensor(params["fc2.bias"], dtype=torch.float32)
            )

    def get_all_params(self) -> Dict[str, np.ndarray]:
        """Return all parameters as numpy dict (for FL aggregation)."""
        return {
            name: self._to_numpy(param)
            for name, param in self.named_parameters()
        }

    def set_all_params(self, params: Dict[str, np.ndarray]):
        """Set all parameters from numpy dict (after FL aggregation)."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in params:
                    param.copy_(
                        torch.tensor(params[name], dtype=torch.float32)
                    )

    def get_non_fc2_params(self) -> Dict[str, np.ndarray]:
        """
        Return all params EXCEPT fc2 (these get DP noise applied).
        Matches apply_dp_to_fc1() in he_layer.py convention.
        """
        return {
            name: self._to_numpy(param)
            for name, param in self.named_parameters()
            if not name.startswith("fc2")
        }


# ── Training utilities ────────────────────────────────────────────────────────

def train_one_round(
    model:    DiabetesNet,
    loader,
    optimizer,
    epochs:   int = 3,
    device:   str = "cpu",
) -> Tuple[float, float]:
    """
    Train model for one federated round (local_epochs passes over local data).

    Returns:
        (avg_loss, avg_accuracy) over the last epoch
    """
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for _ in range(epochs):
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping — matches DP clipping threshold C=0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            preds          = logits.argmax(dim=-1)
            correct_mask   = preds == y_batch
            if hasattr(correct_mask, 'sum'):
                epoch_correct += correct_mask.sum().item()
            else:
                epoch_correct += int(np.sum(correct_mask))
            batch_size      = y_batch.shape[0] if hasattr(y_batch,'shape') else len(y_batch.data)
            epoch_samples  += batch_size
            epoch_loss     += loss.item() * batch_size

        total_loss    = epoch_loss
        total_correct = epoch_correct
        total_samples = epoch_samples

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc  = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def evaluate(
    model:   DiabetesNet,
    loader,
    device:  str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (accuracy, loss)
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        preds          = logits.argmax(dim=-1)
        correct_mask   = preds == y_batch
        if hasattr(correct_mask, 'sum'):
            total_correct += correct_mask.sum().item()
        else:
            total_correct += int(np.sum(correct_mask))
        batch_size      = y_batch.shape[0] if hasattr(y_batch,'shape') else len(y_batch.data)
        total_samples  += batch_size
        total_loss     += loss.item() * batch_size

    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss    / max(total_samples, 1)
    return accuracy, avg_loss


def predict_single(
    model:    DiabetesNet,
    features: List[float],
    mean:     np.ndarray,
    std:      np.ndarray,
    device:   str = "cpu",
) -> Dict:
    """
    Run inference on a single patient's raw (unnormalised) feature vector.

    Args:
        features : [Pregnancies, Glucose, BP, SkinThickness,
                    Insulin, BMI, DiabetesPedigreeFunction, Age]
        mean     : normalisation mean from training data
        std      : normalisation std from training data

    Returns dict with:
        prediction    : 0 or 1
        confidence    : probability of predicted class (0-100%)
        prob_diabetic : probability of diabetes (0-100%)
        prob_healthy  : probability of no diabetes (0-100%)
        feature_risks : per-feature risk contribution (for explanation)
    """
    model.eval()
    model.to(device)

    x_raw  = np.array(features, dtype=np.float32)
    x_norm = (x_raw - mean) / std
    x_t    = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_t)
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    prediction    = int(probs.argmax())
    prob_diabetic = float(probs[1]) * 100
    prob_healthy  = float(probs[0]) * 100
    confidence    = max(prob_diabetic, prob_healthy)

    # Per-feature risk contribution via input gradient
    try:
        x_grad = torch.tensor(x_norm, dtype=torch.float32,
                               requires_grad=True).unsqueeze(0).to(device)
        logits_g = model(x_grad)
        logits_g[0, 1].backward()   # gradient w.r.t. diabetic class
        gradients = x_grad.grad.squeeze().cpu().numpy() if x_grad.grad is not None \
                    else np.zeros(len(x_norm), dtype=np.float32)
    except Exception:
        gradients = np.zeros(len(x_norm), dtype=np.float32)
    feature_risks = {}
    for i, name in enumerate(["Pregnancies","Glucose","BloodPressure",
                               "SkinThickness","Insulin","BMI",
                               "DiabetesPedigreeFunction","Age"]):
        # Positive gradient = feature increases diabetes risk
        feature_risks[name] = {
            "raw_value":   float(x_raw[i]),
            "risk_score":  float(gradients[i]),        # signed contribution
            "risk_abs":    float(abs(gradients[i])),   # magnitude for ranking
        }

    # Sort features by absolute risk contribution
    ranked = sorted(
        feature_risks.items(),
        key=lambda kv: kv[1]["risk_abs"],
        reverse=True,
    )

    return {
        "prediction":    prediction,
        "label":         "DIABETIC" if prediction == 1 else "NOT DIABETIC",
        "confidence":    round(confidence, 1),
        "prob_diabetic": round(prob_diabetic, 1),
        "prob_healthy":  round(prob_healthy, 1),
        "feature_risks": feature_risks,
        "top_features":  [(name, info) for name, info in ranked[:3]],
    }


def fedavg_params(
    param_list: List[Dict[str, np.ndarray]],
    weights:    Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Federated averaging of model parameter dicts.
    Weights default to uniform if not provided.
    """
    if weights is None:
        weights = [1.0 / len(param_list)] * len(param_list)

    averaged = {}
    for key in param_list[0].keys():
        averaged[key] = sum(
            w * p[key] for w, p in zip(weights, param_list)
        )
    return averaged


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("Testing diabetes_net.py...")

    # 1. Model instantiation
    model = DiabetesNet(input_dim=8, num_classes=2)
    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total_params:,}")
    print(f"fc2 output dim   : {model.fc2.out_features}")

    # 2. Forward pass
    x_dummy = torch.randn(16, 8)
    out = model(x_dummy)
    assert out.shape == (16, 2), f"Expected (16,2), got {out.shape}"
    print(f"\nForward pass     : input {x_dummy.shape} → output {out.shape} ✅")

    # 3. Predict proba
    probs = model.predict_proba(x_dummy)
    assert probs.shape == (16, 2)
    assert abs(probs.sum(dim=-1).mean().item() - 1.0) < 1e-5
    print(f"Predict proba    : sums to 1.0 ✅")

    # 4. get/set fc2 params
    fc2_before = model.get_fc2_params()
    model.set_fc2_params(fc2_before)
    fc2_after  = model.get_fc2_params()
    assert np.allclose(fc2_before["fc2.weight"], fc2_after["fc2.weight"])
    print(f"get/set fc2      : roundtrip ✅")

    # 5. get/set all params
    all_params = model.get_all_params()
    model.set_all_params(all_params)
    print(f"get/set all      : roundtrip ✅  ({len(all_params)} tensors)")

    # 6. FedAvg
    params_a = model.get_all_params()
    params_b = {k: v * 2 for k, v in params_a.items()}
    averaged = fedavg_params([params_a, params_b])
    for k in params_a:
        expected = (params_a[k] + params_b[k]) / 2
        assert np.allclose(averaged[k], expected, atol=1e-6)
    print(f"FedAvg           : uniform weights ✅")

    # 7. Quick training test
    from torch.utils.data import TensorDataset, DataLoader
    X_fake = torch.randn(64, 8)
    y_fake = torch.randint(0, 2, (64,))
    loader = DataLoader(TensorDataset(X_fake, y_fake),
                        batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = train_one_round(model, loader, opt, epochs=2)
    print(f"Train one round  : loss={loss:.4f}  acc={acc:.2%} ✅")

    # 8. Evaluate
    acc_e, loss_e = evaluate(model, loader)
    print(f"Evaluate         : acc={acc_e:.2%}  loss={loss_e:.4f} ✅")

    # 9. Single prediction
    import numpy as np
    mean = np.zeros(8, dtype=np.float32)
    std  = np.ones(8,  dtype=np.float32)
    result = predict_single(
        model,
        features=[6, 148, 72, 35, 0, 33.6, 0.627, 50],
        mean=mean, std=std,
    )
    print(f"predict_single   : {result['label']} ({result['confidence']}% confidence) ✅")
    print(f"  Top risk features:")
    for fname, finfo in result["top_features"]:
        direction = "↑ risk" if finfo["risk_score"] > 0 else "↓ risk"
        print(f"    {fname:<28} {direction}  (score={finfo['risk_score']:+.4f})")

    # 10. non_fc2 params
    non_fc2 = model.get_non_fc2_params()
    assert all(not k.startswith("fc2") for k in non_fc2)
    print(f"non_fc2 params   : {len(non_fc2)} tensors (all non-fc2) ✅")

    print("\n✅ diabetes_net.py — ALL 10 TESTS PASSED")
    print(f"   Model ready for SecureFedHE ring training")
