"""
crypto/selective_client.py
Ring 2 — SelectiveHEClient.

Extends the Ring 1 FederatedClient with selective encryption:
  - fc2 weights → CKKS HE encrypted before transmission
  - fc1 weights → Differential Privacy noise added
  - conv blocks → plaintext (unchanged from Ring 1)

The server never sees plaintext fc2 gradients.
The server aggregates enc_fc2 directly using HE addition.
"""

import copy
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.aggregator import get_model_params
from crypto.he_layer import (
    encrypt_fc2,
    apply_dp_to_fc1,
    measure_he_error,
)


class SelectiveHEClient:
    """
    Edge device client with selective homomorphic encryption.

    Args:
        client_id:     Unique identifier.
        train_loader:  This client's private data shard.
        device:        torch device.
        he_ctx:        Shared TenSEAL CKKS context.
        lr:            SGD learning rate.
        local_epochs:  Local training epochs per FL round.
        dp_epsilon:    DP privacy budget for fc1 layer.
        dp_delta:      DP delta parameter.
        dp_sensitivity:Gradient clipping norm.
        validate_he:   If True, measure HE approximation error each round
                       (adds ~50ms overhead; disable after initial validation).
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        device: torch.device,
        he_ctx,
        lr: float = 0.01,
        local_epochs: int = 2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        dp_epsilon: float = 2.0,
        dp_delta: float = 1e-5,
        dp_sensitivity: float = 0.1,
        validate_he: bool = False,
    ):
        self.id = client_id
        self.loader = train_loader
        self.device = device
        self.ctx = he_ctx
        self.lr = lr
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_sensitivity = dp_sensitivity
        self.validate_he = validate_he
        self.dataset_size = len(train_loader.dataset)

    def train(
        self,
        global_model: nn.Module,
    ) -> Tuple[Dict, Dict, Dict[str, np.ndarray], float, float, int, float, float]:
        """
        Full Ring 2 training step.

        Returns:
            enc_fc2:         Encrypted fc2 params (CKKSVector dict)
            fc2_shapes:      Original fc2 shapes for decryption
            plain_params:    All non-fc2 params (conv + dp-noised fc1)
            train_loss:      Mean training loss
            train_acc:       Mean training accuracy
            dataset_size:    For weighted aggregation
            train_time_s:    Wall-clock training time
            enc_overhead_s:  Extra time spent on encryption
        """
        # ── 1. Local training (identical to Ring 1) ────────────────────────
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        epoch_losses, epoch_accs = [], []
        t_train_start = time.perf_counter()

        for _ in range(self.local_epochs):
            running_loss = correct = total = 0
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = local_model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            epoch_losses.append(running_loss / total)
            epoch_accs.append(correct / total)

        train_time = time.perf_counter() - t_train_start

        # ── 2. Extract full model params ───────────────────────────────────
        all_params = get_model_params(local_model)

        # ── 3. Apply DP noise to fc1 (intermediate layer) ─────────────────
        all_params = apply_dp_to_fc1(
            all_params,
            epsilon=self.dp_epsilon,
            delta=self.dp_delta,
            sensitivity=self.dp_sensitivity,
        )

        # ── 4. Encrypt fc2 (classifier layer) — CKKS HE ───────────────────
        t_enc_start = time.perf_counter()

        if self.validate_he:
            err = measure_he_error(all_params["fc2.weight"], self.ctx)
            print(f"  [Client {self.id}] HE approx error: "
                  f"max={err['max_abs_error']:.2e}, "
                  f"mean={err['mean_abs_error']:.2e}")

        enc_fc2, fc2_shapes = encrypt_fc2(all_params, self.ctx)
        enc_overhead = time.perf_counter() - t_enc_start

        # ── 5. Build plaintext param dict (everything except fc2) ──────────
        plain_params = {
            k: v for k, v in all_params.items()
            if k not in ("fc2.weight", "fc2.bias")
        }

        return (
            enc_fc2,
            fc2_shapes,
            plain_params,
            float(np.mean(epoch_losses)),
            float(np.mean(epoch_accs)),
            self.dataset_size,
            train_time,
            enc_overhead,
        )
