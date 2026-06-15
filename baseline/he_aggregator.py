"""
baseline/he_aggregator.py
Ring 2 — Server-side aggregation with Homomorphic Encryption.

The server:
  1. Receives encrypted fc2 from each client  (never sees plaintext fc2)
  2. Aggregates enc_fc2 using HE addition     (FedAvg on ciphertexts)
  3. Decrypts the aggregated fc2              (only the server decrypts)
  4. Receives plaintext conv + DP-noised fc1  (aggregated normally)
  5. Assembles the full updated global model

Security guarantee:
  No individual client's fc2 gradients are ever visible to the server.
  The server only sees the AGGREGATE after decryption.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from crypto.he_layer import aggregate_encrypted_fc2, decrypt_layer


def he_fedavg(
    global_model: nn.Module,
    enc_fc2_updates: List[Dict],
    fc2_shapes: Dict[str, tuple],
    plain_param_updates: List[Dict[str, np.ndarray]],
    client_sizes: List[int],
) -> nn.Module:
    """
    Selective HE FedAvg aggregation.

    Args:
        global_model:       Server's current model (updated in-place).
        enc_fc2_updates:    List of encrypted fc2 dicts from each client.
        fc2_shapes:         Original shapes for decryption.
        plain_param_updates:List of plaintext param dicts (conv + fc1).
        client_sizes:       Dataset sizes for weighted averaging.

    Returns:
        Updated global model.
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    # ── 1. Aggregate encrypted fc2 (HE addition, no decryption) ──────────
    agg_enc_fc2 = aggregate_encrypted_fc2(enc_fc2_updates, client_sizes)

    # ── 2. Decrypt the aggregate fc2 (server's private key) ──────────────
    decrypted_fc2 = {
        key: decrypt_layer(agg_enc_fc2[key], fc2_shapes[key])
        for key in ["fc2.weight", "fc2.bias"]
    }

    # ── 3. Aggregate plaintext params (conv blocks + dp-noised fc1) ───────
    agg_plain: Dict[str, np.ndarray] = {}
    for key in plain_param_updates[0]:
        stacked = np.stack([u[key] for u in plain_param_updates], axis=0)
        w = np.array(weights).reshape([-1] + [1] * (stacked.ndim - 1))
        agg_plain[key] = (stacked * w).sum(axis=0)

    # ── 4. Assemble full state dict ───────────────────────────────────────
    full_state = {**agg_plain, **decrypted_fc2}
    state_tensors = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in full_state.items()
    }
    global_model.load_state_dict(state_tensors, strict=True)
    return global_model


def broadcast_params(model: nn.Module) -> Dict[str, np.ndarray]:
    """Server broadcasts current global params to all clients."""
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}
