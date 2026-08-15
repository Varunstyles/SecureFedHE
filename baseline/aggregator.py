"""
baseline/aggregator.py
Federated Averaging (FedAvg) — McMahan et al. 2017.
This is the server-side aggregation step for Ring 1 (vanilla FL baseline).
In Ring 3 this gets replaced by the ring-topology peer aggregation.
"""

from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn


def fedavg(
    global_model: nn.Module,
    client_updates: List[Dict[str, np.ndarray]],
    client_sizes: List[int],
) -> nn.Module:
    """
    Weighted average of client state dicts by dataset size.

    Args:
        global_model:    The server's current model (mutated in-place).
        client_updates:  List of {param_name: numpy_array} from each client.
        client_sizes:    Number of training samples each client used.

    Returns:
        Updated global model (same object, modified in-place).
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    # Build the averaged state dict. Skip BatchNorm's non-trainable
    # bookkeeping buffers (running_mean, running_var, num_batches_tracked)
    # from weighted-averaging — these are not model weights, are not
    # meant to be federated the same way trainable parameters are, and
    # num_batches_tracked is natively int64 (averaging it as float and
    # casting back corrupts BN's internal momentum counter). Keep the
    # global model's own current buffer values instead.
    ref_state = global_model.state_dict()
    avg_state: Dict[str, torch.Tensor] = {}
    for key in client_updates[0]:
        if key.endswith(("num_batches_tracked", "running_mean", "running_var")):
            continue
        stacked = np.stack([u[key] for u in client_updates], axis=0)      # [C, ...]
        w_array = np.array(weights).reshape([-1] + [1] * (stacked.ndim - 1))
        avg_state[key] = torch.tensor((stacked * w_array).sum(axis=0),
                                      dtype=ref_state[key].dtype)

    global_model.load_state_dict(avg_state, strict=False)
    return global_model


def get_model_params(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract full model state as numpy arrays (for client → server transmission)."""
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}


def set_model_params(model: nn.Module, params: Dict[str, np.ndarray]) -> nn.Module:
    """Load numpy state dict back into a model (server → client broadcast).

    IMPORTANT: BatchNorm's `num_batches_tracked` buffer is natively an
    int64 tensor in PyTorch's state_dict(), used internally as a running
    counter for BN's momentum-averaging formula. Casting it to float32
    unconditionally (as this function previously did for every key) is
    silently accepted by load_state_dict — PyTorch does not error on the
    dtype mismatch here — but corrupts that counter every round, which
    compounds over training and produces exactly the kind of late-round
    accuracy collapse independent of any DP-noise or clipping value.
    Every key's ORIGINAL dtype, not a hardcoded float32, must be preserved.
    """
    ref_state = model.state_dict()
    state = {}
    for k, v in params.items():
        target_dtype = ref_state[k].dtype if k in ref_state else torch.float32
        state[k] = torch.tensor(v, dtype=target_dtype)
    model.load_state_dict(state, strict=True)
    return model
