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

    # Build the averaged state dict
    avg_state: Dict[str, torch.Tensor] = {}
    for key in client_updates[0]:
        stacked = np.stack([u[key] for u in client_updates], axis=0)      # [C, ...]
        w_array = np.array(weights).reshape([-1] + [1] * (stacked.ndim - 1))
        avg_state[key] = torch.tensor((stacked * w_array).sum(axis=0),
                                      dtype=torch.float32)

    global_model.load_state_dict(avg_state, strict=True)
    return global_model


def get_model_params(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract full model state as numpy arrays (for client → server transmission)."""
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}


def set_model_params(model: nn.Module, params: Dict[str, np.ndarray]) -> nn.Module:
    """Load numpy state dict back into a model (server → client broadcast)."""
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()}
    model.load_state_dict(state, strict=True)
    return model
