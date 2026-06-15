"""
network/ring_topology.py
Ring 3 — Decentralised ring aggregation (no central server).

Architecture:
  Nodes: [0] → [1] → [2] → ... → [N-1] → [0]
  Each node trains locally, encrypts its fc2 update,
  and passes a running HE aggregate to its successor.
  After one full ring pass, every node holds the aggregate.
  No central server ever sees plaintext gradients.

This achieves FULL decentralisation:
  - No single point of failure
  - No central server that could be compromised
  - Each node only communicates with its immediate neighbour
"""

import copy
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.aggregator import get_model_params, set_model_params
from crypto.he_layer import (
    encrypt_fc2,
    decrypt_layer,
    apply_dp_to_fc1,
    aggregate_encrypted_fc2,
)


# ── Latency simulation ────────────────────────────────────────────────────────

def simulate_transmission_latency(
    mean_ms: float = 15.0,
    std_ms: float = 5.0,
) -> float:
    """
    Sample a realistic edge-to-edge transmission delay.
    Mean 15ms ± 5ms simulates a LAN/WiFi edge network.
    Returns elapsed time in seconds.
    """
    delay_s = max(0.001, np.random.normal(mean_ms, std_ms) / 1000)
    time.sleep(delay_s)
    return delay_s


# ── Ring Node ─────────────────────────────────────────────────────────────────

class RingNode:
    """
    Single node in the decentralised ring.
    Holds a local model and communicates with its successor only.

    Each node:
      1. Trains locally on its private data
      2. Encrypts fc2 with CKKS HE
      3. Applies DP noise to fc1
      4. Passes encrypted update to next node in ring
    """

    def __init__(
        self,
        node_id: int,
        train_loader: DataLoader,
        device: torch.device,
        he_ctx,
        lr: float = 0.01,
        local_epochs: int = 2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        dp_epsilon: float = 10.0,
        dp_delta: float = 1e-5,
        dp_sensitivity: float = 0.1,
    ):
        self.id = node_id
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
        self.dataset_size = len(train_loader.dataset)

        self.successor: Optional["RingNode"] = None
        self.model: Optional[nn.Module] = None

    def set_successor(self, node: "RingNode"):
        self.successor = node

    def set_model(self, model: nn.Module):
        """Set this node's local model (deep copy of global)."""
        self.model = copy.deepcopy(model)

    def train_locally(self) -> Tuple[Dict[str, np.ndarray], float, float, float]:
        """
        Train local model on private data.

        Returns:
            all_params:   Full model params as numpy
            train_loss:   Average training loss
            train_acc:    Average training accuracy
            train_time:   Wall clock training time
        """
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        epoch_losses, epoch_accs = [], []
        t0 = time.perf_counter()

        for _ in range(self.local_epochs):
            running_loss = correct = total = 0
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            epoch_losses.append(running_loss / total)
            epoch_accs.append(correct / total)

        train_time = time.perf_counter() - t0
        all_params = get_model_params(self.model)

        return (
            all_params,
            float(np.mean(epoch_losses)),
            float(np.mean(epoch_accs)),
            train_time,
        )

    def train_and_get_enc_update(
        self,
    ) -> Tuple[Dict, Dict, Dict[str, np.ndarray], float, float, int, float, float]:
        """
        Train locally, apply DP to fc1, encrypt fc2.

        Returns:
            enc_fc2:        Encrypted fc2 dict
            fc2_shapes:     Original shapes for decryption
            plain_params:   Conv + DP-noised fc1 (plaintext)
            train_loss:     Mean loss
            train_acc:      Mean accuracy
            dataset_size:   For weighted aggregation
            train_time:     Training wall time
            enc_overhead:   Encryption wall time
        """
        # 1. Local training
        all_params, loss, acc, train_time = self.train_locally()

        # 2. Apply DP to fc1
        all_params = apply_dp_to_fc1(
            all_params,
            epsilon=self.dp_epsilon,
            delta=self.dp_delta,
            sensitivity=self.dp_sensitivity,
        )

        # 3. Encrypt fc2
        t_enc = time.perf_counter()
        enc_fc2, fc2_shapes = encrypt_fc2(all_params, self.ctx)
        enc_overhead = time.perf_counter() - t_enc

        # 4. Plain params (everything except fc2)
        plain_params = {
            k: v for k, v in all_params.items()
            if k not in ("fc2.weight", "fc2.bias")
        }

        return (
            enc_fc2, fc2_shapes, plain_params,
            loss, acc, self.dataset_size, train_time, enc_overhead,
        )

    def aggregate_and_forward(
        self,
        incoming_enc_fc2: Dict,
        incoming_plain: Dict[str, np.ndarray],
        incoming_weight: float,
        my_enc_fc2: Dict,
        my_plain: Dict[str, np.ndarray],
        my_weight: float,
    ) -> Tuple[Dict, Dict[str, np.ndarray], float]:
        """
        Aggregate incoming running total with own update using HE addition.
        No decryption happens — this is the key security property.

        Returns:
            combined_enc_fc2:   Aggregated encrypted fc2
            combined_plain:     Aggregated plaintext params
            combined_weight:    Sum of weights
        """
        # Simulate peer-to-peer transmission
        simulate_transmission_latency()

        combined_weight = incoming_weight + my_weight

        # HE addition for fc2 (no decryption needed!)
        combined_enc_fc2 = {}
        for key in ["fc2.weight", "fc2.bias"]:
            combined_enc_fc2[key] = incoming_enc_fc2[key] + my_enc_fc2[key]

        # Plaintext weighted addition for conv + fc1
        combined_plain = {}
        for key in incoming_plain:
            combined_plain[key] = incoming_plain[key] + my_plain[key]

        return combined_enc_fc2, combined_plain, combined_weight


# ── Ring orchestration ────────────────────────────────────────────────────────

def build_ring(nodes: List[RingNode]) -> List[RingNode]:
    """Connect nodes in a ring: 0→1→2→...→N-1→0"""
    for i, node in enumerate(nodes):
        node.set_successor(nodes[(i + 1) % len(nodes)])
    return nodes


def run_ring_round(
    nodes: List[RingNode],
    global_model: nn.Module,
) -> Tuple[nn.Module, float, float, float, float, float]:
    """
    Execute one full ring aggregation round.

    Protocol:
      1. All nodes receive current global model
      2. All nodes train locally + encrypt fc2 + DP fc1
      3. Node 0 starts the ring: sends its weighted update to Node 1
      4. Each subsequent node adds its own weighted update to running total
      5. After full pass: the initiator (Node 0) receives final aggregate
      6. Decrypt fc2, normalise, update all nodes

    Returns:
        updated_model:     Aggregated global model
        avg_train_loss:    Mean training loss across nodes
        avg_train_acc:     Mean training accuracy across nodes
        avg_enc_overhead:  Mean encryption time per node
        total_comm_time:   Total ring communication time
        total_train_time:  Total training time
    """
    n = len(nodes)

    # ── Step 1: Distribute global model to all nodes ──────────────────────
    for node in nodes:
        node.set_model(global_model)

    # ── Step 2: All nodes train locally and prepare encrypted updates ─────
    node_updates = []
    train_losses, train_accs = [], []
    enc_overheads = []

    for node in nodes:
        enc_fc2, fc2_shapes, plain_params, loss, acc, size, t_train, t_enc = \
            node.train_and_get_enc_update()

        # Weight the contributions by dataset size for FedAvg
        weight = size
        weighted_plain = {k: v * weight for k, v in plain_params.items()}
        weighted_enc_fc2 = {
            k: v * weight for k, v in enc_fc2.items()
        }

        node_updates.append({
            "enc_fc2": weighted_enc_fc2,
            "fc2_shapes": fc2_shapes,
            "plain": weighted_plain,
            "weight": weight,
        })

        train_losses.append(loss)
        train_accs.append(acc)
        enc_overheads.append(t_enc)

    # ── Step 3: Ring aggregation pass ─────────────────────────────────────
    # Node 0 starts with its own update
    t_comm_start = time.perf_counter()

    running_enc_fc2 = node_updates[0]["enc_fc2"]
    running_plain = node_updates[0]["plain"]
    running_weight = node_updates[0]["weight"]

    # Pass around the ring: 0→1→2→...→(N-1)
    for i in range(1, n):
        current_node = nodes[i]
        my_update = node_updates[i]

        running_enc_fc2, running_plain, running_weight = \
            current_node.aggregate_and_forward(
                incoming_enc_fc2=running_enc_fc2,
                incoming_plain=running_plain,
                incoming_weight=running_weight,
                my_enc_fc2=my_update["enc_fc2"],
                my_plain=my_update["plain"],
                my_weight=my_update["weight"],
            )

    total_comm_time = time.perf_counter() - t_comm_start

    # ── Step 4: Decrypt and normalise ─────────────────────────────────────
    total_weight = running_weight

    # Decrypt fc2 aggregate and normalise
    fc2_shapes = node_updates[0]["fc2_shapes"]
    decrypted_fc2 = {}
    for key in ["fc2.weight", "fc2.bias"]:
        raw = decrypt_layer(running_enc_fc2[key], fc2_shapes[key])
        decrypted_fc2[key] = raw / total_weight

    # Normalise plaintext aggregate
    normalised_plain = {k: v / total_weight for k, v in running_plain.items()}

    # ── Step 5: Assemble and broadcast ────────────────────────────────────
    full_state = {**normalised_plain, **decrypted_fc2}
    state_tensors = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in full_state.items()
    }
    global_model.load_state_dict(state_tensors, strict=True)

    # Update all nodes with the new global model
    for node in nodes:
        node.set_model(global_model)

    total_train_time = sum(enc_overheads) + total_comm_time

    return (
        global_model,
        float(np.mean(train_losses)),
        float(np.mean(train_accs)),
        float(np.mean(enc_overheads)),
        total_comm_time,
        total_train_time,
    )
