"""
baseline/client.py
Simulated edge-device client.
Each client:
  1. Receives the global model from the server.
  2. Trains for E local epochs on its private data partition.
  3. Returns the updated model parameters (gradients stay local — FL core guarantee).
"""

import copy
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.aggregator import get_model_params


class FederatedClient:
    """
    Represents one edge device in the simulated federated network.

    Args:
        client_id:    Unique identifier.
        train_loader: This client's private data partition.
        device:       torch device to train on.
        lr:           Local SGD learning rate.
        local_epochs: How many epochs to run before sending update back.
        momentum:     SGD momentum.
        weight_decay: L2 regularisation.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        local_epochs: int = 2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        self.id = client_id
        self.loader = train_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset_size = len(train_loader.dataset)

    def train(
        self,
        global_model: nn.Module,
    ) -> Tuple[Dict[str, np.ndarray], float, float, int, float]:
        """
        Receive global weights, train locally, return update.

        Returns:
            updated_params:   State dict as numpy arrays.
            avg_train_loss:   Mean cross-entropy loss across local epochs.
            avg_train_acc:    Mean accuracy across local epochs.
            dataset_size:     Used for weighted FedAvg.
            train_time_s:     Wall-clock seconds spent training.
        """
        # Deep-copy so we don't mutate the global model
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
        t0 = time.perf_counter()

        for _ in range(self.local_epochs):
            running_loss = correct = total = 0
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = local_model(x)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(y)
                correct      += (logits.argmax(1) == y).sum().item()
                total        += len(y)
            epoch_losses.append(running_loss / total)
            epoch_accs.append(correct / total)

        train_time = time.perf_counter() - t0
        return (
            get_model_params(local_model),
            float(np.mean(epoch_losses)),
            float(np.mean(epoch_accs)),
            self.dataset_size,
            train_time,
        )
