"""
data/loader.py
CIFAR-10 loading + non-IID partitioning across N simulated edge clients.

Non-IID is important: real federated settings have skewed data distributions.
Each client gets data dominated by a small subset of classes (Dirichlet partition).
IID partitioning is also available as a sanity-check baseline.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ── Standard CIFAR-10 transforms ────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])


# ── Partitioning ─────────────────────────────────────────────────────────────

def partition_iid(
    dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """Shuffle and split indices equally — unrealistic but useful for sanity checks."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    size = len(indices) // num_clients
    return [indices[i * size:(i + 1) * size] for i in range(num_clients)]


def partition_noniid_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Dirichlet partition — the standard non-IID split used in FL research.
    Lower alpha → more skewed (each client sees fewer classes).
    alpha=0.5 is typical; alpha=0.1 is very heterogeneous.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        # Fix rounding so we don't lose samples
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()
        splits = np.split(cls_idx, np.cumsum(proportions[:-1]))
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices


# ── Public API ────────────────────────────────────────────────────────────────

def load_datasets(
    data_dir: str = "./data/cifar10",
    num_clients: int = 5,
    partition: str = "noniid",   # "iid" or "noniid"
    alpha: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Returns:
        train_loaders  — one DataLoader per simulated client
        test_loader    — single global test loader (server-side evaluation)

    Set verbose=False to suppress the partition summary (useful for
    non-master nodes in the distributed ring so it only prints once).
    """
    train_dataset = datasets.CIFAR10(data_dir, train=True,
                                     download=True, transform=TRAIN_TRANSFORM)
    test_dataset  = datasets.CIFAR10(data_dir, train=False,
                                     download=True, transform=TEST_TRANSFORM)

    if partition == "iid":
        client_idx = partition_iid(train_dataset, num_clients, seed)
    else:
        client_idx = partition_noniid_dirichlet(train_dataset, num_clients, alpha, seed)

    train_loaders = [
        DataLoader(
            Subset(train_dataset, idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        for idx in client_idx
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Print a quick summary so you can verify the partition in logs
    if verbose:
        print(f"\n[DataLoader] Partition: {partition.upper()}")
        print(f"  Clients: {num_clients} | Batch size: {batch_size} | Alpha: {alpha}")
        for i, idx in enumerate(client_idx):
            client_labels = np.array(train_dataset.targets)[idx]
            unique, counts = np.unique(client_labels, return_counts=True)
            dominant = unique[counts.argmax()]
            print(f"  Client {i:02d}: {len(idx):>4} samples | "
                  f"dominant class: {dominant} ({counts.max()} samples)")

    return train_loaders, test_loader
