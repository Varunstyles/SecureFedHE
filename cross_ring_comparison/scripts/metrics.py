"""
evaluation/metrics.py
Round-level profiler + CSV logger.
Wire this up from Day 1 — clean logs are your Phase 4 paper data.
"""

import csv
import os
import time
from dataclasses import dataclass, fields, astuple
from typing import Optional

import psutil
import torch


@dataclass
class RoundMetrics:
    round_num:        int
    phase:            str     # "baseline" | "selectiveHE" | "ring"
    client_id:        int
    train_loss:       float
    train_acc:        float
    eval_loss:        float
    eval_acc:         float
    comm_bytes:       int     # bytes transmitted this round (model update size)
    wall_time_s:      float   # wall-clock time for the full round
    cpu_pct:          float   # average CPU % during the round
    ram_mb:           float   # peak RSS increase during the round
    enc_overhead_s:   float   # extra seconds spent in encryption (0 for baseline)


class Profiler:
    """
    Context manager + manual API for measuring a training round.

    Usage (context manager):
        with profiler.round(round_num=1, phase="baseline", client_id=0) as ctx:
            loss, acc = train(...)
            ctx.train_loss = loss
            ctx.train_acc  = acc
            ...

    Usage (manual):
        profiler.start()
        loss, acc = train(...)
        metrics = profiler.stop(round_num, phase, client_id, loss, acc, ...)
    """

    def __init__(self, log_path: str = "evaluation/logs/metrics.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._write_header()
        self._proc = psutil.Process()
        self._t0: Optional[float] = None
        self._mem0: Optional[float] = None

    def _write_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([field.name for field in fields(RoundMetrics)])

    def start(self):
        self._proc.cpu_percent(interval=None)   # prime the CPU sampler
        self._mem0 = self._proc.memory_info().rss / 1e6
        self._t0 = time.perf_counter()

    def stop(
        self,
        round_num: int,
        phase: str,
        client_id: int,
        train_loss: float,
        train_acc: float,
        eval_loss: float,
        eval_acc: float,
        comm_bytes: int,
        enc_overhead_s: float = 0.0,
    ) -> RoundMetrics:
        elapsed    = time.perf_counter() - self._t0
        cpu_pct    = self._proc.cpu_percent(interval=None)
        ram_delta  = (self._proc.memory_info().rss / 1e6) - self._mem0

        m = RoundMetrics(
            round_num=round_num,
            phase=phase,
            client_id=client_id,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4),
            eval_loss=round(eval_loss, 4),
            eval_acc=round(eval_acc, 4),
            comm_bytes=comm_bytes,
            wall_time_s=round(elapsed, 3),
            cpu_pct=round(cpu_pct, 1),
            ram_mb=round(ram_delta, 1),
            enc_overhead_s=round(enc_overhead_s, 3),
        )
        self._append(m)
        return m

    def _append(self, m: RoundMetrics):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(astuple(m))


def model_size_bytes(model: torch.nn.Module) -> int:
    """Estimate the byte cost of transmitting all model parameters (float32)."""
    return sum(p.numel() for p in model.parameters()) * 4  # 4 bytes per float32


def compute_accuracy(model: torch.nn.Module, loader, device: torch.device) -> tuple:
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out   = model(x)
            loss  = criterion(out, y)
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total
