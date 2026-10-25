"""
variable_length.py
==================
[PAPER] The number of active orders changes over time (dynamic orders arrive, orders finish),
but the neural network must never be rebuilt.  Everything is therefore padded to fixed
tensor widths and accompanied by boolean masks.

    M_max : padded order slots      -> Z [M_max, dz], U [M_max, du], D [M_max, dd]
    S_max : padded service-location slots per entity -> service_location_mask [M_max, S_max]

Example (M_max = 8)
    active orders  O1 O2 O3 O4        -> order_mask [1,1,1,1,0,0,0,0]
    +2 new dynamic orders             -> order_mask [1,1,1,1,1,1,0,0]
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

NEG_INF = -1e9


def pad_to(matrix: np.ndarray, n_rows: int, fill: float = 0.0) -> np.ndarray:
    """Pad (or verify) a [n, d] array to [n_rows, d]."""
    n, d = matrix.shape
    if n > n_rows:
        raise ValueError(f"cannot pad: {n} rows > M_max={n_rows}")
    if n == n_rows:
        return matrix
    out = np.full((n_rows, d), fill, dtype=matrix.dtype)
    out[:n] = matrix
    return out


def make_mask(n_active: int, n_slots: int) -> np.ndarray:
    """1 = real slot, 0 = padding."""
    mask = np.zeros(n_slots, dtype=np.float32)
    mask[:max(0, min(n_active, n_slots))] = 1.0
    return mask


def mask_from_indices(indices: Sequence[int], n_slots: int) -> np.ndarray:
    mask = np.zeros(n_slots, dtype=np.float32)
    for i in indices:
        if 0 <= i < n_slots:
            mask[i] = 1.0
    return mask


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor,
                      neg_inf: float = NEG_INF) -> torch.Tensor:
    """[PAPER] invalid orders are masked BEFORE the softmax.

    `mask`: 1 = feasible, 0 = masked.  Fully-masked rows fall back to a uniform
    distribution so that softmax never produces NaNs (the caller must detect the
    empty-mask case separately).
    """
    mask = mask.to(logits.dtype)
    empty = mask.sum(dim=-1, keepdim=True) <= 0
    safe_mask = torch.where(empty.expand_as(mask), torch.ones_like(mask), mask)
    return logits.masked_fill(safe_mask <= 0, neg_inf)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(apply_action_mask(logits, mask), dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence dimension using a [.., L] mask; shape [.., d]."""
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=-2).clamp(min=1.0)
    return (x * m).sum(dim=-2) / denom


def key_padding_mask(order_mask: torch.Tensor) -> torch.Tensor:
    """torch MultiheadAttention wants True where the position must be IGNORED."""
    return order_mask <= 0


class SlotTable:
    """Maps order_id <-> padded tensor slot, with recycling of finished slots."""

    def __init__(self, n_slots: int, overflow_policy: str = "queue"):
        self.n_slots = n_slots
        self.overflow_policy = overflow_policy
        self.slots: List[Optional[str]] = [None] * n_slots
        self.backlog: List[str] = []

    # ------------------------------------------------------------------ mutation
    def assign(self, order_id: str, recyclable: Optional[List[str]] = None) -> Optional[int]:
        if order_id in self.slots:
            return self.slots.index(order_id)
        for i, slot in enumerate(self.slots):
            if slot is None:
                self.slots[i] = order_id
                return i
        for oid in (recyclable or []):
            if oid in self.slots:
                i = self.slots.index(oid)
                self.slots[i] = order_id
                return i
        if self.overflow_policy == "raise":
            raise RuntimeError(f"no free order slot for {order_id} (M_max={self.n_slots})")
        self.backlog.append(order_id)          # "queue" policy
        return None

    def release(self, order_id: str) -> None:
        if order_id in self.slots:
            self.slots[self.slots.index(order_id)] = None

    def index_of(self, order_id: str) -> Optional[int]:
        return self.slots.index(order_id) if order_id in self.slots else None

    # ------------------------------------------------------------------ queries
    @property
    def occupied(self) -> List[int]:
        return [i for i, s in enumerate(self.slots) if s is not None]

    def occupancy_mask(self) -> np.ndarray:
        return np.array([0.0 if s is None else 1.0 for s in self.slots], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.occupied)
