"""
pesdvs.py
=========
PESDVS = Parallel Encoders + Serial Decoder for the Variable-length State.      [PAPER]

    (Z, U^k, D^k)  ->  Encoder_Z ‖ Encoder_U ‖ Encoder_D  ->  h_z, h_u, h_d
                   ->  6 serial decoder layers (Q=h_z / previous, K=h_u, V=h_d)
                   ->  one embedding per order slot

Everything is padded to M_max and masked, so a variable and growing number of dynamic
orders never changes the network shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from config import Config
from decoder import SerialDecoder
from encoder import ParallelEncoders


def state_to_tensors(state: Dict[str, np.ndarray], device: str = "cpu",
                     batch: bool = True) -> Dict[str, torch.Tensor]:
    """numpy state dict -> torch tensors with a leading batch dimension."""
    out = {}
    for key in ("Z", "U", "D", "order_mask", "service_location_mask"):
        arr = np.asarray(state[key], dtype=np.float32)
        t = torch.from_numpy(arr).to(device)
        out[key] = t.unsqueeze(0) if batch else t
    return out


class PESDVS(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoders = ParallelEncoders(cfg.state.z_dim, cfg.state.u_dim, cfg.state.d_dim,
                                         cfg.network)
        self.decoder = SerialDecoder(cfg.network)

    def forward(self, Z: torch.Tensor, U: torch.Tensor, D: torch.Tensor,
                order_mask: torch.Tensor) -> torch.Tensor:
        """[B, M, *] -> [B, M, hidden_dim]"""
        h_z, h_u, h_d = self.encoders(Z, U, D, order_mask)
        return self.decoder(h_z, h_u, h_d, order_mask)

    def forward_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(state["Z"], state["U"], state["D"], state["order_mask"])
