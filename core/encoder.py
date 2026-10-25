"""
encoder.py
==========
PESDVS — parallel encoders.                                                    [PAPER §22]

Three *parallel* encoders process the three parts of the state independently:

        Encoder_Z (order status)   -> h_z
        Encoder_U (cargo / load)   -> h_u
        Encoder_D (travel info)    -> h_d

Each encoder is

        input projection to hidden_dim (256)
        -> Multi-Head Attention (8 heads)
        -> residual -> LayerNorm
        -> Feed Forward (Tanh)
        -> residual -> LayerNorm

Padded order slots are removed from the attention with a key-padding mask, which is what
makes the variable number of orders possible without ever rebuilding the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import NetworkConfig
from variable_length import key_padding_mask


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, use_tanh: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.Tanh() if use_tanh else nn.ReLU(),        # [PAPER] Tanh where consistent
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """MHA -> residual -> LayerNorm -> FF -> residual -> LayerNorm."""

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ff = FeedForward(cfg.hidden_dim, cfg.ff_dim, cfg.use_tanh_in_ff, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = key_padding_mask(mask)                     # True where padding
        # a fully padded row would make softmax undefined -> keep at least one visible key
        pad = torch.where(pad.all(dim=-1, keepdim=True), torch.zeros_like(pad), pad)
        attended, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        return x


class StateEncoder(nn.Module):
    """One of the three parallel encoders (Z, U or D)."""

    def __init__(self, in_dim: int, cfg: NetworkConfig, n_blocks: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h, mask)
        return h * mask.unsqueeze(-1)                    # zero out padded slots


class ParallelEncoders(nn.Module):
    """Encoder_Z ‖ Encoder_U ‖ Encoder_D  ->  (h_z, h_u, h_d)."""

    def __init__(self, z_dim: int, u_dim: int, d_dim: int, cfg: NetworkConfig,
                 n_blocks: int = 1):
        super().__init__()
        self.encoder_z = StateEncoder(z_dim, cfg, n_blocks)
        self.encoder_u = StateEncoder(u_dim, cfg, n_blocks)
        self.encoder_d = StateEncoder(d_dim, cfg, n_blocks)

    def forward(self, Z: torch.Tensor, U: torch.Tensor, D: torch.Tensor,
                mask: torch.Tensor):
        return (self.encoder_z(Z, mask),
                self.encoder_u(U, mask),
                self.encoder_d(D, mask))
