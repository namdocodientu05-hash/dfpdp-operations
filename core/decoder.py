"""
decoder.py
==========
PESDVS — serial decoder.                                                      [PAPER §23]

Six serial Transformer decoder layers by default.

    layer 1 :  Query = h_z ,  Key = h_u ,  Value = h_d
    layer l :  Query = output of layer l-1 ,  Key = h_u ,  Value = h_d

The decoder returns ONE representation per (active or padded) order slot, which the Actor
turns into one logit per order.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from config import NetworkConfig
from encoder import FeedForward
from variable_length import key_padding_mask


class SerialDecoderLayer(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ff = FeedForward(cfg.hidden_dim, cfg.ff_dim, cfg.use_tanh_in_ff, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, query: torch.Tensor, h_u: torch.Tensor, h_d: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pad = key_padding_mask(mask)
        pad = torch.where(pad.all(dim=-1, keepdim=True), torch.zeros_like(pad), pad)
        attended, _ = self.attn(query, h_u, h_d, key_padding_mask=pad, need_weights=False)
        x = self.norm1(query + attended)
        x = self.norm2(x + self.ff(x))
        return x


class SerialDecoder(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.layers = nn.ModuleList([SerialDecoderLayer(cfg)
                                     for _ in range(cfg.n_decoder_layers)])

    def forward(self, h_z: torch.Tensor, h_u: torch.Tensor, h_d: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = h_z                                    # first query = h_z          [PAPER]
        for layer in self.layers:
            x = layer(x, h_u, h_d, mask)           # later queries = previous output
        return x * mask.unsqueeze(-1)
