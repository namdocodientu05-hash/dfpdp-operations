"""
actor_critic.py
===============
Actor  : PESDVS + one logit per order slot -> masked Softmax over ORDERS ONLY   [PAPER §24]
         The vehicle has already been chosen by the MCDSM; the Actor never chooses the
         vehicle and never chooses pickup-vs-delivery.
Critic : state value V_omega(s_t^k)                                             [PAPER §31]

Vanilla Actor-Critic (NOT PPO / DQN / SAC):

        delta_t      = r_t + gamma * V(s_{t+1}) - V(s_t)
        actor loss   = -delta_t.detach() * log pi_theta(a_t | s_t)
        critic loss  = delta_t^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from encoder import ParallelEncoders
from pesdvs import PESDVS, state_to_tensors
from variable_length import apply_action_mask, masked_mean


# =======================================================================================
class Actor(nn.Module):
    """pi_theta(m | s_t^k) — a probability distribution over ORDERS."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.body = PESDVS(cfg)
        self.head = nn.Linear(cfg.network.hidden_dim, 1)

    def logits(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.body.forward_state(state)                     # [B, M, H]
        logits = self.head(h).squeeze(-1)                      # [B, M]
        clip = self.cfg.network.logit_clip
        if clip and clip > 0:
            logits = clip * torch.tanh(logits / clip)          # stability, keeps ordering
        return logits

    def forward(self, state: Dict[str, torch.Tensor],
                action_mask: torch.Tensor) -> torch.Tensor:
        """Invalid orders are masked BEFORE the Softmax  [PAPER §24]."""
        return torch.softmax(apply_action_mask(self.logits(state), action_mask), dim=-1)

    def distribution(self, state: Dict[str, torch.Tensor],
                     action_mask: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(probs=self.forward(state, action_mask))


# =======================================================================================
class Critic(nn.Module):
    """V_omega(s_t^k). Own parallel encoders + masked mean pooling + MLP.

    [IMPLEMENTATION ASSUMPTION] the paper does not detail the critic's architecture; we use
    the same parallel-encoder front-end with mean pooling, which keeps it permutation
    invariant and variable-length safe.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoders = ParallelEncoders(cfg.state.z_dim, cfg.state.u_dim, cfg.state.d_dim,
                                         cfg.network)
        h = cfg.network.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * h, cfg.network.critic_hidden),
            nn.Tanh(),
            nn.Linear(cfg.network.critic_hidden, 1),
        )

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = state["order_mask"]
        h_z, h_u, h_d = self.encoders(state["Z"], state["U"], state["D"], mask)
        pooled = torch.cat([masked_mean(h_z, mask), masked_mean(h_u, mask),
                            masked_mean(h_d, mask)], dim=-1)
        return self.mlp(pooled).squeeze(-1)                    # [B]


# =======================================================================================
@dataclass
class ActionResult:
    slot: int
    prob: float
    log_prob: torch.Tensor
    entropy: torch.Tensor
    probs: np.ndarray
    mode: str


class ActorCriticAgent:
    """Holds the two networks, the two Adam optimizers and the TD update."""

    def __init__(self, cfg: Config, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or cfg.train.device
        self.actor = Actor(cfg).to(self.device)
        self.critic = Critic(cfg).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)

    # ------------------------------------------------------------------ inference
    def to_tensors(self, state: Dict[str, np.ndarray],
                   action_mask: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        st = state_to_tensors(state, self.device)
        mask = torch.from_numpy(np.asarray(action_mask, dtype=np.float32)
                                ).to(self.device).unsqueeze(0)
        return st, mask

    def act(self, state: Dict[str, np.ndarray], action_mask: np.ndarray,
            mode: str = "sample", best_of_n: int = 1,
            score_fn=None) -> ActionResult:
        """[PAPER §25] training uses categorical sampling; inference may use greedy /
        single stochastic sampling / best-of-N sampling."""
        st, mask = self.to_tensors(state, action_mask)
        probs = self.actor(st, mask)                                  # [1, M]
        dist = torch.distributions.Categorical(probs=probs)
        if mode == "greedy":
            slot = int(torch.argmax(probs, dim=-1).item())
        elif mode == "best_of_n":
            samples = [int(dist.sample().item()) for _ in range(max(1, best_of_n))]
            if score_fn is None:
                slot = max(set(samples), key=samples.count)           # modal sample
            else:
                slot = max(samples, key=score_fn)
        else:                                                          # "sample"
            slot = int(dist.sample().item())
        action = torch.tensor([slot], device=self.device)
        return ActionResult(slot=slot, prob=float(probs[0, slot].item()),
                            log_prob=dist.log_prob(action).squeeze(0),
                            entropy=dist.entropy().squeeze(0),
                            probs=probs.detach().cpu().numpy()[0], mode=mode)

    def value(self, state: Dict[str, np.ndarray]) -> torch.Tensor:
        st = state_to_tensors(state, self.device)
        return self.critic(st).squeeze(0)

    # ------------------------------------------------------------------ learning
    def update(self, state: Dict[str, np.ndarray], log_prob: torch.Tensor,
               reward: float, next_state: Optional[Dict[str, np.ndarray]],
               done: bool, entropy: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """One vanilla Actor-Critic TD(0) update."""
        cfg = self.cfg.train
        v = self.critic(state_to_tensors(state, self.device)).squeeze(0)
        with torch.no_grad():
            if done or next_state is None:
                v_next = torch.zeros((), device=self.device)
            else:
                v_next = self.critic(state_to_tensors(next_state, self.device)).squeeze(0)
            target = torch.as_tensor(reward, dtype=torch.float32,
                                     device=self.device) + cfg.gamma * v_next
        delta = target - v                                   # TD residual

        critic_loss = delta.pow(2)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        actor_loss = -delta.detach() * log_prob
        if entropy is not None and cfg.entropy_coef > 0:
            actor_loss = actor_loss - cfg.entropy_coef * entropy
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        return {"actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "td_error": float(delta.item()),
                "value": float(v.item())}

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str, extra: Optional[dict] = None) -> str:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "config": self.cfg.to_dict(),
                    # [FEDEX EXTENSION] the state layout is stamped into every checkpoint
                    # so a model trained on a different Z/U/D layout can never be loaded
                    # silently and produce nonsense recommendations.
                    "state_version": self.cfg.state.state_version,
                    "state_dims": {"z_dim": self.cfg.state.z_dim,
                                   "u_dim": self.cfg.state.u_dim,
                                   "d_dim": self.cfg.state.d_dim},
                    "extra": extra or {}}, path)
        return path

    def load(self, path: str, strict: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        want = self.cfg.state.state_version
        got = ckpt.get("state_version", "pre_fedex_paper_v0")
        if strict and got != want:
            raise ValueError(
                f"checkpoint '{path}' was trained with state layout '{got}' but this "
                f"build uses '{want}'. The FedEx mixed-flow extension changed the Z and U "
                f"feature widths (z_dim 7->12, u_dim 7->10), so OLD CHECKPOINTS ARE "
                f"INCOMPATIBLE and the model must be retrained "
                f"(python main.py train --episodes 300).")
        self.actor.load_state_dict(ckpt["actor"], strict=strict)
        self.critic.load_state_dict(ckpt["critic"], strict=strict)
        return ckpt.get("extra", {})

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
