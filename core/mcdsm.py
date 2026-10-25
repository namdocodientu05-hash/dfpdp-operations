"""
mcdsm.py
========
MCDSM — the paper's vehicle-selection mechanism.                              [PAPER]

Vehicle selection is *deliberately kept outside the learned policy*: the Actor only ever
selects an ORDER, the MCDSM decides WHICH VEHICLE decides next.  This is also the paper's
task-allocation FAIRNESS mechanism, because the vehicle with the smallest accumulated
working time always moves next:

        k_t = argmin_{k in K_tilde} current_accumulated_time_k

Feasibility rules for a (vehicle k, order m) pair — generalised to mixed FedEx flows.
An order is feasible for a vehicle iff ALL of the following hold:

    order is released                                                        [PAPER]
    AND order is not cancelled and not completed                             [PAPER]
    AND vehicle is available                                                 [PAPER]
    AND if the next operation LOADS cargo   : Q_k - L_k >= w_m               [PAPER]
    AND if the cargo is already ON a vehicle: that vehicle is k              [PAPER, generalised]
    AND if a dispatch decision was ACCEPTED  : that vehicle is k             [EXTENSION —
                        NEW CONSTRAINT, switchable via cfg.mcdsm.lock_committed_orders]
    AND target location / time window is servable                            [REAL-TIME EXT]

For a plain PICKUP_DELIVERY order this reduces EXACTLY to the paper's rules
(z=0 -> capacity, z=1 -> same vehicle, z=2 -> never).  It additionally covers
OUTBOUND (pickup -> depot unload), INBOUND (depot load -> delivery) and
INBOUND_PRELOADED (delivery only, cargo already on board).

No FIFO / LIFO / pickup-delivery adjacency rule is imposed: a vehicle may interleave
operations of different orders freely, e.g.  P1 -> P2 -> D1 -> P3 -> D3 -> D2.

K_tilde = set of vehicles that are available AND have at least one feasible unfinished order.
Masks use 1 = feasible, 0 = masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from entities import Order, Vehicle, Z_COMPLETED, Z_WAIT_DELIVERY, Z_WAIT_PICKUP
from state_manager import StateManager
from utils import get_logger, hhmm

LOG = get_logger("dfpdp.mcdsm")


@dataclass
class MCDSMDecision:
    vehicle: Optional[Vehicle]
    k_tilde: List[str]
    masks: Dict[str, np.ndarray]
    vehicle_times: Dict[str, float]
    feasible_orders: Dict[str, List[str]]
    reason: str = ""

    def as_log_row(self, timestamp: float) -> Dict[str, Any]:
        return dict(
            timestamp=timestamp, clock=hhmm(timestamp),
            selected_vehicle=self.vehicle.vehicle_id if self.vehicle else "",
            k_tilde="|".join(self.k_tilde),
            vehicle_times="|".join(f"{k}:{v:.2f}" for k, v in self.vehicle_times.items()),
            feasible_orders="|".join(
                f"{k}:{','.join(v)}" for k, v in self.feasible_orders.items() if v),
            reason=self.reason,
        )


class MCDSM:
    def __init__(self, cfg: Config, state: StateManager):
        self.cfg = cfg
        self.state = state
        self.n_selections = 0
        self.log: List[Dict[str, Any]] = []
        self.visibility_time = None

    # [IMPLEMENTATION ASSUMPTION] used ONLY by the comparison baselines of §41:
    #   None                 -> event-driven mode, every released order is visible
    #   a timestamp t        -> orders released after t are invisible to the planner
    #                           (models "the plan is only refreshed periodically")
    # [EXTENSION] the horizon now LIVES ON THE STATE, so it hides the order from Z/U/D as
    # well as from the mask.  This property keeps `env.mcdsm.visibility_time = t` — the
    # spelling used by train.py and the §41 baselines — working unchanged.
    @property
    def visibility_time(self) -> Optional[float]:
        return self.state.visibility_time

    @visibility_time.setter
    def visibility_time(self, value: Optional[float]) -> None:
        self.state.visibility_time = value

    # ------------------------------------------------------------------ feasibility
    def order_feasible(self, vehicle: Vehicle, order: Order) -> bool:
        if order.cancelled or order.is_completed or not order.released:
            return False
        if (self.visibility_time is not None
                and not order.cargo_on_vehicle
                and order.release_time > self.visibility_time + 1e-9):
            return False
        # judged at the world clock, so a vehicle idle since the start of the shift cannot
        # look available hours after its shift ended (offline: no-op, see is_available)
        if not vehicle.is_available(self.state.now):
            return False
        # ---- ownership: cargo on board can only be handled by its carrier, and an ---------
        # accepted assignment is exclusive until it completes or its vehicle drops out.
        # `lock_owner` is shared with build_U so the mask and the U features cannot drift
        # apart; with `lock_committed_orders=False` it collapses to the cargo-only rule and
        # this function is behaviourally identical to the pre-extension version.
        owner = self.state.lock_owner(order)
        if owner is not None and owner != vehicle.vehicle_id:
            return False
        # ---- capacity: only binds when the next operation LOADS cargo ------------------
        if order.load_sign > 0 and order.cargo_weight > vehicle.available_capacity + 1e-9:
            return False
        # ---- an unloading operation requires the cargo to actually be on this vehicle ---
        if order.load_sign < 0 and not order.cargo_on_vehicle:
            return False
        return self.state.order_is_servable(vehicle, order)

    def feasible_mask(self, vehicle: Vehicle) -> np.ndarray:
        """1 = feasible order slot for THIS vehicle, 0 = masked."""
        mask = np.zeros(self.cfg.problem.M_max, dtype=np.float32)
        for oid in self.state.active_order_ids:
            order = self.state.orders[oid]
            if order.index is None or order.index < 0:
                continue
            if self.order_feasible(vehicle, order):
                mask[order.index] = 1.0
        return mask

    def feasible_order_ids(self, vehicle: Vehicle) -> List[str]:
        return [oid for oid in self.state.active_order_ids
                if self.order_feasible(vehicle, self.state.orders[oid])]

    # ------------------------------------------------------------------ selection
    def select_vehicle(self, first_step: bool = False) -> MCDSMDecision:
        masks: Dict[str, np.ndarray] = {}
        feasible: Dict[str, List[str]] = {}
        times: Dict[str, float] = {}
        k_tilde: List[str] = []

        for v in self.state.vehicle_list:
            mask = self.feasible_mask(v)
            masks[v.vehicle_id] = mask
            feasible[v.vehicle_id] = self.feasible_order_ids(v)
            times[v.vehicle_id] = v.current_time
            if mask.sum() > 0:
                k_tilde.append(v.vehicle_id)

        if not k_tilde:
            decision = MCDSMDecision(None, [], masks, times, feasible,
                                     reason="no vehicle has a feasible order")
        elif first_step and self.cfg.mcdsm.first_step_initialisation == "paper_first_vehicle":
            # [PAPER-style] at t=0 every accumulated time is identical -> take the first vehicle
            first = min(k_tilde, key=lambda vid: self.state.vehicles[vid].index)
            decision = MCDSMDecision(self.state.vehicles[first], k_tilde, masks, times, feasible,
                                     reason="first-step paper initialisation")
        else:
            best = min(k_tilde, key=lambda vid: (times[vid], self.state.vehicles[vid].index))
            decision = MCDSMDecision(self.state.vehicles[best], k_tilde, masks, times, feasible,
                                     reason="argmin accumulated time")

        self.n_selections += 1
        if self.cfg.mcdsm.log_every_selection:
            self.log.append(decision.as_log_row(self.state.now))
        return decision

    # ------------------------------------------------------------------ fairness
    def workload_vector(self) -> List[float]:
        return [v.working_time for v in self.state.vehicle_list]
