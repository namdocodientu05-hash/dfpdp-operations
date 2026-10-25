"""
environment.py
==============
The DFPDP environment.  It implements exactly the paper's decision cycle:

    current state
        -> MCDSM selects vehicle k                                          [PAPER]
        -> selected vehicle's FULL order state s_t^k = (Z, U^k, D^k)        [PAPER]
        -> Actor produces order probabilities (outside this module)         [PAPER]
        -> selected order m
        -> the ENVIRONMENT determines pickup vs delivery from z_m           [PAPER]
        -> feasible physical service location is selected                   [PAPER, flexible PDP]
        -> action executed, reward r_t = -(travel_time + waiting_time)      [PAPER]
        -> vehicle / order states updated, dynamic orders inserted          [PAPER]
        -> repeat

The Actor NEVER selects the vehicle and NEVER selects pickup-vs-delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from dynamic_orders import DynamicOrderManager
from entities import (Event, Order, ProblemInstance, Vehicle,
                      Z_COMPLETED, Z_WAIT_DELIVERY, Z_WAIT_PICKUP,
                      OP_CUSTOMER_DELIVERY, OP_CUSTOMER_PICKUP,
                      OP_DEPOT_LOAD, OP_DEPOT_UNLOAD)
from event_manager import EventManager, EventResult
from mcdsm import MCDSM, MCDSMDecision
from state_manager import Candidate, StateManager
from utils import fairness_stats, get_logger, hhmm

LOG = get_logger("dfpdp.env")


# =======================================================================================
@dataclass
class Observation:
    """One decision cycle: MCDSM already chose the vehicle."""
    vehicle: Optional[Vehicle]
    state: Dict[str, np.ndarray]
    action_mask: np.ndarray
    mcdsm: MCDSMDecision
    step_index: int
    time: float
    done: bool = False

    @property
    def has_action(self) -> bool:
        return self.vehicle is not None and float(self.action_mask.sum()) > 0


@dataclass
class StepInfo:
    order_id: str = ""
    vehicle_id: str = ""
    operation: str = ""
    flow_type: str = ""
    stage_index: int = 0
    service_location_id: str = ""
    selection_rule: str = ""
    travel_time: float = 0.0
    waiting_time: float = 0.0
    service_duration: float = 0.0
    distance: float = 0.0
    arrival_time: float = 0.0
    completion_time: float = 0.0
    lateness: float = 0.0
    zero_movement: bool = False
    reward: float = 0.0
    reward_baseline: float = 0.0
    reward_extensions: Dict[str, float] = field(default_factory=dict)
    invalid: bool = False
    message: str = ""

    def as_row(self) -> Dict[str, Any]:
        row = {k: v for k, v in self.__dict__.items() if k != "reward_extensions"}
        row.update({f"ext_{k}": v for k, v in self.reward_extensions.items()})
        row["clock"] = hhmm(self.completion_time)
        return row


# =======================================================================================
class DFPDPEnvironment:
    def __init__(self, instance: ProblemInstance, cfg: Config):
        self.cfg = cfg
        self.instance = instance
        self.state = StateManager(instance, cfg)
        self.mcdsm = MCDSM(cfg, self.state)
        self.dynamic = DynamicOrderManager(cfg, self.state)
        self.events = EventManager(cfg, self.state)
        self.step_index = 0
        self.episode_reward = 0.0
        self.step_log: List[StepInfo] = []
        self.reassigned_orders: set = set()
        self._planned_next: Dict[str, str] = {}
        self._last_obs: Optional[Observation] = None
        # [EXTENSION] optional ceiling for the idle fast-forward.  Offline callers never set
        # it, so `_advance_idle_clock` keeps its original behaviour and offline results are
        # provably unchanged.  The online dispatcher sets it to the timestamp of the next
        # exogenous event so the fleet cannot idle PAST an event that has not arrived yet.
        self.clock_horizon: Optional[float] = None

    # ------------------------------------------------------------------ episode setup
    def reset(self, replay_events: bool = False) -> Observation:
        """[§9] initialisation at start of day / start of shift."""
        self.state.reset()
        self.mcdsm = MCDSM(self.cfg, self.state)
        self.dynamic = DynamicOrderManager(self.cfg, self.state)
        self.events = EventManager(self.cfg, self.state)
        self.step_index = 0
        self.episode_reward = 0.0
        self.step_log = []
        self.reassigned_orders = set()
        self._planned_next = {}
        if replay_events and self.instance.events:
            self.events.queue.push_many(
                [e for e in self.instance.events if e.event_type not in ("NEW_ORDER",)])
        return self.observe(first_step=True)

    # ------------------------------------------------------------------ observation
    def observe(self, first_step: bool = False) -> Observation:
        """Runs MCDSM and builds s_t^k for the selected vehicle."""
        self._advance_idle_clock()
        decision = self.mcdsm.select_vehicle(first_step=first_step)
        if decision.vehicle is None:
            obs = Observation(None, self._empty_state(), np.zeros(self.cfg.problem.M_max,
                                                                 dtype=np.float32),
                              decision, self.step_index, self.state.now, done=self.is_done())
        else:
            v = decision.vehicle
            state = self.state.build_state(v)
            obs = Observation(v, state, decision.masks[v.vehicle_id], decision,
                              self.step_index, self.state.now, done=self.is_done())
        self._last_obs = obs
        return obs

    def _empty_state(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        return {
            "Z": np.zeros((cfg.problem.M_max, cfg.state.z_dim), dtype=np.float32),
            "U": np.zeros((cfg.problem.M_max, cfg.state.u_dim), dtype=np.float32),
            "D": np.zeros((cfg.problem.M_max, cfg.state.d_dim), dtype=np.float32),
            "order_mask": np.zeros(cfg.problem.M_max, dtype=np.float32),
            "service_location_mask": np.zeros((cfg.problem.M_max, cfg.problem.S_max),
                                              dtype=np.float32),
            "vehicle_available": np.float32(0.0),
        }

    # ------------------------------------------------------------------ clock
    def _advance_idle_clock(self) -> None:
        """[IMPLEMENTATION ASSUMPTION] simulated global clock.

        The clock of the simulation is the minimum accumulated time of the available
        vehicles (MCDSM always moves that vehicle next).  If nobody has a feasible order
        but dynamic orders are still to come, all available vehicles idle forward to the
        next release time.  Idle waiting for *future* orders is NOT counted as working
        time and produces no reward, because no service is performed.
        """
        available = [v for v in self.state.vehicle_list if v.is_available()]
        if available:
            # [FIX] the simulated clock is MONOTONE.  Assigning the fleet minimum outright
            # let `state.now` fall back BEHIND the timestamp of an event that had just been
            # applied (every `on_*` handler advances it with `max(now, timestamp)`), which
            # erased the planning instant and made every derived ETA start in the past.
            self.state.now = max(self.state.now, min(v.current_time for v in available))
        self.dynamic.release_due(self.state.now, trigger="clock")

        if any(self.mcdsm.feasible_mask(v).sum() > 0 for v in available):
            return
        nxt = self.state.next_release_time
        if nxt is None or not available:
            return
        # [EXTENSION] never idle past a pending exogenous event: online, `next_release_time`
        # is the arrival time of an order the ERP has NOT announced yet, so jumping the whole
        # fleet to it would both leak future information and let the simulation overrun an
        # event (breakdown, cancellation) that is still in flight.
        if self.clock_horizon is not None:
            if nxt > self.clock_horizon + 1e-9:
                return
            nxt = min(nxt, self.clock_horizon)
        for v in available:
            v.current_time = max(v.current_time, nxt)
        self.state.now = max(self.state.now, nxt)
        self.dynamic.release_due(self.state.now, trigger="idle_advance")

    def is_done(self) -> bool:
        if self.state.all_done():
            return True
        available = [v for v in self.state.vehicle_list if v.is_available()]
        if not available:
            return True
        if all(self.mcdsm.feasible_mask(v).sum() == 0 for v in available) \
                and self.state.next_release_time is None:
            return True
        return False

    # ------------------------------------------------------------------ step
    def step(self, order_slot: int) -> Tuple[Observation, float, bool, StepInfo]:
        """Execute the Actor's ORDER choice for the MCDSM-selected vehicle."""
        obs = self._last_obs or self.observe()
        vehicle = obs.vehicle
        info = StepInfo()
        if vehicle is None:
            info.invalid, info.message = True, "no vehicle selected by MCDSM"
            return self.observe(), 0.0, True, info

        # [FIX] the PLANNING INSTANT of this decision.  `on_service_complete` below advances
        # `state.now` to the SIMULATED completion of this one leg; that is a fact about ONE
        # vehicle, not about the world, so it must not become the departure floor of the
        # other vehicles (an idle V2 at 08:00 may still leave at 08:00 after V1 finished a
        # leg at 08:15).  Only exogenous events move the global clock forward.
        planning_now = self.state.now

        order = self._order_at_slot(order_slot)
        if order is None or obs.action_mask[order_slot] <= 0:
            info.invalid = True
            info.message = f"slot {order_slot} is masked / empty"
            return obs, 0.0, self.is_done(), info

        # ---- 1. the ENVIRONMENT resolves the operation from the order's service plan ----
        # [PAPER §26] the Actor never chooses the operation type.  [FEDEX EXTENSION] the
        # operation is now read from the plan instead of being hard-coded from z, so the
        # very same decision cycle drives PICKUP_DELIVERY, OUTBOUND, INBOUND and
        # INBOUND_PRELOADED without the policy knowing which flow it is looking at.
        operation = order.next_operation
        if operation is None:
            info.invalid, info.message = True, f"order {order.order_id} already completed"
            return obs, 0.0, self.is_done(), info

        # ---- 2. physical service location selection  [PAPER §27] -----------------------
        cand, rule = self.state.select_service_location(vehicle, order)
        if cand is None:
            return self._handle_infeasible(order, vehicle, rule, obs)

        # ---- 3. timing -----------------------------------------------------------------
        # [FIX] `Candidate.arrival_time` is the SINGLE source of truth for the ETA: it is
        # computed from the derived departure instant max(vehicle.current_time, state.now),
        # so re-deriving it here from the vehicle clock alone would silently undo the fix.
        arrival = cand.arrival_time
        waiting = cand.waiting_time
        service_start = arrival + waiting
        completion = service_start + cand.service_duration

        # ---- 4. reward  [PAPER §29] r_t = -(travel_time + waiting_time) -----------------
        baseline = -(cand.travel_time + waiting)
        extensions = self._reward_extensions(order, vehicle, cand, completion, operation)
        reward = (baseline + sum(extensions.values())) * self.cfg.reward.reward_scale
        # [FIX] the emergency-reassignment penalty is a ONE-OFF cost of the reassignment,
        # not a per-leg surcharge.  `reassigned_orders` was never cleared, so a two-stage
        # order was charged again at every remaining operation of its plan.  This step is
        # past every early return, so the charge has definitely been applied exactly once.
        self.reassigned_orders.discard(order.order_id)

        # ---- 5. execute ----------------------------------------------------------------
        vehicle.travel_time += cand.travel_time
        vehicle.waiting_time += waiting
        vehicle.distance += cand.distance
        vehicle.working_time += cand.travel_time + waiting + cand.service_duration
        vehicle.current_time = completion
        vehicle.route.append(dict(t=completion, x=cand.location.x, y=cand.location.y,
                                  order_id=order.order_id, operation=operation,
                                  flow_type=order.flow_type,
                                  service_location_id=cand.location.service_location_id))
        self.state.on_service_complete(order.order_id, vehicle.vehicle_id, completion,
                                       operation, cand.location.service_location_id)

        # ---- 6. dynamic orders released in the meantime  [PAPER §10] --------------------
        # [FIX] monotone w.r.t. the PLANNING INSTANT (see `planning_now` above), not w.r.t.
        # the simulated completion `on_service_complete` just stamped onto `state.now`.
        fleet_clock = (min(v.current_time for v in self.state.vehicle_list if v.is_available())
                       if any(v.is_available() for v in self.state.vehicle_list) else completion)
        self.state.now = max(planning_now, fleet_clock)
        self.dynamic.release_due(self.state.now, trigger="step")

        # ---- 7. bookkeeping ------------------------------------------------------------
        info = StepInfo(
            order_id=order.order_id, vehicle_id=vehicle.vehicle_id, operation=operation,
            flow_type=order.flow_type, stage_index=order.stage_index,
            service_location_id=cand.location.service_location_id, selection_rule=rule,
            travel_time=cand.travel_time, waiting_time=waiting,
            service_duration=cand.service_duration, distance=cand.distance,
            arrival_time=arrival, completion_time=completion, lateness=cand.lateness,
            zero_movement=cand.same_location, reward=reward, reward_baseline=baseline,
            reward_extensions=extensions)
        self.step_index += 1
        self.episode_reward += reward
        self.step_log.append(info)
        self._planned_next[vehicle.vehicle_id] = order.order_id

        next_obs = self.observe()
        return next_obs, reward, next_obs.done or self.is_done(), info

    # ------------------------------------------------------------------ helpers
    def _order_at_slot(self, slot: int) -> Optional[Order]:
        if slot is None or slot < 0 or slot >= self.cfg.problem.M_max:
            return None
        oid = self.state.slots.slots[slot]
        return self.state.orders.get(oid) if oid else None

    def _handle_infeasible(self, order: Order, vehicle: Vehicle, rule: str,
                           obs: Observation) -> Tuple[Observation, float, bool, StepInfo]:
        """[§28] no service location with a feasible time window."""
        info = StepInfo(order_id=order.order_id, vehicle_id=vehicle.vehicle_id,
                        selection_rule=rule, invalid=True)
        if rule == "drop_order":
            self.state.on_cancel_order(order.order_id, self.state.now)
            info.message = "order dropped (infeasibility_rule=drop_order)"
        elif rule == "defer_order":
            # the order stays but the vehicle idles forward to the next release / window
            info.message = "order deferred (infeasibility_rule=defer_order)"
            vehicle.current_time += 1.0
        else:
            info.message = f"no service location ({rule})"
        return self.observe(), 0.0, self.is_done(), info

    def _reward_extensions(self, order: Order, vehicle: Vehicle, cand: Candidate,
                           completion: float, operation: str) -> Dict[str, float]:
        """[§30] OPTIONAL real-time reward extensions — all OFF by default."""
        r = self.cfg.reward
        ext: Dict[str, float] = {}
        if r.use_late_delivery_penalty:
            latest = order.latest_delivery_time
            late = 0.0
            if operation == OP_CUSTOMER_DELIVERY and latest is not None and completion > latest:
                late = completion - latest
            late += cand.lateness
            ext["late_delivery"] = -r.late_penalty_per_minute * late
        if r.use_priority_penalty:
            waited = max(0.0, completion - order.release_time) / 60.0
            ext["priority"] = -r.priority_penalty_weight * order.priority * waited
        if r.use_route_change_penalty:
            planned = self._planned_next.get(vehicle.vehicle_id)
            changed = planned is not None and planned != order.order_id
            ext["route_change"] = -r.route_change_penalty if changed else 0.0
        if r.use_emergency_reassignment_penalty:
            ext["emergency_reassignment"] = (
                -r.emergency_reassignment_penalty if order.order_id in self.reassigned_orders else 0.0)
        if r.use_fairness_in_reward:
            wl = [v.working_time for v in self.state.vehicle_list]
            gap_before = max(wl) - min(wl)
            wl2 = list(wl)
            wl2[vehicle.index] += cand.travel_time + cand.waiting_time + cand.service_duration
            gap_after = max(wl2) - min(wl2)
            ext["fairness"] = -r.fairness_weight * max(0.0, gap_after - gap_before)
        return ext

    # ------------------------------------------------------------------ events
    def apply_event(self, event: Event) -> EventResult:
        """[§11] inject a real-time / historical event into the running environment."""
        result = self.events.process(event)
        if (event.event_type == "VEHICLE_BREAKDOWN"
                and result.info.get("reassigned")):
            self.reassigned_orders.update(result.info["reassigned"])
        return result

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> Dict[str, Any]:
        vehicles = self.state.vehicle_list
        orders = list(self.state.orders.values())
        completed = [o for o in orders if o.is_completed]
        late = [o for o in completed if o.late]
        unserved = [o for o in orders if o.released and not o.is_completed and not o.cancelled]
        stats = fairness_stats([v.working_time for v in vehicles])
        served_time = sum(v.working_time for v in vehicles)
        horizon = max(1e-9, max((v.current_time for v in vehicles), default=0.0)
                      - self.cfg.clock.day_start)
        out = {
            "episode_reward": self.episode_reward,
            "n_steps": self.step_index,
            "total_travel_time": sum(v.travel_time for v in vehicles),
            "total_waiting_time": sum(v.waiting_time for v in vehicles),
            "total_distance": sum(v.distance for v in vehicles),
            "completed_orders": len(completed),
            "late_orders": len(late),
            "total_lateness": sum(o.lateness for o in completed),
            "unserved_orders": len(unserved),
            "dynamic_released": len(self.dynamic.records),
            "makespan": max((v.current_time for v in vehicles), default=0.0) - self.cfg.clock.day_start,
            "vehicle_utilisation": served_time / (horizon * max(1, len(vehicles))),
        }
        # ---- [FEDEX EXTENSION] per-flow service level -----------------------------------
        for flow in ("PICKUP_DELIVERY", "OUTBOUND", "INBOUND", "INBOUND_PRELOADED"):
            of_flow = [o for o in orders if o.flow_type == flow and o.released]
            out[f"orders_{flow.lower()}"] = len(of_flow)
            out[f"completed_{flow.lower()}"] = len([o for o in of_flow if o.is_completed])
        out["cargo_left_on_vehicles"] = sum(
            len(v.carrying) for v in vehicles)
        out["final_load_total"] = sum(v.load for v in vehicles)
        out.update(stats)
        return out

    # ------------------------------------------------------------------ export
    def route_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for v in self.state.vehicle_list:
            for i, leg in enumerate(v.route):
                rows.append(dict(vehicle_id=v.vehicle_id, seq=i, **leg, clock=hhmm(leg["t"])))
        return rows

    def order_rows(self) -> List[Dict[str, Any]]:
        return [dict(order_id=o.order_id, order_class=o.order_class, z=o.z,
                     flow_type=o.flow_type, stage=o.stage, stage_index=o.stage_index,
                     n_stages=o.n_stages, cargo_state=o.cargo_state,
                     assigned_vehicle_id=o.assigned_vehicle_id or "",
                     release_time=o.release_time, release_clock=hhmm(o.release_time),
                     pickup_timestamp=o.pickup_timestamp or "",
                     delivery_timestamp=o.delivery_timestamp or "",
                     cargo_weight=o.cargo_weight, priority=o.priority,
                     late=o.late, lateness=o.lateness, cancelled=o.cancelled)
                for o in self.state.orders.values()]

    def vehicle_rows(self) -> List[Dict[str, Any]]:
        return [dict(vehicle_id=v.vehicle_id, vehicle_type=v.vehicle_type, capacity=v.capacity,
                     final_load=v.load, x=v.x, y=v.y, current_time=v.current_time,
                     clock=hhmm(v.current_time), status=v.status, working_time=v.working_time,
                     travel_time=v.travel_time, waiting_time=v.waiting_time,
                     distance=v.distance, n_services=v.n_services)
                for v in self.state.vehicle_list]
