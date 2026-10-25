"""
realtime_inference.py
=====================
MODE B — ONLINE REAL-TIME INFERENCE.                             [REAL-TIME EXTENSION]

    live ERP/OMS orders + live TMS/GPS vehicle events + optional live traffic
        -> Real-Time State Manager  (update Z / U / D)
        -> replanning trigger
        -> MCDSM (vehicle selection)
        -> pretrained PESDVS Actor (order selection)
        -> deterministic feasibility validator
        -> dispatch recommendation
        -> wait for the next event

Two operating modes, switched by `cfg.inference.auto_approve`:

    False (DEPLOYMENT, default)  open loop — `replan()` returns a recommendation and the
                                 state only advances when the real *_COMPLETE event
                                 arrives.  The order is COMMITTED to the recommended
                                 vehicle so no other vehicle can be offered it meanwhile.
    True  (ACADEMIC)             closed loop — every recommendation is assumed fully
                                 approved and is executed at once through `env.step()`,
                                 so the run produces real experimental numbers.

The model is NOT retrained after every event; online operation performs inference only.
Periodic offline retraining on accumulated recent data is a separate, explicit step
(`train.py` on a refreshed workbook, or `main.py retrain`).

Generic ingestion interface:  `dispatcher.process_event(event)`  — no company-specific API.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from actor_critic import ActorCriticAgent
from config import Config
from entities import (EVENT_OPERATION, Event, Order, ProblemInstance, Vehicle,
                      Z_COMPLETED, Z_WAIT_DELIVERY, Z_WAIT_PICKUP)   # noqa: F401 (legacy view)
from environment import DFPDPEnvironment
from event_manager import EventResult
from replan_trigger import ReplanTrigger, TriggerDecision
from utils import ensure_dir, get_logger, hhmm, latency_stats, write_csv

LOG = get_logger("dfpdp.online")


# =======================================================================================
@dataclass
class DispatchDecision:
    event_timestamp: float
    clock: str
    replan_reason: str
    vehicle_id: str = ""
    order_id: str = ""
    operation: str = ""
    flow_type: str = ""                 # [FEDEX EXTENSION]
    stage: str = ""                     # [FEDEX EXTENSION] WAIT_<operation> / COMPLETED
    service_location_id: str = ""
    location_x: float = 0.0
    location_y: float = 0.0
    expected_travel_time: float = 0.0
    expected_waiting_time: float = 0.0
    expected_arrival_time: float = 0.0
    expected_completion_time: float = 0.0
    reward_estimate: float = 0.0
    policy_probability: float = 0.0
    decode_mode: str = ""
    inference_latency_ms: float = 0.0
    decision_latency_ms: float = 0.0
    feasible: bool = True
    validator_notes: str = ""
    k_tilde: str = ""
    n_active_orders: int = 0
    status: str = "OK"
    # ---- [EXTENSION] academic closed loop (`inference.auto_approve`) -------------------
    executed: bool = False              # the recommendation was actually executed
    realized_reward: float = 0.0        # r_t returned by env.step (0.0 when not executed)
    actual_completion_time: float = 0.0 # completion time realised by the environment
    cascade_index: int = 0              # 0 = first decision of the event, >0 = follow-up

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


# =======================================================================================
class FeasibilityValidator:
    """[§36] deterministic safety check run BEFORE any recommendation leaves the system."""

    def __init__(self, cfg: Config, env: DFPDPEnvironment):
        self.cfg = cfg
        self.env = env

    def check(self, vehicle: Vehicle, order: Order) -> (bool, str):
        st = self.env.state
        if order is None:
            return False, "empty order slot"
        if not vehicle.is_available(st.now):
            return False, "vehicle unavailable (breakdown / off shift)"
        if order.cancelled:
            return False, "order cancelled"
        if order.is_completed:
            return False, "order already completed"
        op = order.next_operation
        # ---- [FEDEX EXTENSION] one generic check per operation semantics ---------------
        if order.load_sign > 0:                       # CUSTOMER_PICKUP or DEPOT_LOAD
            if order.cargo_weight > vehicle.available_capacity + 1e-9:
                return False, (f"capacity violated for {op}: w={order.cargo_weight} > "
                               f"Q-L={vehicle.available_capacity}")
        else:                                         # CUSTOMER_DELIVERY or DEPOT_UNLOAD
            if not order.cargo_on_vehicle:
                return False, f"{op} requested but cargo is not loaded yet"
            if order.assigned_vehicle_id != vehicle.vehicle_id:
                return False, (f"{op} must be done by {order.assigned_vehicle_id}, "
                               f"not {vehicle.vehicle_id}")
            if order.order_id not in vehicle.carrying:
                return False, (f"{op}: order {order.order_id} is not in the carrying list "
                               f"of {vehicle.vehicle_id}")
        cand, rule = st.select_service_location(vehicle, order)
        if cand is None:
            return False, f"no available service location ({rule})"
        if not cand.location.available:
            return False, "service location unavailable"
        if not cand.feasible and rule != "allow_late_logged":
            return False, "time window infeasible"
        return True, f"ok ({rule})"


# =======================================================================================
class RealTimeDispatcher:
    """Event-driven online dispatcher around a pretrained Actor."""

    def __init__(self, instance: ProblemInstance, cfg: Config,
                 agent: Optional[ActorCriticAgent] = None,
                 checkpoint: Optional[str] = None,
                 auto_reset: bool = True):
        self.cfg = cfg
        self.env = DFPDPEnvironment(instance, cfg)
        if auto_reset:
            self.env.reset(replay_events=False)
        self.agent = agent or ActorCriticAgent(cfg)
        ckpt = checkpoint or cfg.inference.checkpoint
        self.checkpoint_loaded = False
        if ckpt and os.path.exists(ckpt):
            self.agent.load(ckpt)
            self.checkpoint_loaded = True
            LOG.info("loaded pretrained actor/critic from %s", ckpt)
        else:
            LOG.warning("no checkpoint at %s -- running with an UNTRAINED actor", ckpt)
        self.agent.eval_mode()
        self.trigger = ReplanTrigger(cfg)
        self.validator = FeasibilityValidator(cfg, self.env)
        # [EXTENSION] closed-loop academic mode: every recommendation is assumed approved
        self.auto_approve: bool = bool(cfg.inference.auto_approve)
        self.last_cascade: List[DispatchDecision] = []
        # per-replan diagnostics that parallel `last_cascade` (used by the toy transcript)
        self.last_cascade_diag: List[Dict[str, Any]] = []

        # ---- logs -----------------------------------------------------------------
        self.event_log: List[Dict[str, Any]] = []
        self.replan_log: List[Dict[str, Any]] = []
        self.dispatch_log: List[Dict[str, Any]] = []
        self.vehicle_history: List[Dict[str, Any]] = []
        self.order_history: List[Dict[str, Any]] = []
        self.traffic_history: List[Dict[str, Any]] = []
        self.latencies_ms: List[float] = []
        self.last_decision: Optional[DispatchDecision] = None
        # diagnostics of the last replanning cycle (used by the toy demo / dashboards)
        self.last_obs = None
        self.last_probs: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.on_decision: Optional[Callable[[DispatchDecision], None]] = None

    # ------------------------------------------------------------------ main entry
    def process_event(self, event: Event,
                      horizon: Optional[float] = None) -> Optional[DispatchDecision]:
        """Generic ingestion point.  Returns the FIRST dispatch recommendation, or None.

        `horizon` is the timestamp of the next exogenous event.  In auto-approve mode the
        dispatcher keeps deciding (a "cascade") until the simulated clock passes it, then
        hands control back so the event stream stays chronological.  The full cascade is
        available in `self.last_cascade`; the return type stays a single decision because
        the §7 integration API and the existing tests depend on it.
        """
        t_received = time.perf_counter()
        self.last_cascade = []
        self.last_cascade_diag = []

        # ---- [EXTENSION] auto-approve ignores external service confirmations -----------
        # When every recommendation is executed on the spot, an incoming *_COMPLETE for an
        # operation this dispatcher already performed is a DUPLICATE: replaying it would
        # hit the `operation == expected` assertion in on_service_complete, be swallowed by
        # EventManager.process and be logged as a failed event.  It is filtered here (not
        # in main.py) so `main.py realtime` on the workbook's Realtime_Vehicle_Events sheet
        # works untouched.
        if self.auto_approve and (event.event_type in EVENT_OPERATION
                                  or event.event_type == "SERVICE_COMPLETE"):
            row = event.to_row()
            row.update(clock=hhmm(event.timestamp), handled=True, error="",
                       info="IGNORED_AUTO_APPROVE: execution is driven by the dispatcher, "
                            "not by external confirmations")
            self.event_log.append(row)
            self.replan_log.append(dict(
                timestamp=event.timestamp, replan=False,
                reason=f"{event.event_type}:ignored_auto_approve", detail="{}",
                clock=hhmm(event.timestamp), event_type=event.event_type,
                vehicle_id=event.vehicle_id or "", order_id=event.order_id or ""))
            self._snapshot_states(event.timestamp, event.event_type)
            return None

        # ---- [EXTENSION] auto-approve ignores EXOGENOUS VEHICLE POSITIONS ---------------
        # A GPS_UPDATE describes where the vehicle is on the REAL timeline.  Under
        # auto-approve the simulated timeline is the dispatcher's own, so applying the real
        # position teleports the simulated vehicle for free: the displacement costs no
        # travel time and no distance, which silently removes work from total_travel_time,
        # working_time and hence fairness_gap.  The same argument that filters *_COMPLETE
        # applies verbatim here, so we keep the trigger but drop the coordinates.
        applied_event = event
        if self.auto_approve and event.event_type == "GPS_UPDATE" \
                and (event.payload.get("x") is not None or event.payload.get("y") is not None):
            payload = {k: v for k, v in event.payload.items() if k not in ("x", "y")}
            applied_event = replace(event, payload=payload)
            LOG.debug("auto-approve: dropped exogenous GPS position for %s @%s",
                      event.vehicle_id, hhmm(event.timestamp))

        result: EventResult = self.env.apply_event(applied_event)
        decision: TriggerDecision = self.trigger.evaluate(result)

        self.event_log.append(result.as_log_row())
        self.replan_log.append({**decision.as_log_row(event.timestamp),
                                "clock": hhmm(event.timestamp),
                                "event_type": event.event_type,
                                "vehicle_id": event.vehicle_id or "",
                                "order_id": event.order_id or ""})
        if event.event_type == "TRAFFIC_UPDATE":
            self.traffic_history.append(dict(timestamp=event.timestamp,
                                             clock=hhmm(event.timestamp),
                                             factor=event.payload.get("factor", 1.0),
                                             area=event.payload.get("area", "ALL"),
                                             relative_change=result.info.get("relative_change", 0.0)))
        self._snapshot_states(event.timestamp, event.event_type)

        if not decision.replan:
            return None

        # The horizon also caps the idle fast-forward inside env.observe(), so the fleet
        # cannot idle past an event that has not been delivered yet.  Always restored.
        previous_ceiling = self.env.clock_horizon
        self.env.clock_horizon = horizon
        try:
            first = self.replan(event, decision.reason, t_received, horizon=horizon)
            self.last_cascade = [first]

            # ---- [EXTENSION] cascade: an executed decision frees the vehicle at once ----
            # NOTE: `replan()` observes internally.  Calling `env.observe()` here as well
            # would run `_advance_idle_clock()` twice and could jump every available
            # vehicle's clock to `next_release_time` twice — so we must NOT do it.
            if self.auto_approve and first.executed:
                limit = max(1, self.cfg.inference.auto_approve_max_cascade)
                for i in range(1, limit):
                    d = self.replan(None, f"{decision.reason}:cascade[{i}]",
                                    cascade_index=i, horizon=horizon)
                    d.cascade_index = i
                    self.last_cascade.append(d)
                    if not d.executed:
                        break
                else:
                    # never let a truncated cascade look like a completed one: without this
                    # the last row still reads AUTO_APPROVED and the CSV is indistinguishable
                    LOG.warning("cascade for %s hit auto_approve_max_cascade=%d; truncated",
                                decision.reason, limit)
                    self.last_cascade[-1].status += ":CASCADE_TRUNCATED"
        finally:
            self.env.clock_horizon = previous_ceiling
        return first

    # ------------------------------------------------------------------ replanning
    def replan(self, event: Optional[Event], reason: str,
               t_received: Optional[float] = None,
               cascade_index: int = 0,
               horizon: Optional[float] = None) -> DispatchDecision:
        """1 snapshot -> 2 MCDSM -> 3 pretrained Actor -> 4 recommendation -> 5 validate."""
        t_received = t_received or time.perf_counter()
        cfg = self.cfg

        # ---- 1/2 snapshot + MCDSM ------------------------------------------------------
        obs = self.env.observe()
        # NOTE: read the clock AFTER observe().  `_advance_idle_clock()` may have moved it,
        # and stamping the pre-observe value would mis-date every cascade decision.
        now = event.timestamp if event else self.env.state.now
        self.last_obs = obs
        self.last_mask = obs.action_mask.copy()
        self.last_probs = None
        base = DispatchDecision(event_timestamp=now, clock=hhmm(now), replan_reason=reason,
                                decode_mode=cfg.inference.decode_mode,
                                k_tilde="|".join(obs.mcdsm.k_tilde),
                                n_active_orders=len(self.env.state.active_order_ids),
                                cascade_index=cascade_index)
        # diagnostics of THIS replan (mutated in place below once the Actor has run) so a
        # whole cascade can be transcribed, not only the last cycle
        diag: Dict[str, Any] = {"obs": obs, "mcdsm": obs.mcdsm,
                                "probs": None, "mask": self.last_mask}
        self.last_cascade_diag.append(diag)
        if not obs.has_action:
            base.status = "NO_FEASIBLE_ACTION"
            base.feasible = False
            base.validator_notes = obs.mcdsm.reason
            base.decision_latency_ms = (time.perf_counter() - t_received) * 1000.0
            self.latencies_ms.append(base.decision_latency_ms)
            self.dispatch_log.append(base.as_row())
            self.last_decision = base
            return base

        # ---- [EXTENSION] horizon gate --------------------------------------------------
        # Gate on the DEPARTURE INSTANT the leg would actually use — the same quantity
        # `candidate_locations` derives, `max(vehicle.current_time, state.now)`.
        # Gating on `vehicle.current_time` alone is not enough: that clock can lag behind
        # `state.now` (a vehicle idle since the start of the shift), so a stale-clock vehicle
        # would slip through the gate and then execute a leg that departs AFTER the next
        # exogenous event — running past its own breakdown, for instance.  Gating on
        # `state.now` alone is equally wrong: it is the minimum over ALL available vehicles,
        # so an available vehicle with no feasible order pins it at a stale value.
        # `executed` stays False, so the cascade loop ends here.
        departure = max(obs.vehicle.current_time, self.env.state.now)
        if horizon is not None and departure > horizon + 1e-9:
            base.status = "DEFERRED_TO_NEXT_EVENT"
            base.feasible = False
            base.vehicle_id = obs.vehicle.vehicle_id
            base.validator_notes = (f"departure {hhmm(departure)} is past "
                                    f"the next event at {hhmm(horizon)}")
            base.decision_latency_ms = (time.perf_counter() - t_received) * 1000.0
            self.latencies_ms.append(base.decision_latency_ms)
            self.dispatch_log.append(base.as_row())
            self.last_decision = base
            return base

        vehicle = obs.vehicle
        mask = obs.action_mask.copy()

        # ---- 3/5 Actor + deterministic feasibility validator ---------------------------
        t_inf = time.perf_counter()
        notes = ""
        chosen_slot, prob = None, 0.0
        for attempt in range(cfg.inference.max_reselect_attempts):
            if mask.sum() <= 0:
                break
            res = self.agent.act(obs.state, mask, mode=cfg.inference.decode_mode,
                                 best_of_n=cfg.inference.best_of_n)
            order = self.env._order_at_slot(res.slot)
            ok, msg = (True, "validator disabled")
            if cfg.inference.run_feasibility_validator:
                ok, msg = self.validator.check(vehicle, order)
            self.last_probs = res.probs
            self.last_mask = mask.copy()
            if ok:
                chosen_slot, prob, notes = res.slot, res.prob, msg
                break
            # invalid action from a numerical / model error -> mask it and reselect
            LOG.warning("rejected action slot=%s (%s); masking and reselecting", res.slot, msg)
            notes = f"reselect[{attempt}]: {msg}"
            mask[res.slot] = 0.0
        inference_ms = (time.perf_counter() - t_inf) * 1000.0
        diag["probs"] = self.last_probs
        diag["mask"] = self.last_mask

        if chosen_slot is None:
            base.status = "NO_VALID_ACTION_AFTER_RESELECT"
            base.feasible = False
            base.validator_notes = notes
            base.inference_latency_ms = inference_ms
            base.decision_latency_ms = (time.perf_counter() - t_received) * 1000.0
            self.latencies_ms.append(base.decision_latency_ms)
            self.dispatch_log.append(base.as_row())
            self.last_decision = base
            return base

        order = self.env._order_at_slot(chosen_slot)
        operation = order.next_operation or ""   # [FEDEX EXTENSION] resolved from the plan
        cand, rule = self.env.state.select_service_location(vehicle, order)
        # [FIX] the recommended ETA is `Candidate.arrival_time` — the SINGLE source of truth,
        # derived from max(vehicle.current_time, state.now).  Re-adding the travel time to
        # the vehicle clock here used to publish arrival times EARLIER than the event that
        # triggered the recommendation whenever the vehicle had been idle since the shift start.
        arrival = cand.arrival_time
        completion = arrival + cand.waiting_time + cand.service_duration

        base.vehicle_id = vehicle.vehicle_id
        base.order_id = order.order_id
        base.operation = operation
        base.flow_type = order.flow_type
        base.stage = order.stage
        base.service_location_id = cand.location.service_location_id
        base.location_x, base.location_y = cand.location.x, cand.location.y
        base.expected_travel_time = cand.travel_time
        base.expected_waiting_time = cand.waiting_time
        base.expected_arrival_time = arrival
        base.expected_completion_time = completion
        base.reward_estimate = -(cand.travel_time + cand.waiting_time) * cfg.reward.reward_scale
        base.policy_probability = prob
        base.inference_latency_ms = inference_ms
        base.validator_notes = f"{notes} | location_rule={rule}"
        # measured BEFORE any execution: this is the decision latency of the dispatcher and
        # stays directly comparable to the open-loop run.
        base.decision_latency_ms = (time.perf_counter() - t_received) * 1000.0

        # ---- [EXTENSION] commitment + (optionally) immediate execution ------------------
        # Committing in BOTH modes is the point:
        #   auto_approve = False -> the commitment survives until the real *_COMPLETE event
        #                           arrives, so no other vehicle can be offered this order
        #                           in the meantime  (the deployment case);
        #   auto_approve = True  -> on_service_complete clears it in the same breath, which
        #                           is why the lock is a no-op for closed-loop runs.
        if cfg.mcdsm.lock_committed_orders:
            # A vehicle can only be on its way to ONE order at a time, so a fresh
            # recommendation supersedes whatever this vehicle was committed to before.
            # Without this the commitments accumulate and every superseded order stays
            # locked to a vehicle that is no longer going there.  (Under auto-approve the
            # list is always empty — execution clears the commitment immediately.)
            for stale in self.env.state.committed_orders_of(vehicle.vehicle_id):
                if stale != order.order_id:
                    self.env.state.release_commitment(
                        stale, "SUPERSEDED_BY_NEW_RECOMMENDATION")
            self.env.state.commit_order(order.order_id, vehicle.vehicle_id)
        if self.auto_approve:
            # env.step() reads env._last_obs, which the self.env.observe() call at the top
            # of this method already set -- no second observe() is needed (or allowed).
            _, reward, done, info = self.env.step(chosen_slot)
            base.executed = not info.invalid
            base.realized_reward = reward
            base.actual_completion_time = info.completion_time
            base.status = ("AUTO_APPROVED" if base.executed
                           else f"EXECUTION_FAILED:{info.message}")
            if not base.executed and cfg.mcdsm.lock_committed_orders:
                # never leave an exclusive lock behind for an operation that did not happen
                self.env.state.release_commitment(order.order_id, "EXECUTION_FAILED")

        self.latencies_ms.append(base.decision_latency_ms)
        self.dispatch_log.append(base.as_row())
        self.last_decision = base
        if self.on_decision:
            self.on_decision(base)
        return base

    # ------------------------------------------------------------------ helpers
    def _snapshot_states(self, timestamp: float, event_type: str) -> None:
        for v in self.env.state.vehicle_list:
            self.vehicle_history.append(dict(timestamp=timestamp, clock=hhmm(timestamp),
                                             event_type=event_type, **v.snapshot()))
        for oid in self.env.state.active_order_ids:
            o = self.env.state.orders[oid]
            self.order_history.append(dict(timestamp=timestamp, clock=hhmm(timestamp),
                                           event_type=event_type, **o.snapshot()))

    def apply_confirmation(self, event: Event) -> Optional[DispatchDecision]:
        """Alias kept for readability: PICKUP/DELIVERY confirmations arrive as events too."""
        return self.process_event(event)

    def drain(self, reason: str = "DRAIN",
              max_decisions: Optional[int] = None) -> List[DispatchDecision]:
        """[EXTENSION] keep deciding until no vehicle has a feasible action left.

        Only meaningful with auto-approve: an open-loop recommendation does not change the
        state, so draining would loop forever.  It is therefore a deliberate no-op when
        `inference.auto_approve` is False, which keeps MODE B deployment behaviour intact.
        """
        out: List[DispatchDecision] = []
        if not self.auto_approve:
            return out
        self.last_cascade_diag = []
        limit = (max_decisions if max_decisions is not None
                 else max(self.cfg.inference.auto_approve_max_cascade,
                          4 * len(self.env.state.orders) + 8))
        for i in range(limit):
            d = self.replan(None, f"{reason}[{i}]", cascade_index=i)
            d.cascade_index = i
            out.append(d)
            if not d.executed:
                break
        else:
            LOG.warning("drain(%s) hit its safety bound of %d decisions", reason, limit)
        return out

    def run_replay(self, events: List[Event]) -> List[DispatchDecision]:
        """Replay a whole event stream as a discrete-event driver (tests + toy demo).

        Sorted ONCE (same `clock.event_priority` tie-break as before).  Every event is
        given the timestamp of its successor as the horizon, so an auto-approved cascade
        cannot run past the next exogenous event.  The stream finishes with a `drain()`,
        without which the last legs of the day would never execute and their orders would
        be reported as unserved.
        """
        ordered = sorted(events, key=lambda x: (
            x.timestamp, self.cfg.clock.event_priority.get(x.event_type, 99)))
        decisions: List[DispatchDecision] = []
        for i, e in enumerate(ordered):
            horizon = ordered[i + 1].timestamp if i + 1 < len(ordered) else None
            self.process_event(e, horizon=horizon)
            decisions.extend(self.last_cascade)
        decisions.extend(self.drain(reason="END_OF_EVENT_STREAM"))
        return decisions

    # ------------------------------------------------------------------ outputs
    def latency_summary(self) -> Dict[str, float]:
        return latency_stats(self.latencies_ms)

    def flush_logs(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        out_dir = ensure_dir(out_dir or self.cfg.inference.output_dir)
        paths = {
            "event_log.csv": self.event_log,
            "replan_log.csv": self.replan_log,
            "dispatch_decisions.csv": self.dispatch_log,
            "vehicle_state_history.csv": self.vehicle_history,
            "order_state_history.csv": self.order_history,
            "traffic_history.csv": self.traffic_history,
            "latency_summary.csv": [self.latency_summary()],
            # [EXTENSION] the experimental result of the run.  Only meaningful once the
            # decisions are actually executed (auto-approve); in open loop the environment
            # never moves, so travel / waiting / fairness stay at their initial values.
            "episode_metrics.csv": [self.env.metrics()],
        }
        written = {}
        for name, rows in paths.items():
            written[name] = write_csv(os.path.join(out_dir, name), rows)
        return written


# =======================================================================================
# Adapters / placeholders — no company-specific API is required.
# =======================================================================================
class BaseAdapter:
    """Map a raw external record onto an `Event`.  Subclass and override `to_event`."""
    source = "EXTERNAL"

    def to_event(self, record: Dict[str, Any]) -> Event:      # pragma: no cover - interface
        raise NotImplementedError

    def feed(self, records: List[Dict[str, Any]], dispatcher: RealTimeDispatcher
             ) -> List[DispatchDecision]:
        out = []
        for rec in records:
            d = dispatcher.process_event(self.to_event(rec))
            if d:
                out.append(d)
        return out


class ERPOrderAdapter(BaseAdapter):
    """ERP / OMS -> NEW_ORDER, CANCEL_ORDER, ORDER_PRIORITY_UPDATE, TIME_WINDOW_UPDATE."""
    source = "ERP"

    def to_event(self, record: Dict[str, Any]) -> Event:
        from utils import parse_clock
        return Event(timestamp=parse_clock(record.get("timestamp")),
                     event_type=str(record.get("event_type", "NEW_ORDER")).upper(),
                     order_id=record.get("order_id"),
                     payload={k: v for k, v in record.items()
                              if k not in ("timestamp", "event_type", "order_id")},
                     source=self.source)


class TMSVehicleAdapter(BaseAdapter):
    """TMS / GPS -> GPS_UPDATE, PICKUP_COMPLETE, DELIVERY_COMPLETE, BREAKDOWN, SHIFT_*."""
    source = "TMS"

    def to_event(self, record: Dict[str, Any]) -> Event:
        from utils import parse_clock
        return Event(timestamp=parse_clock(record.get("timestamp")),
                     event_type=str(record.get("event_type", "GPS_UPDATE")).upper(),
                     vehicle_id=record.get("vehicle_id"),
                     order_id=record.get("order_id"),
                     service_location_id=record.get("service_location_id"),
                     payload={k: v for k, v in record.items()
                              if k not in ("timestamp", "event_type", "vehicle_id",
                                           "order_id", "service_location_id")},
                     source=self.source)


class TrafficAdapter(BaseAdapter):
    """Traffic API -> TRAFFIC_UPDATE (multiplicative predicted-travel-time factor)."""
    source = "TRAFFIC_API"

    def to_event(self, record: Dict[str, Any]) -> Event:
        from utils import parse_clock
        return Event(timestamp=parse_clock(record.get("timestamp")),
                     event_type="TRAFFIC_UPDATE",
                     payload={"factor": float(record.get("factor", 1.0)),
                              "area": record.get("area", "ALL")},
                     source=self.source)
