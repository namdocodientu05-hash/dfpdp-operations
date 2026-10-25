"""
state_manager.py
================
Builds the paper state

        s_t^k = ( Z_t , U_t^k , D_t^k )                                        [PAPER]

    Z_t   : status of EVERY currently released order,   z_m in {0,1,2}
    U_t^k : cargo-weight / loading-unloading information of every active order w.r.t. vehicle k
    D_t^k : distance / travel information from the CURRENT location of vehicle k to the
            NEXT relevant service location of every order
                z_m = 0  ->  pickup  service locations
                z_m = 1  ->  delivery service locations
                z_m = 2  ->  masked / zero

and keeps them up to date under real-time events                    [REAL-TIME EXTENSION].

The state of ONE vehicle always contains information about ALL currently released orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from entities import (Order, ProblemInstance, ServiceLocation, Vehicle, euclidean,
                      ROLE_DELIVERY, ROLE_DEPOT, ROLE_PICKUP, VEHICLE_BROKEN, VEHICLE_IDLE,
                      Z_COMPLETED, Z_WAIT_DELIVERY, Z_WAIT_PICKUP,
                      CARGO_NOT_LOADED, CARGO_ON_VEHICLE,
                      FLOW_INBOUND_PRELOADED, OPERATION_ROLE, OPERATION_SEMANTICS, OPERATIONS,
                      OP_CUSTOMER_DELIVERY, OP_CUSTOMER_PICKUP,
                      OP_DEPOT_LOAD, OP_DEPOT_UNLOAD)
from utils import get_logger
from variable_length import SlotTable

LOG = get_logger("dfpdp.state")


# =======================================================================================
# Traffic  [REAL-TIME EXTENSION]
# =======================================================================================
class TrafficModel:
    """Multiplicative congestion factors maintained by TRAFFIC_UPDATE events.

    factor = 1.0 -> free flow ; factor = 1.3 -> travel takes 30 % longer.
    Both representations are kept alive at the same time so that experiments can compare
        BASELINE  : Euclidean distance / speed                            [PAPER]
        EXTENSION : predicted travel time including live congestion       [REAL-TIME]
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.global_factor: float = float(cfg.travel.global_traffic_factor)
        self.area_factors: Dict[str, float] = {}
        self.history: List[Tuple[float, str, float]] = []

    def update(self, timestamp: float, factor: float, area: str = "ALL") -> float:
        """Returns the *relative* change of the predicted travel time (used by the trigger)."""
        area = str(area or "ALL").upper()
        old = self.factor(area)
        if area == "ALL":
            self.global_factor = float(factor)
        else:
            self.area_factors[area] = float(factor)
        self.history.append((timestamp, area, float(factor)))
        return abs(self.factor(area) - old) / max(old, 1e-9)

    def factor(self, area: str = "ALL") -> float:
        area = str(area or "ALL").upper()
        return float(self.area_factors.get(area, self.global_factor))


# =======================================================================================
# Travel time
# =======================================================================================
class TravelModel:
    def __init__(self, cfg: Config, traffic: TrafficModel):
        self.cfg = cfg
        self.traffic = traffic

    def distance(self, a, b) -> float:
        return euclidean(a, b)

    def travel_time(self, vehicle: Vehicle, a, b, t: Optional[float] = None,
                    area: str = "ALL") -> float:
        t = vehicle.current_time if t is None else t
        dist = self.distance(a, b)
        speed = vehicle.speed_at(t, use_profile=self.cfg.travel.use_speed_profile)
        base = dist / speed
        if self.cfg.travel.distance_mode == "traffic":       # [REAL-TIME EXTENSION]
            base *= self.traffic.factor(area)
        return base

    def baseline_travel_time(self, vehicle: Vehicle, a, b) -> float:
        """[PAPER] congestion-free reference used for A/B comparison of the two modes."""
        return self.distance(a, b) / max(vehicle.base_speed, 1e-6)


# =======================================================================================
# Candidate service location
# =======================================================================================
@dataclass
class Candidate:
    location: ServiceLocation
    distance: float
    travel_time: float
    arrival_time: float
    waiting_time: float
    lateness: float
    feasible: bool
    same_location: bool
    service_duration: float

    @property
    def cost(self) -> float:
        return self.travel_time + self.waiting_time

    def as_dict(self) -> Dict[str, Any]:
        return dict(service_location_id=self.location.service_location_id,
                    x=self.location.x, y=self.location.y,
                    distance=self.distance, travel_time=self.travel_time,
                    arrival_time=self.arrival_time, waiting_time=self.waiting_time,
                    lateness=self.lateness, feasible=self.feasible,
                    service_duration=self.service_duration)


# =======================================================================================
# State manager
# =======================================================================================
class StateManager:
    """Owns the live picture of the world: orders, vehicles, slots, traffic."""

    def __init__(self, instance: ProblemInstance, cfg: Config):
        self.cfg = cfg
        self.instance = instance
        self.traffic = TrafficModel(cfg)
        self.travel = TravelModel(cfg, self.traffic)
        self.slots = SlotTable(cfg.problem.M_max, cfg.problem.order_overflow_policy)
        self.now: float = cfg.clock.day_start
        self.orders: Dict[str, Order] = {o.order_id: o for o in instance.orders}
        self.vehicles: Dict[str, Vehicle] = {v.vehicle_id: v for v in instance.vehicles}
        self.vehicle_list: List[Vehicle] = list(instance.vehicles)
        self.active_order_ids: List[str] = []
        self.unreleased_order_ids: List[str] = []
        self.released_history: List[Tuple[float, str]] = []
        self.visibility_time: Optional[float] = None
        self._depot_cache: Dict[str, ServiceLocation] = {}

    # ------------------------------------------------------------------ initialisation
    def reset(self) -> None:
        """[§9] start of day / start of shift."""
        cfg = self.cfg
        self.now = cfg.clock.day_start
        self.traffic = TrafficModel(cfg)
        self.travel = TravelModel(cfg, self.traffic)
        self.slots = SlotTable(cfg.problem.M_max, cfg.problem.order_overflow_policy)
        self.active_order_ids, self.unreleased_order_ids, self.released_history = [], [], []
        # [EXTENSION] planning-horizon of the §41 comparison baselines (None = event-driven).
        # Owned by the state, not by the mask, so an order the planner is defined not to know
        # about is invisible to Z/U/D as well.  `MCDSM.visibility_time` proxies to this.
        self.visibility_time: Optional[float] = None

        for v in self.vehicle_list:
            v.load = 0.0
            v.current_time = max(v.shift_start, cfg.clock.day_start)
            v.status = VEHICLE_IDLE
            v.working_time = v.travel_time = v.waiting_time = v.distance = 0.0
            v.n_services = 0
            v.carrying = []
            v.route = []
            v.observed_speed = None
            depot = self._depot_location(v)
            if depot is not None:
                v.x, v.y = depot.x, depot.y
                v.last_service_location_id = depot.service_location_id

        for o in self.instance.orders:
            o.reset_plan()                    # [FEDEX EXTENSION] stage_index = 0, cargo unloaded
            o.released = False
            o.cancelled = False
            o.late = False
            o.lateness = 0.0
            o.index = -1
            if o.order_class == "STATIC" or o.release_time <= self.now:
                self.release_order(o, self.now)             # expose all STATIC orders
            else:
                self.unreleased_order_ids.append(o.order_id)   # hide DYNAMIC orders

        # [FEDEX EXTENSION] INBOUND_PRELOADED: the cargo is ALREADY on the vehicle when the
        # shift starts.  This runs AFTER the load-zeroing loop above on purpose, otherwise
        # the preloaded weight would be silently wiped out.
        self._init_preloaded_cargo()

    def _load_preloaded_order(self, o: Order, enforce_capacity: bool = False) -> bool:
        """[FEDEX EXTENSION] put an INBOUND_PRELOADED order ON its vehicle.

        Shared by the shift-start initialisation and by the run-time NEW_ORDER path, so the
        two can never drift apart.  Returns False when the order CANNOT be preloaded — the
        caller must then reject it explicitly, because an INBOUND_PRELOADED order whose
        cargo is not on a vehicle is dead on arrival: its only operation is
        CUSTOMER_DELIVERY (load_sign = -1) and `MCDSM.order_feasible` masks an unloading
        operation out for EVERY vehicle while `cargo_on_vehicle` is False.
        """
        vid = o.preloaded_vehicle_id
        if not vid or vid not in self.vehicles:
            LOG.error("order %s is INBOUND_PRELOADED but names no valid vehicle (%r)",
                      o.order_id, vid)
            return False
        v = self.vehicles[vid]
        if enforce_capacity and o.cargo_weight > v.available_capacity + 1e-9:
            LOG.error("order %s cannot be preloaded on %s: w=%.1f > Q-L=%.1f",
                      o.order_id, vid, o.cargo_weight, v.available_capacity)
            return False
        o.mark_preloaded(vid)
        if o.order_id not in v.carrying:
            v.carrying.append(o.order_id)
            v.load += o.cargo_weight
        if v.load > v.capacity + 1e-9:
            LOG.error("vehicle %s starts overloaded: %.1f > %.1f",
                      vid, v.load, v.capacity)
        return True

    def _init_preloaded_cargo(self) -> None:
        for o in self.instance.orders:
            if o.flow_type != FLOW_INBOUND_PRELOADED or o.cancelled:
                continue
            if not self._load_preloaded_order(o):
                # [FIX] do NOT leave the order alive-but-unservable.  It used to keep
                # cargo_state=NOT_LOADED with next_operation=CUSTOMER_DELIVERY, which is
                # masked out for every vehicle from t=0 and can never recover; it then
                # showed up as an "unserved order" nobody could ever have served.
                o.cancelled = True
                LOG.error("order %s cancelled at initialisation: INBOUND_PRELOADED cargo "
                          "could not be placed on a vehicle", o.order_id)

    def _depot_location(self, vehicle: Vehicle) -> Optional[ServiceLocation]:
        if vehicle.depot_id in self._depot_cache:
            return self._depot_cache[vehicle.depot_id]
        locs = self.instance.service_locations.get(vehicle.depot_id, [])
        depot = locs[0] if locs else None
        if depot:
            self._depot_cache[vehicle.depot_id] = depot
        return depot

    # ------------------------------------------------------------------ order release
    def release_order(self, order: Order, timestamp: float) -> Optional[int]:
        """[§10] make a (dynamic) order visible to MCDSM / Actor / Z,U,D."""
        if order.released:
            return order.index
        recyclable = [oid for oid in self.active_order_ids
                      if self.orders[oid].is_completed or self.orders[oid].cancelled]
        slot = self.slots.assign(order.order_id, recyclable=recyclable)
        if slot is None:                       # overflow -> "queue" policy, retry later
            LOG.warning("order %s queued: no free slot (M_max=%d)",
                        order.order_id, self.cfg.problem.M_max)
            return None
        if self.slots.slots[slot] == order.order_id:
            for oid in list(self.active_order_ids):
                if self.orders[oid].index == slot and oid != order.order_id:
                    self.active_order_ids.remove(oid)       # recycled slot
        order.index = slot
        order.released = True
        if order.order_id not in self.active_order_ids:
            self.active_order_ids.append(order.order_id)
        if order.order_id in self.unreleased_order_ids:
            self.unreleased_order_ids.remove(order.order_id)
        self.released_history.append((timestamp, order.order_id))
        return slot

    def release_due_orders(self, timestamp: float) -> List[str]:
        released = []
        for oid in list(self.unreleased_order_ids):
            order = self.orders[oid]
            if order.release_time <= timestamp + 1e-9:
                if self.release_order(order, timestamp) is not None:
                    released.append(oid)
        # retry queued (overflow) orders
        for oid in list(self.slots.backlog):
            if self.slots.assign(oid, recyclable=[o for o in self.active_order_ids
                                                  if self.orders[o].is_completed]) is not None:
                self.slots.backlog.remove(oid)
                self.release_order(self.orders[oid], timestamp)
                released.append(oid)
        return released

    @property
    def next_release_time(self) -> Optional[float]:
        times = [self.orders[o].release_time for o in self.unreleased_order_ids]
        return min(times) if times else None

    # ------------------------------------------------------------------ commitments
    # [EXTENSION] An ACCEPTED dispatch decision reserves the order for one vehicle even
    # before the cargo physically moves.  Offline this window has zero length (the
    # commitment and the execution happen inside the same env.step()), so nothing changes;
    # online it is what stops the next event from re-offering the same order to another
    # vehicle.  Every release path is explicit — a commitment is NEVER dropped silently.
    def commit_order(self, order_id: str, vehicle_id: str) -> Optional[str]:
        """Record an accepted assignment.  Returns the previous commitment, if any."""
        o = self.orders.get(order_id)
        if o is None:
            LOG.error("commit_order: unknown order %s", order_id)
            return None
        previous = o.committed_vehicle_id
        if previous is not None and previous != vehicle_id:
            LOG.warning("order %s was committed to %s, re-committing to %s",
                        order_id, previous, vehicle_id)
        o.committed_vehicle_id = vehicle_id
        return previous

    def release_commitment(self, order_id: str, reason: str = "") -> Optional[str]:
        """Drop the accepted assignment of ONE order.  Returns the released vehicle id."""
        o = self.orders.get(order_id)
        if o is None or o.committed_vehicle_id is None:
            return None
        released = o.committed_vehicle_id
        o.committed_vehicle_id = None
        LOG.debug("released commitment %s -> %s (%s)", order_id, released, reason)
        return released

    def release_commitments_of(self, vehicle_id: str, reason: str = "") -> List[str]:
        """Drop every accepted assignment held by one vehicle.  Returns the order ids."""
        released = [o.order_id for o in self.orders.values()
                    if o.committed_vehicle_id == vehicle_id]
        for oid in released:
            self.orders[oid].committed_vehicle_id = None
        if released:
            LOG.info("released %d commitment(s) of %s (%s): %s",
                     len(released), vehicle_id, reason, released)
        return released

    def committed_orders_of(self, vehicle_id: str) -> List[str]:
        return [o.order_id for o in self.orders.values()
                if o.committed_vehicle_id == vehicle_id]

    # ------------------------------------------------------------------ visibility
    @property
    def visible_order_ids(self) -> List[str]:
        """Orders the planner is allowed to KNOW ABOUT right now.

        [EXTENSION] `visibility_time` used to filter the MCDSM mask only, so a masked-out
        order was still described in Z/U/D — the encoder attended over it, the critic pooled
        it and U[i,3] advertised it as capacity-feasible.  Baselines A_static / B_periodic
        were therefore not information-blind, which biased the §41 comparison in their
        favour.  Filtering here makes the horizon a property of the state itself.

        `visibility_time is None` (mode C, training, MODE B) returns the very same list
        object, so the tensors are byte-identical to the pre-extension build.
        """
        if self.visibility_time is None:
            return self.active_order_ids
        t = self.visibility_time
        return [oid for oid in self.active_order_ids
                if self.orders[oid].cargo_on_vehicle
                or self.orders[oid].release_time <= t + 1e-9]

    def lock_owner(self, order: Order) -> Optional[str]:
        """The ONE vehicle allowed to perform this order's next operation, or None.

        Single source of truth for the exclusivity rule — the MCDSM mask and the U features
        must never disagree, so both call this instead of reading the fields directly.

        Physical ownership (cargo on board) is absolute.  A mere COMMITMENT, by contrast,
        is only as good as the vehicle holding it: if that vehicle is no longer available
        (broken, or simply past its shift end with no SHIFT_END event ever sent), the
        commitment is void and the order goes back to the pool.  Without this the order
        would be masked out for every vehicle and stranded for the rest of the day.
        """
        if order.cargo_on_vehicle and order.assigned_vehicle_id is not None:
            return order.assigned_vehicle_id
        if not self.cfg.mcdsm.lock_committed_orders:
            return None
        holder = order.committed_vehicle_id
        if holder is None:
            return None
        v = self.vehicles.get(holder)
        if v is None or not v.is_available(self.now):
            return None
        return holder

    # ------------------------------------------------------------------ candidates
    def candidate_locations(self, vehicle: Vehicle, order: Order,
                            role: Optional[str] = None) -> List[Candidate]:
        """All physical service locations required by the order's NEXT operation.

        [FEDEX EXTENSION] the role/entity now come from the order's own service plan:
            CUSTOMER_PICKUP   -> consignor PICKUP locations
            CUSTOMER_DELIVERY -> consignee DELIVERY locations
            DEPOT_LOAD        -> source depot      (or the vehicle's depot)
            DEPOT_UNLOAD      -> destination depot (or the vehicle's depot)
        """
        role = role or order.target_role()
        entity = order.target_entity(vehicle)
        locs = [sl for sl in self.instance.service_locations.get(entity, [])
                if sl.entity_role == role and sl.available][: self.cfg.problem.S_max]
        out: List[Candidate] = []
        # [FIX] DEPARTURE INSTANT.  A vehicle cannot leave before it is free (its own
        # accumulated clock) and it cannot leave before the present moment either.  Using
        # `vehicle.current_time` alone let an idle vehicle whose clock still sat at the
        # start of the shift "depart in the past" when a real-time event arrived hours
        # later, which produced ETAs EARLIER than the event that triggered the planning.
        # The departure instant is DERIVED here and never written back onto the vehicle:
        # clamping `vehicle.current_time` would flatten the MCDSM `argmin current_time`
        # fairness criterion (every idle vehicle would tie at `now`).
        depart = max(vehicle.current_time, self.now)
        for sl in locs:
            # the departure instant also drives the hour-of-day speed lookup (`speed_at`)
            tt = self.travel.travel_time(vehicle, vehicle.xy, sl.xy, depart)
            same = (vehicle.last_service_location_id == sl.service_location_id
                    or (abs(vehicle.x - sl.x) < 1e-9 and abs(vehicle.y - sl.y) < 1e-9))
            if same and self.cfg.reward.zero_movement_rule:
                tt = 0.0                                   # [PAPER] zero movement rule
            arrival = depart + tt
            out.append(Candidate(
                location=sl,
                distance=self.travel.distance(vehicle.xy, sl.xy),
                travel_time=tt,
                arrival_time=arrival,
                waiting_time=sl.waiting_time(arrival),
                lateness=sl.lateness(arrival),
                feasible=sl.time_compatible(arrival),
                same_location=same,
                service_duration=sl.service_duration,
            ))
        return out

    def select_service_location(self, vehicle: Vehicle, order: Order
                                ) -> Tuple[Optional[Candidate], str]:
        """[§27/§28] choose the physical service location; returns (candidate, rule_used)."""
        cands = self.candidate_locations(vehicle, order)
        if not cands:
            return None, "no_location"
        feasible = [c for c in cands if c.feasible]
        rule = self.cfg.service_location.selection_rule
        if feasible:
            if rule == "min_travel_plus_wait":             # [REAL-TIME EXTENSION]
                return min(feasible, key=lambda c: c.cost), "min_travel_plus_wait"
            return feasible[0], "paper_time_compatible"    # [PAPER-like] first compatible one
        # ---- no feasible time window: explicit, configured handling --------------------
        handling = self.cfg.service_location.infeasibility_rule
        if handling == "allow_late_logged":
            return min(cands, key=lambda c: c.lateness), "allow_late_logged"
        if handling == "defer_order":
            return None, "defer_order"
        if handling == "drop_order":
            return None, "drop_order"
        raise ValueError(f"unknown infeasibility_rule {handling}")

    def order_is_servable(self, vehicle: Vehicle, order: Order) -> bool:
        """Time-window pre-check used by the MCDSM mask [REAL-TIME EXTENSION, switchable]."""
        if not self.cfg.mcdsm.include_time_window_feasibility_in_mask:
            return True
        if self.cfg.service_location.infeasibility_rule == "allow_late_logged":
            return bool(self.candidate_locations(vehicle, order))
        cand, _ = self.select_service_location(vehicle, order)
        return cand is not None

    # ------------------------------------------------------------------ Z / U / D
    def build_Z(self) -> np.ndarray:
        """[PAPER] Z_t = [z_t^1 ... z_t^M] (padded, global — identical for every vehicle)."""
        cfg = self.cfg
        Z = np.zeros((cfg.problem.M_max, cfg.state.z_dim), dtype=np.float32)
        for oid in self.visible_order_ids:
            o = self.orders[oid]
            i = o.index
            if i is None or i < 0:
                continue
            Z[i, 0] = 1.0 if o.z == Z_WAIT_PICKUP else 0.0
            Z[i, 1] = 1.0 if o.z == Z_WAIT_DELIVERY else 0.0
            Z[i, 2] = 1.0 if o.z == Z_COMPLETED else 0.0
            Z[i, 3] = 1.0 if o.released else 0.0
            Z[i, 4] = 1.0 if o.order_class == "DYNAMIC" else 0.0
            Z[i, 5] = (o.priority / 5.0) if cfg.state.use_priority_feature else 0.0
            Z[i, 6] = 1.0 if o.cancelled else 0.0
            # ---- [FEDEX EXTENSION] flow-agnostic description of the NEXT operation ------
            # Deliberately NOT a one-hot of flow_type: these five numbers describe what the
            # operation DOES, so the policy transfers what it learns between flows and a
            # future flow needs no new feature dimension.
            Z[i, 7] = o.load_sign                    # +1 loads cargo, -1 unloads, 0 done
            Z[i, 8] = o.next_is_depot                # 1 depot operation, 0 customer operation
            Z[i, 9] = o.stage_progress               # how far along the plan the order is
            Z[i, 10] = 1.0 if o.is_final_stage else 0.0
            Z[i, 11] = 1.0 if o.flow_type == FLOW_INBOUND_PRELOADED else 0.0
        return Z

    def build_U(self, vehicle: Vehicle) -> np.ndarray:
        """[PAPER] cargo-weight / loading-unloading information of every active order."""
        cfg = self.cfg
        U = np.zeros((cfg.problem.M_max, cfg.state.u_dim), dtype=np.float32)
        for oid in self.visible_order_ids:
            o = self.orders[oid]
            i = o.index
            if i is None or i < 0 or o.cancelled:
                continue
            w = o.cargo_weight
            rem = vehicle.available_capacity
            # ---- [FEDEX EXTENSION] ONE signed formula replaces the pickup/delivery branch.
            # load_delta = load_sign * w covers all four operations, so CUSTOMER_DELIVERY
            # and DEPOT_UNLOAD are literally the same computation for the network.
            delta = o.load_delta
            load_after = max(0.0, vehicle.load + delta)
            # [EXTENSION] the exclusive owner of the NEXT operation.  Single source of truth
            # shared with MCDSM.order_feasible, so the mask and these features can never
            # disagree.  With `lock_committed_orders=False` it collapses to the cargo-only
            # rule, i.e. exactly the pre-extension behaviour.
            owner = self.lock_owner(o)
            if delta > 0:                       # loading -> capacity is the binding constraint
                feasible = 1.0 if w <= rem + 1e-9 else 0.0
            else:                               # unloading -> ownership is the binding constraint
                feasible = 1.0 if o.assigned_vehicle_id == vehicle.vehicle_id else 0.0
            if owner is not None and owner != vehicle.vehicle_id:
                feasible = 0.0
            dur = 0.0
            locs = self.instance.service_locations.get(o.target_entity(vehicle), [])
            if locs:
                dur = float(np.mean([l.service_duration for l in locs]))
            U[i, 0] = w / cfg.state.weight_scale
            U[i, 1] = w / max(vehicle.capacity, 1e-6)
            U[i, 2] = rem / cfg.state.weight_scale
            U[i, 3] = feasible
            U[i, 4] = load_after / max(vehicle.capacity, 1e-6)
            U[i, 5] = dur / cfg.state.time_scale
            # "this order belongs to k": carrier while loaded, committee otherwise
            U[i, 6] = 1.0 if (owner == vehicle.vehicle_id
                              or (owner is None
                                  and o.assigned_vehicle_id == vehicle.vehicle_id)) else 0.0
            U[i, 7] = delta / max(vehicle.capacity, 1e-6)        # signed load change
            U[i, 8] = 1.0 if o.order_id in vehicle.carrying else 0.0
            # "this order is locked to SOME vehicle" (physically or by an accepted decision)
            U[i, 9] = 1.0 if owner is not None else 0.0
            if o.is_completed:
                U[i, :] = 0.0
        return U

    def build_D(self, vehicle: Vehicle) -> Tuple[np.ndarray, np.ndarray]:
        """[PAPER] travel information vehicle k -> next relevant service locations.

        Returns (D, service_location_mask).  Completed / cancelled orders are zeroed.
        """
        cfg = self.cfg
        D = np.zeros((cfg.problem.M_max, cfg.state.d_dim), dtype=np.float32)
        sl_mask = np.zeros((cfg.problem.M_max, cfg.problem.S_max), dtype=np.float32)
        horizon = max(cfg.clock.day_end - cfg.clock.day_start, 1.0)
        for oid in self.visible_order_ids:
            o = self.orders[oid]
            i = o.index
            if i is None or i < 0:
                continue
            if o.is_completed or o.cancelled:
                continue                                    # [PAPER] z=2 -> masked / zero
            cands = self.candidate_locations(vehicle, o)
            if not cands:
                continue
            for j, c in enumerate(cands[: cfg.problem.S_max]):
                sl_mask[i, j] = 1.0 if c.feasible else 0.0
            chosen, _ = self.select_service_location(vehicle, o)
            ref = chosen or min(cands, key=lambda c: c.cost)
            D[i, 0] = ref.distance / cfg.state.dist_scale
            D[i, 1] = ref.travel_time / cfg.state.time_scale
            D[i, 2] = ref.waiting_time / cfg.state.time_scale
            D[i, 3] = (ref.arrival_time - cfg.clock.day_start) / horizon
            D[i, 4] = float(sl_mask[i].sum()) / cfg.problem.S_max
            if cfg.state.use_time_window_slack:             # [REAL-TIME EXTENSION]
                # the customer promise only binds the CUSTOMER_DELIVERY leg; depot legs
                # are bound by the depot's own opening window  [FEDEX EXTENSION]
                latest = (o.latest_delivery_time
                          if o.next_operation == OP_CUSTOMER_DELIVERY else ref.location.end_time)
                latest = latest if latest is not None else ref.location.end_time
                D[i, 5] = (latest - ref.arrival_time) / cfg.state.time_scale
            if cfg.state.use_realtime_traffic_state:        # [REAL-TIME EXTENSION] R_t^k
                D[i, 6] = self.traffic.factor("ALL") - 1.0
            D[i, 7] = 1.0 if ref.same_location else 0.0
        return D, sl_mask

    def order_padding_mask(self) -> np.ndarray:
        """1 = slot holds a released order (padding mask, independent of feasibility)."""
        mask = np.zeros(self.cfg.problem.M_max, dtype=np.float32)
        for oid in self.visible_order_ids:
            i = self.orders[oid].index
            if i is not None and i >= 0:
                mask[i] = 1.0
        return mask

    def build_state(self, vehicle: Vehicle) -> Dict[str, np.ndarray]:
        """s_t^k = (Z_t, U_t^k, D_t^k) + masks."""
        D, sl_mask = self.build_D(vehicle)
        return {
            "Z": self.build_Z(),
            "U": self.build_U(vehicle),
            "D": D,
            "order_mask": self.order_padding_mask(),
            "service_location_mask": sl_mask,
            "vehicle_available": np.float32(
                1.0 if (vehicle.is_available() or not self.cfg.state.use_vehicle_availability_flag)
                else 0.0),
        }

    # ------------------------------------------------------------------ event handlers
    # (all of these are also used by the offline replay, not only by the live feed)
    def on_new_order(self, order: Order, timestamp: float) -> Optional[int]:
        """[§12] insert -> z=0 -> update Z, U, D for EVERY vehicle -> update masks."""
        self.orders[order.order_id] = order
        if order not in self.instance.orders:
            self.instance.orders.append(order)
        slot = self.release_order(order, timestamp)
        # [FIX] an INBOUND_PRELOADED order announced at RUN TIME must be placed on its
        # vehicle here.  `_init_preloaded_cargo` only ever runs from `reset()`, so before
        # this the ERP feed produced an order with cargo_state=NOT_LOADED and no carrier
        # whose single operation (CUSTOMER_DELIVERY) was masked out for the whole fleet.
        if (order.flow_type == FLOW_INBOUND_PRELOADED and not order.cancelled
                and not order.cargo_on_vehicle):
            if not self._load_preloaded_order(order, enforce_capacity=True):
                order.cancelled = True
                LOG.error("NEW_ORDER %s rejected: INBOUND_PRELOADED cargo could not be "
                          "placed on vehicle %r", order.order_id,
                          order.preloaded_vehicle_id)
        self.now = max(self.now, timestamp)
        return slot

    def on_cancel_order(self, order_id: str, timestamp: float) -> None:
        o = self.orders[order_id]
        if o.cargo_on_vehicle:      # [FEDEX EXTENSION] true for PD, OUTBOUND and INBOUND
            LOG.warning("cancel of order %s ignored: cargo is on board %s",
                        order_id, o.assigned_vehicle_id)
            return
        o.cancelled = True
        self.release_commitment(order_id, "CANCEL_ORDER")   # [EXTENSION] free the vehicle
        if o.order_id in self.active_order_ids:
            self.active_order_ids.remove(o.order_id)
        self.slots.release(o.order_id)
        o.index = -1
        self.now = max(self.now, timestamp)

    def on_gps_update(self, vehicle_id: str, timestamp: float,
                      x: Optional[float] = None, y: Optional[float] = None,
                      speed: Optional[float] = None) -> Dict[str, float]:
        """[§13] update position/speed/time; D_t^k of THAT vehicle is recomputed on demand."""
        v = self.vehicles[vehicle_id]
        expected = (v.x, v.y)
        if x is not None:
            v.x = float(x)
        if y is not None:
            v.y = float(y)
        if speed is not None and float(speed) > 0:
            v.observed_speed = float(speed)
        v.last_gps_timestamp = timestamp
        v.current_time = max(v.current_time, timestamp)
        v.last_service_location_id = None       # the vehicle is between service locations
        self.now = max(self.now, timestamp)
        deviation = euclidean(expected, (v.x, v.y))
        speed_ratio = (v.base_speed / v.observed_speed) if v.observed_speed else 1.0
        return {"route_deviation": deviation, "travel_time_ratio": speed_ratio}

    def on_service_complete(self, order_id: str, vehicle_id: str, timestamp: float,
                            operation: Optional[str] = None,
                            service_location_id: Optional[str] = None) -> Dict[str, Any]:
        """[§14/§15 generalised — FEDEX EXTENSION] confirm ONE operation of a service plan.

        A single handler now covers CUSTOMER_PICKUP, CUSTOMER_DELIVERY, DEPOT_LOAD and
        DEPOT_UNLOAD.  Everything downstream (order stage, cargo ownership, vehicle load,
        carrying list, position, clock, Z/U/D, MCDSM masks, metrics, logs) is updated from
        the same place, so no flow can drift out of sync with another.
        """
        o, v = self.orders[order_id], self.vehicles[vehicle_id]
        expected = o.next_operation
        assert expected is not None, f"order {order_id} is already completed"
        operation = operation or expected
        assert operation == expected, (
            f"order {order_id} expects {expected}, got {operation}")

        # ---- ownership: once cargo is on board only its carrier may continue ------------
        if o.ownership_locked:
            assert o.assigned_vehicle_id == vehicle_id, (
                f"order {order_id} is on board {o.assigned_vehicle_id}, not {vehicle_id}")
        if o.load_sign > 0:
            assert o.cargo_weight <= v.available_capacity + 1e-9, (
                f"capacity violated loading {order_id} on {vehicle_id}: "
                f"w={o.cargo_weight} > Q-L={v.available_capacity}")

        # [EXTENSION] a confirmation for an operation another vehicle had accepted is a
        # planning inconsistency, not a state corruption: the cargo assertions above are
        # the real safety net, so we log it and let the physical fact win.
        if (o.committed_vehicle_id is not None
                and o.committed_vehicle_id != vehicle_id):
            LOG.warning("order %s was committed to %s but %s completed %s",
                        order_id, o.committed_vehicle_id, vehicle_id, operation)

        delta = o.load_delta
        o.advance_stage(timestamp=timestamp, vehicle_id=vehicle_id,
                        service_location_id=service_location_id)
        # [EXTENSION] the commitment covered exactly THIS operation.  Mid-plan the cargo
        # lock (`ownership_locked`) takes over, and the next decision re-commits the order.
        self.release_commitment(order_id, f"{operation}_COMPLETE")

        # ---- vehicle load / carrying list ----------------------------------------------
        v.load = max(0.0, v.load + delta)
        if delta > 0:
            if order_id not in v.carrying:
                v.carrying.append(order_id)
        else:
            if order_id in v.carrying:
                v.carrying.remove(order_id)

        # ---- lateness is judged on the CUSTOMER promise only ---------------------------
        if (operation == OP_CUSTOMER_DELIVERY and o.latest_delivery_time is not None
                and timestamp > o.latest_delivery_time + 1e-9):
            o.late = True
            o.lateness = timestamp - o.latest_delivery_time

        # ---- position / clock -----------------------------------------------------------
        if service_location_id:
            sl = self.instance.location_by_id(service_location_id)
            v.x, v.y = sl.x, sl.y
            v.last_service_location_id = sl.service_location_id
        v.current_time = max(v.current_time, timestamp)
        v.n_services += 1
        self.now = max(self.now, timestamp)
        return {"order_id": order_id, "operation": operation, "z": o.z,
                "stage": o.stage, "completed": o.is_completed,
                "vehicle_load": v.load, "carrying": list(v.carrying)}

    # ---- [PAPER] thin aliases so existing feeds / tests / adapters keep working ---------
    def on_pickup_complete(self, order_id: str, vehicle_id: str, timestamp: float,
                           service_location_id: Optional[str] = None) -> Dict[str, Any]:
        """[§14] confirms whatever LOADING operation the plan expects next."""
        o = self.orders[order_id]
        op = o.next_operation
        assert op is not None and o.load_sign > 0, (
            f"PICKUP_COMPLETE on order {order_id} whose next operation is {op}")
        return self.on_service_complete(order_id, vehicle_id, timestamp, op,
                                        service_location_id)

    def on_delivery_complete(self, order_id: str, vehicle_id: str, timestamp: float,
                             service_location_id: Optional[str] = None) -> Dict[str, Any]:
        """[§15] confirms whatever UNLOADING operation the plan expects next."""
        o = self.orders[order_id]
        op = o.next_operation
        assert op is not None and o.load_sign < 0, (
            f"DELIVERY_COMPLETE on order {order_id} whose next operation is {op}")
        return self.on_service_complete(order_id, vehicle_id, timestamp, op,
                                        service_location_id)

    @staticmethod
    def _can_roll_back_for_transshipment(order: Order) -> bool:
        """Can this on-board order be handed over to another vehicle by a stage rollback?

        Only if the rollback really lands on a LOADING operation.  Otherwise the order
        would end up with `cargo_state=NOT_LOADED` AND an unloading `next_operation`, a
        combination `MCDSM.order_feasible` and the online feasibility validator both reject
        for every vehicle -- a permanent dead state.
        """
        if order.load_sign > 0:
            return True                     # already waiting for a loading operation
        previous = order.stage_index - 1
        if previous < 0 or previous >= len(order.service_plan):
            return False
        return OPERATION_SEMANTICS[order.service_plan[previous]][0] > 0

    def on_vehicle_breakdown(self, vehicle_id: str, timestamp: float) -> Dict[str, Any]:
        """[§16] IMPLEMENTATION ASSUMPTION — configurable, no silent reassignment."""
        v = self.vehicles[vehicle_id]
        v.status = VEHICLE_BROKEN
        v.active = False
        self.now = max(self.now, timestamp)
        policy = self.cfg.breakdown.policy
        affected = list(v.carrying)
        # [EXTENSION] THE breakdown exception: an accepted-but-not-yet-executed assignment
        # is ALWAYS released, whatever the cargo policy is.  A broken vehicle cannot serve
        # what it promised, so the order must become visible to the rest of the fleet again.
        # Cargo ownership (orders physically on board) stays policy-driven and unchanged.
        released_commitments = self.release_commitments_of(vehicle_id, "VEHICLE_BREAKDOWN")
        info = {"policy": policy, "vehicle_id": vehicle_id, "carrying": affected,
                "reassigned": [], "not_transshippable": [],
                "released_commitments": released_commitments}
        if policy == "B_emergency_transshipment" and self.cfg.breakdown.allow_auto_reassign_picked_cargo:
            # [LIMITATION] the transshipment is modelled as a ROLLBACK of one plan stage:
            # the order goes back to "not loaded" and the rescuing vehicle re-executes the
            # preceding LOADING operation at its original service location (the consignor,
            # or the source depot).  The physical cargo, however, is sitting on the BROKEN
            # vehicle wherever it stopped, which may be far from that location.  Modelling
            # the real handover would require an explicit rendezvous point, i.e. a change
            # to the mathematical formulation -- deliberately NOT done here.  Until then the
            # travel time reported for a transshipped leg is optimistic.
            kept: List[str] = []
            for oid in affected:
                o = self.orders[oid]
                if not self._can_roll_back_for_transshipment(o):
                    # [FIX] rolling this order back would leave it in a DEAD state:
                    # cargo_state=NOT_LOADED together with an unloading next_operation is
                    # masked out for EVERY vehicle, with no code path that ever recovers
                    # (an INBOUND_PRELOADED order has a single CUSTOMER_DELIVERY stage, so
                    # `stage_index - 1` cannot reach a loading operation).  Leave the cargo
                    # exactly where it physically is and export it for manual dispatch, the
                    # same treatment policy C gives every order.
                    kept.append(oid)
                    info["not_transshippable"].append(oid)
                    LOG.warning("order %s stays on the broken vehicle %s: its plan has no "
                                "loading operation to roll back to (flow=%s, stage %d/%d)",
                                oid, vehicle_id, o.flow_type, o.stage_index, o.n_stages)
                    continue
                # [FEDEX EXTENSION] transshipment: the cargo changes hands, so it goes back
                # to "not loaded" and its remaining plan restarts with a loading operation.
                o.assigned_vehicle_id = None
                o.cargo_state = CARGO_NOT_LOADED
                if o.load_sign < 0:
                    o.stage_index = max(0, o.stage_index - 1)
                info["reassigned"].append(oid)
                LOG.warning("order %s transshipped off %s: the rescuing vehicle is routed "
                            "to the ORIGINAL loading location (%s), not to where the cargo "
                            "physically is -- see the [LIMITATION] note above",
                            oid, vehicle_id, o.target_entity(v))
            # [FIX] recomputed from what is REALLY still on board, never hard-set to
            # []/0.0: `cargo_left_on_vehicles` and `final_load_total` are derived from
            # these two fields and used to under-report stranded cargo.
            v.carrying = kept
            v.load = sum(self.orders[x].cargo_weight for x in kept)
        elif policy == "C_manual_intervention":
            info["exported_for_manual_dispatch"] = affected
        # policy A (default): cargo stays on board until VEHICLE_RECOVERY
        return info

    def on_vehicle_recovery(self, vehicle_id: str, timestamp: float) -> None:
        v = self.vehicles[vehicle_id]
        v.status = VEHICLE_IDLE
        v.active = True
        v.current_time = max(v.current_time, timestamp)
        self.now = max(self.now, timestamp)

    def on_traffic_update(self, timestamp: float, factor: float, area: str = "ALL") -> float:
        self.now = max(self.now, timestamp)
        return self.traffic.update(timestamp, factor, area)

    def on_time_window_update(self, service_location_id: str, begin: Optional[float],
                              end: Optional[float], timestamp: float) -> None:
        sl = self.instance.location_by_id(service_location_id)
        if begin is not None:
            sl.begin_time = float(begin)
        if end is not None:
            sl.end_time = float(end)
        self.now = max(self.now, timestamp)

    def on_priority_update(self, order_id: str, priority: float, timestamp: float) -> None:
        self.orders[order_id].priority = float(priority)
        self.now = max(self.now, timestamp)

    def on_shift_start(self, vehicle_id: Optional[str], timestamp: float) -> None:
        targets = [self.vehicles[vehicle_id]] if vehicle_id else self.vehicle_list
        for v in targets:
            v.status = VEHICLE_IDLE
            v.current_time = max(v.current_time, timestamp)
        self.now = max(self.now, timestamp)

    def on_shift_end(self, vehicle_id: Optional[str], timestamp: float) -> None:
        targets = [self.vehicles[vehicle_id]] if vehicle_id else self.vehicle_list
        for v in targets:
            v.status = "OFF_SHIFT"
            v.active = False
            # [EXTENSION] an off-shift vehicle can no longer honour what it accepted
            self.release_commitments_of(v.vehicle_id, "SHIFT_END")
        self.now = max(self.now, timestamp)

    # ------------------------------------------------------------------ queries
    def unfinished_orders(self) -> List[Order]:
        return [self.orders[o] for o in self.active_order_ids
                if not self.orders[o].is_completed and not self.orders[o].cancelled]

    def all_done(self) -> bool:
        return not self.unfinished_orders() and not self.unreleased_order_ids

    def snapshot(self) -> Dict[str, Any]:
        return {
            "now": self.now,
            "vehicles": [v.snapshot() for v in self.vehicle_list],
            "orders": [self.orders[o].snapshot() for o in self.active_order_ids],
            "Z": [int(self.orders[o].z) for o in self.active_order_ids],
            "traffic_factor": self.traffic.factor("ALL"),
        }
