"""
entities.py
===========
Domain objects of the Dynamic Flexible Pickup and Delivery Problem (DFPDP).

[PAPER] order status  z_m in {0: waiting for pickup, 1: waiting for delivery, 2: completed}
        with the ONLY legal transitions 0 -> 1 -> 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------- constants
Z_WAIT_PICKUP = 0
Z_WAIT_DELIVERY = 1
Z_COMPLETED = 2

ROLE_DEPOT = "DEPOT"
ROLE_PICKUP = "PICKUP"
ROLE_DELIVERY = "DELIVERY"

ORDER_STATIC = "STATIC"
ORDER_DYNAMIC = "DYNAMIC"

# =======================================================================================
# [FEDEX EXTENSION] mixed inbound / outbound flows
# =======================================================================================
# The paper hard-codes ONE lifecycle (pickup -> delivery).  FedEx needs four, so the
# lifecycle becomes DATA (a per-order service plan) instead of CODE.
OP_CUSTOMER_PICKUP = "CUSTOMER_PICKUP"
OP_CUSTOMER_DELIVERY = "CUSTOMER_DELIVERY"
OP_DEPOT_LOAD = "DEPOT_LOAD"
OP_DEPOT_UNLOAD = "DEPOT_UNLOAD"

OPERATIONS = (OP_CUSTOMER_PICKUP, OP_CUSTOMER_DELIVERY, OP_DEPOT_LOAD, OP_DEPOT_UNLOAD)

# ---------------------------------------------------------------------------------------
# [DESIGN] the two orthogonal bits that fully describe any service operation.
#
#                     |  loads cargo (+1)      |  unloads cargo (-1)
#   ------------------+------------------------+------------------------
#   customer  (0)     |  CUSTOMER_PICKUP       |  CUSTOMER_DELIVERY
#   depot     (1)     |  DEPOT_LOAD            |  DEPOT_UNLOAD
#
# The DRL policy is given (load_sign, is_depot) rather than a 4-way one-hot of the
# operation or a one-hot of the flow type.  Two consequences:
#   1. the network sees immediately that CUSTOMER_DELIVERY and DEPOT_UNLOAD share the
#      SAME capacity mechanics, so what it learns on one flow transfers to the other
#      (important because FedEx volumes are very unbalanced across flows);
#   2. a NEW flow (e.g. cross-dock P -> depot unload -> depot load -> D) needs zero new
#      feature dimensions and zero architectural change - only a new service plan.
OPERATION_SEMANTICS: Dict[str, tuple] = {      # operation -> (load_sign, is_depot)
    OP_CUSTOMER_PICKUP:   (+1.0, 0.0),
    OP_CUSTOMER_DELIVERY: (-1.0, 0.0),
    OP_DEPOT_LOAD:        (+1.0, 1.0),
    OP_DEPOT_UNLOAD:      (-1.0, 1.0),
}

OPERATION_ROLE: Dict[str, str] = {             # operation -> service-location role
    OP_CUSTOMER_PICKUP:   ROLE_PICKUP,
    OP_CUSTOMER_DELIVERY: ROLE_DELIVERY,
    OP_DEPOT_LOAD:        ROLE_DEPOT,
    OP_DEPOT_UNLOAD:      ROLE_DEPOT,
}

STAGE_COMPLETED = "COMPLETED"

def stage_of(operation: Optional[str]) -> str:
    """Waiting stage name for the operation that comes next."""
    return STAGE_COMPLETED if operation is None else f"WAIT_{operation}"

STAGES = tuple(stage_of(op) for op in OPERATIONS) + (STAGE_COMPLETED,)

# ---- flow types and their required service plans ---------------------------------------
FLOW_PICKUP_DELIVERY = "PICKUP_DELIVERY"       # [PAPER] the original DFPDP lifecycle
FLOW_OUTBOUND = "OUTBOUND"                     # [FEDEX] customer -> depot
FLOW_INBOUND = "INBOUND"                       # [FEDEX] depot -> customer
FLOW_INBOUND_PRELOADED = "INBOUND_PRELOADED"   # [FEDEX] already on the vehicle at shift start

SERVICE_PLANS: Dict[str, tuple] = {
    FLOW_PICKUP_DELIVERY:   (OP_CUSTOMER_PICKUP, OP_CUSTOMER_DELIVERY),
    FLOW_OUTBOUND:          (OP_CUSTOMER_PICKUP, OP_DEPOT_UNLOAD),
    FLOW_INBOUND:           (OP_DEPOT_LOAD, OP_CUSTOMER_DELIVERY),
    FLOW_INBOUND_PRELOADED: (OP_CUSTOMER_DELIVERY,),
}
FLOW_TYPES = tuple(SERVICE_PLANS)

# cargo bookkeeping (drives ownership: once loaded, only the carrier may continue)
CARGO_NOT_LOADED = "NOT_LOADED"
CARGO_ON_VEHICLE = "ON_VEHICLE"
CARGO_UNLOADED = "UNLOADED"

VEHICLE_IDLE = "IDLE"
VEHICLE_BUSY = "BUSY"
VEHICLE_BROKEN = "BROKEN"
VEHICLE_OFF_SHIFT = "OFF_SHIFT"

EVENT_TYPES = (
    "NEW_ORDER", "CANCEL_ORDER", "ORDER_PRIORITY_UPDATE", "TIME_WINDOW_UPDATE",
    "GPS_UPDATE", "PICKUP_COMPLETE", "DELIVERY_COMPLETE",
    "VEHICLE_BREAKDOWN", "VEHICLE_RECOVERY", "TRAFFIC_UPDATE",
    "SHIFT_START", "SHIFT_END",
    # [FEDEX EXTENSION] generic + operation-specific service confirmations.
    # PICKUP_COMPLETE / DELIVERY_COMPLETE remain valid aliases of the customer variants,
    # so existing ERP/TMS feeds keep working unchanged.
    "SERVICE_COMPLETE",
    "CUSTOMER_PICKUP_COMPLETE", "CUSTOMER_DELIVERY_COMPLETE",
    "DEPOT_LOAD_COMPLETE", "DEPOT_UNLOAD_COMPLETE",
)

#: event_type -> operation it confirms  [FEDEX EXTENSION]
EVENT_OPERATION: Dict[str, str] = {
    "PICKUP_COMPLETE":            OP_CUSTOMER_PICKUP,
    "CUSTOMER_PICKUP_COMPLETE":   OP_CUSTOMER_PICKUP,
    "DELIVERY_COMPLETE":          OP_CUSTOMER_DELIVERY,
    "CUSTOMER_DELIVERY_COMPLETE": OP_CUSTOMER_DELIVERY,
    "DEPOT_LOAD_COMPLETE":        OP_DEPOT_LOAD,
    "DEPOT_UNLOAD_COMPLETE":      OP_DEPOT_UNLOAD,
}


# ---------------------------------------------------------------------------------------
@dataclass
class ServiceLocation:
    """A *physical* place where an entity (consignor / consignee / depot) can be served.

    [PAPER] "flexible" in DFPDP == one entity may own SEVERAL physical service locations.
    """
    service_location_id: str
    entity_id: str
    entity_role: str                    # DEPOT | PICKUP | DELIVERY
    x: float
    y: float
    begin_time: float                   # B  (minutes since midnight)
    end_time: float                     # E
    service_duration: float = 0.0
    available: bool = True
    lat: Optional[float] = None
    lon: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self):
        return (self.x, self.y)

    def time_compatible(self, arrival_time: float) -> bool:
        """Arrival may be early (we then wait) but never after E."""
        return self.available and arrival_time <= self.end_time + 1e-9

    def waiting_time(self, arrival_time: float) -> float:
        return max(0.0, self.begin_time - arrival_time)

    def lateness(self, arrival_time: float) -> float:
        return max(0.0, arrival_time - self.end_time)


# ---------------------------------------------------------------------------------------
@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_type: str = "TRUCK"
    capacity: float = 100.0             # Q_k        (static)
    base_speed: float = 0.8             # (static) distance units / minute
    depot_id: str = "DEPOT"
    shift_start: float = 8 * 60.0
    shift_end: float = 18 * 60.0
    speed_profile: Dict[int, float] = field(default_factory=dict)   # hour -> speed multiplier
    active: bool = True

    # ---- dynamic variables -------------------------------------------------------------
    load: float = 0.0                   # L_k
    x: float = 0.0
    y: float = 0.0
    current_time: float = 8 * 60.0      # current accumulated time (MCDSM criterion)
    status: str = VEHICLE_IDLE
    observed_speed: Optional[float] = None      # [REAL-TIME EXTENSION] from GPS
    last_gps_timestamp: Optional[float] = None
    last_service_location_id: Optional[str] = None

    # ---- bookkeeping -------------------------------------------------------------------
    index: int = 0
    working_time: float = 0.0           # accumulated travel + wait + service (fairness metric)
    travel_time: float = 0.0
    waiting_time: float = 0.0
    distance: float = 0.0
    n_services: int = 0
    carrying: List[str] = field(default_factory=list)   # order_ids currently on board
    route: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------------------------
    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def available_capacity(self) -> float:
        return self.capacity - self.load

    def is_available(self, now: Optional[float] = None) -> bool:
        """Is this vehicle able to START new work?

        [EXTENSION] `now` is the world clock.  Availability is judged at the instant the
        vehicle would actually depart — `max(own clock, world clock)` — not at its own
        accumulated time alone.  A vehicle idle since the start of the shift keeps a stale
        `current_time`, and judging by that clock alone made it look available hours after
        its shift had ended, so the dispatcher happily issued `status=OK` recommendations
        that put a driver on the road past their shift end.

        Taking `now` on its own would be equally wrong: it would make an out-of-shift
        vehicle look available whenever some *other* vehicle is still in shift.

        Offline this is a no-op: `state.now == min(current_time over available vehicles)`,
        so `max(...)` collapses to `current_time` for every vehicle that reaches this test.
        """
        if not self.active or self.status == VEHICLE_BROKEN:
            return False
        t = self.current_time if now is None else max(self.current_time, now)
        return t <= self.shift_end + 1e-9

    def speed_at(self, t: float, use_profile: bool = True) -> float:
        spd = self.observed_speed if self.observed_speed else self.base_speed
        if use_profile and self.speed_profile:
            hour = int((t // 60) % 24)
            spd *= float(self.speed_profile.get(hour, 1.0))
        return max(spd, 1e-6)

    def snapshot(self) -> Dict[str, Any]:
        return dict(vehicle_id=self.vehicle_id, x=self.x, y=self.y, load=self.load,
                    current_time=self.current_time, status=self.status,
                    working_time=self.working_time, carrying=list(self.carrying))


# ---------------------------------------------------------------------------------------
@dataclass
class Order:
    """An order + its own service plan.

    [PAPER] the original model has ONE global lifecycle z = 0 -> 1 -> 2.
    [FEDEX EXTENSION] the lifecycle becomes per-order data: `service_plan` is an ordered
    list of operations and `stage_index` says how far along it is.  `z` survives as a
    DERIVED property so every paper-side formula, log and figure keeps working:

        z = 0  cargo not yet loaded          (whatever the next operation is)
        z = 1  cargo currently on a vehicle  (whatever the remaining operation is)
        z = 2  every operation of the plan has been executed
    """
    order_id: str
    consignor_id: str
    consignee_id: str
    cargo_weight: float
    order_class: str = ORDER_STATIC     # STATIC | DYNAMIC
    release_time: float = 0.0
    created_timestamp: float = 0.0
    priority: float = 0.0               # 0 = normal, higher = more urgent
    external_order_id: str = ""
    latest_delivery_time: Optional[float] = None    # used by slack / lateness metrics

    # ---- [FEDEX EXTENSION] flow definition ---------------------------------------------
    flow_type: str = FLOW_PICKUP_DELIVERY
    service_plan: tuple = ()            # filled from flow_type when empty
    source_depot_id: str = ""           # DEPOT_LOAD target   ("" -> the vehicle's own depot)
    dest_depot_id: str = ""             # DEPOT_UNLOAD target ("" -> the vehicle's own depot)
    preloaded_vehicle_id: str = ""      # INBOUND_PRELOADED: who already carries the cargo

    # ---- state -------------------------------------------------------------------------
    stage_index: int = 0
    cargo_state: str = CARGO_NOT_LOADED
    assigned_vehicle_id: Optional[str] = None       # owner once cargo is on board
    # [EXTENSION] set when a dispatch decision is ACCEPTED (before the cargo physically
    # moves).  It makes the assignment exclusive to that vehicle until the operation
    # completes, the order is cancelled, the shift ends, or the vehicle breaks down.
    committed_vehicle_id: Optional[str] = None
    pickup_timestamp: Optional[float] = None
    delivery_timestamp: Optional[float] = None
    released: bool = False
    cancelled: bool = False
    late: bool = False
    lateness: float = 0.0

    # ---- bookkeeping -------------------------------------------------------------------
    index: int = -1                     # slot inside the padded tensors
    pickup_location_id: Optional[str] = None
    delivery_location_id: Optional[str] = None
    stage_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------------------------
    def __post_init__(self):
        self.flow_type = str(self.flow_type or FLOW_PICKUP_DELIVERY).strip().upper()
        if self.flow_type not in SERVICE_PLANS:
            raise ValueError(f"unknown flow_type '{self.flow_type}' for order {self.order_id}")
        if not self.service_plan:
            self.service_plan = SERVICE_PLANS[self.flow_type]
        self.service_plan = tuple(self.service_plan)
        for op in self.service_plan:
            if op not in OPERATIONS:
                raise ValueError(f"unknown operation '{op}' in plan of order {self.order_id}")

    # ------------------------------------------------------------------ plan queries
    @property
    def n_stages(self) -> int:
        return len(self.service_plan)

    @property
    def next_operation(self) -> Optional[str]:
        """The operation the environment must execute if this order is selected."""
        if self.stage_index >= len(self.service_plan):
            return None
        return self.service_plan[self.stage_index]

    @property
    def stage(self) -> str:
        return stage_of(self.next_operation)

    @property
    def is_completed(self) -> bool:
        return self.stage_index >= len(self.service_plan)

    @property
    def is_final_stage(self) -> bool:
        return (not self.is_completed) and self.stage_index == len(self.service_plan) - 1

    @property
    def stage_progress(self) -> float:
        return self.stage_index / max(1, len(self.service_plan))

    @property
    def load_sign(self) -> float:
        """+1 the next operation loads cargo, -1 it unloads, 0 the order is finished."""
        op = self.next_operation
        return 0.0 if op is None else OPERATION_SEMANTICS[op][0]

    @property
    def next_is_depot(self) -> float:
        op = self.next_operation
        return 0.0 if op is None else OPERATION_SEMANTICS[op][1]

    @property
    def load_delta(self) -> float:
        """Signed change of the vehicle load if the next operation is executed."""
        return self.load_sign * self.cargo_weight

    @property
    def cargo_on_vehicle(self) -> bool:
        return self.cargo_state == CARGO_ON_VEHICLE

    @property
    def ownership_locked(self) -> bool:
        """Once cargo is on board, ONLY `assigned_vehicle_id` may continue the plan.

        Deliberately CARGO-based only: it backs the hard assertions of
        `StateManager.on_service_complete` and the U[i,9] feature, both of which must keep
        meaning "the cargo is physically on that vehicle".  The softer, *planning* level
        lock lives in `locked_to_vehicle_id`.
        """
        return self.cargo_on_vehicle and self.assigned_vehicle_id is not None

    @property
    def locked_to_vehicle_id(self) -> Optional[str]:
        """Vehicle that exclusively owns the NEXT operation: physical carrier, else commitment.

        [EXTENSION] `None` means the order is still open to every vehicle.
        """
        if self.cargo_on_vehicle and self.assigned_vehicle_id is not None:
            return self.assigned_vehicle_id
        return self.committed_vehicle_id

    # ---- paper compatibility -----------------------------------------------------------
    @property
    def z(self) -> int:
        """[PAPER] derived 0 / 1 / 2 view of the plan (read-only)."""
        if self.is_completed:
            return Z_COMPLETED
        return Z_WAIT_DELIVERY if self.cargo_on_vehicle else Z_WAIT_PICKUP

    @property
    def pickup_vehicle_id(self) -> Optional[str]:
        """[PAPER] alias kept for the same-vehicle-delivery rule and old log readers."""
        return self.assigned_vehicle_id

    @pickup_vehicle_id.setter
    def pickup_vehicle_id(self, value: Optional[str]) -> None:
        self.assigned_vehicle_id = value

    def set_status(self, new_z: int, strict: bool = True) -> None:
        """[PAPER] compatibility shim: 0 -> 1 -> 2 only, executed as one plan advance."""
        if strict and new_z != self.z + 1:
            raise ValueError(
                f"illegal status transition for order {self.order_id}: {self.z} -> {new_z}")
        self.advance_stage()

    # ------------------------------------------------------------------ transitions
    def advance_stage(self, timestamp: Optional[float] = None,
                      vehicle_id: Optional[str] = None,
                      service_location_id: Optional[str] = None) -> str:
        """Execute the next operation of the plan and move the cargo state with it."""
        op = self.next_operation
        if op is None:
            raise ValueError(f"order {self.order_id} is already completed")
        sign = OPERATION_SEMANTICS[op][0]
        self.cargo_state = CARGO_ON_VEHICLE if sign > 0 else CARGO_UNLOADED
        if sign > 0 and vehicle_id is not None:
            self.assigned_vehicle_id = vehicle_id
        if op == OP_CUSTOMER_PICKUP:
            self.pickup_timestamp = timestamp
            self.pickup_location_id = service_location_id
        elif op == OP_CUSTOMER_DELIVERY:
            self.delivery_timestamp = timestamp
            self.delivery_location_id = service_location_id
        self.stage_index += 1
        self.stage_history.append(dict(operation=op, timestamp=timestamp,
                                       vehicle_id=vehicle_id,
                                       service_location_id=service_location_id,
                                       stage_index=self.stage_index))
        if self.is_completed:
            self.assigned_vehicle_id = self.assigned_vehicle_id or vehicle_id
        return op

    def reset_plan(self) -> None:
        self.stage_index = 0
        self.cargo_state = CARGO_NOT_LOADED
        self.assigned_vehicle_id = None
        self.committed_vehicle_id = None        # [EXTENSION] no accepted assignment at t=0
        self.pickup_timestamp = self.delivery_timestamp = None
        self.pickup_location_id = self.delivery_location_id = None
        self.stage_history = []

    def mark_preloaded(self, vehicle_id: str) -> None:
        """[FEDEX EXTENSION] INBOUND_PRELOADED: cargo is on the vehicle before t=0."""
        self.cargo_state = CARGO_ON_VEHICLE
        self.assigned_vehicle_id = vehicle_id
        self.stage_index = 0

    # ------------------------------------------------------------------ targets
    def target_role(self) -> str:
        """Service-location role required by the NEXT operation."""
        op = self.next_operation
        return ROLE_DELIVERY if op is None else OPERATION_ROLE[op]

    def target_entity(self, vehicle: Any = None) -> str:
        """Entity whose physical locations satisfy the next operation.

        [IMPLEMENTATION ASSUMPTION] when an order does not name its own depot, the depot
        operation happens at the depot of the vehicle performing it - which is why this
        needs the vehicle and why D_t^k is genuinely vehicle-dependent for depot legs.
        """
        op = self.next_operation
        depot_id = ""
        if vehicle is not None:
            depot_id = getattr(vehicle, "depot_id", str(vehicle))
        if op == OP_CUSTOMER_PICKUP:
            return self.consignor_id
        if op == OP_CUSTOMER_DELIVERY or op is None:
            return self.consignee_id
        if op == OP_DEPOT_LOAD:
            return self.source_depot_id or depot_id
        if op == OP_DEPOT_UNLOAD:
            return self.dest_depot_id or depot_id
        raise ValueError(f"unknown operation {op}")

    def requires_role(self, role: str) -> bool:
        """Does this order's plan ever need a location of that role?  (used by validation)"""
        return any(OPERATION_ROLE[op] == role for op in self.service_plan)

    # ------------------------------------------------------------------ export
    def snapshot(self) -> Dict[str, Any]:
        return dict(order_id=self.order_id, z=self.z, order_class=self.order_class,
                    flow_type=self.flow_type, stage=self.stage,
                    stage_index=self.stage_index, n_stages=self.n_stages,
                    next_operation=self.next_operation or "",
                    cargo_state=self.cargo_state,
                    assigned_vehicle_id=self.assigned_vehicle_id,
                    committed_vehicle_id=self.committed_vehicle_id,   # [EXTENSION]
                    locked_to_vehicle_id=self.locked_to_vehicle_id,   # [EXTENSION]
                    pickup_vehicle_id=self.assigned_vehicle_id,
                    pickup_timestamp=self.pickup_timestamp,
                    delivery_timestamp=self.delivery_timestamp,
                    released=self.released, cancelled=self.cancelled, late=self.late)


# ---------------------------------------------------------------------------------------
@dataclass(order=False)
class Event:
    """Generic real-time / historical event.  [REAL-TIME EXTENSION]"""
    timestamp: float
    event_type: str
    order_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    service_location_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "SIM"                 # SIM | ERP | TMS | TRAFFIC_API | HISTORICAL
    event_id: str = ""

    def __post_init__(self):
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.event_type}")

    def sort_key(self, priority_map: Dict[str, int]) -> tuple:
        return (self.timestamp, priority_map.get(self.event_type, 99), self.event_id)

    def to_row(self) -> Dict[str, Any]:
        return dict(timestamp=self.timestamp, event_type=self.event_type,
                    order_id=self.order_id or "", vehicle_id=self.vehicle_id or "",
                    service_location_id=self.service_location_id or "",
                    source=self.source, payload=str(self.payload))


# ---------------------------------------------------------------------------------------
@dataclass
class ProblemInstance:
    """Everything needed to run one episode / one operating day."""
    vehicles: List[Vehicle]
    orders: List[Order]
    service_locations: Dict[str, List[ServiceLocation]]   # entity_id -> physical locations
    speed_profile: Dict[int, float] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)     # historical / replay events
    day: str = "D0"
    data_source_tag: str = "REAL_HISTORICAL"
    meta: Dict[str, Any] = field(default_factory=dict)

    def locations_for(self, entity_id: str, role: str) -> List[ServiceLocation]:
        return [sl for sl in self.service_locations.get(entity_id, [])
                if sl.entity_role == role and sl.available]

    def order_by_id(self, order_id: str) -> Order:
        for o in self.orders:
            if o.order_id == order_id:
                return o
        raise KeyError(order_id)

    def vehicle_by_id(self, vehicle_id: str) -> Vehicle:
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        raise KeyError(vehicle_id)

    def location_by_id(self, sl_id: str) -> ServiceLocation:
        for group in self.service_locations.values():
            for sl in group:
                if sl.service_location_id == sl_id:
                    return sl
        raise KeyError(sl_id)


# ---------------------------------------------------------------------------------------
def euclidean(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
