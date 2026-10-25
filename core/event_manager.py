"""
event_manager.py
================
Event-driven engine.                                                 [REAL-TIME EXTENSION]

Every event carries `event_timestamp`, `event_type`, the relevant entity ids and a payload.
Events are processed strictly chronologically (ties broken by `clock.event_priority`).

The same engine serves
    * MODE A : replay of historical company events (offline training)
    * MODE B : the live feed (ERP/OMS, TMS/GPS, traffic API)
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import Config
from entities import Event, Order, ORDER_DYNAMIC, EVENT_OPERATION, FLOW_PICKUP_DELIVERY
from state_manager import StateManager
from utils import get_logger, hhmm

LOG = get_logger("dfpdp.events")


# =======================================================================================
class EventQueue:
    """Chronological priority queue (stable for equal timestamps)."""

    def __init__(self, cfg: Config, events: Optional[List[Event]] = None):
        self.cfg = cfg
        self._heap: List[tuple] = []
        self._counter = itertools.count()
        for e in (events or []):
            self.push(e)

    def push(self, event: Event) -> None:
        prio = self.cfg.clock.event_priority.get(event.event_type, 99)
        heapq.heappush(self._heap, (event.timestamp, prio, next(self._counter), event))

    def push_many(self, events: List[Event]) -> None:
        for e in events:
            self.push(e)

    def pop(self) -> Optional[Event]:
        return heapq.heappop(self._heap)[3] if self._heap else None

    def peek_time(self) -> Optional[float]:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# =======================================================================================
@dataclass
class EventResult:
    event: Event
    handled: bool
    info: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_log_row(self) -> Dict[str, Any]:
        row = self.event.to_row()
        row.update(clock=hhmm(self.event.timestamp), handled=self.handled,
                   error=self.error, info=str(self.info))
        return row


class EventManager:
    """Routes events to the StateManager handlers and keeps an audit log."""

    def __init__(self, cfg: Config, state: StateManager,
                 order_factory: Optional[Callable[[Event], Order]] = None):
        self.cfg = cfg
        self.state = state
        self.queue = EventQueue(cfg)
        self.event_log: List[EventResult] = []
        self.order_factory = order_factory or self._default_order_factory
        self.subscribers: List[Callable[[EventResult], None]] = []

    # ------------------------------------------------------------------ plumbing
    def subscribe(self, fn: Callable[[EventResult], None]) -> None:
        self.subscribers.append(fn)

    def push(self, event: Event) -> None:
        self.queue.push(event)

    def _default_order_factory(self, event: Event) -> Order:
        """Build an Order from a NEW_ORDER payload if the order is not already known."""
        p = event.payload
        oid = event.order_id or f"RT-{len(self.state.orders) + 1}"
        return Order(
            order_id=oid,
            consignor_id=str(p.get("consignor_id", "")),
            consignee_id=str(p.get("consignee_id", "")),
            cargo_weight=float(p.get("cargo_weight", 0.0)),
            order_class=ORDER_DYNAMIC,
            release_time=event.timestamp,
            created_timestamp=float(p.get("created_timestamp", event.timestamp)),
            priority=float(p.get("priority", 0.0)),
            external_order_id=str(p.get("external_order_id", oid)),
            latest_delivery_time=p.get("latest_delivery_time"),
            # [FEDEX EXTENSION] the ERP feed may declare the flow; default stays paper-like
            flow_type=str(p.get("flow_type", FLOW_PICKUP_DELIVERY) or FLOW_PICKUP_DELIVERY),
            source_depot_id=str(p.get("source_depot_id", "") or ""),
            dest_depot_id=str(p.get("dest_depot_id", "") or ""),
            preloaded_vehicle_id=str(p.get("preloaded_vehicle_id", "") or ""),
        )

    def _apply_payload_position(self, event: Event) -> None:
        """TMS confirmations often carry the GPS position of the completed service."""
        p = event.payload
        if event.vehicle_id and ("x" in p or "y" in p):
            v = self.state.vehicles[event.vehicle_id]
            if p.get("x") is not None:
                v.x = float(p["x"])
            if p.get("y") is not None:
                v.y = float(p["y"])

    # ------------------------------------------------------------------ dispatch
    def process(self, event: Event) -> EventResult:
        st = self.state
        info: Dict[str, Any] = {}
        error = ""
        try:
            et = event.event_type
            if et == "NEW_ORDER":
                order = st.orders.get(event.order_id) or self.order_factory(event)
                order.release_time = min(order.release_time or event.timestamp, event.timestamp)
                slot = st.on_new_order(order, event.timestamp)
                info = {"slot": slot, "order_id": order.order_id,
                        "n_active": len(st.active_order_ids)}
            elif et == "CANCEL_ORDER":
                st.on_cancel_order(event.order_id, event.timestamp)
                info = {"order_id": event.order_id}
            elif et == "ORDER_PRIORITY_UPDATE":
                st.on_priority_update(event.order_id,
                                      float(event.payload.get("priority", 0.0)), event.timestamp)
                info = {"priority": event.payload.get("priority")}
            elif et == "TIME_WINDOW_UPDATE":
                st.on_time_window_update(
                    event.service_location_id,
                    event.payload.get("begin_time"), event.payload.get("end_time"),
                    event.timestamp)
                info = {"service_location_id": event.service_location_id}
            elif et == "GPS_UPDATE":
                info = st.on_gps_update(event.vehicle_id, event.timestamp,
                                        event.payload.get("x"), event.payload.get("y"),
                                        event.payload.get("speed"))
            elif et in EVENT_OPERATION or et == "SERVICE_COMPLETE":
                # [FEDEX EXTENSION] one branch for every service confirmation.
                #   SERVICE_COMPLETE          -> payload["operation"] names the operation
                #   CUSTOMER_PICKUP_COMPLETE  -> CUSTOMER_PICKUP        (etc.)
                #   PICKUP_COMPLETE / DELIVERY_COMPLETE  -> legacy customer aliases
                # An operation that is NOT the one the plan expects is rejected by
                # on_service_complete, so a stray confirmation cannot corrupt the state.
                operation = EVENT_OPERATION.get(et) or event.payload.get("operation")
                info = st.on_service_complete(event.order_id, event.vehicle_id,
                                              event.timestamp, operation,
                                              event.service_location_id)
                self._apply_payload_position(event)
            elif et == "VEHICLE_BREAKDOWN":
                info = st.on_vehicle_breakdown(event.vehicle_id, event.timestamp)
            elif et == "VEHICLE_RECOVERY":
                st.on_vehicle_recovery(event.vehicle_id, event.timestamp)
                info = {"vehicle_id": event.vehicle_id}
            elif et == "TRAFFIC_UPDATE":
                rel = st.on_traffic_update(event.timestamp,
                                           float(event.payload.get("factor", 1.0)),
                                           str(event.payload.get("area", "ALL")))
                info = {"relative_change": rel, "factor": event.payload.get("factor")}
            elif et == "SHIFT_START":
                st.on_shift_start(event.vehicle_id, event.timestamp)
                info = {"vehicle_id": event.vehicle_id}
            elif et == "SHIFT_END":
                st.on_shift_end(event.vehicle_id, event.timestamp)
                info = {"vehicle_id": event.vehicle_id}
            else:
                raise ValueError(f"unhandled event type {et}")
            handled = True
        except (AssertionError, KeyError, ValueError) as exc:
            handled = False
            error = f"{type(exc).__name__}: {exc}"
            LOG.error("event %s @%s failed: %s", event.event_type, hhmm(event.timestamp), error)

        # dynamic orders whose release time has passed become visible as well
        st.release_due_orders(event.timestamp)

        result = EventResult(event=event, handled=handled, info=info, error=error)
        self.event_log.append(result)
        for fn in self.subscribers:
            fn(result)
        return result

    # ------------------------------------------------------------------ drivers
    def run_until(self, timestamp: float) -> List[EventResult]:
        out = []
        while self.queue and (self.queue.peek_time() or 0.0) <= timestamp + 1e-9:
            out.append(self.process(self.queue.pop()))
        return out

    def run_all(self) -> List[EventResult]:
        out = []
        while self.queue:
            out.append(self.process(self.queue.pop()))
        return out

    # ------------------------------------------------------------------ export
    def log_rows(self) -> List[Dict[str, Any]]:
        return [r.as_log_row() for r in self.event_log]
