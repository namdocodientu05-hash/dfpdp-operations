"""
dynamic_orders.py
=================
[PAPER §"dynamic orders"] A dynamic order must be COMPLETELY invisible to

    * MCDSM
    * the Actor
    * the state Z / U / D

until its release condition occurs.  When released it is inserted into `active_orders`,
initialised with z = 0, its U information is appended and its pickup distances are computed
in D for EVERY vehicle; the order padding masks grow accordingly and the order becomes
selectable from the *next* decision cycle onwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import Config
from entities import Event, Order, ORDER_DYNAMIC
from state_manager import StateManager
from utils import get_logger, hhmm

LOG = get_logger("dfpdp.dynamic")


@dataclass
class ReleaseRecord:
    timestamp: float
    order_id: str
    slot: Optional[int]
    n_active_after: int
    trigger: str


class DynamicOrderManager:
    """Thin coordinator on top of StateManager's slot bookkeeping."""

    def __init__(self, cfg: Config, state: StateManager):
        self.cfg = cfg
        self.state = state
        self.records: List[ReleaseRecord] = []

    # ------------------------------------------------------------------ properties
    @property
    def active_orders(self) -> List[Order]:
        return [self.state.orders[o] for o in self.state.active_order_ids]

    @property
    def unreleased_dynamic_orders(self) -> List[Order]:
        return [self.state.orders[o] for o in self.state.unreleased_order_ids]

    # ------------------------------------------------------------------ release
    def release_due(self, timestamp: float, trigger: str = "clock") -> List[str]:
        """Time-driven release (offline replay / simulated episodes)."""
        released = self.state.release_due_orders(timestamp)
        for oid in released:
            self._record(timestamp, oid, trigger)
        return released

    def release_event(self, order: Order, timestamp: float,
                      trigger: str = "NEW_ORDER") -> Optional[int]:
        """Event-driven release (live ERP/OMS feed)."""
        slot = self.state.on_new_order(order, timestamp)
        self._record(timestamp, order.order_id, trigger)
        return slot

    def _record(self, timestamp: float, order_id: str, trigger: str) -> None:
        order = self.state.orders[order_id]
        rec = ReleaseRecord(timestamp=timestamp, order_id=order_id, slot=order.index,
                            n_active_after=len(self.state.active_order_ids), trigger=trigger)
        self.records.append(rec)
        LOG.debug("%s released %s into slot %s (%d active)",
                  hhmm(timestamp), order_id, order.index, rec.n_active_after)

    # ------------------------------------------------------------------ helpers
    def pending_release_events(self) -> List[Event]:
        """Turn the not-yet-released dynamic orders into NEW_ORDER events for the replay."""
        return [Event(timestamp=o.release_time, event_type="NEW_ORDER", order_id=o.order_id,
                      payload={"order_class": ORDER_DYNAMIC}, source="SIM",
                      event_id=f"REL-{o.order_id}")
                for o in self.unreleased_dynamic_orders]

    def timeline(self) -> List[Dict[str, Any]]:
        return [dict(timestamp=r.timestamp, clock=hhmm(r.timestamp), order_id=r.order_id,
                     slot=r.slot, n_active=r.n_active_after, trigger=r.trigger)
                for r in self.records]
