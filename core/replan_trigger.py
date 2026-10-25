"""
replan_trigger.py
=================
[REAL-TIME EXTENSION §18]  The DRL policy must NOT be invoked after every incoming data
packet.  This module decides whether an event justifies a replanning cycle.

Defaults
    NEW_ORDER / any service confirmation / VEHICLE_BREAKDOWN /
    TIME_WINDOW_UPDATE                        -> replan
    TRAFFIC_UPDATE                            -> replan only if the predicted travel time
                                                 changed by at least rho
    GPS_UPDATE                                -> state update only, unless
                                                   * route deviation  > threshold, or
                                                   * predicted lateness occurs, or
                                                   * travel-time deterioration > threshold
Every threshold lives in `config.ReplanConfig`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import Config
from event_manager import EventResult
from utils import get_logger

LOG = get_logger("dfpdp.trigger")


@dataclass
class TriggerDecision:
    replan: bool
    reason: str
    detail: Dict[str, Any]

    def as_log_row(self, timestamp: float) -> Dict[str, Any]:
        return dict(timestamp=timestamp, replan=self.replan, reason=self.reason,
                    detail=str(self.detail))


class ReplanTrigger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_replan_wall: Optional[float] = None
        self.n_evaluated = 0
        self.n_triggered = 0

    # ------------------------------------------------------------------ main entry
    def evaluate(self, result: EventResult) -> TriggerDecision:
        self.n_evaluated += 1
        c = self.cfg.replan
        et = result.event.event_type
        info = result.info or {}
        decision = TriggerDecision(False, f"{et}:no_trigger", info)

        if not result.handled:
            decision = TriggerDecision(False, f"{et}:event_failed", info)
        elif et == "NEW_ORDER" and c.replan_on_new_order:
            decision = TriggerDecision(True, "NEW_ORDER", info)
        elif et == "CANCEL_ORDER" and c.replan_on_cancel_order:
            decision = TriggerDecision(True, "CANCEL_ORDER", info)
        # [FEDEX EXTENSION] every service confirmation frees the vehicle for its next
        # decision, so all four operations trigger exactly like the paper's two did.
        # The loading ops reuse `replan_on_pickup_complete`, the unloading ops reuse
        # `replan_on_delivery_complete`, so the existing switches keep their meaning.
        elif et in ("PICKUP_COMPLETE", "CUSTOMER_PICKUP_COMPLETE",
                    "DEPOT_LOAD_COMPLETE") and c.replan_on_pickup_complete:
            decision = TriggerDecision(True, et, info)
        elif et in ("DELIVERY_COMPLETE", "CUSTOMER_DELIVERY_COMPLETE",
                    "DEPOT_UNLOAD_COMPLETE") and c.replan_on_delivery_complete:
            decision = TriggerDecision(True, et, info)
        elif et == "SERVICE_COMPLETE":
            op = str(result.info.get("operation") or result.event.payload.get("operation", ""))
            loads = op in ("CUSTOMER_PICKUP", "DEPOT_LOAD")
            if (loads and c.replan_on_pickup_complete) or (
                    not loads and c.replan_on_delivery_complete):
                decision = TriggerDecision(True, f"SERVICE_COMPLETE:{op}", info)
        elif et == "VEHICLE_BREAKDOWN" and c.replan_on_vehicle_breakdown:
            decision = TriggerDecision(True, "VEHICLE_BREAKDOWN", info)
        elif et == "VEHICLE_RECOVERY" and c.replan_on_vehicle_recovery:
            decision = TriggerDecision(True, "VEHICLE_RECOVERY", info)
        elif et == "TIME_WINDOW_UPDATE" and c.replan_on_time_window_update:
            decision = TriggerDecision(True, "TIME_WINDOW_UPDATE", info)
        elif et == "ORDER_PRIORITY_UPDATE" and c.replan_on_priority_update:
            decision = TriggerDecision(True, "ORDER_PRIORITY_UPDATE", info)
        elif et == "SHIFT_START" and c.replan_on_shift_start:
            decision = TriggerDecision(True, "SHIFT_START", info)
        elif et == "TRAFFIC_UPDATE":
            rel = float(info.get("relative_change", 0.0))
            if rel >= c.traffic_rho:
                decision = TriggerDecision(
                    True, f"TRAFFIC_UPDATE:rel_change={rel:.3f}>=rho={c.traffic_rho}", info)
            else:
                decision = TriggerDecision(
                    False, f"TRAFFIC_UPDATE:rel_change={rel:.3f}<rho={c.traffic_rho}", info)
        elif et == "GPS_UPDATE":
            dev = float(info.get("route_deviation", 0.0))
            ratio = float(info.get("travel_time_ratio", 1.0))
            lateness = float(info.get("predicted_lateness", 0.0))
            if dev > c.gps_route_deviation_threshold:
                decision = TriggerDecision(True, f"GPS_UPDATE:route_deviation={dev:.2f}", info)
            elif lateness > c.gps_lateness_threshold:
                decision = TriggerDecision(True, f"GPS_UPDATE:predicted_lateness={lateness:.1f}", info)
            elif (ratio - 1.0) > c.gps_travel_time_deterioration:
                decision = TriggerDecision(
                    True, f"GPS_UPDATE:travel_time_deterioration={ratio - 1.0:.2f}", info)
            else:
                decision = TriggerDecision(False, "GPS_UPDATE:state_update_only", info)

        # ---- rate limiting -------------------------------------------------------------
        if decision.replan and c.min_seconds_between_replans > 0:
            now = time.perf_counter()
            if (self._last_replan_wall is not None
                    and now - self._last_replan_wall < c.min_seconds_between_replans):
                decision = TriggerDecision(False, decision.reason + ":rate_limited", decision.detail)
            else:
                self._last_replan_wall = now

        if decision.replan:
            self.n_triggered += 1
        return decision

    # ------------------------------------------------------------------ periodic mode
    def periodic_due(self, now: float, last_replan_time: Optional[float]) -> bool:
        """Used by experiment B (periodic replanning every N minutes)."""
        if last_replan_time is None:
            return True
        return (now - last_replan_time) >= self.cfg.replan.periodic_replan_minutes
