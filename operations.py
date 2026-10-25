"""Configurable, finite demo runs sharing the original state and feasibility rules."""
import copy
import random
import time
import uuid

import numpy as np
from road_routes import road_position
from traffic_profiles import (REGION_LABELS, REGION_MEAN, traffic_payload, speed_kmh)

from simulator import (SimulationEngine as LegacyEngine, CORE, DEPOT, SAMPLE_LOCATIONS,
                       Config, Vehicle, Order, ProblemInstance, ServiceLocation, Event,
                       ROLE_DEPOT, ROLE_PICKUP, ROLE_DELIVERY, ORDER_STATIC,
                       FLOW_OUTBOUND, FLOW_INBOUND, FLOW_PICKUP_DELIVERY,
                       OP_LABELS, local_xy, local_latlon, hhmm, torch)

ALGORITHMS = {"drl": "DRL · PESDVS", "nearest": "Điểm gần nhất", "deadline": "Hạn giao sớm nhất"}
MODELS = {"pilot": "current_pilot_best.pt", "best": "best_actor_critic.pt"}

# Đội xe tham chiếu: (nhóm, model, tải trọng kg, thể tích m³, hệ số tốc độ, số lượng) — tổng 24 xe.
FLEET_MODELS = [
    ("VAN", "Cargo van", 1134.0, 3.85, 1.0, 10),
    ("VAN", "E-Transit", 1270.0, 7.90, 1.0, 5),
    ("VAN", "BrightDrop 600", 1420.0, 17.41, 1.0, 5),
    ("TRUCK", "Truck 12 ft", 2268.0, 22.20, 0.9, 2),
    ("TRUCK", "Truck 20 ft", 5897.0, 37.00, 0.9, 2),
]
BASE_SPEED = .58  # distance units / minute ứng với hệ số tốc độ 1.0


def fleet_specs():
    """24 xe xếp xen kẽ theo tỷ lệ từng model, để chọn N xe đầu vẫn có đủ các loại."""
    units = [((k + .5) / count, i, k) for i, (*_, count) in enumerate(FLEET_MODELS) for k in range(count)]
    return [dict(zip(("group", "model", "capacity", "volume_m3", "speed_factor"), FLEET_MODELS[i][:5]))
            for _, i, _ in sorted(units)]


class SimulationEngine(LegacyEngine):
    SHIFT_START = 8 * 60.0
    SHIFT_END = 22 * 60.0

    def __init__(self, vehicles=6, orders=30, dynamic_percent=30, algorithm="drl",
                 model="pilot", seed=42, days=3, traffic_region="HCMC", start_weekday=0):
        if not 1 <= vehicles <= 24 or not 1 <= orders <= 100:
            raise ValueError("Quy mô cho phép: 1–24 xe và 1–100 đơn.")
        if not 1 <= int(days) <= 7:
            raise ValueError("Thời lượng mô phỏng cho phép: 1–7 ngày.")
        traffic_region = str(traffic_region).upper()
        if traffic_region not in REGION_LABELS:
            raise ValueError("Khu vực giao thông không hợp lệ.")
        if not 0 <= int(start_weekday) <= 6:
            raise ValueError("Ngày bắt đầu không hợp lệ.")
        if not 0 <= dynamic_percent <= 100 or algorithm not in ALGORITHMS or model not in MODELS:
            raise ValueError("Cấu hình mô phỏng không hợp lệ.")
        self.settings = dict(vehicles=vehicles, orders=orders, dynamic_percent=dynamic_percent,
                             algorithm=algorithm, model=model, seed=seed, days=int(days),
                             traffic_region=traffic_region, start_weekday=int(start_weekday))
        self.finish_status = "ready"
        self.status_message = "Sẵn sàng. Bấm Bắt đầu để điều phối đội xe."
        self.deferred_breakdowns = set()
        self.policy_decisions = 0
        self.road_router = None
        self.day_closing = False
        self.daily_summaries = []
        self.carryover_history = []
        self.decision_trace = []
        self.latest_vehicle_ranking = []
        self.profile_speed_kmh = None
        self.telemetry = []
        super().__init__()

    def _checkpoint_path(self):
        return CORE / "checkpoints" / MODELS[self.settings["model"]]

    def _config_from_checkpoint(self, path):
        if not path.is_file():
            raise ValueError(f"Thiếu checkpoint {path.name}.")
        # Use only the bundled local checkpoints; keep feature scales from training.
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = Config.from_dict(raw.get("config", {}))
        self.training_scale = dict(vehicles=cfg.problem.K_max, orders=cfg.problem.M_max)
        cfg.problem.K_max = max(cfg.problem.K_max, self.settings["vehicles"])
        cfg.problem.M_max = max(cfg.problem.M_max, 100)
        cfg.inference.checkpoint = str(path)
        cfg.inference.auto_approve = False
        cfg.inference.decode_mode = "greedy"
        cfg.inference.write_logs = False
        cfg.mcdsm.lock_committed_orders = True
        cfg.service_location.infeasibility_rule = "allow_late_logged"
        cfg.clock.day_start = self.SHIFT_START
        # Multi-day horizon is handled by the prototype orchestration layer.
        cfg.clock.day_end = self.settings["days"] * 1440.0 + self.SHIFT_END
        return cfg

    def _build_runtime(self, seed=None):
        self.run_id = uuid.uuid4().hex
        self.road_positions = {}
        self.road_stops = {}
        self.road_depot = None
        self.routing_status = "ready"
        self.routing_message = ""
        self.finish_status = "ready"
        self.status_message = "Sẵn sàng. Bấm Bắt đầu để điều phối đội xe."
        self.deferred_breakdowns = set()
        self.policy_decisions = 0
        self.order_seq = 100
        self.last_choice = None
        self.day_closing = False
        self.daily_summaries = []
        self.carryover_history = []
        self.decision_trace = []
        self.latest_vehicle_ranking = []
        self.profile_speed_kmh = None
        self.telemetry = []
        super()._build_runtime(self.settings["seed"])
        self.events = []
        self._apply_calibrated_traffic()
        self._log("SYSTEM", f"Mô phỏng mới: {self.settings['vehicles']} xe, {self.settings['orders']} đơn, {self.settings['days']} ngày · {ALGORITHMS[self.settings['algorithm']] }.")
        self._log("TRAFFIC_UPDATE", f"Đã nạp hồ sơ tốc độ {REGION_LABELS[self.settings['traffic_region']]} theo giờ/ngày; tốc độ hiện tại {self.profile_speed_kmh:.1f} km/h.")

    def reset(self, new_seed=False):
        with self.lock:
            if new_seed:
                self.settings["seed"] = random.SystemRandom().randint(0, 999999)
            self._build_runtime()

    def set_speed(self, minutes_per_tick):
        with self.lock:
            self.sim_minutes_per_tick = max(0.25, min(240.0, float(minutes_per_tick)))
            self._log("SYSTEM", f"Tốc độ mô phỏng: 1 giây = {self.sim_minutes_per_tick:g} phút mô phỏng.")

    def _initial_instance(self):
        self.order_origins = {}
        self.initial_plan_assignments = {}
        dx, dy = local_xy(DEPOT["lat"], DEPOT["lon"])
        locations = {DEPOT["id"]: [ServiceLocation(
            service_location_id="SGNA-DEPOT-01", entity_id=DEPOT["id"], entity_role=ROLE_DEPOT,
            x=dx, y=dy, begin_time=480, end_time=1e9, service_duration=4,
            lat=DEPOT["lat"], lon=DEPOT["lon"], meta=dict(name=DEPOT["name"], address=DEPOT["address"]))]}
        specs = fleet_specs()[:self.settings["vehicles"]]
        self.vehicle_specs = {f"V{i+1}": s for i, s in enumerate(specs)}
        vehicles = [Vehicle(f"V{i+1}", s["group"], s["capacity"], BASE_SPEED * s["speed_factor"],
                            DEPOT["id"], 480, 1e9, index=i)
                    for i, s in enumerate(specs)]
        n_dynamic = round(self.settings["orders"] * self.settings["dynamic_percent"] / 100)
        n_static = self.settings["orders"] - n_dynamic
        self.blueprints = []
        orders = []
        keys = list(SAMPLE_LOCATIONS)
        flows = [FLOW_OUTBOUND, FLOW_INBOUND, FLOW_PICKUP_DELIVERY]
        # Dynamic demand is distributed across the selected multi-day horizon rather than
        # being compressed into a single morning.  Each day releases between 08:30–17:30.
        day_counts = [n_dynamic // self.settings["days"] for _ in range(self.settings["days"])]
        for d in range(n_dynamic % self.settings["days"]):
            day_counts[d] += 1
        day_seen = [0 for _ in day_counts]
        dynamic_seen = 0
        for i in range(self.settings["orders"]):
            p, d = self.rng.sample(keys, 2)
            flow = flows[i % 3]
            # Slightly distribute stops around reference addresses so 100 parcels do
            # not share only 24 exact coordinates. These are explicitly demo locations.
            def place(key):
                out = dict(SAMPLE_LOCATIONS[key])
                out["lat"] += self.rng.uniform(-.004, .004)
                out["lon"] += self.rng.uniform(-.004, .004)
                out["name"] = "Điểm mô phỏng gần " + out["name"]
                out["address"] = out["name"]
                return out
            pickup = place(p) if flow != FLOW_INBOUND else None
            delivery = place(d) if flow != FLOW_OUTBOUND else None
            weight = float(self.rng.randint(30, 180))
            window = float(self.rng.randint(180, 360))
            priority = float(self.rng.randint(1, 4))
            if i >= n_static:
                j = dynamic_seen
                # Round-robin across days keeps demand visible throughout the horizon.
                day = j % self.settings["days"]
                k = day_seen[day]
                day_seen[day] += 1
                dynamic_seen += 1
                count = max(1, day_counts[day])
                release_in_day = 510.0 if count == 1 else 510.0 + k * 540.0 / max(1, count - 1)
                release = day * 1440.0 + release_in_day
                self.blueprints.append(dict(schedule_id=f"AUTO-{j+1:03d}",
                    release_time=release, released=False, day_index=day,
                    pickup=pickup, delivery=delivery, flow_type=flow, weight=weight,
                    window_minutes=window, priority=priority))
                continue
            oid = f"DH-{i+1:03d}"
            consignor, consignee = DEPOT["id"], DEPOT["id"]
            if pickup:
                consignor = oid + "-P"
                self._add_customer_location(locations, consignor, ROLE_PICKUP, pickup, 480, 480+window, 5)
            if delivery:
                consignee = oid + "-D"
                self._add_customer_location(locations, consignee, ROLE_DELIVERY, delivery, 480, 480+window, 5)
            orders.append(Order(order_id=oid, consignor_id=consignor, consignee_id=consignee,
                cargo_weight=weight, order_class=ORDER_STATIC, release_time=480, created_timestamp=480,
                priority=priority, latest_delivery_time=480+window, flow_type=flow,
                source_depot_id=DEPOT["id"] if flow == FLOW_INBOUND else "",
                dest_depot_id=DEPOT["id"] if flow == FLOW_OUTBOUND else ""))
            self.order_origins[oid] = "STATIC"
        return ProblemInstance(vehicles=vehicles, orders=orders, service_locations=locations,
                               speed_profile={}, events=[], day="MULTIDAY-DEMO", data_source_tag="SYNTHETIC_MULTI_DAY_TRAFFIC_CALIBRATED")

    def _build_departure_schedule(self):
        self.departure_times = {v.vehicle_id: 480.0 for v in self.dispatcher.env.state.vehicle_list}
        self.departed = {vid: False for vid in self.departure_times}

    def _build_initial_plans(self):
        self.vehicle_plans = {vid: [] for vid in self.departure_times}

    def _build_auto_dynamic_schedule(self):
        self.auto_dynamic_schedule = sorted(copy.deepcopy(self.blueprints), key=lambda x: x["release_time"])
        self.auto_dynamic_released = self.auto_dynamic_skipped = self.auto_order_seq = 0

    # ------------------------------------------------------------------ multi-day + calibrated traffic
    def _day_index(self, t=None):
        t = self.display_time if t is None else float(t)
        return max(0, int(t // 1440.0))

    def _minute_of_day(self, t=None):
        t = self.display_time if t is None else float(t)
        return t % 1440.0

    def _day_start_abs(self, day=None):
        day = self._day_index() if day is None else int(day)
        return day * 1440.0 + self.SHIFT_START

    def _day_end_abs(self, day=None):
        day = self._day_index() if day is None else int(day)
        return day * 1440.0 + self.SHIFT_END

    def _apply_calibrated_traffic(self):
        """Apply the agreed hourly regional speed profile to model travel time.

        The profile sets the fleet-level observed road speed while preserving each
        vehicle's nominal relative speed.  Manual TRAFFIC_UPDATE factors remain a
        separate incident multiplier in the core TrafficModel.
        """
        if not getattr(self, 'dispatcher', None):
            return
        region = self.settings["traffic_region"]
        target = float(speed_kmh(region, self.display_time, self.settings["start_weekday"]))
        fleet = self.dispatcher.env.state.vehicle_list
        mean_base = sum(v.base_speed for v in fleet) / max(1, len(fleet))
        for v in fleet:
            relative = v.base_speed / max(mean_base, 1e-9)
            v.observed_speed = max(0.05, target / 60.0 * relative)
            # Prevent double application of the legacy prototype hourly multipliers.
            v.speed_profile = {}
        self.profile_speed_kmh = target

    def _current_carryover_orders(self):
        st = self.dispatcher.env.state
        return [o for o in st.orders.values() if o.released and not o.cancelled and not o.is_completed]

    def _finalize_day_summary(self, day):
        if any(x.get("day_index") == day for x in self.daily_summaries):
            return
        st = self.dispatcher.env.state
        carry = self._current_carryover_orders()
        completed = sum(1 for o in st.orders.values() if o.is_completed and o.release_time < (day + 1) * 1440.0)
        late = sum(1 for o in st.orders.values() if o.late and o.release_time < (day + 1) * 1440.0)
        summary = {
            "day_index": day,
            "day_number": day + 1,
            "completed_cumulative": completed,
            "late_cumulative": late,
            "carryover_count": len(carry),
            "carryover_ids": [o.order_id for o in carry[:30]],
            "distance_km": round(sum(v.distance for v in st.vehicle_list), 2),
        }
        self.daily_summaries.append(summary)
        self.carryover_history.append({"day_index": day, "order_ids": [o.order_id for o in carry]})
        self._log("SYSTEM", f"Chốt ngày {day+1}: {len(carry)} đơn chưa hoàn tất được chuyển sang ngày kế tiếp.")

    def _begin_next_day(self):
        day = self._day_index()
        next_day = day + 1
        if next_day >= self.settings["days"]:
            return False
        self._finalize_day_summary(day)
        st = self.dispatcher.env.state
        next_start = next_day * 1440.0 + self.SHIFT_START
        # Planning commitments without physical cargo are released so the next morning
        # may be re-optimized.  Cargo ownership is intentionally preserved.
        for o in st.orders.values():
            if not o.cargo_on_vehicle and o.committed_vehicle_id:
                st.release_commitment(o.order_id, reason="OVERNIGHT_REPLAN")
        for v in st.vehicle_list:
            v.current_time = max(v.current_time, next_start)
            if v.active:
                v.status = "IDLE"
        self.vehicle_plans = {v.vehicle_id: [] for v in st.vehicle_list}
        self.departure_times = {v.vehicle_id: next_start for v in st.vehicle_list}
        self.departed = {v.vehicle_id: False for v in st.vehicle_list}
        self.display_time = next_start
        st.now = next_start
        self.day_closing = False
        self._apply_calibrated_traffic()
        self._start_due_departures()
        self._release_due_auto_orders()
        self._log("SYSTEM", f"Bắt đầu ngày {next_day+1}: tái đánh giá {len(self._current_carryover_orders())} đơn chuyển tiếp cùng các đơn mới trong ngày.")
        return True

    def _close_or_roll_day(self):
        if not self.day_closing or self.active_legs:
            return False
        # Once all current legs/returns finish, move to the next operating morning.
        if self._begin_next_day():
            self._dispatch_idle()
            return True
        day = min(self._day_index(), self.settings["days"] - 1)
        self._finalize_day_summary(day)
        remaining = self._current_carryover_orders()
        future = any(not x["released"] for x in self.auto_dynamic_schedule)
        if remaining or future:
            self.finish_status = "horizon_end"
            self.status_message = f"Đã hết {self.settings['days']} ngày mô phỏng; còn {len(remaining)} đơn chưa hoàn tất."
        else:
            self.finish_status = "completed"
            self.status_message = "Đã hoàn tất toàn bộ đơn trong thời lượng mô phỏng."
        self.running = False
        self._log("SYSTEM", self.status_message)
        return True

    def _record_telemetry(self):
        st = self.dispatcher.env.state
        active_orders = [o for o in st.orders.values() if o.released and not o.cancelled and not o.is_completed]
        active_legs = list(self.active_legs.values())
        point = {
            "timestamp": round(float(self.display_time), 3),
            "day_number": self._day_index() + 1,
            "minute_of_day": round(float(self._minute_of_day()), 3),
            "completed": int(sum(1 for o in st.orders.values() if o.is_completed)),
            "active_orders": int(len(active_orders)),
            "busy_vehicles": int(len(active_legs)),
            "carryover": int(len(self._current_carryover_orders())),
            "late_orders": int(sum(1 for o in st.orders.values() if o.late)),
            "distance_km": round(float(sum(v.distance for v in st.vehicle_list)), 3),
            "traffic_speed_kmh": round(float(self.profile_speed_kmh or 0.0), 3),
            "decisions": int(self.policy_decisions),
        }
        if not self.telemetry or self.telemetry[-1] != point:
            self.telemetry.append(point)
            self.telemetry = self.telemetry[-480:]

    def _decision_summary(self):
        if not self.decision_trace:
            return {}
        tr = self.decision_trace[-1]
        probs = [float(x.get("probability") or 0.0) for x in tr.get("top_candidates", [])]
        probs = [p for p in probs if p > 0]
        peak = max(probs) if probs else None
        dominance = None
        if len(probs) >= 2:
            dominance = peak - sorted(probs, reverse=True)[1]
        return {
            "chosen_order": tr.get("chosen_order"),
            "vehicle_id": tr.get("vehicle_id"),
            "latency_ms": tr.get("latency_ms"),
            "feasible_orders": tr.get("feasible_orders"),
            "chosen_probability": tr.get("chosen_probability"),
            "peak_probability": round(peak, 6) if peak is not None else None,
            "dominance_gap": round(dominance, 6) if dominance is not None else None,
        }

    def _record_decision_trace(self, vehicle, candidates, order, algorithm, latency_ms, action=None):
        st = self.dispatcher.env.state
        state = st.build_state(vehicle)
        rows = []
        probs = getattr(action, "probs", None)
        for cand in candidates:
            c, _ = self._candidate_for_order(vehicle, cand)
            rows.append({
                "order_id": cand.order_id,
                "operation": OP_LABELS.get(cand.next_operation, cand.next_operation),
                "travel_wait": round(float(c.cost), 3) if c is not None else None,
                "deadline": cand.latest_delivery_time,
                "priority": cand.priority,
                "probability": round(float(probs[cand.index]), 6) if probs is not None else None,
                "selected": cand.order_id == order.order_id,
            })
        if algorithm == "drl":
            rows.sort(key=lambda r: -(r["probability"] or 0.0))
        elif algorithm == "deadline":
            rows.sort(key=lambda r: (r["deadline"] or 1e12, r["travel_wait"] or 1e12))
        else:
            rows.sort(key=lambda r: (r["travel_wait"] or 1e12, r["order_id"]))
        idx = order.index
        trace = {
            "trace_id": f"TRACE-{self.policy_decisions+1:05d}",
            "timestamp": self.display_time,
            "day_number": self._day_index() + 1,
            "algorithm": algorithm,
            "algorithm_label": ALGORITHMS[algorithm],
            "vehicle_id": vehicle.vehicle_id,
            "vehicle_current_time": vehicle.current_time,
            "feasible_orders": len(candidates),
            "chosen_order": order.order_id,
            "chosen_operation": OP_LABELS.get(order.next_operation, order.next_operation),
            "latency_ms": round(float(latency_ms), 3),
            "entropy": round(float(action.entropy.item()), 6) if action is not None else None,
            "chosen_probability": round(float(action.prob), 6) if action is not None else None,
            "top_candidates": rows[:8],
            "state_shape": {"Z": list(state["Z"].shape), "U": list(state["U"].shape), "D": list(state["D"].shape)},
            "chosen_features": {
                "Z": [round(float(x), 4) for x in state["Z"][idx].tolist()],
                "U": [round(float(x), 4) for x in state["U"][idx].tolist()],
                "D": [round(float(x), 4) for x in state["D"][idx].tolist()],
            },
            "pipeline": [
                {"key": "fleet", "label": "Lọc xe khả thi / thứ tự thời gian tích lũy", "status": "done"},
                {"key": "state", "label": "Tạo trạng thái Z · U · D và feasibility mask", "status": "done"},
                {"key": "policy", "label": "Tính điểm / xác suất cho các đơn khả thi", "status": "done"},
                {"key": "select", "label": f"Chọn {order.order_id}", "status": "done"},
                {"key": "route", "label": "Khóa quyết định và bắt đầu chặng đường bộ", "status": "done"},
            ],
        }
        self.decision_trace.append(trace)
        self.decision_trace = self.decision_trace[-60:]
        return trace

    def _release_due_auto_orders(self):
        if not self.auto_dynamic_enabled:
            return
        for item in self.auto_dynamic_schedule:
            if item["released"] or item["release_time"] > self.display_time + 1e-9:
                continue
            self.create_order(pickup=item["pickup"], delivery=item["delivery"], weight=item["weight"],
                note=item["schedule_id"], flow_type=item["flow_type"], window_minutes=item["window_minutes"],
                priority=item["priority"], source="AUTO")
            item["released"] = True
            self.auto_dynamic_released += 1

    def create_order(self, *args, **kwargs):
        with self.lock:
            reserved = sum(not x["released"] for x in self.auto_dynamic_schedule)
            current = len(self.dispatcher.env.state.orders)
            if kwargs.get("source", "MANUAL") != "AUTO" and current + reserved >= 100:
                raise ValueError("Đã đạt giới hạn 100 đơn (gồm đơn đang chờ phát sinh). Hãy tạo ca mới.")
            if current >= 100:
                raise ValueError("Ca mô phỏng chỉ được có tối đa 100 đơn.")
            out = super().create_order(*args, **kwargs)
            if self.events and self.events[-1]["kind"] == "NEW_ORDER":
                self.events.pop()
            self._log("NEW_ORDER", f"{out.order_id}: đã nhận đơn mới; sẽ xét khi xe sẵn sàng bằng {ALGORITHMS[self.settings['algorithm']] }.", out.order_id)
            if self.finish_status in {"completed", "blocked", "waiting"}:
                self.finish_status = "paused"
                self.status_message = "Có đơn mới. Bấm Tiếp tục để chạy."
            return out

    def _assign_fallback_plan(self, order_id, preferred_vehicle_id=None):
        # Assignment occurs at a free vehicle's next boundary using the chosen policy.
        return None

    def _process_event(self, event):
        self.display_time = max(self.display_time, event.timestamp)
        result = self.dispatcher.env.apply_event(event)
        if not result.handled:
            raise ValueError(result.error or f"Không xử lý được {event.event_type}")
        self.dispatcher.event_log.append(result.as_log_row())
        self._log("EVENT", f"{event.event_type} · {event.order_id or event.vehicle_id or ''}",
                  event.order_id or "", event.vehicle_id or "")
        return None

    def _start_due_departures(self):
        self.departed = {vid: True for vid in self.departure_times}

    def _try_start_vehicle(self, vehicle_id, initial=False):
        if not self.running or vehicle_id in self.active_legs:
            return None
        st = self.dispatcher.env.state
        st.now = self.display_time
        self._apply_calibrated_traffic()
        v = st.vehicles[vehicle_id]
        if not v.is_available(self.display_time):
            return None

        # Outside the operating window, vehicles close the day by returning to depot.
        minute = self._minute_of_day()
        if self.day_closing or minute >= self.SHIFT_END or minute < self.SHIFT_START:
            start_lat, start_lon = self.road_positions.get(vehicle_id, [*local_latlon(v.x, v.y)])
            if abs(start_lat-DEPOT["lat"])+abs(start_lon-DEPOT["lon"]) < 1e-5:
                return None
            leg = self._build_return_leg(vehicle_id)
            if leg:
                leg["source"] = "Đóng ngày vận hành"
                self.active_legs[vehicle_id] = leg
                self._log("EXECUTE", f"{vehicle_id} kết thúc ngày và quay về kho.", vehicle_id=vehicle_id)
            return leg

        candidates = [o for o in st.orders.values() if self.dispatcher.env.mcdsm.order_feasible(v, o)
                      and self._candidate_for_order(v, o)[0] is not None]
        if candidates:
            t0 = time.perf_counter()
            probability = None
            action = None
            if self.settings["algorithm"] == "drl":
                mask = np.zeros(self.cfg.problem.M_max, dtype=np.float32)
                for o in candidates:
                    mask[o.index] = 1
                with torch.inference_mode():
                    action = self.dispatcher.agent.act(st.build_state(v), mask, mode="greedy")
                order = next(o for o in candidates if o.index == action.slot)
                probability = action.prob
            elif self.settings["algorithm"] == "deadline":
                order = min(candidates, key=lambda o: (o.latest_delivery_time or 1e9,
                            self._candidate_for_order(v, o)[0].cost, o.order_id))
            else:
                order = min(candidates, key=lambda o: (self._candidate_for_order(v, o)[0].cost, o.order_id))
            latency_ms = (time.perf_counter()-t0)*1000
            trace = self._record_decision_trace(v, candidates, order, self.settings["algorithm"], latency_ms, action)
            leg = self._build_leg(vehicle_id, order.order_id, source=ALGORITHMS[self.settings["algorithm"]])
            leg["trace_id"] = trace["trace_id"]
            leg["traffic_speed_kmh"] = self.profile_speed_kmh
            st.commit_order(order.order_id, vehicle_id)
            self._apply_recommendation_to_plan(dict(vehicle_id=vehicle_id, order_id=order.order_id))
            self.active_legs[vehicle_id] = leg
            self.policy_decisions += 1
            self.last_choice = dict(vehicle_id=vehicle_id, order_id=order.order_id,
                operation=OP_LABELS.get(order.next_operation, order.next_operation),
                algorithm=ALGORITHMS[self.settings["algorithm"]], probability=probability,
                trace_id=trace["trace_id"], latency_ms=round(latency_ms, 2))
            self._log("DECISION", f"{vehicle_id} → {order.order_id}: {self.last_choice['operation']} · {self.last_choice['algorithm']}", order.order_id, vehicle_id)
            return leg
        if not v.carrying:
            leg = self._build_return_leg(vehicle_id)
            if leg:
                self.active_legs[vehicle_id] = leg
                return leg
        return None

    def _dispatch_idle(self):
        # Same accumulated-time fairness criterion as MCDSM, restricted to free vehicles.
        st = self.dispatcher.env.state
        ranked = []
        for v in sorted(st.vehicle_list, key=lambda v: (v.current_time, v.index)):
            if v.vehicle_id in self.active_legs or not v.active:
                continue
            feasible = sum(1 for o in st.orders.values() if self.dispatcher.env.mcdsm.order_feasible(v, o))
            ranked.append({"vehicle_id": v.vehicle_id, "current_time": round(v.current_time, 3),
                           "working_time": round(v.working_time, 3), "feasible_orders": feasible})
        self.latest_vehicle_ranking = ranked
        for row in ranked:
            self._try_start_vehicle(row["vehicle_id"])

    def _ensure_road_routes(self):
        if self.road_router is None:
            return True
        pending, failures = 0, 0
        for vid, leg in self.active_legs.items():
            if leg.get("road_points"):
                continue
            start = self.road_positions.get(vid, [leg["start_lat"], leg["start_lon"]])
            end = [leg["target_lat"], leg["target_lon"]]
            leg.setdefault("road_key", f"{self.run_id}:{vid}:{leg['start_time']}:{leg['order_id']}")
            # Loading/unloading at the same stop has no vehicle movement.
            stationary = abs(leg["start_lat"]-end[0])+abs(leg["start_lon"]-end[1]) < 1e-7
            if stationary:
                leg["road_points"] = [start]
                leg["road_status"] = "ready"
                continue
            route = self.road_router.request([*start, *end])
            leg["road_status"] = route["status"]
            if route["status"] == "ready":
                leg["road_points"] = route["points"]
                leg["road_provider"] = route["provider"]
                self.road_positions[vid] = list(route["points"][0])
                self.road_stops[leg["service_location_id"]] = list(route["points"][-1])
                if leg.get("return_to_depot"):
                    self.road_depot = list(route["points"][-1])
                elif abs(leg["start_lat"]-DEPOT["lat"])+abs(leg["start_lon"]-DEPOT["lon"]) < 1e-7:
                    self.road_depot = list(route["points"][0])
            elif route["status"] == "error":
                failures += 1
            else:
                pending += 1
        self.routing_status = "error" if failures else "loading" if pending else "ready"
        self.routing_message = (f"Chưa tải được đường cho {failures} xe. Xe đang chờ; bấm Thử lại tuyến đường."
                                if failures else f"Đang chuẩn bị đường cho {pending} xe…" if pending else "")
        if self.road_depot:
            self.road_stops["SGNA-DEPOT-01"] = self.road_depot
            # Stationary depot loading tasks should share the accessible road entrance.
            for leg in self.active_legs.values():
                if len(leg.get("road_points", [])) == 1 and leg["service_location_id"] == "SGNA-DEPOT-01":
                    leg["road_points"] = [self.road_depot]
        return not pending and not failures

    def _routes_ready_or_retry(self):
        if self._ensure_road_routes():
            return True
        if self.routing_status == "error" and self.road_router:
            # Tự thử lại tuyến lỗi ở nhịp sau thay vì để xe chạy không có đường.
            self.road_router.retry_failed()
            return self._ensure_road_routes()
        return False

    def retry_routes(self):
        with self.lock:
            if self.road_router:
                self.road_router.retry_failed()
                self._ensure_road_routes()

    def _finish_leg(self, vehicle_id):
        leg = self.active_legs.get(vehicle_id)
        if leg and leg.get("road_points"):
            self.road_positions[vehicle_id] = list(leg["road_points"][-1])
        super()._finish_leg(vehicle_id)

    def start(self):
        with self.lock:
            self.running = True
            self.finish_status = "running"
            self.status_message = f"Đang điều phối ngày {self._day_index()+1}/{self.settings['days']}."
            self._start_due_departures()
            self._release_due_auto_orders()
            self._dispatch_idle()
            self._ensure_road_routes()
            self._check_end()
            self._record_telemetry()

    def pause(self):
        with self.lock:
            self.running = False
            if self.finish_status == "running":
                self.finish_status = "paused"
                self.status_message = "Đã tạm dừng. Bấm Tiếp tục khi bạn sẵn sàng."

    def _check_end(self):
        remaining = self._current_carryover_orders()
        future = any(not x["released"] for x in self.auto_dynamic_schedule)
        if self.active_legs:
            return
        if self.day_closing:
            self._close_or_roll_day()
            return
        # If all demand for the full horizon has been resolved, the run can end early.
        if not remaining and not future:
            self._finalize_day_summary(min(self._day_index(), self.settings["days"] - 1))
            self.finish_status = "completed"
            self.status_message = "Mô phỏng hoàn tất: tất cả đơn đã xử lý và đội xe đã về kho."
            self.running = False
            self._log("SYSTEM", self.status_message)
            return
        if future and not self.auto_dynamic_enabled and not remaining:
            self.finish_status = "waiting"
            self.status_message = "Còn đơn động chưa phát sinh. Bật đơn động rồi bấm Tiếp tục."
            self.running = False
            self._log("SYSTEM", self.status_message)

    def tick(self, accrue=True):
        """Advance simulated time; returns True while waiting for a road route.

        Each real-time tick adds `sim_minutes_per_tick` to a small time budget (capped at two
        ticks).  Time never advances while an active leg lacks road geometry — otherwise its
        progress grows while the vehicle stands still and it later jumps/dashes in a straight
        line.  Minutes held back while waiting stay in the budget, so follow-up polls
        (accrue=False) catch up as soon as the route arrives instead of slowing the run.
        """
        with self.lock:
            if not self.running:
                self._time_budget = 0.0
                return False
            if accrue:
                self._time_budget = min(getattr(self, "_time_budget", 0.0) + self.sim_minutes_per_tick,
                                        2 * self.sim_minutes_per_tick)
            if not self._routes_ready_or_retry():
                return True
            start_time = self.display_time
            waiting = False
            target = self.display_time + getattr(self, "_time_budget", 0.0)
            safety = 0
            while self.running and self.display_time < target - 1e-9 and safety < 500:
                safety += 1
                self._ensure_road_routes()
                day = self._day_index()
                day_end = self._day_end_abs(day)
                times = [target] + [l["completion_time"] for l in self.active_legs.values()]
                if not self.day_closing and self.display_time < day_end + 1e-9:
                    times.append(day_end)
                if self.auto_dynamic_enabled and not self.day_closing:
                    times += [x["release_time"] for x in self.auto_dynamic_schedule
                              if not x["released"] and x["release_time"] >= self.display_time - 1e-9]
                next_time = min(times)
                self.display_time = max(self.display_time, next_time)
                self.dispatcher.env.state.now = self.display_time
                self._apply_calibrated_traffic()

                # Complete service/return boundaries first.
                for vid in sorted(list(self.active_legs), key=lambda k: self.active_legs[k]["completion_time"]):
                    if self.active_legs[vid]["completion_time"] <= self.display_time + 1e-9:
                        self._finish_leg(vid)
                        if vid in self.deferred_breakdowns:
                            self.deferred_breakdowns.remove(vid)
                            super().vehicle_breakdown(vid)

                # Shift-end boundary: finish current legs, return fleet to depot, carry work over.
                if not self.day_closing and self.display_time >= day_end - 1e-9:
                    self.day_closing = True
                    self._log("SYSTEM", f"Đến 22:00 ngày {day+1}: dừng nhận tác vụ mới, hoàn tất chặng đang chạy và đưa xe về kho.")

                if not self.day_closing:
                    self._release_due_auto_orders()
                self._dispatch_idle()
                # Chặng mới (kể cả đơn động, quay về kho) phải có đường bộ trước khi thời gian trôi tiếp.
                if not self._ensure_road_routes():
                    waiting = True
                    break

                if self.day_closing and not self.active_legs:
                    if self._close_or_roll_day():
                        # A rollover jumps overnight to the next 08:00.  Stop this real-time
                        # tick here; the next browser tick resumes the new operating day.
                        break
                self._check_end()
                if next_time >= target - 1e-9:
                    break

            # An overnight rollover consumes far more than the budget, which simply resets it.
            self._time_budget = max(0.0, self._time_budget - (self.display_time - start_time))
            for leg in self.active_legs.values():
                leg["progress"] = min(1., max(0., (self.display_time-leg["start_time"]) /
                                               max(.001, leg["arrival_time"]-leg["start_time"])))
            self._record_telemetry()
            return waiting

    def manual_replan(self):
        with self.lock:
            self._dispatch_idle()
            return getattr(self, "last_choice", None)

    def vehicle_breakdown(self, vehicle_id):
        with self.lock:
            if vehicle_id not in self.dispatcher.env.state.vehicles:
                raise ValueError("Không tìm thấy xe.")
            if vehicle_id in self.active_legs:
                self.deferred_breakdowns.add(vehicle_id)
                self._log("SYSTEM", f"{vehicle_id}: sẽ dừng sau chặng hiện tại.")
            else:
                super().vehicle_breakdown(vehicle_id)

    def vehicle_recovery(self, vehicle_id):
        with self.lock:
            self.deferred_breakdowns.discard(vehicle_id)
            super().vehicle_recovery(vehicle_id)

    def cancel_order(self, order_id):
        with self.lock:
            if any(x["order_id"] == order_id for x in self.active_legs.values()):
                raise ValueError("Đơn đang được phục vụ; hãy chờ chặng hiện tại hoàn tất trước khi hủy.")
            super().cancel_order(order_id)

    def state(self):
        with self.lock:
            data = super().state()
            if self.road_depot:
                data["depot"] = {**data["depot"], "lat": self.road_depot[0], "lon": self.road_depot[1]}
            for order in data["orders"]:
                for role in ["pickup", "delivery"]:
                    place = order[role]
                    point = self.road_stops.get(place["service_location_id"])
                    if point:
                        place["lat"], place["lon"] = point
            for v in data["vehicles"]:
                i = int(v["vehicle_id"][1:])
                v["label"] = f"Xe {i:02d}"
                v["color"] = f"hsl({(i*137)%360}, 65%, 42%)"
                spec = getattr(self, "vehicle_specs", {}).get(v["vehicle_id"])
                if spec:
                    v["model"], v["volume_m3"], v["speed_factor"] = spec["model"], spec["volume_m3"], spec["speed_factor"]
                v["breakdown_pending"] = v["vehicle_id"] in self.deferred_breakdowns
                leg = self.active_legs.get(v["vehicle_id"])
                if leg and leg.get("road_points"):
                    progress = (self.display_time-leg["start_time"]) / max(.001, leg["arrival_time"]-leg["start_time"])
                    v["lat"], v["lon"] = road_position(leg["road_points"], progress)
                elif self.road_router:
                    v["lat"], v["lon"] = self.road_positions.get(v["vehicle_id"], self.road_depot or [DEPOT["lat"], DEPOT["lon"]])
                v["road_waiting"] = bool(leg and self.road_router and not leg.get("road_points"))
            data["settings"] = dict(self.settings)
            data["run_id"] = self.run_id
            data["routing"] = {"enabled": self.road_router is not None, "status": self.routing_status,
                               "message": self.routing_message}
            data["clock"]["advancing"] = self.running and self.routing_status == "ready"
            data["clock"]["minutes_per_tick"] = self.sim_minutes_per_tick
            # Keep enough completed segments for one buffered animation interval at 24 vehicles.
            # The browser replays motion ~2 s behind the server, so it needs the legs completed in that window
            # (longer at high simulation speeds).
            window = max(45.0, 4 * self.sim_minutes_per_tick)
            recent = [dict(x) for x in self.operation_history if x["completion_time"] >= self.display_time-window]
            data["operation_history"] = recent + [{**copy.deepcopy(x), "completed": False} for x in self.active_legs.values()]
            day_idx = min(self._day_index(), self.settings["days"] - 1)
            selected_traffic = traffic_payload(self.settings["traffic_region"], self.display_time,
                                               self.settings["start_weekday"])
            all_profiles = {key: traffic_payload(key, self.display_time, self.settings["start_weekday"])
                            for key in REGION_LABELS}
            carry = self._current_carryover_orders()
            data["run"] = dict(status=self.finish_status, message=self.status_message,
                total=len(data["orders"])+data["auto_dynamic"]["remaining"],
                cancelled=sum(o["cancelled"] for o in data["orders"]),
                distance_km=round(sum(v["distance"] for v in data["vehicles"]), 2),
                elapsed_minutes=round(self.display_time-self.SHIFT_START, 1), decisions=self.policy_decisions,
                last_choice=getattr(self, "last_choice", None))
            data["clock"].update({
                "day_number": day_idx + 1, "days_total": self.settings["days"],
                "weekday": selected_traffic["weekday_label"],
                "shift_start": hhmm(self._day_start_abs(day_idx)),
                "shift_end": hhmm(self._day_end_abs(day_idx)),
                "day_closing": self.day_closing,
            })
            data["calendar"] = {
                "day_index": day_idx, "day_number": day_idx + 1, "days_total": self.settings["days"],
                "weekday": selected_traffic["weekday_label"],
                "shift_start_minute": self._day_start_abs(day_idx),
                "shift_end_minute": self._day_end_abs(day_idx),
                "is_closing": self.day_closing,
                "daily_summaries": copy.deepcopy(self.daily_summaries),
                "carryover_current": len(carry),
                "carryover_ids": [o.order_id for o in carry[:40]],
            }
            data["traffic_profile"] = {
                "selected": selected_traffic,
                "regions": all_profiles,
                "manual_incident_factor": round(float(self.dispatcher.env.state.traffic.factor("ALL")), 3),
                "effective_note": "Tốc độ hiệu chỉnh theo giờ/ngày × hệ số sự cố thủ công (nếu có)",
            }
            latest_trace = copy.deepcopy(self.decision_trace[-1]) if self.decision_trace else None
            data["intelligence"] = {
                "algorithm": self.settings["algorithm"],
                "algorithm_label": ALGORITHMS[self.settings["algorithm"]],
                "latest": latest_trace,
                "history": copy.deepcopy(self.decision_trace[-16:]),
                "vehicle_ranking": copy.deepcopy(self.latest_vehicle_ranking[:12]),
                "active_trace_ids": [x.get("trace_id") for x in self.active_legs.values() if x.get("trace_id")],
                "decision_summary": self._decision_summary(),
            }
            data["analytics"] = {
                "telemetry": copy.deepcopy(self.telemetry[-240:]),
                "history_window_note": "Cửa sổ hiển thị gần nhất cho biểu đồ thời gian thực.",
            }
            data["model"]["used"] = self.settings["algorithm"] == "drl"
            data["summary"]["decisions"] = self.policy_decisions
            data["summary"]["replans"] = self.policy_decisions
            data["summary"]["carryover"] = len(carry)
            data["summary"]["traffic_speed_kmh"] = round(float(self.profile_speed_kmh or selected_traffic["speed_kmh"]), 1)
            data["model"]["training_scale"] = self.training_scale
            data["model"]["scale_warning"] = (self.settings["vehicles"] > self.training_scale["vehicles"]
                    or self.settings["orders"] > self.training_scale["orders"])
            return data
