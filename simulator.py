from __future__ import annotations

import copy
import json
import math
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import torch  # noqa: E402
from config import Config  # noqa: E402
from entities import (  # noqa: E402
    CARGO_ON_VEHICLE,
    FLOW_INBOUND,
    FLOW_INBOUND_PRELOADED,
    FLOW_OUTBOUND,
    FLOW_PICKUP_DELIVERY,
    OP_CUSTOMER_DELIVERY,
    OP_CUSTOMER_PICKUP,
    OP_DEPOT_LOAD,
    OP_DEPOT_UNLOAD,
    ORDER_DYNAMIC,
    ORDER_STATIC,
    ROLE_DELIVERY,
    ROLE_DEPOT,
    ROLE_PICKUP,
    Event,
    Order,
    ProblemInstance,
    ServiceLocation,
    Vehicle,
)
from realtime_inference import DispatchDecision, RealTimeDispatcher  # noqa: E402
from utils import hhmm  # noqa: E402

# ======================================================================================
# BẢN ĐỒ / KHU VỰC VẬN HÀNH
# ======================================================================================
HCMC_CENTER = (10.7769, 106.7009)
DEPOT = {
    "id": "SGNA_DEPOT",
    "name": "Kho trung tâm SGNA",
    "address": "Khu vực sân bay Tân Sơn Nhất, TP.HCM",
    "lat": 10.8188,
    "lon": 106.6519,
    "district": "Tân Bình",
}

# Các điểm mô phỏng đều là các mốc thực/địa danh khu vực TP.HCM. Đây là lớp hiển thị
# prototype; mô hình DRL vẫn nhận tọa độ cục bộ (km) thông qua local_xy().
SAMPLE_LOCATIONS: Dict[str, Dict[str, Any]] = {
    # Trung tâm
    "BEN_THANH": {"name": "Chợ Bến Thành", "address": "Chợ Bến Thành, Quận 1, TP.HCM", "lat": 10.7724, "lon": 106.6980, "district": "Quận 1"},
    "NGUYEN_HUE": {"name": "Phố đi bộ Nguyễn Huệ", "address": "Nguyễn Huệ, Quận 1, TP.HCM", "lat": 10.7758, "lon": 106.7036, "district": "Quận 1"},
    "HCMUT": {"name": "ĐH Bách Khoa TP.HCM", "address": "268 Lý Thường Kiệt, Quận 10, TP.HCM", "lat": 10.7720, "lon": 106.6579, "district": "Quận 10"},
    "PHU_NHUAN": {"name": "Phú Nhuận", "address": "Nguyễn Văn Trỗi, Phú Nhuận, TP.HCM", "lat": 10.7983, "lon": 106.6792, "district": "Phú Nhuận"},
    # Nam TP.HCM — use case trọng tâm
    "CRESCENT": {"name": "Crescent Mall", "address": "101 Tôn Dật Tiên, Quận 7, TP.HCM", "lat": 10.7296, "lon": 106.7180, "district": "Quận 7"},
    "SC_VIVO": {"name": "SC VivoCity", "address": "1058 Nguyễn Văn Linh, Quận 7, TP.HCM", "lat": 10.7293, "lon": 106.7037, "district": "Quận 7"},
    "LOTTE_Q7": {"name": "Lotte Mart Quận 7", "address": "469 Nguyễn Hữu Thọ, Quận 7, TP.HCM", "lat": 10.7411, "lon": 106.7008, "district": "Quận 7"},
    "PHU_MY_HUNG": {"name": "Phú Mỹ Hưng", "address": "Nguyễn Đức Cảnh, Quận 7, TP.HCM", "lat": 10.7328, "lon": 106.7212, "district": "Quận 7"},
    "NHA_BE": {"name": "Khu trung tâm Nhà Bè", "address": "Nguyễn Hữu Thọ, Nhà Bè, TP.HCM", "lat": 10.6955, "lon": 106.7048, "district": "Nhà Bè"},
    # Đông
    "LANDMARK81": {"name": "Landmark 81", "address": "Vinhomes Central Park, Bình Thạnh, TP.HCM", "lat": 10.7948, "lon": 106.7219, "district": "Bình Thạnh"},
    "THAO_DIEN": {"name": "Thảo Điền", "address": "Thảo Điền, TP Thủ Đức, TP.HCM", "lat": 10.8045, "lon": 106.7355, "district": "TP Thủ Đức"},
    "GIGAMALL": {"name": "Gigamall Thủ Đức", "address": "240 Phạm Văn Đồng, TP Thủ Đức, TP.HCM", "lat": 10.8281, "lon": 106.7218, "district": "TP Thủ Đức"},
    "SALA": {"name": "Khu đô thị Sala", "address": "Mai Chí Thọ, TP Thủ Đức, TP.HCM", "lat": 10.7714, "lon": 106.7246, "district": "TP Thủ Đức"},
    # Tây
    "TAN_PHU": {"name": "Tân Phú", "address": "Lũy Bán Bích, Tân Phú, TP.HCM", "lat": 10.7902, "lon": 106.6284, "district": "Tân Phú"},
    "AEON_TAN_PHU": {"name": "AEON Mall Tân Phú", "address": "30 Bờ Bao Tân Thắng, Tân Phú, TP.HCM", "lat": 10.8018, "lon": 106.6177, "district": "Tân Phú"},
    "BINH_TAN": {"name": "Bình Tân", "address": "Tỉnh lộ 10, Bình Tân, TP.HCM", "lat": 10.7626, "lon": 106.6032, "district": "Bình Tân"},
    "AN_DONG": {"name": "An Đông", "address": "An Dương Vương, Quận 5, TP.HCM", "lat": 10.7584, "lon": 106.6707, "district": "Quận 5"},
    # Bắc
    "GO_VAP": {"name": "Gò Vấp", "address": "Nguyễn Oanh, Gò Vấp, TP.HCM", "lat": 10.8388, "lon": 106.6650, "district": "Gò Vấp"},
    "QUANG_TRUNG": {"name": "Công viên phần mềm Quang Trung", "address": "Tô Ký, Quận 12, TP.HCM", "lat": 10.8507, "lon": 106.6278, "district": "Quận 12"},
    "HOC_MON": {"name": "Hóc Môn", "address": "Quốc lộ 22, Hóc Môn, TP.HCM", "lat": 10.8864, "lon": 106.5922, "district": "Hóc Môn"},
    # Tây Nam
    "BINH_CHANH": {"name": "Bình Chánh", "address": "Khu Trung Sơn, Bình Chánh, TP.HCM", "lat": 10.7377, "lon": 106.6885, "district": "Bình Chánh"},
    "TRUNG_SON": {"name": "Khu Trung Sơn", "address": "Đường số 9A, Bình Chánh, TP.HCM", "lat": 10.7363, "lon": 106.6868, "district": "Bình Chánh"},
    "HIM_LAM": {"name": "Khu dân cư Him Lam", "address": "Nguyễn Thị Thập, Quận 7, TP.HCM", "lat": 10.7398, "lon": 106.6994, "district": "Quận 7"},
    "QUAN_8": {"name": "Quận 8", "address": "Tạ Quang Bửu, Quận 8, TP.HCM", "lat": 10.7384, "lon": 106.6659, "district": "Quận 8"},
}

VEHICLE_COLORS = {
    "V1": "#1D4ED8", "V2": "#7C3AED", "V3": "#0F766E",
    "V4": "#C2410C", "V5": "#B91C1C", "V6": "#0369A1",
}
VEHICLE_ZONES = {
    "V1": "Nam TP.HCM", "V2": "Đông TP.HCM", "V3": "Trung tâm",
    "V4": "Tây TP.HCM", "V5": "Bắc TP.HCM", "V6": "Tây Nam TP.HCM",
}

FLOW_LABELS = {
    FLOW_PICKUP_DELIVERY: "Lấy tại khách → giao cho khách",
    FLOW_OUTBOUND: "Lấy tại khách → dỡ tại kho",
    FLOW_INBOUND: "Chất tại kho → giao cho khách",
    FLOW_INBOUND_PRELOADED: "Đã chất sẵn tại kho → giao cho khách",
}
OP_LABELS = {
    OP_CUSTOMER_PICKUP: "Lấy hàng tại khách",
    OP_CUSTOMER_DELIVERY: "Giao hàng cho khách",
    OP_DEPOT_LOAD: "Chất hàng tại kho",
    OP_DEPOT_UNLOAD: "Dỡ hàng tại kho",
}
OP_COMPLETE_EVENT = {
    OP_CUSTOMER_PICKUP: "CUSTOMER_PICKUP_COMPLETE",
    OP_CUSTOMER_DELIVERY: "CUSTOMER_DELIVERY_COMPLETE",
    OP_DEPOT_LOAD: "DEPOT_LOAD_COMPLETE",
    OP_DEPOT_UNLOAD: "DEPOT_UNLOAD_COMPLETE",
}


def local_xy(lat: float, lon: float) -> Tuple[float, float]:
    return ((lon - HCMC_CENTER[1]) * 109.0, (lat - HCMC_CENTER[0]) * 111.0)


def local_latlon(x: float, y: float) -> Tuple[float, float]:
    return (HCMC_CENTER[0] + y / 111.0, HCMC_CENTER[1] + x / 109.0)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def _point_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": d.get("name", ""), "address": d.get("address", ""),
        "lat": float(d["lat"]), "lon": float(d["lon"]),
        "district": d.get("district", "TP.HCM"),
    }


class SimulationEngine:
    """Lớp vận hành của prototype đặt trên RealTimeDispatcher hiện tại.

    Hai lớp được tách rõ:
      1) LỚP VẬN HÀNH: mỗi xe có manifest đầu ca, giờ xuất phát riêng và một tuyến còn lại.
         Nhiều xe có thể chạy đồng thời. Đây là cách mô phỏng thực tế kiểu FedEx.
      2) LỚP QUYẾT ĐỊNH: MCDSM + PESDVS vẫn chỉ tạo *next-action recommendation*.
         Khuyến nghị mới có thể đổi thứ tự phần tuyến còn lại hoặc chen pickup mới; xe đang
         giữa chặng không bị quay đầu/teleport.

    Core model không bị sửa. Prototype chỉ bổ sung orchestration / visualization layer.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.sim_minutes_per_tick = 2.0  # 1 giây thật = 2 phút mô phỏng
        self.running = False
        self.auto_apply_decisions = True
        # V7: bộ sinh nhu cầu động chạy tự động trong suốt ca. Người dùng vẫn có thể
        # tạo đơn thủ công từ trang /customer; hai nguồn được phân biệt rõ trên dashboard.
        self.auto_dynamic_enabled = True
        self.auto_dynamic_schedule: List[Dict[str, Any]] = []
        self.auto_dynamic_released = 0
        self.auto_dynamic_skipped = 0
        self.order_seq = 100
        self.auto_order_seq = 0
        self.order_origins: Dict[str, str] = {}
        self.initial_plan_assignments: Dict[str, str] = {}
        self.decision_seq = 0
        self.scenario_seed = int(time.time()) % 100000
        self.active_legs: Dict[str, Dict[str, Any]] = {}
        self.pending_decisions: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.operation_history: List[Dict[str, Any]] = []
        self.vehicle_plans: Dict[str, List[str]] = {}
        self.departure_times: Dict[str, float] = {}
        self.departed: Dict[str, bool] = {}
        self._build_runtime(self.scenario_seed)

    # ------------------------------------------------------------------ model/runtime
    def _checkpoint_path(self) -> Path:
        current = CORE / "checkpoints" / "current_pilot_best.pt"
        return current if current.exists() else CORE / "checkpoints" / "best_actor_critic.pt"

    def _config_from_checkpoint(self, path: Path) -> Config:
        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
            cfg = Config.from_dict(raw.get("config", {}))
        except Exception:
            cfg = Config()
        cfg.inference.checkpoint = str(path)
        cfg.inference.auto_approve = False
        cfg.inference.decode_mode = "greedy"
        cfg.inference.write_logs = False
        cfg.mcdsm.lock_committed_orders = True
        cfg.clock.day_start = 480.0
        cfg.clock.day_end = max(float(cfg.clock.day_end), 1320.0)
        return cfg

    def _build_runtime(self, seed: Optional[int] = None) -> None:
        self.scenario_seed = int(seed if seed is not None else self.scenario_seed)
        self.rng = random.Random(self.scenario_seed)
        ckpt = self._checkpoint_path()
        self.cfg = self._config_from_checkpoint(ckpt)
        instance = self._initial_instance()
        self.dispatcher = RealTimeDispatcher(instance, self.cfg, checkpoint=str(ckpt))
        self.display_time = float(self.cfg.clock.day_start)
        self.active_legs = {}
        self.pending_decisions = {}
        self.events = []
        self.operation_history = []
        self.decision_seq = 0
        self.running = False
        self._build_departure_schedule()
        self._build_initial_plans()
        self._build_auto_dynamic_schedule()

        # Khởi tạo ca ở tầng state nhưng KHÔNG gọi dispatcher.process_event(), vì đầu ca
        # không cần AI "cho phép" từng xe xuất kho. Xe sẽ tự rời kho theo departure_time.
        result = self.dispatcher.env.apply_event(Event(
            timestamp=self.display_time, event_type="SHIFT_START", source="PROTOTYPE", event_id="SHIFT-START"
        ))
        self._log("SYSTEM", "Khởi tạo ca vận hành. Các xe có danh sách hàng đầu ca và tự xuất kho theo lịch; mô hình chỉ can thiệp vào quyết định kế tiếp khi có ranh giới vận hành/sự kiện.")
        self._log("MANIFEST", "Kế hoạch đầu ca đã được tạo cho cả ba luồng: giao từ kho, lấy về kho và lấy–giao trực tiếp. Hàng delivery đầu ca được preload; các tác vụ pickup tĩnh đã biết trước được đưa vào tuyến khu vực.")
        self._log("SYSTEM", f"Bộ sinh đơn động tự động đang BẬT: {len(self.auto_dynamic_schedule)} yêu cầu được phân bổ xuyên suốt ca, nhưng chỉ được tiết lộ cho mô hình đúng thời điểm phát sinh.")
        self._log("SYSTEM", f"Mã ngẫu nhiên kịch bản: {self.scenario_seed}. Trạng thái bắt đầu ca: {'Hợp lệ' if result.handled else result.error}.")

    def reset(self, new_seed: bool = False) -> None:
        with self.lock:
            seed = (int(time.time() * 1000) % 1000000) if new_seed else self.scenario_seed
            self._build_runtime(seed)

    def start(self) -> None:
        with self.lock:
            if not self.running:
                self.running = True
                self._log("SYSTEM", "Bắt đầu mô phỏng vận hành. Các xe đến giờ xuất phát sẽ tự rời kho theo kế hoạch đầu ca.")
                self._start_due_departures()

    def pause(self) -> None:
        with self.lock:
            self.running = False
            self._log("SYSTEM", "Tạm dừng đồng hồ mô phỏng.")

    def set_speed(self, minutes_per_tick: float) -> None:
        with self.lock:
            self.sim_minutes_per_tick = max(0.25, min(15.0, float(minutes_per_tick)))
            self._log("SYSTEM", f"Tốc độ mô phỏng: 1 giây = {self.sim_minutes_per_tick:g} phút mô phỏng.")

    # ------------------------------------------------------------------ scenario data
    def _initial_instance(self) -> ProblemInstance:
        """Tạo kịch bản V7 có đủ ba business flow và tải công việc trải dài cả ngày.

        FLOW 1  depot -> pickup -> depot            = FLOW_OUTBOUND
        FLOW 2  depot -> delivery -> depot          = FLOW_INBOUND_PRELOADED / FLOW_INBOUND
        FLOW 3  depot -> pickup -> delivery -> depot = FLOW_PICKUP_DELIVERY

        ``INBOUND_PRELOADED`` chỉ là cách triển khai kỹ thuật của Flow 2 ở đầu ca: hàng
        delivery đã nằm trên xe trước giờ departure. Chặng quay về depot là orchestration
        của tour, không phải một action Actor riêng.
        """
        self.order_origins = {}
        self.initial_plan_assignments = {}
        depot_x, depot_y = local_xy(DEPOT["lat"], DEPOT["lon"])
        locations: Dict[str, List[ServiceLocation]] = {
            DEPOT["id"]: [ServiceLocation(
                service_location_id="SGNA-DEPOT-01", entity_id=DEPOT["id"], entity_role=ROLE_DEPOT,
                x=depot_x, y=depot_y, begin_time=480.0, end_time=1320.0,
                service_duration=8.0, available=True, lat=DEPOT["lat"], lon=DEPOT["lon"],
                meta={"name": DEPOT["name"], "address": DEPOT["address"]},
            )]
        }
        vehicles = [
            Vehicle("V1", "VAN", 1500.0, 0.58, DEPOT["id"], 480.0, 1320.0, index=0),
            Vehicle("V2", "VAN", 1500.0, 0.56, DEPOT["id"], 480.0, 1320.0, index=1),
            Vehicle("V3", "VAN", 1500.0, 0.60, DEPOT["id"], 480.0, 1320.0, index=2),
            Vehicle("V4", "TRUCK", 6000.0, 0.50, DEPOT["id"], 480.0, 1320.0, index=3),
            Vehicle("V5", "TRUCK", 6000.0, 0.48, DEPOT["id"], 480.0, 1320.0, index=4),
            Vehicle("V6", "TRUCK", 6000.0, 0.52, DEPOT["id"], 480.0, 1320.0, index=5),
        ]
        for v in vehicles:
            v.speed_profile = {8: 0.85, 9: 0.78, 10: 0.92, 11: 0.90, 12: 0.86,
                               13: 0.80, 14: 0.75, 15: 0.70, 16: 0.78, 17: 0.72,
                               18: 0.78, 19: 0.90, 20: 0.96, 21: 1.0}

        orders: List[Order] = []

        # ------------------------------------------------------------------ FLOW 2 STATIC
        # 24 delivery đầu ca. Time windows được trải rộng đến đầu/giữa buổi chiều để
        # initial plan không kết thúc quá sớm, đồng thời vẫn còn dư địa chèn pickup động.
        preload_specs = [
            # V1 — Nam TP.HCM, use case chính
            ("F2-S01", "V1", "LOTTE_Q7", 150.0, 500.0, 690.0),
            ("F2-S02", "V1", "SC_VIVO", 165.0, 520.0, 735.0),
            ("F2-S03", "V1", "CRESCENT", 145.0, 550.0, 795.0),
            ("F2-S04", "V1", "PHU_MY_HUNG", 155.0, 590.0, 845.0),
            ("F2-S05", "V1", "NHA_BE", 185.0, 640.0, 900.0),
            # V2 — Đông
            ("F2-E01", "V2", "LANDMARK81", 190.0, 500.0, 700.0),
            ("F2-E02", "V2", "SALA", 175.0, 530.0, 750.0),
            ("F2-E03", "V2", "THAO_DIEN", 205.0, 575.0, 815.0),
            ("F2-E04", "V2", "GIGAMALL", 225.0, 625.0, 880.0),
            # V3 — Trung tâm
            ("F2-C01", "V3", "PHU_NHUAN", 145.0, 495.0, 665.0),
            ("F2-C02", "V3", "BEN_THANH", 160.0, 525.0, 720.0),
            ("F2-C03", "V3", "NGUYEN_HUE", 150.0, 565.0, 775.0),
            ("F2-C04", "V3", "HCMUT", 180.0, 615.0, 835.0),
            # V4 — Tây
            ("F2-W01", "V4", "AEON_TAN_PHU", 620.0, 500.0, 720.0),
            ("F2-W02", "V4", "TAN_PHU", 650.0, 540.0, 780.0),
            ("F2-W03", "V4", "BINH_TAN", 780.0, 600.0, 850.0),
            ("F2-W04", "V4", "AN_DONG", 520.0, 650.0, 905.0),
            # V5 — Bắc
            ("F2-N01", "V5", "GO_VAP", 760.0, 520.0, 735.0),
            ("F2-N02", "V5", "QUANG_TRUNG", 900.0, 585.0, 825.0),
            ("F2-N03", "V5", "HOC_MON", 980.0, 660.0, 920.0),
            # V6 — Tây Nam
            ("F2-SW01", "V6", "AN_DONG", 560.0, 505.0, 690.0),
            ("F2-SW02", "V6", "QUAN_8", 660.0, 555.0, 755.0),
            ("F2-SW03", "V6", "TRUNG_SON", 610.0, 615.0, 825.0),
            ("F2-SW04", "V6", "BINH_CHANH", 720.0, 675.0, 900.0),
        ]
        for oid, vid, sid, weight, begin, end in preload_specs:
            place = SAMPLE_LOCATIONS[sid]
            consignee = f"{oid}-D"
            self._add_customer_location(locations, consignee, ROLE_DELIVERY, place, begin, end, 9.0)
            orders.append(Order(
                order_id=oid, consignor_id=DEPOT["id"], consignee_id=consignee,
                cargo_weight=weight, order_class=ORDER_STATIC, release_time=480.0,
                created_timestamp=450.0, priority=1.0, latest_delivery_time=end,
                flow_type=FLOW_INBOUND_PRELOADED, source_depot_id=DEPOT["id"],
                preloaded_vehicle_id=vid,
            ))
            self.order_origins[oid] = "STATIC"
            self.initial_plan_assignments[oid] = vid

        # ------------------------------------------------------------- FLOW 1 + 3 STATIC
        # Các order này đã biết trước lúc 08:00 nên Initial Planning được phép đưa chúng
        # vào tuyến khu vực. Chúng KHÔNG preload hàng: khi tới pickup mới tăng tải xe.
        static_mixed = [
            # oid, flow, vehicle, pickup, delivery, kg, pickup-begin, deadline
            ("F1-S01", FLOW_OUTBOUND, "V1", "HIM_LAM", None, 90.0, 560.0, 760.0),
            ("F3-S01", FLOW_PICKUP_DELIVERY, "V1", "SC_VIVO", "NHA_BE", 75.0, 610.0, 860.0),
            ("F1-E01", FLOW_OUTBOUND, "V2", "THAO_DIEN", None, 110.0, 575.0, 790.0),
            ("F3-E01", FLOW_PICKUP_DELIVERY, "V2", "SALA", "GIGAMALL", 95.0, 625.0, 875.0),
            ("F1-C01", FLOW_OUTBOUND, "V3", "HCMUT", None, 80.0, 550.0, 750.0),
            ("F3-C01", FLOW_PICKUP_DELIVERY, "V3", "BEN_THANH", "PHU_NHUAN", 70.0, 600.0, 825.0),
            ("F1-W01", FLOW_OUTBOUND, "V4", "TAN_PHU", None, 280.0, 590.0, 820.0),
            ("F3-W01", FLOW_PICKUP_DELIVERY, "V4", "AEON_TAN_PHU", "BINH_TAN", 340.0, 650.0, 910.0),
            ("F1-N01", FLOW_OUTBOUND, "V5", "GO_VAP", None, 310.0, 600.0, 840.0),
            ("F3-N01", FLOW_PICKUP_DELIVERY, "V5", "QUANG_TRUNG", "HOC_MON", 360.0, 670.0, 940.0),
            ("F1-SW01", FLOW_OUTBOUND, "V6", "QUAN_8", None, 260.0, 585.0, 810.0),
            ("F3-SW01", FLOW_PICKUP_DELIVERY, "V6", "TRUNG_SON", "BINH_CHANH", 300.0, 645.0, 900.0),
        ]
        for oid, flow, vid, pickup_sid, delivery_sid, weight, begin, end in static_mixed:
            consignor_id = f"{oid}-P"
            pickup = SAMPLE_LOCATIONS[pickup_sid]
            self._add_customer_location(locations, consignor_id, ROLE_PICKUP, pickup, begin, min(end - 25.0, begin + 150.0), 8.0)
            consignee_id = DEPOT["id"]
            dest_depot_id = DEPOT["id"] if flow == FLOW_OUTBOUND else ""
            if flow == FLOW_PICKUP_DELIVERY:
                consignee_id = f"{oid}-D"
                delivery = SAMPLE_LOCATIONS[delivery_sid]
                self._add_customer_location(locations, consignee_id, ROLE_DELIVERY, delivery, begin + 20.0, end, 9.0)
            orders.append(Order(
                order_id=oid, consignor_id=consignor_id, consignee_id=consignee_id,
                cargo_weight=weight, order_class=ORDER_STATIC, release_time=480.0,
                created_timestamp=455.0, priority=1.5, latest_delivery_time=end,
                flow_type=flow, dest_depot_id=dest_depot_id,
            ))
            self.order_origins[oid] = "STATIC"
            self.initial_plan_assignments[oid] = vid

        return ProblemInstance(
            vehicles=vehicles, orders=orders, service_locations=locations,
            speed_profile={}, events=[], day="HCMC-PROTOTYPE-V7", data_source_tag="PROTOTYPE_3FLOW_DYNAMIC",
            meta={
                "depot": DEPOT,
                "note": "Initial zone planning + three operational flows + automatic online demand",
                "business_flows": {
                    "FLOW_1": "Depot → Pickup → Depot",
                    "FLOW_2": "Depot → Delivery → Depot",
                    "FLOW_3": "Depot → Pickup → Delivery → Depot",
                },
            },
        )

    @staticmethod
    def _add_customer_location(locations: Dict[str, List[ServiceLocation]], entity_id: str,
                               role: str, place: Dict[str, Any], begin: float, end: float,
                               service_duration: float) -> str:
        x, y = local_xy(float(place["lat"]), float(place["lon"]))
        suffix = "P" if role == ROLE_PICKUP else "D"
        sl_id = f"{entity_id}-{suffix}-01"
        locations.setdefault(entity_id, []).append(ServiceLocation(
            service_location_id=sl_id, entity_id=entity_id, entity_role=role,
            x=x, y=y, begin_time=float(begin), end_time=float(end),
            service_duration=float(service_duration), available=True,
            lat=float(place["lat"]), lon=float(place["lon"]),
            meta={"name": place.get("name", entity_id), "address": place.get("address", "")},
        ))
        return sl_id

    def _build_departure_schedule(self) -> None:
        # 4 xe ra lúc 08:00; 2 xe còn lại xuất phát trễ ngẫu nhiên nhưng tái lập theo seed.
        self.departure_times = {
            "V1": 480.0, "V2": 480.0, "V3": 480.0, "V4": 480.0,
            "V5": 480.0 + float(self.rng.randint(7, 16)),
            "V6": 480.0 + float(self.rng.randint(18, 35)),
        }
        self.departed = {vid: False for vid in self.departure_times}

    def _build_initial_plans(self) -> None:
        """Initial Planning: phân vùng + manifest + sequence baseline trước khi ca chạy.

        Hàng Flow 2 đã preload được giữ đúng xe. Các Flow 1/3 tĩnh được gán theo khu vực
        bằng ``initial_plan_assignments``. Đây là operational baseline, KHÔNG gắn nhãn là
        output của Actor. Khi có event online, MCDSM + PESDVS được quyền thay đổi phần
        tuyến còn lại ở ranh giới tác vụ.
        """
        st = self.dispatcher.env.state
        self.vehicle_plans = {v.vehicle_id: [] for v in st.vehicle_list}
        for v in st.vehicle_list:
            assigned = []
            for oid, o in st.orders.items():
                if o.is_completed or o.cancelled or not o.released:
                    continue
                if oid in v.carrying or self.initial_plan_assignments.get(oid) == v.vehicle_id:
                    assigned.append(oid)

            current = (DEPOT["lat"], DEPOT["lon"])
            current_time = float(self.departure_times.get(v.vehicle_id, 480.0))
            remaining = list(dict.fromkeys(assigned))
            plan: List[str] = []
            baseline_speed_kmh = 24.0 if v.vehicle_type == "VAN" else 21.0

            def target_of(o: Order) -> Dict[str, Any]:
                pickup, delivery = self._order_locations(o)
                if o.next_operation == OP_CUSTOMER_PICKUP:
                    return pickup
                if o.next_operation == OP_CUSTOMER_DELIVERY:
                    return delivery
                return _point_payload(DEPOT)

            while remaining:
                def score(oid: str) -> float:
                    o = st.orders[oid]
                    target = target_of(o)
                    d = haversine_km(current, (target["lat"], target["lon"]))
                    travel_min = d / baseline_speed_kmh * 60.0
                    arrival = current_time + travel_min
                    begin = float(target.get("begin_time", current_time))
                    end = float(target.get("end_time", o.latest_delivery_time or 1320.0))
                    wait = max(0.0, begin - arrival)
                    late = max(0.0, arrival - end)
                    # Pickup về depot được ưu tiên vừa phải khi xe đang đi gần; Flow 3 giữ
                    # deadline mạnh hơn vì còn một stage delivery phía sau.
                    flow_bias = -4.0 if o.flow_type == FLOW_OUTBOUND else (-2.0 if o.flow_type == FLOW_PICKUP_DELIVERY else 0.0)
                    urgency = max(0.0, end - current_time) * 0.006
                    return travel_min + 0.65 * wait + 14.0 * late + urgency + flow_bias

                best = min(remaining, key=score)
                plan.append(best)
                o = st.orders[best]
                target = target_of(o)
                d = haversine_km(current, (target["lat"], target["lon"]))
                travel_min = d / baseline_speed_kmh * 60.0
                arrival = current_time + travel_min
                current_time = max(arrival, float(target.get("begin_time", arrival))) + float(target.get("service_duration", 8.0) or 8.0)
                current = (target["lat"], target["lon"])
                remaining.remove(best)
            self.vehicle_plans[v.vehicle_id] = plan

    def _build_auto_dynamic_schedule(self) -> None:
        """Sinh 54 yêu cầu động ẩn, trải từ ~08:45 đến ~19:30.

        Tần suất trung bình khoảng 5 đơn/giờ, đủ để 6 xe còn việc trong phần lớn ca thay
        vì hết backlog vào khoảng 11:00. Mỗi xe nhận một ``preferred_vehicle_id`` theo
        khu vực làm fallback vận hành; MCDSM + PESDVS vẫn có thể chọn xe khác khi event
        thực sự được release. Không một order nào trong lịch này được đưa vào state sớm.
        """
        zone_points = {
            "V1": ["HIM_LAM", "LOTTE_Q7", "SC_VIVO", "CRESCENT", "PHU_MY_HUNG", "NHA_BE"],
            "V2": ["LANDMARK81", "SALA", "THAO_DIEN", "GIGAMALL"],
            "V3": ["PHU_NHUAN", "BEN_THANH", "NGUYEN_HUE", "HCMUT"],
            "V4": ["AEON_TAN_PHU", "TAN_PHU", "BINH_TAN", "AN_DONG"],
            "V5": ["GO_VAP", "QUANG_TRUNG", "HOC_MON"],
            "V6": ["AN_DONG", "QUAN_8", "TRUNG_SON", "BINH_CHANH"],
        }
        vehicles = ["V1", "V2", "V3", "V4", "V5", "V6"]
        flows = [FLOW_OUTBOUND, FLOW_PICKUP_DELIVERY, FLOW_INBOUND]
        schedule: List[Dict[str, Any]] = []
        n_events = 54
        start_minute = 525.0
        interval = 12.0
        for i in range(n_events):
            vid = vehicles[i % len(vehicles)]
            flow = flows[(i + i // len(vehicles)) % len(flows)]
            pts = zone_points[vid]
            cycle = i // len(vehicles)
            p = pts[cycle % len(pts)]
            d = pts[(cycle + 1 + (i % 2)) % len(pts)]
            if d == p:
                d = pts[(cycle + 1) % len(pts)]
            if flow == FLOW_INBOUND:
                p = None
            elif flow == FLOW_OUTBOUND:
                d = None

            is_truck = vid in {"V4", "V5", "V6"}
            base_weight = 165.0 if is_truck else 62.0
            weight = base_weight + float((i * 17) % (150 if is_truck else 55))
            tw = float(145 + (i % 5) * 15)       # 145–205 phút
            priority = float(2.0 + (i % 4) * 0.5)
            release = start_minute + i * interval + self.rng.randint(-3, 3)
            schedule.append({
                "schedule_id": f"AUTO-{i+1:02d}", "release_time": float(release),
                "flow_type": flow, "preferred_vehicle_id": vid,
                "pickup_id": p, "delivery_id": d, "weight": round(weight, 1),
                "window_minutes": tw, "priority": priority, "released": False,
            })
        self.auto_dynamic_schedule = sorted(schedule, key=lambda x: x["release_time"])
        self.auto_dynamic_released = 0
        self.auto_dynamic_skipped = 0
        self.auto_order_seq = 0

    def _release_due_auto_orders(self) -> None:
        if not self.auto_dynamic_enabled:
            return
        for item in self.auto_dynamic_schedule:
            if item["released"] or float(item["release_time"]) > self.display_time + 1e-9:
                continue
            # Tránh bơm quá dày khi state gần đầy M_max. Nếu chưa có slot tái chế, trì
            # 10 phút mô phỏng thay vì làm model overflow hàng loạt.
            st = self.dispatcher.env.state
            unfinished = sum(1 for o in st.orders.values() if o.released and not o.cancelled and not o.is_completed)
            if unfinished >= max(8, int(self.cfg.problem.M_max) - 3):
                item["release_time"] = float(self.display_time + 10.0)
                self._log("SYSTEM", f"Tạm lùi {item['schedule_id']} thêm 10 phút vì state đang có {unfinished} đơn chưa hoàn tất (M_max={self.cfg.problem.M_max}).")
                continue
            pickup = dict(SAMPLE_LOCATIONS[item["pickup_id"]]) if item.get("pickup_id") else None
            delivery = dict(SAMPLE_LOCATIONS[item["delivery_id"]]) if item.get("delivery_id") else None
            try:
                self.create_order(
                    pickup=pickup, delivery=delivery, weight=float(item["weight"]),
                    note=f"Nhu cầu tự động {item['schedule_id']}", flow_type=item["flow_type"],
                    window_minutes=float(item["window_minutes"]), priority=float(item["priority"]),
                    source="AUTO", preferred_vehicle_id=item.get("preferred_vehicle_id"),
                )
                item["released"] = True
                self.auto_dynamic_released += 1
            except Exception as exc:
                # Không để một đơn mô phỏng làm dừng toàn bộ đồng hồ.
                item["released"] = True
                self.auto_dynamic_skipped += 1
                self._log("SYSTEM", f"Không thể phát sinh {item['schedule_id']}: {exc}")

    def _fallback_vehicle_for_order(self, order: Order, preferred_vehicle_id: Optional[str] = None) -> Optional[str]:
        st = self.dispatcher.env.state
        if preferred_vehicle_id in st.vehicles:
            v = st.vehicles[preferred_vehicle_id]
            if v.active and v.is_available(self.display_time) and (order.load_sign <= 0 or order.cargo_weight <= v.available_capacity + 1e-9):
                return preferred_vehicle_id
        pickup, delivery = self._order_locations(order)
        target = pickup if order.next_operation == OP_CUSTOMER_PICKUP else (delivery if order.next_operation == OP_CUSTOMER_DELIVERY else _point_payload(DEPOT))
        candidates = []
        for v in st.vehicle_list:
            if not v.active or not v.is_available(self.display_time):
                continue
            if order.load_sign > 0 and order.cargo_weight > v.available_capacity + 1e-9:
                continue
            lat, lon = self._visual_vehicle_position(v)
            candidates.append((haversine_km((lat, lon), (target["lat"], target["lon"])), v.working_time, v.vehicle_id))
        return min(candidates)[2] if candidates else None

    def _assign_fallback_plan(self, order_id: str, preferred_vehicle_id: Optional[str] = None) -> Optional[str]:
        st = self.dispatcher.env.state
        o = st.orders.get(order_id)
        if o is None or o.cancelled or o.is_completed:
            return None
        # Nếu model đã đưa order này vào một plan thì không ghi đè.
        if any(order_id in p for p in self.vehicle_plans.values()):
            return next((vid for vid, p in self.vehicle_plans.items() if order_id in p), None)
        vid = self._fallback_vehicle_for_order(o, preferred_vehicle_id)
        if not vid:
            return None
        self.vehicle_plans.setdefault(vid, []).append(order_id)
        self._log("SYSTEM", f"{order_id} được thêm vào tuyến dự phòng khu vực của {vid}; nếu MCDSM + PESDVS chọn xe khác ở lần tái xét kế tiếp, tuyến sẽ tự cập nhật.", order_id, vid)
        self._try_start_vehicle(vid)
        return vid

    def set_auto_dynamic(self, enabled: bool) -> None:
        with self.lock:
            self.auto_dynamic_enabled = bool(enabled)
            self._log("SYSTEM", f"Bộ sinh đơn động tự động: {'BẬT' if self.auto_dynamic_enabled else 'TẮT'}.")
            if self.auto_dynamic_enabled:
                self._release_due_auto_orders()

    # ------------------------------------------------------------------ event / decision
    def _log(self, kind: str, text: str, order_id: str = "", vehicle_id: str = "") -> None:
        self.events.append({
            "clock": hhmm(self.display_time), "timestamp": round(self.display_time, 2),
            "kind": kind, "text": text, "order_id": order_id, "vehicle_id": vehicle_id,
        })
        self.events = self.events[-220:]

    def _capture_decision(self, decision: Optional[DispatchDecision]) -> None:
        if decision is None:
            return
        self.decision_seq += 1
        row = decision.as_row()
        row["decision_id"] = f"DEC-{self.decision_seq:04d}"
        row["operation_label"] = OP_LABELS.get(row.get("operation", ""), row.get("operation", ""))
        row["flow_label"] = FLOW_LABELS.get(row.get("flow_type", ""), row.get("flow_type", ""))
        row["expected_arrival_clock"] = hhmm(float(row.get("expected_arrival_time", 0.0))) if row.get("expected_arrival_time") else ""
        row["expected_completion_clock"] = hhmm(float(row.get("expected_completion_time", 0.0))) if row.get("expected_completion_time") else ""
        vid = row.get("vehicle_id") or "SYSTEM"
        self.pending_decisions[vid] = row
        if row.get("status") == "OK" and row.get("vehicle_id") and row.get("order_id"):
            self._apply_recommendation_to_plan(row)
            self._log(
                "DECISION",
                f"Mô hình đề xuất {row['vehicle_id']} thực hiện {row['order_id']} / {row['operation_label']} tiếp theo (xác suất={row.get('policy_probability', 0):.3f}). Phần tuyến còn lại được cập nhật, không ngắt chặng đang chạy.",
                row["order_id"], row["vehicle_id"],
            )
            if self.auto_apply_decisions:
                self._try_start_vehicle(row["vehicle_id"])
        else:
            self._log("DECISION", f"Trạng thái quyết định: {row.get('status')} · {row.get('validator_notes', '')}")

    def _apply_recommendation_to_plan(self, row: Dict[str, Any]) -> None:
        vid, oid = row.get("vehicle_id"), row.get("order_id")
        if not vid or not oid:
            return
        # Một order chỉ được nằm trong đúng một remaining plan. Recommendation mới
        # supersede assignment fallback/initial trước đó.
        for other_plan in self.vehicle_plans.values():
            other_plan[:] = [x for x in other_plan if x != oid]
        plan = self.vehicle_plans.setdefault(vid, [])
        plan.insert(0, oid)

    def _process_event(self, event: Event) -> Optional[DispatchDecision]:
        self.display_time = max(self.display_time, float(event.timestamp))
        decision = self.dispatcher.process_event(event)
        self._log("EVENT", f"{event.event_type} · {event.order_id or event.vehicle_id or ''}".strip(" ·"), event.order_id or "", event.vehicle_id or "")
        self._capture_decision(decision)
        return decision

    # ------------------------------------------------------------------ operational plan / concurrent movement
    def _start_due_departures(self) -> None:
        for vid, depart_t in sorted(self.departure_times.items(), key=lambda x: x[1]):
            if not self.departed.get(vid) and self.display_time + 1e-9 >= depart_t:
                self.departed[vid] = True
                self._log("DEPARTURE", f"{vid} xuất kho lúc {hhmm(depart_t)} với danh sách hàng đầu ca, khu vực {VEHICLE_ZONES.get(vid, 'TP.HCM')}.", vehicle_id=vid)
                self._try_start_vehicle(vid, initial=True)

    def _candidate_for_order(self, vehicle: Vehicle, order: Order):
        st = self.dispatcher.env.state
        cand, rule = st.select_service_location(vehicle, order)
        if cand is None:
            return None, rule
        return cand, rule

    def _build_leg(self, vehicle_id: str, order_id: str, decision_id: str = "", source: str = "Kế hoạch hiện tại") -> Dict[str, Any]:
        st = self.dispatcher.env.state
        v = st.vehicles[vehicle_id]
        o = st.orders[order_id]
        cand, rule = self._candidate_for_order(v, o)
        if cand is None:
            raise ValueError(f"Không tìm được địa điểm khả thi cho {vehicle_id}/{order_id}: {rule}")
        sl = cand.location
        start_lat, start_lon = local_latlon(v.x, v.y)
        target_lat = float(sl.lat) if sl.lat is not None else local_latlon(sl.x, sl.y)[0]
        target_lon = float(sl.lon) if sl.lon is not None else local_latlon(sl.x, sl.y)[1]
        depart = max(float(self.display_time), float(v.current_time), float(self.departure_times.get(vehicle_id, 480.0)))
        travel = float(cand.travel_time)
        arrival = depart + travel
        wait = max(0.0, float(cand.waiting_time))
        service = max(0.0, float(cand.service_duration))
        completion = arrival + wait + service
        return {
            "decision_id": decision_id,
            "vehicle_id": vehicle_id,
            "order_id": order_id,
            "operation": o.next_operation,
            "operation_label": OP_LABELS.get(o.next_operation or "", o.next_operation or ""),
            "service_location_id": sl.service_location_id,
            "source": source,
            "selection_rule": rule,
            "start_time": depart,
            "arrival_time": arrival,
            "completion_time": completion,
            "travel_time": travel,
            "waiting_time": wait,
            "service_time": service,
            "distance_km_model": float(cand.distance),
            "start_lat": start_lat, "start_lon": start_lon,
            "target_lat": target_lat, "target_lon": target_lon,
            "progress": 0.0,
        }

    def _build_return_leg(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Operational tour closure: xe rảnh và không còn hàng sẽ quay về depot.

        Đây không phải action của Actor/PESDVS; nó là lớp vận hành để ba business flow
        đều được hiển thị có điểm kết thúc tại kho. Nếu xe đã ở depot thì không tạo leg.
        """
        st = self.dispatcher.env.state
        v = st.vehicles[vehicle_id]
        depot_sl = st.instance.service_locations[DEPOT["id"]][0]
        start_lat, start_lon = local_latlon(v.x, v.y)
        if haversine_km((start_lat, start_lon), (DEPOT["lat"], DEPOT["lon"])) < 0.15:
            return None
        depart = max(float(self.display_time), float(v.current_time))
        distance = float(st.travel.distance(v.xy, (depot_sl.x, depot_sl.y)))
        travel = float(st.travel.travel_time(v, v.xy, (depot_sl.x, depot_sl.y), depart))
        completion = depart + travel
        return {
            "decision_id": "", "vehicle_id": vehicle_id, "order_id": f"{vehicle_id}-RETURN",
            "operation": "RETURN_TO_DEPOT", "operation_label": "Trở về kho",
            "service_location_id": depot_sl.service_location_id,
            "source": "Kết thúc vòng tuyến / chờ nhu cầu mới", "selection_rule": "operational_return",
            "start_time": depart, "arrival_time": completion, "completion_time": completion,
            "travel_time": travel, "waiting_time": 0.0, "service_time": 0.0,
            "distance_km_model": distance, "start_lat": start_lat, "start_lon": start_lon,
            "target_lat": DEPOT["lat"], "target_lon": DEPOT["lon"], "progress": 0.0,
            "return_to_depot": True,
        }

    def _try_start_vehicle(self, vehicle_id: str, initial: bool = False) -> Optional[Dict[str, Any]]:
        if not self.running or not self.departed.get(vehicle_id, False):
            return None
        if vehicle_id in self.active_legs:
            return None
        st = self.dispatcher.env.state
        v = st.vehicles.get(vehicle_id)
        if v is None or not v.active:
            return None

        # Model recommendation has priority over the baseline remaining plan.
        d = self.pending_decisions.get(vehicle_id)
        if d and d.get("status") == "OK" and d.get("feasible", True):
            oid = d.get("order_id")
            if oid in st.orders and not st.orders[oid].is_completed and not st.orders[oid].cancelled:
                try:
                    leg = self._build_leg(vehicle_id, oid, d.get("decision_id", ""), "Khuyến nghị mô hình")
                    self.pending_decisions.pop(vehicle_id, None)
                    rec_order = st.orders.get(oid)
                    if rec_order is not None and not rec_order.cargo_on_vehicle and not rec_order.committed_vehicle_id:
                        rec_order.committed_vehicle_id = vehicle_id
                    self.active_legs[vehicle_id] = leg
                    self._log("EXECUTE", f"{vehicle_id} bắt đầu {leg['operation_label']} cho {oid} theo khuyến nghị mô hình.", oid, vehicle_id)
                    return copy.deepcopy(leg)
                except ValueError as exc:
                    self._log("SYSTEM", f"Không thể khởi hành theo khuyến nghị {vehicle_id}/{oid}: {exc}", oid, vehicle_id)

        plan = self.vehicle_plans.setdefault(vehicle_id, [])
        # Bỏ đơn đã hoàn thành/hủy khỏi tuyến còn lại.
        plan[:] = [oid for oid in plan if oid in st.orders and not st.orders[oid].is_completed and not st.orders[oid].cancelled]
        for oid in list(plan):
            o = st.orders[oid]
            owner = o.assigned_vehicle_id if o.cargo_on_vehicle else o.committed_vehicle_id
            if owner and owner != vehicle_id:
                continue
            try:
                leg = self._build_leg(vehicle_id, oid, "", "Kế hoạch đầu ca" if initial else "Tuyến còn lại")
            except ValueError:
                continue
            # Khi một tác vụ đã thực sự bắt đầu, khóa planning-level commitment vào xe
            # đó cho đến SERVICE_COMPLETE. Nếu không, một NEW_ORDER khác giữa chặng có thể
            # làm dispatcher chuyển commitment sang xe khác dù xe hiện tại đã chạy tới điểm.
            if not o.cargo_on_vehicle and not o.committed_vehicle_id:
                o.committed_vehicle_id = vehicle_id
            self.active_legs[vehicle_id] = leg
            self._log("EXECUTE", f"{vehicle_id} bắt đầu {leg['operation_label']} cho {oid} theo {'kế hoạch đầu ca' if initial else 'tuyến còn lại'}.", oid, vehicle_id)
            return copy.deepcopy(leg)

        # Không còn task đã biết: đóng vòng tuyến bằng chặng quay về depot nếu xe không
        # còn mang hàng. Xe có thể lại rời kho khi một order động mới được release.
        if not v.carrying:
            return_leg = self._build_return_leg(vehicle_id)
            if return_leg is not None:
                self.active_legs[vehicle_id] = return_leg
                self._log("EXECUTE", f"{vehicle_id} không còn tác vụ đã biết, bắt đầu quay về kho để chờ nhu cầu mới.", vehicle_id=vehicle_id)
                return copy.deepcopy(return_leg)
        return None

    def tick(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.display_time = min(float(self.cfg.clock.day_end), self.display_time + self.sim_minutes_per_tick)
            self._start_due_departures()
            self._release_due_auto_orders()

            completed_vids: List[str] = []
            for vid, leg in list(self.active_legs.items()):
                start_t, arrival_t = float(leg["start_time"]), float(leg["arrival_time"])
                if arrival_t <= start_t + 1e-9:
                    progress = 1.0
                else:
                    progress = (min(self.display_time, arrival_t) - start_t) / (arrival_t - start_t)
                leg["progress"] = max(0.0, min(1.0, progress))
                if self.display_time >= float(leg["completion_time"]) - 1e-9:
                    completed_vids.append(vid)

            # Hoàn thành theo timestamp để event log/clock có thứ tự ổn định.
            completed_vids.sort(key=lambda vid: self.active_legs[vid]["completion_time"])
            for vid in completed_vids:
                self._finish_leg(vid)

            # Xe rảnh sẽ tiếp tục phần tuyến còn lại. Nếu có recommendation mới, nó được ưu tiên.
            for vid in self.departure_times:
                self._try_start_vehicle(vid)

    def _finish_leg(self, vehicle_id: str) -> None:
        leg = self.active_legs.pop(vehicle_id, None)
        if not leg:
            return
        st = self.dispatcher.env.state
        v = st.vehicles[vehicle_id]
        oid = leg["order_id"]
        completion = float(leg["completion_time"])

        if leg.get("return_to_depot"):
            v.travel_time += float(leg.get("travel_time", 0.0))
            v.working_time += float(leg.get("travel_time", 0.0))
            v.distance += float(leg.get("distance_km_model", 0.0))
            depot_sl = st.instance.service_locations[DEPOT["id"]][0]
            v.x, v.y = depot_sl.x, depot_sl.y
            v.current_time = max(v.current_time, completion)
            v.last_service_location_id = depot_sl.service_location_id
            v.route.append({"x": depot_sl.x, "y": depot_sl.y, "order_id": "", "operation": "RETURN_TO_DEPOT", "timestamp": completion})
            self.operation_history.append({**copy.deepcopy(leg), "completed": True, "complete_clock": hhmm(completion)})
            self.operation_history = self.operation_history[-220:]
            self._log("SERVICE", f"{vehicle_id} đã trở về kho và sẵn sàng nhận vòng tuyến tiếp theo.", vehicle_id=vehicle_id)
            return

        o = st.orders[oid]
        self.display_time = max(self.display_time, completion)

        # Metrics vận hành cho các leg do prototype orchestration thực thi.
        v.travel_time += float(leg.get("travel_time", 0.0))
        v.waiting_time += float(leg.get("waiting_time", 0.0))
        v.working_time += float(leg.get("travel_time", 0.0)) + float(leg.get("waiting_time", 0.0)) + float(leg.get("service_time", 0.0))
        v.distance += float(leg.get("distance_km_model", 0.0))
        target_sl = st.instance.location_by_id(leg["service_location_id"])
        v.route.append({"x": target_sl.x, "y": target_sl.y, "order_id": oid, "operation": leg["operation"], "timestamp": completion})

        event_type = OP_COMPLETE_EVENT[leg["operation"]]
        event = Event(
            timestamp=completion, event_type=event_type,
            order_id=oid, vehicle_id=vehicle_id,
            service_location_id=leg["service_location_id"], source="PROTOTYPE",
            event_id=f"{event_type}-{oid}-{int(completion*10)}",
        )
        self.operation_history.append({
            **copy.deepcopy(leg), "completed": True, "complete_clock": hhmm(completion),
        })
        self.operation_history = self.operation_history[-180:]

        # Nếu order nhiều stage (pickup rồi delivery/unload), đưa stage còn lại về cuối plan.
        plan = self.vehicle_plans.setdefault(vehicle_id, [])
        plan[:] = [x for x in plan if x != oid]

        self._process_event(event)
        o_after = st.orders[oid]
        if not o_after.is_completed and not o_after.cancelled:
            # Service completion có thể đồng thời trigger một recommendation mới. Không
            # được append lại vào plan cũ nếu dispatcher vừa sắp xếp order vào plan khác.
            already_planned = any(oid in p for p in self.vehicle_plans.values())
            if not already_planned:
                owner = o_after.assigned_vehicle_id or o_after.committed_vehicle_id or vehicle_id
                self.vehicle_plans.setdefault(owner, []).append(oid)
        self._log("SERVICE", f"{vehicle_id} hoàn tất {OP_LABELS.get(leg['operation'], leg['operation'])} của {oid}.", oid, vehicle_id)

    # ------------------------------------------------------------------ manual / model decision API
    def execute_decision(self, decision_id: str) -> Dict[str, Any]:
        with self.lock:
            row = next((d for d in self.pending_decisions.values() if d.get("decision_id") == decision_id), None)
            if row is None:
                raise ValueError("Quyết định không còn hiệu lực hoặc đã được thay thế.")
            if row.get("status") != "OK" or not row.get("feasible", True):
                raise ValueError(f"Quyết định không thể thực hiện: {row.get('status')}")
            vid = row["vehicle_id"]
            self._apply_recommendation_to_plan(row)
            if vid in self.active_legs:
                # Không quay đầu giữa chặng; giữ recommendation cho ranh giới tiếp theo.
                self._log("DECISION", f"Khuyến nghị {decision_id} đã được chấp nhận cho {vid}; sẽ áp dụng ngay sau chặng đang chạy.", row.get("order_id", ""), vid)
                return {"queued": True, "vehicle_id": vid, "decision_id": decision_id}
            self.running = True
            leg = self._try_start_vehicle(vid)
            if leg is None:
                raise ValueError("Xe chưa đến giờ xuất phát hoặc không thể bắt đầu tác vụ này.")
            return leg

    def manual_replan(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            d = self.dispatcher.replan(None, "MANUAL_REPLAN")
            self._capture_decision(d)
            return d.as_row()

    # ------------------------------------------------------------------ dynamic orders
    def create_order(self, pickup: Optional[Dict[str, Any]], delivery: Optional[Dict[str, Any]],
                     weight: float, note: str, flow_type: str = FLOW_PICKUP_DELIVERY,
                     window_minutes: float = 180.0, priority: float = 2.0,
                     source: str = "MANUAL", preferred_vehicle_id: Optional[str] = None) -> Order:
        with self.lock:
            flow = str(flow_type or FLOW_PICKUP_DELIVERY).upper()
            if flow not in {FLOW_PICKUP_DELIVERY, FLOW_OUTBOUND, FLOW_INBOUND}:
                raise ValueError("Prototype chỉ cho tạo ba loại đơn phát sinh: lấy–giao khách, lấy về kho hoặc từ kho giao khách.")
            source = str(source or "MANUAL").upper()
            if source == "AUTO":
                self.auto_order_seq += 1
                oid = f"AU-{self.auto_order_seq:03d}"
            else:
                self.order_seq += 1
                oid = f"DY-{self.order_seq:03d}"
            now = float(self.display_time)
            locations = self.dispatcher.env.state.instance.service_locations
            consignor_id = DEPOT["id"]
            consignee_id = DEPOT["id"]
            source_depot_id = dest_depot_id = ""
            latest = now + float(window_minutes)

            if flow in {FLOW_PICKUP_DELIVERY, FLOW_OUTBOUND}:
                if pickup is None:
                    raise ValueError("Luồng này cần điểm lấy hàng.")
                consignor_id = f"{oid}-P"
                self._add_customer_location(locations, consignor_id, ROLE_PICKUP, pickup,
                                            now, now + min(120.0, window_minutes), 6.0)
            if flow in {FLOW_PICKUP_DELIVERY, FLOW_INBOUND}:
                if delivery is None:
                    raise ValueError("Luồng này cần điểm giao hàng.")
                consignee_id = f"{oid}-D"
                self._add_customer_location(locations, consignee_id, ROLE_DELIVERY, delivery,
                                            now + (15.0 if flow == FLOW_PICKUP_DELIVERY else 0.0),
                                            latest, 7.0)
            if flow == FLOW_OUTBOUND:
                dest_depot_id = DEPOT["id"]
            elif flow == FLOW_INBOUND:
                source_depot_id = DEPOT["id"]

            order = Order(
                order_id=oid, consignor_id=consignor_id, consignee_id=consignee_id,
                cargo_weight=float(weight), order_class=ORDER_DYNAMIC,
                release_time=now, created_timestamp=now, priority=float(priority),
                external_order_id=oid, latest_delivery_time=latest,
                flow_type=flow, source_depot_id=source_depot_id, dest_depot_id=dest_depot_id,
            )
            self.dispatcher.env.state.instance.orders.append(order)
            self.dispatcher.env.state.orders[oid] = order
            event = Event(
                timestamp=now, event_type="NEW_ORDER", order_id=oid,
                payload={
                    "consignor_id": consignor_id, "consignee_id": consignee_id,
                    "cargo_weight": float(weight), "created_timestamp": now,
                    "priority": float(priority), "latest_delivery_time": latest,
                    "flow_type": flow, "source_depot_id": source_depot_id,
                    "dest_depot_id": dest_depot_id, "note": note,
                },
                source=("AUTO_DEMAND" if source == "AUTO" else "CUSTOMER_UI"), event_id=f"NEW-{oid}",
            )
            # Khác bản V5: event được đưa vào ngay cả khi nhiều xe đang chạy. Recommendation
            # chỉ thay đổi phần tuyến còn lại, không ngắt chặng hiện tại.
            self.order_origins[oid] = "AUTO" if source == "AUTO" else "MANUAL"
            self._process_event(event)
            # Nếu model không đưa order mới vào một plan ở lần trigger hiện tại, operational
            # layer vẫn có một tuyến dự phòng theo vùng/độ gần để đội xe không đứng chờ vô lý.
            fallback_vid = self._assign_fallback_plan(oid, preferred_vehicle_id)
            origin_text = "tự động" if source == "AUTO" else "thủ công"
            self._log("NEW_ORDER", f"{oid} ({origin_text}) phát sinh lúc {hhmm(now)}; mô hình đánh giá ngay. Tuyến fallback: {fallback_vid or 'chưa gán'}.", oid, fallback_vid or "")
            return order

    def cancel_order(self, order_id: str) -> None:
        with self.lock:
            o = self.dispatcher.env.state.orders.get(order_id)
            if o is None:
                raise ValueError("Không tìm thấy đơn hàng.")
            if o.cargo_on_vehicle:
                raise ValueError("Không cho hủy đơn khi hàng đã ở trên xe.")
            self._process_event(Event(timestamp=self.display_time, event_type="CANCEL_ORDER",
                                      order_id=order_id, source="PROTOTYPE",
                                      event_id=f"CANCEL-{order_id}-{int(self.display_time)}"))
            for plan in self.vehicle_plans.values():
                plan[:] = [x for x in plan if x != order_id]

    # ------------------------------------------------------------------ operational events
    def traffic_update(self, factor: float, area: str = "ALL") -> Dict[str, Any]:
        with self.lock:
            old = float(self.dispatcher.env.state.traffic.factor(area))
            self._process_event(Event(timestamp=self.display_time, event_type="TRAFFIC_UPDATE",
                                      payload={"factor": float(factor), "area": area},
                                      source="TRAFFIC_API", event_id=f"TRAFFIC-{time.time_ns()}"))
            return {"old_factor": old, "new_factor": float(factor),
                    "relative_change": abs(float(factor) - old) / max(abs(old), 1e-9),
                    "threshold": self.cfg.replan.traffic_rho}

    def vehicle_breakdown(self, vehicle_id: str) -> None:
        with self.lock:
            if vehicle_id in self.active_legs:
                # Prototype không teleport/rollback giữa chặng; sự cố ghi nhận ở điểm hiện tại và
                # dừng kế hoạch sau khi chặng mô phỏng hiện tại kết thúc.
                self._log("SYSTEM", f"Ghi nhận cảnh báo hỏng {vehicle_id}; xe sẽ dừng sau chặng đang chạy để tránh teleport giữa đường.", vehicle_id=vehicle_id)
            self._process_event(Event(timestamp=self.display_time, event_type="VEHICLE_BREAKDOWN",
                                      vehicle_id=vehicle_id, source="PROTOTYPE",
                                      event_id=f"BREAK-{vehicle_id}-{time.time_ns()}"))

    def vehicle_recovery(self, vehicle_id: str) -> None:
        with self.lock:
            self._process_event(Event(timestamp=self.display_time, event_type="VEHICLE_RECOVERY",
                                      vehicle_id=vehicle_id, source="PROTOTYPE",
                                      event_id=f"RECOVER-{vehicle_id}-{time.time_ns()}"))
            self._try_start_vehicle(vehicle_id)

    @staticmethod
    def _business_flow_number(flow_type: str) -> int:
        if flow_type == FLOW_OUTBOUND:
            return 1
        if flow_type in {FLOW_INBOUND, FLOW_INBOUND_PRELOADED}:
            return 2
        if flow_type == FLOW_PICKUP_DELIVERY:
            return 3
        return 0

    @staticmethod
    def _business_flow_short(flow_type: str) -> str:
        n = SimulationEngine._business_flow_number(flow_type)
        return {
            1: "Lấy về kho",
            2: "Giao từ kho",
            3: "Lấy & giao trực tiếp",
        }.get(n, flow_type)

    # ------------------------------------------------------------------ serialization
    def _location_payload(self, sl: ServiceLocation) -> Dict[str, Any]:
        lat, lon = ((float(sl.lat), float(sl.lon)) if sl.lat is not None and sl.lon is not None
                    else local_latlon(sl.x, sl.y))
        return {
            "service_location_id": sl.service_location_id,
            "entity_id": sl.entity_id, "role": sl.entity_role,
            "name": sl.meta.get("name", sl.entity_id), "address": sl.meta.get("address", ""),
            "lat": lat, "lon": lon, "begin_time": sl.begin_time, "end_time": sl.end_time,
            "begin_clock": hhmm(sl.begin_time), "end_clock": hhmm(sl.end_time),
        }

    def _order_locations(self, o: Order) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        depot_sl = self.dispatcher.env.state.instance.service_locations[DEPOT["id"]][0]
        pickup = self._location_payload(depot_sl)
        delivery = self._location_payload(depot_sl)
        if o.flow_type in {FLOW_PICKUP_DELIVERY, FLOW_OUTBOUND}:
            locs = self.dispatcher.env.state.instance.locations_for(o.consignor_id, ROLE_PICKUP)
            if locs:
                pickup = self._location_payload(locs[0])
        if o.flow_type in {FLOW_PICKUP_DELIVERY, FLOW_INBOUND, FLOW_INBOUND_PRELOADED}:
            locs = self.dispatcher.env.state.instance.locations_for(o.consignee_id, ROLE_DELIVERY)
            if locs:
                delivery = self._location_payload(locs[0])
        return pickup, delivery

    def _visual_vehicle_position(self, v: Vehicle) -> Tuple[float, float]:
        leg = self.active_legs.get(v.vehicle_id)
        if leg is not None:
            p = max(0.0, min(1.0, float(leg.get("progress", 0.0))))
            # Map UI sẽ thay nội suy thẳng bằng pointAlong(OSRM geometry). Tọa độ này chỉ là
            # fallback state cho client trước khi geometry đường bộ tải xong.
            lat = leg["start_lat"] + (leg["target_lat"] - leg["start_lat"]) * p
            lon = leg["start_lon"] + (leg["target_lon"] - leg["start_lon"]) * p
            return lat, lon
        return local_latlon(v.x, v.y)

    def _plan_payload(self, vehicle_id: str) -> List[Dict[str, Any]]:
        st = self.dispatcher.env.state
        rows: List[Dict[str, Any]] = []
        for pos, oid in enumerate(self.vehicle_plans.get(vehicle_id, []), start=1):
            o = st.orders.get(oid)
            if o is None or o.cancelled or o.is_completed:
                continue
            pickup, delivery = self._order_locations(o)
            target = pickup if o.next_operation == OP_CUSTOMER_PICKUP else (delivery if o.next_operation == OP_CUSTOMER_DELIVERY else DEPOT)
            rows.append({
                "position": pos, "order_id": oid, "operation": o.next_operation,
                "operation_label": OP_LABELS.get(o.next_operation or "", "Đã hoàn tất"),
                "weight": round(o.cargo_weight, 1), "priority": o.priority,
                "target": target,
                "latest_clock": hhmm(o.latest_delivery_time) if o.latest_delivery_time is not None else "",
                "dynamic": o.order_class == ORDER_DYNAMIC,
                "origin": self.order_origins.get(oid, "STATIC"),
                "flow_type": o.flow_type,
                "flow_number": self._business_flow_number(o.flow_type),
                "flow_short": self._business_flow_short(o.flow_type),
                "stage": o.stage,
                "stage_index": o.stage_index,
                "n_stages": o.n_stages,
                "committed": o.committed_vehicle_id == vehicle_id,
                "onboard": o.assigned_vehicle_id == vehicle_id and o.cargo_on_vehicle,
            })
        return rows

    def state(self) -> Dict[str, Any]:
        with self.lock:
            st = self.dispatcher.env.state
            vehicles = []
            for v in st.vehicle_list:
                lat, lon = self._visual_vehicle_position(v)
                committed = st.committed_orders_of(v.vehicle_id)
                visual_status = "EN_ROUTE" if v.vehicle_id in self.active_legs else ("SCHEDULED" if not self.departed.get(v.vehicle_id, False) else v.status)
                plan = self._plan_payload(v.vehicle_id)
                vehicles.append({
                    "vehicle_id": v.vehicle_id, "vehicle_type": v.vehicle_type,
                    "zone": VEHICLE_ZONES.get(v.vehicle_id, "TP.HCM"),
                    "departure_time": self.departure_times.get(v.vehicle_id, 480.0),
                    "departure_clock": hhmm(self.departure_times.get(v.vehicle_id, 480.0)),
                    "departed": self.departed.get(v.vehicle_id, False),
                    "lat": lat, "lon": lon, "status": visual_status,
                    "capacity": round(v.capacity, 1), "load": round(v.load, 1),
                    "remaining_capacity": round(v.available_capacity, 1),
                    "load_pct": round(100.0 * v.load / max(v.capacity, 1e-9), 1),
                    "carrying": list(v.carrying), "committed_orders": committed,
                    "current_time": round(v.current_time, 2), "current_clock": hhmm(v.current_time),
                    "working_time": round(v.working_time, 2), "travel_time": round(v.travel_time, 2),
                    "waiting_time": round(v.waiting_time, 2), "distance": round(v.distance, 2),
                    "n_services": v.n_services, "color": VEHICLE_COLORS.get(v.vehicle_id, "#475467"),
                    "broken": (not v.active),
                    "route_history": [[*local_latlon(float(r.get("x", v.x)), float(r.get("y", v.y)))] for r in v.route],
                    "operational_plan": plan,
                })

            orders = []
            for o in sorted(st.orders.values(), key=lambda x: (x.is_completed, x.order_id)):
                pickup, delivery = self._order_locations(o)
                orders.append({
                    "order_id": o.order_id, "order_class": o.order_class,
                    "flow_type": o.flow_type, "flow_label": FLOW_LABELS.get(o.flow_type, o.flow_type),
                    "flow_number": self._business_flow_number(o.flow_type),
                    "flow_short": self._business_flow_short(o.flow_type),
                    "origin": self.order_origins.get(o.order_id, "STATIC"),
                    "stage": o.stage, "stage_index": o.stage_index, "n_stages": o.n_stages,
                    "next_operation": o.next_operation or "", "next_operation_label": OP_LABELS.get(o.next_operation or "", "Đã hoàn tất"),
                    "cargo_state": o.cargo_state, "weight": round(o.cargo_weight, 1),
                    "released": o.released, "cancelled": o.cancelled, "completed": o.is_completed,
                    "assigned_vehicle_id": o.assigned_vehicle_id,
                    "committed_vehicle_id": o.committed_vehicle_id,
                    "locked_to_vehicle_id": o.locked_to_vehicle_id,
                    "priority": o.priority, "late": o.late, "lateness": round(o.lateness, 2),
                    "release_time": o.release_time, "release_clock": hhmm(o.release_time),
                    "latest_delivery_time": o.latest_delivery_time,
                    "latest_delivery_clock": hhmm(o.latest_delivery_time) if o.latest_delivery_time is not None else "",
                    "pickup": pickup, "delivery": delivery,
                })

            # Bảng điều hành ba flow nghiệp vụ. INBOUND_PRELOADED và INBOUND đều
            # được gộp thành Flow 2 ở cấp business.
            flow_summary: Dict[str, Dict[str, Any]] = {
                "1": {"number": 1, "title": "Lấy hàng về kho", "path": "KHO → LẤY → KHO", "active": 0, "completed": 0, "static": 0, "dynamic": 0, "auto": 0, "manual": 0, "onboard": 0},
                "2": {"number": 2, "title": "Giao hàng từ kho", "path": "KHO → GIAO → KHO", "active": 0, "completed": 0, "static": 0, "dynamic": 0, "auto": 0, "manual": 0, "onboard": 0},
                "3": {"number": 3, "title": "Lấy & giao trực tiếp", "path": "KHO → LẤY → GIAO → KHO", "active": 0, "completed": 0, "static": 0, "dynamic": 0, "auto": 0, "manual": 0, "onboard": 0},
            }
            for o in st.orders.values():
                key = str(self._business_flow_number(o.flow_type))
                if key not in flow_summary:
                    continue
                fs = flow_summary[key]
                if o.is_completed:
                    fs["completed"] += 1
                elif o.released and not o.cancelled:
                    fs["active"] += 1
                if o.order_class == ORDER_STATIC:
                    fs["static"] += 1
                else:
                    fs["dynamic"] += 1
                    origin = self.order_origins.get(o.order_id, "MANUAL").lower()
                    if origin in fs:
                        fs[origin] += 1
                if o.cargo_on_vehicle and not o.is_completed:
                    fs["onboard"] += 1

            next_auto = next((x for x in self.auto_dynamic_schedule if not x.get("released")), None)
            auto_dynamic = {
                "enabled": self.auto_dynamic_enabled,
                "scheduled_total": len(self.auto_dynamic_schedule),
                "released": self.auto_dynamic_released,
                "remaining": sum(1 for x in self.auto_dynamic_schedule if not x.get("released")),
                "skipped": self.auto_dynamic_skipped,
                "next_release_time": float(next_auto["release_time"]) if next_auto else None,
                "next_release_clock": hhmm(float(next_auto["release_time"])) if next_auto else "—",
            }

            decisions = list(self.pending_decisions.values())
            decisions.sort(key=lambda d: d.get("decision_id", ""), reverse=True)
            completed = sum(1 for o in st.orders.values() if o.is_completed)
            onboard = sum(1 for o in st.orders.values() if o.cargo_state == CARGO_ON_VEHICLE and not o.is_completed)
            active = sum(1 for o in st.orders.values() if o.released and not o.cancelled and not o.is_completed)
            late = sum(1 for o in st.orders.values() if o.late)
            traffic_factor = float(st.traffic.factor("ALL"))
            n_static = sum(1 for o in st.orders.values() if o.order_class == ORDER_STATIC)
            n_dynamic = sum(1 for o in st.orders.values() if o.order_class == ORDER_DYNAMIC)
            departed_count = sum(1 for x in self.departed.values() if x)
            busy_ratio = (len(self.active_legs) / departed_count) if departed_count else 0.0
            model_meta = {
                "checkpoint": self._checkpoint_path().name,
                "checkpoint_loaded": bool(self.dispatcher.checkpoint_loaded),
                "state_version": self.cfg.state.state_version,
                "m_max": self.cfg.problem.M_max, "k_max": self.cfg.problem.K_max,
                "z_dim": self.cfg.state.z_dim, "u_dim": self.cfg.state.u_dim, "d_dim": self.cfg.state.d_dim,
                "traffic_state": self.cfg.state.use_realtime_traffic_state,
                "slack_state": self.cfg.state.use_time_window_slack,
                "priority_state": self.cfg.state.use_priority_feature,
                "decode_mode": self.cfg.inference.decode_mode,
                "auto_approve": self.cfg.inference.auto_approve,
                "architecture": "MCDSM chọn xe → PESDVS chọn đơn → môi trường chọn tác vụ/địa điểm",
            }
            current_history = [dict(x) for x in self.operation_history[-90:]]
            for leg in self.active_legs.values():
                current_history.append({**copy.deepcopy(leg), "completed": False, "complete_clock": ""})
            return {
                "clock": {
                    "clock": hhmm(self.display_time), "minutes": round(self.display_time, 2),
                    "speed_rule": f"1 giây = {self.sim_minutes_per_tick:g} phút mô phỏng",
                    "running": self.running,
                },
                "scenario": {
                    "seed": self.scenario_seed,
                    "mode": "Nhiều xe chạy đồng thời + quyết định trực tuyến cho phần tuyến còn lại",
                    "auto_apply_decisions": self.auto_apply_decisions,
                },
                "depot": DEPOT,
                "vehicles": vehicles, "orders": orders,
                "pending_decisions": decisions,
                "active_legs": [copy.deepcopy(x) for x in self.active_legs.values()],
                "active_leg": copy.deepcopy(next(iter(self.active_legs.values()), None)),  # tương thích V5
                "operation_history": current_history,
                "summary": {
                    "active": active, "onboard": onboard, "completed": completed,
                    "late": late, "decisions": len(self.dispatcher.dispatch_log),
                    "replans": self.dispatcher.trigger.n_triggered,
                    "traffic_factor": round(traffic_factor, 3),
                    "vehicles_en_route": len(self.active_legs),
                    "vehicles_departed": departed_count,
                    "vehicle_activity_pct": round(100.0 * busy_ratio, 1),
                    "static_orders": n_static, "dynamic_orders": n_dynamic,
                    "auto_dynamic_released": self.auto_dynamic_released,
                },
                "events": list(reversed(self.events[-100:])),
                "model": model_meta,
                "samples": SAMPLE_LOCATIONS,
                "flow_labels": FLOW_LABELS,
                "operation_labels": OP_LABELS,
                "flow_summary": flow_summary,
                "auto_dynamic": auto_dynamic,
            }


def geocode_address(address: str) -> Optional[Dict[str, Any]]:
    """Định vị địa chỉ TP.HCM bằng Nominatim khi Internet khả dụng."""
    q = address.strip()
    if not q:
        return None
    params = urllib.parse.urlencode({
        "q": q if "Hồ Chí Minh" in q or "TP.HCM" in q else q + ", Hồ Chí Minh, Việt Nam",
        "format": "jsonv2", "limit": 1, "countrycodes": "vn",
    })
    req = urllib.request.Request(
        "https://nominatim.openstreetmap.org/search?" + params,
        headers={"User-Agent": "Eureka-DFPDP-Operations-Prototype/6.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None
        item = data[0]
        lat, lon = float(item["lat"]), float(item["lon"])
        if not (10.45 <= lat <= 11.30 and 106.20 <= lon <= 107.30):
            return None
        return {"name": q, "address": item.get("display_name", q), "lat": lat, "lon": lon,
                "district": "TP.HCM", "source": "Nominatim / OpenStreetMap"}
    except Exception:
        return None
