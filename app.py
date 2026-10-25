from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator import (
    FLOW_INBOUND,
    FLOW_OUTBOUND,
    FLOW_PICKUP_DELIVERY,
    SAMPLE_LOCATIONS,
    SimulationEngine,
    geocode_address,
)
from operations import SimulationEngine, ALGORITHMS, MODELS
from road_routes import RoadRoutes

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
STATIC = WEB / "static"


# Bộ định tuyến đường bộ cho lớp hiển thị prototype.
# V7 thử tuần tự nhiều máy chủ OSRM công khai để tăng khả năng hoạt động trên máy người dùng.
# Trình duyệt cũng có cơ chế gọi trực tiếp các provider này trước khi quay về endpoint localhost.
ROUTING_PROVIDERS = [
    x.strip().rstrip('/') for x in os.environ.get(
        "ROUTING_PROVIDERS",
        "https://router.project-osrm.org,https://routing.openstreetmap.de/routed-car"
    ).split(',') if x.strip()
]
_route_cache: Dict[tuple, Dict[str, Any]] = {}
_route_cache_lock = threading.Lock()

def _fetch_osrm(base_url: str, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    url = (
        f"{base_url}/route/v1/driving/"
        f"{float(start_lon):.6f},{float(start_lat):.6f};{float(end_lon):.6f},{float(end_lat):.6f}"
        "?overview=full&geometries=geojson&steps=false&alternatives=false"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 Eureka-DFPDP-Operations-Prototype/9.2",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=7) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message") or data.get("code") or "OSRM không trả về tuyến đường")
    route = data["routes"][0]
    coords = route.get("geometry", {}).get("coordinates", [])
    points = [[float(lat), float(lon)] for lon, lat in coords]
    if len(points) < 2:
        raise RuntimeError("Tuyến đường không có đủ điểm hình học")
    return {
        "ok": True, "source": base_url, "provider": base_url,
        "points": points, "distance_m": float(route.get("distance", 0.0)),
        "duration_s": float(route.get("duration", 0.0)), "fallback": False,
    }

def _road_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    key = tuple(round(float(v), 5) for v in (start_lat, start_lon, end_lat, end_lon))
    with _route_cache_lock:
        cached = _route_cache.get(key)
    if cached is not None:
        return dict(cached)
    for lat, lon in ((start_lat, start_lon), (end_lat, end_lon)):
        if not (10.45 <= float(lat) <= 11.30 and 106.20 <= float(lon) <= 107.30):
            raise ValueError("Tọa độ nằm ngoài phạm vi mô phỏng TP.HCM")
    attempts = []
    for provider in ROUTING_PROVIDERS:
        try:
            result = _fetch_osrm(provider, start_lat, start_lon, end_lat, end_lon)
            result["attempts"] = attempts + [{"provider": provider, "ok": True}]
            with _route_cache_lock:
                _route_cache[key] = dict(result)
            return result
        except Exception as exc:
            attempts.append({"provider": provider, "ok": False, "error": str(exc)})
    result = {
        "ok": False, "source": "Đường nối dự phòng", "provider": "fallback",
        "points": [[float(start_lat), float(start_lon)], [float(end_lat), float(end_lon)]],
        "distance_m": 0.0, "duration_s": 0.0, "fallback": True,
        "warning": "Không kết nối được các máy chủ định tuyến công khai. Trình duyệt sẽ thử gọi trực tiếp provider trước khi dùng đường nối dự phòng.",
        "attempts": attempts,
    }
    # Failed lookups must be retryable after connectivity recovers.
    return result

app = FastAPI(title="Eureka DFPDP Intelligence Operations Platform", version="9.2-telemetry-reporting")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
engine = SimulationEngine()
road_router = RoadRoutes(_road_route)
engine.road_router = road_router


class SocketHub:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.connections.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = []
        for ws in list(self.connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = SocketHub()
_sim_task: Optional[asyncio.Task] = None


async def publish() -> Dict[str, Any]:
    snapshot = engine.state()
    await hub.broadcast(snapshot)
    return snapshot


async def simulation_loop() -> None:
    loop = asyncio.get_running_loop()
    next_at = loop.time() + 1.0
    while True:
        await asyncio.sleep(max(0.0, next_at - loop.time()))
        next_at = max(next_at + 1.0, loop.time() + 0.5)
        waiting = await asyncio.to_thread(engine.tick)
        # While a new leg waits for its road geometry, re-check quickly so the held-back
        # minutes are used as soon as the route arrives; broadcasts stay at one per second.
        while waiting and loop.time() < next_at - 0.3:
            await asyncio.sleep(0.2)
            waiting = await asyncio.to_thread(engine.tick, False)
        await hub.broadcast(engine.state())


@app.on_event("startup")
async def _startup() -> None:
    global _sim_task
    if _sim_task is None or _sim_task.done():
        _sim_task = asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _sim_task
    if _sim_task:
        _sim_task.cancel()
        _sim_task = None
    road_router.close()


@app.get("/")
def root():
    return RedirectResponse("/control")


@app.get("/control")
def control_page():
    return FileResponse(WEB / "dashboard.html")


@app.get("/customer")
def customer_page():
    return FileResponse(WEB / "customer.html")


@app.get("/customer-web")
def customer_web_page():
    # Desktop order injection view uses the same current-model form.
    return FileResponse(WEB / "customer.html")


@app.get("/report")
def report_page():
    return FileResponse(WEB / "report.html")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "dfpdp-operations", "version": app.version}


@app.get("/api/state")
def state():
    return engine.state()


@app.get("/api/samples")
def samples():
    return SAMPLE_LOCATIONS


@app.get("/api/road-route")
async def road_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    try:
        return await asyncio.to_thread(_road_route, start_lat, start_lon, end_lat, end_lon)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/routing/config")
def routing_config():
    return {"providers": ROUTING_PROVIDERS, "mode": "browser-direct-first + backend-fallback"}


class PlaceInput(BaseModel):
    sample_id: Optional[str] = None
    address: Optional[str] = None


class CreateOrderPayload(BaseModel):
    flow_type: str = Field(default=FLOW_PICKUP_DELIVERY)
    pickup: Optional[PlaceInput] = None
    delivery: Optional[PlaceInput] = None
    weight: float = Field(default=50.0, gt=0, le=6000)
    priority: float = Field(default=2.0, ge=0, le=5)
    window_minutes: float = Field(default=180.0, ge=30, le=480)
    note: str = Field(default="", max_length=240)


class DecisionPayload(BaseModel):
    decision_id: str


class TrafficPayload(BaseModel):
    factor: float = Field(default=1.3, ge=0.5, le=3.0)
    area: str = "ALL"


class SpeedPayload(BaseModel):
    minutes_per_tick: float = Field(default=5.0, ge=0.25, le=240.0)


class AutoDynamicPayload(BaseModel):
    enabled: bool = True


class ScenarioPayload(BaseModel):
    vehicles: int = Field(default=6, ge=1, le=24, strict=True)
    orders: int = Field(default=30, ge=1, le=100, strict=True)
    dynamic_percent: int = Field(default=30, ge=0, le=100, strict=True)
    algorithm: str = "drl"
    model: str = "pilot"
    seed: int = Field(default=42, ge=0, le=999999, strict=True)
    days: int = Field(default=3, ge=1, le=7, strict=True)
    traffic_region: str = "HCMC"
    start_weekday: int = Field(default=0, ge=0, le=6, strict=True)


@app.get("/api/options")
def options():
    return {"algorithms": ALGORITHMS, "models": MODELS, "max_vehicles": 24, "max_orders": 100,
            "max_days": 7, "traffic_regions": {"HCMC":"TP.HCM"}}


@app.post("/api/simulation/configure")
async def configure(payload: ScenarioPayload):
    global engine
    # Build successfully before replacing the current run.
    try:
        replacement = await asyncio.to_thread(SimulationEngine, **payload.model_dump())
        replacement.road_router = road_router
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    engine = replacement
    return {"ok": True, "state": await publish()}


def resolve_place(place: Optional[PlaceInput], required: bool = True) -> Optional[Dict[str, Any]]:
    if place is None:
        if required:
            raise HTTPException(400, "Thiếu địa điểm bắt buộc.")
        return None
    if place.sample_id:
        if place.sample_id not in SAMPLE_LOCATIONS:
            raise HTTPException(400, f"Địa chỉ mẫu không hợp lệ: {place.sample_id}")
        return dict(SAMPLE_LOCATIONS[place.sample_id])
    text = (place.address or "").strip()
    if not text:
        if required:
            raise HTTPException(400, "Hãy nhập địa chỉ hoặc chọn một địa chỉ mẫu.")
        return None
    low = text.lower()
    for item in SAMPLE_LOCATIONS.values():
        if low in {item["name"].lower(), item["address"].lower()}:
            return dict(item)
    found = geocode_address(text)
    if found is None:
        raise HTTPException(
            400,
            "Không định vị được địa chỉ tùy chỉnh. Hãy dùng địa chỉ mẫu hoặc bật Internet để gọi OpenStreetMap Nominatim.",
        )
    return found


@app.post("/api/orders")
async def create_order(payload: CreateOrderPayload):
    flow = payload.flow_type.upper().strip()
    if flow not in {FLOW_PICKUP_DELIVERY, FLOW_OUTBOUND, FLOW_INBOUND}:
        raise HTTPException(400, "Loại luồng vận chuyển không hợp lệ.")
    pickup_required = flow in {FLOW_PICKUP_DELIVERY, FLOW_OUTBOUND}
    delivery_required = flow in {FLOW_PICKUP_DELIVERY, FLOW_INBOUND}
    pickup = await asyncio.to_thread(resolve_place, payload.pickup, pickup_required)
    delivery = await asyncio.to_thread(resolve_place, payload.delivery, delivery_required)
    if pickup and delivery and abs(pickup["lat"] - delivery["lat"]) < 1e-7 and abs(pickup["lon"] - delivery["lon"]) < 1e-7:
        raise HTTPException(400, "Điểm lấy hàng và điểm giao hàng phải khác nhau.")
    try:
        order = engine.create_order(
            pickup=pickup, delivery=delivery, weight=payload.weight, note=payload.note,
            flow_type=flow, window_minutes=payload.window_minutes, priority=payload.priority,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    snapshot = await publish()
    return {"ok": True, "order_id": order.order_id, "state": snapshot}


@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    try:
        engine.cancel_order(order_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "state": await publish()}


@app.post("/api/decision/execute")
async def execute_decision(payload: DecisionPayload):
    try:
        leg = engine.execute_decision(payload.decision_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "leg": leg, "state": await publish()}


@app.post("/api/decision/replan")
async def replan():
    try:
        engine.manual_replan()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "state": await publish()}


@app.post("/api/events/traffic")
async def traffic(payload: TrafficPayload):
    try:
        info = engine.traffic_update(payload.factor, payload.area)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "traffic": info, "state": await publish()}


@app.post("/api/events/breakdown/{vehicle_id}")
async def breakdown(vehicle_id: str):
    try:
        engine.vehicle_breakdown(vehicle_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "state": await publish()}


@app.post("/api/events/recovery/{vehicle_id}")
async def recovery(vehicle_id: str):
    try:
        engine.vehicle_recovery(vehicle_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/start")
async def simulation_start():
    engine.start()
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/retry-routes")
async def retry_routes():
    engine.retry_routes()
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/pause")
async def simulation_pause():
    engine.pause()
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/speed")
async def simulation_speed(payload: SpeedPayload):
    engine.set_speed(payload.minutes_per_tick)
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/auto-dynamic")
async def simulation_auto_dynamic(payload: AutoDynamicPayload):
    engine.set_auto_dynamic(payload.enabled)
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/reset")
async def reset():
    engine.reset(new_seed=False)
    return {"ok": True, "state": await publish()}


@app.post("/api/simulation/new")
async def new_scenario():
    engine.reset(new_seed=True)
    return {"ok": True, "state": await publish()}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        await ws.send_json(engine.state())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)
