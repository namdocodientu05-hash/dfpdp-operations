"""Bounded, shared road-geometry requests; a failed route is never a straight line."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor


class RoadRoutes:
    def __init__(self, resolver, workers=3):
        self.resolver = resolver
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="road-route")
        self.lock = threading.Lock()
        self.jobs = {}

    def request(self, coords):
        key = tuple(round(float(x), 6) for x in coords)
        with self.lock:
            if key not in self.jobs:
                self.jobs[key] = self.pool.submit(self._resolve, key)
            job = self.jobs[key]
        if not job.done():
            return {"status": "loading"}
        return job.result()

    def _resolve(self, coords):
        try:
            route = self.resolver(*coords)
            points = route.get("points", [])
            if not route.get("ok") or route.get("fallback") or len(points) < 2:
                raise ValueError("Chưa lấy được tuyến đường bộ. Kiểm tra Internet rồi thử lại.")
            if any(len(p) != 2 or not all(math.isfinite(float(x)) for x in p) for p in points):
                raise ValueError("Tuyến đường trả về không hợp lệ.")
            return {"status": "ready", "points": points, "provider": route.get("provider", "OSRM")}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def retry_failed(self):
        with self.lock:
            self.jobs = {k: v for k, v in self.jobs.items()
                         if not v.done() or v.result().get("status") != "error"}

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def road_position(points, progress):
    """Distance interpolation on consecutive road vertices, never endpoint-to-endpoint."""
    if len(points) == 1:
        return list(points[0])
    lengths = [math.hypot(b[0]-a[0], (b[1]-a[1])*math.cos(math.radians(a[0])))
               for a, b in zip(points, points[1:])]
    distance = sum(lengths) * max(0, min(1, progress))
    for a, b, length in zip(points, points[1:], lengths):
        if distance <= length and length > 0:
            ratio = distance / length
            return [a[0]+(b[0]-a[0])*ratio, a[1]+(b[1]-a[1])*ratio]
        distance -= length
    return list(points[-1])
