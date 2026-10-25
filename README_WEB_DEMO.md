# Eureka DFPDP Operations Platform V9.1 — Intelligence / Traffic / Multi-day

## Chạy nhanh trên Windows

1. Giải nén toàn bộ ZIP vào một thư mục riêng.
2. Double-click `run_demo.bat`.
3. Lần đầu chương trình sẽ tạo `.venv` và cài các thư viện trong `requirements.txt`.
4. Trình duyệt tự mở tại `http://127.0.0.1:8000/control`.
5. Trang tạo đơn thủ công: `http://127.0.0.1:8000/customer`.

> Bản đồ OpenStreetMap và hình học tuyến OSRM cần Internet. Core mô hình vẫn chạy khi Internet lỗi; khi đó giao diện dùng đường nối dự phòng.

## V9.1 bổ sung gì?

### 1. Dashboard khoa học / vận hành
- Bản đồ đường bộ OSM/OSRM.
- Lưới phân tích 4×4 trực tiếp trên bản đồ, kèm số đơn đang hoạt động trong từng ô.
- KPI vận hành, fleet panel, bảng đơn, lịch sử chặng, event log.
- Traffic intelligence 24 giờ với vị trí giờ hiện tại.
- Multi-day timeline và carry-over monitor.
- Model observability: pipeline quyết định, top candidates, xác suất Actor và state Z/U/D của order được chọn.

### 2. Traffic profile được nhúng trực tiếp
Một hồ sơ tốc độ TP.HCM được đóng gói offline trong `traffic_profiles.py`:
- TP.HCM: hồ sơ tham chiếu 24 giờ × 7 ngày, mean 21.1 km/h.

Hồ sơ này được lấy từ `HCMC_reference` của nhóm và là hồ sơ duy nhất được hiển thị/áp dụng trong nguyên mẫu hiện tại.

### 3. Multi-day simulation
- Chọn 1–7 ngày.
- Ca vận hành mỗi ngày: 08:00–22:00.
- 22:00: không nhận tác vụ mới, xe hoàn tất chặng đang chạy rồi quay về kho.
- Đơn chưa hoàn thành được giữ nguyên trạng thái và đưa sang ngày kế tiếp.
- Commitment planning chưa có cargo được release để sáng hôm sau tái đánh giá; cargo ownership thực tế vẫn được giữ.
- Dynamic demand được phân bố xuyên suốt các ngày, mỗi ngày khoảng 08:30–17:30.
- Nếu hết horizon vẫn còn đơn, trạng thái là `horizon_end` và dashboard hiển thị lượng carry-over còn lại.

### 4. Theo dõi “bên trong” bộ ra quyết định
Khi một xe rảnh cần tác vụ mới, V9.1 ghi một decision trace gồm:
1. Lọc xe khả thi / sắp thứ tự theo thời gian tích lũy.
2. Tạo state `Z, U, D` + feasibility mask.
3. Tính điểm/xác suất trên các order khả thi.
4. Chọn order.
5. Commit và bắt đầu chặng đường bộ.

Nếu chọn DRL, bảng candidates hiển thị probability của PESDVS Actor. Nếu chọn Nearest/Deadline, bảng vẫn hiển thị các candidate và score/tiêu chí baseline để so sánh.

## Điểm cần hiểu đúng về khoa học

- V9.1 **không train lại checkpoint**. Hai checkpoint được giữ nguyên từ source trước.
- V9.1 tăng observability, traffic calibration, multi-day orchestration và visualization; không được dùng để claim rằng neural policy đã mạnh hơn nếu chưa benchmark/retrain.
- Core AI vẫn là PESDVS Actor-Critic; MCDSM/feasibility rules vẫn kiểm soát tính khả thi.
- OSRM dùng cho geometry đường bộ ở giao diện. Core travel state dùng calibrated speed profile + traffic factor trên travel model hiện tại, không biến OSRM thành một optimizer mới.
- Return-to-depot và overnight rollover là orchestration của prototype, không phải action mới của Actor.

## Kiểm thử

```bash
python -m unittest test_operations -v
python smoke_test.py
```

V9.1 có test riêng cho traffic profile, multi-day carry-over, dynamic demand xuyên ngày, AI observability và tạo đơn thủ công.
