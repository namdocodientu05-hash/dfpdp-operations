# MODEL ALIGNMENT — V9.1

## Core không thay đổi kiến trúc học

Pipeline nghiên cứu vẫn là:

`feasibility / MCDSM logic → vehicle context → PESDVS Actor → order → environment`

Actor không chọn vehicle và không tự tạo full route trong một forward pass. Route được hình thành từ chuỗi next-order decisions.

## Lớp V9.1 bổ sung

### Traffic calibration
`traffic_profiles.py` cung cấp tốc độ theo `(region, weekday, hour)`. Trước mỗi decision boundary, prototype cập nhật `observed_speed` của fleet theo calibrated speed. Manual traffic factor của core vẫn được áp dụng độc lập như một incident multiplier.

### Multi-day orchestration
- Shift: 08:00–22:00.
- Current leg được hoàn tất trước khi đóng ngày.
- Fleet quay về depot.
- Đơn chưa xong vẫn nằm trong StateManager và được tiếp tục ở ngày sau.
- Non-physical commitments được release qua đêm; cargo ownership không bị xóa.

### Decision observability
V9.1 chỉ đọc các tensor/outputs đã có:
- shape `Z/U/D`;
- feature vector của selected order;
- feasible candidate set;
- Actor probabilities;
- selected action;
- latency.

Không chỉnh weights, logits hay action selection để phục vụ visualization.

## Không được claim
- Không có retraining mới trong V9.1.
- Không có 2-opt / insertion / local search / GA / SA / OR-Tools nhúng vào neural policy.
- Không có live Google traffic feed.
- Không có online learning.
