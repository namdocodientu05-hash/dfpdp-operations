# Hướng dẫn demo V9.1

## Kịch bản demo khuyến nghị

1. Chọn `AI · DRL PESDVS`, 6 xe, 40–60 đơn, 40–60% đơn động, 3 ngày, traffic TP.HCM.
2. Bấm **Tạo mô phỏng mới** rồi **Bắt đầu**.
3. Quan sát thanh traffic: 08:00–09:00 và 17:00–18:00 tốc độ giảm rõ.
4. Quan sát **Bên trong bộ ra quyết định**: pipeline chạy qua state/mask → candidates → selected order.
5. Chọn một xe trên map để xem tuyến đường bộ và lịch sử chặng.
6. Bật lưới phân tích để nhìn mật độ đơn theo ô.
7. Tăng tốc mô phỏng lên 1–4 giờ/giây để đi đến 22:00.
8. Cuối ngày, xem panel **Theo dõi chuyển ngày**. Các đơn tồn được liệt kê và tiếp tục từ 08:00 hôm sau.
9. Có thể tạo thêm đơn thủ công từ `/customer`, mô phỏng breakdown, hoặc thêm traffic incident factor.
10. Dùng cùng seed để đổi sang Nearest/Deadline và so sánh.
