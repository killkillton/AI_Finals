# AI Adventures — Giao diện đồ án AI

Ứng dụng PyGame gồm menu và ba màn hình: N-Queens, Tank Battle, Kakuro.
Toàn bộ chữ hiển thị trong ứng dụng dùng tiếng Anh.

## Cài và chạy

Dùng Python 3.10 trở lên. Mở terminal tại thư mục chứa `main.py`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows: thay lệnh kích hoạt bằng `.venv\Scripts\activate`.

Chọn task trên menu; bấm **Main menu** hoặc **Esc** trong task để quay lại.

## Các chức năng

- **N-Queens:** đã nối solver thật. Bấm ô để đặt/gỡ hậu, hoặc chọn ô khác cùng
  cột để di chuyển. **Solve** giữ các quân đặt tay và giải phần còn lại.
  Có Standard/Custom Min-Conflicts, Glucose3, phát lại từng bước và biểu đồ CSV.
  Viền đỏ báo các quân đang ăn nhau; **Clear board** xóa bàn. N từ 4–200;
  Glucose3 trong GUI giới hạn N ≤ 64. Min-Conflicts hết giới hạn bước chỉ có
  nghĩa là chưa tìm thấy, không chứng minh vô nghiệm.
- **Tank Battle:** đã có giao diện, chọn bot, nhập bot `.py`, HP xe/căn cứ và
  hiển thị từng lượt. P1: W/A/S/D di chuyển, F bắn, G đứng yên. P2: phím mũi
  tên di chuyển, K bắn, L đứng yên. Chờ nhóm thêm engine/AI để chơi thật.
- **Kakuro:** đã nối `KakuroCSP` (đọc bàn cờ từ file JSON trong `Task3/boards/`,
  lan truyền ràng buộc AC-3 + Forward Checking và quay lui với MRV/Degree + LCV);
  hiển thị trực quan miền giá trị, gán số, cắt tỉa và phát hiện mâu thuẫn (`CONFLICT`);
  hỗ trợ Play, Step, Quick solve và chỉnh tốc độ.

Khi chưa có code Task 2, màn hình đó hiển thị **PREVIEW**, không giả lập
kết quả thuật toán.

## Cấu trúc

```text
main.py                       # Điểm chạy ứng dụng
requirements.txt              # pygame, python-sat
frontend/                     # Menu, giao diện, kết nối backend
assets/                       # Hai ảnh cần cho giao diện
Task1/                        # Solver, script benchmark, CSV
Task3/                        # Kakuro CSP solver và các bàn cờ JSON (boards/)
backend_config.example.json   # Mẫu chỉnh đường dẫn import
tests/                       # Kiểm tra solver và giao diện
```

## Ghép code Task 2/3

Mặc định giao diện tự import các module sau khi khởi động:

```text
Task2/tank_game_engine.py      # TankGameEngine, Action
Task2/agents.py                # BaseTankAgent và các lớp bot
Task3/kakuro_csp.py            # KakuroCSP
```

Thêm `__init__.py` vào Task2/ và Task3/. Nếu dùng tên module khác, sao chép
`backend_config.example.json` thành `backend_config.json` và sửa đường dẫn.
Khởi động lại ứng dụng sau khi thêm/sửa backend.

Các API theo PDF phân công nhóm:

- `TankGameEngine()`, `reset() -> dict`, `clone() -> engine`,
  `step(action_p1, action_p2) -> (state, sub_ticks)`.
- Bot kế thừa `BaseTankAgent`, khởi tạo bằng `player_id`, `max_depth`;
  `get_action(engine_copy) -> (Action, metrics)`.
- State có `turn`, `grid_size` (rows, cols), `tanks`, `bases`, `walls`,
  `bullets`, `is_terminal`, `winner`. Tọa độ `x` là cột, `y` là hàng.
  Mỗi xe/căn cứ có `hp`, `alive`; thanh máu đọc trực tiếp HP, tối đa 6/9.
- `KakuroCSP(board_id="board_5x5")` hoặc `board_10x10`;
  `get_board_layout() -> {rows, cols, cells}`;
  `solve_step_by_step()` yield snapshot `event_type`, `focus_cell`,
  `assignments`, `domains`, `pruned_cells`, `stats`.
- `assignments` và `domains` dùng khóa tuple `(r, c)`. Các sự kiện:
  `SELECT_VAR`, `ASSIGN`, `PROPAGATE`, `CONFLICT`, `BACKTRACK`, `SOLVED`.
  Khi quay lui, snapshot cần chứa miền giá trị đã khôi phục.

## Kiểm tra

```sh
python -m unittest discover -s tests
```

Tests dùng SDL dummy, không cần mở cửa sổ. Bản gửi không kèm `.venv` hoặc
cache Python; người nhận cài thư viện bằng `requirements.txt`.
