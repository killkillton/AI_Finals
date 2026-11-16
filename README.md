# AI Adventures — Giao diện đồ án AI

Ứng dụng PyGame gồm menu và ba màn hình: N-Queens, Tank Battle và Kakuro.
Toàn bộ chữ hiển thị trong ứng dụng dùng tiếng Anh.

## Yêu cầu hệ thống

- Python 3.10 trở lên.
- Git (không bắt buộc, nhưng cần nếu muốn lấy mã nguồn bằng `git clone`).
- `pygame` để chạy giao diện.
- `python-sat` để chạy N-Queens bằng Glucose3.
- `matplotlib` để chạy các script thí nghiệm và tạo biểu đồ.
- `jupyter`, `ipython` và `pandas` để chạy các notebook.

Các module như `enum`, `csv`, `json`, `math`, `pathlib`, `random`,
`statistics`, `threading`, `time` và `typing` là thư viện chuẩn của Python,
không cần cài riêng. Danh sách package đầy đủ nằm trong
[`requirements.txt`](requirements.txt).

## Cài đặt

Mở terminal tại thư mục chứa `main.py` và tạo môi trường ảo.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS/Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

Sau khi kích hoạt môi trường ảo:

```sh
python main.py
```

Chọn task trên menu; bấm **Main menu** hoặc **Esc** trong task để quay lại.

## Chạy thí nghiệm và notebook

Các notebook phải được mở từ thư mục gốc của project để các import `Task1`,
`Task2` và `Task3` hoạt động:

```sh
python -m jupyter notebook
```

Có thể chạy các script thí nghiệm bằng:

```sh
python Task1/run_benchmarks.py
python Task2/experiments/run_experiments.py
python Task3/experiments/run_experiments.py
```

Script Task 2 và Task 3 ghi CSV/PNG vào thư mục `experiments/results/`.
Các thí nghiệm có thể mất nhiều thời gian hơn chạy giao diện.

## Các chức năng

- **N-Queens:** solver Glucose3 và Standard/Custom Min-Conflicts, phát lại từng
  bước và biểu đồ CSV. N từ 4–200; Glucose3 trong GUI giới hạn N ≤ 64.
- **Tank Battle:** giao diện, bốn bot tìm kiếm đối kháng, bot người và import
  bot `.py` bên ngoài. P1 dùng W/A/S/D, F, G; P2 dùng phím mũi tên, K, L.
- **Kakuro:** đọc bàn cờ JSON trong `Task3/boards/`, AC-3, Forward Checking,
  MRV/Degree/LCV, hiển thị miền giá trị, lan truyền, quay lui và mâu thuẫn.

## Cấu trúc

```text
main.py                       # Điểm chạy ứng dụng
requirements.txt              # Package runtime, thí nghiệm và notebook
frontend/                     # Menu, giao diện, kết nối backend
assets/                       # Ảnh dùng cho giao diện
Task1/                        # N-Queens, benchmark và CSV
Task2/                        # Tank Battle, bot và thí nghiệm
Task3/                        # Kakuro, boards, notebook và thí nghiệm
backend_config.example.json   # Mẫu chỉnh đường dẫn import
tests/                        # Kiểm tra solver và giao diện
```

## Ghép code backend với UI

Mặc định giao diện tự import:

```text
Task2/tank_game_engine.py      # TankGameEngine, Action
Task2/agents.py                # BaseTankAgent và các lớp bot
Task3/kakuro_csp.py            # KakuroCSP
```

Nếu dùng module khác, sao chép `backend_config.example.json` thành
`backend_config.json` rồi sửa đường dẫn module. Khởi động lại ứng dụng sau khi
thêm hoặc sửa backend.

Các API chính:

- `TankGameEngine()`, `reset() -> dict`, `clone() -> engine`,
  `step(action_p1, action_p2) -> (state, sub_ticks)`.
- Bot kế thừa `BaseTankAgent`, khởi tạo bằng `player_id`, `max_depth`;
  `get_action(engine_copy) -> (Action, metrics)`.
- `KakuroCSP(board_id="board_5x5")` hoặc `"board_10x10"`;
  `get_board_layout()` và `solve_step_by_step()`.

## Kiểm tra

```sh
python -m unittest discover -s tests
```

Các test giao diện dùng SDL dummy nên không cần mở cửa sổ. Có thể kiểm tra
nhanh các dependency bằng:

```sh
python -c "import matplotlib, pandas, pygame, pysat; print('Dependencies OK')"
```

Không commit `.venv`, `__pycache__` hoặc cache Python vào source package.
