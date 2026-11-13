from __future__ import annotations

import argparse
import csv
import os
import queue
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from .backends import Backends, ROOT, preview_kakuro, preview_tank
from .menu import GameMenu
from .queens_scene import QueensScene
from .tank_scene import TankScene
from .rendering import Canvas
from .kakuro_scene import KakuroScene

BG = "#0C1220"
PANEL = "#151F30"
CARD = "#1D2A3D"
LINE = "#2B3A51"
TEXT = "#ECF2FA"
MUTED = "#94A6BD"
TEAL = "#57E2C0"
BLUE = "#76ABFF"
ORANGE = "#FFBD73"
RED = "#FF8193"
PURPLE = "#B8A1FF"


class App:
    def __init__(self, size=(1280, 860)):
        pygame.init()
        pygame.display.set_caption("AI Adventures · Introduction to AI")
        self.resize(size)
        self.clock = pygame.time.Clock()
        self.fonts = {}
        self.buttons = []
        self.running = True
        self.task = 0
        self.page = "menu"
        self.menu = GameMenu(ROOT / "assets" / "menu_background.png")
        self.queens_scene = QueensScene()
        self.tank_scene = TankScene()
        self.kakuro_scene = KakuroScene()
        self.backends = Backends()
        self.messages = {i: [] for i in range(3)}
        self.jobs = queue.Queue()
        self.busy = False
        self.work_task = None
        self.modal = None
        self.modal_value = ""
        self.n = 8
        self.algorithm = 1
        self.algorithms = ["Min-Conflicts · Standard", "Min-Conflicts · Custom", "PySAT · Glucose3"]
        self.queen_board = []
        self.fixed_queens = {}
        self.queen_conflicts = set()
        self.queen_result = {}
        self.history = []
        self.history_index = 0
        self.queen_play = False
        self.show_chart = False
        self.tank_engine = None
        self.tank_state = preview_tank()
        self.tank_frames = []
        self.tank_play = False
        self.tank_metrics = {}
        self.bot_names = ["Human", "Human"]
        self.bot_instances = [None, None]
        self.human_actions = [None, None]
        self.depth = 3
        self.kakuro = None
        self.kakuro_layout = preview_kakuro()
        self.kakuro_event = {}
        self.kakuro_generator = None
        self.kakuro_done = False
        self.kakuro_play = False
        self.board_id = "board_5x5"
        self.speed_index = 1
        self.speeds = [1, 4, 12, 30]
        self.last_tick = time.monotonic()
        self.last_tank_tick = 0
        self.log(0, "Click cells to place queens, or press Solve to start from an empty board.")
        self.reset_tank()
        self.reset_kakuro()

    def resize(self, size):
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.screen = Canvas(size, depth=32)

    def font(self, size=18, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("Arial", size, bold=bold)
        return self.fonts[key]

    @property
    def queen_style(self):
        return self.page == "task" and self.task == 0

    @property
    def tank_style(self):
        return self.page == "task" and self.task == 1

    @property
    def kakuro_style(self):
        return self.page == "task" and self.task == 2

    def text(self, value, x, y, size=18, color=TEXT, bold=False, max_width=None):
        value = str(value)
        if self.queen_style:
            color = self.queens_scene.color(color)
        elif self.tank_style:
            color = self.tank_scene.color(color)
        elif self.kakuro_style:
            color = self.kakuro_scene.color(color)
        font = self.font(size, bold)
        if max_width:
            while value and font.size(value)[0] > max_width:
                value = value[:-2] if value.endswith("…") else value[:-1]
                value += "…"
        self.screen.blit(font.render(value, True, color), (int(x), int(y)))

    def wrapped(self, value, x, y, width, size=16, color=MUTED, limit=4):
        lines = []
        # Preserve explicit line breaks and empty lines before wrapping by width.
        for paragraph in str(value).split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                if self.font(size).size(candidate)[0] > width and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        for i, line in enumerate(lines[:limit]):
            self.text(line, x, y + i * (size + 6), size, color, max_width=width)
        return min(len(lines), limit) * (size + 6)

    def box(self, rect, color=PANEL, radius=14, border=None):
        if self.queen_style:
            return self.queens_scene.panel(self, rect, color, radius, border)
        if self.tank_style:
            return self.tank_scene.panel(self, rect, color, radius, border)
        if self.kakuro_style:
            return self.kakuro_scene.panel(self, rect, color, radius, border)
        rect = pygame.Rect(rect)
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=radius)
        return rect

    def button(self, label, rect, action, primary=False, enabled=True, selected=False):
        if self.queen_style:
            return self.queens_scene.button(self, label, rect, action, primary, enabled, selected)
        if self.tank_style:
            return self.tank_scene.button(self, label, rect, action, primary, enabled, selected)
        if self.kakuro_style:
            return self.kakuro_scene.button(self, label, rect, action, primary, enabled, selected)
        rect = pygame.Rect(rect)
        hovered = rect.collidepoint(pygame.mouse.get_pos()) and enabled
        fill = TEAL if primary and enabled else ("#293D54" if hovered or selected else CARD)
        self.box(rect, fill, 9, TEAL if selected else None)
        color = BG if primary and enabled else (TEXT if enabled else "#617187")
        font = self.font(16, primary)
        while font.size(label)[0] > rect.w - 18 and len(label) > 2:
            label = label[:-2] + "…" if label.endswith("…") else label[:-1] + "…"
        image = font.render(label, True, color)
        self.screen.blit(image, image.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, action))

    def log(self, task, message):
        stamp = time.strftime("%H:%M:%S")
        self.messages[task].append(f"{stamp}  {message}")
        self.messages[task] = self.messages[task][-40:]

    def launch(self, task, fn, on_success):
        if self.busy:
            return
        self.busy, self.work_task = True, task

        def run():
            try:
                self.jobs.put((task, on_success, fn(), None))
            except Exception as exc:
                self.jobs.put((task, on_success, None, exc))
        threading.Thread(target=run, daemon=True).start()

    def receive_jobs(self):
        try:
            task, callback, result, error = self.jobs.get_nowait()
        except queue.Empty:
            return
        self.busy, self.work_task = False, None
        if error:
            self.tank_play = self.kakuro_play = False
            if isinstance(error, ModuleNotFoundError) and (error.name or "").split(".")[0] == "pysat":
                message = "PySAT is missing: run python -m pip install -r requirements.txt"
            else:
                message = f"Error: {error}"
            self.log(task, message)
            self.modal = ("Notice", None)
            self.modal_value = message
        else:
            try:
                callback(result)
            except Exception as exc:
                self.tank_play = self.kakuro_play = False
                self.log(task, f"Invalid solver data: {exc}")
                self.modal = ("Unable to read data", None)
                self.modal_value = str(exc)

    def ask(self, title, value, callback):
        self.modal, self.modal_value = (title, callback), str(value)
        pygame.key.start_text_input()

    def submit_modal(self):
        callback = self.modal[1]
        if callback:
            try:
                callback(self.modal_value.strip())
            except Exception as exc:
                self.log(self.task, str(exc))
                self.modal = ("Notice", None)
                self.modal_value = str(exc)
                return
        self.modal = None
        pygame.key.stop_text_input()

    def set_n(self, value):
        n = int(value)
        if not 4 <= n <= 200:
            raise ValueError("Enter N between 4 and 200.")
        self.n = n
        self.clear_queens()

    def clear_queens(self):
        if self.busy:
            return
        self.fixed_queens = {}
        self.queen_conflicts = set()
        self.queen_board, self.history, self.queen_result = [], [], {}
        self.history_index = 0
        self.queen_play = False

    def place_queen(self, row, col):
        if self.busy or not (0 <= row < self.n and 0 <= col < self.n):
            return
        if self.fixed_queens.get(col) == row:
            del self.fixed_queens[col]
        else:
            self.fixed_queens[col] = row
        self.queen_conflicts = set()
        positions = list(self.fixed_queens.items())
        for i, (c1, r1) in enumerate(positions):
            for c2, r2 in positions[i + 1:]:
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    self.queen_conflicts.update((c1, c2))
        self.queen_board = [self.fixed_queens.get(c, -1) for c in range(self.n)] if self.fixed_queens else []
        self.queen_result, self.history = {}, []
        self.history_index = 0
        self.queen_play = False
        if self.queen_conflicts:
            self.log(0, "Queens with red borders attack each other. Move or remove them before solving.")
        else:
            self.log(0, f"Locked {len(self.fixed_queens)}/{self.n} queens. Press Solve to complete the board.")

    def solve_queens(self):
        if self.busy:
            return
        if self.queen_conflicts:
            self.modal = ("The placed queens attack each other", None)
            self.modal_value = "Move or remove the queens with red borders before solving. Your placements will stay locked."
            return
        if self.algorithm == 2 and self.n > 64:
            self.log(0, "SAT is limited to N ≤ 64 in this interface to limit memory use. Use Min-Conflicts for larger boards.")
            return
        from Task1.n_queens_solver import NQueenSolver
        solver = NQueenSolver(self.n, fixed_queens=self.fixed_queens)
        self.show_chart = False
        self.queen_play = False
        self.log(0, f"Solving N={self.n} · {self.algorithms[self.algorithm]}")
        algorithm = self.algorithm

        def solve():
            if algorithm == 2:
                return solver.solve_pysat()
            return solver.solve_min_conflicts(use_custom_heuristic=algorithm == 1,
                                              max_steps=10000, record_history=True)

        def done(result):
            if result["solved"] and not solver.board_helper.is_valid_solution(result["board"]):
                raise ValueError("The solver returned an invalid board.")
            if result["solved"] and any(result["board"][c] != r for c, r in solver.fixed_queens.items()):
                raise ValueError("The solver moved a locked queen.")
            self.queen_result = result
            self.history = result.get("history", [])
            self.history_index = 0
            self.queen_board = result["board"] or ([self.fixed_queens.get(c, -1) for c in range(self.n)] if self.fixed_queens else [])
            status = "Solution found" if result["solved"] else (
                "No solution exists with these locked queens" if result["algorithm"] == "Glucose3"
                else "Step limit reached; no solution found (not a proof of impossibility)")
            self.log(0, f"{status} · {result['runtime_ms']:.3f} ms")
        self.launch(0, solve, done)

    def replay(self):
        if self.history:
            self.history_index = 0
            self.queen_board = self.history[0][2]
            self.queen_play = True

    def open_task(self, task):
        self.task = task
        self.page = "task"
        self.last_tick = time.monotonic()

    def go_menu(self):
        self.queen_play = self.tank_play = self.kakuro_play = False
        self.page = "menu"

    def reset_tank(self):
        self.tank_scene.reset_health()
        self.tank_play = False
        self.tank_frames = []
        self.tank_metrics = {}
        self.human_actions = [None, None]
        self.bot_instances = [None, None]
        self.tank_engine = None
        self.tank_state = preview_tank()
        if self.backends.tank:
            try:
                self.tank_engine = self.backends.tank()
                self.tank_state = self.tank_engine.reset()
                self.log(1, "Choose both players, then start the match.")
            except Exception as exc:
                self.tank_engine = None
                self.log(1, f"Unable to start the game: {exc}")
        else:
            self.log(1, "Preview only · Tank Battle gameplay is still in development.")
        for i, name in enumerate(self.bot_names):
            if name != "Human" and name not in self.backends.agents:
                self.bot_names[i] = "Human"

    def cycle_bot(self, player):
        choices = ["Human"] + list(self.backends.agents)
        index = choices.index(self.bot_names[player]) if self.bot_names[player] in choices else 0
        self.bot_names[player] = choices[(index + 1) % len(choices)]
        self.bot_instances[player] = None
        self.tank_play = False
        self.human_actions[player] = None

    def import_bot(self, path):
        names = self.backends.import_bot(path)
        self.log(1, f"Imported bots: {', '.join(names)}. Click a player selector to choose one.")

    def tank_step(self):
        if not self.tank_engine or self.tank_frames or self.tank_state.get("is_terminal"):
            return
        for i, name in enumerate(self.bot_names):
            if name == "Human" and self.human_actions[i] is None:
                return
        engine = self.tank_engine
        names, pending, instances = list(self.bot_names), list(self.human_actions), list(self.bot_instances)
        classes, action_enum, depth = dict(self.backends.agents), self.backends.action, self.depth
        self.human_actions = [None, None]

        def step():
            actions, metrics = [], {}
            for i, name in enumerate(names):
                if name == "Human":
                    actions.append(action_enum[pending[i]])
                else:
                    if instances[i] is None:
                        instances[i] = classes[name](player_id=i + 1, max_depth=depth)
                    action, metric = instances[i].get_action(engine.clone())
                    actions.append(action)
                    metrics[i + 1] = metric
            state, frames = engine.step(*actions)
            return state, list(frames), metrics, instances

        def done(result):
            state, frames, metrics, instances = result
            self.bot_instances = instances
            self.tank_metrics = metrics
            self.tank_frames = frames + [state]
            self.last_tank_tick = time.monotonic()
        self.launch(1, step, done)

    def reset_kakuro(self):
        self.kakuro_play = self.kakuro_done = False
        self.kakuro = self.kakuro_generator = None
        self.kakuro_layout = preview_kakuro()
        self.kakuro_event = {}
        if self.backends.kakuro:
            try:
                self.kakuro = self.backends.kakuro(board_id=self.board_id)
                self.kakuro_layout = self.kakuro.get_board_layout()
                self.kakuro_generator = iter(self.kakuro.solve_step_by_step())
                self.log(2, f"Loaded {self.board_id}. Press Step or Play.")
            except Exception as exc:
                self.kakuro = self.kakuro_generator = None
                self.log(2, f"Unable to start Kakuro: {exc}")
        else:
            self.log(2, "Sample board · Kakuro solving is still in development.")

    def change_kakuro(self):
        self.board_id = "board_10x10" if self.board_id == "board_5x5" else "board_5x5"
        self.reset_kakuro()

    def kakuro_step(self, instant=False):
        if self.kakuro_generator is None or self.kakuro_done:
            return
        generator = self.kakuro_generator

        def work():
            latest = None
            # Instant mode drains in bounded batches, preserving a responsive UI
            # and allowing Pause between batches even on a long search.
            for _ in range(500 if instant else 1):
                try:
                    latest = next(generator)
                except StopIteration:
                    return latest, True
                if latest.get("event_type") == "SOLVED":
                    return latest, True
            return latest, False

        def done(result):
            event, ended = result
            if event is not None:
                self.kakuro_event = event
                stats = event.get("stats", {})
                self.log(2, f"{event['event_type']} · cell {event.get('focus_cell')} · step {stats.get('steps', '—')}")
            self.kakuro_done = ended
            if ended:
                self.kakuro_play = False
                if self.kakuro_event.get("event_type") != "SOLVED":
                    self.log(2, "Search ended without a SOLVED event.")
        self.launch(2, work, done)

    def cycle_speed(self):
        self.speed_index = (self.speed_index + 1) % len(self.speeds)

    def draw_header(self, width):
        titles = ["N-Queens Playground", "Tank Battle Arena", "Kakuro Visualizer"]
        subtitles = ["Place N queens · explore algorithms · replay solutions",
                     "Two opponents · one arena · explore AI decisions",
                     "Explore domains, constraint propagation, and backtracking"]
        self.text("AI ADVENTURES  /  TASK 0" + str(self.task + 1), 28, 25, 12, TEAL, True)
        self.text(titles[self.task], 27, 50, 30, bold=True)
        self.text(subtitles[self.task], 29, 91, 15, MUTED, max_width=width - 240)
        self.button("← Main menu", (width - 170, 32, 142, 41), self.go_menu)

    def draw_board(self, rect):
        self.box(rect)
        label = ["CHESSBOARD", "BATTLEFIELD", "KAKURO BOARD"][self.task]
        self.text(label, rect.x + 20, rect.y + 17, 12, MUTED, True)
        badge = f"LOCKED {len(self.fixed_queens)}/{self.n} QUEENS · CLICK TO PLACE / REMOVE" if self.task == 0 else (
            "MATCH" if self.task == 1 and self.tank_engine else "DOMAINS & SEARCH" if self.task == 2 and self.kakuro else "PREVIEW")
        self.text(badge, rect.x + 20, rect.y + 40, 12, ORANGE if badge == "PREVIEW" else TEAL)
        area = pygame.Rect(rect.x + 25, rect.y + 76, rect.w - 50, rect.h - 100)
        if self.task == 0 and self.show_chart:
            self.draw_chart(area)
            return
        if self.task == 0:
            rows = cols = self.n
        elif self.task == 1:
            rows, cols = self.tank_state["grid_size"]
        else:
            rows, cols = self.kakuro_layout["rows"], self.kakuro_layout["cols"]
        unit = min(area.w / cols, area.h / rows)
        origin = (area.centerx - unit * cols / 2, area.centery - unit * rows / 2)
        if self.task == 0:
            self.draw_queens(origin, unit)
        elif self.task == 1:
            self.draw_tank(origin, unit, rows, cols)
        else:
            self.draw_kakuro(origin, unit)

    def draw_queens(self, origin, unit):
        self.queens_scene.grid(self, origin, unit)

    def draw_tank(self, origin, unit, rows, cols):
        self.tank_scene.battlefield(self, origin, unit, rows, cols)

    def draw_kakuro(self, origin, unit):
        self.kakuro_scene.grid(self, origin, unit)

    def draw_chart(self, area):
        path = ROOT / "Task1" / "task1_min_conflicts_benchmark.csv"
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            groups = {}
            for row in rows:
                groups.setdefault(row["variant"], []).append((int(row["N"]), float(row["avg_runtime_ms"])))
            points = [point for series in groups.values() for point in series]
            max_x, max_y = max(x for x, y in points), max(y for x, y in points)
        except Exception as exc:
            self.wrapped(f"Unable to read CSV: {exc}", area.x, area.y, area.w)
            return
        self.text("Min-Conflicts · Average runtime (ms)", area.x, area.y, 16)
        plot = pygame.Rect(area.x + 45, area.y + 55, area.w - 65, area.h - 120)
        for i in range(5):
            y = plot.bottom - plot.h * i / 4
            pygame.draw.line(self.screen, self.queens_scene.color(LINE) if self.queen_style else LINE, (plot.x, y), (plot.right, y))
            self.text(f"{max_y * i / 4:.1f}", area.x, y - 8, 12, MUTED)
        for index, (name, series) in enumerate(sorted(groups.items())):
            color = TEAL if "Custom" in name else PURPLE
            if self.queen_style:
                color = "#51745C" if "Custom" in name else "#AE7738"
            series.sort()
            coords = [(plot.x + x / max(max_x, 1) * plot.w, plot.bottom - y / max(max_y, .001) * plot.h) for x, y in series]
            if len(coords) > 1:
                pygame.draw.lines(self.screen, color, False, coords, 2)
            for point in coords:
                pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 4)
            self.text(name.replace("MinConflicts_", ""), area.x + index * 150, area.bottom - 25, 14, color)
        self.text("N →", plot.right - 28, plot.bottom + 13, 13, MUTED)
        self.text(f"{max_x}", plot.right - 18, plot.bottom - 23, 12, MUTED)

    def draw_controls(self, rect):
        self.box(rect)
        x, y, w = rect.x + 20, rect.y + 20, rect.w - 40
        self.text("CONTROLS", x, y, 12, MUTED, True)
        y += 36
        available = not self.busy

        def btn(label, fn, primary=False, enabled=True, height=40):
            nonlocal y
            self.button(label, (x, y, w, height), fn, primary, enabled)
            y += height + 10

        if self.task == 0:
            self.text("Board size", x, y, 15, MUTED)
            y += 26
            btn(f"N = {self.n}   ·   Click to edit", lambda: self.ask("Board size N (4–200)", self.n, self.set_n), enabled=available)
            self.text("Algorithm · click to change", x, y + 2, 15, MUTED)
            y += 28
            btn(self.algorithms[self.algorithm], lambda: setattr(self, "algorithm", (self.algorithm + 1) % 3), enabled=available)
            btn("Solving…" if self.busy and self.work_task == 0 else "Solve", self.solve_queens, True, available)
            btn("Clear board", self.clear_queens, enabled=available)
            btn("Replay steps", self.replay, enabled=available and bool(self.history))
            btn("Pause / Resume", lambda: setattr(self, "queen_play", not self.queen_play), enabled=bool(self.history))
            btn("Board / CSV chart", lambda: setattr(self, "show_chart", not self.show_chart))
            btn(f"Replay speed: {self.speeds[self.speed_index]} steps/s", self.cycle_speed)
            self.wrapped("Click to place/remove; click another row to move. Gold: locked. Red: conflict.", x, y + 3, w, 13, limit=3)
        elif self.task == 1:
            ready = self.tank_engine is not None
            for i in range(2):
                self.text(f"PLAYER {i + 1}", x, y, 12, "#70BE52" if i == 0 else "#E35D4E", True)
                y += 23
                btn(self.bot_names[i], lambda i=i: self.cycle_bot(i), enabled=available and ready and not self.tank_frames)
            btn("Import external bot (.py)", lambda: self.ask("Path to bot file (.py)", "", self.import_bot),
                enabled=available and self.backends.agent_base is not None)
            btn(f"Search depth: {self.depth}", self.change_depth, enabled=available and ready and not self.tank_frames)
            btn("Pause" if self.tank_play else "Start match", lambda: setattr(self, "tank_play", not self.tank_play), True, ready and (available or self.tank_play) and not self.tank_state.get("is_terminal"))
            btn("Next turn", self.tank_step, enabled=ready and available and not self.tank_play and not self.tank_frames and not self.tank_state.get("is_terminal"))
            btn("Restart match", self.reset_tank, enabled=available)
            self.wrapped("P1 · Move: W A S D\nFire: F  |  Stay: G\n\nP2 · Move: Arrow keys\nFire: K  |  Stay: L\nPick one action per turn.", x, y + 4, w, 13, limit=7)
        else:
            ready = self.kakuro_generator is not None and not self.kakuro_done
            btn(f"Board: {'5 × 5' if self.board_id == 'board_5x5' else '10 × 10'}", self.change_kakuro,
                enabled=available and self.backends.kakuro is not None)
            btn("Pause" if self.kakuro_play else "Play steps", self.toggle_kakuro, True, ready and (available or self.kakuro_play))
            btn("Step", lambda: self.kakuro_step(), enabled=ready and available and not self.kakuro_play)
            btn("Quick solve", self.instant_kakuro, enabled=ready and available)
            btn(f"Speed: {self.speeds[self.speed_index]} steps/s", self.cycle_speed)
            btn("Reset", self.reset_kakuro, enabled=available)
            y += 7
            for label, color in [("Select / Propagate", BLUE), ("Assign / Solved", TEAL),
                                 ("Pruned values", ORANGE), ("Conflict / Backtrack", RED)]:
                pygame.draw.circle(self.screen, self.kakuro_scene.color(color), (x + 5, y + 8), 4)
                self.text(label, x + 18, y, 14, MUTED)
                y += 29
            self.wrapped("Small digits show each cell’s domain. Values update with solver events.", x, y + 4, w, 14)

    def change_depth(self):
        self.depth = self.depth % 5 + 1
        self.bot_instances = [None, None]
        self.tank_play = False

    def toggle_kakuro(self):
        self.instant_mode = False
        self.kakuro_play = not self.kakuro_play

    def instant_kakuro(self):
        self.instant_mode = True
        self.kakuro_play = True

    def draw_metrics(self, rect):
        if self.task == 0:
            result = self.queen_result
            status = "Solving" if self.busy and self.work_task == 0 else (
                "Conflict" if self.queen_conflicts else "Replaying" if self.queen_play else "Solved" if result.get("solved")
                else "No solution" if result.get("algorithm") == "Glucose3" else "Not found" if result
                else f"Placed {len(self.fixed_queens)}/{self.n}" if self.fixed_queens else "Not solved")
            metrics = [("STATUS", status), ("RUNTIME", f"{result.get('runtime_ms', 0):.3f} ms"),
                       ("STEPS / CNF", result.get("steps", result.get("num_clauses", "—"))),
                       ("REPLAY", f"{self.history_index}/{len(self.history)}")]
        elif self.task == 1:
            state = self.tank_state
            metrics = [("TURN", state.get("turn", 0))]
            for player in [1, 2]:
                tanks, bases = state.get("tanks", {}), state.get("bases", {})
                tank = tanks.get(player, tanks.get(str(player), {}))
                base = bases.get(player, bases.get(str(player), {}))
                metrics.append((f"P{player} · TANK / BASE", f"{tank.get('hp', 0)}/6 · {base.get('hp', 0)}/9"))
            winner = state.get("winner")
            status = f"Winner: {winner}" if state.get("is_terminal") else "Playing" if self.tank_play else "Pause" if self.tank_engine else "Preview"
            metrics.append(("STATUS", status))
        else:
            stats = self.kakuro_event.get("stats", {})
            metrics = [("EVENT", self.kakuro_event.get("event_type", "Not started")),
                       ("STEPS", stats.get("steps", 0)), ("BACKTRACKS / PRUNED", f"{stats.get('backtracks', 0)} / {stats.get('prunings', 0)}"),
                       ("RUNTIME", f"{stats.get('elapsed_ms', 0):.2f} ms")]
        width = (rect.w - 30) / 4
        for i, (label, value) in enumerate(metrics):
            x = rect.x + i * (width + 10)
            self.box((x, rect.y, width, rect.h))
            self.text(label, x + 14, rect.y + 13, 11, MUTED, True, max_width=width - 25)
            self.text(value, x + 14, rect.y + 37, 19, TEAL if i == 0 else TEXT, True, max_width=width - 25)

    def draw_log(self, rect):
        self.box(rect)
        self.text("ACTIVITY LOG", rect.x + 17, rect.y + 12, 11, MUTED, True)
        lines = list(self.messages[self.task])
        if self.task == 1:
            for player, metric in self.tank_metrics.items():
                lines.append(f"AI P{player} · {metric.get('time_ms', 0):.2f} ms · {metric.get('explored_states', 0)} states · depth {metric.get('depth_reached', '—')}")
            if self.tank_engine:
                queued = [f"P{i+1}: {self.human_actions[i] or 'awaiting input'}" for i in range(2) if self.bot_names[i] == "Human"]
                if queued:
                    lines.append("Queued actions · " + " | ".join(queued))
        count = max(1, (rect.h - 40) // 22)
        for i, line in enumerate(lines[-count:]):
            self.text(line, rect.x + 17, rect.y + 36 + i * 22, 14, MUTED, max_width=rect.w - 34)

    def draw_modal(self):
        width, height = self.screen.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        rect = pygame.Rect(width // 2 - 285, height // 2 - 125, 570, 250)
        self.box(rect, PANEL, 16, LINE)
        title, callback = self.modal
        self.text(title, rect.x + 25, rect.y + 22, 21, bold=True)
        if callback:
            input_color = "#FAEDCB" if self.queen_style or self.kakuro_style else BG
            self.box((rect.x + 24, rect.y + 75, rect.w - 48, 48), input_color, 8, TEAL)
            # Tail remains visible when entering a long absolute path.
            value = self.modal_value
            while self.font(17).size(value)[0] > rect.w - 75:
                value = value[1:]
            self.text(value + "|", rect.x + 35, rect.y + 88, 17)
            self.text("Enter: confirm · Esc: cancel · Cmd/Ctrl+V: paste", rect.x + 25, rect.y + 139, 13, MUTED)
        else:
            self.wrapped(self.modal_value, rect.x + 25, rect.y + 70, rect.w - 50, 16, TEXT, 4)
        self.button("Confirm" if callback else "Close", (rect.right - 154, rect.bottom - 59, 130, 37), self.submit_modal, True)

    def draw(self):
        self.buttons = []
        width, height = self.screen.get_size()
        if self.page == "menu":
            self.menu.draw(self)
        elif self.task == 0:
            self.queens_scene.draw(self)
        elif self.task == 1:
            self.tank_scene.draw(self)
        else:
            self.kakuro_scene.draw(self)
        if self.modal:
            self.buttons = []
            self.draw_modal()
        # Only an opaque, fully composed frame is copied onto the native window.
        self.window.blit(self.screen, (0, 0))
        pygame.display.flip()

    def key(self, event):
        if self.modal:
            if event.key == pygame.K_ESCAPE:
                self.modal = None
                pygame.key.stop_text_input()
            elif event.key == pygame.K_RETURN:
                self.submit_modal()
            elif self.modal[1] and event.key == pygame.K_BACKSPACE:
                self.modal_value = self.modal_value[:-1]
            elif self.modal[1] and event.key == pygame.K_v and event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
                try:
                    pygame.scrap.init()
                    pasted = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if pasted:
                        self.modal_value += pasted.decode("utf-8").rstrip("\0")
                except (pygame.error, UnicodeError):
                    self.log(self.task, "Unable to read clipboard. Type the path instead.")
            return
        if self.page == "menu":
            self.menu.key(self, event)
            return
        if event.key == pygame.K_ESCAPE:
            self.go_menu()
            return
        if self.task != 1 or not self.tank_engine or self.busy or self.tank_frames or self.tank_state.get("is_terminal"):
            return
        keys = [{pygame.K_w: "MOVE_UP", pygame.K_s: "MOVE_DOWN", pygame.K_a: "MOVE_LEFT", pygame.K_d: "MOVE_RIGHT", pygame.K_f: "FIRE", pygame.K_g: "STAY"},
                {pygame.K_UP: "MOVE_UP", pygame.K_DOWN: "MOVE_DOWN", pygame.K_LEFT: "MOVE_LEFT", pygame.K_RIGHT: "MOVE_RIGHT", pygame.K_k: "FIRE", pygame.K_l: "STAY"}]
        for i, mapping in enumerate(keys):
            if self.bot_names[i] == "Human" and event.key in mapping:
                self.human_actions[i] = mapping[event.key]

    def update(self):
        self.receive_jobs()
        if self.page == "menu":
            return
        now = time.monotonic()
        if self.task == 1 and self.tank_frames and now - self.last_tank_tick >= .09:
            self.tank_state = self.tank_frames.pop(0)
            self.last_tank_tick = now
            if not self.tank_frames and self.tank_state.get("is_terminal"):
                self.tank_play = False
                self.log(1, f"Game over · winner: {self.tank_state.get('winner')}")
        if self.modal:
            return
        if self.task == 1 and self.tank_play and not self.busy and not self.tank_frames:
            self.tank_step()
        interval = 1 / self.speeds[self.speed_index]
        if now - self.last_tick < interval:
            return
        self.last_tick = now
        if self.task == 0 and self.queen_play and self.history:
            if self.history_index < len(self.history):
                self.queen_board = self.history[self.history_index][2]
                self.history_index += 1
            else:
                self.queen_play = False
                self.queen_board = self.queen_result["board"]
        if self.task == 2 and self.kakuro_play and not self.busy:
            self.kakuro_step(getattr(self, "instant_mode", False))

    def handle_click(self, position):
        for rect, callback in reversed(self.buttons):
            if rect.collidepoint(position):
                callback()
                return
        if self.queen_style and not self.show_chart and not self.modal and not self.busy:
            cell = self.queens_scene.cell_at(position)
            if cell is not None:
                self.place_queen(*cell)

    def run(self, frames=None, screenshot=None):
        count = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.resize((max(1100, event.w), max(820, event.h)))
                elif event.type == pygame.KEYDOWN:
                    self.key(event)
                elif event.type == pygame.TEXTINPUT and self.modal and self.modal[1]:
                    self.modal_value += event.text
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.DROPFILE and self.page == "task" and self.task == 1 and not self.busy:
                    try:
                        self.import_bot(event.file)
                    except Exception as exc:
                        self.log(1, str(exc))
            self.update()
            self.draw()
            self.clock.tick(60)
            count += 1
            if frames and count >= frames:
                if screenshot:
                    pygame.image.save(self.screen, screenshot)
                break
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="AI final project desktop UI")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="Open a task directly; default opens the main menu")
    parser.add_argument("--frames", type=int, help="Exit after this many frames (smoke test)")
    parser.add_argument("--screenshot", help="Save final frame as PNG when --frames is set")
    args = parser.parse_args()
    app = App()
    if args.task is not None:
        app.open_task(args.task - 1)
    app.run(args.frames, args.screenshot)


if __name__ == "__main__":
    main()
