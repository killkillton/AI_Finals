"""Contract integration tests, without opening a desktop window."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from enum import Enum
from pathlib import Path
import random
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import pygame
from frontend.app import App
from frontend.backends import preview_kakuro, preview_tank


class Action(Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    FIRE = "FIRE"
    STAY = "STAY"


class Engine:
    def __init__(self):
        self.turn = 0
        self.clones = []

    def reset(self):
        self.turn = 0
        return preview_tank()

    def clone(self):
        clone = Engine()
        self.clones.append(clone)
        return clone

    def step(self, first, second):
        assert first == second == Action.STAY
        self.turn += 1
        state = preview_tank()
        state["turn"] = self.turn
        return state, [preview_tank()]


class BaseAgent:
    def __init__(self, player_id, max_depth=3):
        self.player_id = player_id


class Bot(BaseAgent):
    def get_action(self, engine_copy):
        engine_copy.turn = 999
        return Action.STAY, {"time_ms": 1, "explored_states": 4, "depth_reached": 2}


class Kakuro:
    def __init__(self, board_id="board_5x5"):
        self.board_id = board_id

    def get_board_layout(self):
        return preview_kakuro()

    def solve_step_by_step(self):
        for kind in ["PROPAGATE", "BACKTRACK", "SOLVED"]:
            yield {
                "event_type": kind, "focus_cell": (1, 1), "assigned_value": None,
                "pruned_cells": [(1, 2, [8, 9])] if kind == "PROPAGATE" else [],
                "assignments": {(1, 1): 1} if kind == "SOLVED" else {},
                "domains": {(1, 1): [1, 2, 3] if kind == "PROPAGATE" else list(range(1, 10))},
                "stats": {"steps": 1, "backtracks": 1, "prunings": 2, "elapsed_ms": 1},
            }


class FrontendTests(unittest.TestCase):
    def setUp(self):
        # Keep the tests independent of future teammate backend additions.
        with patch("frontend.backends.load_module", side_effect=ModuleNotFoundError("Not installed")):
            self.app = App()
        self.app.open_task(0)

    def tearDown(self):
        pygame.quit()

    def wait_job(self):
        deadline = time.monotonic() + 5
        while self.app.busy and time.monotonic() < deadline:
            self.app.receive_jobs()
            time.sleep(.005)
        self.assertFalse(self.app.busy, "Background job did not finish")
        self.assertIsNone(self.app.modal, self.app.modal_value)

    def test_screens_render_without_backend_at_minimum_size(self):
        self.app.resize((1100, 820))
        for task in range(3):
            self.app.task = task
            self.app.draw()
        self.app.task, self.app.show_chart = 0, True
        self.app.draw()
        self.assertFalse(self.app.tank_play)
        self.assertIsNone(self.app.kakuro_generator)

    def test_real_local_solver_and_replay_without_sat(self):
        random.seed(42)
        self.app.n = 4
        self.app.solve_queens()
        self.wait_job()
        self.assertTrue(self.app.queen_result["solved"])
        from Task1.n_queens_solver import NQueenBoard
        self.assertTrue(NQueenBoard(4).is_valid_solution(self.app.queen_board))
        self.app.replay()
        self.assertTrue(self.app.history)
        self.app.draw()

    def test_bots_get_separate_copies_and_frames_are_queued(self):
        self.app.backends.tank = Engine
        self.app.backends.action = Action
        self.app.backends.agents = {"Bot": Bot}
        self.app.bot_names = ["Bot", "Bot"]
        self.app.reset_tank()
        self.app.tank_step()
        self.wait_job()
        self.assertEqual(self.app.tank_engine.turn, 1)
        self.assertEqual(len(self.app.tank_engine.clones), 2)
        self.assertIsNot(self.app.tank_engine.clones[0], self.app.tank_engine.clones[1])
        self.assertEqual(self.app.tank_frames[-1]["turn"], 1)
        self.assertEqual(set(self.app.tank_metrics), {1, 2})

    def test_human_turn_waits_for_both_actions(self):
        self.app.backends.tank = Engine
        self.app.backends.action = Action
        self.app.reset_tank()
        self.app.tank_step()
        self.assertEqual(self.app.tank_engine.turn, 0)
        self.app.human_actions = ["STAY", "STAY"]
        self.app.tank_step()
        self.wait_job()
        self.assertEqual(self.app.tank_engine.turn, 1)
        self.assertEqual(self.app.human_actions, [None, None])

    def test_damage_snapshots_update_tank_and_base_health_and_reset(self):
        class DamageEngine(Engine):
            def step(self, first, second):
                state, _ = super().step(first, second)
                state["tanks"][1]["hp"] = 5
                state["bases"][2]["hp"] = 8
                return state, [preview_tank()]

        self.app.backends.tank = DamageEngine
        self.app.backends.action = Action
        self.app.open_task(1)
        self.app.reset_tank()
        self.app.draw()
        self.assertEqual(self.app.tank_scene.health_visuals[("tank", 1)]["hp"], 6)
        self.assertEqual(self.app.tank_scene.health_visuals[("base", 2)]["hp"], 9)
        self.app.human_actions = ["STAY", "STAY"]
        self.app.tank_step()
        self.wait_job()
        for _ in range(2):
            self.app.last_tank_tick = 0
            self.app.update()
            self.app.draw()
        self.assertEqual(self.app.tank_scene.health_visuals[("tank", 1)]["hp"], 5)
        self.assertEqual(self.app.tank_scene.health_visuals[("base", 2)]["hp"], 8)
        self.app.reset_tank()
        self.app.draw()
        self.assertEqual(self.app.tank_scene.health_visuals[("tank", 1)]["fill"], 1)
        self.assertEqual(self.app.tank_scene.health_visuals[("base", 2)]["fill"], 1)

    def test_kakuro_restores_domains_and_stops_at_solved(self):
        self.app.backends.kakuro = Kakuro
        self.app.reset_kakuro()
        self.app.task = 2
        self.app.kakuro_step()
        self.wait_job()
        self.assertEqual(self.app.kakuro_event["domains"][(1, 1)], [1, 2, 3])
        self.app.draw()
        self.app.kakuro_step()
        self.wait_job()
        self.assertEqual(len(self.app.kakuro_event["domains"][(1, 1)]), 9)
        self.app.kakuro_step()
        self.wait_job()
        self.assertTrue(self.app.kakuro_done)
        self.assertEqual(self.app.kakuro_event["event_type"], "SOLVED")

    def test_external_bot_contract(self):
        self.app.backends.agent_base = BaseAgent
        module = types.ModuleType("test_agent_contract")
        module.BaseTankAgent = BaseAgent
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"test_agent_contract": module}):
            path = Path(folder) / "bot.py"
            path.write_text("from test_agent_contract import BaseTankAgent\nclass ExternalBot(BaseTankAgent):\n    pass\n")
            self.assertEqual(self.app.backends.import_bot(path), ["ExternalBot"])
            path.write_text("class WrongBot:\n    pass\n")
            with self.assertRaises(ValueError):
                self.app.backends.import_bot(path)

    def test_missing_sat_dependency_shows_install_message(self):
        self.app.algorithm = 2
        with patch.dict("sys.modules", {"pysat": None}):
            self.app.solve_queens()
            deadline = time.monotonic() + 5
            while self.app.busy and time.monotonic() < deadline:
                self.app.receive_jobs()
                time.sleep(.005)
        self.assertFalse(self.app.busy)
        self.assertIsNotNone(self.app.modal)
        self.assertIn("requirements.txt", self.app.modal_value)

    def test_pause_is_available_while_generator_is_busy(self):
        self.app.backends.kakuro = Kakuro
        self.app.reset_kakuro()
        self.app.task = 2
        self.app.kakuro_play = True
        self.app.busy, self.app.work_task = True, 2
        self.app.draw()
        # The play/pause button is the second control beneath the board selector.
        pause = next(callback for rect, callback in self.app.buttons
                     if getattr(callback, "__name__", "") == "toggle_kakuro")
        pause()
        self.assertFalse(self.app.kakuro_play)
        self.app.busy = False

    def test_main_menu_click_and_escape_navigation(self):
        self.app.go_menu()
        self.app.draw()
        self.assertEqual(len(self.app.buttons), 4)
        # The third wooden sign opens Kakuro, not a task-selection overlay.
        self.app.buttons[2][1]()
        self.assertEqual((self.app.page, self.app.task), ("task", 2))
        self.app.kakuro_play = True
        self.app.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertEqual(self.app.page, "menu")
        self.assertFalse(self.app.kakuro_play)

    def test_menu_keyboard_and_task_specific_controls(self):
        self.app.go_menu()
        self.app.menu.selected = 0
        self.app.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        self.app.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        self.assertEqual((self.app.page, self.app.task), ("task", 1))
        with patch.object(self.app, "button", wraps=self.app.button) as button:
            self.app.draw()
            labels = [call.args[0] for call in button.call_args_list]
        self.assertIn("← Main menu", labels)
        self.assertFalse(any("backend" in label.lower() for label in labels))
        self.assertNotIn("Solve", labels)
        self.assertNotIn("Quick solve", labels)

    def test_tank_control_instructions_keep_explicit_line_breaks(self):
        self.app.open_task(1)
        with patch.object(self.app, "text", wraps=self.app.text) as text:
            self.app.draw()
        rendered = {call.args[0]: call.args[1:3] for call in text.call_args_list}
        p1 = rendered["P1 · Move: W A S D"]
        p2 = rendered["P2 · Move: Arrow keys"]
        self.assertEqual(p1[0], p2[0])
        self.assertGreater(p2[1], p1[1])
        self.assertIn("Fire: F | Stay: G", rendered)
        self.assertIn("Fire: K | Stay: L", rendered)

    def test_transparent_text_and_queen_leave_scene_visible(self):
        from frontend.rendering import Canvas
        self.assertIsInstance(self.app.screen, Canvas)
        background = pygame.Color("#46A5C4")
        self.app.screen.fill(background)
        title = self.app.font(37, True).render("N-QUEENS", True, "#FFF0CA")
        self.assertEqual(title.get_at((0, 0)).a, 0)
        self.app.screen.blit(title, (10, 10))
        self.assertEqual(self.app.screen.get_at((10, 10)), background)
        piece = self.app.queens_scene.sprite(68)
        self.assertEqual(piece.get_at((0, 0)).a, 0)
        self.app.screen.blit(piece, (20, 80))
        self.assertEqual(self.app.screen.get_at((20, 80)), background)
        self.app.resize((1100, 820))
        self.assertIsInstance(self.app.screen, Canvas)
        self.assertEqual(self.app.screen.get_size(), self.app.window.get_size())

    def test_board_click_places_moves_removes_and_clears_queens(self):
        self.app.draw()
        (ox, oy), unit, _ = self.app.queens_scene.grid_geometry

        def click(row, col):
            self.app.handle_click((ox + (col + .5) * unit, oy + (row + .5) * unit))

        click(1, 0)
        self.assertEqual(self.app.fixed_queens, {0: 1})
        click(3, 0)
        self.assertEqual(self.app.fixed_queens, {0: 3})
        click(3, 0)
        self.assertFalse(self.app.fixed_queens)
        click(0, 0)
        click(1, 1)
        self.assertEqual(self.app.queen_conflicts, {0, 1})
        self.app.solve_queens()
        self.assertFalse(self.app.busy)
        self.assertIsNotNone(self.app.modal)
        self.app.modal = None
        self.app.clear_queens()
        self.assertFalse(self.app.fixed_queens)
        self.assertFalse(self.app.queen_board)

    def test_ui_solve_preserves_manual_placements_and_edit_clears_replay(self):
        random.seed(42)
        self.app.place_queen(0, 0)
        self.app.place_queen(4, 1)
        self.app.solve_queens()
        self.wait_job()
        self.assertTrue(self.app.queen_result["solved"])
        self.assertEqual(self.app.queen_board[:2], [0, 4])
        self.app.replay()
        self.app.place_queen(7, 2)
        self.assertFalse(self.app.queen_play)
        self.assertFalse(self.app.history)
        self.assertFalse(self.app.queen_result)
        self.app.set_n("4")
        self.assertFalse(self.app.fixed_queens)

    def test_board_edit_is_disabled_during_work_and_chart_view(self):
        self.app.draw()
        (ox, oy), unit, _ = self.app.queens_scene.grid_geometry
        position = (ox + unit / 2, oy + unit / 2)
        self.app.busy = True
        self.app.handle_click(position)
        self.assertFalse(self.app.fixed_queens)
        self.app.busy = False
        self.app.show_chart = True
        self.app.handle_click(position)
        self.assertFalse(self.app.fixed_queens)


if __name__ == "__main__":
    unittest.main()
