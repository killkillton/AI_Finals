import json
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import pygame

from Task3.kakuro_csp import KakuroCSP
from frontend.app import App


class KakuroCSPTests(unittest.TestCase):
    """Automated unit and integration tests for Task 3A & Contract 3 (KakuroCSP)."""

    def test_invalid_board_id_or_missing_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            KakuroCSP("board_invalid")
        with self.assertRaises(ValueError):
            KakuroCSP(board_path="nonexistent_board.json")

    def test_board_layouts_match_contract_3_and_exact_pdf_clues(self):
        expected_5x5_clues = {
            (0, 2): (21, None),
            (0, 3): (14, None),
            (1, 1): (14, 12),
            (1, 4): (11, None),
            (2, 0): (None, 21),
            (3, 0): (None, 11),
            (4, 1): (None, 16),
        }
        expected_10x10_clues = {
            (0, 2): (44, None),
            (0, 3): (16, None),
            (0, 4): (9, None),
            (0, 6): (15, None),
            (0, 7): (44, None),
            (1, 1): (None, 22),
            (1, 5): (11, 3),
            (1, 8): (14, None),
            (2, 1): (17, 28),
            (3, 0): (None, 14),
            (3, 4): (12, 20),
            (4, 0): (None, 17),
            (4, 3): (29, 22),
            (4, 8): (7, None),
            (5, 1): (12, 28),
            (5, 6): (None, 12),
            (6, 0): (None, 14),
            (6, 5): (7, None),
            (6, 6): (12, 6),
            (7, 0): (None, 39),
            (8, 1): (None, 16),
            (8, 4): (None, 8),
        }

        cases = (
            ("board_5x5", (5, 5), 12, expected_5x5_clues),
            ("board_10x10", (10, 10), 49, expected_10x10_clues),
        )
        for board_id, expected_dims, expected_whites, expected_clues in cases:
            with self.subTest(board_id=board_id):
                csp = KakuroCSP(board_id)
                layout = csp.get_board_layout()
                self.assertEqual((layout["rows"], layout["cols"]), expected_dims)
                self.assertEqual(len(layout["cells"]), expected_dims[0] * expected_dims[1])

                actual_clues = {}
                white_count = 0
                for cell in layout["cells"]:
                    r, c = cell["r"], cell["c"]
                    self.assertIn(cell["type"], ("BLOCK", "CLUE", "WHITE"))
                    if cell["type"] == "WHITE":
                        white_count += 1
                    elif cell["type"] == "CLUE":
                        actual_clues[(r, c)] = (cell.get("down_sum"), cell.get("right_sum"))

                self.assertEqual(white_count, expected_whites)
                self.assertEqual(actual_clues, expected_clues)

    def test_general_json_parser_supports_custom_files_and_cell_schemas(self):
        custom_board = {
            "board_id": "custom_3x3",
            "rows": 3,
            "cols": 3,
            "grid": [
                ["#", "3/-", "4/-"],
                ["-/4", "W", "W"],
                ["-/3", "W", "W"],
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "custom_3x3.json"
            json_path.write_text(json.dumps(custom_board), encoding="utf-8")

            csp = KakuroCSP.from_json(json_path)
            self.assertEqual((csp.rows, csp.cols), (3, 3))
            self.assertEqual(len(csp.variables), 4)
            events = list(csp.solve_step_by_step())
            self.assertEqual(events[-1]["event_type"], "SOLVED")
            self.assertTrue(csp.is_valid_solution(events[-1]["assignments"]))

            # Also verify explicit 'cells' JSON schema (even when shuffled)
            layout_dict = csp.get_board_layout()
            layout_dict["cells"] = list(reversed(layout_dict["cells"]))
            csp_from_cells = KakuroCSP.from_json(layout_dict)
            self.assertEqual(len(csp_from_cells.variables), 4)
            self.assertEqual(len(csp_from_cells.constraints), 4)

    def test_json_parser_rejects_malformed_boards_and_cells(self):
        # 1. Duplicate coordinates in 'cells'
        dup_cells = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"r": 0, "c": 0, "type": "BLOCK"},
                {"r": 0, "c": 0, "type": "BLOCK"},
                {"r": 1, "c": 0, "type": "BLOCK"},
                {"r": 1, "c": 1, "type": "BLOCK"},
            ],
        }
        with self.assertRaises(ValueError):
            KakuroCSP.from_json(dup_cells)

        # 2. Missing coordinates in 'cells'
        missing_cells = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"r": 0, "c": 0, "type": "BLOCK"},
                {"r": 0, "c": 1, "type": "BLOCK"},
                {"r": 1, "c": 0, "type": "BLOCK"},
            ],
        }
        with self.assertRaises(ValueError):
            KakuroCSP.from_json(missing_cells)

        # 3. Out-of-bounds coordinates in 'cells'
        oob_cells = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"r": 0, "c": 0, "type": "BLOCK"},
                {"r": 0, "c": 1, "type": "BLOCK"},
                {"r": 1, "c": 0, "type": "BLOCK"},
                {"r": 2, "c": 1, "type": "BLOCK"},
            ],
        }
        with self.assertRaises(ValueError):
            KakuroCSP.from_json(oob_cells)

        # 4. Invalid cell type in 'cells' or 'grid'
        invalid_type = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"r": 0, "c": 0, "type": "UNKNOWN"},
                {"r": 0, "c": 1, "type": "BLOCK"},
                {"r": 1, "c": 0, "type": "BLOCK"},
                {"r": 1, "c": 1, "type": "BLOCK"},
            ],
        }
        with self.assertRaises(ValueError):
            KakuroCSP.from_json(invalid_type)

        # 5. CLUE with neither down_sum nor right_sum (in dict, list, and string formats)
        for bad_clue in (
            {"r": 0, "c": 0, "type": "CLUE", "down_sum": None, "right_sum": None},
            ["C", None, None],
            "-/-",
        ):
            with self.subTest(bad_clue=bad_clue), self.assertRaises(ValueError):
                KakuroCSP.from_json({"rows": 1, "cols": 1, "grid": [[bad_clue]]})

    def test_every_white_cell_has_horizontal_and_vertical_constraints(self):
        for board_id in ("board_5x5", "board_10x10"):
            with self.subTest(board_id=board_id):
                csp = KakuroCSP(board_id)
                for var in csp.variables:
                    self.assertEqual(csp.domains[var], list(range(1, 10)))
                    dirs = {c.direction for c in csp.cell_constraints[var]}
                    self.assertEqual(
                        dirs,
                        {"right", "down"},
                        f"Cell {var} on {board_id} is missing a run constraint: {dirs}",
                    )

    def test_ac3_and_forward_checking_prune_domains_and_detect_wipeout(self):
        csp = KakuroCSP("board_5x5")
        assignments = {}
        domains = {v: list(csp.domains[v]) for v in csp.variables}

        ok, pruned = csp.propagate_ac3(assignments, domains)
        self.assertTrue(ok)
        self.assertTrue(len(pruned) > 0)
        self.assertNotIn(1, domains[(1, 2)])
        self.assertNotIn(2, domains[(1, 2)])
        self.assertNotIn(6, domains[(1, 2)])

        # Assigning a supported value (1,2)=8 forces (1,3)=4 in the horizontal run (1,2)+(1,3)=12
        valid_assignments = {(1, 2): 8}
        valid_domains = {v: list(csp.domains[v]) for v in csp.variables}
        valid_domains[(1, 2)] = [8]
        fc_ok, fc_pruned = csp.forward_check((1, 2), 8, valid_assignments, valid_domains)
        self.assertTrue(fc_ok)
        self.assertEqual(valid_domains[(1, 3)], [4])
        self.assertTrue(any(r == 1 and c == 3 for r, c, _ in fc_pruned))

        # Force a contradiction: assign (1,2)=8 when (1,3)'s domain only has [9]
        contradiction_domains = {v: list(csp.domains[v]) for v in csp.variables}
        contradiction_domains[(1, 3)] = [9]
        fc_ok_fail, _ = csp.forward_check((1, 2), 8, {(1, 2): 8}, contradiction_domains)
        self.assertFalse(fc_ok_fail)

    def test_solve_step_by_step_uses_forward_checking_and_ac3_on_every_step(self):
        valid_events = {"SELECT_VAR", "ASSIGN", "PROPAGATE", "CONFLICT", "BACKTRACK", "SOLVED"}

        # 1. 5x5 board solves to completion and calls both forward_check and propagate_ac3 on every assignment
        csp_5x5 = KakuroCSP("board_5x5")
        with (
            patch.object(csp_5x5, "forward_check", wraps=csp_5x5.forward_check) as fc_spy,
            patch.object(csp_5x5, "propagate_ac3", wraps=csp_5x5.propagate_ac3) as ac3_spy,
        ):
            events_5x5 = list(csp_5x5.solve_step_by_step())
            self.assertEqual(fc_spy.call_count, len(csp_5x5.variables))
            # 1 initial AC-3 pass + 1 AC-3 pass after each valid assignment
            self.assertEqual(ac3_spy.call_count, 1 + fc_spy.call_count)

        for ev in events_5x5:
            self.assertIn(ev["event_type"], valid_events)
            self.assertEqual(
                set(ev["stats"].keys()),
                {"steps", "backtracks", "prunings", "elapsed_ms"},
            )
        self.assertEqual(events_5x5[-1]["event_type"], "SOLVED")
        self.assertTrue(csp_5x5.is_valid_solution(events_5x5[-1]["assignments"]))
        self.assertEqual(len(events_5x5[-1]["domains"]), 0)

        # 2. Exact PDF 10x10 board is mathematically unsolvable and terminates with CONFLICT
        csp_10x10 = KakuroCSP("board_10x10")
        events_10x10 = list(csp_10x10.solve_step_by_step())
        self.assertGreater(len(events_10x10), 0)
        self.assertEqual(events_10x10[-1]["event_type"], "CONFLICT")
        self.assertFalse(csp_10x10.is_valid_solution(events_10x10[-1]["assignments"]))

    def test_frontend_app_integrates_with_real_kakuro_backend(self):
        app = App()
        try:
            app.open_task(2)
            # 5x5 solves to SOLVED
            self.assertEqual(app.kakuro.board_id, "board_5x5")
            app.kakuro_step(instant=True)
            deadline = time.monotonic() + 5
            while app.busy and time.monotonic() < deadline:
                app.receive_jobs()
                time.sleep(0.005)
            self.assertFalse(app.busy)
            self.assertTrue(app.kakuro_done)
            self.assertEqual(app.kakuro_event["event_type"], "SOLVED")
            app.draw()

            # 10x10 (exact PDF) detects unsolvability and ends cleanly at CONFLICT
            app.change_kakuro()
            self.assertEqual(app.kakuro.board_id, "board_10x10")
            app.kakuro_step(instant=True)
            deadline = time.monotonic() + 5
            while app.busy and time.monotonic() < deadline:
                app.receive_jobs()
                time.sleep(0.005)
            self.assertFalse(app.busy)
            self.assertTrue(app.kakuro_done)
            self.assertEqual(app.kakuro_event["event_type"], "CONFLICT")
            app.draw()
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
