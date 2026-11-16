import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Task3.kakuro_csp import KakuroCSP

TASK3_DIR = ROOT / "Task3"
EVENT_KEYS = {"event_type", "focus_cell", "assigned_value", "pruned_cells", "assignments", "domains", "stats"}
EVENT_TYPES = {"SELECT_VAR", "ASSIGN", "PROPAGATE", "CONFLICT", "BACKTRACK", "SOLVED"}


class TestParsing(unittest.TestCase):
    def test_board_5x5(self):
        csp = KakuroCSP("board_5x5")
        self.assertEqual(len(csp.variables), 12)
        self.assertEqual(len(csp.constraints), 8)

    def test_board_10x10(self):
        csp = KakuroCSP("board_10x10")
        self.assertEqual(len(csp.variables), 48)
        self.assertEqual(len(csp.constraints), 28)
        self.assertEqual(csp.get_board_layout()["cells"][3 * 10 + 3]["type"], "BLOCK")

    def test_layout_keys(self):
        layout = KakuroCSP("board_5x5").get_board_layout()
        self.assertEqual(set(layout), {"rows", "cols", "cells"})
        self.assertEqual(len(layout["cells"]), 25)


class TestPropagation(unittest.TestCase):
    def test_forward_check_removes_value_and_bounds(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        assignments = {(1, 2): 9}
        domains[(1, 2)] = [9]
        ok, pruned = csp.forward_check((1, 2), 9, assignments, domains)
        self.assertTrue(ok)
        self.assertNotIn(9, domains[(1, 3)])
        self.assertNotIn(9, domains[(2, 2)])
        self.assertTrue(pruned)

    def test_forward_check_wipeout(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        domains[(1, 3)] = [1]
        assignments = {(1, 2): 1}
        domains[(1, 2)] = [1]
        ok, _ = csp.forward_check((1, 2), 1, assignments, domains)
        self.assertFalse(ok)

    def test_ac3_prunes(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        ok, pruned = csp.propagate_ac3({}, domains)
        self.assertTrue(ok)
        self.assertTrue(pruned)
        self.assertTrue(all(domains[v] for v in csp.variables))

    def test_ac3_detects_wipeout(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        domains[(1, 2)] = [1]
        domains[(1, 3)] = [1]
        ok, _ = csp.propagate_ac3({}, domains)
        self.assertFalse(ok)


class TestHeuristics(unittest.TestCase):
    def test_mrv_picks_smallest_domain(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        domains[(3, 4)] = [1, 2]
        self.assertEqual(csp._select_unassigned_variable({}, domains), (3, 4))

    def test_degree_breaks_ties(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        chosen = csp._select_unassigned_variable({}, domains)
        best = max(len(csp.peers[v]) for v in csp.variables)
        self.assertEqual(len(csp.peers[chosen]), best)

    def test_static_order(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        domains[(3, 4)] = [1]
        self.assertEqual(
            csp._select_unassigned_variable({}, domains, use_mrv=False, use_degree=False),
            csp.variables[0],
        )

    def test_lcv_order(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        var = (1, 2)
        ordered = csp._order_domain_values(var, {}, domains)
        self.assertEqual(sorted(ordered), [3, 4, 5, 7, 8, 9])
        self.assertEqual(csp._order_domain_values(var, {}, domains, use_lcv=False), list(range(1, 10)))

    def test_legal_run_values_sum_44(self):
        csp = KakuroCSP("board_10x10")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        self.assertEqual(csp._get_legal_run_values((2, 2), {}, domains), list(range(2, 8)))

    def test_lcv_pushes_wipeout_last(self):
        csp = KakuroCSP("board_5x5")
        domains = {v: list(range(1, 10)) for v in csp.variables}
        domains[(1, 3)] = [5]
        ordered = csp._order_domain_values((1, 2), {}, domains)
        self.assertEqual(ordered[0], 7)


class TestSolver(unittest.TestCase):
    def check_board(self, board_id):
        csp = KakuroCSP(board_id)
        events = list(csp.solve_step_by_step())
        self.assertEqual(events[-1]["event_type"], "SOLVED")
        self.assertTrue(csp.is_valid_solution(events[-1]["assignments"]))
        for ev in events:
            self.assertEqual(set(ev), EVENT_KEYS)
            self.assertIn(ev["event_type"], EVENT_TYPES)
            self.assertEqual(set(ev["stats"]), {"steps", "backtracks", "prunings", "elapsed_ms"})

        result = KakuroCSP(board_id).solve()
        self.assertEqual(
            set(result),
            {"board_id", "solved", "assignments", "steps", "assignments_tried",
             "backtracks", "prunings", "elapsed_ms"},
        )
        self.assertTrue(result["solved"])
        self.assertTrue(csp.is_valid_solution(result["assignments"]))
        return result

    def test_5x5(self):
        self.check_board("board_5x5")

    def test_10x10(self):
        result = self.check_board("board_10x10")
        self.assertEqual(result["backtracks"], 0)

    def test_all_configs_solve(self):
        for use_ac3 in (False, True):
            for heur in (False, True):
                csp = KakuroCSP("board_5x5")
                res = csp.solve(heur, heur, heur, use_ac3, True)
                self.assertTrue(res["solved"])
                self.assertTrue(csp.is_valid_solution(res["assignments"]))


class TestIntegrity(unittest.TestCase):
    def test_no_forbidden_strings_and_ascii(self):
        forbidden = ["Con" + "tract", "Lec" + "ture", "Sl" + "ide", "T" + "in", "N" + "h"]
        this_file = Path(__file__).resolve()
        for path in TASK3_DIR.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".png"}:
                continue
            if path.resolve() == this_file or path.suffix == ".ipynb":
                continue
            data = path.read_bytes()
            try:
                text = data.decode("ascii")
            except UnicodeDecodeError:
                self.fail(f"non-ASCII content in {path}")
            for word in forbidden:
                self.assertNotIn(word, text, f"{word!r} found in {path}")


if __name__ == "__main__":
    unittest.main()




