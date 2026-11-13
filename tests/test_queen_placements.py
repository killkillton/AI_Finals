import importlib.util
import random
import unittest

from Task1.n_queens_solver import NQueenSolver


class FixedPlacementTests(unittest.TestCase):
    def test_local_search_keeps_fixed_queens_through_moves_and_restarts(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                random.seed(42)
                fixed = {0: 0, 1: 4}
                solver = NQueenSolver(8, fixed)
                result = solver.solve_min_conflicts(custom, record_history=True)
                self.assertTrue(result["solved"])
                self.assertTrue(solver.board_helper.is_valid_solution(result["board"]))
                self.assertGreater(result["restarts"], 0)
                for _, _, board in result["history"]:
                    for col, row in fixed.items():
                        self.assertEqual(board[col], row)
                for col, row in fixed.items():
                    self.assertEqual(result["board"][col], row)

    def test_invalid_or_conflicting_placements_are_rejected(self):
        for fixed in ({0: 0, 1: 1}, {0: 1, 2: 1}, {8: 0}, {0: -1}, {"0": 1}):
            with self.subTest(fixed=fixed), self.assertRaises(ValueError):
                NQueenSolver(8, fixed)

    def test_complete_manual_solution_needs_no_repairs(self):
        fixed = dict(enumerate([1, 3, 0, 2]))
        for custom in (False, True):
            result = NQueenSolver(4, fixed).solve_min_conflicts(custom, max_steps=0)
            self.assertTrue(result["solved"])
            self.assertEqual(result["steps"], 0)

    def test_incomplete_board_at_zero_budget_is_not_a_solution(self):
        result = NQueenSolver(4, {0: 0}).solve_min_conflicts(max_steps=0)
        self.assertFalse(result["solved"])
        self.assertEqual(result["board"][0], 0)

    def test_cnf_keeps_original_clauses_and_adds_fixed_unit_clauses(self):
        base = NQueenSolver(4).generate_cnf_clauses()
        fixed = NQueenSolver(4, {0: 1}).generate_cnf_clauses()
        self.assertEqual(len(base), 80)
        self.assertEqual(fixed[:-1], base)
        self.assertEqual(fixed[-1], [5])

    @unittest.skipUnless(importlib.util.find_spec("pysat"), "python-sat not installed")
    def test_sat_completes_or_proves_unsatisfiable_with_fixed_queen(self):
        solver = NQueenSolver(4, {0: 1})
        result = solver.solve_pysat()
        self.assertTrue(result["solved"])
        self.assertTrue(solver.board_helper.is_valid_solution(result["board"]))
        self.assertEqual(result["board"][0], 1)
        self.assertEqual(result["num_clauses"], 81)
        # A corner queen on a 4x4 board is non-attacking on its own, but cannot
        # be extended to a complete solution. SAT must report UNSAT.
        impossible = NQueenSolver(4, {0: 0}).solve_pysat()
        self.assertFalse(impossible["solved"])
        self.assertEqual(impossible["board"], [])
