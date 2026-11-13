import csv
import importlib.util
from pathlib import Path
import unittest

from Task1.n_queens_solver import NQueenBoard, NQueenSolver


class Task1ContractTests(unittest.TestCase):
    """Automated unit tests verifying Contract 1 (Task 1: N-Queens) schemas and CSVs."""

    def test_board_variable_bijection_and_validation(self):
        board = NQueenBoard(8)
        for r in range(8):
            for c in range(8):
                var = board.var_id(r, c)
                self.assertTrue(1 <= var <= 64)
                self.assertEqual(board.pos_from_var(var), (r, c))

        self.assertTrue(board.is_valid_solution([0, 4, 7, 5, 2, 6, 1, 3]))
        self.assertFalse(board.is_valid_solution([0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertFalse(board.is_valid_solution([0, 0, 7, 5, 2, 6, 1, 3]))

    @unittest.skipUnless(importlib.util.find_spec("pysat"), "python-sat not installed")
    def test_pysat_solver_schema_and_counts(self):
        for n, expected_clauses in ((4, 80), (8, 736)):
            with self.subTest(n=n):
                solver = NQueenSolver(n)
                res = solver.solve_pysat()
                self.assertEqual(
                    set(res.keys()),
                    {
                        "algorithm",
                        "n",
                        "solved",
                        "board",
                        "num_variables",
                        "num_clauses",
                        "runtime_ms",
                    },
                )
                self.assertTrue(res["solved"])
                self.assertEqual(res["num_variables"], n * n)
                self.assertEqual(res["num_clauses"], expected_clauses)
                self.assertTrue(solver.board_helper.is_valid_solution(res["board"]))

    def test_min_conflicts_standard_and_custom_heuristic(self):
        solver = NQueenSolver(8)
        for use_custom in (False, True):
            with self.subTest(use_custom_heuristic=use_custom):
                res = solver.solve_min_conflicts(
                    use_custom_heuristic=use_custom,
                    max_steps=5000,
                    record_history=True,
                )
                self.assertEqual(
                    set(res.keys()),
                    {
                        "algorithm",
                        "n",
                        "solved",
                        "board",
                        "steps",
                        "restarts",
                        "runtime_ms",
                        "history",
                    },
                )
                self.assertTrue(res["solved"])
                self.assertTrue(solver.board_helper.is_valid_solution(res["board"]))
                self.assertGreater(len(res["history"]), 0)

    def test_benchmark_csv_files_exist_and_have_valid_rows(self):
        task1_dir = Path(__file__).resolve().parent.parent / "Task1"
        pysat_csv = task1_dir / "task1_pysat_benchmark.csv"
        mc_csv = task1_dir / "task1_min_conflicts_benchmark.csv"
        self.assertTrue(pysat_csv.exists())
        self.assertTrue(mc_csv.exists())

        with pysat_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 5)
            self.assertEqual(
                set(rows[0].keys()),
                {"N", "num_variables", "num_clauses", "runtime_ms"},
            )

        with mc_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 5)
            self.assertEqual(
                set(rows[0].keys()),
                {"N", "variant", "avg_steps", "avg_runtime_ms", "success_rate"},
            )


if __name__ == "__main__":
    unittest.main()
