import random
import time
from typing import Dict, List, Tuple
from pysat.solvers import Glucose3


class NQueenBoard:
    """
    Represents an N x N chessboard for the N-Queens problem.
    Provides helper methods for conflict tracking, propositional variable mapping,
    and board validation.
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("Board size N must be a positive integer.")
        self.n = n

    def var_id(self, row: int, col: int) -> int:
        """
        Maps a 0-indexed (row, col) board square to a 1-indexed propositional
        variable ID in the range [1, N*N] for SAT solvers.
        """
        return row * self.n + col + 1

    def pos_from_var(self, var: int) -> Tuple[int, int]:
        """
        Maps a 1-indexed propositional variable ID back to 0-indexed (row, col).
        """
        idx = abs(var) - 1
        return idx // self.n, idx % self.n

    def is_valid_solution(self, board: List[int]) -> bool:
        """
        Verifies whether a 1D board array (col_idx -> row_idx) has N queens
        with no two queens attacking each other.
        """
        if len(board) != self.n or any(r < 0 or r >= self.n for r in board):
            return False

        for c1 in range(self.n):
            r1 = board[c1]
            for c2 in range(c1 + 1, self.n):
                r2 = board[c2]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True


class NQueenSolver:
    """
    Contract 1: Task 1 - N-Queens Solver (Tin <-> Nh).
    Implements Propositional Logic (CNF + PySAT Glucose3) and Local Search
    (Standard vs. Custom Heuristic Min-Conflicts).
    """

    def __init__(self, n: int):
        self.n = n
        self.board_helper = NQueenBoard(n)

    def generate_cnf_clauses(self) -> List[List[int]]:
        """
        Generates Conjunctive Normal Form (CNF) clauses governing the placement
        of N queens on an N x N board:
        1. At least one queen per column.
        2. At most one queen per column.
        3. At most one queen per row.
        4. At most one queen per main/anti-diagonal.
        """
        n = self.n
        clauses: List[List[int]] = []

        # 1. At least one queen in each column: (X_{0,c} v X_{1,c} v ... v X_{N-1,c})
        for c in range(n):
            clauses.append([self.board_helper.var_id(r, c) for r in range(n)])

        # 2. At most one queen in each column: (~X_{r1,c} v ~X_{r2,c})
        for c in range(n):
            for r1 in range(n):
                u = self.board_helper.var_id(r1, c)
                for r2 in range(r1 + 1, n):
                    v = self.board_helper.var_id(r2, c)
                    clauses.append([-u, -v])

        # 3. At most one queen in each row: (~X_{r,c1} v ~X_{r,c2})
        for r in range(n):
            for c1 in range(n):
                u = self.board_helper.var_id(r, c1)
                for c2 in range(c1 + 1, n):
                    v = self.board_helper.var_id(r, c2)
                    clauses.append([-u, -v])

        # 4. At most one queen on each diagonal: (~X_{r1,c1} v ~X_{r2,c2})
        for r1 in range(n):
            for c1 in range(n):
                u = self.board_helper.var_id(r1, c1)
                # Down-right diagonal (row + 1, col + 1)
                dr, dc = r1 + 1, c1 + 1
                while dr < n and dc < n:
                    v = self.board_helper.var_id(dr, dc)
                    clauses.append([-u, -v])
                    dr += 1
                    dc += 1

                # Down-left diagonal (row + 1, col - 1)
                dr, dc = r1 + 1, c1 - 1
                while dr < n and dc >= 0:
                    v = self.board_helper.var_id(dr, dc)
                    clauses.append([-u, -v])
                    dr += 1
                    dc -= 1

        return clauses

    def solve_pysat(self) -> dict:
        """
        Solves N-Queens using PySAT Glucose3.
        Returns:
        {
            "algorithm": "Glucose3",
            "n": int,
            "solved": bool,
            "board": list[int],     # 1D list of length N: col_idx -> row_idx of queen
            "num_variables": int,   # N * N
            "num_clauses": int,     # REQUIRED metric for Report Section 1.2
            "runtime_ms": float     # REQUIRED metric for Report Section 1.2
        }
        """
        start_time = time.perf_counter()
        clauses = self.generate_cnf_clauses()

        board = [-1] * self.n
        solved = False

        with Glucose3() as solver:
            for clause in clauses:
                solver.add_clause(clause)

            if solver.solve():
                solved = True
                model = solver.get_model()
                # Positive literals in the model correspond to squares containing a queen
                for lit in model:
                    if lit > 0:
                        r, c = self.board_helper.pos_from_var(lit)
                        board[c] = r

        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "algorithm": "Glucose3",
            "n": self.n,
            "solved": solved,
            "board": board if solved else [],
            "num_variables": self.n * self.n,
            "num_clauses": len(clauses),
            "runtime_ms": round(runtime_ms, 4),
        }

    def _init_conflicts_tables(
        self, board: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Builds O(1) lookup tables for the number of queens on each row,
        main diagonal (r - c + N - 1), and anti-diagonal (r + c).
        """
        n = self.n
        row_counts = [0] * n
        diag1_counts = [0] * (2 * n - 1)
        diag2_counts = [0] * (2 * n - 1)

        for col, row in enumerate(board):
            row_counts[row] += 1
            diag1_counts[row - col + n - 1] += 1
            diag2_counts[row + col] += 1

        return row_counts, diag1_counts, diag2_counts

    def _count_conflicts(
        self,
        row: int,
        col: int,
         current_row: int,
        row_counts: List[int],
        diag1_counts: List[int],
        diag2_counts: List[int],
    ) -> int:
        """
        Computes how many other queens attack square (row, col), excluding the
        queen currently placed at (current_row, col).
        """
        n = self.n
        conflicts = (
            row_counts[row]
            + diag1_counts[row - col + n - 1]
            + diag2_counts[row + col]
        )
        # If checking the square where this column's queen already sits, subtract its own 3 counts
        if row == current_row:
            conflicts -= 3
        return conflicts

    def _initialize_board(
        self, use_custom_heuristic: bool
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Initializes the board state and conflict tables.
        - Standard: Places one queen per column in a uniformly random row.
        - Custom Heuristic: Greedily places each column's queen in a row that minimizes
          immediate conflicts with already placed queens (ties broken randomly).
        """
        n = self.n
        board = [0] * n
        row_counts = [0] * n
        diag1_counts = [0] * (2 * n - 1)
        diag2_counts = [0] * (2 * n - 1)

        if not use_custom_heuristic:
            for col in range(n):
                row = random.randrange(n)
                board[col] = row
                row_counts[row] += 1
                diag1_counts[row - col + n - 1] += 1
                diag2_counts[row + col] += 1
        else:
            for col in range(n):
                min_conf = n + 1
                candidates = []
                for row in range(n):
                    conf = (
                        row_counts[row]
                        + diag1_counts[row - col + n - 1]
                        + diag2_counts[row + col]
                    )
                    if conf < min_conf:
                        min_conf = conf
                        candidates = [row]
                    elif conf == min_conf:
                        candidates.append(row)

                chosen_row = random.choice(candidates)
                board[col] = chosen_row
                row_counts[chosen_row] += 1
                diag1_counts[chosen_row - col + n - 1] += 1
                diag2_counts[chosen_row + col] += 1

        return board, row_counts, diag1_counts, diag2_counts

    def solve_min_conflicts(
        self,
        use_custom_heuristic: bool = False,
        max_steps: int = 10000,
        record_history: bool = False,
    ) -> dict:
        """
        Solves N-Queens using Standard or Proposed Heuristic Min-Conflicts.
        - Standard (Lecture 08): Random initial placement; picks a random conflicted column.
        - Custom Heuristic: Greedy min-conflict initial placement; picks the column with the
          highest number of conflicts (Max-Conflicts variable selection heuristic) with a
          recently-moved tie-breaker to avoid 2-cycles.

        Returns:
        {
            "algorithm": "MinConflicts_Custom" | "MinConflicts_Standard",
            "n": int,
            "solved": bool,
            "board": list[int],     # Final queen positions
            "steps": int,           # Number of repair steps taken
            "runtime_ms": float,
            "history": list[tuple[int, int, list[int]]] # Optional: [(step, col, board), ...]
        }
        """
        start_time = time.perf_counter()
        n = self.n
        algo_name = (
            "MinConflicts_Custom" if use_custom_heuristic else "MinConflicts_Standard"
        )

        board, row_counts, diag1_counts, diag2_counts = self._initialize_board(
            use_custom_heuristic
        )
        history: List[Tuple[int, int, List[int]]] = []
        last_col = -1
        # Short tabu tenure for Custom Heuristic to escape strict local minima plateaus
        tabu_until = [0] * n
        tabu_tenure = max(1, min(4, n // 4))

        for step in range(max_steps):
            # Gather all columns whose queen is currently under attack
            conflicted_cols = []
            max_col_conf = -1
            max_conf_cols = []

            for col in range(n):
                r = board[col]
                conf = self._count_conflicts(
                    r, col, r, row_counts, diag1_counts, diag2_counts
                )
                if conf > 0:
                    conflicted_cols.append(col)
                    if use_custom_heuristic and step >= tabu_until[col]:
                        if conf > max_col_conf:
                            max_col_conf = conf
                            max_conf_cols = [col]
                        elif conf == max_col_conf:
                            max_conf_cols.append(col)

            # No conflicts remain -> valid solution found
            if not conflicted_cols:
                runtime_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "algorithm": algo_name,
                    "n": n,
                    "solved": True,
                    "board": board,
                    "steps": step,
                    "runtime_ms": round(runtime_ms, 4),
                    "history": history,
                }

            # Variable selection: Standard picks randomly; Custom picks most-conflicted non-tabu variable
            if not use_custom_heuristic:
                col = random.choice(conflicted_cols)
            else:
                if not max_conf_cols:
                    # Fallback if all conflicted columns are currently tabu
                    candidates = [c for c in conflicted_cols if c != last_col] or conflicted_cols
                    col = random.choice(candidates)
                else:
                    col = random.choice(max_conf_cols)

            # Value selection (Min-Conflicts): pick row in 'col' that minimizes conflicts
            current_row = board[col]
            min_conf = n + 1
            best_rows = []

            for r in range(n):
                # When evaluating a move in custom mode, prefer moving to a different row if possible
                conf = self._count_conflicts(
                    r, col, current_row, row_counts, diag1_counts, diag2_counts
                )
                if conf < min_conf:
                    min_conf = conf
                    best_rows = [r]
                elif conf == min_conf:
                    best_rows.append(r)

            # If multiple rows tie for minimum conflicts in custom mode, avoid staying on current_row
            if use_custom_heuristic and len(best_rows) > 1 and current_row in best_rows:
                best_rows.remove(current_row)

            new_row = random.choice(best_rows)

            # Update conflict lookup tables in O(1) time
            if new_row != current_row:
                row_counts[current_row] -= 1
                diag1_counts[current_row - col + n - 1] -= 1
                diag2_counts[current_row + col] -= 1

                board[col] = new_row
                row_counts[new_row] += 1
                diag1_counts[new_row - col + n - 1] += 1
                diag2_counts[new_row + col] += 1

            last_col = col
            if use_custom_heuristic:
                tabu_until[col] = step + tabu_tenure
            if record_history:
                history.append((step + 1, col, list(board)))

        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "algorithm": algo_name,
            "n": n,
            "solved": False,
            "board": board,
            "steps": max_steps,
            "runtime_ms": round(runtime_ms, 4),
            "history": history,
        }
