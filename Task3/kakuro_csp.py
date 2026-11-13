from collections import deque
import json
from pathlib import Path
import time
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple, Union


class KakuroConstraint:
    """
    Represents a single Kakuro run constraint (either horizontal 'right' or vertical 'down').
    Enforces two simultaneous rules over its ordered list of WHITE variable cells:
    1. AllDiff: No two cells in the run may share the same digit (1..9).
    2. Exact Sum: The sum of digits assigned to all cells in the run must equal target_sum.
    """

    def __init__(self, target_sum: int, cells: List[Tuple[int, int]], direction: str):
        self.target_sum = target_sum
        self.cells = list(cells)
        self.direction = direction  # "right" or "down"


class KakuroCSP:
    """
    Contract 3: Task 3 - Kakuro CSP (Tin & Ng <-> Nh).
    Task 3A (Tin):
    - General JSON Kakuro board parser supporting 'board_5x5', 'board_10x10', or any
      custom JSON board file/dictionary.
    - Models variables (WHITE cells), domains ({1..9}), and AllDiff + Sum constraints.
    - Combines Forward Checking (FC) and Generalized AC-3 (Arc Consistency) constraint
      propagation with MRV/Degree + LCV Backtracking Search.
    """

    BOARDS_DIR = Path(__file__).resolve().parent / "boards"

    def __init__(
        self,
        board_id: str = "board_5x5",
        board_path: Optional[Union[str, Path]] = None,
        board_data: Optional[Dict[str, Any]] = None,
    ):
        self.board_id = board_id
        if board_data is not None:
            raw_data = board_data
        else:
            resolved_path = self._resolve_board_path(board_id, board_path)
            raw_data = self.load_board_json(resolved_path)

        self.rows, self.cols, self.cells = self._parse_board_data(raw_data)

        # CSP Model Components:
        # 1. Variables: list of (r, c) coordinates for every WHITE cell
        self.variables: List[Tuple[int, int]] = [
            (cell["r"], cell["c"]) for cell in self.cells if cell["type"] == "WHITE"
        ]
        # 2. Domains: mapping (r, c) -> sorted list of valid digits in {1..9}
        self.domains: Dict[Tuple[int, int], List[int]] = {
            var: list(range(1, 10)) for var in self.variables
        }
        # 3. Constraints: list of KakuroConstraint runs & lookup from each cell to its runs
        self.constraints: List[KakuroConstraint] = []
        self.cell_constraints: Dict[Tuple[int, int], List[KakuroConstraint]] = {
            var: [] for var in self.variables
        }
        self.peers: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            var: set() for var in self.variables
        }
        self._extract_constraints()

    @classmethod
    def from_json(cls, source: Union[str, Path, Dict[str, Any]]) -> "KakuroCSP":
        """Convenience factory to construct a KakuroCSP from any JSON file path or dict."""
        if isinstance(source, dict):
            board_id = str(source.get("board_id", "custom_board"))
            return cls(board_id=board_id, board_data=source)
        path = Path(source)
        return cls(board_id=path.stem, board_path=path)

    @classmethod
    def _resolve_board_path(
        cls, board_id: str, board_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Resolves a board_id or explicit file path to a JSON board definition file."""
        if board_path is not None:
            path = Path(board_path)
            if not path.is_file():
                raise ValueError(f"Kakuro board JSON file not found: '{path}'.")
            return path

        direct_path = Path(board_id)
        if direct_path.suffix.lower() == ".json" and direct_path.is_file():
            return direct_path

        candidate = cls.BOARDS_DIR / f"{board_id}.json"
        if candidate.is_file():
            return candidate

        raise ValueError(
            f"Unsupported board_id '{board_id}'. Expected 'board_5x5', 'board_10x10', "
            f"or a valid JSON file in '{cls.BOARDS_DIR}'."
        )

    @staticmethod
    def load_board_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Reads and parses a JSON file describing a Kakuro board."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Kakuro JSON root must be an object in '{file_path}'.")
        return data

    @staticmethod
    def _validate_clue_sums(
        r: int, c: int, down_sum: Optional[int], right_sum: Optional[int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Validates that a CLUE cell defines at least one positive integer sum."""
        if down_sum is None and right_sum is None:
            raise ValueError(f"CLUE cell at ({r}, {c}) must specify at least one sum.")
        if down_sum is not None and down_sum <= 0:
            raise ValueError(f"CLUE down_sum at ({r}, {c}) must be positive, got {down_sum}.")
        if right_sum is not None and right_sum <= 0:
            raise ValueError(f"CLUE right_sum at ({r}, {c}) must be positive, got {right_sum}.")
        return down_sum, right_sum

    @classmethod
    def _parse_cell_token(cls, r: int, c: int, token: Any) -> dict:
        """
        Parses and validates a single cell representation from JSON into the Contract 3
        cell dictionary.
        Supported token formats:
        - String shorthand:
          - "#" or "B" or "BLOCK" -> BLOCK cell
          - "W" or "." or "WHITE" -> WHITE variable cell
          - "down/right" (e.g. "14/12", "21/-", "-/16") -> CLUE cell
        - List/Tuple shorthand: ["B"], ["W"], or ["C", down_sum, right_sum]
        - Dictionary format: {"type": "BLOCK"|"CLUE"|"WHITE", "down_sum": ..., "right_sum": ...}
        """
        if isinstance(token, str):
            text = token.strip()
            upper = text.upper()
            if upper in ("#", "B", "BLOCK"):
                return {"r": r, "c": c, "type": "BLOCK"}
            if upper in ("W", ".", "WHITE"):
                return {"r": r, "c": c, "type": "WHITE"}
            if "/" in text:
                down_str, right_str = [part.strip() for part in text.split("/", 1)]
                down_sum = None if down_str in ("", "-", "None", "null") else int(down_str)
                right_sum = None if right_str in ("", "-", "None", "null") else int(right_str)
                down_sum, right_sum = cls._validate_clue_sums(r, c, down_sum, right_sum)
                return {
                    "r": r,
                    "c": c,
                    "type": "CLUE",
                    "down_sum": down_sum,
                    "right_sum": right_sum,
                }
            raise ValueError(f"Unrecognized cell token '{token}' at ({r}, {c}).")

        if isinstance(token, (list, tuple)) and len(token) >= 1:
            kind = str(token[0]).upper()
            if kind in ("B", "BLOCK"):
                return {"r": r, "c": c, "type": "BLOCK"}
            if kind in ("W", "WHITE"):
                return {"r": r, "c": c, "type": "WHITE"}
            if kind in ("C", "CLUE") and len(token) == 3:
                down_sum = int(token[1]) if token[1] is not None else None
                right_sum = int(token[2]) if token[2] is not None else None
                down_sum, right_sum = cls._validate_clue_sums(r, c, down_sum, right_sum)
                return {
                    "r": r,
                    "c": c,
                    "type": "CLUE",
                    "down_sum": down_sum,
                    "right_sum": right_sum,
                }

        if isinstance(token, dict):
            cell_type = str(token.get("type", "")).upper()
            if cell_type == "BLOCK":
                return {"r": r, "c": c, "type": "BLOCK"}
            if cell_type == "WHITE":
                return {"r": r, "c": c, "type": "WHITE"}
            if cell_type == "CLUE":
                down_raw = token.get("down_sum")
                right_raw = token.get("right_sum")
                down_sum = int(down_raw) if down_raw is not None else None
                right_sum = int(right_raw) if right_raw is not None else None
                down_sum, right_sum = cls._validate_clue_sums(r, c, down_sum, right_sum)
                return {
                    "r": r,
                    "c": c,
                    "type": "CLUE",
                    "down_sum": down_sum,
                    "right_sum": right_sum,
                }

        raise ValueError(f"Invalid cell definition at ({r}, {c}): {token!r}")

    @classmethod
    def _parse_board_data(cls, data: Dict[str, Any]) -> Tuple[int, int, List[dict]]:
        """
        Parses and validates a JSON board dictionary supporting either:
        1. A 2D 'grid' array of tokens, or
        2. An explicit 'cells' list with 'rows' and 'cols'.
        """
        if "grid" in data:
            grid = data["grid"]
            if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                raise ValueError("JSON 'grid' must be a non-empty 2D list.")
            rows = int(data.get("rows", len(grid)))
            cols = int(data.get("cols", len(grid[0])))
            if rows <= 0 or cols <= 0:
                raise ValueError("Board dimensions 'rows' and 'cols' must be positive.")
            if len(grid) != rows or any(not isinstance(row, list) or len(row) != cols for row in grid):
                raise ValueError("JSON 'grid' dimensions do not match 'rows' and 'cols'.")
            cells: List[dict] = []
            for r in range(rows):
                for c in range(cols):
                    cells.append(cls._parse_cell_token(r, c, grid[r][c]))
            return rows, cols, cells

        if "cells" in data and "rows" in data and "cols" in data:
            rows = int(data["rows"])
            cols = int(data["cols"])
            if rows <= 0 or cols <= 0:
                raise ValueError("Board dimensions 'rows' and 'cols' must be positive.")
            raw_cells = data["cells"]
            if not isinstance(raw_cells, list):
                raise ValueError("JSON 'cells' must be a list of cell objects.")

            cell_by_coord: Dict[Tuple[int, int], dict] = {}
            for item in raw_cells:
                if not isinstance(item, dict) or "r" not in item or "c" not in item:
                    raise ValueError(f"Each cell in 'cells' must be a dict with 'r' and 'c': {item!r}")
                r = int(item["r"])
                c = int(item["c"])
                if not (0 <= r < rows and 0 <= c < cols):
                    raise ValueError(
                        f"Cell coordinate ({r}, {c}) is outside board dimensions ({rows}x{cols})."
                    )
                if (r, c) in cell_by_coord:
                    raise ValueError(f"Duplicate cell coordinate ({r}, {c}) in 'cells'.")
                cell_by_coord[(r, c)] = cls._parse_cell_token(r, c, item)

            expected_coords = {(r, c) for r in range(rows) for c in range(cols)}
            missing_coords = expected_coords - set(cell_by_coord.keys())
            if missing_coords:
                first_missing = min(missing_coords)
                raise ValueError(
                    f"Missing {len(missing_coords)} cell coordinate(s) in 'cells', e.g. {first_missing}."
                )

            cells = [cell_by_coord[(r, c)] for r in range(rows) for c in range(cols)]
            return rows, cols, cells

        raise ValueError("Kakuro board JSON must contain either 'grid' or ('rows', 'cols', 'cells').")

    def _extract_constraints(self) -> None:
        """
        Scans all CLUE cells on the board and builds horizontal ('right') and
        vertical ('down') KakuroConstraint objects over contiguous WHITE cells.
        """
        cell_map = {(cell["r"], cell["c"]): cell for cell in self.cells}

        for cell in self.cells:
            if cell["type"] != "CLUE":
                continue
            r, c = cell["r"], cell["c"]

            # Horizontal run to the right
            if cell.get("right_sum") is not None:
                run_cells = []
                cc = c + 1
                while cc < self.cols and cell_map[(r, cc)]["type"] == "WHITE":
                    run_cells.append((r, cc))
                    cc += 1
                if run_cells:
                    constraint = KakuroConstraint(cell["right_sum"], run_cells, "right")
                    self.constraints.append(constraint)
                    for var in run_cells:
                        self.cell_constraints[var].append(constraint)
                        self.peers[var].update(other for other in run_cells if other != var)

            # Vertical run downwards
            if cell.get("down_sum") is not None:
                run_cells = []
                rr = r + 1
                while rr < self.rows and cell_map[(rr, c)]["type"] == "WHITE":
                    run_cells.append((rr, c))
                    rr += 1
                if run_cells:
                    constraint = KakuroConstraint(cell["down_sum"], run_cells, "down")
                    self.constraints.append(constraint)
                    for var in run_cells:
                        self.cell_constraints[var].append(constraint)
                        self.peers[var].update(other for other in run_cells if other != var)

    def get_board_layout(self) -> dict:
        """
        Called once by Nh's UI to draw the static Kakuro grid.
        Returns:
        {
            "rows": int,
            "cols": int,
            "cells": [
                {"r": int, "c": int, "type": "BLOCK"},
                {"r": int, "c": int, "type": "CLUE", "down_sum": int|None, "right_sum": int|None},
                {"r": int, "c": int, "type": "WHITE"}
            ]
        }
        """
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cells": [dict(cell) for cell in self.cells],
        }

    # =========================================================================
    # Task 3A: Constraint Propagation Module (Forward Checking & AC-3)
    # =========================================================================

    @staticmethod
    def _has_valid_run_completion(
        unassigned_domains: List[List[int]],
        target_remaining: int,
        used_values: Set[int],
        idx: int = 0,
    ) -> bool:
        """
        Checks whether there exists a valid assignment of distinct digits to the
        remaining unassigned cells in a run that sums to 'target_remaining'.
        """
        remaining_count = len(unassigned_domains) - idx
        if remaining_count == 0:
            return target_remaining == 0

        # Fast arithmetic bounding using minimum and maximum possible sums of k distinct digits in 1..9
        min_possible = remaining_count * (remaining_count + 1) // 2
        max_possible = remaining_count * (19 - remaining_count) // 2
        if target_remaining < min_possible or target_remaining > max_possible:
            return False

        for val in unassigned_domains[idx]:
            if val in used_values or val > target_remaining:
                continue
            used_values.add(val)
            if KakuroCSP._has_valid_run_completion(
                unassigned_domains, target_remaining - val, used_values, idx + 1
            ):
                used_values.remove(val)
                return True
            used_values.remove(val)

        return False

    def is_value_supported_in_constraint(
        self,
        constraint: KakuroConstraint,
        target_cell: Tuple[int, int],
        value: int,
        assignments: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], List[int]],
    ) -> bool:
        """
        Checks whether assigning 'target_cell = value' is supported by at least one
        valid tuple of distinct values from the current domains of the other cells
        in 'constraint'. Used by AC-3 to enforce full run-level arc consistency.
        """
        used_values: Set[int] = {value}
        assigned_sum = value
        other_domains: List[List[int]] = []

        for cell in constraint.cells:
            if cell == target_cell:
                continue
            if cell in assignments:
                assigned_val = assignments[cell]
                if assigned_val in used_values:
                    return False
                used_values.add(assigned_val)
                assigned_sum += assigned_val
            else:
                other_domains.append(domains[cell])

        target_remaining = constraint.target_sum - assigned_sum
        if target_remaining < 0:
            return False

        # Sort smaller domains first without mutating cell domains in-place
        other_domains = sorted([list(d) for d in other_domains], key=len)
        return self._has_valid_run_completion(
            other_domains, target_remaining, set(used_values)
        )

    def propagate_ac3(
        self,
        assignments: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], List[int]],
        initial_constraints: Optional[List[KakuroConstraint]] = None,
    ) -> Tuple[bool, List[Tuple[int, int, List[int]]]]:
        """
        Generalized AC-3 (Arc Consistency) propagation across Kakuro Sum + AllDiff constraints.
        Uses a FIFO deque to iteratively prune unsupported values from unassigned variable
        domains until a fixed point is reached or a domain wipeout occurs.

        Returns:
            (consistent, pruned_cells)
            - consistent: False if any variable's domain becomes empty (wipeout), True otherwise.
            - pruned_cells: list of (r, c, removed_values) for UI visualization.
        """
        queue: Deque[KakuroConstraint] = deque(
            initial_constraints if initial_constraints is not None else self.constraints
        )
        in_queue: Set[KakuroConstraint] = set(queue)
        pruned_map: Dict[Tuple[int, int], List[int]] = {}

        while queue:
            constraint = queue.popleft()
            in_queue.discard(constraint)

            for cell in constraint.cells:
                if cell in assignments:
                    continue

                current_dom = domains[cell]
                removed: List[int] = []
                new_dom: List[int] = []

                for val in current_dom:
                    if self.is_value_supported_in_constraint(
                        constraint, cell, val, assignments, domains
                    ):
                        new_dom.append(val)
                    else:
                        removed.append(val)

                if removed:
                    domains[cell] = new_dom
                    pruned_map.setdefault(cell, []).extend(removed)

                    # Domain wipeout detected -> inconsistency
                    if not new_dom:
                        pruned_list = [
                            (r, c, sorted(set(vals))) for (r, c), vals in pruned_map.items()
                        ]
                        return False, pruned_list

                    # Enqueue all constraints involving 'cell' to propagate further
                    for neighbor_constraint in self.cell_constraints[cell]:
                        if neighbor_constraint not in in_queue:
                            queue.append(neighbor_constraint)
                            in_queue.add(neighbor_constraint)

        pruned_list = [(r, c, sorted(set(vals))) for (r, c), vals in pruned_map.items()]
        return True, pruned_list

    def forward_check(
        self,
        focus_cell: Tuple[int, int],
        assigned_value: int,
        assignments: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], List[int]],
    ) -> Tuple[bool, List[Tuple[int, int, List[int]]]]:
        """
        Performs lightweight Forward Checking (FC) on all unassigned peers sharing a
        horizontal or vertical run with 'focus_cell' after assigning 'assigned_value':
        1. AllDiff pruning: removes 'assigned_value' (and any already-assigned run values).
        2. Direct arithmetic sum-bound pruning: removes peer values that cannot possibly
           fit the remaining run sum given the number of unassigned cells left in the run.
        Full multi-cell domain tuple checks are left to AC-3 (`propagate_ac3`).

        Returns:
            (consistent, pruned_cells)
            - consistent: False if any neighbor's domain is wiped out, True otherwise.
            - pruned_cells: list of (r, c, removed_values) for UI visualization.
        """
        pruned_map: Dict[Tuple[int, int], List[int]] = {}

        for constraint in self.cell_constraints[focus_cell]:
            used_values: Set[int] = {assigned_value}
            assigned_sum = assigned_value
            unassigned_peers: List[Tuple[int, int]] = []

            for cell in constraint.cells:
                if cell == focus_cell:
                    continue
                if cell in assignments:
                    val_c = assignments[cell]
                    used_values.add(val_c)
                    assigned_sum += val_c
                else:
                    unassigned_peers.append(cell)

            remaining_sum = constraint.target_sum - assigned_sum
            other_count = len(unassigned_peers) - 1
            min_other = other_count * (other_count + 1) // 2
            max_other = other_count * (19 - other_count) // 2

            for peer in unassigned_peers:
                current_dom = domains[peer]
                removed: List[int] = []
                new_dom: List[int] = []

                for val in current_dom:
                    after_val = remaining_sum - val
                    if val not in used_values and min_other <= after_val <= max_other:
                        new_dom.append(val)
                    else:
                        removed.append(val)

                if removed:
                    domains[peer] = new_dom
                    pruned_map.setdefault(peer, []).extend(removed)
                    if not new_dom:
                        pruned_list = [
                            (r, c, sorted(set(vals))) for (r, c), vals in pruned_map.items()
                        ]
                        return False, pruned_list

        pruned_list = [(r, c, sorted(set(vals))) for (r, c), vals in pruned_map.items()]
        return True, pruned_list

    def is_valid_solution(self, assignments: Dict[Tuple[int, int], int]) -> bool:
        """
        Verifies that all WHITE cells are assigned a digit in 1..9 and every
        horizontal and vertical KakuroConstraint has distinct digits summing to target_sum.
        """
        if set(assignments.keys()) != set(self.variables):
            return False
        if any(not (1 <= val <= 9) for val in assignments.values()):
            return False

        for constraint in self.constraints:
            vals = [assignments[cell] for cell in constraint.cells]
            if len(set(vals)) != len(vals):
                return False
            if sum(vals) != constraint.target_sum:
                return False

        return True

    # =========================================================================
    # Task 3B Integration Hook: Step-by-Step Solver Generator (Contract 3)
    # =========================================================================

    def _select_unassigned_variable(
        self,
        assignments: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], List[int]],
    ) -> Tuple[int, int]:
        """
        Selects the next unassigned variable using Lecture 08 heuristics:
        1. Minimum Remaining Values (MRV): smallest domain size first.
        2. Degree Heuristic (tie-breaker): most unassigned peers first.
        """
        unassigned = [v for v in self.variables if v not in assignments]
        return min(
            unassigned,
            key=lambda var: (
                len(domains[var]),
                -sum(1 for peer in self.peers[var] if peer not in assignments),
                var,
            ),
        )

    def _order_domain_values(
        self,
        var: Tuple[int, int],
        assignments: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], List[int]],
    ) -> List[int]:
        """
        Orders domain values for 'var' using the Least Constraining Value (LCV) heuristic:
        prefers values that rule out the fewest values in neighboring unassigned cells.
        """
        if len(domains[var]) <= 1:
            return list(domains[var])

        def conflict_impact(val: int) -> int:
            eliminated = 0
            for peer in self.peers[var]:
                if peer not in assignments and val in domains[peer]:
                    eliminated += 1
            return eliminated

        return sorted(domains[var], key=lambda v: (conflict_impact(v), v))

    @staticmethod
    def _merge_pruned_lists(
        first: List[Tuple[int, int, List[int]]],
        second: List[Tuple[int, int, List[int]]],
    ) -> List[Tuple[int, int, List[int]]]:
        """Merges two pruned_cells lists from Forward Checking and AC-3 without duplicates."""
        merged: Dict[Tuple[int, int], Set[int]] = {}
        order: List[Tuple[int, int]] = []
        for source in (first, second):
            for r, c, vals in source:
                key = (r, c)
                if key not in merged:
                    merged[key] = set()
                    order.append(key)
                merged[key].update(vals)
        return [(r, c, sorted(merged[(r, c)])) for r, c in order]

    def solve_step_by_step(self) -> Iterator[dict]:
        """
        Generator yielding a state event every time the solver acts.
        Nh's UI calls `next(generator)` on a timer (Play) or button click (Step).
        """
        start_time = time.perf_counter()
        assignments: Dict[Tuple[int, int], int] = {}
        domains: Dict[Tuple[int, int], List[int]] = {
            var: list(self.domains[var]) for var in self.variables
        }
        stats = {"steps": 0, "backtracks": 0, "prunings": 0, "elapsed_ms": 0.0}

        def make_event(
            event_type: str,
            focus_cell: Tuple[int, int],
            assigned_value: Optional[int] = None,
            pruned_cells: Optional[List[Tuple[int, int, List[int]]]] = None,
        ) -> dict:
            stats["elapsed_ms"] = round((time.perf_counter() - start_time) * 1000.0, 4)
            return {
                "event_type": event_type,
                "focus_cell": focus_cell,
                "assigned_value": assigned_value,
                "pruned_cells": list(pruned_cells) if pruned_cells else [],
                "assignments": dict(assignments),
                "domains": {
                    var: list(domains[var])
                    for var in self.variables
                    if var not in assignments
                },
                "stats": dict(stats),
            }

        # Step 1: Initial AC-3 Constraint Propagation before search
        consistent, initial_pruned = self.propagate_ac3(assignments, domains)
        if initial_pruned:
            stats["steps"] += 1
            stats["prunings"] += sum(len(vals) for _, _, vals in initial_pruned)
            first_focus = (initial_pruned[0][0], initial_pruned[0][1])
            yield make_event("PROPAGATE", first_focus, None, initial_pruned)

        if not consistent:
            stats["steps"] += 1
            yield make_event("CONFLICT", self.variables[0], None, [])
            return

        # Step 2: Backtracking Search with Forward Checking (FC) followed by AC-3 (MAC)
        def backtrack() -> Iterator[Tuple[bool, dict]]:
            if len(assignments) == len(self.variables):
                stats["steps"] += 1
                last_cell = next(reversed(assignments)) if assignments else self.variables[0]
                yield True, make_event("SOLVED", last_cell, assignments.get(last_cell), [])
                return

            var = self._select_unassigned_variable(assignments, domains)
            stats["steps"] += 1
            yield False, make_event("SELECT_VAR", var, None, [])

            for val in self._order_domain_values(var, assignments, domains):
                saved_domains = {v: list(domains[v]) for v in self.variables}
                assignments[var] = val
                domains[var] = [val]

                stats["steps"] += 1
                yield False, make_event("ASSIGN", var, val, [])

                # Phase A: Lightweight Forward Checking on immediate peers in var's runs
                ok, fc_pruned = self.forward_check(var, val, assignments, domains)
                pruned = list(fc_pruned)

                # Phase B: Full AC-3 run-support propagation starting from var's runs
                # and any intersecting runs of peers narrowed by Forward Checking
                if ok:
                    ac3_seeds: List[KakuroConstraint] = list(self.cell_constraints[var])
                    for r_p, c_p, _ in fc_pruned:
                        for neighbor_c in self.cell_constraints[(r_p, c_p)]:
                            if neighbor_c not in ac3_seeds:
                                ac3_seeds.append(neighbor_c)
                    ok, ac3_pruned = self.propagate_ac3(
                        assignments, domains, initial_constraints=ac3_seeds
                    )
                    pruned = self._merge_pruned_lists(fc_pruned, ac3_pruned)

                if pruned:
                    stats["steps"] += 1
                    stats["prunings"] += sum(len(vals) for _, _, vals in pruned)
                    yield False, make_event("PROPAGATE", var, val, pruned)

                if ok:
                    solved_flag = False
                    for is_solved, ev in backtrack():
                        yield is_solved, ev
                        if is_solved:
                            solved_flag = True
                            break
                    if solved_flag:
                        return
                else:
                    stats["steps"] += 1
                    yield False, make_event("CONFLICT", var, val, pruned)

                # Undo assignment and restore domains on backtrack
                del assignments[var]
                for v in self.variables:
                    domains[v] = list(saved_domains[v])
                stats["steps"] += 1
                stats["backtracks"] += 1
                yield False, make_event("BACKTRACK", var, None, [])

        for is_solved, event in backtrack():
            yield event
            if is_solved:
                return
