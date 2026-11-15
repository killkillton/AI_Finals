"""
Adversarial Search Agents for Battle City Tank Combat.

Formal Alignment with Course Lecture 07 (AI-lec07-Adversarial Search.pptx):
- Slides 4-6: Zero-Sum Game formulation:
  S_0 (Initial State), PLAYER(s), ACTIONS(s), RESULT(s, a),
  TERMINAL-TEST(s), and UTILITY(s, p).
- Slides 9-14: Minimax algorithm (`MINIMAX-DECISION`, `MAX-VALUE`, `MIN-VALUE`),
  time complexity O(b^m), space complexity O(bm), and bounded lookahead motivation.
- Slides 15-26: Alpha-Beta Pruning (`ALPHA-BETA-SEARCH`) and Slide 25 properties:
  * "Pruning does not affect the result"
  * "Good move ordering improves effectiveness of pruning"
  * "Killer move heuristic: explore the best move first"
  * "Transposition table avoids re-evaluation a state"
- Slides 27-29: Heuristic Minimax (`H-MINIMAX(s, d)`), replacing `TERMINAL-TEST`
  with `CUTOFF-TEST(s, d)` and `UTILITY` with `EVAL(s)`.

Implemented Agents:
1. `MinimaxAgent`: Bounded-depth Minimax search using basic material payoff (`evaluate_basic`).
2. `AlphaBetaAgent`: Bounded-depth Alpha-Beta Pruning using basic material payoff (`evaluate_basic`).
3. `HeuristicMinimaxAgent`: `H-MINIMAX(s, d)` using full tactical evaluation (`evaluate_tactical`).
4. `HeuristicAlphaBetaAgent`: `H-MINIMAX(s, d)` with Alpha-Beta pruning, move ordering,
   killer-move heuristic, and Transposition Table caching (`evaluate_tactical`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from Task2.heuristics import (
    evaluate_basic,
    evaluate_tactical,
    evaluate_terminal,
    order_actions,
)
from Task2.tank_game_engine import (
    CANONICAL_ACTION_RANK,
    DIR_DELTAS,
    GRID_SIZE,
    Action,
    Direction,
    TankGameEngine,
    canonical_action_rank,
)

# Transposition Table bound flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2


def _subtree_node_estimate(branching_factor: int, rem_depth: int) -> int:
    """
    Return the number of subtree game states eliminated when a branch at
    remaining `rem_depth` is pruned in a tree with local `branching_factor`.
    """
    if rem_depth <= 1:
        return 1
    b = max(2, branching_factor)
    total = 0
    power = 1
    for _ in range(rem_depth):
        total += power
        power *= b
    return total


class TranspositionTable:
    """
    State Transposition Table for Alpha-Beta Search (Lecture 07 Slide 25:
    "Transposition table avoids re-evaluation a state").
    Caches evaluated game states keyed by `engine.state_key(turn_player, pending_action)`.
    """

    def __init__(self, max_entries: int = 100000) -> None:
        self.max_entries = max_entries
        self._table: Dict[Tuple, Tuple[int, int, float, Optional[Action]]] = {}
        self.hits: int = 0

    def clear(self) -> None:
        self._table.clear()
        self.hits = 0

    def lookup(
        self,
        key: Tuple,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[Optional[float], Optional[Action]]:
        """
        Look up a cached state evaluation.
        Increments `self.hits` ONLY when returning a valid cached utility/heuristic
        value (`val`) for the requested remaining `depth` that satisfies the
        exact/alpha/beta bound conditions.
        """
        entry = self._table.get(key)
        if entry is None:
            return None, None
        stored_depth, flag, val, best_act = entry
        if stored_depth == depth and flag == TT_EXACT:
            self.hits += 1
            return val, best_act
        return None, best_act

    def store(
        self,
        key: Tuple,
        depth: int,
        flag: int,
        val: float,
        best_action: Optional[Action],
    ) -> None:
        if len(self._table) >= self.max_entries and key not in self._table:
            self._table.clear()
        existing = self._table.get(key)
        if (
            existing is None
            or existing[0] < depth
            or (existing[0] == depth and (flag == TT_EXACT or existing[1] != TT_EXACT))
        ):
            self._table[key] = (depth, flag, val, best_action)


def _apply_joint_step(
    engine: TankGameEngine,
    max_player: int,
    max_action: Action,
    min_action: Action,
) -> TankGameEngine:
    """
    Advance `engine` by one simultaneous game cycle given `max_action` for
    `max_player` and `min_action` for `3 - max_player`.
    """
    if max_player == 1:
        return engine.simulate_action(max_action, min_action)
    return engine.simulate_action(min_action, max_action)


def _apply_tactical_half_ply(
    engine: TankGameEngine,
    max_player: int,
    max_action: Action,
) -> TankGameEngine:
    """
    Resolve an odd final search ply (`depth == 1`) by pairing `max_action` with
    a deterministic reactive opponent policy (evasive dodge if under immediate
    bullet threat, return fire if aligned on an unobstructed firing line, or
    `Action.STAY` otherwise) and advancing one physics cycle.
    """
    opp_id = 3 - max_player
    t_max = engine.tanks[max_player]
    t_min = engine.tanks[opp_id]
    min_action = Action.STAY

    if t_min["alive"]:
        tx, ty = t_min["x"], t_min["y"]
        dodged = False
        v_candidates = (
            ((Action.MOVE_UP, Direction.UP), (Action.MOVE_DOWN, Direction.DOWN))
            if opp_id == 1
            else ((Action.MOVE_DOWN, Direction.DOWN), (Action.MOVE_UP, Direction.UP))
        )
        h_candidates = (
            ((Action.MOVE_RIGHT, Direction.RIGHT), (Action.MOVE_LEFT, Direction.LEFT))
            if opp_id == 1
            else ((Action.MOVE_LEFT, Direction.LEFT), (Action.MOVE_RIGHT, Direction.RIGHT))
        )
        for b in engine.bullets:
            if b["owner"] == max_player:
                bdx, bdy = DIR_DELTAS[b["dir"]]
                if bdx != 0 and b["y"] == ty and 0 < (tx - b["x"]) * bdx <= 3:
                    for cand, d in v_candidates:
                        dx, dy = DIR_DELTAS[d]
                        nx, ny = tx + dx, ty + dy
                        if (
                            0 <= nx < GRID_SIZE
                            and 0 <= ny < GRID_SIZE
                            and (nx, ny) not in engine.walls
                        ):
                            min_action = cand
                            dodged = True
                            break
                    break
                if bdy != 0 and b["x"] == tx and 0 < (ty - b["y"]) * bdy <= 3:
                    for cand, d in h_candidates:
                        dx, dy = DIR_DELTAS[d]
                        nx, ny = tx + dx, ty + dy
                        if (
                            0 <= nx < GRID_SIZE
                            and 0 <= ny < GRID_SIZE
                            and (nx, ny) not in engine.walls
                        ):
                            min_action = cand
                            dodged = True
                            break
                    break

        # If MIN does not need to dodge and has an unobstructed shot facing MAX, return fire
        if not dodged and t_max["alive"]:
            mx, my = t_max["x"], t_max["y"]
            if tx == mx and ty != my:
                step = 1 if my > ty else -1
                req_dir = Direction.DOWN if step > 0 else Direction.UP
                if t_min["dir"] == req_dir and all(
                    (tx, y) not in engine.walls for y in range(ty + step, my, step)
                ):
                    min_action = Action.FIRE
            elif ty == my and tx != mx:
                step = 1 if mx > tx else -1
                req_dir = Direction.RIGHT if step > 0 else Direction.LEFT
                if t_min["dir"] == req_dir and all(
                    (x, ty) not in engine.walls for x in range(tx + step, mx, step)
                ):
                    min_action = Action.FIRE

    return _apply_joint_step(engine, max_player, max_action, min_action)


class BaseTankAgent(ABC):
    """
    Abstract base class for all Tank Battle adversarial search agents.
    """

    ALGORITHM_NAME: str = "Base Tank Agent"

    def __init__(
        self,
        player_id: int,
        max_depth: int = 3,
        name: Optional[str] = None,
    ) -> None:
        if player_id not in (1, 2):
            raise ValueError(f"player_id must be 1 or 2, got {player_id}")
        self.player_id: int = player_id
        self.opponent_id: int = 3 - player_id
        self.max_depth: int = max(1, int(max_depth))
        self.name: str = name or self.ALGORITHM_NAME

        # Performance telemetry per decision
        self.explored_states: int = 0
        self.pruned_branches: int = 0
        self.tt_hits: int = 0
        self.decision_time_ms: float = 0.0
        self.root_best_value: float = 0.0

        # Anti-oscillation and anti-stagnation memory across live game turns
        self._recent_positions: List[Tuple[int, int]] = []
        self._recent_hp_sigs: List[Tuple[int, int, int, int, int, int]] = []

    def reset_metrics(self) -> None:
        self.explored_states = 0
        self.pruned_branches = 0
        self.tt_hits = 0
        self.decision_time_ms = 0.0
        self.root_best_value = 0.0

    def _board_hp_signature(
        self, engine: TankGameEngine
    ) -> Tuple[int, int, int, int, int, int]:
        return (
            engine.tanks[1]["hp"],
            engine.tanks[2]["hp"],
            engine.bases[1]["hp"],
            engine.bases[2]["hp"],
            len(engine.walls),
            sum(engine.walls.values()),
        )

    def _has_valid_fire_target(self, engine: TankGameEngine) -> bool:
        """
        Return True if firing along `self.player_id`'s current facing direction
        will reach an enemy tank, enemy base, or non-friendly-base wall within
        effective range (<= 6 cells for walls, <= 12 cells for enemy tank/base).
        """
        t = engine.tanks[self.player_id]
        if not t["alive"]:
            return False
        opp_t = engine.tanks[self.opponent_id]
        my_b = engine.bases[self.player_id]
        opp_b = engine.bases[self.opponent_id]

        tx, ty = t["x"], t["y"]
        fdx, fdy = DIR_DELTAS[t["dir"]]
        cx, cy = tx, ty
        for step in range(1, 13):
            cx += fdx
            cy += fdy
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                return False
            if (cx, cy) == (my_b["x"], my_b["y"]):
                return False
            if (cx, cy) in engine.walls:
                if abs(cx - my_b["x"]) + abs(cy - my_b["y"]) <= 2:
                    return False
                if abs(cx - opp_b["x"]) + abs(cy - opp_b["y"]) <= 2:
                    return step <= 6
                if (
                    cx == opp_t["x"]
                    or cy == opp_t["y"]
                    or cx == opp_b["x"]
                    or cy == opp_b["y"]
                ):
                    return step <= 4
                return False
            if opp_t["alive"] and (cx, cy) == (opp_t["x"], opp_t["y"]):
                return True
            if opp_b["hp"] > 0 and (cx, cy) == (opp_b["x"], opp_b["y"]):
                return True
        return False

    def _record_position_and_get_penalty(
        self,
        engine: TankGameEngine,
        action: Action,
    ) -> float:
        """
        Compute live-play anti-oscillation and anti-stagnation penalty so tanks
        do not oscillate between two cells or sit in place firing indefinitely
        when no HP damage is being dealt. Returns 0.0 on a fresh agent before
        multi-turn history accumulates so single-state search optimality is
        unaffected.
        """
        if not self._recent_positions:
            return 0.0

        t = engine.tanks[self.player_id]
        opp_tank = engine.tanks[self.opponent_id]
        my_base = engine.bases[self.player_id]
        tx, ty = t["x"], t["y"]
        dir_map = {
            Action.MOVE_UP: Direction.UP,
            Action.MOVE_DOWN: Direction.DOWN,
            Action.MOVE_LEFT: Direction.LEFT,
            Action.MOVE_RIGHT: Direction.RIGHT,
        }

        if action in dir_map:
            dx, dy = DIR_DELTAS[dir_map[action]]
            nx, ny = tx + dx, ty + dy
            if not (
                0 <= nx < GRID_SIZE
                and 0 <= ny < GRID_SIZE
                and (nx, ny) not in engine.walls
            ):
                nx, ny = tx, ty
        else:
            nx, ny = tx, ty

        repeats = self._recent_positions.count((nx, ny))
        cur_sig = self._board_hp_signature(engine)

        # Count consecutive recent turns spent on (tx, ty) without any HP change
        stagnant_repeats = 0
        for pos, sig in zip(
            reversed(self._recent_positions), reversed(self._recent_hp_sigs)
        ):
            if pos == (tx, ty) and sig == cur_sig:
                stagnant_repeats += 1
            else:
                break

        penalty = 0.0

        # Penalize repeatedly attempting to step into a cell contested by the enemy tank
        if (
            action in dir_map
            and stagnant_repeats >= 2
            and opp_tank["alive"]
            and abs(nx - opp_tank["x"]) + abs(ny - opp_tank["y"]) <= 1
        ):
            penalty -= 95.0 * stagnant_repeats

        # Never allow firing at friendly base or friendly base shield walls in live play
        close_enemy_in_front = False
        if action == Action.FIRE:
            fdx, fdy = DIR_DELTAS[t["dir"]]
            cx, cy = tx, ty
            for step in range(1, 10):
                cx += fdx
                cy += fdy
                if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                    break
                if (cx, cy) in engine.walls:
                    if abs(cx - my_base["x"]) + abs(cy - my_base["y"]) <= 2:
                        penalty -= 250.0
                    break
                if (cx, cy) == (my_base["x"], my_base["y"]):
                    penalty -= 250.0
                    break
                if (
                    opp_tank["alive"]
                    and (cx, cy) == (opp_tank["x"], opp_tank["y"])
                    and step <= 2
                ):
                    close_enemy_in_front = True
                    break

        if action == Action.FIRE and close_enemy_in_front:
            return penalty

        valid_target = self._has_valid_fire_target(engine)
        if action == Action.FIRE and valid_target and stagnant_repeats < 5:
            return penalty

        if stagnant_repeats >= 2 and action in (Action.STAY, Action.FIRE):
            penalty -= 120.0 * stagnant_repeats
        elif action == Action.STAY and repeats >= 2:
            penalty -= 25.0 * repeats
        elif action != Action.FIRE and repeats >= 2:
            penalty -= 12.0 * (repeats - 1)

        return penalty

    def update_position_memory(self, engine: TankGameEngine) -> None:
        t = engine.tanks[self.player_id]
        self._recent_positions.append((t["x"], t["y"]))
        self._recent_hp_sigs.append(self._board_hp_signature(engine))
        if len(self._recent_positions) > 8:
            self._recent_positions.pop(0)
        if len(self._recent_hp_sigs) > 8:
            self._recent_hp_sigs.pop(0)

    def _update_history(self, engine: TankGameEngine) -> None:
        self.update_position_memory(engine)

    def _build_metrics(self, best_score: float) -> Dict[str, Any]:
        total_branches = self.explored_states + self.pruned_branches
        pruning_rate = (
            round(100.0 * self.pruned_branches / total_branches, 2)
            if total_branches > 0
            else 0.0
        )
        return {
            "algorithm": self.ALGORITHM_NAME,
            "time_ms": round(self.decision_time_ms, 4),
            "explored_states": self.explored_states,
            "depth_reached": self.max_depth,
            "pruned_branches": self.pruned_branches,
            "pruning_rate": pruning_rate,
            "tt_hits": self.tt_hits,
            "best_score": round(best_score, 4),
        }

    @abstractmethod
    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        """Select and return `(best_action, metrics_dict)` for `self.player_id`."""


class MinimaxAgent(BaseTankAgent):
    """
    Standard Bounded-Depth Minimax Agent (Lecture 07 Slides 9-14:
    `MINIMAX-DECISION`, `MAX-VALUE`, `MIN-VALUE`).
    Evaluates terminal states with exact `UTILITY(s, p)` and bounded-depth leaf
    states with basic material payoff (`evaluate_basic`).
    """

    ALGORITHM_NAME = "Minimax Search"

    def __init__(self, player_id: int, max_depth: int = 3) -> None:
        super().__init__(
            player_id=player_id,
            max_depth=max_depth,
            name=self.ALGORITHM_NAME,
        )

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        best_action = legal_actions[0]
        best_score = -math.inf

        for act in legal_actions:
            if self.max_depth <= 1:
                child = _apply_tactical_half_ply(engine_copy, self.player_id, act)
                raw_score = self._minimax(child, 0, False, None)
            else:
                raw_score = self._minimax(engine_copy, self.max_depth - 1, False, act)
            score = raw_score + self._record_position_and_get_penalty(engine_copy, act)
            if score > best_score + 1e-6:
                best_score = score
                best_action = act
            elif abs(score - best_score) <= 1e-6:
                if canonical_action_rank(self.player_id, act) > canonical_action_rank(
                    self.player_id, best_action
                ):
                    best_action = act

        self.update_position_memory(engine_copy)
        self.root_best_value = best_score
        self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_action, self._build_metrics(best_score)

    def _minimax(
        self,
        engine: TankGameEngine,
        depth: int,
        is_max_turn: bool,
        pending_max_action: Optional[Action],
    ) -> float:
        self.explored_states += 1

        if engine.is_terminal():
            return evaluate_terminal(engine, self.player_id, depth)
        if depth <= 0:
            return evaluate_basic(engine, self.player_id, 0)

        if is_max_turn:
            v = -math.inf
            actions = engine.get_legal_actions(self.player_id)
            if depth == 1:
                for act in actions:
                    child = _apply_tactical_half_ply(engine, self.player_id, act)
                    val = self._minimax(child, 0, False, None)
                    if val > v:
                        v = val
                return v

            for act in actions:
                val = self._minimax(engine, depth - 1, False, act)
                if val > v:
                    v = val
            return v
        else:
            v = math.inf
            actions = engine.get_legal_actions(self.opponent_id)
            max_act = (
                pending_max_action if pending_max_action is not None else Action.STAY
            )
            for min_act in actions:
                child = _apply_joint_step(engine, self.player_id, max_act, min_act)
                val = self._minimax(child, depth - 1, True, None)
                if val < v:
                    v = val
            return v


class AlphaBetaAgent(BaseTankAgent):
    """
    Standard Bounded-Depth Alpha-Beta Pruning Agent (Lecture 07 Slides 15-26:
    `ALPHA-BETA-SEARCH`).
    Uses the same `evaluate_basic` payoff function as `MinimaxAgent` so that
    pruning preserves the exact Minimax decision while reducing explored states.
    """

    ALGORITHM_NAME = "Alpha-Beta Search"

    def __init__(
        self,
        player_id: int,
        max_depth: int = 3,
        use_move_ordering: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            max_depth=max_depth,
            name=self.ALGORITHM_NAME,
        )
        self.use_move_ordering = use_move_ordering

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        ordered = (
            order_actions(engine_copy, self.player_id, legal_actions)
            if self.use_move_ordering
            else legal_actions
        )

        best_action = ordered[0]
        best_score = -math.inf
        beta = math.inf

        for act in ordered:
            pen = self._record_position_and_get_penalty(engine_copy, act)
            root_alpha = best_score - pen - 1e-5
            if self.max_depth <= 1:
                child = _apply_tactical_half_ply(engine_copy, self.player_id, act)
                raw_score = self._alphabeta(child, 0, root_alpha, beta, False, None)
            else:
                raw_score = self._alphabeta(
                    engine_copy, self.max_depth - 1, root_alpha, beta, False, act
                )
            score = raw_score + pen
            if score > best_score + 1e-6:
                best_score = score
                best_action = act
            elif abs(score - best_score) <= 1e-6:
                if canonical_action_rank(self.player_id, act) > canonical_action_rank(
                    self.player_id, best_action
                ):
                    best_action = act

        self.update_position_memory(engine_copy)
        self.root_best_value = best_score
        self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_action, self._build_metrics(best_score)

    def _alphabeta(
        self,
        engine: TankGameEngine,
        depth: int,
        alpha: float,
        beta: float,
        is_max_turn: bool,
        pending_max_action: Optional[Action],
    ) -> float:
        self.explored_states += 1

        if engine.is_terminal():
            return evaluate_terminal(engine, self.player_id, depth)
        if depth <= 0:
            return evaluate_basic(engine, self.player_id, 0)

        if is_max_turn:
            v = -math.inf
            raw_actions = engine.get_legal_actions(self.player_id)
            actions = (
                order_actions(engine, self.player_id, raw_actions)
                if self.use_move_ordering
                else raw_actions
            )
            n_act = len(actions)

            if depth == 1:
                for idx, act in enumerate(actions):
                    child = _apply_tactical_half_ply(engine, self.player_id, act)
                    val = self._alphabeta(child, 0, alpha, beta, False, None)
                    if val > v:
                        v = val
                    if v >= beta:
                        skipped = max(1, n_act - (idx + 1))
                        self.pruned_branches += skipped
                        return v
                    if v > alpha:
                        alpha = v
                return v

            for idx, act in enumerate(actions):
                val = self._alphabeta(engine, depth - 1, alpha, beta, False, act)
                if val > v:
                    v = val
                if v >= beta:
                    skipped = max(1, n_act - (idx + 1))
                    self.pruned_branches += skipped * _subtree_node_estimate(
                        n_act, depth
                    )
                    return v
                if v > alpha:
                    alpha = v
            return v
        else:
            v = math.inf
            raw_actions = engine.get_legal_actions(self.opponent_id)
            actions = (
                order_actions(engine, self.opponent_id, raw_actions)
                if self.use_move_ordering
                else raw_actions
            )
            n_act = len(actions)
            max_act = (
                pending_max_action if pending_max_action is not None else Action.STAY
            )

            for idx, min_act in enumerate(actions):
                child = _apply_joint_step(engine, self.player_id, max_act, min_act)
                val = self._alphabeta(child, depth - 1, alpha, beta, True, None)
                if val < v:
                    v = val
                if v <= alpha:
                    skipped = max(1, n_act - (idx + 1))
                    self.pruned_branches += skipped * _subtree_node_estimate(
                        n_act, depth
                    )
                    return v
                if v < beta:
                    beta = v
            return v


class HeuristicMinimaxAgent(BaseTankAgent):
    """
    Heuristic Minimax Agent implementing `H-MINIMAX(s, d)` (Lecture 07 Slides 27-29).
    Replaces `TERMINAL-TEST` with `CUTOFF-TEST(s, d)` and `UTILITY` with the
    multi-feature tactical heuristic evaluation function `EVAL(s)` (`evaluate_tactical`),
    without Alpha-Beta branch pruning or move ordering.
    """

    ALGORITHM_NAME = "Heuristic Minimax Search"

    def __init__(self, player_id: int, max_depth: int = 3) -> None:
        super().__init__(
            player_id=player_id,
            max_depth=max_depth,
            name=self.ALGORITHM_NAME,
        )

    def cutoff_test(self, engine: TankGameEngine, depth: int) -> bool:
        """CUTOFF-TEST(s, d) (Lecture 07 Slide 28)."""
        return engine.is_terminal() or depth <= 0

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        best_action = legal_actions[0]
        best_score = -math.inf

        for act in legal_actions:
            if self.max_depth <= 1:
                child = _apply_tactical_half_ply(engine_copy, self.player_id, act)
                raw_score = self._heuristic_minimax(child, 0, False, None)
            else:
                raw_score = self._heuristic_minimax(
                    engine_copy, self.max_depth - 1, False, act
                )
            score = raw_score + self._record_position_and_get_penalty(engine_copy, act)
            if score > best_score + 1e-6:
                best_score = score
                best_action = act
            elif abs(score - best_score) <= 1e-6:
                if canonical_action_rank(self.player_id, act) > canonical_action_rank(
                    self.player_id, best_action
                ):
                    best_action = act

        self.update_position_memory(engine_copy)
        self.root_best_value = best_score
        self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_action, self._build_metrics(best_score)

    def _heuristic_minimax(
        self,
        engine: TankGameEngine,
        depth: int,
        is_max_turn: bool,
        pending_max_action: Optional[Action],
    ) -> float:
        self.explored_states += 1

        if self.cutoff_test(engine, depth):
            return evaluate_tactical(engine, self.player_id, depth)

        if is_max_turn:
            v = -math.inf
            actions = engine.get_legal_actions(self.player_id)
            if depth == 1:
                for act in actions:
                    child = _apply_tactical_half_ply(engine, self.player_id, act)
                    val = self._heuristic_minimax(child, 0, False, None)
                    if val > v:
                        v = val
                return v

            for act in actions:
                val = self._heuristic_minimax(engine, depth - 1, False, act)
                if val > v:
                    v = val
            return v
        else:
            v = math.inf
            actions = engine.get_legal_actions(self.opponent_id)
            max_act = (
                pending_max_action if pending_max_action is not None else Action.STAY
            )
            for min_act in actions:
                child = _apply_joint_step(engine, self.player_id, max_act, min_act)
                val = self._heuristic_minimax(child, depth - 1, True, None)
                if val < v:
                    v = val
            return v


class HeuristicAlphaBetaAgent(BaseTankAgent):
    """
    Heuristic Alpha-Beta Agent combining `H-MINIMAX(s, d)` (Lecture 07 Slides 27-29)
    with Alpha-Beta Pruning, Move Ordering, Killer Move Heuristic, and
    Transposition Table state caching (Lecture 07 Slide 25).
    """

    ALGORITHM_NAME = "Heuristic Minimax Search with Alpha-Beta Pruning"

    def __init__(self, player_id: int, max_depth: int = 3) -> None:
        super().__init__(
            player_id=player_id,
            max_depth=max_depth,
            name=self.ALGORITHM_NAME,
        )
        self.tt = TranspositionTable(max_entries=100000)
        self._killer_moves: Dict[int, Action] = {}

    def cutoff_test(self, engine: TankGameEngine, depth: int) -> bool:
        """CUTOFF-TEST(s, d) (Lecture 07 Slide 28)."""
        return engine.is_terminal() or depth <= 0

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()
        self.tt.clear()
        self._killer_moves.clear()

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        root_key = engine_copy.state_key(self.player_id, None)
        _, tt_best = self.tt.lookup(root_key, self.max_depth, -math.inf, math.inf)
        ordered = order_actions(
            engine_copy, self.player_id, legal_actions, tt_best_action=tt_best
        )

        best_action = ordered[0]
        best_score = -math.inf
        beta = math.inf

        for act in ordered:
            pen = self._record_position_and_get_penalty(engine_copy, act)
            root_alpha = best_score - pen - 1e-5
            if self.max_depth <= 1:
                child = _apply_tactical_half_ply(engine_copy, self.player_id, act)
                raw_score = self._heuristic_alphabeta(
                    child, 0, root_alpha, beta, False, None
                )
            else:
                raw_score = self._heuristic_alphabeta(
                    engine_copy, self.max_depth - 1, root_alpha, beta, False, act
                )
            score = raw_score + pen
            if score > best_score + 1e-6:
                best_score = score
                best_action = act
            elif abs(score - best_score) <= 1e-6:
                if canonical_action_rank(self.player_id, act) > canonical_action_rank(
                    self.player_id, best_action
                ):
                    best_action = act

        self.tt.store(root_key, self.max_depth, TT_EXACT, best_score, best_action)
        self.update_position_memory(engine_copy)
        self.tt_hits = self.tt.hits
        self.root_best_value = best_score
        self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_action, self._build_metrics(best_score)

    def _heuristic_alphabeta(
        self,
        engine: TankGameEngine,
        depth: int,
        alpha: float,
        beta: float,
        is_max_turn: bool,
        pending_max_action: Optional[Action],
    ) -> float:
        if engine.is_terminal():
            self.explored_states += 1
            return evaluate_terminal(engine, self.player_id, depth)

        turn_player = self.player_id if is_max_turn else self.opponent_id
        state_key = engine.state_key(turn_player, pending_max_action)
        cached_val, tt_best_action = self.tt.lookup(state_key, depth, alpha, beta)
        if cached_val is not None:
            b_est = len(engine.get_legal_actions(turn_player))
            self.pruned_branches += _subtree_node_estimate(b_est, max(1, depth + 1))
            return cached_val

        self.explored_states += 1

        if depth <= 0:
            val = evaluate_tactical(engine, self.player_id, 0)
            self.tt.store(state_key, 0, TT_EXACT, val, None)
            return val

        alpha_orig = alpha
        beta_orig = beta
        killer = self._killer_moves.get(depth)

        if is_max_turn:
            v = -math.inf
            best_act: Optional[Action] = None
            actions = order_actions(
                engine,
                self.player_id,
                engine.get_legal_actions(self.player_id),
                tt_best_action=tt_best_action,
                killer_action=killer,
            )
            n_act = len(actions)

            if depth == 1:
                for idx, act in enumerate(actions):
                    child = _apply_tactical_half_ply(engine, self.player_id, act)
                    val = self._heuristic_alphabeta(child, 0, alpha, beta, False, None)
                    if val > v:
                        v = val
                        best_act = act
                    if v >= beta:
                        skipped = max(1, n_act - (idx + 1))
                        self.pruned_branches += skipped
                        self._killer_moves[depth] = act
                        break
                    if v > alpha:
                        alpha = v
            else:
                for idx, act in enumerate(actions):
                    val = self._heuristic_alphabeta(
                        engine, depth - 1, alpha, beta, False, act
                    )
                    if val > v:
                        v = val
                        best_act = act
                    if v >= beta:
                        skipped = max(1, n_act - (idx + 1))
                        self.pruned_branches += skipped * _subtree_node_estimate(
                            n_act, depth
                        )
                        self._killer_moves[depth] = act
                        break
                    if v > alpha:
                        alpha = v

            flag = (
                TT_EXACT
                if alpha_orig < v < beta_orig
                else (TT_LOWERBOUND if v >= beta_orig else TT_UPPERBOUND)
            )
            self.tt.store(state_key, depth, flag, v, best_act)
            return v
        else:
            v = math.inf
            best_act = None
            actions = order_actions(
                engine,
                self.opponent_id,
                engine.get_legal_actions(self.opponent_id),
                tt_best_action=tt_best_action,
                killer_action=killer,
            )
            n_act = len(actions)
            max_act = (
                pending_max_action if pending_max_action is not None else Action.STAY
            )

            for idx, min_act in enumerate(actions):
                child = _apply_joint_step(engine, self.player_id, max_act, min_act)
                val = self._heuristic_alphabeta(
                    child, depth - 1, alpha, beta, True, None
                )
                if val < v:
                    v = val
                    best_act = min_act
                if v <= alpha:
                    skipped = max(1, n_act - (idx + 1))
                    self.pruned_branches += skipped * _subtree_node_estimate(
                        n_act, depth
                    )
                    self._killer_moves[depth] = min_act
                    break
                if v < beta:
                    beta = v

            flag = (
                TT_EXACT
                if alpha_orig < v < beta_orig
                else (TT_UPPERBOUND if v <= alpha_orig else TT_LOWERBOUND)
            )
            self.tt.store(state_key, depth, flag, v, best_act)
            return v


AGENT_REGISTRY: Dict[str, Type[BaseTankAgent]] = {
    "Minimax": MinimaxAgent,
    "MinimaxAgent": MinimaxAgent,
    "Minimax Search": MinimaxAgent,
    "Alpha-Beta": AlphaBetaAgent,
    "AlphaBeta": AlphaBetaAgent,
    "AlphaBetaAgent": AlphaBetaAgent,
    "Alpha-Beta Search": AlphaBetaAgent,
    "Heuristic Minimax": HeuristicMinimaxAgent,
    "HeuristicMinimax": HeuristicMinimaxAgent,
    "HeuristicMinimaxAgent": HeuristicMinimaxAgent,
    "Heuristic Minimax Search": HeuristicMinimaxAgent,
    "Heuristic Alpha-Beta": HeuristicAlphaBetaAgent,
    "HeuristicAlphaBeta": HeuristicAlphaBetaAgent,
    "HeuristicAlphaBetaAgent": HeuristicAlphaBetaAgent,
    "Heuristic Minimax Search with Alpha-Beta Pruning": HeuristicAlphaBetaAgent,
}


def get_agent_by_name(
    name: str,
    player_id: int = 1,
    max_depth: int = 3,
) -> BaseTankAgent:
    """Instantiate a registered Tank Battle agent by display name or class name."""
    if name in AGENT_REGISTRY:
        return AGENT_REGISTRY[name](player_id=player_id, max_depth=max_depth)
    norm = (name or "").strip().lower()
    for k, cls in AGENT_REGISTRY.items():
        if k.lower() == norm:
            return cls(player_id=player_id, max_depth=max_depth)
    raise ValueError(f"Unknown agent name: {name}")


def load_external_agent(file_path: str | Path) -> Dict[str, Type[BaseTankAgent]]:
    """
    Dynamically load external student group bot classes inheriting from
    `BaseTankAgent` from a Python file path.
    """
    path = Path(file_path).resolve()
    if not path.is_file() or path.suffix.lower() != ".py":
        raise ValueError(f"Invalid external bot file path: {path}")

    module_name = f"task2_external_bot_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    found: Dict[str, Type[BaseTankAgent]] = {}
    for cls_name, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and obj is not BaseTankAgent
            and issubclass(obj, BaseTankAgent)
            and not inspect.isabstract(obj)
            and obj.__module__ == module_name
        ):
            found[cls_name] = obj
            AGENT_REGISTRY[cls_name] = obj

    if not found:
        raise ValueError(f"No BaseTankAgent subclass found in {path}")
    return found
