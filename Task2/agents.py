"""
Adversarial search agents for the Tank Battle game.

Implements bounded-depth Minimax, Alpha-Beta pruning, and heuristic-guided
variants with move ordering, killer-move tracking, and transposition caching.
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
    Stores previously evaluated search states keyed by engine state signature.
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
        Look up a cached state at the exact remaining depth.
        Returns a cached value when an exact score or valid alpha/beta bound
        cutoff matches, otherwise returns only the cached best action hint.
        """
        entry = self._table.get(key)
        if entry is None:
            return None, None
        stored_depth, flag, val, best_act = entry
        if stored_depth == depth:
            if (
                flag == TT_EXACT
                or (flag == TT_LOWERBOUND and val >= beta)
                or (flag == TT_UPPERBOUND and val <= alpha)
            ):
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


class BaseTankAgent(ABC):
    """
    Abstract base class for all Tank Battle adversarial search agents.
    """

    name: str = "BaseAgent"
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
        self.depth_reached: int = 0
        self.decision_time_ms: float = 0.0
        self.root_best_value: float = 0.0

        # Anti-oscillation and anti-stagnation memory across live game turns
        self._recent_positions: List[Tuple[int, int]] = []
        self._recent_hp_sigs: List[Tuple[int, int, int, int, int, int]] = []

    def reset_metrics(self) -> None:
        self.explored_states = 0
        self.pruned_branches = 0
        self.tt_hits = 0
        self.depth_reached = 0
        self.decision_time_ms = 0.0
        self.root_best_value = 0.0

    def _note_depth(self, remaining_depth: int) -> None:
        reached = max(0, self.max_depth - remaining_depth)
        if reached > self.depth_reached:
            self.depth_reached = reached

    def _board_hp_signature(
        self, engine: TankGameEngine
    ) -> Tuple[int, int, int, int, int, int]:
        opp_id = self.opponent_id
        return (
            engine.tanks[opp_id]["hp"],
            engine.bases[opp_id]["hp"],
            len(engine.walls),
            sum(engine.walls.values()),
            engine.tanks[self.player_id]["hp"],
            engine.bases[self.player_id]["hp"],
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
                if abs(cx - opp_b["x"]) + abs(cy - opp_b["y"]) == 1:
                    return step <= 4
                if (
                    (cx == opp_t["x"] and tx == opp_t["x"])
                    or (cy == opp_t["y"] and ty == opp_t["y"])
                ):
                    return step <= 3
                return False
            if opp_t["alive"] and (cx, cy) == (opp_t["x"], opp_t["y"]):
                return step <= 6
            if opp_b["hp"] > 0 and (cx, cy) == (opp_b["x"], opp_b["y"]):
                return step <= 5
        return False

    def _record_position_and_get_penalty(
        self,
        engine: TankGameEngine,
        action: Action,
    ) -> float:
        """
        Penalize wasted shots, repeated moves, contested-cell stalls, and
        friendly-base shots during live play.
        """
        t = engine.tanks[self.player_id]
        opp_tank = engine.tanks[self.opponent_id]
        my_base = engine.bases[self.player_id]
        tx, ty = t["x"], t["y"]

        valid_target = self._has_valid_fire_target(engine)
        if len(self._recent_positions) < 2:
            return 0.0

        dir_map = {
            Action.MOVE_UP: Direction.UP,
            Action.MOVE_DOWN: Direction.DOWN,
            Action.MOVE_LEFT: Direction.LEFT,
            Action.MOVE_RIGHT: Direction.RIGHT,
        }

        if action in dir_map:
            dx, dy = DIR_DELTAS[dir_map[action]]
            nx, ny = tx + dx, ty + dy
            if not engine.is_cell_walkable(nx, ny, self.player_id):
                nx, ny = tx, ty
        else:
            nx, ny = tx, ty

        repeats = self._recent_positions.count((nx, ny))
        cur_sig = self._board_hp_signature(engine)

        # Count consecutive recent turns spent on (tx, ty) without reducing enemy tank/base/wall HP
        stagnant_repeats = 0
        for pos, sig in zip(
            reversed(self._recent_positions), reversed(self._recent_hp_sigs)
        ):
            if pos == (tx, ty) and sig[:4] == cur_sig[:4]:
                stagnant_repeats += 1
            else:
                break

        penalty = 0.0
        if action == Action.FIRE and not valid_target:
            penalty -= 50.0

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

        fdx, fdy = DIR_DELTAS[t["dir"]]
        opp_in_front_on_line = opp_tank["alive"] and (
            (fdx != 0 and opp_tank["y"] == ty and (opp_tank["x"] - tx) * fdx > 0)
            or (fdy != 0 and opp_tank["x"] == tx and (opp_tank["y"] - ty) * fdy > 0)
        )
        if action == Action.FIRE and close_enemy_in_front and stagnant_repeats < 3:
            return penalty

        if action == Action.FIRE and valid_target:
            if not opp_in_front_on_line or stagnant_repeats < 4:
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
        rounded_score = round(best_score, 4)
        return {
            "algorithm": self.ALGORITHM_NAME,
            "time_ms": round(self.decision_time_ms, 4),
            "explored_states": self.explored_states,
            "depth_reached": self.depth_reached,
            "eval_score": rounded_score,
            "best_score": rounded_score,
            "pruned_branches": self.pruned_branches,
            "pruning_rate": pruning_rate,
            "tt_hits": self.tt_hits,
        }

    @abstractmethod
    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        """Select and return `(best_action, metrics_dict)` for `self.player_id`."""


class MinimaxAgent(BaseTankAgent):
    """
    Depth-bounded Minimax agent using terminal utilities and basic material
    payoff at cutoff nodes.
    """

    name: str = "MinimaxAgent"
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

        if engine_copy.is_terminal():
            self.explored_states = 1
            self.depth_reached = 0
            best_score = evaluate_terminal(engine_copy, self.player_id, self.max_depth)
            self.root_best_value = best_score
            self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
            return Action.STAY, self._build_metrics(best_score)

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        opp_actions = engine_copy.get_legal_actions(self.opponent_id)
        best_action = legal_actions[0]
        best_score = -math.inf

        for act in legal_actions:
            if self.max_depth <= 1:
                raw_score = math.inf
                for min_act in opp_actions:
                    child = _apply_joint_step(engine_copy, self.player_id, act, min_act)
                    val = self._minimax(child, 0, True, None)
                    if val < raw_score:
                        raw_score = val
            else:
                raw_score = self._minimax(engine_copy, self.max_depth - 1, False, act)
            score = raw_score
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
        self._note_depth(depth)

        if engine.is_terminal():
            return evaluate_terminal(engine, self.player_id, depth)
        if depth <= 0:
            return evaluate_basic(engine, self.player_id, 0)

        if is_max_turn:
            v = -math.inf
            actions = engine.get_legal_actions(self.player_id)
            if depth == 1:
                for act in actions:
                    child = _apply_joint_step(
                        engine, self.player_id, act, Action.STAY
                    )
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
    Depth-bounded Alpha-Beta pruning agent using the same basic material payoff
    as MinimaxAgent.
    """

    name: str = "AlphaBetaAgent"
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

        if engine_copy.is_terminal():
            self.explored_states = 1
            self.depth_reached = 0
            best_score = evaluate_terminal(engine_copy, self.player_id, self.max_depth)
            self.root_best_value = best_score
            self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
            return Action.STAY, self._build_metrics(best_score)

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        ordered = (
            order_actions(engine_copy, self.player_id, legal_actions)
            if self.use_move_ordering
            else legal_actions
        )
        raw_opp = engine_copy.get_legal_actions(self.opponent_id)
        opp_actions = (
            order_actions(engine_copy, self.opponent_id, raw_opp)
            if self.use_move_ordering
            else raw_opp
        )
        n_opp = len(opp_actions)

        best_action = ordered[0]
        best_score = -math.inf
        beta = math.inf

        for act in ordered:
            root_alpha = best_score - 1e-5
            if self.max_depth <= 1:
                raw_score = math.inf
                min_beta = beta
                for m_idx, min_act in enumerate(opp_actions):
                    child = _apply_joint_step(engine_copy, self.player_id, act, min_act)
                    val = self._alphabeta(child, 0, root_alpha, min_beta, True, None)
                    if val < raw_score:
                        raw_score = val
                    if raw_score <= root_alpha:
                        skipped = n_opp - (m_idx + 1)
                        if skipped > 0:
                            self.pruned_branches += skipped
                        break
                    if raw_score < min_beta:
                        min_beta = raw_score
            else:
                raw_score = self._alphabeta(
                    engine_copy, self.max_depth - 1, root_alpha, beta, False, act
                )
            score = raw_score
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
        self._note_depth(depth)

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
                    child = _apply_joint_step(
                        engine, self.player_id, act, Action.STAY
                    )
                    val = self._alphabeta(child, 0, alpha, beta, False, None)
                    if val > v:
                        v = val
                    if v >= beta:
                        skipped = n_act - (idx + 1)
                        if skipped > 0:
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
                    skipped = n_act - (idx + 1)
                    if skipped > 0:
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
                    skipped = n_act - (idx + 1)
                    if skipped > 0:
                        self.pruned_branches += skipped * _subtree_node_estimate(
                            n_act, depth
                        )
                    return v
                if v < beta:
                    beta = v
            return v


class HeuristicMinimaxAgent(BaseTankAgent):
    """
    Depth-limited Heuristic Minimax agent using multi-feature tactical leaf
    evaluation without alpha-beta pruning.
    """

    name: str = "HeuristicMinimaxAgent"
    ALGORITHM_NAME = "Heuristic Minimax Search"

    def __init__(self, player_id: int, max_depth: int = 3) -> None:
        super().__init__(
            player_id=player_id,
            max_depth=max_depth,
            name=self.ALGORITHM_NAME,
        )

    def cutoff_test(self, engine: TankGameEngine, depth: int) -> bool:
        """Return True at terminal states or when depth reaches zero."""
        return engine.is_terminal() or depth <= 0

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()

        if engine_copy.is_terminal():
            self.explored_states = 1
            self.depth_reached = 0
            best_score = evaluate_terminal(engine_copy, self.player_id, self.max_depth)
            self.root_best_value = best_score
            self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
            return Action.STAY, self._build_metrics(best_score)

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        opp_actions = engine_copy.get_legal_actions(self.opponent_id)
        best_action = legal_actions[0]
        best_score = -math.inf

        for act in legal_actions:
            if self.max_depth <= 1:
                raw_score = math.inf
                for min_act in opp_actions:
                    child = _apply_joint_step(engine_copy, self.player_id, act, min_act)
                    val = self._heuristic_minimax(child, 0, True, None)
                    if val < raw_score:
                        raw_score = val
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
        self._note_depth(depth)

        if self.cutoff_test(engine, depth):
            return evaluate_tactical(engine, self.player_id, depth)

        if is_max_turn:
            v = -math.inf
            actions = engine.get_legal_actions(self.player_id)
            if depth == 1:
                for act in actions:
                    child = _apply_joint_step(
                        engine, self.player_id, act, Action.STAY
                    )
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
    Heuristic Minimax agent with alpha-beta pruning, move ordering, killer-move
    tracking, and a transposition table.
    """

    name: str = "HeuristicAlphaBetaAgent"
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
        """Return True at terminal states or when depth reaches zero."""
        return engine.is_terminal() or depth <= 0

    def get_action(
        self, engine_copy: TankGameEngine
    ) -> Tuple[Action, Dict[str, Any]]:
        t0 = time.perf_counter()
        self.reset_metrics()
        self.tt.clear()
        self._killer_moves.clear()

        if engine_copy.is_terminal():
            self.explored_states = 1
            self.depth_reached = 0
            best_score = evaluate_terminal(engine_copy, self.player_id, self.max_depth)
            self.root_best_value = best_score
            self.decision_time_ms = (time.perf_counter() - t0) * 1000.0
            return Action.STAY, self._build_metrics(best_score)

        legal_actions = engine_copy.get_legal_actions(self.player_id)
        root_key = engine_copy.state_key(self.player_id, None)
        _, tt_best = self.tt.lookup(root_key, self.max_depth, -math.inf, math.inf)
        ordered = order_actions(
            engine_copy, self.player_id, legal_actions, tt_best_action=tt_best
        )
        opp_actions = order_actions(
            engine_copy,
            self.opponent_id,
            engine_copy.get_legal_actions(self.opponent_id),
        )
        n_opp = len(opp_actions)

        best_action = ordered[0]
        best_score = -math.inf
        beta = math.inf

        for act in ordered:
            pen = self._record_position_and_get_penalty(engine_copy, act)
            root_alpha = best_score - pen - 1e-5
            if self.max_depth <= 1:
                raw_score = math.inf
                min_beta = beta
                for m_idx, min_act in enumerate(opp_actions):
                    child = _apply_joint_step(engine_copy, self.player_id, act, min_act)
                    val = self._heuristic_alphabeta(
                        child, 0, root_alpha, min_beta, True, None
                    )
                    if val < raw_score:
                        raw_score = val
                    if raw_score <= root_alpha:
                        skipped = n_opp - (m_idx + 1)
                        if skipped > 0:
                            self.pruned_branches += skipped
                        break
                    if raw_score < min_beta:
                        min_beta = raw_score
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
        self._note_depth(depth)
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
                    child = _apply_joint_step(
                        engine, self.player_id, act, Action.STAY
                    )
                    val = self._heuristic_alphabeta(
                        child, 0, alpha, beta, False, None
                    )
                    if val > v:
                        v = val
                        best_act = act
                    if v >= beta:
                        skipped = n_act - (idx + 1)
                        if skipped > 0:
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
                        skipped = n_act - (idx + 1)
                        if skipped > 0:
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
                    skipped = n_act - (idx + 1)
                    if skipped > 0:
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
