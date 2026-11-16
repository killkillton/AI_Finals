from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from Task2.agents import BaseTankAgent, _apply_joint_step
from Task2.heuristics import evaluate_basic
from Task2.tank_game_engine import Action, TankGameEngine


class ExternalGroupBot(BaseTankAgent):
    """
    Sample external bot class demonstrating dynamic loading via BaseTankAgent.
    Uses a greedy 2-ply lookahead with basic material and approach scoring.
    """

    name: str = "ExternalGroupBot"
    ALGORITHM_NAME = "External Group Bot"

    def get_action(self, engine_copy: TankGameEngine) -> Tuple[Action, Dict[str, Any]]:
        start = time.perf_counter()
        if engine_copy.is_terminal():
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            score = round(evaluate_basic(engine_copy, self.player_id, 0), 4)
            return Action.STAY, {
                "algorithm": self.ALGORITHM_NAME,
                "time_ms": round(elapsed_ms, 4),
                "explored_states": 1,
                "depth_reached": 0,
                "eval_score": score,
                "best_score": score,
                "pruned_branches": 0,
                "pruning_rate": 0.0,
                "tt_hits": 0,
            }

        explored = 0
        candidates = engine_copy.get_legal_actions(self.player_id)
        opp_candidates = engine_copy.get_legal_actions(self.opponent_id)

        best_action = candidates[0]
        best_score = float("-inf")

        for my_act in candidates:
            worst_reply_score = float("inf")
            for opp_act in opp_candidates:
                explored += 1
                child = _apply_joint_step(
                    engine_copy, self.player_id, my_act, opp_act
                )
                score = evaluate_basic(child, self.player_id, 1)
                if score < worst_reply_score:
                    worst_reply_score = score
            if worst_reply_score > best_score:
                best_score = worst_reply_score
                best_action = my_act

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rounded_score = round(best_score, 4)
        metrics = {
            "algorithm": self.ALGORITHM_NAME,
            "time_ms": round(elapsed_ms, 4),
            "explored_states": explored,
            "depth_reached": 2,
            "eval_score": rounded_score,
            "best_score": rounded_score,
            "pruned_branches": 0,
            "pruning_rate": 0.0,
            "tt_hits": 0,
        }
        return best_action, metrics
