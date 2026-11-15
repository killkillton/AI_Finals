from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from Task2.agents import BaseTankAgent, _apply_joint_step
from Task2.heuristics import evaluate_tactical, order_actions
from Task2.tank_game_engine import Action, TankGameEngine


class ExternalGroupBot(BaseTankAgent):
    """
    Reference external student group bot implementation demonstrating dynamic
    bot loading via the PyGame interface or importlib loader.
    Uses a greedy 2-ply lookahead with tactical evaluation and bullet evasion.
    """

    ALGORITHM_NAME = "External Group Bot"

    def get_action(self, engine_copy: TankGameEngine) -> Tuple[Action, Dict[str, Any]]:
        start = time.perf_counter()
        explored = 0
        candidates = order_actions(
            engine_copy,
            self.player_id,
            engine_copy.get_legal_actions(self.player_id),
        )
        opp_candidates = order_actions(
            engine_copy,
            self.opponent_id,
            engine_copy.get_legal_actions(self.opponent_id),
        )[:3]

        best_action = candidates[0]
        best_score = float("-inf")

        for my_act in candidates:
            worst_reply_score = float("inf")
            for opp_act in opp_candidates:
                explored += 1
                child = _apply_joint_step(
                    engine_copy, self.player_id, my_act, opp_act
                )
                score = evaluate_tactical(child, self.player_id, 1)
                if score < worst_reply_score:
                    worst_reply_score = score
            if worst_reply_score > best_score:
                best_score = worst_reply_score
                best_action = my_act

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = {
            "algorithm": self.ALGORITHM_NAME,
            "time_ms": round(elapsed_ms, 4),
            "explored_states": explored,
            "depth_reached": min(2, self.max_depth),
            "pruned_branches": 0,
            "pruning_rate": 0.0,
            "tt_hits": 0,
            "best_score": round(best_score, 4),
        }
        return best_action, metrics
