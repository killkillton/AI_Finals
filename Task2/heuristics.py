"""
Heuristic evaluation functions and move ordering for the Tank Battle game.

Provides terminal utility scoring, basic material evaluation, multi-feature
tactical evaluation, and alpha-beta move ordering. All evaluation functions
are strictly zero-sum between Player 1 and Player 2.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from Task2.tank_game_engine import (
    ALL_ACTIONS,
    CANONICAL_ACTION_RANK,
    DIR_DELTAS,
    GRID_SIZE,
    OPPOSITE_DIR,
    Action,
    Direction,
    TankGameEngine,
    canonical_action_rank,
)

WIN_SCORE: float = 100000.0
TERMINAL_WIN_SCORE: float = WIN_SCORE


def evaluate_terminal(
    engine: TankGameEngine,
    max_player: int,
    remaining_depth: int = 0,
) -> float:
    """
    Return the exact zero-sum terminal utility from max_player's perspective.
    """
    min_player = 3 - max_player
    t_max = engine.tanks[max_player]
    t_min = engine.tanks[min_player]
    b_max = engine.bases[max_player]
    b_min = engine.bases[min_player]

    hp_diff = (t_max["hp"] - t_min["hp"]) * 200.0 + (b_max["hp"] - b_min["hp"]) * 150.0
    if engine.winner == max_player:
        return WIN_SCORE + remaining_depth * 500.0 + hp_diff
    if engine.winner == min_player:
        return -WIN_SCORE - remaining_depth * 500.0 + hp_diff
    return hp_diff


def _basic_pos_score(engine: TankGameEngine, player_id: int) -> float:
    """
    Compute a simple Manhattan proximity tie-breaker toward enemy targets.
    """
    opp_id = 3 - player_id
    my_tank = engine.tanks[player_id]
    opp_tank = engine.tanks[opp_id]
    opp_base = engine.bases[opp_id]
    if not my_tank["alive"]:
        return -50.0

    tx, ty = my_tank["x"], my_tank["y"]
    t_dir = my_tank["dir"]
    dist_tank = abs(tx - opp_tank["x"]) + abs(ty - opp_tank["y"])
    dist_base = abs(tx - opp_base["x"]) + abs(ty - opp_base["y"])

    clear_tank_los = False
    facing_bonus = 0.0
    for target, bonus, is_tank in (
        (opp_tank, 0.8, True),
        (opp_base, 0.4, False),
    ):
        if tx == target["x"] and ty != target["y"]:
            step = 1 if target["y"] > ty else -1
            req_dir = Direction.DOWN if step > 0 else Direction.UP
            walls_between = sum(
                1 for y in range(ty + step, target["y"], step) if (tx, y) in engine.walls
            )
            if walls_between == 0:
                if is_tank:
                    clear_tank_los = True
                if t_dir == req_dir:
                    facing_bonus += bonus
        elif ty == target["y"] and tx != target["x"]:
            step = 1 if target["x"] > tx else -1
            req_dir = Direction.RIGHT if step > 0 else Direction.LEFT
            walls_between = sum(
                1 for x in range(tx + step, target["x"], step) if (x, ty) in engine.walls
            )
            if walls_between == 0:
                if is_tank:
                    clear_tank_los = True
                if t_dir == req_dir:
                    facing_bonus += bonus

    if clear_tank_los and dist_tank <= 6:
        return facing_bonus
    return -(1.2 * dist_tank + 0.4 * dist_base) + facing_bonus


def evaluate_basic(
    engine: TankGameEngine,
    max_player: int,
    remaining_depth: int = 0,
) -> float:
    """
    Evaluate states using terminal utility or HP difference plus Manhattan proximity.
    """
    if engine.is_terminal():
        return evaluate_terminal(engine, max_player, remaining_depth)

    min_player = 3 - max_player
    t_max = engine.tanks[max_player]
    t_min = engine.tanks[min_player]
    b_max = engine.bases[max_player]
    b_min = engine.bases[min_player]

    hp_payoff = 300.0 * (t_max["hp"] - t_min["hp"]) + 220.0 * (b_max["hp"] - b_min["hp"])
    pos_tiebreak = _basic_pos_score(engine, max_player) - _basic_pos_score(engine, min_player)
    return hp_payoff + pos_tiebreak


def evaluate_tactical(
    engine: TankGameEngine,
    max_player: int,
    remaining_depth: int = 0,
) -> float:
    """
    Evaluate non-terminal board states using HP balance, bullet threats,
    wall-aware path distance, firing alignment, mobility, and base defense.
    """
    if engine.is_terminal():
        return evaluate_terminal(engine, max_player, remaining_depth)

    min_player = 3 - max_player
    t_max = engine.tanks[max_player]
    t_min = engine.tanks[min_player]
    b_max = engine.bases[max_player]
    b_min = engine.bases[min_player]

    w_tank_hp = 420.0 * (t_max["hp"] - t_min["hp"])
    w_base_hp = 280.0 * (b_max["hp"] - b_min["hp"])
    w_bullets = _evaluate_bullets(engine, max_player)
    w_alignment = (
        _alignment_score_for_player(engine, max_player)
        - _alignment_score_for_player(engine, min_player)
    )
    w_mobility = 2.0 * (_mobility(engine, t_max) - _mobility(engine, t_min))
    w_defense = (
        _base_defense_score(engine, max_player)
        - _base_defense_score(engine, min_player)
    )

    return w_tank_hp + w_base_hp + w_bullets + w_alignment + w_mobility + w_defense


def _bullet_score_for_player(engine: TankGameEngine, pid: int) -> float:
    """
    Evaluate active bullets for player pid based on projected line impacts.
    """
    opp_id = 3 - pid
    my_tank = engine.tanks[pid]
    opp_tank = engine.tanks[opp_id]
    my_base = engine.bases[pid]
    opp_base = engine.bases[opp_id]
    walls = engine.walls

    score = 0.0
    for b in engine.bullets:
        if b["owner"] != pid:
            continue

        bx, by = b["x"], b["y"]
        bdir = b["dir"]
        dx, dy = DIR_DELTAS[bdir]
        cx, cy = bx, by

        # Check if this bullet will intercept an opposing enemy bullet on the same line
        intercepted = False
        for ob in engine.bullets:
            if ob["owner"] == pid or ob["dir"] != OPPOSITE_DIR[bdir]:
                continue
            if dx != 0 and ob["y"] == by and (ob["x"] - bx) * dx > 0:
                step = dx
                clear_between = all(
                    (x, by) not in walls for x in range(bx + step, ob["x"], step)
                )
                if clear_between:
                    intercepted = True
                    break
            elif dy != 0 and ob["x"] == bx and (ob["y"] - by) * dy > 0:
                step = dy
                clear_between = all(
                    (bx, y) not in walls for y in range(by + step, ob["y"], step)
                )
                if clear_between:
                    intercepted = True
                    break

        if intercepted:
            score += 45.0
            continue

        for step_idx in range(1, 10):
            cx += dx
            cy += dy
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                break

            if (cx, cy) in walls:
                # Strongly penalize shooting friendly base shield walls
                if abs(cx - my_base["x"]) + abs(cy - my_base["y"]) <= 2:
                    score -= 60.0
                elif abs(cx - opp_base["x"]) + abs(cy - opp_base["y"]) <= 2:
                    score += 10.0
                elif (
                    (cx == opp_tank["x"] and cx == my_tank["x"])
                    or (cy == opp_tank["y"] and cy == my_tank["y"])
                    or (cx == opp_base["x"] and cx == my_tank["x"])
                    or (cy == opp_base["y"] and cy == my_tank["y"])
                ):
                    score += 6.0
                break

            if opp_tank["alive"] and (cx, cy) == (opp_tank["x"], opp_tank["y"]):
                cycles_to_hit = (step_idx + 2) // 3
                if cycles_to_hit == 1:
                    score += 180.0
                elif cycles_to_hit == 2:
                    score += 95.0
                else:
                    score += 45.0
                break

            if opp_base["hp"] > 0 and (cx, cy) == (opp_base["x"], opp_base["y"]):
                cycles_to_hit = (step_idx + 2) // 3
                if cycles_to_hit == 1:
                    score += 150.0
                elif cycles_to_hit == 2:
                    score += 90.0
                else:
                    score += 45.0
                break

            if my_base["hp"] > 0 and (cx, cy) == (my_base["x"], my_base["y"]):
                score -= 260.0
                break

    return score


def _evaluate_bullets(engine: TankGameEngine, max_player: int) -> float:
    """
    Strictly antisymmetric bullet evaluation:
      _evaluate_bullets(engine, 1) == -_evaluate_bullets(engine, 2)
    """
    min_player = 3 - max_player
    return _bullet_score_for_player(engine, max_player) - _bullet_score_for_player(
        engine, min_player
    )


_LANE_CACHE: Dict[Tuple, Dict[Tuple[int, int], float]] = {}
_CELL_CACHE: Dict[Tuple, Dict[Tuple[int, int], float]] = {}


def _build_lane_distance_map(
    engine: TankGameEngine,
    target_x: int,
    target_y: int,
    seed_lanes: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    Compute Dijkstra path costs to clear firing lanes or directly to the target cell.
    """
    wall_sig = tuple(sorted(engine.walls.items()))
    cache_dict = _LANE_CACHE if seed_lanes else _CELL_CACHE
    cache_key = (wall_sig, target_x, target_y)
    cached = cache_dict.get(cache_key)
    if cached is not None:
        return cached

    if len(cache_dict) > 256:
        cache_dict.clear()

    dist_map: Dict[Tuple[int, int], float] = {}
    queue: deque = deque()

    if seed_lanes:
        # Seed unobstructed firing lanes up to 6 cells away from target
        for d in Direction:
            dx, dy = DIR_DELTAS[d]
            for step in range(1, 7):
                sx, sy = target_x + dx * step, target_y + dy * step
                if not (0 <= sx < GRID_SIZE and 0 <= sy < GRID_SIZE):
                    break
                if (sx, sy) in engine.walls:
                    break
                if (sx, sy) not in dist_map:
                    dist_map[(sx, sy)] = 0.0
                    queue.append((sx, sy, 0.0))

    if not queue:
        dist_map[(target_x, target_y)] = 0.0
        queue.append((target_x, target_y, 0.0))

    while queue:
        cx, cy, cost = queue.popleft()
        if cost > dist_map.get((cx, cy), 1e9):
            continue
        for d in Direction:
            dx, dy = DIR_DELTAS[d]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            nxt = (nx, ny)
            step_cost = 1.0 + (0.75 * engine.walls[nxt] if nxt in engine.walls else 0.0)
            new_cost = cost + step_cost
            if new_cost < dist_map.get(nxt, 1e9):
                dist_map[nxt] = new_cost
                queue.append((nx, ny, new_cost))

    cache_dict[cache_key] = dist_map
    return dist_map


def _query_lane_distance(
    engine: TankGameEngine,
    target_x: int,
    target_y: int,
    x: int,
    y: int,
    seed_lanes: bool = True,
) -> float:
    dist_map = _build_lane_distance_map(engine, target_x, target_y, seed_lanes=seed_lanes)
    return dist_map.get((x, y), float(abs(target_x - x) + abs(target_y - y)))


def _alignment_score_for_player(engine: TankGameEngine, player_id: int) -> float:
    """
    Evaluate offensive alignment, wall-breach progress, friendly base shield
    preservation, and shortest-path pursuit for `player_id`.
    """
    opp_id = 3 - player_id
    my_tank = engine.tanks[player_id]
    opp_tank = engine.tanks[opp_id]
    my_base = engine.bases[player_id]
    opp_base = engine.bases[opp_id]

    if not my_tank["alive"]:
        return -100.0

    tx, ty = my_tank["x"], my_tank["y"]
    t_dir = my_tank["dir"]
    score = 0.0

    # Reward keeping protective brick walls intact near the friendly base.
    for (wx, wy), whp in engine.walls.items():
        if abs(wx - my_base["x"]) + abs(wy - my_base["y"]) <= 2:
            score += 14.0 * whp

    # 2. Line-of-sight alignment with enemy tank (primary) and enemy base (secondary)
    tank_vulnerability = 1.0 + 0.15 * max(0, 6 - opp_tank["hp"])
    base_vulnerability = 1.0 + 0.10 * max(0, 9 - opp_base["hp"])

    for target, weight in (
        (opp_tank, 1.85 * tank_vulnerability),
        (opp_base, 1.20 * base_vulnerability),
    ):
        if tx == target["x"] and ty != target["y"]:
            step = 1 if target["y"] > ty else -1
            req_dir = Direction.DOWN if step > 0 else Direction.UP
            walls_between = 0
            wall_hp_between = 0
            for y in range(ty + step, target["y"], step):
                if (tx, y) in engine.walls:
                    walls_between += 1
                    wall_hp_between += engine.walls[(tx, y)]
            dist = abs(target["y"] - ty)
            if walls_between == 0:
                score += (28.0 - 1.2 * min(dist, 12)) * weight
                if t_dir == req_dir:
                    score += 12.0 * weight
            elif walls_between == 1:
                score += max(2.0, 14.0 - 0.5 * min(dist, 12) - 2.5 * wall_hp_between) * weight
                if t_dir == req_dir:
                    score += 4.0 * weight
            elif walls_between == 2:
                score += max(0.5, 6.0 - 0.3 * min(dist, 12) - 1.0 * wall_hp_between) * weight

        elif ty == target["y"] and tx != target["x"]:
            step = 1 if target["x"] > tx else -1
            req_dir = Direction.RIGHT if step > 0 else Direction.LEFT
            walls_between = 0
            wall_hp_between = 0
            for x in range(tx + step, target["x"], step):
                if (x, ty) in engine.walls:
                    walls_between += 1
                    wall_hp_between += engine.walls[(x, ty)]
            dist = abs(target["x"] - tx)
            if walls_between == 0:
                score += (28.0 - 1.2 * min(dist, 12)) * weight
                if t_dir == req_dir:
                    score += 12.0 * weight
            elif walls_between == 1:
                score += max(2.0, 14.0 - 0.5 * min(dist, 12) - 2.5 * wall_hp_between) * weight
                if t_dir == req_dir:
                    score += 4.0 * weight
            elif walls_between == 2:
                score += max(0.5, 6.0 - 0.3 * min(dist, 12) - 1.0 * wall_hp_between) * weight

    # 3. Wall-aware shortest path pursuit toward enemy tank (primary) and enemy base (secondary)
    lane_cost_base = _query_lane_distance(
        engine, opp_base["x"], opp_base["y"], tx, ty, seed_lanes=True
    )
    intercept_x = opp_tank["x"]
    intercept_y = int(round((2 * opp_tank["y"] + opp_base["y"]) / 3.0))
    lane_cost_tank = _query_lane_distance(
        engine, intercept_x, intercept_y, tx, ty, seed_lanes=False
    )
    score -= (1.8 * lane_cost_base + 4.2 * lane_cost_tank)

    return score


def _mobility(engine: TankGameEngine, tank: Dict) -> int:
    """Count walkable adjacent cells around `tank`."""
    if not tank["alive"]:
        return 0
    tx, ty = tank["x"], tank["y"]
    free = 0
    for d in Direction:
        dx, dy = DIR_DELTAS[d]
        nx, ny = tx + dx, ty + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in engine.walls:
            free += 1
    return free


def _base_defense_score(engine: TankGameEngine, player_id: int) -> float:
    """
    Evaluate defensive positioning when the opponent threatens `player_id`'s base.
    """
    opp_id = 3 - player_id
    my_base = engine.bases[player_id]
    my_tank = engine.tanks[player_id]
    opp_tank = engine.tanks[opp_id]

    bx, by = my_base["x"], my_base["y"]
    ox, oy = opp_tank["x"], opp_tank["y"]
    tx, ty = my_tank["x"], my_tank["y"]

    opp_dist_to_my_base = abs(ox - bx) + abs(oy - by)
    my_dist_to_my_base = abs(tx - bx) + abs(ty - by)

    score = 0.0
    if opp_dist_to_my_base <= 5:
        score -= (6 - opp_dist_to_my_base) * 8.0
        if my_dist_to_my_base <= opp_dist_to_my_base:
            score += 12.0
    return score


def order_actions(
    engine: TankGameEngine,
    player_id: int,
    actions: Optional[List[Action]] = None,
    tt_best_action: Optional[Action] = None,
    killer_action: Optional[Action] = None,
) -> List[Action]:
    """
    Order candidate actions to improve alpha-beta pruning efficiency.
    Prioritizes transposition table hints, killer moves, bullet evasion, and shots.
    """
    if actions is None:
        actions = engine.get_legal_actions(player_id)

    opp_id = 3 - player_id
    t = engine.tanks[player_id]
    opp_t = engine.tanks[opp_id]
    my_b = engine.bases[player_id]
    opp_b = engine.bases[opp_id]
    tx, ty = t["x"], t["y"]
    tdir = t["dir"]

    under_imminent_fire = False
    incoming_axis: Optional[str] = None
    for b in engine.bullets:
        if b["owner"] == opp_id:
            bdx, bdy = DIR_DELTAS[b["dir"]]
            if bdx != 0 and b["y"] == ty and 0 < (tx - b["x"]) * bdx <= 4:
                under_imminent_fire = True
                incoming_axis = "H"
                break
            if bdy != 0 and b["x"] == tx and 0 < (ty - b["y"]) * bdy <= 4:
                under_imminent_fire = True
                incoming_axis = "V"
                break

    intercept_x = opp_t["x"]
    intercept_y = int(round((2 * opp_t["y"] + opp_b["y"]) / 3.0))
    cur_tank_cost = _query_lane_distance(
        engine, intercept_x, intercept_y, tx, ty, seed_lanes=False
    )
    cur_base_cost = _query_lane_distance(
        engine, opp_b["x"], opp_b["y"], tx, ty, seed_lanes=True
    )

    scored: List[Tuple[float, Action]] = []
    for act in actions:
        s = 0.0
        if tt_best_action is not None and act == tt_best_action:
            s += 1000.0
        if killer_action is not None and act == killer_action:
            s += 500.0

        if under_imminent_fire:
            if incoming_axis == "H" and act in (Action.MOVE_UP, Action.MOVE_DOWN):
                s += 300.0
            elif incoming_axis == "V" and act in (Action.MOVE_LEFT, Action.MOVE_RIGHT):
                s += 300.0
            elif act == Action.STAY:
                s -= 150.0

        if act == Action.FIRE:
            fdx, fdy = DIR_DELTAS[tdir]
            cx, cy = tx, ty
            for step in range(1, 10):
                cx += fdx
                cy += fdy
                if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                    break
                if (cx, cy) in engine.walls:
                    if abs(cx - my_b["x"]) + abs(cy - my_b["y"]) <= 2:
                        s -= 90.0
                    elif abs(cx - opp_b["x"]) + abs(cy - opp_b["y"]) <= 2:
                        s += 180.0 - step * 8.0
                    elif (
                        (cx == opp_t["x"] and cx == tx)
                        or (cy == opp_t["y"] and cy == ty)
                        or (cx == opp_b["x"] and cx == tx)
                        or (cy == opp_b["y"] and cy == ty)
                    ):
                        s += 60.0 - step * 5.0
                    break
                if (cx, cy) == (my_b["x"], my_b["y"]):
                    s -= 150.0
                    break
                if (cx, cy) == (opp_t["x"], opp_t["y"]) or (cx, cy) == (opp_b["x"], opp_b["y"]):
                    s += 220.0 - step * 10.0
                    break
        elif act in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            dir_map = {
                Action.MOVE_UP: Direction.UP,
                Action.MOVE_DOWN: Direction.DOWN,
                Action.MOVE_LEFT: Direction.LEFT,
                Action.MOVE_RIGHT: Direction.RIGHT,
            }
            d = dir_map[act]
            dx, dy = DIR_DELTAS[d]
            nx, ny = tx + dx, ty + dy
            new_tank_cost = _query_lane_distance(
                engine, intercept_x, intercept_y, nx, ny, seed_lanes=False
            )
            new_base_cost = _query_lane_distance(
                engine, opp_b["x"], opp_b["y"], nx, ny, seed_lanes=True
            )
            s += (cur_tank_cost - new_tank_cost) * 16.0 + (cur_base_cost - new_base_cost) * 8.0
            if nx == opp_t["x"] or ny == opp_t["y"]:
                s += 20.0
            if nx == opp_b["x"] or ny == opp_b["y"]:
                s += 12.0
        else:
            s -= 25.0

        scored.append((s, act))

    scored.sort(
        key=lambda item: (item[0], canonical_action_rank(player_id, item[1])),
        reverse=True,
    )
    return [act for _, act in scored]
