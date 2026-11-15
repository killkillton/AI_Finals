"""
Turn-based engine for the 13x13 Tank Battle game.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

GRID_SIZE: int = 13

MAX_TANK_HP: int = 6
TANK_INITIAL_HP: int = MAX_TANK_HP

MAX_BASE_HP: int = 9
BASE_INITIAL_HP: int = MAX_BASE_HP

MAX_WALL_HP: int = 3
WALL_INITIAL_HP: int = MAX_WALL_HP

BULLET_SPEED: int = 3
BULLET_DAMAGE: int = 1
DAMAGE_PER_HIT: int = BULLET_DAMAGE

DEFAULT_MAX_TURNS: int = 200


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Action(str, Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    FIRE = "FIRE"
    STAY = "STAY"


# Canonical action ordering and deterministic tie-break rank
ALL_ACTIONS: Tuple[Action, ...] = (
    Action.FIRE,
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.STAY,
)

CANONICAL_ACTION_RANK: Dict[Action, int] = {
    Action.FIRE: 6,
    Action.MOVE_UP: 5,
    Action.MOVE_RIGHT: 4,
    Action.MOVE_LEFT: 3,
    Action.MOVE_DOWN: 2,
    Action.STAY: 1,
}

CANONICAL_ACTION_RANK_P2: Dict[Action, int] = {
    Action.FIRE: 6,
    Action.MOVE_DOWN: 5,
    Action.MOVE_LEFT: 4,
    Action.MOVE_RIGHT: 3,
    Action.MOVE_UP: 2,
    Action.STAY: 1,
}


def canonical_action_rank(player_id: int, action: Action) -> int:
    """Return the player-symmetric tie-break rank for an action."""
    if player_id == 2:
        return CANONICAL_ACTION_RANK_P2.get(action, 0)
    return CANONICAL_ACTION_RANK.get(action, 0)

DIR_DELTAS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

ACTION_TO_DIR: Dict[Action, Direction] = {
    Action.MOVE_UP: Direction.UP,
    Action.MOVE_DOWN: Direction.DOWN,
    Action.MOVE_LEFT: Direction.LEFT,
    Action.MOVE_RIGHT: Direction.RIGHT,
}

OPPOSITE_DIR: Dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Standard 37-wall symmetric arena layout
DEFAULT_WALL_COORDS: Tuple[Tuple[int, int], ...] = (
    # Top Base (Player 2 at (6, 0)) shield walls (5 walls)
    (5, 0),
    (7, 0),
    (5, 1),
    (6, 1),
    (7, 1),
    # Bottom Base (Player 1 at (6, 12)) shield walls (5 walls)
    (5, 12),
    (7, 12),
    (5, 11),
    (6, 11),
    (7, 11),
    # Upper brick columns (8 walls)
    (1, 2),
    (1, 3),
    (3, 2),
    (3, 3),
    (9, 2),
    (9, 3),
    (11, 2),
    (11, 3),
    # Lower brick columns (8 walls)
    (1, 9),
    (1, 10),
    (3, 9),
    (3, 10),
    (9, 9),
    (9, 10),
    (11, 9),
    (11, 10),
    # Side border cover walls (2 walls)
    (0, 5),
    (12, 7),
    # Mid-field left and right bunkers (4 walls)
    (2, 6),
    (3, 6),
    (9, 6),
    (10, 6),
    # Center cross obstacle (5 walls)
    (6, 5),
    (5, 6),
    (6, 6),
    (7, 6),
    (6, 7),
)


class TankGameEngine:
    """Turn-based engine for the Tank Battle game on a 13x13 grid."""

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.grid_size: Tuple[int, int] = (GRID_SIZE, GRID_SIZE)
        self.max_turns: int = max_turns
        self.turn: int = 0
        self.tanks: Dict[int, Dict[str, Any]] = {}
        self.bases: Dict[int, Dict[str, Any]] = {}
        self.walls: Dict[Tuple[int, int], int] = {}
        self.bullets: List[Dict[str, Any]] = []
        self.is_terminal_flag: bool = False
        self.winner: Optional[int] = None

        if initial_state is not None:
            self._load_from_dict(initial_state)
        else:
            self.reset()

    @property
    def terminal(self) -> bool:
        return self.is_terminal_flag

    @terminal.setter
    def terminal(self, value: bool) -> None:
        self.is_terminal_flag = bool(value)

    def reset(self) -> Dict[str, Any]:
        """Reset the arena to the initial state S_0 and return the state dict."""
        self.turn = 0
        self.tanks = {
            1: {
                "x": 2,
                "y": 10,
                "dir": Direction.UP,
                "hp": MAX_TANK_HP,
                "alive": True,
            },
            2: {
                "x": 10,
                "y": 2,
                "dir": Direction.DOWN,
                "hp": MAX_TANK_HP,
                "alive": True,
            },
        }
        self.bases = {
            1: {"x": 6, "y": 12, "hp": MAX_BASE_HP, "alive": True},
            2: {"x": 6, "y": 0, "hp": MAX_BASE_HP, "alive": True},
        }
        self.walls = {(wx, wy): MAX_WALL_HP for wx, wy in DEFAULT_WALL_COORDS}
        self.bullets = []
        self.is_terminal_flag = False
        self.winner = None
        return self.get_state()

    def _load_from_dict(self, state: Dict[str, Any]) -> None:
        """Load internal state from a serialized state dictionary."""
        self.turn = int(state.get("turn", 0))
        gs = state.get("grid_size", (GRID_SIZE, GRID_SIZE))
        self.grid_size = (int(gs[0]), int(gs[1]))

        self.tanks = {}
        raw_tanks = state.get("tanks", {})
        for pid in (1, 2):
            t = raw_tanks.get(pid, raw_tanks.get(str(pid), {}))
            hp = int(t.get("hp", MAX_TANK_HP))
            d_val = t.get("dir", "UP" if pid == 1 else "DOWN")
            direction = d_val if isinstance(d_val, Direction) else Direction(str(d_val))
            self.tanks[pid] = {
                "x": int(t.get("x", 2 if pid == 1 else 10)),
                "y": int(t.get("y", 10 if pid == 1 else 2)),
                "dir": direction,
                "hp": hp,
                "alive": bool(t.get("alive", hp > 0)),
            }

        self.bases = {}
        raw_bases = state.get("bases", {})
        for pid in (1, 2):
            b = raw_bases.get(pid, raw_bases.get(str(pid), {}))
            hp = int(b.get("hp", MAX_BASE_HP))
            self.bases[pid] = {
                "x": int(b.get("x", 6)),
                "y": int(b.get("y", 12 if pid == 1 else 0)),
                "hp": hp,
                "alive": bool(b.get("alive", hp > 0)),
            }

        self.walls = {}
        for w in state.get("walls", []):
            hp = int(w.get("hp", MAX_WALL_HP))
            if hp > 0:
                self.walls[(int(w["x"]), int(w["y"]))] = hp

        self.bullets = []
        for bul in state.get("bullets", []):
            d_val = bul.get("dir", "UP")
            direction = d_val if isinstance(d_val, Direction) else Direction(str(d_val))
            self.bullets.append(
                {
                    "x": int(bul["x"]),
                    "y": int(bul["y"]),
                    "dir": direction,
                    "owner": int(bul.get("owner", 1)),
                }
            )

        self.is_terminal_flag = bool(state.get("is_terminal", False))
        self.winner = state.get("winner", None)

    def clone(self) -> "TankGameEngine":
        """Fast deep copy of the engine for adversarial search tree expansion."""
        cloned = TankGameEngine.__new__(TankGameEngine)
        cloned.grid_size = self.grid_size
        cloned.max_turns = self.max_turns
        cloned.turn = self.turn
        cloned.tanks = {
            1: dict(self.tanks[1]),
            2: dict(self.tanks[2]),
        }
        cloned.bases = {
            1: dict(self.bases[1]),
            2: dict(self.bases[2]),
        }
        cloned.walls = dict(self.walls)
        cloned.bullets = [dict(b) for b in self.bullets]
        cloned.is_terminal_flag = self.is_terminal_flag
        cloned.winner = self.winner
        return cloned

    def get_state(self) -> Dict[str, Any]:
        """Return the current game state formatted for the frontend UI."""
        return {
            "turn": self.turn,
            "grid_size": self.grid_size,
            "tanks": {
                pid: {
                    "x": t["x"],
                    "y": t["y"],
                    "dir": t["dir"].value if isinstance(t["dir"], Direction) else str(t["dir"]),
                    "hp": t["hp"],
                    "alive": t["alive"],
                }
                for pid, t in self.tanks.items()
            },
            "bases": {
                pid: {
                    "x": b["x"],
                    "y": b["y"],
                    "hp": b["hp"],
                    "alive": b["alive"],
                }
                for pid, b in self.bases.items()
            },
            "walls": [
                {"x": wx, "y": wy, "hp": hp}
                for (wx, wy), hp in sorted(self.walls.items())
                if hp > 0
            ],
            "bullets": [
                {
                    "x": b["x"],
                    "y": b["y"],
                    "dir": b["dir"].value if isinstance(b["dir"], Direction) else str(b["dir"]),
                    "owner": b["owner"],
                }
                for b in self.bullets
            ],
            "is_terminal": self.is_terminal_flag,
            "winner": self.winner,
        }

    def player(self, ply_index: int = 1) -> int:
        """Return the active player (1 or 2) at the given ply."""
        return 1 if ply_index % 2 == 1 else 2

    def actions(self, player_id: int = 1) -> List[Action]:
        """Return candidate actions for the given player."""
        t = self.tanks.get(player_id)
        if not t or not t["alive"] or self.is_terminal_flag:
            return [Action.STAY]
        return list(ALL_ACTIONS)

    def result(
        self, action_p1: Action | str, action_p2: Action | str
    ) -> "TankGameEngine":
        """Return the successor state without mutating self."""
        return self.simulate_action(action_p1, action_p2)

    def terminal_test(self) -> bool:
        """Return True if the game has ended."""
        return self.is_terminal()

    def utility(self, player_id: int) -> float:
        """Return zero-sum terminal payoff (+1.0, -1.0, or 0.0) for player_id."""
        if self.winner == player_id:
            return 1.0
        if self.winner == (3 - player_id):
            return -1.0
        return 0.0

    def is_terminal(self) -> bool:
        return self.is_terminal_flag

    def get_legal_actions(self, player_id: int) -> List[Action]:
        """Return legal actions for `player_id` in canonical priority order."""
        t = self.tanks.get(player_id)
        if not t or not t["alive"] or self.is_terminal_flag:
            return [Action.STAY]

        legal: List[Action] = [Action.FIRE]
        tx, ty = t["x"], t["y"]
        opp_id = 3 - player_id
        opp_t = self.tanks[opp_id]
        opp_pos = (opp_t["x"], opp_t["y"]) if opp_t["alive"] else (-1, -1)
        base_positions = {
            (self.bases[1]["x"], self.bases[1]["y"]),
            (self.bases[2]["x"], self.bases[2]["y"]),
        }

        move_order = (
            (
                Action.MOVE_UP,
                Action.MOVE_RIGHT,
                Action.MOVE_LEFT,
                Action.MOVE_DOWN,
            )
            if player_id == 1
            else (
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
                Action.MOVE_UP,
            )
        )
        for act in move_order:
            d = ACTION_TO_DIR[act]
            dx, dy = DIR_DELTAS[d]
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                if (
                    (nx, ny) not in self.walls
                    and (nx, ny) not in base_positions
                    and (nx, ny) != opp_pos
                ):
                    legal.append(act)

        legal.append(Action.STAY)
        return legal

    @staticmethod
    def _coerce_action(act: Action | str) -> Action:
        if isinstance(act, Action):
            return act
        return Action(str(act))

    def _is_cell_blocked_for_tank(
        self, x: int, y: int, moving_pid: int
    ) -> bool:
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return True
        if (x, y) in self.walls:
            return True
        for b in self.bases.values():
            if b["alive"] and (b["x"], b["y"]) == (x, y):
                return True
        return False

    def step(
        self,
        action_p1: Action | str,
        action_p2: Action | str,
        record_frames: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute one simultaneous game cycle (turn) for Player 1 and Player 2:
        1. Process tank movement (1 cell) and facing direction updates.
        2. Spawn new bullets at the tank's coordinates for tanks choosing FIRE.
        3. Advance all active bullets up to BULLET_SPEED (3) single-cell sub-steps,
           resolving head-on swaps, open-cell bullet collisions, wall hits,
           and tank/base hits at each sub-step.
        """
        if self.is_terminal_flag:
            return self.get_state(), []

        a1 = self._coerce_action(action_p1)
        a2 = self._coerce_action(action_p2)
        actions = {1: a1, 2: a2}

        # Step 1: Tank movement & facing update
        desired_pos: Dict[int, Tuple[int, int]] = {}
        for pid in (1, 2):
            t = self.tanks[pid]
            if not t["alive"]:
                desired_pos[pid] = (t["x"], t["y"])
                continue
            act = actions[pid]
            if act in ACTION_TO_DIR:
                new_dir = ACTION_TO_DIR[act]
                t["dir"] = new_dir
                dx, dy = DIR_DELTAS[new_dir]
                nx, ny = t["x"] + dx, t["y"] + dy
                if not self._is_cell_blocked_for_tank(nx, ny, pid):
                    desired_pos[pid] = (nx, ny)
                else:
                    desired_pos[pid] = (t["x"], t["y"])
            else:
                desired_pos[pid] = (t["x"], t["y"])

        # Prevent two tanks from moving into the same cell or swapping cells head-on
        p1_old = (self.tanks[1]["x"], self.tanks[1]["y"])
        p2_old = (self.tanks[2]["x"], self.tanks[2]["y"])
        p1_new = desired_pos[1]
        p2_new = desired_pos[2]

        if p1_new == p2_new or (p1_new == p2_old and p2_new == p1_old):
            p1_new = p1_old
            p2_new = p2_old

        self.tanks[1]["x"], self.tanks[1]["y"] = p1_new
        self.tanks[2]["x"], self.tanks[2]["y"] = p2_new

        # Step 2: Spawn bullets for tanks executing FIRE
        for pid in (1, 2):
            t = self.tanks[pid]
            if t["alive"] and actions[pid] == Action.FIRE:
                self.bullets.append(
                    {
                        "x": t["x"],
                        "y": t["y"],
                        "dir": t["dir"],
                        "owner": pid,
                    }
                )

        # Step 3: Advance bullets cell-by-cell for BULLET_SPEED (3) sub-steps
        frames: List[Dict[str, Any]] = []
        for sub_step in range(BULLET_SPEED):
            if not self.bullets or self.is_terminal_flag:
                break

            old_positions = [(b["x"], b["y"]) for b in self.bullets]
            new_positions: List[Tuple[int, int]] = []
            for b in self.bullets:
                dx, dy = DIR_DELTAS[b["dir"]]
                new_positions.append((b["x"] + dx, b["y"] + dy))

            cancelled: Set[int] = set()
            n_bul = len(self.bullets)
            walls_at_substep_start = set(self.walls.keys())

            # Cancel opposing bullets meeting in an open cell or swapping head-on;
            # bullets entering a wall cell both damage the wall.
            for i in range(n_bul):
                for j in range(i + 1, n_bul):
                    if self.bullets[i]["owner"] == self.bullets[j]["owner"]:
                        continue
                    same_open_cell = (
                        new_positions[i] == new_positions[j]
                        and new_positions[i] not in walls_at_substep_start
                    )
                    head_on_swap = (
                        new_positions[i] == old_positions[j]
                        and new_positions[j] == old_positions[i]
                    )
                    if same_open_cell or head_on_swap:
                        cancelled.add(i)
                        cancelled.add(j)

            surviving_bullets: List[Dict[str, Any]] = []
            for idx, b in enumerate(self.bullets):
                if idx in cancelled:
                    continue

                nx, ny = new_positions[idx]
                # Out of bounds
                if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    continue

                # Wall collision
                if (nx, ny) in self.walls:
                    self.walls[(nx, ny)] -= BULLET_DAMAGE
                    if self.walls[(nx, ny)] <= 0:
                        del self.walls[(nx, ny)]
                    continue

                # Base collision
                hit_base = False
                for pid, base in self.bases.items():
                    if base["alive"] and (base["x"], base["y"]) == (nx, ny):
                        base["hp"] = max(0, base["hp"] - BULLET_DAMAGE)
                        if base["hp"] <= 0:
                            base["alive"] = False
                        hit_base = True
                        break
                if hit_base:
                    continue

                # Tank collision (only hits opposing tank)
                hit_tank = False
                for pid, tank in self.tanks.items():
                    if (
                        pid != b["owner"]
                        and tank["alive"]
                        and (tank["x"], tank["y"]) == (nx, ny)
                    ):
                        tank["hp"] = max(0, tank["hp"] - BULLET_DAMAGE)
                        if tank["hp"] <= 0:
                            tank["alive"] = False
                        hit_tank = True
                        break
                if hit_tank:
                    continue

                b["x"], b["y"] = nx, ny
                surviving_bullets.append(b)

            self.bullets = surviving_bullets
            self._check_terminal()

            if record_frames and sub_step < BULLET_SPEED - 1 and not self.is_terminal_flag:
                frames.append(self.get_state())

        self.turn += 1
        self._check_terminal()
        final_state = self.get_state()
        return final_state, frames

    def _check_terminal(self) -> None:
        """Evaluate terminal conditions and determine the game winner."""
        p1_dead = (not self.tanks[1]["alive"]) or (not self.bases[1]["alive"])
        p2_dead = (not self.tanks[2]["alive"]) or (not self.bases[2]["alive"])

        if p1_dead and not p2_dead:
            self.is_terminal_flag = True
            self.winner = 2
            return
        if p2_dead and not p1_dead:
            self.is_terminal_flag = True
            self.winner = 1
            return
        if p1_dead and p2_dead:
            self.is_terminal_flag = True
            p1_total = self.tanks[1]["hp"] + self.bases[1]["hp"]
            p2_total = self.tanks[2]["hp"] + self.bases[2]["hp"]
            if p1_total > p2_total:
                self.winner = 1
            elif p2_total > p1_total:
                self.winner = 2
            elif self.bases[1]["hp"] > self.bases[2]["hp"]:
                self.winner = 1
            elif self.bases[2]["hp"] > self.bases[1]["hp"]:
                self.winner = 2
            else:
                self.winner = "Draw"
            return

        if self.turn >= self.max_turns:
            self.is_terminal_flag = True
            p1_score = self.tanks[1]["hp"] * 2 + self.bases[1]["hp"]
            p2_score = self.tanks[2]["hp"] * 2 + self.bases[2]["hp"]
            if p1_score > p2_score:
                self.winner = 1
            elif p2_score > p1_score:
                self.winner = 2
            else:
                self.winner = "Draw"

    def simulate_action(
        self, action_p1: Action | str, action_p2: Action | str
    ) -> "TankGameEngine":
        """Return a new TankGameEngine after advancing one cycle without frame recording."""
        child = self.clone()
        child.step(action_p1, action_p2, record_frames=False)
        return child

    def state_key(
        self,
        turn_player: int,
        pending_action: Optional[Action] = None,
    ) -> Tuple:
        """Return an immutable hashable key uniquely identifying the ply state."""
        t1 = self.tanks[1]
        t2 = self.tanks[2]
        b1 = self.bases[1]
        b2 = self.bases[2]
        walls_tuple = tuple(sorted(self.walls.items()))
        bullets_tuple = tuple(
            sorted((b["x"], b["y"], b["dir"].value, b["owner"]) for b in self.bullets)
        )
        return (
            turn_player,
            pending_action.value if pending_action is not None else None,
            (t1["x"], t1["y"], t1["dir"].value, t1["hp"]),
            (t2["x"], t2["y"], t2["dir"].value, t2["hp"]),
            (b1["hp"], b2["hp"]),
            walls_tuple,
            bullets_tuple,
        )


def create_scenario(scenario_name: str = "standard_arena") -> TankGameEngine:
    """Create one of the four benchmark game scenarios by name."""
    name = (scenario_name or "standard_arena").strip().lower()
    if name == "standard_arena":
        return TankGameEngine()

    if name == "open_duel":
        return TankGameEngine(
            initial_state={
                "turn": 8,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 4, "y": 9, "dir": "UP", "hp": 4, "alive": True},
                    2: {"x": 4, "y": 3, "dir": "DOWN", "hp": 4, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [
                    {"x": 2, "y": 6, "hp": 2},
                    {"x": 6, "y": 6, "hp": 2},
                    {"x": 8, "y": 5, "hp": 3},
                    {"x": 8, "y": 7, "hp": 3},
                ],
                "bullets": [
                    {"x": 4, "y": 6, "dir": "DOWN", "owner": 2},
                ],
                "is_terminal": False,
                "winner": None,
            }
        )

    if name == "base_siege":
        return TankGameEngine(
            initial_state={
                "turn": 14,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 6, "y": 5, "dir": "UP", "hp": 5, "alive": True},
                    2: {"x": 6, "y": 7, "dir": "DOWN", "hp": 5, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 3, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 3, "alive": True},
                },
                "walls": [
                    {"x": 6, "y": 1, "hp": 1},
                    {"x": 5, "y": 1, "hp": 3},
                    {"x": 7, "y": 1, "hp": 3},
                    {"x": 6, "y": 11, "hp": 1},
                    {"x": 5, "y": 11, "hp": 3},
                    {"x": 7, "y": 11, "hp": 3},
                ],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )

    if name == "corridor_skirmish":
        walls = [
            {"x": 2, "y": 4, "hp": 3},
            {"x": 2, "y": 8, "hp": 3},
            {"x": 4, "y": 5, "hp": 3},
            {"x": 4, "y": 7, "hp": 3},
            {"x": 5, "y": 12, "hp": 3},
            {"x": 5, "y": 11, "hp": 3},
            {"x": 6, "y": 11, "hp": 3},
            {"x": 7, "y": 11, "hp": 3},
            {"x": 7, "y": 12, "hp": 3},
            {"x": 5, "y": 0, "hp": 3},
            {"x": 5, "y": 1, "hp": 3},
            {"x": 6, "y": 1, "hp": 3},
            {"x": 7, "y": 1, "hp": 3},
            {"x": 7, "y": 0, "hp": 3},
        ]
        return TankGameEngine(
            initial_state={
                "turn": 6,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 3, "y": 8, "dir": "UP", "hp": 4, "alive": True},
                    2: {"x": 3, "y": 4, "dir": "DOWN", "hp": 4, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": walls,
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )

    raise ValueError(f"Unknown scenario_name: {scenario_name}")
