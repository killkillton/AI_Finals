from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

from pathlib import Path
import time
import unittest
import pygame

from Task2.agents import (
    AlphaBetaAgent,
    BaseTankAgent,
    HeuristicAlphaBetaAgent,
    HeuristicMinimaxAgent,
    MinimaxAgent,
    get_agent_by_name,
    load_external_agent,
)
from Task2.heuristics import (
    evaluate_basic,
    evaluate_tactical,
    evaluate_terminal,
)
from Task2.tank_game_engine import (
    BASE_INITIAL_HP,
    BULLET_SPEED,
    DAMAGE_PER_HIT,
    DEFAULT_WALL_COORDS,
    TANK_INITIAL_HP,
    WALL_INITIAL_HP,
    Action,
    TankGameEngine,
    create_scenario,
)
from frontend.app import App
from frontend.backends import Backends


class Task2EngineAndSearchTests(unittest.TestCase):
    """Unit and integration tests for Task 2: Adversarial Search (Tank Battle Game)."""

    def test_entity_initial_health_and_37_wall_battle_city_arena(self):
        self.assertEqual(TANK_INITIAL_HP, 6)
        self.assertEqual(BASE_INITIAL_HP, 9)
        self.assertEqual(WALL_INITIAL_HP, 3)
        self.assertEqual(DAMAGE_PER_HIT, 1)
        self.assertEqual(BULLET_SPEED, 3)
        self.assertEqual(len(DEFAULT_WALL_COORDS), 37)
        self.assertIn((0, 5), DEFAULT_WALL_COORDS)
        self.assertIn((12, 7), DEFAULT_WALL_COORDS)

        engine = TankGameEngine()
        state = engine.reset()
        self.assertEqual(state["tanks"][1]["hp"], 6)
        self.assertEqual(state["tanks"][2]["hp"], 6)
        self.assertEqual(state["bases"][1]["hp"], 9)
        self.assertEqual(state["bases"][2]["hp"], 9)
        self.assertEqual(len(state["walls"]), 37)
        for wall in state["walls"]:
            self.assertEqual(wall["hp"], 3)

    def test_lecture_07_formal_methods(self):
        engine = TankGameEngine()
        self.assertEqual(engine.player(1), 1)
        self.assertEqual(engine.player(2), 2)
        self.assertEqual(len(engine.actions(1)), 6)
        self.assertFalse(engine.terminal_test())
        next_engine = engine.result(Action.MOVE_UP, Action.MOVE_DOWN)
        self.assertEqual(engine.turn, 0)
        self.assertEqual(next_engine.turn, 1)

    def test_six_tank_actions_movement_and_blocking(self):
        engine = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 2, "y": 2, "dir": "UP", "hp": 6, "alive": True},
                    2: {"x": 10, "y": 10, "dir": "DOWN", "hp": 6, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [{"x": 2, "y": 1, "hp": 3}],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )

        # Moving UP into a wall at (2, 1) keeps tank 1 at (2, 2) facing UP
        state, _ = engine.step(Action.MOVE_UP, Action.STAY)
        self.assertEqual((state["tanks"][1]["x"], state["tanks"][1]["y"]), (2, 2))
        self.assertEqual(state["tanks"][1]["dir"], "UP")

        # Moving RIGHT, DOWN, LEFT updates coordinates by 1 cell and updates direction
        state, _ = engine.step(Action.MOVE_RIGHT, Action.STAY)
        self.assertEqual((state["tanks"][1]["x"], state["tanks"][1]["y"]), (3, 2))
        self.assertEqual(state["tanks"][1]["dir"], "RIGHT")

        state, _ = engine.step(Action.MOVE_DOWN, Action.STAY)
        self.assertEqual((state["tanks"][1]["x"], state["tanks"][1]["y"]), (3, 3))
        self.assertEqual(state["tanks"][1]["dir"], "DOWN")

        state, _ = engine.step(Action.MOVE_LEFT, Action.STAY)
        self.assertEqual((state["tanks"][1]["x"], state["tanks"][1]["y"]), (2, 3))
        self.assertEqual(state["tanks"][1]["dir"], "LEFT")

    def test_bullet_cell_by_cell_movement_and_wall_destruction(self):
        engine = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 2, "y": 8, "dir": "UP", "hp": 6, "alive": True},
                    2: {"x": 10, "y": 2, "dir": "DOWN", "hp": 6, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [{"x": 2, "y": 6, "hp": 3}],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )

        # Tank 1 at (2, 8) fires UP. Sub-step 1 moves to (2, 7); Sub-step 2 hits wall at (2, 6) and stops immediately.
        state, sub_ticks = engine.step(Action.FIRE, Action.STAY)
        self.assertEqual(len(state["bullets"]), 0)
        self.assertEqual(len(state["walls"]), 1)
        self.assertEqual(state["walls"][0]["hp"], 2)
        self.assertTrue(len(sub_ticks) >= 1)

        # Two more shots destroy the wall completely (HP: 2 -> 1 -> 0)
        engine.step(Action.FIRE, Action.STAY)
        state, _ = engine.step(Action.FIRE, Action.STAY)
        self.assertEqual(len(state["walls"]), 0)

        # Next shot travels unobstructed 3 cells from (2, 8) to (2, 5)
        state, sub_ticks = engine.step(Action.FIRE, Action.STAY)
        self.assertEqual(len(state["bullets"]), 1)
        self.assertEqual((state["bullets"][0]["x"], state["bullets"][0]["y"]), (2, 5))
        self.assertEqual(len(sub_ticks), 2)
        self.assertEqual((sub_ticks[0]["bullets"][0]["x"], sub_ticks[0]["bullets"][0]["y"]), (2, 7))
        self.assertEqual((sub_ticks[1]["bullets"][0]["x"], sub_ticks[1]["bullets"][0]["y"]), (2, 6))

    def test_opposing_bullets_entering_wall_damage_wall_not_cancel(self):
        # Verify fix for Immortal Wall (6, 6) bug: two opposing bullets entering
        # a wall cell on the same sub-step both damage the wall (HP: 3 -> 1) instead
        # of cancelling each other out in mid-air.
        engine = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 4, "y": 6, "dir": "RIGHT", "hp": 6, "alive": True},
                    2: {"x": 8, "y": 6, "dir": "LEFT", "hp": 6, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [{"x": 6, "y": 6, "hp": 3}],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        # Sub-step 1: bullets move to (5, 6) and (7, 6).
        # Sub-step 2: both bullets enter wall (6, 6) and deal 1 damage each (HP: 3 -> 1).
        state, _ = engine.step(Action.FIRE, Action.FIRE)
        self.assertEqual(len(state["bullets"]), 0)
        self.assertEqual(engine.walls.get((6, 6)), 1)

        # Next shot destroys (6, 6) completely
        state2, _ = engine.step(Action.FIRE, Action.STAY)
        self.assertNotIn((6, 6), engine.walls)

    def test_terminal_condition_when_tank_or_base_destroyed(self):
        # 1. Tank destruction ends game immediately
        engine = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 5, "y": 5, "dir": "RIGHT", "hp": 6, "alive": True},
                    2: {"x": 7, "y": 5, "dir": "LEFT", "hp": 1, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        state, _ = engine.step(Action.FIRE, Action.STAY)
        self.assertTrue(state["is_terminal"])
        self.assertEqual(state["winner"], 1)
        self.assertEqual(state["tanks"][2]["hp"], 0)
        self.assertFalse(state["tanks"][2]["alive"])

        # 2. Base destruction ends game immediately
        engine_base = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 2, "y": 10, "dir": "UP", "hp": 6, "alive": True},
                    2: {"x": 6, "y": 9, "dir": "DOWN", "hp": 6, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 1, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        state_base, _ = engine_base.step(Action.STAY, Action.FIRE)
        self.assertTrue(state_base["is_terminal"])
        self.assertEqual(state_base["winner"], 2)
        self.assertEqual(state_base["bases"][1]["hp"], 0)
        self.assertFalse(state_base["bases"][1]["alive"])

    def test_clone_isolation(self):
        engine = TankGameEngine()
        cloned = engine.clone()
        cloned.step(Action.MOVE_UP, Action.FIRE)
        orig_state = engine.get_state()
        self.assertEqual(orig_state["turn"], 0)
        self.assertEqual((orig_state["tanks"][1]["x"], orig_state["tanks"][1]["y"]), (2, 10))
        self.assertEqual(len(orig_state["bullets"]), 0)

    def test_strict_zero_sum_symmetry_across_all_scenarios(self):
        # Verify EVAL(s, 1) == -EVAL(s, 2) for both evaluate_basic and evaluate_tactical
        for scenario_name in ("standard_arena", "open_duel", "base_siege", "corridor_skirmish"):
            with self.subTest(scenario=scenario_name):
                eng = create_scenario(scenario_name)
                self.assertAlmostEqual(
                    evaluate_basic(eng, 1) + evaluate_basic(eng, 2), 0.0, places=9
                )
                self.assertAlmostEqual(
                    evaluate_tactical(eng, 1) + evaluate_tactical(eng, 2), 0.0, places=9
                )
                # Also verify after a joint step with active bullets
                eng.step(Action.FIRE, Action.FIRE)
                self.assertAlmostEqual(
                    evaluate_basic(eng, 1) + evaluate_basic(eng, 2), 0.0, places=9
                )
                self.assertAlmostEqual(
                    evaluate_tactical(eng, 1) + evaluate_tactical(eng, 2), 0.0, places=9
                )

    def test_wall_breach_increases_score_and_friendly_base_shield_preserved(self):
        # 1. Destroying a 1-HP wall blocking Player 1's line of sight to Player 2's
        # tank/base must strictly INCREASE evaluate_tactical for Player 1.
        eng_blocked = TankGameEngine(
            initial_state={
                "turn": 5,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 6, "y": 5, "dir": "UP", "hp": 6, "alive": True},
                    2: {"x": 10, "y": 4, "dir": "LEFT", "hp": 6, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [{"x": 6, "y": 2, "hp": 1}],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        eng_cleared = eng_blocked.clone()
        del eng_cleared.walls[(6, 2)]
        self.assertGreater(
            evaluate_tactical(eng_cleared, 1),
            evaluate_tactical(eng_blocked, 1),
        )

        # 2. Damaging a friendly base shield wall (e.g. (6, 11) for Player 1) must
        # strictly DECREASE evaluate_tactical for Player 1.
        eng_full_shield = TankGameEngine()
        eng_damaged_shield = eng_full_shield.clone()
        eng_damaged_shield.walls[(6, 11)] = 2
        self.assertLess(
            evaluate_tactical(eng_damaged_shield, 1),
            evaluate_tactical(eng_full_shield, 1),
        )

    def test_minimax_vs_alphabeta_exact_equivalence_and_pruning(self):
        # Lecture 07 Slide 25 Invariant: Pruning does not affect the result
        for scenario_name in ("standard_arena", "open_duel", "base_siege", "corridor_skirmish"):
            with self.subTest(scenario=scenario_name):
                engine = create_scenario(scenario_name)

                mm = MinimaxAgent(player_id=1, max_depth=3)
                ab = AlphaBetaAgent(player_id=1, max_depth=3)
                act_mm, met_mm = mm.get_action(engine.clone())
                act_ab, met_ab = ab.get_action(engine.clone())

                self.assertEqual(met_mm["best_score"], met_ab["best_score"])
                self.assertEqual(act_mm, act_ab)
                self.assertLess(met_ab["explored_states"], met_mm["explored_states"])
                self.assertGreater(met_ab["pruning_rate"], 0.0)

                hmm = HeuristicMinimaxAgent(player_id=1, max_depth=3)
                hab = HeuristicAlphaBetaAgent(player_id=1, max_depth=3)
                act_hmm, met_hmm = hmm.get_action(engine.clone())
                act_hab, met_hab = hab.get_action(engine.clone())

                self.assertEqual(met_hmm["best_score"], met_hab["best_score"])
                self.assertEqual(act_hmm, act_hab)
                self.assertLess(met_hab["explored_states"], met_hmm["explored_states"])
                self.assertLess(met_hab["explored_states"], met_ab["explored_states"])
                self.assertGreater(met_hab["pruning_rate"], met_ab["pruning_rate"])
                self.assertGreater(met_hab["pruning_rate"], 50.0)
                self.assertGreater(met_hab["tt_hits"], 0)

    def test_no_stuttering_penalty_when_firing_at_target(self):
        engine = create_scenario("base_siege")
        agent = HeuristicAlphaBetaAgent(player_id=1, max_depth=3)
        # Simulate multiple turns at the same cell while aiming at wall/base
        for _ in range(4):
            agent._update_history(engine)
        self.assertTrue(agent._has_valid_fire_target(engine))
        self.assertEqual(agent._record_position_and_get_penalty(engine, Action.FIRE), 0.0)

    def test_heuristic_agents_dodge_incoming_bullet(self):
        # In open_duel, a bullet is at (4, 6) moving DOWN (3 cells/cycle -> reaches (4, 9) this turn)
        # where Player 1's tank sits at (4, 9). Both basic and heuristic agents at depth 3 must dodge.
        engine = create_scenario("open_duel")
        for agent_cls in (MinimaxAgent, AlphaBetaAgent, HeuristicMinimaxAgent, HeuristicAlphaBetaAgent):
            with self.subTest(agent=agent_cls.__name__):
                agent = agent_cls(player_id=1, max_depth=3)
                action, _ = agent.get_action(engine.clone())
                self.assertIn(action, (Action.MOVE_LEFT, Action.MOVE_RIGHT))

    def test_head_on_and_same_cell_bullet_cancellation(self):
        # 1. Head-on swap cancellation: Tank 1 at (3, 5) facing RIGHT, Tank 2 at (6, 5) facing LEFT (distance 3).
        engine_swap = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 3, "y": 5, "dir": "RIGHT", "hp": 1, "alive": True},
                    2: {"x": 6, "y": 5, "dir": "LEFT", "hp": 1, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        state_swap, _ = engine_swap.step(Action.FIRE, Action.FIRE)
        self.assertEqual(len(state_swap["bullets"]), 0)
        self.assertEqual(state_swap["tanks"][1]["hp"], 1)
        self.assertEqual(state_swap["tanks"][2]["hp"], 1)
        self.assertFalse(state_swap["is_terminal"])

        # 2. Same-cell impact cancellation: Tank 1 at (3, 5) facing RIGHT, Tank 2 at (7, 5) facing LEFT (distance 4).
        engine_same = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 3, "y": 5, "dir": "RIGHT", "hp": 1, "alive": True},
                    2: {"x": 7, "y": 5, "dir": "LEFT", "hp": 1, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 9, "alive": True},
                },
                "walls": [],
                "bullets": [],
                "is_terminal": False,
                "winner": None,
            }
        )
        state_same, _ = engine_same.step(Action.FIRE, Action.FIRE)
        self.assertEqual(len(state_same["bullets"]), 0)
        self.assertEqual(state_same["tanks"][1]["hp"], 1)
        self.assertEqual(state_same["tanks"][2]["hp"], 1)
        self.assertFalse(state_same["is_terminal"])

    def test_simultaneous_substep_impacts_and_clear_winner_tiebreak(self):
        # Two bullets on separate lanes both reach opposing tanks on the same sub-step.
        # Both impacts apply in that sub-step; Player 1 has higher Base HP (9 vs 7) so Player 1 wins.
        engine = TankGameEngine(
            initial_state={
                "turn": 0,
                "grid_size": (13, 13),
                "tanks": {
                    1: {"x": 2, "y": 8, "dir": "UP", "hp": 1, "alive": True},
                    2: {"x": 10, "y": 4, "dir": "DOWN", "hp": 1, "alive": True},
                },
                "bases": {
                    1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                    2: {"x": 6, "y": 0, "hp": 7, "alive": True},
                },
                "walls": [],
                "bullets": [
                    {"x": 10, "y": 6, "dir": "UP", "owner": 1},
                    {"x": 2, "y": 6, "dir": "DOWN", "owner": 2},
                ],
                "is_terminal": False,
                "winner": None,
            }
        )
        state, _ = engine.step(Action.STAY, Action.STAY)
        self.assertTrue(state["is_terminal"])
        self.assertEqual(state["winner"], 1)
        self.assertFalse(state["tanks"][1]["alive"])
        self.assertFalse(state["tanks"][2]["alive"])

    def test_depth3_minimax_and_alphabeta_fire_when_aligned(self):
        # When aligned on a clear 1-cell-wide corridor at distance 4 (beyond 1-cycle 3-cell bullet travel),
        # depth-3 MinimaxAgent and AlphaBetaAgent must actively choose FIRE rather than walking passively.
        walls = (
            [{"x": 2, "y": y, "hp": 3} for y in range(2, 11)]
            + [{"x": 4, "y": y, "hp": 3} for y in range(2, 11)]
        )
        engine = TankGameEngine(
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
        for pid in (1, 2):
            mm = MinimaxAgent(player_id=pid, max_depth=3)
            ab = AlphaBetaAgent(player_id=pid, max_depth=3)
            act_mm, _ = mm.get_action(engine.clone())
            act_ab, _ = ab.get_action(engine.clone())
            self.assertEqual(act_mm, Action.FIRE)
            self.assertEqual(act_ab, Action.FIRE)

        # Also verify that in a live match on the 37-wall standard arena at depth=3,
        # HeuristicAlphaBetaAgent vs MinimaxAgent actively fires shots and engages in combat.
        live_engine = TankGameEngine(max_turns=30)
        p1 = HeuristicAlphaBetaAgent(player_id=1, max_depth=3)
        p2 = MinimaxAgent(player_id=2, max_depth=3)
        fired_count = 0
        initial_wall_hp = sum(live_engine.walls.values())
        for _ in range(25):
            if live_engine.terminal:
                break
            a1, _ = p1.get_action(live_engine.clone())
            a2, _ = p2.get_action(live_engine.clone())
            if a1 == Action.FIRE:
                fired_count += 1
            if a2 == Action.FIRE:
                fired_count += 1
            live_engine.step(a1, a2, record_frames=False)
        self.assertGreater(fired_count, 0)
        self.assertLess(sum(live_engine.walls.values()), initial_wall_hp)

    def test_academic_integrity_and_pure_ascii_compliance(self):
        task2_dir = Path(__file__).resolve().parent.parent
        forbidden_substrings = (
            "Slide 33",
            "Contract 2",
            "LEGACY_PREVIEW_WALL_COORDS",
            "simulate_Action",
            "facing_Bonus",
            "killer_Action",
            "Player None",
            "\u2014",
            "\u2013",
            "\u2212",
            "\u2026",
        )
        for path in task2_dir.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".ipynb"):
                if path.name == "test_task2.py":
                    continue
                content = path.read_text(encoding="utf-8")
                for bad in forbidden_substrings:
                    self.assertNotIn(
                        bad,
                        content,
                        f"Forbidden substring {bad!r} found in {path}",
                    )

    def test_external_bot_loader_and_registry(self):
        bot_path = Path(__file__).resolve().parent.parent / "external_bots" / "sample_group_bot.py"
        loaded = load_external_agent(bot_path)
        self.assertIn("ExternalGroupBot", loaded)
        bot = loaded["ExternalGroupBot"](player_id=2, max_depth=2)
        act, metrics = bot.get_action(TankGameEngine())
        self.assertIsInstance(act, Action)
        self.assertIn("time_ms", metrics)
        self.assertIn("explored_states", metrics)

        for name in ("Minimax", "Alpha-Beta", "Heuristic Minimax", "Heuristic Alpha-Beta"):
            agent = get_agent_by_name(name, player_id=1, max_depth=2)
            self.assertIsInstance(agent, BaseTankAgent)

    def test_frontend_app_seamless_connection_with_task2(self):
        backends = Backends()
        self.assertIs(backends.tank, TankGameEngine)
        self.assertIs(backends.action, Action)
        self.assertIs(backends.agent_base, BaseTankAgent)
        self.assertEqual(
            set(backends.agents.keys()),
            {
                "MinimaxAgent",
                "AlphaBetaAgent",
                "HeuristicMinimaxAgent",
                "HeuristicAlphaBetaAgent",
            },
        )

        app = App()
        try:
            app.open_task(1)
            self.assertIsNotNone(app.tank_engine)
            self.assertEqual(app.tank_state["turn"], 0)
            self.assertEqual(len(app.tank_state["walls"]), 37)

            # Select HeuristicAlphaBetaAgent for both players and step a turn
            app.bot_names = ["HeuristicAlphaBetaAgent", "HeuristicAlphaBetaAgent"]
            app.depth = 2
            app.tank_step()
            deadline = time.monotonic() + 5.0
            while app.busy and time.monotonic() < deadline:
                app.receive_jobs()
                time.sleep(0.005)
            self.assertFalse(app.busy)
            self.assertTrue(len(app.tank_frames) >= 1)
            self.assertEqual(set(app.tank_metrics.keys()), {1, 2})

            while app.tank_frames:
                app.last_tank_tick = 0
                app.update()
                app.draw()
            self.assertEqual(app.tank_state["turn"], 1)

            # Test importing external student group bot via UI hook
            bot_path = Path(__file__).resolve().parent.parent / "external_bots" / "sample_group_bot.py"
            app.import_bot(str(bot_path))
            self.assertIn("ExternalGroupBot", app.backends.agents)
        finally:
            pygame.quit()

    def test_experiments_artifacts_and_notebook_outputs_exist(self):
        import csv
        import json

        task2_dir = Path(__file__).resolve().parent.parent
        results_dir = task2_dir / "experiments" / "results"
        expected_csvs = [
            "search_performance_by_depth.csv",
            "scenario_comparison_depth3.csv",
            "tournament_matchups.csv",
            "tournament_summary.csv",
        ]
        for csv_name in expected_csvs:
            path = results_dir / csv_name
            self.assertTrue(path.is_file(), f"Missing benchmark CSV: {path}")
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                self.assertGreater(len(rows), 0)

        expected_pngs = [
            "execution_time_comparison.png",
            "explored_states_and_pruning.png",
            "tournament_win_rates_and_hp.png",
        ]
        for png_name in expected_pngs:
            path = results_dir / png_name
            self.assertTrue(path.is_file(), f"Missing benchmark chart: {path}")
            self.assertGreater(path.stat().st_size, 1000)

        nb_path = task2_dir / "Task2_Experiments_and_Formulation.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing executed Jupyter Notebook")
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb_data.get("nbformat"), 4)
        code_cells = [c for c in nb_data.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            self.assertIsNotNone(cell.get("execution_count"))
            self.assertGreater(len(cell.get("outputs", [])), 0)


if __name__ == "__main__":
    unittest.main()
