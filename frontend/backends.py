"""Small integration boundary; no game physics or CSP solving lives in the UI.

See backend_config.example.json to override module names when teammates deliver
their code. Classes and payloads follow the team's PDF API contracts.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "tank_module": "Task2.tank_game_engine",
    "agents_module": "Task2.agents",
    "kakuro_module": "Task3.kakuro_csp",
}


def load_module(name):
    importlib.invalidate_caches()
    # Refresh discovers files added since launch without reloading modules that
    # may still own a running solver or existing Enum instances.
    return importlib.import_module(name)


class Backends:
    def __init__(self):
        self.tank = self.kakuro = self.action = self.agent_base = None
        self.agents = {}
        self.errors = {}
        self.refresh()

    def refresh(self):
        self.errors = {}
        config = dict(DEFAULTS)
        try:
            path = ROOT / "backend_config.json"
            if path.exists():
                config.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            self.errors["config"] = f"backend_config.json: {exc}"
        self.config = config
        try:
            module = load_module(config["tank_module"])
            self.tank = getattr(module, "TankGameEngine")
            self.action = getattr(module, "Action")
        except Exception as exc:
            self.tank = self.action = None
            self.errors["tank"] = str(exc)
        try:
            module = load_module(config["agents_module"])
            self.agent_base = getattr(module, "BaseTankAgent")
            self.agents = {
                name: cls for name, cls in vars(module).items()
                if inspect.isclass(cls) and cls is not self.agent_base
                and issubclass(cls, self.agent_base) and not inspect.isabstract(cls)
            }
        except Exception as exc:
            self.agent_base = None
            self.agents = {}
            self.errors["agents"] = str(exc)
        try:
            self.kakuro = getattr(load_module(config["kakuro_module"]), "KakuroCSP")
        except Exception as exc:
            self.kakuro = None
            self.errors["kakuro"] = str(exc)

    def import_bot(self, path):
        if self.agent_base is None:
            raise RuntimeError("The agents module must define BaseTankAgent before importing bots.")
        path = Path(path).resolve()
        if path.suffix.lower() != ".py" or not path.is_file():
            raise ValueError("Choose an existing .py file.")
        name = f"external_bot_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("Unable to read the bot module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            found = {
                cls_name: cls for cls_name, cls in vars(module).items()
                if inspect.isclass(cls) and cls is not self.agent_base
                and issubclass(cls, self.agent_base)
                and cls.__module__ == name and not inspect.isabstract(cls)
            }
            if not found:
                raise ValueError("The file must define a class inheriting from BaseTankAgent.")
            self.agents.update(found)
            return list(found)
        except Exception:
            sys.modules.pop(name, None)
            raise


def preview_tank():
    """Static layout only: never presented as an actual engine result."""
    return {
        "turn": 0, "grid_size": (13, 13),
        "tanks": {1: {"x": 2, "y": 10, "dir": "UP", "hp": 6, "alive": True},
                  2: {"x": 10, "y": 2, "dir": "DOWN", "hp": 6, "alive": True}},
        "bases": {1: {"x": 6, "y": 12, "hp": 9, "alive": True},
                  2: {"x": 6, "y": 0, "hp": 9, "alive": True}},
        "walls": [{"x": x, "y": y, "hp": 3}
                  for x, y in [(3, 3), (4, 3), (8, 3), (9, 3), (6, 5),
                               (5, 6), (6, 6), (7, 6), (6, 7), (3, 9), (4, 9), (8, 9), (9, 9)]],
        "bullets": [], "is_terminal": False, "winner": None,
    }


def preview_kakuro():
    cells = []
    for r in range(5):
        for c in range(5):
            cell = {"r": r, "c": c, "type": "BLOCK"}
            if 1 <= r <= 3 and 1 <= c <= 3:
                cell["type"] = "WHITE"
            elif r == 0 and 1 <= c <= 3:
                cell.update(type="CLUE", down_sum=6, right_sum=None)
            elif c == 0 and 1 <= r <= 3:
                cell.update(type="CLUE", down_sum=None, right_sum=6)
            cells.append(cell)
    return {"rows": 5, "cols": 5, "cells": cells}
