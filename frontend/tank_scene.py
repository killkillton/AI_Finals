"""Bright, rounded arcade illustration inspired by the user's tank reference.

All artwork is native PyGame drawing, so it scales without external assets.
Game state and physics are still supplied exclusively by the engine.
"""
import math
import random
import time

import pygame


GOLD = "#FFD15C"
CREAM = "#FFF8E6"
GREEN = "#56A64D"
RED = "#DF685B"


class TankScene:
    COLORS = {
        "#ECF2FA": "#4C5839", "#94A6BD": "#7F805D",
        "#57E2C0": "#98701F", "#B8A1FF": RED,
        "#0C1220": "#FFFDF4", "#151F30": CREAM,
        "#1D2A3D": "#E4EDCC", "#2B3A51": "#D6C499",
        "#FFBD73": "#AB7038", "#76ABFF": "#709361",
        "#70BE52": "#438540", "#E35D4E": "#BA5149",
    }

    def __init__(self):
        self.background = None
        self.background_size = None
        self.sprite_cache = {}
        self.grid_cache = {}
        self.base_cache = {}
        self.health_visuals = {}

    def reset_health(self):
        self.health_visuals.clear()

    def health_bar(self, app, rect, hp, maximum, key):
        """Render real HP with a short eased loss animation and damage flash."""
        rect = pygame.Rect(rect)
        now = time.monotonic()
        hp = max(0, min(maximum, hp))
        target = hp / maximum
        visual = self.health_visuals.setdefault(key, {"hp": hp, "fill": target, "time": now, "flash": 0})
        elapsed = min(.1, now - visual["time"])
        if hp < visual["hp"]:
            visual["flash"] = now + .35
        elif hp > visual["hp"]:
            visual["fill"] = target
        visual["hp"], visual["time"] = hp, now
        visual["fill"] = max(target, visual["fill"] - elapsed * .8)
        color = "#79B85A" if target > .5 else "#ECC04F" if target > .25 else "#E37661"
        pygame.draw.rect(app.screen, "#DADFC7", rect, border_radius=max(2, rect.h // 2))
        inner = rect.inflate(-2, -2)
        if inner.w > 0 and inner.h > 0:
            # Pale trail makes the amount lost on the latest hit easy to see.
            trailing = inner.copy()
            trailing.w = round(inner.w * visual["fill"])
            pygame.draw.rect(app.screen, "#EFAD82", trailing, border_radius=max(1, inner.h // 2))
            fill = inner.copy()
            fill.w = round(inner.w * target)
            if fill.w:
                pygame.draw.rect(app.screen, color, fill, border_radius=max(1, inner.h // 2))
        edge = "#E57C63" if now < visual["flash"] else "#A8B791"
        pygame.draw.rect(app.screen, edge, rect, 1, border_radius=max(2, rect.h // 2))

    def base_sprite(self, unit, player):
        size = max(8, round(unit * .85))
        key = (size, player)
        if key not in self.base_cache:
            image = pygame.Surface((160, 160), pygame.SRCALPHA)
            team = GREEN if player == 1 else RED
            shade = "#42803F" if player == 1 else "#B84F46"
            pygame.draw.ellipse(image, (75, 98, 57, 50), (9, 137, 143, 16))
            pygame.draw.rect(image, "#BAAF91", (15, 130, 130, 17), border_radius=7)
            pygame.draw.rect(image, "#E4D8B9", (19, 127, 122, 13), border_radius=5)
            # Main hall, twin towers, tiled roofs and a central arched gate.
            pygame.draw.rect(image, "#BCAA84", (41, 62, 78, 72), border_radius=7)
            pygame.draw.rect(image, "#FFF1D0", (45, 60, 70, 72), border_radius=5)
            for x in (17, 108):
                pygame.draw.rect(image, "#BCAA84", (x, 52, 36, 85), border_radius=5)
                pygame.draw.rect(image, "#F5E7C6", (x + 4, 51, 28, 83), border_radius=4)
                for y in (77, 103, 126):
                    pygame.draw.line(image, "#D7C5A1", (x + 5, y), (x + 31, y), 3)
                pygame.draw.polygon(image, shade, [(x - 5, 56), (x + 18, 26), (x + 41, 56)])
                pygame.draw.polygon(image, team, [(x - 5, 52), (x + 18, 22), (x + 41, 52)])
                pygame.draw.line(image, "#FFE3A4", (x + 3, 48), (x + 17, 29), 3)
                pygame.draw.rect(image, "#7A9D9B", (x + 12, 64, 13, 22), border_radius=6)
                pygame.draw.line(image, "#FFF6D9", (x + 18, 66), (x + 18, 82), 2)
            pygame.draw.polygon(image, shade, [(36, 67), (80, 39), (124, 67)])
            pygame.draw.polygon(image, team, [(36, 62), (80, 34), (124, 62)])
            pygame.draw.rect(image, "#B7A378", (66, 98, 29, 36), border_radius=14)
            pygame.draw.rect(image, "#795D3D", (70, 102, 21, 32), border_top_left_radius=11, border_top_right_radius=11)
            pygame.draw.line(image, "#B9945C", (80, 108), (80, 131), 2)
            pygame.draw.circle(image, "#F6D67A", (85, 120), 2)
            pygame.draw.circle(image, "#F7D98B", (80, 81), 12)
            pygame.draw.circle(image, team, (80, 81), 8)
            pygame.draw.line(image, "#A88C5D", (80, 8), (80, 38), 4)
            pygame.draw.polygon(image, team, [(82, 8), (110, 15), (82, 24)])
            self.base_cache[key] = pygame.transform.smoothscale(image, (size, size))
            if len(self.base_cache) > 30:
                self.base_cache = {key: self.base_cache[key]}
        return self.base_cache[key]

    def color(self, color):
        return self.COLORS.get(color, color)

    def panel(self, app, rect, color, radius, border):
        rect = pygame.Rect(rect)
        if color != "#151F30":
            pygame.draw.rect(app.screen, self.color(color), rect, border_radius=radius)
            if border:
                pygame.draw.rect(app.screen, self.color(border), rect, 1, border_radius=radius)
            return rect
        pygame.draw.rect(app.screen, "#B5C397", rect.move(0, 4), border_radius=18)
        pygame.draw.rect(app.screen, CREAM, rect, border_radius=18)
        pygame.draw.rect(app.screen, "#D7BF87", rect, 2, border_radius=18)
        pygame.draw.line(app.screen, "#FFE3A1", (rect.x + 18, rect.y + 3), (rect.right - 18, rect.y + 3), 2)
        return rect

    def button(self, app, label, rect, action, primary, enabled, selected):
        rect = pygame.Rect(rect)
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        fill = GOLD if primary and enabled else "#DFEDBC" if hover else "#EBF1D8" if enabled else "#EEEBDC"
        border = "#D8AC45" if primary or selected else "#B5C68F" if enabled else "#D5D0B9"
        pygame.draw.rect(app.screen, "#D2CBA9", rect.move(0, 3), border_radius=12)
        pygame.draw.rect(app.screen, fill, rect, border_radius=12)
        pygame.draw.rect(app.screen, border, rect, 2 if primary else 1, border_radius=12)
        font = app.font(16, primary)
        while font.size(label)[0] > rect.w - 37 and len(label) > 2:
            label = label[:-2] + "…" if label.endswith("…") else label[:-1] + "…"
        color = "#655024" if primary and enabled else "#4E633D" if enabled else "#9A947B"
        image = font.render(label, True, color)
        app.screen.blit(image, image.get_rect(center=rect.center))
        if enabled:
            app.buttons.append((rect, action))

    @staticmethod
    def crosshair(screen, center, radius, color, width=2):
        cx, cy = center
        pygame.draw.circle(screen, color, center, radius, width)
        pygame.draw.circle(screen, color, center, max(3, int(radius * .23)), width)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            pygame.draw.line(screen, color, (cx + dx * radius * .42, cy + dy * radius * .42),
                             (cx + dx * radius, cy + dy * radius), width)

    def backdrop(self, screen):
        width, height = screen.get_size()
        if self.background_size != (width, height):
            backdrop = pygame.Surface((width, height))
            for y in range(height):
                t = y / height
                pygame.draw.line(backdrop, (203 + int(39 * t), 232 - int(2 * t), 229 - int(44 * t)), (0, y), (width, y))
            rng = random.Random(72)
            for _ in range(9):
                x, y = rng.randrange(width), rng.randrange(int(height * .65))
                for dx, dy, radius in [(-28, 6, 24), (0, -4, 32), (30, 6, 24)]:
                    pygame.draw.circle(backdrop, "#F5FBED", (x + dx, y + dy), radius)
            pygame.draw.circle(backdrop, "#FFF2B7", (width - 78, 57), 43)
            pygame.draw.circle(backdrop, GOLD, (width - 78, 57), 33)
            for index, color in enumerate(["#CDDC9F", "#B9D48B", "#A9C878"]):
                floor = height - 110 + index * 40
                points = [(0, height)] + [(x, floor - int(26 * math.sin(x / 93 + index) + 16 * math.cos(x / 41)))
                                          for x in range(0, width + 30, 20)] + [(width, height)]
                pygame.draw.polygon(backdrop, color, points)
            self.background, self.background_size = backdrop, (width, height)
        screen.blit(self.background, (0, 0))

    @staticmethod
    def tank_art(player, direction="RIGHT"):
        """Chunky side-view toy tank matching the supplied illustration."""
        image = pygame.Surface((180, 118), pygame.SRCALPHA)
        body = "#57A746" if player == 1 else "#DF5E51"
        light = "#ABDF87" if player == 1 else "#FFAA91"
        dark = "#387A33" if player == 1 else "#AE423B"
        pygame.draw.ellipse(image, (75, 98, 57, 45), (14, 101, 142, 14))
        pygame.draw.rect(image, "#65776A", (13, 84, 140, 27), border_radius=13)
        pygame.draw.rect(image, "#B8C6AE", (13, 84, 140, 27), 2, border_radius=13)
        for x in range(28, 145, 19):
            pygame.draw.circle(image, "#B0BFAC", (x, 98), 9)
            pygame.draw.circle(image, "#596D5E", (x, 98), 6)
            pygame.draw.circle(image, "#DCE4CD", (x - 2, 96), 2)
        pygame.draw.rect(image, "#B4C7A7", (23, 59, 124, 39), border_radius=14)
        pygame.draw.rect(image, body, (19, 55, 124, 39), border_radius=13)
        pygame.draw.rect(image, light, (27, 58, 105, 6), border_radius=3)
        pygame.draw.line(image, dark, (32, 77), (128, 77), 4)
        pygame.draw.line(image, dark, (74, 68), (74, 88), 2)
        pygame.draw.rect(image, "#BDCCAE", (47, 32, 73, 37), border_radius=18)
        pygame.draw.rect(image, body, (44, 27, 73, 37), border_radius=18)
        pygame.draw.rect(image, light, (53, 31, 43, 7), border_radius=4)
        if direction in ("UP", "DOWN"):
            top, length = (3, 42) if direction == "UP" else (40, 52)
            pygame.draw.rect(image, dark, (103, top, 12, length), border_radius=5)
            pygame.draw.rect(image, light, (104, top + 4, 3, length - 8), border_radius=2)
            muzzle_y = top - 1 if direction == "UP" else top + length - 4
            pygame.draw.ellipse(image, "#C8D1B7", (99, muzzle_y, 20, 9))
            pygame.draw.ellipse(image, "#5B705D", (103, muzzle_y + 2, 12, 5))
        else:
            pygame.draw.rect(image, dark, (94, 36, 77, 12), border_radius=5)
            pygame.draw.rect(image, light, (103, 37, 63, 3), border_radius=2)
            pygame.draw.ellipse(image, "#C8D1B7", (164, 32, 13, 21))
            pygame.draw.ellipse(image, "#5B705D", (167, 36, 7, 13))
        pygame.draw.circle(image, light, (79, 29), 13)
        pygame.draw.circle(image, "#456749", (79, 29), 9)
        pygame.draw.circle(image, "#7FA46F", (76, 26), 3)
        return image

    @staticmethod
    def side_tank(screen, center, scale, player, facing_right=True):
        image = TankScene.tank_art(player)
        if not facing_right:
            image = pygame.transform.flip(image, True, False)
        image = pygame.transform.smoothscale(image, (int(180 * scale), int(118 * scale)))
        screen.blit(image, image.get_rect(center=center))

    def tank_sprite(self, unit, player, direction):
        size = max(8, round(unit * .98))
        key = (size, player, direction)
        if key not in self.sprite_cache:
            image = self.tank_art(player, direction)
            if direction == "LEFT":
                image = pygame.transform.flip(image, True, False)
            scale = size / max(image.get_size())
            dimensions = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
            self.sprite_cache[key] = pygame.transform.smoothscale(image, dimensions)
            if len(self.sprite_cache) > 80:
                self.sprite_cache = {key: self.sprite_cache[key]}
        return self.sprite_cache[key]

    def battlefield(self, app, origin, unit, rows, cols):
        ox, oy = origin
        key = (rows, cols, round(unit, 5))
        if key not in self.grid_cache:
            board = pygame.Surface((math.ceil(cols * unit), math.ceil(rows * unit)))
            board.fill("#C0D49E")
            for r in range(rows):
                for c in range(cols):
                    tile = pygame.Rect(c * unit, r * unit, math.ceil(unit), math.ceil(unit))
                    pygame.draw.rect(board, "#DEEBBD" if (r + c) % 2 else "#E8F0CE", tile.inflate(-2, -2), border_radius=max(2, int(unit * .14)))
                    if unit >= 20 and (r * 7 + c * 3) % 11 == 0:
                        x, y = tile.x + unit * .25, tile.y + unit * .7
                        pygame.draw.line(board, "#C0D69C", (x, y), (x - 2, y - 4), 1)
                        pygame.draw.line(board, "#C0D69C", (x, y), (x + 3, y - 3), 1)
            self.grid_cache = {key: board}
        frame = pygame.Rect(ox - 6, oy - 6, math.ceil(cols * unit) + 12, math.ceil(rows * unit) + 12)
        pygame.draw.rect(app.screen, "#B0C58F", frame.move(0, 3), border_radius=10)
        pygame.draw.rect(app.screen, "#C5D7A6", frame, border_radius=10)
        pygame.draw.rect(app.screen, "#9CB77C", frame, 2, border_radius=10)
        app.screen.blit(self.grid_cache[key], (ox, oy))
        state = app.tank_state
        for wall in state.get("walls", []):
            tile = pygame.Rect(ox + wall["x"] * unit + 2, oy + wall["y"] * unit + 2, unit - 4, unit - 4)
            pygame.draw.rect(app.screen, "#B5BC8D", tile.move(0, 2), border_radius=6)
            pygame.draw.rect(app.screen, "#F0C48C", tile, border_radius=6)
            pygame.draw.rect(app.screen, "#C79E69", tile, 1, border_radius=6)
            pygame.draw.line(app.screen, "#FFE7BB", (tile.x + 4, tile.y + 3), (tile.right - 5, tile.y + 3), 2)
            pygame.draw.line(app.screen, "#D9A971", (tile.x + 2, tile.centery), (tile.right - 3, tile.centery), 1)
            pygame.draw.line(app.screen, "#D9A971", (tile.centerx, tile.top + 3), tile.center, 1)
            if unit >= 24:
                app.text(wall["hp"], tile.x + 4, tile.y + 2, 10, "#876039", True)
        for player_id, base in state.get("bases", {}).items():
            if not base.get("alive", True):
                continue
            player = int(player_id)
            cx, cy = ox + (base["x"] + .5) * unit, oy + (base["y"] + .5) * unit
            sprite = self.base_sprite(unit, player)
            app.screen.blit(sprite, sprite.get_rect(center=(round(cx), round(cy - unit * .04))))
            self.health_bar(app, (cx - unit * .38, cy + unit * .39, unit * .76, max(4, unit * .11)),
                            base.get("hp", 0), 9, ("base", player))
        for player_id, tank in state.get("tanks", {}).items():
            if not tank.get("alive", True):
                continue
            player = int(player_id)
            cx, cy = ox + (tank["x"] + .5) * unit, oy + (tank["y"] + .5) * unit
            sprite = self.tank_sprite(unit, player, tank["dir"])
            app.screen.blit(sprite, sprite.get_rect(center=(round(cx), round(cy))))
            self.health_bar(app, (cx - unit * .38, cy + unit * .39, unit * .76, max(4, unit * .11)),
                            tank.get("hp", 0), 6, ("tank", player))
        for bullet in state.get("bullets", []):
            center = (round(ox + (bullet["x"] + .5) * unit), round(oy + (bullet["y"] + .5) * unit))
            pygame.draw.circle(app.screen, "#C9782E", center, max(4, int(unit * .16)))
            pygame.draw.circle(app.screen, GOLD, center, max(3, int(unit * .10)))
            pygame.draw.circle(app.screen, "#FFF4AB", center, max(1, int(unit * .045)))

    @staticmethod
    def entity(state, kind, player):
        entities = state.get(kind, {})
        return entities.get(player, entities.get(str(player), {}))

    def draw_metrics(self, app, rect):
        card_width = (rect.w - 30) / 4
        state = app.tank_state
        for i in range(4):
            card = pygame.Rect(rect.x + i * (card_width + 10), rect.y, card_width, rect.h)
            app.box(card)
            if i in (1, 2):
                player = i
                color = GREEN if player == 1 else RED
                app.text(f"PLAYER {player} · HEALTH", card.x + 11, card.y + 9, 10, color, True, max_width=card.w - 20)
                for row, (kind, label, maximum, key) in enumerate([
                    ("tanks", "TANK", 6, "tank"), ("bases", "BASE", 9, "base")]):
                    entity = self.entity(state, kind, player)
                    hp = max(0, min(maximum, entity.get("hp", 0))) if entity.get("alive", True) else 0
                    y = card.y + 27 + row * 20
                    app.text(f"{label} {hp}/{maximum}", card.x + 11, y, 11, "#61724A", True)
                    self.health_bar(app, (card.x + 78, y + 1, card.w - 90, 12), hp, maximum, (key, player))
            else:
                if i == 0:
                    label, value = "TURN", state.get("turn", 0)
                else:
                    label = "STATUS"
                    value = f"Winner: {state.get('winner')}" if state.get("is_terminal") else (
                        "Playing" if app.tank_play else "Pause" if app.tank_engine else "Preview")
                app.text(label, card.x + 14, card.y + 11, 11, "#7F805D", True)
                app.text(value, card.x + 14, card.y + 36, 19, "#98701F", True, max_width=card.w - 25)

    def draw(self, app):
        width, height = app.screen.get_size()
        self.backdrop(app.screen)
        app.button("← Main menu", (28, 27, 142, 40), app.go_menu)
        title = app.font(45, True).render("TANK", True, GOLD)
        shadow = app.font(45, True).render("TANK", True, "#8C783C")
        title_rect = title.get_rect(center=(width // 2, 34))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 2)]:
            app.screen.blit(shadow, title_rect.move(dx, dy))
        app.screen.blit(title, title_rect)
        pygame.draw.line(app.screen, "#D2B056", (width // 2 - 65, 60), (width // 2 + 65, 60), 2)
        title = app.font(23, True).render("BATTLE", True, "#587345")
        app.screen.blit(title, title.get_rect(center=(width // 2, 81)))
        self.side_tank(app.screen, (width // 2 - 170, 51), .72, 1)
        self.side_tank(app.screen, (width // 2 + 170, 51), .72, 2, False)
        self.crosshair(app.screen, (width - 162, 48), 21, "#B99E56")
        left, top, gap, controls = 28, 116, 20, 310
        board_width, board_height = width - 56 - gap - controls, height - 319
        app.draw_board(pygame.Rect(left, top, board_width, board_height))
        app.draw_controls(pygame.Rect(left + board_width + gap, top, controls, board_height + 84))
        self.draw_metrics(app, pygame.Rect(left, top + board_height + 14, board_width, 70))
        app.draw_log(pygame.Rect(left, height - 101, width - 56, 82))
        if app.busy and app.work_task == 1:
            app.text("AI is choosing an action…", 193, 39, 13, "#76632E")
