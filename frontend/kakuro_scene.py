"""Warm storybook Kakuro board using the same village and wood UI as the menu."""
import math

import pygame


class KakuroScene:
    COLORS = {
        "#ECF2FA": "#493C2A", "#94A6BD": "#84735A",
        "#57E2C0": "#579660", "#0C1220": "#FFF8E8",
        "#2B3A51": "#CCB487", "#FFBD73": "#D89543",
        "#76ABFF": "#5897BD", "#FF8193": "#D36E70",
    }
    EVENTS = {
        "SELECT_VAR": ("#E1EFF7", "#5897BD"),
        "PROPAGATE": ("#E1EFF7", "#5897BD"),
        "ASSIGN": ("#E4F0D6", "#579660"),
        "SOLVED": ("#E4F0D6", "#579660"),
        "CONFLICT": ("#F8E1DA", "#D36E70"),
        "BACKTRACK": ("#F8E1DA", "#D36E70"),
    }

    def color(self, color):
        return self.COLORS.get(color, color)

    def panel(self, app, rect, color, radius, border):
        return app.queens_scene.panel(app, rect, color, radius, border)

    def button(self, app, label, rect, action, primary, enabled, selected):
        return app.queens_scene.button(app, label, rect, action, primary, enabled, selected)

    def grid(self, app, origin, unit):
        ox, oy = origin
        layout, event = app.kakuro_layout, app.kakuro_event
        assignments, domains = event.get("assignments", {}), event.get("domains", {})
        focus = event.get("focus_cell")
        focus = tuple(focus) if focus is not None else None
        pruned = {(r, c) for r, c, _ in event.get("pruned_cells", [])}
        kind = event.get("event_type")
        event_fill, event_edge = self.EVENTS.get(kind, ("#E1EFF7", "#5897BD"))
        frame = pygame.Rect(ox - 7, oy - 7, math.ceil(layout["cols"] * unit) + 14,
                            math.ceil(layout["rows"] * unit) + 14)
        pygame.draw.rect(app.screen, "#775436", frame.move(0, 4), border_radius=7)
        pygame.draw.rect(app.screen, "#9D764B", frame, border_radius=7)
        pygame.draw.rect(app.screen, "#E6C895", frame, 2, border_radius=7)
        for cell in layout["cells"]:
            r, c, cell_type = cell["r"], cell["c"], cell["type"]
            rect = pygame.Rect(ox + c * unit, oy + r * unit, unit, unit).inflate(-2, -2)
            radius = max(2, min(4, int(unit * .06)))
            active = (r, c) == focus or kind == "SOLVED" and cell_type == "WHITE"
            edge, fill = "#9B764C", "#EBD3AB"
            if active:
                fill, edge = event_fill, event_edge
            if (r, c) in pruned:
                fill, edge = "#FFEDCC", "#D89543"
            if cell_type == "WHITE":
                pygame.draw.rect(app.screen, "#AA8654", rect.move(0, 2), border_radius=radius)
                pygame.draw.rect(app.screen, fill, rect, border_radius=radius)
                pygame.draw.line(app.screen, "#F9E9CA", (rect.x + 4, rect.y + 2), (rect.right - 5, rect.y + 2), 1)
                pygame.draw.rect(app.screen, edge, rect, 3 if active or (r, c) in pruned else 1, border_radius=radius)
                if (r, c) in assignments:
                    font = app.font(max(16, int(unit * .54)), True)
                    text = str(assignments[(r, c)])
                    value = font.render(text, True, "#FFF6DF")
                    outline = font.render(text, True, "#73553A")
                    bounds = value.get_rect(center=rect.center)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 2)]:
                        app.screen.blit(outline, bounds.move(dx, dy))
                    app.screen.blit(value, bounds)
                else:
                    for value in domains.get((r, c), range(1, 10)):
                        x = ox + c * unit + ((value - 1) % 3 + .5) * unit / 3
                        y = oy + r * unit + ((value - 1) // 3 + .5) * unit / 3
                        image = app.font(max(10, int(unit * .18))).render(str(value), True, "#72765B")
                        app.screen.blit(image, image.get_rect(center=(round(x), round(y))))
            else:
                pygame.draw.rect(app.screen, "#AD855A" if cell_type == "CLUE" else "#99744E", rect, border_radius=radius)
                pygame.draw.rect(app.screen, "#765336", rect, 1, border_radius=radius)
                pygame.draw.line(app.screen, "#CAA87B", (rect.x + 4, rect.y + 2), (rect.right - 5, rect.y + 2), 1)
                if cell_type == "CLUE":
                    pygame.draw.line(app.screen, "#DDC397", (rect.x + 3, rect.y + 3), (rect.right - 4, rect.bottom - 4), 1)
                    size = max(12, int(unit * .24))
                    for key, center in [("down_sum", (rect.x + rect.w * .25, rect.y + rect.h * .73)),
                                        ("right_sum", (rect.x + rect.w * .75, rect.y + rect.h * .27))]:
                        if cell.get(key) is not None:
                            label = app.font(size, True).render(str(cell[key]), True, "#FFF0CC")
                            app.screen.blit(label, label.get_rect(center=center))
                elif unit > 25:
                    cx, cy, size = rect.centerx, rect.centery, unit * .09
                    pygame.draw.polygon(app.screen, "#99764F", [(cx, cy-size), (cx+size, cy), (cx, cy+size), (cx-size, cy)], 1)

    def playfield(self, app, rect):
        app.box(rect)
        mat = rect.inflate(-17, -17)
        pygame.draw.rect(app.screen, "#C89A78", mat, border_radius=17)
        pygame.draw.rect(app.screen, "#E3BF91", mat, 1, border_radius=17)
        # Quiet botanical ornaments sit behind the board, never over the digits.
        for corner_x, corner_y, flip in [(mat.x + 26, mat.y + 95, 1), (mat.right - 26, mat.bottom - 25, -1)]:
            for i in range(4):
                cx, cy = corner_x + flip * i * 12, corner_y + flip * i * 24
                pygame.draw.line(app.screen, "#B68565", (cx, cy), (cx + flip * 12, cy + flip * 25), 2)
                pygame.draw.ellipse(app.screen, "#BF8E6D", (cx - 13, cy - 17, 25, 13), 2)
                pygame.draw.ellipse(app.screen, "#BF8E6D", (cx + 3, cy - 3, 13, 25), 2)
        app.text("KAKURO BOARD", rect.x + 24, rect.y + 20, 12, "#FFF3D7", True)
        badge = "DOMAINS & SEARCH" if app.kakuro else "PREVIEW"
        app.text(badge, rect.x + 24, rect.y + 42, 11, "#F7E3C1")
        area = pygame.Rect(rect.x + 30, rect.y + 80, rect.w - 60, rect.h - 110)
        rows, cols = app.kakuro_layout["rows"], app.kakuro_layout["cols"]
        unit = min(area.w / cols, area.h / rows)
        origin = (area.centerx - cols * unit / 2, area.centery - rows * unit / 2)
        self.grid(app, origin, unit)

    def draw(self, app):
        width, height = app.screen.get_size()
        app.menu.backdrop(app.screen)
        app.button("← Main menu", (28, 28, 142, 40), app.go_menu)
        controls, gap = 290, 20
        board_x = 28 + controls + gap
        board_width = width - board_x - 28
        title_x = board_x + board_width // 2
        for text, size, y in [("KAKURO", 39, 41), ("THE NUMBER GARDEN  /  CONSTRAINT SATISFACTION", 14, 80)]:
            shadow = app.font(size, size > 20).render(text, True, "#39432A")
            title = app.font(size, size > 20).render(text, True, "#FFF0CA")
            rect = title.get_rect(center=(title_x, y))
            app.screen.blit(shadow, rect.move(1, 2))
            app.screen.blit(title, rect)
        # Small number-tile badge, drawn with genuine alpha on the scene canvas.
        for i, value in enumerate([1, 2, 3, 4]):
            tile = pygame.Rect(width - 91 + i % 2 * 27, 23 + i // 2 * 27, 24, 24)
            pygame.draw.rect(app.screen, "#FFF0CC", tile, border_radius=6)
            pygame.draw.rect(app.screen, "#BC985B", tile, 1, border_radius=6)
            label = app.font(15, True).render(str(value), True, "#726137")
            app.screen.blit(label, label.get_rect(center=tile.center))
        top, board_height = 112, height - 315
        app.draw_controls(pygame.Rect(28, top, controls, board_height + 84))
        self.playfield(app, pygame.Rect(board_x, top, board_width, board_height))
        app.draw_metrics(pygame.Rect(board_x, top + board_height + 14, board_width, 70))
        app.draw_log(pygame.Rect(28, height - 101, width - 56, 82))
        if app.busy and app.work_task == 2:
            app.text("Exploring the next step…", width - 310, 47, 14, "#FFF0CA", True)
