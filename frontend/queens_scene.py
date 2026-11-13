"""Wood, parchment, and gold presentation for the N-Queens challenge."""
import math

import pygame

from .backends import ROOT


class QueensScene:
    COLORS = {
        "#ECF2FA": "#493522", "#94A6BD": "#80674D",
        "#57E2C0": "#895C23", "#0C1220": "#302318",
        "#2B3A51": "#C5A67A", "#FFBD73": "#9A5426",
        "#76ABFF": "#536F61", "#B8A1FF": "#977347",
    }

    def __init__(self):
        path = ROOT / "assets" / "gold_queen.png"
        self.piece = pygame.image.load(str(path)).convert_alpha() if path.exists() else None
        if self.piece is not None:
            # Ignore transparent padding when fitting the supplied piece in a cell.
            bounds = self.piece.get_bounding_rect(min_alpha=128)
            self.piece = self.piece.subsurface(bounds).copy()
        self.piece_cache = {}
        self.board_cache = {}
        self.grid_geometry = None

    def cell_at(self, position):
        if self.grid_geometry is None:
            return None
        (ox, oy), unit, n = self.grid_geometry
        x, y = position
        if ox <= x < ox + n * unit and oy <= y < oy + n * unit:
            return int((y - oy) / unit), int((x - ox) / unit)
        return None

    def color(self, color):
        return self.COLORS.get(color, color)

    def panel(self, app, rect, color, radius, border):
        rect = pygame.Rect(rect)
        if color != "#151F30":
            pygame.draw.rect(app.screen, self.color(color), rect, border_radius=radius)
            if border:
                pygame.draw.rect(app.screen, self.color(border), rect, 1, border_radius=radius)
            return rect
        pygame.draw.rect(app.screen, "#82613D", rect.move(0, 5), border_radius=25)
        pygame.draw.rect(app.screen, "#A67948", rect, border_radius=25)
        pygame.draw.rect(app.screen, "#E0BB7C", rect.inflate(-3, -3), 1, border_radius=23)
        inner = rect.inflate(-13, -13)
        pygame.draw.rect(app.screen, "#F5E7C7", inner, border_radius=19)
        pygame.draw.rect(app.screen, "#C9AA75", inner, 1, border_radius=19)
        # Quiet paper fibers, kept away from text and board detail.
        for y in range(inner.y + 9, inner.bottom - 3, 7):
            pygame.draw.line(app.screen, "#EEE0BD", (inner.x + 18, y), (inner.right - 19, y))
        for x in (rect.x + 13, rect.right - 14):
            for y in (rect.y + 13, rect.bottom - 14):
                pygame.draw.circle(app.screen, "#D9B578", (x, y), 2)
        return rect

    def button(self, app, label, rect, action, primary, enabled, selected):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos()) and enabled
        fill = ("#A96B31" if hover else "#8B562B") if enabled else "#C6B18A"
        edge = "#E6BE79" if primary and enabled or hover or selected else "#A78755"
        radius = min(20, rect.h // 2)
        pygame.draw.rect(app.screen, "#A18155", rect.move(0, 3), border_radius=radius)
        pygame.draw.rect(app.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(app.screen, edge, rect, 2, border_radius=radius)
        for y in (rect.y + 7, rect.bottom - 7):
            pygame.draw.line(app.screen, "#A5703A" if enabled else "#BCA47E",
                             (rect.x + 8, y), (rect.right - 9, y), 1)
        for x in (rect.x + 8, rect.right - 9):
            pygame.draw.circle(app.screen, "#E0C391" if enabled else "#A38E69", (x, rect.centery), 2)
        font = app.font(16, primary)
        while font.size(label)[0] > rect.w - 34 and len(label) > 2:
            label = label[:-2] + "…" if label.endswith("…") else label[:-1] + "…"
        text = font.render(label, True, "#FFF1D3" if enabled else "#786346")
        if enabled:
            shadow = font.render(label, True, "#58391F")
            app.screen.blit(shadow, shadow.get_rect(center=(rect.centerx, rect.centery + 1)))
        app.screen.blit(text, text.get_rect(center=rect.center))
        if enabled:
            app.buttons.append((rect, action))

    def sprite(self, height):
        height = max(3, int(height))
        if self.piece is None:
            return None
        if height not in self.piece_cache:
            width = max(1, round(height * self.piece.get_width() / self.piece.get_height()))
            self.piece_cache[height] = pygame.transform.smoothscale(self.piece, (width, height))
            if len(self.piece_cache) > 40:
                self.piece_cache = {height: self.piece_cache[height]}
        return self.piece_cache[height]

    def grid(self, app, origin, unit):
        self.grid_geometry = (origin, unit, app.n)
        ox, oy = origin
        size = math.ceil(unit * app.n)
        cache_key = (app.n, round(unit, 5))
        if cache_key not in self.board_cache:
            board = pygame.Surface((size, size))
            for r in range(app.n):
                for c in range(app.n):
                    dark = (r + c) % 2
                    tile = pygame.Rect(c * unit, r * unit, math.ceil(unit), math.ceil(unit))
                    pygame.draw.rect(board, "#95663E" if dark else "#E9CEA0", tile)
                    if unit >= 20:
                        for fraction in (.23, .76):
                            y = tile.y + unit * fraction
                            pygame.draw.line(board, "#9D7048" if dark else "#E2C596",
                                             (tile.x + 2, y), (tile.right - 2, y))
            self.board_cache = {cache_key: board}
        frame = pygame.Rect(ox - 12, oy - 12, size + 24, size + 24)
        # The board and every square stay sharp, unlike the rounded UI panels.
        pygame.draw.rect(app.screen, "#4C321E", frame.move(0, 5))
        pygame.draw.rect(app.screen, "#724827", frame)
        pygame.draw.rect(app.screen, "#B58A4D", frame.inflate(-4, -4), 2)
        app.screen.blit(self.board_cache[cache_key], (ox, oy))
        for col, row in app.fixed_queens.items():
            tile = pygame.Rect(ox + col * unit, oy + row * unit, unit, unit)
            color = "#B43329" if col in app.queen_conflicts else "#C69B37"
            pygame.draw.rect(app.screen, color, tile.inflate(-2, -2), max(1, int(unit * .05)))
        if not app.busy and not app.modal:
            hover = self.cell_at(pygame.mouse.get_pos())
            if hover is not None:
                row, col = hover
                pygame.draw.rect(app.screen, "#FFF0B3", (ox + col * unit, oy + row * unit, unit, unit), 2)
        piece = self.sprite(unit * .91)
        for col, row in enumerate(app.queen_board):
            if not 0 <= row < app.n:
                continue
            cx, cy = ox + (col + .5) * unit, oy + (row + .5) * unit
            if unit < 9:
                pygame.draw.circle(app.screen, "#FFDF77", (int(cx), int(cy)), max(1, int(unit * .35)))
            elif piece is not None:
                shadow = pygame.Rect(0, 0, max(3, unit * .49), max(2, unit * .10))
                shadow.center = (cx, cy + unit * .39)
                pygame.draw.ellipse(app.screen, "#75512F", shadow)
                app.screen.blit(piece, piece.get_rect(center=(round(cx), round(cy))))
        if app.n <= 16:
            for i in range(app.n):
                label = app.font(10, True).render(chr(65 + i), True, "#D7B778")
                app.screen.blit(label, label.get_rect(center=(ox + (i + .5) * unit, oy + size + 7)))
        if not app.queen_board:
            label = app.font(14, True).render("Click to place queens · or press Solve", True, "#FFF1D3")
            bounds = label.get_rect(center=(ox + size / 2, oy + size / 2))
            pygame.draw.rect(app.screen, "#624328", bounds.inflate(30, 22), border_radius=19)
            pygame.draw.rect(app.screen, "#D7B476", bounds.inflate(30, 22), 2, border_radius=19)
            app.screen.blit(label, bounds)

    def draw(self, app):
        width, height = app.screen.get_size()
        # Share the menu backdrop exactly, without a second dimming overlay.
        app.menu.backdrop(app.screen)
        app.button("← Main menu", (28, 28, 142, 40), app.go_menu)
        title = app.font(37, True).render("N-QUEENS", True, "#FFF0CA")
        title_shadow = app.font(37, True).render("N-QUEENS", True, "#39432A")
        app.screen.blit(title_shadow, title_shadow.get_rect(center=(width // 2 + 2, 44)))
        app.screen.blit(title, title.get_rect(center=(width // 2, 41)))
        subtitle = app.font(14).render("THE QUEENS CHALLENGE  /  LOGIC & LOCAL SEARCH", True, "#F1DDB4")
        subtitle_shadow = app.font(14).render("THE QUEENS CHALLENGE  /  LOGIC & LOCAL SEARCH", True, "#39432A")
        app.screen.blit(subtitle_shadow, subtitle_shadow.get_rect(center=(width // 2 + 1, 81)))
        app.screen.blit(subtitle, subtitle.get_rect(center=(width // 2, 80)))
        piece = self.sprite(68)
        if piece:
            app.screen.blit(piece, (width - 79, 14))
        left, top, gap, controls = 28, 112, 20, 310
        board_width, board_height = width - 56 - gap - controls, height - 315
        app.draw_board(pygame.Rect(left, top, board_width, board_height))
        app.draw_controls(pygame.Rect(left + board_width + gap, top, controls, board_height + 84))
        app.draw_metrics(pygame.Rect(left, top + board_height + 14, board_width, 70))
        app.draw_log(pygame.Rect(left, height - 101, width - 56, 82))
        if app.busy and app.work_task == 0:
            app.text("Finding a solution…", width - 298, 50, 16, "#FFF0CA", True)
