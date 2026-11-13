"""Storybook main menu. Artwork is decorative; every sign is a real button."""
import math
import random

import pygame


class GameMenu:
    def __init__(self, background_path):
        self.background = pygame.image.load(str(background_path)).convert() if background_path.exists() else None
        self.scaled_background = None
        self.cached_size = None
        self.selected = 0
        self.mouse_position = None
        self.sign_rects = []

    def backdrop(self, screen):
        width, height = screen.get_size()
        if self.cached_size != (width, height):
            if self.background:
                scale = max(width / self.background.get_width(), height / self.background.get_height())
                image = pygame.transform.smoothscale(self.background,
                    (math.ceil(self.background.get_width() * scale), math.ceil(self.background.get_height() * scale)))
                self.scaled_background = pygame.Surface((width, height))
                self.scaled_background.blit(image, image.get_rect(center=(width // 2, height // 2)))
            else:
                self.scaled_background = pygame.Surface((width, height))
                self.scaled_background.fill("#3E5945")
            # A graduated shade grounds the title without hiding the scenery.
            shade = pygame.Surface((width, height), pygame.SRCALPHA)
            for y in range(height):
                alpha = int(14 + 150 * (y / height) ** 3)
                pygame.draw.line(shade, (9, 23, 17, alpha), (0, y), (width, y))
            self.scaled_background.blit(shade, (0, 0))
            self.cached_size = (width, height)
        screen.blit(self.scaled_background, (0, 0))

    def centered(self, app, text, center, size, color, bold=False):
        font = app.font(size, bold)
        shadow = font.render(text, True, "#392619")
        label = font.render(text, True, color)
        rect = label.get_rect(center=center)
        app.screen.blit(shadow, rect.move(0, 2))
        app.screen.blit(label, rect)

    def sign(self, app, rect, title, caption, index, callback, active):
        x, y, w, h = rect
        screen = app.screen
        shift = -7 if active else 0
        x += shift
        points = [(x + 25, y), (x + w, y + 2), (x + w - 3, y + h),
                  (x + 26, y + h - 3), (x, y + h // 2)]
        pygame.draw.polygon(screen, "#32251B", [(a + 4, b + 8) for a, b in points])
        pygame.draw.polygon(screen, "#E4BB7A" if active else "#A28C64", points)
        inner = [(x + 29, y + 6), (x + w - 6, y + 8), (x + w - 8, y + h - 7),
                 (x + 30, y + h - 9), (x + 9, y + h // 2)]
        pygame.draw.polygon(screen, "#A65E2E" if active else "#784626", inner)
        # Seeded grain stays still between frames and at different frame rates.
        grain = random.Random(40 + index)
        for i in range(11):
            gy = y + 12 + i * (h - 25) / 11
            start = x + grain.randint(33, 68)
            end = x + w - grain.randint(12, 40)
            pygame.draw.line(screen, "#B77843" if active else "#935B31", (start, gy), (end, gy + grain.randint(-2, 2)), 1)
        # Two steel brackets and screws match the physical signpost design.
        for fraction in [.25, .86]:
            bx = x + int(w * fraction)
            pygame.draw.rect(screen, "#4D5147", (bx, y - 5, 19, 28), border_radius=3)
            pygame.draw.rect(screen, "#B0AF94", (bx + 2, y - 3, 15, 24), 2, border_radius=3)
            pygame.draw.circle(screen, "#D3C9A7", (bx + 9, y + 7), 3)
            pygame.draw.line(screen, "#605D4C", (bx + 7, y + 7), (bx + 11, y + 7))
        if active:
            pygame.draw.polygon(screen, "#FCE3A4", [(x - 19, y + h // 2 - 6), (x - 8, y + h // 2), (x - 19, y + h // 2 + 6)])
        size = max(24, min(35, int(w * .105)))
        self.centered(app, title, (x + w * .54, y + h * .45), size, "#FFF5DF", True)
        self.centered(app, caption, (x + w * .54, y + h * .79), 12, "#F0D5AE")
        app.buttons.append((rect.inflate(18, 12), callback))

    def draw(self, app):
        screen = app.screen
        width, height = screen.get_size()
        self.backdrop(screen)
        mouse = pygame.mouse.get_pos()
        self.sign_rects = []
        x, w = int(width * .66), int(width * .285)
        h, gap, top = int(height * .103), int(height * .035), int(height * .20)
        # Wooden support behind all four signs.
        post_x = x + int(w * .84)
        pygame.draw.rect(screen, "#3E2A1C", (post_x + 5, top - 66, 49, int(height * .76)), border_radius=5)
        pygame.draw.rect(screen, "#9D663A", (post_x, top - 74, 43, int(height * .76)), border_radius=4)
        pygame.draw.rect(screen, "#C18848", (post_x + 5, top - 72, 7, int(height * .75)), border_radius=2)
        pygame.draw.polygon(screen, "#D3A367", [(post_x - 4, top - 73), (post_x + 43, top - 79),
                                              (post_x + 48, top - 66), (post_x, top - 61)])
        items = [("N-Queens", "01  /  LOGIC & LOCAL SEARCH", lambda: app.open_task(0)),
                 ("Tank Battle", "02  /  ADVERSARIAL SEARCH", lambda: app.open_task(1)),
                 ("Kakuro", "03  /  CONSTRAINT SATISFACTION", lambda: app.open_task(2)),
                 ("Quit", "SEE YOU NEXT TIME, EXPLORER", lambda: setattr(app, "running", False))]
        rects = [pygame.Rect(x, top + i * (h + gap), w, h) for i in range(4)]
        if mouse != self.mouse_position:
            for i, rect in enumerate(rects):
                if rect.inflate(18, 12).collidepoint(mouse):
                    self.selected = i
            self.mouse_position = mouse
        for i, ((title, caption, callback), rect) in enumerate(zip(items, rects)):
            self.sign_rects.append(rect)
            self.sign(app, rect, title, caption, i, callback, i == self.selected)

        left = int(width * .055)
        app.text("INTRODUCTION TO ARTIFICIAL INTELLIGENCE", left, int(height * .065), 13, "#F8EDD3", True)
        pygame.draw.line(screen, "#D7B67B", (left, int(height * .105)), (left + 78, int(height * .105)), 2)
        title_y = int(height * .65)
        for text, y, size in [("AI", title_y, 57), ("ADVENTURES", title_y + 56, 59)]:
            font = app.font(size, True)
            shadow = font.render(text, True, "#193426")
            label = font.render(text, True, "#FFF1CE")
            screen.blit(shadow, (left + 3, y + 4))
            screen.blit(label, (left, y))
        app.text("Three challenges. One adventure.", left + 3, title_y + 129, 19, "#F8EDD8")
        app.text("Choose a sign to begin", left + 3, title_y + 162, 15, "#E2D5B5")
        app.text("TDTU  /  FINAL PROJECT", left, height - 36, 11, "#DFD5BC", True)
        app.text("↑ ↓  Select    Enter  Open    Esc  Quit", width - 320, height - 36, 12, "#F4E5C7")

    def key(self, app, event):
        if event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % 4
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % 4
        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
            if self.selected < 3:
                app.open_task(self.selected)
            else:
                app.running = False
        elif event.key == pygame.K_ESCAPE:
            app.running = False
