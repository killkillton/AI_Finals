"""Compose alpha artwork in software before presenting an opaque window frame."""
import pygame


class Canvas(pygame.Surface):
    def blit(self, source, dest, area=None, special_flags=0):
        # Use SDL's alpha compositor consistently for fonts, PNG sprites, and
        # translucent overlays. A regular RGB canvas isolates that composition
        # from display-specific pixel formats (including macOS window surfaces).
        if special_flags == 0 and source.get_flags() & pygame.SRCALPHA:
            special_flags = pygame.BLEND_ALPHA_SDL2
        return super().blit(source, dest, area, special_flags)
