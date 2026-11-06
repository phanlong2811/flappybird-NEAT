"""Drawing only: a dusk scene with stone pillars and translucent birds."""
import random
from functools import cache

import pygame

from game import BIRD_R, BIRD_X, FLOOR_Y, GAP_H, HEIGHT, PIPE_SPEED, PIPE_W, WIDTH, World

SKY_TOP, SKY_LOW = (24, 16, 56), (214, 98, 130)
STONE, STONE_DARK, STONE_LIGHT = (218, 200, 160), (150, 128, 98), (244, 232, 200)
GROUND, GROUND_RIM = (43, 18, 51), (240, 160, 70)
BIRD_BODY, BIRD_WING, BIRD_ALPHA = (110, 235, 255), (30, 120, 190), 110
CAP_H = 24


@cache
def _background() -> pygame.Surface:
    surf = pygame.Surface((WIDTH, HEIGHT))
    for y in range(FLOOR_Y):
        t = (y / FLOOR_Y) ** 1.5
        pygame.draw.line(surf, [a + (b - a) * t for a, b in zip(SKY_TOP, SKY_LOW)], (0, y), (WIDTH, y))
    rng = random.Random(1)
    for _ in range(70):
        x, y = rng.randrange(WIDTH), rng.randrange(FLOOR_Y * 2 // 3)
        shade = rng.randrange(120, 230)
        surf.set_at((x, y), (shade, shade, 255))
    halo = pygame.Surface((120, 120), pygame.SRCALPHA)
    for r in range(60, 28, -1):
        pygame.draw.circle(halo, (250, 240, 215, 2), (60, 60), r)
    surf.blit(halo, (250, 30))
    pygame.draw.circle(surf, (250, 240, 215), (310, 90), 28)
    pygame.draw.circle(surf, (225, 212, 190), (302, 96), 6)
    pygame.draw.circle(surf, (225, 212, 190), (320, 80), 4)
    return surf


@cache
def _bird_sprite() -> pygame.Surface:
    size = 4 * BIRD_R
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    pygame.draw.circle(sprite, BIRD_BODY, (c, c), BIRD_R)
    pygame.draw.ellipse(sprite, BIRD_WING, (c - BIRD_R, c - 2, BIRD_R + 2, BIRD_R - 2))
    pygame.draw.polygon(sprite, (255, 150, 90), [(c + BIRD_R - 2, c - 3), (c + BIRD_R + 7, c + 1), (c + BIRD_R - 2, c + 4)])
    pygame.draw.circle(sprite, (20, 12, 40), (c + 5, c - 4), 2)
    sprite.set_alpha(BIRD_ALPHA)
    return sprite


@cache
def _font(size: int) -> pygame.font.Font:
    return pygame.font.Font(None, size)


def _pillar(surface, x, top, bottom, cap_y):
    pygame.draw.rect(surface, STONE, (x, top, PIPE_W, bottom - top))
    pygame.draw.rect(surface, STONE_DARK, (x + PIPE_W - 10, top, 10, bottom - top))
    pygame.draw.rect(surface, STONE_LIGHT, (x - 4, cap_y, PIPE_W + 8, CAP_H), border_radius=4)
    pygame.draw.rect(surface, STONE_DARK, (x - 4, cap_y, PIPE_W + 8, CAP_H), 2, border_radius=4)


def draw(surface: pygame.Surface, world: World, hud: dict[str, str]) -> None:
    surface.blit(_background(), (0, 0))
    for pipe in world.pipes:
        x = round(pipe.x)
        gap_top, gap_bottom = round(pipe.gap_y - GAP_H / 2), round(pipe.gap_y + GAP_H / 2)
        _pillar(surface, x, 0, gap_top, gap_top - CAP_H)
        _pillar(surface, x, gap_bottom, FLOOR_Y, gap_bottom)
    pygame.draw.rect(surface, GROUND, (0, FLOOR_Y, WIDTH, HEIGHT - FLOOR_Y))
    pygame.draw.line(surface, GROUND_RIM, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)
    offset = round(world.frame * PIPE_SPEED) % 24
    for x in range(-offset, WIDTH, 24):
        pygame.draw.line(surface, (70, 34, 80), (x, FLOOR_Y + 8), (x - 10, HEIGHT), 2)
    sprite = _bird_sprite()
    for bird in world.birds:
        if bird.alive:
            tilted = pygame.transform.rotate(sprite, -max(-30, min(60, bird.vy * 5)))
            surface.blit(tilted, tilted.get_rect(center=(BIRD_X, round(bird.y))))
    score = _font(56).render(str(world.score), True, (255, 245, 225))
    surface.blit(score, score.get_rect(center=(WIDTH // 2, (FLOOR_Y + HEIGHT) // 2 + 2)))
    panel = pygame.Surface((170, 8 + 20 * len(hud)), pygame.SRCALPHA)
    panel.fill((10, 6, 30, 150))
    surface.blit(panel, (8, 8))
    for i, (key, value) in enumerate(hud.items()):
        text = _font(22).render(f"{key} {value}", True, (235, 225, 245))
        surface.blit(text, (14, 12 + 20 * i))
