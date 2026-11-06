import random
from collections.abc import Sequence
from dataclasses import dataclass

WIDTH, HEIGHT = 400, 600
FLOOR_Y = 540
BIRD_X, BIRD_R = 100, 12
GRAVITY, FLAP_VY, MAX_FALL = 0.5, -7.0, 10.0
PIPE_W, GAP_H, PIPE_SPEED, PIPE_SPACING = 60, 160, 3.0, 220
GAP_MARGIN = 40
MAX_GAP_SHIFT = 130


@dataclass
class Bird:
    y: float
    vy: float = 0.0
    alive: bool = True


@dataclass
class Pipe:
    x: float
    gap_center: float
    passed: bool = False


class World:
    def __init__(self, n_birds: int, seed: int | None = None):
        self.rng = random.Random(seed)
        self.birds = [Bird(y=FLOOR_Y / 2) for _ in range(n_birds)]
        self.pipes = [self._new_pipe(WIDTH, FLOOR_Y / 2)]
        self.frame = 0
        self.score = 0

    def _new_pipe(self, x: float, prev_gap_center: float) -> Pipe:
        min_gap_center = GAP_H / 2 + GAP_MARGIN
        gap_center = prev_gap_center + self.rng.uniform(-MAX_GAP_SHIFT, MAX_GAP_SHIFT)
        return Pipe(x, min(FLOOR_Y - min_gap_center, max(min_gap_center, gap_center)))

    def next_pipe(self) -> Pipe:
        return next(p for p in self.pipes if not p.passed)

    def step(self, flaps: Sequence[bool]) -> None:
        self.frame += 1
        for bird, flap in zip(self.birds, flaps):
            if bird.alive:
                if flap:
                    bird.vy = FLAP_VY
                bird.vy = min(bird.vy + GRAVITY, MAX_FALL)
                bird.y += bird.vy
        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED
        while self.pipes[-1].x < WIDTH:
            last = self.pipes[-1]
            self.pipes.append(self._new_pipe(last.x + PIPE_SPACING, last.gap_center))
        self.pipes = [p for p in self.pipes if p.x + PIPE_W > 0]
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_W < BIRD_X - BIRD_R:
                pipe.passed = True
                self.score += 1
        for bird in self.birds:
            if bird.alive and self._crashed(bird):
                bird.alive = False

    def _crashed(self, bird: Bird) -> bool:
        if bird.y - BIRD_R < 0 or bird.y + BIRD_R > FLOOR_Y:
            return True
        return any(
            p.x < BIRD_X + BIRD_R
            and p.x + PIPE_W > BIRD_X - BIRD_R
            and abs(bird.y - p.gap_center) > GAP_H / 2 - BIRD_R
            for p in self.pipes
        )

    @property
    def done(self) -> bool:
        return not any(b.alive for b in self.birds)
