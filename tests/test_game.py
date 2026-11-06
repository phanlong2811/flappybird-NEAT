import pytest

from game import BIRD_X, FLOOR_Y, GAP_H, PIPE_W, Pipe, World


def test_idle_bird_hits_the_floor():
    world = World(1, seed=0)
    # Free fall from mid-height at gravity 0.5 px/frame^2 reaches the floor in under 100 frames.
    for _ in range(100):
        world.step([False])
    assert world.done
    assert world.birds[0].y > FLOOR_Y - 30


def test_bird_in_pipe_body_dies_on_that_frame():
    world = World(2, seed=0)
    world.pipes = [Pipe(x=BIRD_X, gap_y=300)]
    world.birds[0].y = 300 - GAP_H / 2 - 30  # inside the upper pipe
    world.birds[1].y = 300  # centre of the gap
    world.step([False, False])
    assert [b.alive for b in world.birds] == [False, True]


def oracle(world):
    pipe = world.next_pipe()
    return [b.y > pipe.gap_y and b.vy >= 0 for b in world.birds]


@pytest.mark.parametrize("seed", range(10))
def test_oracle_passes_twenty_pipes(seed):
    world = World(1, seed=seed)
    while not world.done and world.score < 20:
        world.step(oracle(world))
    assert world.score == 20


def test_same_seed_same_course():
    a, b = World(1, seed=7), World(1, seed=7)
    for _ in range(600):
        a.step(oracle(a))
        b.step(oracle(b))
    assert [p.gap_y for p in a.pipes] == [p.gap_y for p in b.pipes]
    assert a.score == b.score > 0


def test_pipe_scrolls_out_and_is_scored():
    world = World(1, seed=0)
    world.pipes = [Pipe(x=BIRD_X - PIPE_W - 20, gap_y=300)]
    world.birds[0].y = 300
    world.step([False])
    assert world.score == 1
