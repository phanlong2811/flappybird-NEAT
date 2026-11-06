"""NEAT agent: observation, per-generation evaluation and training."""
import pickle
from pathlib import Path

import neat

from game import BIRD_X, FLOOR_Y, GAP_H, HEIGHT, MAX_FALL, WIDTH, Bird, World

CONFIG_PATH = Path(__file__).parent / "config-feedforward.txt"
WINNER_PATH = Path("winner.pkl")
COURSES = 10  # courses flown per generation; a genome scores its worst one


class Stop(Exception):
    """Raised by an on_frame hook to end the run cleanly."""


def load_config() -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG_PATH))


def observe(world: World, bird: Bird) -> list[float]:
    pipe = world.next_pipe()
    return [
        bird.y / FLOOR_Y,
        bird.vy / MAX_FALL,
        (pipe.x - BIRD_X) / WIDTH,
        (pipe.gap_y - GAP_H / 2 - bird.y) / HEIGHT,
        (pipe.gap_y + GAP_H / 2 - bird.y) / HEIGHT,
    ]


def run_world(world, nets, on_frame=None, info=None, max_score=None) -> list[float]:
    """Play nets[i] as bird i until all die or max_score is reached. Returns each bird's fitness:
    pipes passed plus a survival bonus below 1, so only passing max_score pipes reaches max_score."""
    fitness = [0.0] * len(nets)
    while not world.done and (max_score is None or world.score < max_score):
        world.step([b.alive and net.activate(observe(world, b))[0] > 0.5
                    for b, net in zip(world.birds, nets)])
        for i, bird in enumerate(world.birds):
            if bird.alive:  # a bird keeps the score it had on its last living frame
                fitness[i] = world.score + world.frame / (world.frame + 100)
        if on_frame:
            on_frame(world, info or {})
    return fitness


def train(generations: int = 50, seed: int = 0, on_frame=None):
    """Evolve for up to `generations`, save the winner to winner.pkl and return it.
    Training ends when a genome passes `fitness_threshold` pipes on every course of a generation."""
    config = load_config()
    pop = neat.Population(config, seed=seed)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    def evaluate(genomes, config):
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        best = max((g.fitness for g in stats.most_fit_genomes), default=0.0)
        info = {"generation": pop.generation, "best": best}
        fits = []
        for course in range(COURSES):
            world = World(len(nets), seed=seed * 100_000 + pop.generation * COURSES + course)
            hook = on_frame if course == 0 else None  # draw one course per generation, run the rest headless
            fits.append(run_world(world, nets, hook, info, config.fitness_threshold))
        for (_, genome), per_course in zip(genomes, zip(*fits)):
            genome.fitness = min(per_course)

    try:
        winner = pop.run(evaluate, generations)
    except Stop:
        winner = max(stats.most_fit_genomes, key=lambda g: g.fitness, default=None)
    if winner:
        WINNER_PATH.write_bytes(pickle.dumps(winner))
    return winner


def play_winner(seed: int | None = None, on_frame=None) -> int:
    """Replay winner.pkl alone on a fresh course until it dies or on_frame raises Stop."""
    genome = pickle.loads(WINNER_PATH.read_bytes())
    net = neat.nn.FeedForwardNetwork.create(genome, load_config())
    world = World(1, seed)
    try:
        run_world(world, [net], on_frame)
    except Stop:
        pass
    return world.score
