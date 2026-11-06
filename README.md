# NEAT pipe dodger

NEAT (neat-python 2.x) evolves small neural networks that fly a bird through a gap in a never-ending row of pillars. The window uses only pygame shapes, with no image assets.

```
uv sync
uv run python main.py train --render   # watch the swarm learn; hold SPACE to speed up
uv run python main.py watch            # replay winner.pkl until it dies or you close the window
```

`train` runs headless and fast without `--render`. Both commands stop cleanly when you close the window.

## How it works

`game.py` holds the rules and no drawing, `agent.py` the NEAT code, `render.py` the drawing, and `main.py` the CLI.

- **Inputs (5, all scaled to about -1..1).** Bird height, bird vertical speed, horizontal distance to the next pipe, and the vertical distance from the bird to the top and to the bottom of that pipe's gap.
- **Output (1).** A sigmoid. The bird flaps when it is above 0.5.
- **Fitness.** On one course: pipes passed, plus a survival bonus that grows with frames survived and stays below 1. Early genomes pass no pipes, and the bonus still ranks them. A genome reaches fitness `n` only by passing `n` pipes.
- **Ten courses per generation.** Every genome flies the same 10 courses, one after another, and its fitness is the lowest of the 10. Reliability means clearing every course, so the worst course is the score. Course seeds come from `seed * 100_000 + generation * 10 + k`: the courses change every generation, which stops the population memorising one layout, and a given `--seed` still gives the same run. `--render` draws the first course of each generation and runs the other 9 headless.
- **Gap placement.** Each gap sits at most 130 px from the previous one, so a bird that can steer toward the gap can always reach it.
- **Stopping.** A course ends when a bird passes `fitness_threshold` pipes (100). Training ends after a generation in which a genome scores that on all 10 courses, and that genome is saved as `winner.pkl`.

## Tuning `config-feedforward.txt`

The file started from neat-python's `examples/xor/config-feedforward`. The keys most worth changing:

- `pop_size`. Larger populations find a winner in fewer generations and cost more per generation.
- `fitness_threshold`. The pipe count a winner must pass on every course. It is the only place the cap lives.
- `COURSES_PER_GENERATION` in `agent.py`. More courses per generation make the winner less likely to fail on a layout it never saw, and make each generation slower.
- `conn_add_prob`, `node_add_prob`. Higher values grow networks faster. `num_hidden = 0` starts with no hidden nodes.
- `weight_mutate_rate`, `weight_mutate_power`. These set how far a child's weights drift from its parent's.
- `compatibility_threshold`. A lower value splits the population into more species.
