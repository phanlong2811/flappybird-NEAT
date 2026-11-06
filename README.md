# NEAT pipe dodger

NEAT (neat-python 2.x) evolves small neural networks that fly a bird through a gap in a never-ending row of pillars. The window uses only pygame shapes, with no image assets.

```
uv sync
uv run python main.py train --seed 1 --render   # watch the swarm learn; hold SPACE to speed up
uv run python main.py watch                     # replay winner.pkl until it dies or you close the window
```

`train` runs headless and fast without `--render`. Both commands stop cleanly when you close the window.

## How it works

`game.py` holds the rules and no drawing, `agent.py` the NEAT code, `render.py` the drawing, and `main.py` the CLI.

- **Inputs (5, all scaled to about -1..1).** Bird height, bird vertical speed, horizontal distance to the next pipe, and the vertical distance from the bird to the top and to the bottom of that pipe's gap.
- **Output (1).** A sigmoid. The bird flaps when it is above 0.5.
- **Fitness.** Pipes passed when the bird died, plus 0.01 per frame survived. Early genomes pass no pipes, and the survival term still ranks them.
- **One shared course per generation.** Every genome flies the same pipes, so scores compare fairly. The course is seeded from `seed + generation`: it changes every generation, which stops the population memorising one layout, and a given `--seed` still gives the same run.
- **Gap placement.** Each gap sits at most 130 px from the previous one, so a bird that can steer toward the gap can always reach it.
- **Stopping.** A generation ends when a bird passes `MAX_SCORE` (100) pipes. `fitness_threshold` in the config equals the fitness of a bird that is still alive at that point, so reaching the cap ends training and saves `winner.pkl`. The winner is the first genome to clear one course. Some seeds produce a winner that fails on other courses, so try another `--seed` if `watch` crashes early.

## Tuning `config-feedforward.txt`

The file started from neat-python's `examples/xor/config-feedforward`. The keys most worth changing:

- `pop_size`. Larger populations find a winner in fewer generations and cost more per generation.
- `fitness_threshold`. It must stay at 100 + 0.01 * (frames needed to pass 100 pipes), about 173.8. Change `MAX_SCORE`, the pipe spacing or the pipe speed in the code and you must recompute it.
- `conn_add_prob`, `node_add_prob`. Higher values grow networks faster. `num_hidden = 0` starts with no hidden nodes.
- `weight_mutate_rate`, `weight_mutate_power`. These set how far a child's weights drift from its parent's.
- `compatibility_threshold`. A lower value splits the population into more species.
