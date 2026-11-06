"""CLI: `train` evolves a winner, `watch` plays it."""
import argparse
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

import agent
import render
from game import HEIGHT, WIDTH

FPS = 60


def make_hook(title: str, extra=lambda info: {}):
    """Build an on_frame hook that draws each frame, stops on window close and runs unthrottled while SPACE is held."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()

    def on_frame(world, info):
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            raise agent.Stop
        alive = sum(b.alive for b in world.birds)
        render.draw(screen, world, {"alive": f"{alive}/{len(world.birds)}", **extra(info)})
        pygame.display.flip()
        if not pygame.key.get_pressed()[pygame.K_SPACE]:
            clock.tick(FPS)

    return on_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="NEAT pipe dodger")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="evolve a network and save winner.pkl")
    train.add_argument("--generations", type=int, default=50)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--render", action="store_true", help="show training live (hold SPACE to speed up)")
    watch = sub.add_parser("watch", help="play winner.pkl in a window")
    watch.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.command == "train":
        hook = None
        if args.render:
            hook = make_hook("NEAT pipe dodger: training", lambda i: {
                "generation": str(i["generation"]), "best fitness": f"{i['best']:.1f}"})
        if agent.train(args.generations, args.seed, hook) is None:
            print("Stopped before the first generation finished; nothing saved.")
    elif not agent.WINNER_PATH.exists():
        print("No winner.pkl found. Run `python main.py train` first.")
    else:
        print(f"Score: {agent.play_winner(args.seed, make_hook('NEAT pipe dodger: best bird'))}")


if __name__ == "__main__":
    main()
