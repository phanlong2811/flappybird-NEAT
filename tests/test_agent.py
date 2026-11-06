import pickle

import neat

import agent

HELD_OUT_SEEDS = [9_000_001, 9_000_002, 9_000_003]


def test_train_converges_and_winner_clears_unseen_courses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cap = agent.load_config().fitness_threshold

    winner = agent.train(generations=50, seed=1)

    assert isinstance(pickle.loads((tmp_path / "winner.pkl").read_bytes()), neat.DefaultGenome)
    assert winner.fitness >= cap

    def stop_at_cap(world, info):
        if world.score >= cap:
            raise agent.Stop

    assert [agent.play_winner(seed, stop_at_cap) for seed in HELD_OUT_SEEDS] == [cap] * len(HELD_OUT_SEEDS)
