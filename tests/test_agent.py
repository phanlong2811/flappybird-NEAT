import pickle

import neat

import agent


def test_train_reaches_cap_and_winner_generalises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []  # (generation, score) for every frame

    def record(world, info):
        seen.append((info["generation"], world.score))

    agent.train(generations=50, seed=1, on_frame=record)

    first_cap = next(gen for gen, score in seen if score >= agent.MAX_SCORE)
    assert first_cap < 50
    assert seen[-1][0] == first_cap  # reaching the cap ended training
    assert isinstance(pickle.loads((tmp_path / "winner.pkl").read_bytes()), neat.DefaultGenome)

    def stop_at_50(world, info):
        if world.score >= 50:
            raise agent.Stop

    assert agent.play_winner(seed=987654, on_frame=stop_at_50) >= 50
