import json
import os
import time

import numpy as np
import pytest

from fhp_evaluation import cohort as c
from fhp_evaluation.game import load_fhp_game
from fhp_evaluation.rule_agents import published_rule_agents


def fake_score(task):
    return dict(task=c.portable(task), elapsed_seconds=1.0, summary=dict(n=2, mean=1.0))


def hung_score(task):
    time.sleep(30)
    return fake_score(task)


def killed_score(task):
    os._exit(137)


def task(kind="rule", identity="a"):
    return dict(id=identity, match_id=identity, kind=kind, deals=2, opponent="test",
                a=dict(id="p", path="/unneeded", sha256="a"*64, experiment="exp1", seed=0, hours=24, nodes=100))


def test_duplicate_reproducibility_and_split_schedule(monkeypatch):
    seen = []
    monkeypatch.setattr(c, "play_hand", lambda game, policies, **kw: (seen.append(kw) or (1, -1)))
    assert c.duplicate_samples(None, "a", "b", 3, 7) == [0, 0, 0]
    rng = np.random.default_rng(7)
    chance = rng.integers(0, 2**63-1, size=3, dtype=np.int64)
    action = rng.integers(0, 2**63-1, size=3, dtype=np.int64)
    assert seen[::2] == seen[1::2] == [dict(chance_seed=int(a), action_seed=int(b)) for a,b in zip(chance,action)]


def test_real_duplicate_play():
    game = load_fhp_game()
    a = published_rule_agents(game)["tight_passive"]
    assert c.duplicate_samples(game, a, a, 3, 9) == [0, 0, 0]


def test_atomic_cache_resume_contract_and_corruption(tmp_path):
    tasks = [task()]
    a = c.execute_tasks(tasks, tmp_path, {"code": 1}, workers=1, seconds=30, scorer=fake_score)
    # No worker is launched when complete, even with a tiny time budget.
    b = c.execute_tasks(tasks, tmp_path, {"code": 1}, workers=1, seconds=.001, scorer=hung_score)
    assert a == b
    tasks[0]["a"]["path"] = "/relocated"
    assert c.execute_tasks(tasks, tmp_path, {"code": 1}, workers=1, seconds=1, scorer=hung_score) == b
    with pytest.raises(ValueError, match="Changed"):
        c.execute_tasks(tasks, tmp_path, {"code": 2}, workers=1, seconds=1)
    path = tmp_path / "task_results/a.json"
    data = json.loads(path.read_text())
    data["result"]["summary"]["mean"] = 2
    c.write_json(path, data)
    with pytest.raises(ValueError, match="Invalid cached"):
        c.execute_tasks(tasks, tmp_path, {"code": 1}, workers=1, seconds=1)


def test_hard_deadline_terminates_workers(tmp_path):
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        c.execute_tasks([task()], tmp_path, {}, workers=1, seconds=.5, scorer=hung_score)
    assert time.monotonic() - start < 10


def test_killed_worker_fails_fast(tmp_path):
    from concurrent.futures.process import BrokenProcessPool
    start = time.monotonic()
    with pytest.raises(BrokenProcessPool):
        c.execute_tasks([task()], tmp_path, {}, workers=1, seconds=60, scorer=killed_score)
    assert time.monotonic() - start < 20


def test_cost_gate_requires_every_lbr_target_and_margin():
    a = task("lbr")
    assert c.cost_projection([dict(task=a, elapsed_seconds=4)], [dict(a, deals=100)], 2) == 720
    other = dict(a, a=dict(a["a"], id="unknown"))
    with pytest.raises(ValueError, match="Unprofiled"):
        c.cost_projection([dict(task=a, elapsed_seconds=4)], [other], 2)


def test_lbr_shards_are_not_training_replications():
    results = []
    for seed in range(3):
        for shard in range(2):
            t = task("lbr", f"s{seed}_shard{shard}")
            t.update(match_id=f"seed{seed}", shard=shard, a=dict(t["a"], seed=seed))
            results.append(dict(task=t, paired_mbb=[seed, seed+2]))
    rows = c.collapse(results)
    assert len(rows) == 3 and all(r["match_n"] == 4 for r in rows)
    summary = c.aggregate(rows)[0]
    assert summary["seed_n"] == 3 and summary["seed_mean"] == 2
    assert summary["seed_se"] == pytest.approx(1 / np.sqrt(3))


def test_pair_differences_within_training_seed():
    rows = [dict(kind="rule", experiment=f"exp{e}", opponent="test", hours=24,
                 earlier_hours=None, seed=s, nodes=1, match_mean=s+e) for e in (1,2,3) for s in range(3)]
    differences = c.paired_differences(rows)
    assert len(differences) == 9
    assert {r["seed_n"] for r in c.aggregate(differences)} == {3}
    assert all(r["seed_se"] == 0 for r in c.aggregate(differences))
