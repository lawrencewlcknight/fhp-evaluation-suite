"""Small-deck exact differentials and production protocol regression tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyspiel
import pytest

from fhp_evaluation.best_response.adapters import PolicyAdapter
from fhp_evaluation.best_response.flop import FlopBestResponse, make_state, validate_game
from fhp_evaluation.best_response.validation import FixedPolicy
from fhp_evaluation.duplicate import play_hand as reference_play
from fhp_evaluation.game import FHP_GAME_PARAMETERS
from fhp_evaluation.resolving.experiment import config, tasks
from fhp_evaluation.resolving.policy import ResolvingPolicy, play_hand
from fhp_evaluation.resolving.solver import Deadline, prepare, response_values, certify, solve


@pytest.fixture
def small():
    return pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))


@pytest.mark.parametrize("hero", [0, 1])
@pytest.mark.parametrize("pre", [(1, 1), (2, 1), (2, 2, 2, 1)])
def test_certificate_differential_and_improvement(small, hero, pre):
    target = PolicyAdapter(FixedPolicy(small, "random"))
    p = prepare(small, target, (0, 3, 6), pre, hero, Deadline())
    baseline = response_values(p, p.blueprint, Deadline())
    reference = FlopBestResponse(small, target).solve((0, 3, 6), pre, 1 - hero,
                                                    opponent_reach=p.reach)
    np.testing.assert_allclose(baseline, reference.weighted_values / 100, atol=1e-12)
    _, candidate, stats = solve(small, target, (0, 3, 6), pre, hero, seconds=None, iterations=100)
    assert stats["accepted"] and candidate is not None
    assert certify(p, candidate, baseline, Deadline())["passed"]
    np.testing.assert_array_less(response_values(p, candidate, Deadline()), baseline + 1e-9)
    # Check the new candidate through the older independent tree traversal too,
    # not just by calling its acceptance function twice.
    from fhp_evaluation.best_response.flop import betting_history, observed_cards
    class Candidate:
        def action_probabilities(self, state, player):
            own, _ = observed_cards(small, state, player)
            _, post = betting_history(small, state)
            node_index = next(i for i, n in enumerate(p.nodes) if n.path == post)
            row = candidate[node_index][p.space.lookup[own]]
            return {a: row[j] for j, (a, _) in enumerate(p.nodes[node_index].children)}
    reference = FlopBestResponse(small, Candidate()).solve((0, 3, 6), pre, 1 - hero, opponent_reach=p.reach)
    np.testing.assert_allclose(response_values(p, candidate, Deadline()), reference.weighted_values / 100, atol=1e-12)


def test_reject_bad_candidate_and_invalid_probabilities(small):
    target = PolicyAdapter(FixedPolicy(small, "random"))
    p = prepare(small, target, (0, 3, 6), (1, 1), 0, Deadline())
    baseline = response_values(p, p.blueprint, Deadline())
    always_fold = {}
    for i, v in p.blueprint.items():
        always_fold[i] = np.zeros_like(v)
        always_fold[i][:, 0] = 1
    assert not certify(p, always_fold, baseline, Deadline())["passed"]
    malformed = copy.deepcopy(p.blueprint)
    next(iter(malformed.values()))[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Malformed"):
        response_values(p, malformed, Deadline())


def test_zero_budget_matches_original_simulator(small):
    a, b = [PolicyAdapter(FixedPolicy(small, k)) for k in ("random", "uniform")]
    search = ResolvingPolicy(small, a, seconds=0)
    for seat in (0, 1):
        for seed in range(15):
            original = (a, b) if seat == 0 else (b, a)
            revised = (search, b) if seat == 0 else (b, search)
            expected = reference_play(small, original, chance_seed=seed, action_seed=10 + seed)
            actual, traces = play_hand(small, revised, chance_seed=seed, action_seed=10 + seed)
            assert actual == expected
            assert all(not x["accepted"] for x in traces)


def test_deadline_never_publishes_unchecked_strategy(small):
    a = PolicyAdapter(FixedPolicy(small))
    ticks = iter([i * .01 for i in range(1000)])
    _, candidate, stats = solve(small, a, (0, 3, 6), (1, 1), 0, seconds=.1, clock=lambda: next(ticks))
    assert candidate is None and not stats["accepted"]


def test_opponent_cards_never_enter_solver(small, monkeypatch):
    a = PolicyAdapter(FixedPolicy(small))
    calls = []
    def fixed_solve(game, adapter, board, pre, hero, **kwargs):
        calls.append((board, pre, hero))
        return None, None, dict(accepted=False, reason="fixture")
    monkeypatch.setattr("fhp_evaluation.resolving.policy.solve", fixed_solve)
    search = ResolvingPolicy(small, a)
    for other in ((4, 5), (4, 7)):
        s = small.new_initial_state()
        for action in (1, 2, *other, 1, 1, 0, 3, 6):
            s.apply_action(action)
        search.begin_hand()
        search.observe_flop_entry(s, 0)
    assert calls == [((0, 3, 6), (1, 1), 0)] * 2


def test_protocol_size_and_registry():
    spec = config()
    plan = tasks(spec)
    assert len(plan) == 480
    assert sum(t["count"] * 2 for t in plan) == 240000
    assert sum(t["count"] * 2 for t in plan if t["opponent"] != "blueprint") == 210000
    assert len(spec["checkpoint_registry"]) == 9
    assert len(set(t["id"] for t in plan)) == len(plan)


def test_resume_hash_and_seed_stream_independent_of_shards(tmp_path):
    from fhp_evaluation.cohort import write_json
    from fhp_evaluation.resolving.runner import read_shard, sign_shard, deal_stream
    spec = config()
    a = deal_stream(spec, 0, "blueprint")
    spec["pairs_per_shard"] = 500
    b = deal_stream(spec, 0, "blueprint")
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    t = dict(id="fake", seed=0, opponent="blueprint", start=2, count=2)
    result = dict(task=t, samples=[dict(pair_index=2), dict(pair_index=3)])
    path = tmp_path / "shard.json"
    write_json(path, sign_shard(result, "identity"))
    assert read_shard(path, t, "identity") == result
    with pytest.raises(ValueError, match="Incompatible"):
        read_shard(path, t, "wrong")
    broken = sign_shard(result, "identity")
    broken["payload"]["samples"].pop()
    write_json(path, broken)
    with pytest.raises(ValueError, match="corrupted"):
        read_shard(path, t, "identity")


def test_paired_inference_and_complete_cohort(tmp_path):
    from fhp_evaluation.resolving.runner import summarise, write_report
    spec = config()
    spec.update(pairs_per_match=2, pairs_per_shard=2)
    results = []
    for t in tasks(spec):
        value = float(t["seed"] + 1)
        samples = [dict(pair_index=i, search_seats=[10 + value, 12 + value],
                        baseline_seats=None if t["opponent"] == "blueprint" else [10., 12.]) for i in range(2)]
        results.append(dict(task=t, samples=samples, traces=[], search_gameplay_seconds=1., baseline_gameplay_seconds=.1))
    result = summarise(results, spec)
    assert result["external_panel_uplift_across_training_seeds"]["seed_values"] == [1., 2., 3.]
    assert result["external_panel_uplift_across_training_seeds"]["n"] == 3
    assert result["primary_across_training_seeds"]["mean"] == 13.
    write_report(tmp_path, result)
    assert (tmp_path / "report.md").is_file()
    with pytest.raises(ValueError, match="Incomplete"):
        summarise(results[:-1], spec)


def test_feasibility_gates():
    from fhp_evaluation.resolving.runner import enforce_gate
    spec = config()
    good = dict(attempts=24, acceptance_rate=.5, overrun_rate=0., maximum_seconds=.499)
    enforce_gate(good, spec)
    for key, value in (("attempts", 0), ("acceptance_rate", 0.), ("overrun_rate", .2), ("maximum_seconds", 2.)):
        with pytest.raises(RuntimeError, match="feasibility"):
            enforce_gate(dict(good, **{key: value}), spec)


def test_native_production_network_and_stdlib_cloud_prepare(tmp_path):
    root = Path(__file__).resolve().parents[1]
    native = root.parent / "fhp_ucv_escher/fhp-ucv-escher-experiments"
    if not (native / "fhp_escher/hand_board_features.py").is_file():
        pytest.skip("Native UCV checkout not installed")
    proc = subprocess.run([sys.executable, str(root / "tests/resolving_native_probe.py"), str(native), str(tmp_path)],
                          cwd=root, env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checks = json.loads((tmp_path / "native_probe.json").read_text())
    assert checks["passed"] and checks["preflop_full_range_max_error"] < 1e-7
    output = tmp_path / "batch"
    proc = subprocess.run([sys.executable, "-S", str(root / "gcp/exp23_ucv_flop_resolving_batch.py"),
                           "prepare", "--project", "clever-overview-399515", "--region", "europe-west1",
                           "--bucket", "gs://clever-overview-399515-fhp-escher-results", "--service-account",
                           "fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com", "--run-id",
                           "exp23-search-unittest", "--native-repo", str(native), "--run-dir", str(output)],
                          cwd=root, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    job = json.loads((output / "job.json").read_text())
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    assert job["taskGroups"][0]["taskCount"] == 1
    request = json.loads((output / "request.json").read_text())
    assert len(request["experiment"]["checkpoint_registry"]) == 9
    assert request["source"]["source_only"]
    assert (output / "source.tar.gz").stat().st_size < 16 * 1024**2
