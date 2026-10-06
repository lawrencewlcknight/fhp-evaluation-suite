from itertools import combinations
import json
import sys

import numpy as np
import pyspiel
import pytest
import torch

from fhp_evaluation.best_response.adapters import PolicyAdapter, load_target
from fhp_evaluation.best_response.flop import (
    FlopBestResponse, FullFlopResponsePolicy, HandSpace, make_state, validate_game,
)
from fhp_evaluation.best_response.reference import scalar_flop_response, exact_small_game_response
from fhp_evaluation.best_response.validation import FixedPolicy, run_validation
from fhp_evaluation.best_response.runner import run_bounded
from fhp_evaluation.best_response.worker import compare, profile
from fhp_evaluation.game import FHP_GAME_PARAMETERS
from fhp_evaluation.lbr import LBRConfig


@pytest.fixture
def small():
    return pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))


def test_independent_validation():
    result = run_validation()
    assert len(result["records"]) == 42
    assert result["max_absolute_error"] < 1e-8
    assert not result["full_game_fhp_exploitability_implemented"]


@pytest.mark.parametrize("board", [(0, 3, 6), (1, 2, 7)])
def test_terminal_vectors_against_dense_scalar(small, board):
    space = HandSpace(small, validate_game(small), board)
    weights = np.random.default_rng(21).random(len(space.hands))
    for i, own in enumerate(space.hands):
        mass, payoff = 0., 0.
        for j, other in enumerate(space.hands):
            if set(own) & set(other):
                continue
            state = small.new_initial_state()
            for card in (*own, *other):
                state.apply_action(int(card))
            state.apply_action(1)
            state.apply_action(1)
            for card in board:
                state.apply_action(card)
            state.apply_action(1)
            state.apply_action(1)
            mass += weights[j]
            payoff += weights[j] * state.returns()[0]
        assert space.mass(weights)[i] == pytest.approx(mass, abs=1e-10)
        assert space.showdown(weights, 100., 100.)[i] == pytest.approx(payoff, abs=1e-10)


@pytest.mark.parametrize("player", [0, 1])
def test_full_deck_spot_matches_openspiel_returns(game, player):
    target = FixedPolicy(game, "random")
    result = FlopBestResponse(game, target).solve((16, 20, 32), (2, 1), player)
    for index in (0, len(result.hands) // 2):
        expected = scalar_flop_response(game, target, result.board, result.preflop,
                                        player, result.hands[index])
        assert result.weighted_values[index] == pytest.approx(expected["weighted_value"], abs=1e-7)
        assert result.opponent_mass[index] == pytest.approx(expected["opponent_mass"], abs=1e-9)


def test_zero_reach_still_defines_a_legal_response(small):
    engine = FlopBestResponse(small, FixedPolicy(small, "fold"))
    result = engine.solve((0, 3, 6), (1, 1), 0, opponent_reach=np.zeros(10))
    np.testing.assert_array_equal(result.weighted_values, 0.)
    assert len(result.actions) == 4


def test_hidden_cards_do_not_change_actions_and_cache_is_bounded(game):
    target = PolicyAdapter(FixedPolicy(game, "random"))
    response = FullFlopResponsePolicy(game, target, cache_boards=1,
                                      config=LBRConfig(preflop_rollout_samples=4))
    def state(other, board=(16, 20, 32)):
        s = game.new_initial_state()
        for a in (*other, 0, 4, 1, 1, *board):
            s.apply_action(a)
        return s
    first, second = state((8, 12)), state((24, 28))
    assert first.information_state_string(1) == second.information_state_string(1)
    assert response.action_probabilities(first) == response.action_probabilities(second)
    response.action_probabilities(state((8, 12), (17, 21, 33)))
    assert len(response.cache) == 1


def test_batch_bound_legality_and_scalar_parity(game):
    target = PolicyAdapter(FixedPolicy(game), batch_size=2)
    states = [make_state(game, tuple(range(52)), h, 0) for h in ((0, 4), (1, 5), (2, 6))]
    assert target.check_scalar_parity(states)["passed"]
    assert target.stats["batches"] == 2
    class Illegal:
        def action_probabilities(self, *_):
            return {99: 1.}
    with pytest.raises(ValueError, match="Illegal"):
        PolicyAdapter(Illegal()).batch_probabilities(states, 0)
    class InvalidReference:
        def action_probabilities(self, state, player):
            return {a: float("nan") for a in state.legal_actions(player)}
    with pytest.raises(ValueError, match="non-finite"):
        PolicyAdapter(FixedPolicy(game), reference=InvalidReference()).check_scalar_parity(states)
    class Sampled:
        selected_iterations = {0: 1}
    with pytest.raises(ValueError, match="behavioural"):
        PolicyAdapter(Sampled())


def test_raw_checkpoint_batch_parity_and_encoded_rejection(game, tmp_path):
    path = tmp_path / "raw.pt"
    payload = dict(game="FHP", policy_state_dict={
        "layers.0.weight": torch.randn(8, 190), "layers.0.bias": torch.randn(8),
        "layers.1.weight": torch.randn(3, 8), "layers.1.bias": torch.randn(3)})
    torch.save(payload, path)
    target = load_target(game, path, family="raw_mlp", batch_size=2)
    states = [make_state(game, tuple(range(52)), h, 0) for h in ((0, 4), (1, 5), (2, 6))]
    assert target.check_scalar_parity(states)["passed"]
    payload["feature_encoder"] = {"name": "unsupported"}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unencoded"):
        load_target(game, path, family="raw_mlp")


def test_reject_wrong_game_and_invalid_cards(game):
    with pytest.raises(ValueError, match="universal_poker"):
        FlopBestResponse(pyspiel.load_game("kuhn_poker"), None)
    wrong = pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, maxRaises="1 1"))
    with pytest.raises(ValueError, match="contract"):
        FlopBestResponse(wrong, None)
    with pytest.raises(ValueError, match="distinct"):
        HandSpace(game, tuple(range(52)), (0, 0, 1))
    with pytest.raises(ValueError, match="state limit"):
        exact_small_game_response(game, FixedPolicy(game), 0, max_states=100)
    with pytest.raises(ValueError, match="Invalid"):
        make_state(game, tuple(range(52)), (0, 99), 0)
    with pytest.raises(ValueError, match="Illegal preflop"):
        FlopBestResponse(game, FixedPolicy(game)).solve((0, 3, 6), (99,), 0)


def test_paired_comparison_is_explicitly_not_exact(game, tmp_path):
    target = PolicyAdapter(FixedPolicy(game, "uniform"))
    partial = tmp_path / "partial.json"
    result = compare(game, target, dict(deals=2, seed=24, search_seed=91, rollouts=2), progress_path=partial)
    assert not result["exact_full_game"]
    assert result["metric"] == "exploitability_lower_bound_estimate"
    assert result["paired_improvement"]["mean"] == pytest.approx(
        result["full_flop"]["mean"] - result["lbr"]["mean"])
    progress = json.loads(partial.read_text())
    assert progress["completed_pairs"] == 2
    assert not progress["complete"]
    assert progress["pair_payoffs_mbb"] == result["pair_payoffs_mbb"]
    assert "ci95_low" not in progress


def test_paired_comparison_resumes_exact_completed_prefix(game, tmp_path, monkeypatch):
    target = PolicyAdapter(FixedPolicy(game, "uniform"))
    partial = tmp_path / "partial.json"
    spec = dict(deals=2, seed=2401, search_seed=91, rollouts=2, resume_partial=True)
    first = compare(game, target, spec, progress_path=partial)
    monkeypatch.setattr("fhp_evaluation.best_response.worker.play_hand",
                        lambda *_, **__: pytest.fail("completed pairs were replayed"))
    second = compare(game, target, spec, progress_path=partial)
    assert second["pair_payoffs_mbb"] == first["pair_payoffs_mbb"]


def test_paired_comparison_refuses_partial_from_another_stream(game, tmp_path):
    target = PolicyAdapter(FixedPolicy(game, "uniform"))
    partial = tmp_path / "partial.json"
    compare(game, target, dict(deals=1, seed=24, search_seed=91, rollouts=2,
                               resume_partial=True), progress_path=partial)
    with pytest.raises(ValueError, match="identity"):
        compare(game, target, dict(deals=1, seed=25, search_seed=91, rollouts=2,
                                   resume_partial=True), progress_path=partial)


def test_production_resumes_interrupted_prefix_with_identical_scores(game, tmp_path, monkeypatch):
    from fhp_evaluation.best_response import worker
    partial = tmp_path / "partial.json"
    spec = dict(deals=3, seed=20261007, search_seed=731, rollouts=2,
                resume_partial=True, record_pair_timings=True)
    expected = compare(game, PolicyAdapter(FixedPolicy(game, "uniform")), spec)
    original_play = worker.play_hand
    calls = []
    def interrupt_after_one_pair(*args, **kwargs):
        if len(calls) == 4:
            raise RuntimeError("simulated interruption")
        calls.append(True)
        return original_play(*args, **kwargs)
    monkeypatch.setattr(worker, "play_hand", interrupt_after_one_pair)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        compare(game, PolicyAdapter(FixedPolicy(game, "uniform")), spec, progress_path=partial)
    prefix = json.loads(partial.read_text())
    assert prefix["completed_pairs"] == 1 and len(prefix["pair_elapsed_seconds"]) == 1
    monkeypatch.setattr(worker, "play_hand", original_play)
    actual = compare(game, PolicyAdapter(FixedPolicy(game, "uniform")), spec, progress_path=partial)
    assert actual["pair_payoffs_mbb"] == expected["pair_payoffs_mbb"]
    assert actual["seat_payoffs_chips"] == expected["seat_payoffs_chips"]
    assert actual["pair_elapsed_seconds"][0] == prefix["pair_elapsed_seconds"][0]
    assert len(actual["pair_elapsed_seconds"]) == 3


@pytest.mark.parametrize("kind,code,limit,seconds", [
    ("time_limit", "import time; time.sleep(10)", 2048, .3),
    ("memory_limit", "import time; x=bytearray(100*1024**2); time.sleep(10)", 40, 10),
    ("failed", "raise RuntimeError('intentional failure')", 2048, 10),
])
def test_watchdog_and_failure_diagnostics(tmp_path, kind, code, limit, seconds):
    root = tmp_path / kind
    result = run_bounded(dict(mode="test"), root, max_seconds=seconds,
                         max_rss_mb=limit, command=[sys.executable, "-c", code])
    assert result["status"] == kind
    assert json.loads((root / "run.json").read_text())["status"] == kind
    assert (root / "resources.jsonl").is_file()
    if kind == "failed":
        assert "intentional failure" in (root / "stderr.log").read_text()


def test_watchdog_success_resume_and_identity_guard(tmp_path):
    # Actual internal worker exercises subprocess imports, parity and result persistence.
    spec = dict(mode="profile", family="control", threads=1, control="uniform",
                batch_size=64, model_batch_size=2, boards=1, repeats=1, seed=21)
    root = tmp_path / "profile"
    result = run_bounded(spec, root, max_seconds=60, max_rss_mb=2048)
    assert result["status"] == "succeeded", (root / "stderr.log").read_text()
    assert result["result"]["scope"] == "timing_only_not_exploitability"
    assert run_bounded(spec, root, max_seconds=60, max_rss_mb=2048, resume=True) == result
    with pytest.raises(FileExistsError):
        run_bounded(dict(spec, seed=22), root, max_seconds=60, max_rss_mb=2048, resume=True)


def test_worker_launch_failure_is_recorded(tmp_path):
    root = tmp_path / "launch_failure"
    result = run_bounded(dict(mode="test"), root,
                         command=[str(tmp_path / "nonexistent_executable")])
    assert result["status"] == "failed"
    assert "launch failed" in result["error"]
    assert json.loads((root / "run.json").read_text())["status"] == "failed"
