from itertools import combinations
import json

import numpy as np
import pytest

from fhp_evaluation.best_response.adapters import PolicyAdapter
from fhp_evaluation.best_response.flop import make_state, validate_game
from fhp_evaluation.best_response.validation import FixedPolicy
from fhp_evaluation.game import load_fhp_game
from fhp_evaluation.safeguard.experiment import config, count, tasks, stream_seed, opponents
from fhp_evaluation.safeguard.matches import paired_matches
from fhp_evaluation.safeguard.policy import ExactCallSafeguard, TerminalCallVerifier, rank_code
from fhp_evaluation.safeguard.report import describe, seed_summary
from fhp_evaluation.safeguard.validation import CASES, fixture, validate_verifier, validate_native


@pytest.fixture
def game():
    return load_fhp_game()


def test_exhaustive_independent_payoffs_and_symmetries():
    result = validate_verifier()
    assert result["passed"]
    assert result["independent_payoff_comparisons"] == 21620


def test_ties_and_losing_hands(game):
    verifier = TerminalCallVerifier(game)
    hand, board, _ = CASES[3]
    state = fixture(game, hand, board, 1)
    _, called, folded = verifier.payoffs(state, 1)
    assert np.any(called == 0)  # tied nut straights
    assert np.all(called >= folded)
    hand, board, _ = CASES[4]
    assert not verifier.certificate(fixture(game, hand, board, 1), 1)["dominates"]


def test_only_terminal_call_fold_changes(game):
    base = PolicyAdapter(FixedPolicy(game, "uniform"))
    guard = ExactCallSafeguard(game, base)
    hand, board, _ = CASES[0]
    deck = validate_game(game)
    # Preflop capped, flop nonterminal call/check, flop raise still legal.
    for pre, public, post in (((2,2,2), (), ()), ((1,1), board, ()), ((1,1), board, (2,))):
        state = make_state(game, deck, hand, 1, board=public, preflop=pre, flop=post)
        player = state.current_player()
        assert guard.action_probabilities(state, player) == base.action_probabilities(state, player)
        assert guard.verifier.certificate(state, player) is None
    assert validate_native(game, base)["passed"]


def test_exact_mass_transfer_and_unchanged_weights(game):
    class Example:
        def action_probabilities(self, state, player_id=None):
            return {0: .9972, 1: .0028}
    base = PolicyAdapter(Example())
    guarded = ExactCallSafeguard(game, base)
    winning = fixture(game, *CASES[0][:2], 1)
    losing = fixture(game, *CASES[4][:2], 1)
    before = base.action_probabilities(winning)
    assert guarded.action_probabilities(winning) == {0: 0., 1: 1.}
    assert guarded.action_probabilities(losing) == before
    assert base.action_probabilities(winning) == before
    assert guarded.check_scalar_parity([winning, losing])["passed"]
    # Both response engines must retain the deployed adapter, not unwrap it.
    from fhp_evaluation.best_response.flop import FlopBestResponse, FullFlopResponsePolicy
    assert FlopBestResponse(game, guarded).target is guarded
    assert FullFlopResponsePolicy(game, guarded).target is guarded
    assert guarded.preflop_policy(game, None).target_policy is guarded


def test_bounded_caches_no_hidden_card_dependency(game):
    verifier = TerminalCallVerifier(game, board_cache=1, certificate_cache=2)
    for h, b, _ in CASES:
        verifier.certificate(fixture(game, h, b, 1), 1)
    assert len(verifier.boards) <= 1 and len(verifier.certificates) <= 2
    h, b, _ = CASES[0]
    a = fixture(game, h, b, 1)
    other = tuple(c for c in range(52) if c not in (*h, *b))[-2:]
    z = fixture(game, h, b, 1, other)
    assert verifier.certificate(a, 1) == verifier.certificate(z, 1)
    assert verifier.stats["certificate_hits"]
    json.dumps(verifier.diagnostics(), allow_nan=False)


def test_rank_encoding_order():
    from fhp_evaluation.cards import five_card_rank
    rng = np.random.default_rng(443)
    hands = [rng.choice(52, 5, replace=False) for _ in range(500)]
    assert sorted(range(len(hands)), key=lambda i: rank_code(hands[i])) == sorted(
        range(len(hands)), key=lambda i: five_card_rank(hands[i]))


def test_coupled_games_shard_and_batch_invariance(game):
    base = PolicyAdapter(FixedPolicy(game, "uniform"))
    opponent = PolicyAdapter(FixedPolicy(game, "raise"))
    spec = config("smoke")
    task = dict(seed=0, opponent="raise", start=0, count=100)
    full, _ = paired_matches(game, base, opponent, task, spec)
    pieces = [paired_matches(game, base, opponent, dict(task, start=i, count=20),
                             dict(spec, batch_size=7))[0] for i in range(0, 100, 20)]
    assert np.array_equal(full, np.concatenate(pieces))
    assert np.all(full[:, 2:4] >= 0)
    # Ordinary self-play duplicate scores are exactly zero, not just approximately.
    self_play, _ = paired_matches(game, base, base, task, spec)
    assert np.array_equal(self_play[:, 0], np.zeros(100))


def test_terminal_coupling_equals_two_literal_played_versions(game):
    # Independent simulator with inverse-CDF RNG identical to the paired evaluator.
    base = PolicyAdapter(FixedPolicy(game, "uniform"))
    opponent = PolicyAdapter(FixedPolicy(game, "raise"))
    guard = ExactCallSafeguard(game, base)
    spec = config("smoke")
    task = dict(seed=0, opponent="raise", start=0, count=160)
    rows, _ = paired_matches(game, base, opponent, task, spec)
    def literal(policies, chance_seed, action_seed, seat):
        state = game.new_initial_state()
        chance = np.random.default_rng(chance_seed)
        action = np.random.default_rng(action_seed)
        while not state.is_terminal():
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                chosen = int(chance.choice(acts, p=probs))
            else:
                p = state.current_player()
                raw = policies[p].action_probabilities(state, p)
                probs = [raw.get(a, 0.) for a in range(3)]
                chosen = int(np.searchsorted(np.cumsum(probs), action.random(), side="right"))
            state.apply_action(chosen)
        return state.returns()[seat]
    for index in range(task["count"]):
        rng = np.random.default_rng(stream_seed(spec, 0, "raise", index))
        chance, action = map(int, rng.integers(0, 2**63-1, size=2))
        for col, focal in enumerate((base, guard)):
            value = .5 * 10 * (literal((focal, opponent), chance, action, 0)
                               + literal((opponent, focal), chance, action, 1))
            assert value == rows[index, col]


def test_prespecified_design_and_rare_event_reporting():
    spec = config("full")
    schedule = list(tasks(spec))
    assert spec["seeds"] == [0,1,2]
    assert len({r["id"] for r in schedule}) == len(schedule)
    for seed in spec["seeds"]:
        for opponent in opponents(spec, seed):
            assert sum(r["count"] for r in schedule if r["seed"] == seed and r["opponent"] == opponent) == count(spec, seed, opponent)
    assert sum(r["count"] for r in schedule if r["opponent"] == f"exp16_seed{r['seed']}_48h") == 3_000_000
    empty = describe(np.zeros((10,8)))
    assert empty["zero_event_warning"]
    assert empty["pair_activation_probability_95_upper"] > 0
    assert seed_summary([1,2,3])["n"] == 3
    json.dumps(empty, allow_nan=False)


def test_coupling_when_safeguard_actually_fires(game):
    winning = fixture(game, *CASES[0][:2], 1)
    class ForcedGame:
        def new_initial_state(self):
            return winning.clone()
    base = PolicyAdapter(FixedPolicy(game, "uniform"))
    opponent = PolicyAdapter(FixedPolicy(game, "call"))
    rows, _ = paired_matches(ForcedGame(), base, opponent,
        dict(seed=0, opponent="forced", start=0, count=100), config("smoke"),
        verifier=TerminalCallVerifier(game))
    assert np.all(rows[:, 4] == 1)
    assert np.all(rows[:, 3] == .5 * 10 * .5 * 1300)
    assert set(rows[:, 2]) == {0., .5 * 10 * 1300}


def test_strategic_spec_exposes_safeguard_to_hypothetical_queries():
    from fhp_evaluation.safeguard.runner import strategic_spec
    spec = strategic_spec(config("full"), "/input", "/native")
    assert len(spec["candidates"]) == 6
    assert len(spec["opponents"]) == 8
    assert spec["policies"]["exp16_seed0_48h_guarded"]["deployment"] == "exact_terminal_call_v1"
    assert "deployment" not in spec["policies"]["exp16_seed0_48h"]
