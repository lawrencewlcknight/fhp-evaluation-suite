import copy
from itertools import permutations

import numpy as np
import pyspiel
import pytest

from fhp_evaluation.best_response.adapters import PolicyAdapter
from fhp_evaluation.best_response.flop import FlopBestResponse
from fhp_evaluation.best_response.validation import FixedPolicy
from fhp_evaluation.game import FHP_GAME_PARAMETERS
from fhp_evaluation.strategic_audit.checks import check_adapter
from fhp_evaluation.strategic_audit.commands import create_spec
from fhp_evaluation.strategic_audit.engine import audit_entry
from fhp_evaluation.strategic_audit.library import build_library, canonical_board, board_schedule, validate_library
from fhp_evaluation.strategic_audit.metrics import win_tie_loss
from fhp_evaluation.strategic_audit.report import Accumulator, summarise_decisions
from fhp_evaluation.strategic_audit.tables import BoardContext, build_tables, key, validate_tables
from fhp_evaluation.strategic_audit.validation import validate, scalar_decision
from fhp_evaluation.strategic_audit.workflow import completed, mark, sampled_positions, validate_spec, run


@pytest.fixture
def small_context():
    game = pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))
    return BoardContext(game, (0, 3, 6))


@pytest.fixture(scope="module")
def library():
    return build_library(reference_deals=3)


def test_library_partition_and_isomorphism(library):
    assert len(library["boards"]) == 96
    assert library["partition_counts"] == dict(development=64, assessment=32)
    assert len({r["texture"] for r in library["boards"]}) == 6
    assert len(board_schedule(library, "development")) == 64
    for permutation in permutations(range(4)):
        assert canonical_board((0, 5, 10)) == canonical_board(tuple(4*(c//4) + permutation[c%4] for c in (0, 5, 10)))
    with pytest.raises(ValueError, match="Reference sampling"):
        board_schedule(library, "assessment", True)


def test_library_rejects_leakage(library):
    bad = copy.deepcopy(library)
    heldout = next(r["board"] for r in bad["boards"] if r["partition"] == "assessment")
    private = [c for c in range(52) if c not in heldout][:4]
    bad["reference_deals"][0]["cards"] = private + heldout
    with pytest.raises(ValueError, match="leakage"):
        validate_library(bad)
    bad = copy.deepcopy(library)
    bad["boards"].append(bad["boards"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_library(bad)


def test_independent_all_reduced_deck_decisions():
    result = validate()
    assert result["comparisons"] > 1000
    assert result["max_absolute_error_bb"] < 1e-9


@pytest.mark.parametrize("seat", [0, 1])
def test_full_deck_spot_checks(game, seat):
    context = BoardContext(game, (16, 20, 32))
    candidate, opponent = FixedPolicy(game, "uniform"), FixedPolicy(game, "random")
    result = audit_entry(context, build_tables(context, PolicyAdapter(candidate)),
                         build_tables(context, PolicyAdapter(opponent)), (2, 1), seat)
    for d in result["decisions"][:2]:
        for i in (0, 500):
            ref = scalar_decision(game, candidate, opponent, context.space.board, (2, 1), d.path, seat, context.space.hands[i])
            assert d.local_gap_bb[i] == pytest.approx(ref["local"], abs=1e-9)
            assert d.response_gap_bb[i] == pytest.approx(ref["remaining"], abs=1e-9)


def test_compatible_br_and_fixed_continuation_are_distinct(small_context):
    c = small_context
    candidate, opponent = FixedPolicy(c.game, "uniform"), FixedPolicy(c.game, "random")
    cp, op = build_tables(c, PolicyAdapter(candidate)), build_tables(c, PolicyAdapter(opponent))
    result = audit_entry(c, cp, op, (1, 1), 1)
    old = FlopBestResponse(c.game, opponent).solve(c.space.board, (1, 1), 1)
    np.testing.assert_allclose(result["best_response_bb"], old.weighted_values / old.opponent_mass / 100, atol=1e-12)
    assert any(np.any(d.response_gap_bb > d.local_gap_bb + 1e-5) for d in result["decisions"])


def test_zero_reach_is_not_uniform_fallback(small_context):
    c = small_context
    cp = build_tables(c, PolicyAdapter(FixedPolicy(c.game, "uniform")))
    op = build_tables(c, PolicyAdapter(FixedPolicy(c.game, "fold")))
    result = audit_entry(c, cp, op, (2, 1), 0)
    assert all(np.isnan(d.local_gap_bb).all() for d in result["decisions"])
    acc = Accumulator()
    summarise_decisions(c, (2, 1), result, acc)
    assert acc.summaries()["all"]["count"] == 0
    assert acc.summaries()["all"]["unsupported"] > 0


def test_pot_odds_equals_exact_call_fold_difference(small_context):
    c = small_context
    tables = build_tables(c, PolicyAdapter(FixedPolicy(c.game, "uniform")))
    found = 0
    for seat in (0, 1):
        result = audit_entry(c, tables, tables, (1, 1), seat)
        for d in result["decisions"]:
            assert d.raises_remaining >= 0
            if d.threshold is not None:
                found += 1
                assert d.legal == (0, 1)
                np.testing.assert_allclose(d.q_bb[:, 1] - d.q_bb[:, 0],
                    (d.pot_bb + d.call_cost_bb) * d.equity - d.call_cost_bb, atol=1e-12)
                assert d.raises_remaining == 0
                if d.global_unbeatable.any():
                    assert np.all(d.q_bb[d.global_unbeatable, 1] > d.q_bb[d.global_unbeatable, 0])
    assert found


def test_win_tie_loss_matches_blocker_mass(small_context):
    space = small_context.space
    weights = np.random.default_rng(3).random(len(space.hands))
    wins, ties, losses = win_tie_loss(space, weights)
    np.testing.assert_allclose(wins + ties + losses, space.mass(weights), atol=1e-12)
    np.testing.assert_allclose(wins - losses, space.showdown(weights, 1., 1.), atol=1e-12)


def test_table_validation_rejects_bad_and_illegal_mass(small_context):
    c = small_context
    tables = build_tables(c, PolicyAdapter(FixedPolicy(c.game, "uniform")))
    tables[key((1, 1), ())][0, 0] = .1
    with pytest.raises(ValueError, match="Invalid cached"):
        validate_tables(c, tables)


def test_information_privacy_and_symmetry_checks(game):
    c = BoardContext(game, (16, 20, 32))
    result = check_adapter(c, PolicyAdapter(FixedPolicy(game, "uniform")), require_suit_invariance=True)
    assert result["suit_max_error"] == result["hidden_card_max_error"] == 0
    class Cheater:
        def action_probabilities(self, state, player):
            legal = state.legal_actions()
            enemy = state.history()[2] if player == 0 else state.history()[0]
            return {legal[0 if enemy < 30 else -1]: 1.}
    with pytest.raises(ValueError, match="Hidden-card"):
        check_adapter(c, PolicyAdapter(Cheater()))


def test_noninvariant_policy_is_diagnostic_unless_required(game):
    c = BoardContext(game, (16, 20, 32))
    result = check_adapter(c, PolicyAdapter(FixedPolicy(game, "random")))
    assert result["suit_max_error"] > 0
    with pytest.raises(ValueError, match="suit invariance"):
        check_adapter(c, PolicyAdapter(FixedPolicy(game, "random")), require_suit_invariance=True)


def test_checksum_and_identity_fail_closed(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    mark(tmp_path, "one", "result.json")
    assert completed(tmp_path, "one", "result.json")
    with pytest.raises(ValueError, match="identity"):
        completed(tmp_path, "two", "result.json")
    (tmp_path / "result.json").write_text('{"changed":1}')
    with pytest.raises(ValueError, match="checksum"):
        completed(tmp_path, "one", "result.json")


def test_spec_rejects_duplicate_seed_and_path_traversal():
    spec = create_spec(smoke=True)
    validate_spec(spec)
    spec["policies"]["../evil"] = spec["policies"]["uniform"]
    with pytest.raises(ValueError, match="filesystem-safe"):
        validate_spec(spec)
    spec = create_spec(smoke=True)
    spec["policies"]["duplicate"] = dict(spec["policies"]["uniform"])
    spec["candidates"].append("duplicate")
    with pytest.raises(ValueError, match="Duplicate cohort"):
        validate_spec(spec)


def test_assessment_requires_unlock_before_output(tmp_path, library):
    with pytest.raises(ValueError, match="unlock-assessment"):
        run(create_spec(smoke=True), library, tmp_path / "run", partition="assessment")
    assert not (tmp_path / "run").exists()


def test_resume_rejects_changed_manifest(tmp_path, library):
    import json
    path = tmp_path / "run"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"different": True}))
    with pytest.raises(ValueError, match="identical inputs"):
        run(create_spec(smoke=True), library, path, resume=True)


def test_synthetic_tilt_changes_range_not_candidate(small_context):
    c = small_context
    tables = build_tables(c, PolicyAdapter(FixedPolicy(c.game, "uniform")))
    original = {k: v.copy() for k, v in tables.items()}
    first = audit_entry(c, tables, tables, (1, 1), 1)
    tilted = audit_entry(c, tables, tables, (1, 1), 1, range_tilt=2)
    assert not np.allclose(first["decisions"][0].equity, tilted["decisions"][0].equity)
    for k in tables:
        np.testing.assert_array_equal(tables[k], original[k])


def test_frozen_sampling_reproducible_and_candidate_independent(game):
    c = BoardContext(game, (16, 20, 32))
    tables = build_tables(c, PolicyAdapter(FixedPolicy(game, "call")))
    deal = dict(cards=[0, 4, 8, 12, 16, 20, 32], action_seed=77)
    first = list(sampled_positions(c, deal, tables, tables))
    assert first == list(sampled_positions(c, deal, tables, tables))
    assert {row[1] for row in first} == {0, 1}


def test_native_raw_checkpoint_full_audit_tables(game, tmp_path):
    import torch
    from fhp_evaluation.best_response.adapters import load_target
    path = tmp_path / "policy.pt"
    torch.save(dict(game="FHP", policy_state_dict={
        "layers.0.weight": torch.zeros(8, 190), "layers.0.bias": torch.zeros(8),
        "layers.1.weight": torch.zeros(3, 8), "layers.1.bias": torch.zeros(3)}), path)
    adapter = load_target(game, path, family="raw_mlp")
    context = BoardContext(game, (16, 20, 32))
    check_adapter(context, adapter, require_suit_invariance=True)
    tables = build_tables(context, adapter)
    assert len(tables) > 40
    for name, _, _, _, legal in context.schemas():
        np.testing.assert_allclose(tables[name][:, legal], 1/len(legal), atol=1e-7)


def test_cached_rule_equity_preserves_five_native_agents(game):
    from fhp_evaluation.strategic_audit.tables import accelerate_rule
    from fhp_evaluation.rule_agents import published_rule_agents
    from fhp_evaluation.best_response.flop import make_state
    context = BoardContext(game, (16, 20, 32))
    for original in published_rule_agents(game).values():
        fast = accelerate_rule(context, PolicyAdapter(original))
        # Exact published thresholds and bluff probabilities are NOT reimplemented.
        for i in (0, 70, 500, 1175):
            for pre, post in (((1, 1), ()), ((1, 1), (2,)), ((2, 1), (2, 2, 2))):
                nodes = context.trees[pre]
                node = next(n for n in nodes if n.path == post)
                state = make_state(game, context.deck, context.space.hands[i], node.player,
                                   board=context.space.board, preflop=pre, flop=post)
                actual = fast.action_probabilities(state, node.player)
                expected = original.action_probabilities(state, node.player)
                np.testing.assert_array_equal([actual.get(a, 0.) for a in range(3)],
                                              [expected.get(a, 0.) for a in range(3)])
        # Do not mutate the shared agent's strength function.
        assert fast.native is not original
