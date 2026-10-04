from __future__ import annotations

import pytest

from fhp_evaluation.rule_agents import (
    CALL,
    FOLD,
    RAISE,
    HandStrengthPolicy,
    StrengthBands,
    TightAggressivePolicy,
    published_rule_agents,
)

from conftest import deal_preflop


def test_reversed_strength_bands_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        StrengthBands(-100.0, -300.0)


def test_loose_aggressive_transposition_is_corrected(game):
    agent = published_rule_agents(game)["loose_aggressive"]
    assert agent.bands == StrengthBands(-300.0, -100.0)
    assert agent.action_for_strength(-400.0, (FOLD, CALL, RAISE)) == FOLD
    assert agent.action_for_strength(-200.0, (FOLD, CALL, RAISE)) == CALL
    assert agent.action_for_strength(0.0, (FOLD, CALL, RAISE)) == RAISE


@pytest.mark.parametrize(
    ("bands", "strength", "expected"),
    [
        (StrengthBands(-100, 100), -101, FOLD),
        (StrengthBands(-100, 100), -100, CALL),
        (StrengthBands(-100, 100), 99, CALL),
        (StrengthBands(-100, 100), 100, RAISE),
    ],
)
def test_band_boundaries(game, bands, strength, expected):
    agent = HandStrengthPolicy(game, bands, name="test")
    assert agent.action_for_strength(strength, (FOLD, CALL, RAISE)) == expected


def test_tight_aggressive_weak_hand_bluff_distribution(game):
    state = deal_preflop(game)
    agent = TightAggressivePolicy(game, strength_fn=lambda private, public: -200.0)
    assert agent.action_probabilities(state, 0) == {RAISE: 0.2, FOLD: 0.8}


def test_all_five_agents_return_normalised_legal_probabilities(game):
    state = deal_preflop(game)
    agents = published_rule_agents(game)
    assert len(agents) == 5
    for agent in agents.values():
        probabilities = agent.action_probabilities(state, 0)
        assert set(probabilities) <= set(state.legal_actions(0))
        assert sum(probabilities.values()) == pytest.approx(1.0)

