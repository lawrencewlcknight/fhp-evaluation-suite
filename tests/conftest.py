from __future__ import annotations

import pytest

from fhp_evaluation.game import load_fhp_game


@pytest.fixture(scope="session")
def game():
    return load_fhp_game()


def deal_preflop(game, cards=(48, 49, 0, 1)):
    state = game.new_initial_state()
    for card in cards:
        assert state.is_chance_node()
        state.apply_action(card)
    return state
