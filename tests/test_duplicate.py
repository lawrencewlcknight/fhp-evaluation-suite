from __future__ import annotations

from open_spiel.python import policy
import pytest

from fhp_evaluation.duplicate import evaluate_duplicate_match


class PreferredActionPolicy(policy.Policy):
    def __init__(self, game, preference):
        super().__init__(game, list(range(game.num_players())))
        self.preference = tuple(preference)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = set(state.legal_actions(player))
        action = next(action for action in self.preference if action in legal)
        return {action: 1.0}


def test_duplicate_self_play_cancels_exactly(game):
    caller = PreferredActionPolicy(game, (1, 0, 2))
    result = evaluate_duplicate_match(game, caller, caller, num_deals=20, seed=7)
    assert result.mean_chips_per_hand == 0.0
    assert result.std_chips_per_pair == 0.0


def test_duplicate_is_reproducible_and_antisymmetric(game):
    caller = PreferredActionPolicy(game, (1, 0, 2))
    raiser = PreferredActionPolicy(game, (2, 1, 0))
    first = evaluate_duplicate_match(game, caller, raiser, num_deals=30, seed=19)
    repeated = evaluate_duplicate_match(game, caller, raiser, num_deals=30, seed=19)
    reverse = evaluate_duplicate_match(game, raiser, caller, num_deals=30, seed=19)
    assert first == repeated
    assert first.mean_chips_per_hand == pytest.approx(-reverse.mean_chips_per_hand)

