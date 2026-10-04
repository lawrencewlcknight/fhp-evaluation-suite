from __future__ import annotations

from open_spiel.python import policy

from fhp_evaluation.lbr import LBRConfig, LocalBestResponsePolicy, evaluate_lbr

from conftest import deal_preflop


class PreferredActionPolicy(policy.Policy):
    def __init__(self, game, preference):
        super().__init__(game, list(range(game.num_players())))
        self.preference = tuple(preference)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = set(state.legal_actions(player))
        action = next(action for action in self.preference if action in legal)
        return {action: 1.0}


def test_lbr_raises_into_an_always_fold_target(game):
    target = PreferredActionPolicy(game, (0, 1, 2))
    lbr = LocalBestResponsePolicy(
        game, target, config=LBRConfig(preflop_rollout_samples=32, seed=5)
    )
    state = deal_preflop(game)
    assert lbr.action_probabilities(state, 0) == {2: 1.0}


def test_lbr_does_not_observe_the_actual_opponent_hand(game):
    target = PreferredActionPolicy(game, (1, 0, 2))
    config = LBRConfig(preflop_rollout_samples=32, seed=11)
    state_a = deal_preflop(game, cards=(48, 49, 0, 1))
    state_b = deal_preflop(game, cards=(48, 49, 20, 21))
    action_a = LocalBestResponsePolicy(game, target, config=config).action_probabilities(state_a, 0)
    action_b = LocalBestResponsePolicy(game, target, config=config).action_probabilities(state_b, 0)
    assert state_a.information_state_string(0) == state_b.information_state_string(0)
    assert action_a == action_b


def test_lbr_range_update_path_after_an_observed_call(game):
    target = PreferredActionPolicy(game, (1, 0, 2))
    state = deal_preflop(game, cards=(0, 1, 48, 49))
    state.apply_action(1)
    assert state.current_player() == 1
    lbr = LocalBestResponsePolicy(
        game, target, config=LBRConfig(preflop_rollout_samples=16, seed=13)
    )
    probabilities = lbr.action_probabilities(state, 1)
    assert set(probabilities) <= set(state.legal_actions(1))
    assert sum(probabilities.values()) == 1.0


def test_lbr_duplicate_evaluation_is_labelled_as_a_lower_bound(game):
    target = PreferredActionPolicy(game, (0, 1, 2))
    result = evaluate_lbr(
        game,
        target,
        num_deals=2,
        seed=3,
        config=LBRConfig(preflop_rollout_samples=16, seed=3),
        target_name="always_fold",
    )
    assert result["metric"] == "lbr_lower_bound"
    assert "not_exact_exploitability" in result["interpretation"]
    assert result["mean_chips_per_hand"] > 0.0
