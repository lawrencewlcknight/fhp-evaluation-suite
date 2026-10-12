"""Exp27 learning controls: known weak FHP opponents and exact Leduc BR values."""
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch
from torch import nn
from open_spiel.python import policy as spiel_policy
from open_spiel.python.algorithms import best_response, expected_game_score

from .learning import attacker_loss


class TensorEncoder:
    def __init__(self, game):
        self.policy_layout = SimpleNamespace(total_size=game.information_state_tensor_size())

    def information_state(self, state, player):
        return np.asarray(state.information_state_tensor(player), dtype=np.float32)


class ControlPolicy:
    """No hidden-state access: legal uniform, 80%-fold/20%-call, or check/call.

    Always folding would terminate every seat-1 training hand before that
    learner acts. The fixed 20% continuation supplies decisions in both seats.
    """
    def __init__(self, encoder, kind):
        if kind not in ("leduc_uniform", "fhp_fold", "fhp_call"):
            raise ValueError("Unknown control")
        self.feature_encoder, self.kind = encoder, kind
        self.model = nn.Identity()  # Immutable, parameter-free reference.

    def batch(self, states):
        x = np.asarray([self.feature_encoder.information_state(s, s.current_player()) for s in states], np.float32)
        mask = np.asarray([s.legal_actions_mask() for s in states], bool)
        if self.kind == "leduc_uniform":
            values = mask / mask.sum(1)[:, None]
        else:
            values = np.zeros((len(states), 3), dtype=np.float32)
            values[:, 1] = 1.
            if self.kind == "fhp_fold":
                values[mask[:, 0], 0] = .8
                values[mask[:, 0], 1] = .2
        return x, mask, values

    def action_probabilities(self, state, player_id=None):
        if player_id is not None and player_id != state.current_player():
            raise ValueError("Acting-player information only")
        values = self.batch([state])[2][0]
        return {a: float(values[a]) for a in state.legal_actions()}


def leduc_reference():
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    return game, ControlPolicy(TensorEncoder(game), "leduc_uniform")


def tabular(game, actor):
    """Materialise all information sets in batches; never enumerate FHP."""
    if game.get_type().short_name != "leduc_poker":
        raise ValueError("Exact enumeration is restricted to Leduc")
    table = spiel_policy.TabularPolicy(game)
    for start in range(0, len(table.states), 512):
        values = actor.batch(table.states[start:start + 512])[2]
        table.action_probability_array[start:start + len(values)] = values
    return table


def exact_oracle(game, reference):
    opponent = tabular(game, reference)
    root = game.new_initial_state()
    values = [float(best_response.BestResponsePolicy(game, seat, opponent).value(root)) for seat in (0, 1)]
    self_play = expected_game_score.policy_value(root, [opponent, opponent]).tolist()
    return dict(per_seat_chips=values, mean_chips=float(np.mean(values)),
                uniform_self_play_chips=self_play, game=str(game),
                method="OpenSpiel BestResponsePolicy, cut_threshold=0; exact expected values, no deal sampling")


def exact_score(game, actors, reference):
    opponent = tabular(game, reference)
    values = []
    for seat, actor in enumerate(actors):
        policies = [opponent, opponent]
        policies[seat] = tabular(game, actor)
        values.append(float(expected_game_score.policy_value(game.new_initial_state(), policies)[seat]))
    return dict(per_seat_chips=values, mean_chips=float(np.mean(values)))


def stochastic_loss_check():
    """Exact finite-distribution diagnostic, not a claim about FHP's cause.

    Y=-1 with p=.8 and +4 with p=.2 has mean 0. Half-MSE minimises at
    zero; beta=1 Smooth L1 minimises at -.75. Demonstrate this analytically
    and by enumerating the expected loss, without sample/optimizer noise.
    """
    labels = torch.tensor([-1., -1., -1., -1., 4.], dtype=torch.float64)
    grid = torch.linspace(-1, 1, 801, dtype=torch.float64)
    minima = {}
    for kind, expected in (("smooth_l1", -.75), ("half_mse", 0.)):
        losses = torch.stack([attacker_loss(q.expand_as(labels), labels, kind) for q in grid])
        optimum = float(grid[losses.argmin()])
        if abs(optimum - expected) > 1e-10:
            raise ValueError("Loss expectation diagnostic failed")
        minima[kind] = optimum
    return dict(target_mean=0., optima=minima, passed=True,
                caveat="Loss-target mismatch can occur; this does not prove it explains FHP performance")
