"""Fail-fast checks using real OpenSpiel histories and independent scalar queries."""
import numpy as np
import torch
from torch import nn

from ..duplicate import play_hand
from ..game import BIG_BLIND_CHIPS
from .learning import NetworkPolicy, TransitionReplay, attacker_model, collect, double_dqn_targets, model_hash


class RecordedGame:
    def __init__(self, game):
        self.game, self.states = game, []

    def new_initial_state(self):
        state = self.game.new_initial_state()
        self.states.append(state)
        return state


class FixedPolicy:
    def __init__(self, encoder, action):
        self.feature_encoder, self.action = encoder, action

    def batch(self, states):
        x = np.asarray([self.feature_encoder.information_state(s, s.current_player()) for s in states], np.float32)
        masks = np.asarray([s.legal_actions_mask() for s in states], bool)
        actions = [self.action if m[self.action] else 1 for m in masks]
        return x, masks, np.eye(3)[actions]

    def action_probabilities(self, state, player_id=None):
        p = self.batch([state])[2][0]
        return {a: float(p[a]) for a in state.legal_actions()}


def verify_collector(game, learner, opponent, seed=261):
    """Replay every collected history with scalar inference; check reward and TD rows.

    Histories are simulator-only diagnostic data, never features or labels other
    than ordinary terminal reward. This tests the actual production collector.
    """
    recording = RecordedGame(game)
    seats = [i % 2 for i in range(32)]
    episodes = collect(recording, learner, [opponent] * 32, seats, np.random.default_rng(seed))
    nodes = 0
    for end, episode, seat in zip(recording.states, episodes, seats):
        np.testing.assert_allclose(episode.reward, end.returns()[seat] / BIG_BLIND_CHIPS, atol=0)
        np.testing.assert_allclose(sum(end.returns()), 0, atol=1e-10)
        state, rows = game.new_initial_state(), []
        for action in end.history():
            if not state.is_chance_node():
                actor = learner if state.current_player() == seat else opponent
                x, mask, probabilities = actor.batch([state])
                scalar = actor.action_probabilities(state, state.current_player())
                np.testing.assert_allclose(probabilities[0], [scalar.get(a, 0.) for a in range(3)], atol=3e-6)
                if state.current_player() == seat:
                    rows.append((x[0], mask[0], action))
                nodes += 1
            state.apply_action(action)
        np.testing.assert_array_equal(state.returns(), end.returns())
        if len(rows) != len(episode.rows):
            raise ValueError("Collector omitted a learning decision")
        for actual, expected in zip(episode.rows, rows):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
        if rows:
            replay = TransitionReplay(len(rows), len(rows[0][0]))
            replay.append([episode])
            np.testing.assert_array_equal(replay.reward, [0.] * (len(rows) - 1) + [episode.reward])
            for j in range(len(rows) - 1):
                np.testing.assert_array_equal(replay.next_x[j], rows[j + 1][0])
                np.testing.assert_array_equal(replay.next_mask[j], rows[j + 1][1])
            if not replay.done[-1] or replay.next_mask[-1].any() or replay.next_x[-1].any():
                raise ValueError("Invalid terminal transition")
    return dict(hands=len(episodes), decisions=nodes, reward_unit="big blinds")


def correctness_checks(game, target):
    encoder = target.native.feature_encoder
    reference = NetworkPolicy(target.native.model, encoder)
    before = model_hash(reference.model)
    attacker = NetworkPolicy(attacker_model(encoder.policy_layout.total_size, 26), encoder, greedy=True)
    audits = [verify_collector(game, attacker, reference), verify_collector(game, reference, attacker, 262)]
    # Known net blind payoffs test both seat signs and chip -> BB conversion.
    folder, caller, raiser = (FixedPolicy(encoder, a) for a in (0, 1, 2))
    for policies, expected in (((folder, caller), (-50., 50.)), ((raiser, folder), (100., -100.))):
        actual = play_hand(game, policies, chance_seed=26, action_seed=27)
        np.testing.assert_array_equal(actual, expected)
        recording = RecordedGame(game)
        eps = collect(recording, policies[0], [policies[1]], [0], np.random.default_rng(26))
        np.testing.assert_allclose(eps[0].reward, expected[0] / 100)
    # Showdown rewards exercise both winners, not just fold-only paths.
    audits.append(verify_collector(game, caller, caller, 263))
    # Independent native loader vs collection wrapper on a diverse state batch.
    recording = RecordedGame(game)
    collect(recording, reference, [reference] * 64, [i % 2 for i in range(64)], np.random.default_rng(264))
    states = []
    for terminal in recording.states:
        state = game.new_initial_state()
        for action in terminal.history():
            if not state.is_chance_node():
                states.append(state.clone())
            state.apply_action(action)
    batch = reference.batch(states)[2]
    errors = []
    for i, state in enumerate(states):
        native = target.action_probabilities(state, state.current_player())
        errors.append(float(np.max(np.abs(batch[i] - [native.get(a, 0.) for a in range(3)]))))
    if max(errors) > 3e-6:
        raise ValueError("Collector reference differs from native evaluation policy")
    # Hidden-opponent-card invariance for both players at their first decision.
    for hero, deals in ((0, ((0, 4, 8, 12), (0, 4, 16, 20))),
                        (1, ((8, 12, 0, 4), (16, 20, 0, 4)))):
        queries = []
        for cards in deals:
            s = game.new_initial_state()
            for c in cards:
                s.apply_action(c)
            if hero == 1:
                s.apply_action(1)
            queries.append(s)
        if queries[0].information_state_string(hero) != queries[1].information_state_string(hero):
            raise ValueError("Invalid hidden-card diagnostic construction")
        np.testing.assert_array_equal(*[encoder.information_state(s, hero) for s in queries])
        np.testing.assert_array_equal(*attacker.batch(queries)[2])
    online, target_q = nn.Linear(1, 3), nn.Linear(1, 3)
    with torch.no_grad():
        online.weight.zero_(); target_q.weight.zero_()
        online.bias.copy_(torch.tensor([999., 5., 2.]))
        target_q.bias.copy_(torch.tensor([999., 3., 8.]))
    desired = double_dqn_targets(online, target_q, torch.zeros(2, 1),
                                torch.tensor([[False, True, True], [False, False, False]]),
                                torch.tensor([1., -2.]), torch.tensor([False, True]))
    np.testing.assert_array_equal(desired.numpy(), [4., -2.])
    if model_hash(reference.model) != before:
        raise ValueError("Diagnostic changed frozen defender")
    return dict(passed=True, collector_audits=audits, known_fold_rewards_chips=[[-50, 50], [100, -100]],
                native_parity_states=len(states), native_parity_max_error=max(errors),
                hidden_cards_checked_for_both_seats=True, double_dqn_terminal_and_mask_checks=True,
                source_model_hash=before, note="Correctness checks, not evidence of attacker strength")
