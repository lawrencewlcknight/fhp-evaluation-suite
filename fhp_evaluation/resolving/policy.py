"""A per-hand continuation, solved before either player acts on the flop."""
import time

import numpy as np

from ..best_response.flop import betting_history, observed_cards
from ..duplicate import _normalised_distribution, _sample
from .solver import solve


class ResolvingPolicy:
    def __init__(self, game, blueprint, *, seconds=.5, iterations=10000):
        self.game, self.blueprint = game, blueprint
        self.seconds, self.iterations = seconds, iterations
        self.begin_hand()

    def begin_hand(self):
        self.problem = self.strategy = self.stats = None
        self.hero = self.entry = self.board = None
        self.node_by_path = {}

    def observe_flop_entry(self, state, hero):
        """Public-state callback. Actual private cards are NOT passed to solve.

        Invoke at entry, even when the other player is to act first. Delaying
        until our first action could select different entire continuations
        conditional on that action and invalidate the root certificate.
        """
        if self.stats is not None:
            return
        started = time.perf_counter()
        _, board = observed_cards(self.game, state, hero)
        if len(board) != 3:
            return
        pre, post = betting_history(self.game, state)
        if post:
            raise ValueError("The game driver missed the public flop-entry callback")
        self.hero, self.entry, self.board = hero, tuple(pre), tuple(sorted(board))
        remaining = max(0., self.seconds - (time.perf_counter() - started) - .002)
        self.problem, self.strategy, self.stats = solve(
            self.game, self.blueprint, self.board, self.entry, hero,
            seconds=remaining, iterations=self.iterations)
        self.stats.update(flop_decisions=0, changed_flop_decisions=0, policy_total_variation_sum=0.)
        if self.strategy is not None:
            self.node_by_path = {n.path: i for i, n in enumerate(self.problem.nodes)}
        self.stats["total_seconds"] = time.perf_counter() - started
        self.stats["budget_seconds"] = self.seconds
        if self.stats["total_seconds"] > self.seconds:
            self.strategy = None
            self.stats.update(accepted=False, reason="deadline_overrun")

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        own, board = observed_cards(self.game, state, player)
        if len(board) != 3:
            return self.blueprint.action_probabilities(state, player)
        if self.stats is None:
            raise ValueError("Use the resolving game driver, not a stateless policy evaluator")
        pre, post = betting_history(self.game, state)
        if (player != self.hero or tuple(pre) != self.entry or tuple(sorted(board)) != self.board):
            raise ValueError("Stale continuation: reset the wrapper at every hand")
        self.stats["flop_decisions"] += 1
        if self.strategy is None:
            return self.blueprint.action_probabilities(state, player)
        i = self.node_by_path[tuple(post)]
        hand_index = self.problem.space.lookup[tuple(sorted(own))]
        row = self.strategy[i][hand_index]
        tv = .5 * float(np.abs(row - self.problem.blueprint[i][hand_index]).sum())
        self.stats["policy_total_variation_sum"] += tv
        self.stats["changed_flop_decisions"] += int(tv > 1e-6)
        return {a: float(row[j]) for j, (a, _) in enumerate(self.problem.nodes[i].children)}


def play_hand(game, policies, *, chance_seed, action_seed):
    """Repository duplicate-deal simulator with explicit public-entry callbacks.

    Chance and action streams are independent. Search consumes neither stream.
    Returned traces contain timing/certificates, never opponent hole cards.
    """
    for policy in policies:
        if isinstance(policy, ResolvingPolicy):
            policy.begin_hand()
    chance_rng = np.random.default_rng(int(chance_seed))
    action_rng = np.random.default_rng(int(action_seed))
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            probs = np.asarray([p for _, p in outcomes])
            state.apply_action(outcomes[_sample(probs / probs.sum(), chance_rng)][0])
            continue
        for hero, policy in enumerate(policies):
            if isinstance(policy, ResolvingPolicy) and policy.stats is None:
                policy.observe_flop_entry(state, hero)
        player = state.current_player()
        actions, probs = _normalised_distribution(state, policies[player], player)
        state.apply_action(actions[_sample(probs, action_rng)])
    traces = [dict(p.stats) for p in policies if isinstance(p, ResolvingPolicy) and p.stats is not None]
    return tuple(map(float, state.returns())), traces
