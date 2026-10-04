"""Independent, deliberately small-game-only correctness oracles.

These enumerate explicit OpenSpiel states and are NOT production FHP engines.
The state cap is enforced before a full tree can accidentally be constructed.
"""

from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np

from ..duplicate import _normalised_distribution


def exact_small_game_response(game, target, responder, *, max_states=100_000):
    states, children, probabilities, reaches = [], [], [], []
    groups = defaultdict(list)

    def collect(state, reach):
        if len(states) >= max_states:
            raise ValueError("Small-game reference state limit exceeded")
        index = len(states)
        states.append(state)
        children.append({})
        probabilities.append({})
        reaches.append(reach)
        if state.is_terminal():
            return index
        if state.is_chance_node():
            distribution = dict(state.chance_outcomes())
        elif state.current_player() == responder:
            groups[state.information_state_string(responder)].append(index)
            distribution = {a: 1. for a in state.legal_actions()}
        else:
            actions, values = _normalised_distribution(state, target, state.current_player())
            distribution = dict(zip(actions, values))
        probabilities[index] = distribution
        for action, probability in distribution.items():
            children[index][action] = collect(state.child(action), reach * probability)
        return index

    collect(game.new_initial_state(), 1.)

    @lru_cache(None)
    def decision(info):
        members = groups[info]
        actions = tuple(children[members[0]])
        if any(tuple(children[j]) != actions for j in members):
            raise ValueError("Inconsistent legal actions within an information set")
        return max(actions, key=lambda a: (sum(reaches[j] * value(children[j][a])
                                               for j in members), -a))

    @lru_cache(None)
    def value(index):
        state = states[index]
        if state.is_terminal():
            return state.returns()[responder]
        if not state.is_chance_node() and state.current_player() == responder:
            return value(children[index][decision(state.information_state_string(responder))])
        return sum(p * value(children[index][a]) for a, p in probabilities[index].items())

    return dict(value=float(value(0)), states=len(states), information_sets=len(groups))


def scalar_flop_response(game, target, board, preflop, responder, private):
    """One known hand; enumerate unknown hands and all betting via OpenSpiel.

This is independently computed from terminal returns, not HandSpace's rank or
blocker formulas. Suitable for reduced decks or a few full-deck spot checks.
"""
    excluded = set(board) | set(private)
    deck = game.new_initial_state().legal_actions()
    opponent = 1 - responder
    hypotheses = []
    for other in combinations([c for c in deck if c not in excluded], 2):
        state = game.new_initial_state()
        holes = tuple(private) + other if responder == 0 else other + tuple(private)
        for card in holes:
            state.apply_action(int(card))
        weight = 1.
        for action in preflop:
            if state.current_player() == opponent:
                legal, probabilities = _normalised_distribution(state, target, opponent)
                weight *= probabilities[legal.index(action)]
            state.apply_action(int(action))
        for card in board:
            state.apply_action(int(card))
        hypotheses.append((state, weight))

    def walk(rows):
        first = rows[0][0]
        if first.is_terminal():
            return sum(weight * state.returns()[responder] for state, weight in rows)
        if first.is_chance_node():
            raise ValueError("Unexpected future chance")
        if first.current_player() == responder:
            assert len({s.information_state_string(responder) for s, _ in rows}) == 1
            return max(walk([(state.child(a), w) for state, w in rows])
                       for a in first.legal_actions())
        output = 0.
        probabilities = [dict(zip(*_normalised_distribution(s, target, opponent))) for s, _ in rows]
        for action in first.legal_actions():
            output += walk([(s.child(action), w * p[action])
                            for (s, w), p in zip(rows, probabilities)])
        return output
    return dict(weighted_value=float(walk(hypotheses)),
                opponent_mass=float(sum(w for _, w in hypotheses)))
