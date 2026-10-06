"""Independent scalar OpenSpiel oracle: no vector rank/blocker calculations."""

from itertools import combinations
import numpy as np
import pyspiel

from ..best_response.adapters import PolicyAdapter
from ..best_response.validation import FixedPolicy
from ..game import FHP_GAME_PARAMETERS
from .engine import audit_entry
from .tables import BoardContext, build_tables


def scalar_decision(game, candidate, opponent, board, pre, post, seat, hand):
    rows = []
    for other in combinations([c for c in game.new_initial_state().legal_actions()
                                if c not in set(board) | set(hand)], 2):
        state = game.new_initial_state()
        for c in (*hand, *other) if seat == 0 else (*other, *hand):
            state.apply_action(int(c))
        weight = 1.
        for a in pre:
            if state.current_player() == 1 - seat:
                weight *= opponent.action_probabilities(state, 1 - seat).get(a, 0.)
            state.apply_action(a)
        for c in board:
            state.apply_action(int(c))
        for a in post:
            if state.current_player() == 1 - seat:
                weight *= opponent.action_probabilities(state, 1 - seat).get(a, 0.)
            state.apply_action(a)
        rows.append((state, weight))
    mass = sum(w for _, w in rows)
    if mass == 0:
        return None
    def walk(hypotheses, optimise):
        first = hypotheses[0][0]
        if first.is_terminal():
            return sum(w * s.returns()[seat] for s, w in hypotheses)
        if first.current_player() == seat:
            # Sum hidden worlds before choosing a single information-set action.
            values = {a: walk([(s.child(a), w) for s, w in hypotheses], optimise)
                      for a in first.legal_actions()}
            if optimise:
                return max(values.values())
            probabilities = candidate.action_probabilities(first, seat)
            return sum(probabilities.get(a, 0.) * v for a, v in values.items())
        return sum(walk([(s.child(a), w * opponent.action_probabilities(s, 1 - seat).get(a, 0.))
                         for s, w in hypotheses], optimise) for a in first.legal_actions())
    q = {a: walk([(s.child(a), w) for s, w in rows], False) / mass / 100
         for a in rows[0][0].legal_actions()}
    p = candidate.action_probabilities(rows[0][0], seat)
    current = sum(p.get(a, 0.) * v for a, v in q.items())
    return dict(q=q, local=max(q.values()) - current,
                remaining=walk(rows, True) / mass / 100 - current, mass=mass)


def validate():
    game = pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))
    count, maximum = 0, 0.
    for board in ((0, 3, 6), (1, 2, 7)):
        context = BoardContext(game, board)
        candidate = FixedPolicy(game, "random")
        cp = build_tables(context, PolicyAdapter(candidate))
        for kind in ("uniform", "random", "call", "fold"):
            opponent = FixedPolicy(game, kind)
            op = build_tables(context, PolicyAdapter(opponent))
            for pre in context.entries:
                for seat in (0, 1):
                    actual = audit_entry(context, cp, op, pre, seat)
                    for decision in actual["decisions"]:
                        for i, hand in enumerate(context.space.hands):
                            expected = scalar_decision(game, candidate, opponent, board, pre, decision.path, seat, hand)
                            count += 1
                            if expected is None:
                                if np.isfinite(decision.local_gap_bb[i]):
                                    raise AssertionError("Zero-reach position was scored")
                                continue
                            errors = [abs(decision.q_bb[i, a] - v) for a, v in expected["q"].items()]
                            errors += [abs(decision.local_gap_bb[i] - expected["local"]),
                                       abs(decision.response_gap_bb[i] - expected["remaining"])]
                            maximum = max(maximum, *errors)
                            if max(errors) > 1e-9:
                                raise AssertionError(f"Independent scalar mismatch: {board, pre, seat, decision.path, hand, errors}")
    return dict(status="passed", comparisons=count, max_absolute_error_bb=maximum,
                oracle="explicit OpenSpiel hidden-hand enumeration, fixed and optimal continuation",
                exact_whole_game=False)
