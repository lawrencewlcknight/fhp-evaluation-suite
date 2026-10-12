"""Independent OpenSpiel certification checks, mandatory before cloud evaluation."""
from itertools import permutations

import numpy as np

from ..best_response.adapters import PolicyAdapter
from ..best_response.flop import make_state, validate_game
from ..best_response.validation import FixedPolicy
from ..game import load_fhp_game
from .policy import ExactCallSafeguard, TerminalCallVerifier

CASES = (
    ((2, 14), (6, 10, 50), True),       # wheel straight flush
    ((48, 25), (24, 26, 27), True),    # quads, ace side card
    ((12, 25), (24, 26, 27), True),    # same quads, five side card
    ((28, 32), (16, 21, 26), True),    # nut straight, ties possible
    ((0, 5), (16, 21, 26), False),     # weak holding on the same board
)
PRE = (1, 2, 2, 2, 1)


def fixture(game, hand, board, player, other=None):
    post = (2, 2, 2) if player == 0 else (1, 2, 2, 2)
    if other is None:
        return make_state(game, validate_game(game), hand, player, board=board, preflop=PRE, flop=post)
    state = game.new_initial_state()
    holes = (*hand, *other) if player == 0 else (*other, *hand)
    for action in (*holes, *PRE, *board, *post):
        if action not in state.legal_actions():
            raise AssertionError("Invalid verification fixture")
        state.apply_action(int(action))
    return state


def validate_verifier():
    game = load_fhp_game()
    verifier = TerminalCallVerifier(game)
    records = []
    for hand, board, expected in CASES:
        for player in (0, 1):
            state = fixture(game, hand, board, player)
            holdings, call, fold = verifier.payoffs(state, player)
            error = 0.
            for i, other in enumerate(holdings):
                varied = fixture(game, hand, board, player, other)
                error = max(error, abs(varied.child(1).returns()[player] - call[i]),
                            abs(varied.child(0).returns()[player] - fold[i]))
            proof = verifier.certificate(state, player)
            if error != 0 or proof["dominates"] != expected:
                raise AssertionError("Independent OpenSpiel payoff verification failed")
            records.append(dict(hand=hand, board=board, player=player, proof=proof, max_error=error))
    # Test every global suit permutation and both private/public dealing orders.
    for hand, board, expected in CASES:
        for perm in permutations(range(4)):
            transform = lambda c: 4*(c//4)+perm[c%4]
            changed = fixture(game, tuple(map(transform, hand[::-1])),
                              tuple(map(transform, board[::-1])), 1)
            if verifier.certificate(changed, 1)["dominates"] != expected:
                raise AssertionError("Suit/order-dependent safeguard")
    return dict(passed=True, independent_payoff_comparisons=2*1081*len(records), records=records,
                suit_permutations=24, diagnostics=verifier.diagnostics())


def validate_native(game, adapter):
    wrapper = ExactCallSafeguard(game, adapter)
    states = [fixture(game, h, b, p) for h, b, _ in CASES for p in (0, 1)]
    baseline_parity = adapter.check_scalar_parity(states)
    deployed_parity = wrapper.check_scalar_parity(states)
    errors = []
    for hand, board, expected in CASES:
        for player in (0, 1):
            state = fixture(game, hand, board, player)
            a = adapter.action_probabilities(state, player)
            b = wrapper.action_probabilities(state, player)
            wanted = {0: 0., 1: a[0]+a[1]} if expected else a
            if b != wanted:
                raise AssertionError("Safeguard changed something other than certified fold mass")
            other = tuple(c for c in range(52) if c not in (*hand, *board))[-2:]
            alternate = fixture(game, hand, board, player, other)
            changed = wrapper.action_probabilities(alternate, player)
            errors.append(max(abs(changed[k]-b[k]) for k in b))
    if max(errors) > 1e-12:
        raise AssertionError("Actual opponent cards leaked into deployed policy")
    return dict(passed=True, baseline_parity=baseline_parity, deployed_parity=deployed_parity,
                hidden_card_max_error=max(errors), verifier=wrapper.verifier.diagnostics())
