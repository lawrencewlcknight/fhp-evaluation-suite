"""Information-set privacy, native parity, card-order and suit-symmetry diagnostics."""

from itertools import permutations
import numpy as np

from ..best_response.flop import make_state


def check_adapter(context, adapter, *, require_suit_invariance=False, tolerance=3e-6):
    game, deck = context.game, context.deck
    board, hand = context.space.board, tuple(context.space.hands[0])
    pre = context.entries[0]
    states = []
    # Several histories, both seats; no actual opponent cards enter model input.
    for node in context.trees[pre]:
        if node.player >= 0:
            states.append(make_state(game, deck, hand, node.player, board=board,
                                     preflop=pre, flop=node.path))
    parity = adapter.check_scalar_parity(states)
    hidden_error = order_error = 0.
    suit_errors = []
    for player in (0, 1):
        node = next(n for n in context.trees[pre] if n.player == player)
        base = make_state(game, deck, hand, player, board=board, preflop=pre, flop=node.path)
        expected = adapter.batch_probabilities([base], player)[0]
        other = tuple(c for c in deck if c not in set(board) | set(hand))[-2:]
        changed = game.new_initial_state()
        holes = (*hand, *other) if player == 0 else (*other, *hand)
        for action in (*holes, *pre, *board, *node.path):
            changed.apply_action(int(action))
        if base.information_state_string(player) != changed.information_state_string(player):
            raise AssertionError("Privacy test failed to preserve the information set")
        hidden_error = max(hidden_error, float(np.max(abs(
            adapter.batch_probabilities([changed], player)[0] - expected))))
        # Reverse both hole-card order and the flop dealing order.
        ordered = make_state(game, deck, hand[::-1], player, board=board[::-1],
                             preflop=pre, flop=node.path)
        order_error = max(order_error, float(np.max(abs(
            adapter.batch_probabilities([ordered], player)[0] - expected))))
        suits = int(game.get_parameters()["numSuits"])
        renamed = []
        for p in permutations(range(suits)):
            transform = lambda c: suits * (int(c) // suits) + p[int(c) % suits]
            renamed.append(make_state(game, deck, tuple(map(transform, hand)), player,
                                      board=tuple(map(transform, board)), preflop=pre, flop=node.path))
        suit_errors.extend(np.max(abs(adapter.batch_probabilities(renamed, player) - expected), axis=1))
    if hidden_error > tolerance or order_error > tolerance:
        raise ValueError("Hidden-card or card-order invariance failure")
    if require_suit_invariance and max(suit_errors) > tolerance:
        raise ValueError("Required suit invariance failed")
    return dict(native_scalar_parity=parity, hidden_card_max_error=hidden_error,
                card_order_max_error=order_error, suit_max_error=float(max(suit_errors)),
                suit_mean_error=float(np.mean(suit_errors)), tolerance=tolerance,
                suit_invariance_required=require_suit_invariance,
                scope="two seats, one hand, several histories on this board; not exhaustive")
