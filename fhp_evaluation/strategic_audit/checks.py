"""Information-set privacy, native parity, card-order and suit-symmetry diagnostics."""

from itertools import permutations
import numpy as np

from ..best_response.flop import make_state


def checked_adapter(context, adapter, *, require_suit_invariance=False, float64_fallback=False):
    """Escalate precision only on a scalar/batched mismatch, never waive a gate.

    Successful FP32 boards retain their original path. A fallback board must
    repeat ALL checks, then use the returned FP64 adapter for all table exports.
    If FP64 fails, propagate the error; no unchecked probabilities are accepted.
    """
    try:
        checks = check_adapter(context, adapter, require_suit_invariance=require_suit_invariance)
        return adapter, checks
    except ValueError as error:
        if (not float64_fallback or adapter.backend != "sd_cfr"
                or not str(error).startswith("Native/batched policy mismatch:")):
            raise
        original_error = str(error)
    precise = adapter.sd_float64_copy()
    checks = check_adapter(context, precise, require_suit_invariance=require_suit_invariance)
    checks["numerical_fallback"] = dict(original_error=original_error, from_precision="float32",
        to_precision="float64", tolerance_unchanged=True,
        scope="complete board export, both scalar and batched implementations; identical archived FP32 weights")
    return precise, checks


def check_exported_parity(context, adapter, tables, tolerance=3e-6):
    """Check actual exported rows, not just a separately shaped probe batch.

    These scalar queries are already cached by check_adapter, so this adds no
    extra archive sweeps. Covers both seats and every history of the first
    preflop entry for the first compatible hand; it is not exhaustive.
    """
    from .tables import key
    pre, hand = context.entries[0], tuple(context.space.hands[0])
    maximum = 0.
    for node in context.trees[pre]:
        if node.player < 0:
            continue
        state = make_state(context.game, context.deck, hand, node.player,
                           board=context.space.board, preflop=pre, flop=node.path)
        raw = adapter.reference.action_probabilities(state, node.player)
        expected = np.array([raw.get(a, 0.) for a in range(3)], dtype=np.float64)
        expected /= expected.sum()
        actual = tables[key(pre, node.path)][0]
        error = float(np.max(np.abs(expected - actual)))
        maximum = max(maximum, error)
        if error > tolerance:
            raise ValueError(f"Export/reference policy mismatch: {error} > {tolerance}; "
                             f"history={list(state.history())}; table={actual.tolist()}; scalar={expected.tolist()}")
    return dict(passed=True, max_absolute_error=maximum, tolerance=tolerance,
                scope="actual table rows for the existing first-hand, first-entry probe histories; not exhaustive")


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
    parity = adapter.check_scalar_parity(states, tolerance=tolerance)
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
