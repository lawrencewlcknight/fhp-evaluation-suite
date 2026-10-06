"""Portable, exact behavioural tables avoid native-package collisions and repeat inference."""

import numpy as np

from ..best_response.flop import HandSpace, compile_tree, make_state, validate_game


def key(pre, post=None):
    return "pre_" + "".join(map(str, pre)) if post is None else (
        "flop_" + "".join(map(str, pre)) + "_" + "".join(map(str, post)))


def preflop_nodes(game):
    state = game.new_initial_state()
    for c in range(4):
        state.apply_action(c)
    nodes, entries = {}, []
    def walk(s, path):
        if s.is_terminal():
            return
        if s.is_chance_node():
            entries.append(path)
            return
        nodes[path] = (s.current_player(), tuple(s.legal_actions()))
        for action in s.legal_actions():
            walk(s.child(action), path + (action,))
    walk(state, ())
    return nodes, tuple(entries)


class BoardContext:
    def __init__(self, game, board):
        self.game, self.deck = game, validate_game(game)
        self.space = HandSpace(game, self.deck, board)
        self.pre_nodes, self.entries = preflop_nodes(game)
        self.trees = {pre: compile_tree(make_state(game, self.deck, self.space.hands[0], 0,
                                                 board=board, preflop=pre)) for pre in self.entries}

    def schemas(self):
        for pre, (player, legal) in self.pre_nodes.items():
            yield key(pre), pre, None, player, legal
        for pre, nodes in self.trees.items():
            for node in nodes:
                if node.player >= 0:
                    yield key(pre, node.path), pre, node.path, node.player, tuple(a for a, _ in node.children)


def build_tables(context, adapter):
    tables = {}
    for name, pre, post, player, _ in context.schemas():
        values = np.empty((len(context.space.hands), 3), dtype=np.float64)
        for start in range(0, len(values), adapter.batch_size):
            states = [make_state(context.game, context.deck, h, player,
                                 board=() if post is None else context.space.board,
                                 preflop=pre, flop=() if post is None else post)
                      for h in context.space.hands[start:start + adapter.batch_size]]
            values[start:start + len(states)] = adapter.batch_probabilities(states, player)
        tables[name] = values
    validate_tables(context, tables)
    return tables


def accelerate_rule(context, adapter):
    """Cache exact uniform flop equity once per board, keeping native rule logic.

    The generic published agents enumerate hidden hands on EVERY policy call.
    Repeating that at every public history would dominate this audit. This is an
    audit-local evaluator optimisation, not a change to the shared rule agents.
    """
    import copy
    from ..best_response.adapters import PolicyAdapter
    from ..equity import published_hand_strength
    from .metrics import win_tie_loss
    space = context.space
    weights = np.ones(len(space.hands))
    wins, ties, _ = win_tie_loss(space, weights)
    strength = (wins + .5 * ties) / space.mass(weights) * 1326. - 663.
    lookup = {tuple(map(int, h)): float(s) for h, s in zip(space.hands, strength)}
    board = tuple(space.board)
    def cached(private, public):
        if tuple(sorted(public)) == board:
            return lookup[tuple(sorted(private))]
        return published_hand_strength(private, public)
    native = copy.copy(adapter.native)
    native._strength_fn = cached
    return PolicyAdapter(native, batch_size=adapter.batch_size, reference=adapter.native,
                         metadata=dict(adapter.metadata, exact_rule_equity_cache="per_board_vector"))


def validate_tables(context, tables):
    expected = set()
    for name, _, _, _, legal in context.schemas():
        expected.add(name)
        values = tables[name]
        illegal = [a for a in range(3) if a not in legal]
        if (values.shape != (len(context.space.hands), 3) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values[:, illegal] != 0)
                or not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=1e-12)):
            raise ValueError(f"Invalid cached probabilities: {name}")
    if set(tables) != expected:
        raise ValueError("Unexpected cached table keys")


def reach_preflop(context, tables, pre, player):
    weights = np.ones(len(context.space.hands))
    for i, action in enumerate(pre):
        if context.pre_nodes[pre[:i]][0] == player:
            weights *= tables[key(pre[:i])][:, action]
    return weights


def load_tables(path, context):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    validate_tables(context, result)
    return result
