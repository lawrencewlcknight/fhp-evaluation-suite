"""Exact, range-vector best responses AFTER the final FHP chance event.

Opponent reach excludes the responder's choices. At a responder node we take
one maximum per own hand AFTER summing hidden opponent hands, never one maximum
per fully observed deal. There are no chance nodes below a flop root.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np
import pyspiel

from ..cards import cards_from_information_state, five_card_rank
from ..game import FHP_GAME_PARAMETERS
from ..lbr import _spent
from .adapters import PolicyAdapter


def validate_game(game):
    parameters = game.get_parameters()
    if game.get_type().short_name != "universal_poker":
        raise ValueError("Flop engine requires universal_poker")
    ranks, suits = int(parameters["numRanks"]), int(parameters["numSuits"])
    if not (1 <= ranks <= 13 and 1 <= suits <= 4 and ranks * suits >= 8):
        raise ValueError("Expected a deck of 8..52 standard cards")
    expected = pyspiel.load_game("universal_poker", dict(
        FHP_GAME_PARAMETERS, numRanks=ranks, numSuits=suits)).get_parameters()
    if parameters != expected:
        raise ValueError("Unsupported game contract: only canonical FHP or its reduced deck is supported")
    return tuple(game.new_initial_state().legal_actions())


def observed_cards(game, state, player):
    private, board = cards_from_information_state(state, player)
    suits = int(game.get_parameters()["numSuits"])
    def internal(cards):
        return tuple(sorted((c // 4) * suits + c % 4 for c in cards))
    return internal(private), internal(board)


def betting_history(game, state):
    cursor, dealt, preflop, flop = game.new_initial_state(), 0, [], []
    for action in state.history():
        if cursor.is_chance_node():
            dealt += 1
        else:
            (preflop if dealt == 4 else flop).append(int(action))
        cursor.apply_action(action)
    return tuple(preflop), tuple(flop)


def make_state(game, deck, hand, player, *, board=(), preflop=(), flop=()):
    """Representative state with the queried player's hand, never actual enemy cards."""
    if (player not in (0, 1) or len(hand) != 2 or len(set(hand)) != 2
            or len(set(board)) != len(board) or len(board) not in (0, 3)
            or not (set(hand) | set(board)) <= set(deck)):
        raise ValueError("Invalid player or cards")
    excluded = set(hand) | set(board)
    other = tuple(c for c in deck if c not in excluded)[:2]
    if len(other) != 2 or len(set(hand)) != 2 or set(hand) & set(board):
        raise ValueError("Invalid or overlapping cards")
    holes = tuple(hand) + other if player == 0 else other + tuple(hand)
    state = game.new_initial_state()
    for action in holes:
        state.apply_action(int(action))
    for action in preflop:
        if state.is_terminal() or state.is_chance_node() or action not in state.legal_actions():
            raise ValueError("Invalid preflop betting prefix")
        state.apply_action(int(action))
    for card in board:
        if not state.is_chance_node() or card not in state.legal_actions():
            raise ValueError("Preflop prefix must reach the flop chance event")
        state.apply_action(int(card))
    for action in flop:
        if state.is_terminal() or state.is_chance_node() or action not in state.legal_actions():
            raise ValueError("Invalid flop betting prefix")
        state.apply_action(int(action))
    return state


@dataclass(frozen=True)
class Node:
    path: tuple[int, ...]
    player: int
    children: tuple[tuple[int, int], ...]
    spent: tuple[float, float]
    fold_returns: tuple[float, float] | None = None


def compile_tree(root):
    """Compile only the public betting skeleton, not a full private-card tree."""
    nodes = []
    def walk(state, path):
        index = len(nodes)
        nodes.append(None)
        if state.is_chance_node():
            raise ValueError("Exact flop continuation must have no future chance")
        if state.is_terminal():
            nodes[index] = Node(path, -1, (), _spent(state),
                                tuple(state.returns()) if path and path[-1] == 0 else None)
        else:
            children = tuple((a, walk(state.child(a), path + (a,))) for a in state.legal_actions())
            nodes[index] = Node(path, state.current_player(), children, _spent(state))
        return index
    walk(root, ())
    return tuple(nodes)


class HandSpace:
    """Exact blocker-aware O(K + 52 R) terminal operations, no K-by-K payoff matrix."""

    def __init__(self, game, deck, board):
        board = tuple(sorted(map(int, board)))
        if len(board) != 3 or len(set(board)) != 3 or not set(board) <= set(deck):
            raise ValueError("Three distinct legal flop cards required")
        self.board = board
        self.hands = np.asarray(list(combinations([c for c in deck if c not in board], 2)),
                                dtype=np.int32)
        self.lookup = {tuple(hand): i for i, hand in enumerate(self.hands)}
        self.deck_size = len(deck)
        suits = int(game.get_parameters()["numSuits"])
        ranks = [five_card_rank([(int(c) // suits) * 4 + int(c) % suits
                                 for c in (*hand, *board)]) for hand in self.hands]
        ordered = {rank: i for i, rank in enumerate(sorted(set(ranks)))}
        self.codes = np.asarray([ordered[rank] for rank in ranks], dtype=np.int32)
        self.rank_count = len(ordered)
        self.card_rank_indices = (self.hands * self.rank_count + self.codes[:, None]).ravel()

    def mass(self, weights):
        totals = np.bincount(self.hands.ravel(), weights=np.repeat(weights, 2),
                             minlength=self.deck_size)
        return weights.sum() - totals[self.hands].sum(axis=1) + weights

    def showdown(self, weights, own_spent, other_spent):
        counts = np.bincount(self.codes, weights=weights, minlength=self.rank_count)
        cumulative = np.cumsum(counts)
        wins = (cumulative - counts)[self.codes]
        losses = (weights.sum() - cumulative)[self.codes]
        by_card = np.bincount(self.card_rank_indices, weights=np.repeat(weights, 2),
                              minlength=self.deck_size * self.rank_count)
        by_card = by_card.reshape(self.deck_size, self.rank_count)
        prefix = np.cumsum(by_card, axis=1)
        lesser, greater = prefix - by_card, prefix[:, -1:] - prefix
        # A hand overlapping both own cards is the identical hand and hence a tie;
        # its contribution to win/loss sums is zero, so no extra correction there.
        wins -= lesser[self.hands, self.codes[:, None]].sum(axis=1)
        losses -= greater[self.hands, self.codes[:, None]].sum(axis=1)
        ties = self.mass(weights) - wins - losses
        return other_spent * wins - own_spent * losses + .5 * (other_spent - own_spent) * ties


@dataclass
class FlopResult:
    board: tuple[int, ...]
    preflop: tuple[int, ...]
    responder: int
    hands: np.ndarray
    weighted_values: np.ndarray
    opponent_mass: np.ndarray
    actions: dict[tuple[int, ...], np.ndarray]
    elapsed_seconds: float

    def action(self, private, history):
        matches = np.flatnonzero(np.all(self.hands == np.asarray(sorted(private)), axis=1))
        if len(matches) != 1 or tuple(history) not in self.actions:
            raise ValueError("Hand or decision not present in this flop response")
        return int(self.actions[tuple(history)][matches[0]])


class FlopBestResponse:
    def __init__(self, game, target):
        self.game, self.deck = game, validate_game(game)
        self.target = target if isinstance(target, PolicyAdapter) else PolicyAdapter(target)

    def preflop_reach(self, hands, board, preflop, opponent):
        states = [make_state(self.game, self.deck, h, opponent, board=(), preflop=())
                  for h in hands]
        weights = np.ones(len(hands), dtype=np.float64)
        for action in preflop:
            if states[0].is_chance_node() or states[0].is_terminal():
                raise ValueError("Invalid preflop sequence")
            if action not in states[0].legal_actions():
                raise ValueError("Illegal preflop action")
            if states[0].current_player() == opponent:
                weights *= self.target.batch_probabilities(states, opponent)[:, action]
            for state in states:
                if action not in state.legal_actions():
                    raise ValueError("Illegal preflop action")
                state.apply_action(int(action))
        if not states[0].is_chance_node():
            raise ValueError("Preflop sequence must finish the first betting round")
        return weights

    def solve(self, board, preflop, responder, *, opponent_reach=None):
        started = time.perf_counter()
        if responder not in (0, 1):
            raise ValueError("Responder must be 0 or 1")
        space = HandSpace(self.game, self.deck, board)
        preflop = tuple(map(int, preflop))
        opponent = 1 - responder
        weights = (self.preflop_reach(space.hands, space.board, preflop, opponent)
                   if opponent_reach is None else np.asarray(opponent_reach, dtype=np.float64))
        if weights.shape != (len(space.hands),) or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("Invalid opponent hand reach")
        representative = make_state(self.game, self.deck, space.hands[0], opponent,
                                    board=space.board, preflop=preflop)
        nodes = compile_tree(representative)
        decisions = {}

        def walk(index, reach):
            node = nodes[index]
            if node.player == -1:
                if node.fold_returns is not None:
                    return node.fold_returns[responder] * space.mass(reach)
                return space.showdown(reach, node.spent[responder], node.spent[opponent])
            if node.player == opponent:
                # Stream states in bounded batches rather than retaining K game
                # objects at every public node. Only the small probability array survives.
                probabilities = np.empty((len(space.hands), 3), dtype=np.float64)
                if not np.any(reach):
                    probabilities.fill(0.)
                    probabilities[:, [a for a, _ in node.children]] = 1. / len(node.children)
                for start in range(0, len(space.hands) if np.any(reach) else 0, self.target.batch_size):
                    hands = space.hands[start:start + self.target.batch_size]
                    states = [make_state(self.game, self.deck, h, opponent, board=space.board,
                                         preflop=preflop, flop=node.path) for h in hands]
                    probabilities[start:start + len(hands)] = self.target.batch_probabilities(states, opponent)
                return sum((walk(child, reach * probabilities[:, action])
                            for action, child in node.children), np.zeros(len(reach)))
            values = np.stack([walk(child, reach) for _, child in node.children])
            choice = np.argmax(values, axis=0)
            decisions[node.path] = np.asarray([a for a, _ in node.children], dtype=np.int8)[choice]
            return np.max(values, axis=0)

        result = walk(0, weights)
        return FlopResult(space.board, preflop, responder, space.hands, result,
                          space.mass(weights), decisions, time.perf_counter() - started)


class FullFlopResponsePolicy:
    """Fixed preflop LBR + exact remaining flop response. Legal lower-bound probe."""

    def __init__(self, game, target, *, config=None, cache_boards=2):
        from ..lbr import LBRConfig
        if int(cache_boards) < 1:
            raise ValueError("Positive board cache size required")
        self.engine = FlopBestResponse(game, target)
        self.target = self.engine.target
        self.game = game
        self.preflop = self.target.preflop_policy(game, config or LBRConfig())
        self.cache, self.cache_boards = OrderedDict(), int(cache_boards)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        if state.is_terminal() or state.is_chance_node() or state.current_player() != player:
            raise ValueError("Response queried outside acting information set")
        private, board = observed_cards(self.game, state, player)
        if not board:
            # Existing LBR is unchanged; bound its deterministic memoisation too.
            if len(self.preflop._action_cache) >= 2048:
                self.preflop._action_cache.clear()
            return self.preflop.action_probabilities(state, player)
        pre, post = betting_history(self.game, state)
        key = (board, pre, player)
        if key not in self.cache:
            self.cache[key] = self.engine.solve(board, pre, player)
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_boards:
            self.cache.popitem(last=False)
        result = self.cache[key]
        return {result.action(private, post): 1.}
