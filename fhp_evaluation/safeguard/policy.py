"""Certify terminal call >= fold for EVERY compatible opponent holding.

Only the canonical FHP contract is supported. Rank enumeration is shared across
hands on a board; no actual opponent cards, range estimates or neural values are
used by the certificate. There is deliberately no numerical safety tolerance.
"""
from collections import OrderedDict
from itertools import combinations
import time

import numpy as np

from ..best_response.adapters import PolicyAdapter
from ..best_response.flop import validate_game
from ..cards import cards_from_information_state, five_card_rank
from ..game import load_fhp_game
from ..lbr import _spent


def rank_code(cards):
    rank = five_card_rank(cards)
    # Fixed-length base-15 encoding preserves lexicographic poker rank ordering.
    return sum(v * 15 ** (5 - i) for i, v in enumerate(rank))


class TerminalCallVerifier:
    def __init__(self, game, *, board_cache=4096, certificate_cache=32768):
        if game.get_parameters() != load_fhp_game().get_parameters():
            raise ValueError("Terminal-call safeguard supports canonical FHP only")
        self.deck = validate_game(game)
        if board_cache < 1 or certificate_cache < 1:
            raise ValueError("Positive bounded cache sizes required")
        self.board_limit, self.certificate_limit = board_cache, certificate_cache
        self.boards, self.certificates = OrderedDict(), OrderedDict()
        self.stats = dict(queries=0, eligible=0, certified=0, board_misses=0,
                          certificate_hits=0, seconds=0., maximum_seconds=0.)
        self.edges = np.geomspace(1e-7, 10., 65)
        self.latency = {k: np.zeros(66, dtype=np.int64)
                        for k in ("all", "cold_board", "cached_board", "ineligible")}

    @staticmethod
    def retain(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def board_ranks(self, board):
        board = tuple(sorted(board))
        if board in self.boards:
            self.boards.move_to_end(board)
            return self.boards[board]
        hands = np.asarray(list(combinations([c for c in self.deck if c not in board], 2)), dtype=np.int16)
        codes = np.asarray([rank_code((*h, *board)) for h in hands], dtype=np.int32)
        self.stats["board_misses"] += 1
        return self.retain(self.boards, board, (hands, codes), self.board_limit)

    def payoffs(self, state, player):
        """Return all 1,081 (call, fold) payoffs, or None when ineligible.

        OpenSpiel children are used only for terminal status/public contributions.
        In particular the actual deal's showdown return is NEVER read here.
        """
        if state.is_terminal() or state.is_chance_node() or player != state.current_player():
            return None
        if set(state.legal_actions(player)) != {0, 1}:
            return None
        hand, board = cards_from_information_state(state, player)
        if len(board) != 3:
            return None
        called, folded = state.child(1), state.child(0)
        if not called.is_terminal() or not folded.is_terminal():
            return None
        before, after = _spent(state), _spent(called)
        if after[player] <= before[player]:
            return None
        hands, codes = self.board_ranks(board)
        compatible = ~np.isin(hands, hand).any(axis=1)
        if int(compatible.sum()) != 1081:
            raise AssertionError("Invalid compatible-hand support")
        ranks = codes[compatible]
        own = rank_code((*hand, *board))
        call = np.where(own > ranks, after[1-player],
                        np.where(own < ranks, -after[player], .5 * (after[1-player] - after[player])))
        fold = np.full(len(call), -before[player], dtype=np.float64)
        return hands[compatible], call, fold

    def certificate(self, state, player):
        started = time.perf_counter()
        misses = self.stats["board_misses"]
        self.stats["queries"] += 1
        # The information state contains only observable information.
        key = (player, state.information_state_string(player))
        if key in self.certificates:
            self.stats["certificate_hits"] += 1
            self.certificates.move_to_end(key)
            result = self.certificates[key]
        else:
            values = self.payoffs(state, player)
            result = None if values is None else dict(
                holdings=len(values[0]), minimum_gain_chips=float(np.min(values[1] - values[2])),
                maximum_gain_chips=float(np.max(values[1] - values[2])))
            if result is not None:
                result["dominates"] = result["minimum_gain_chips"] >= 0.
            self.retain(self.certificates, key, result, self.certificate_limit)
        self.stats["eligible"] += result is not None
        self.stats["certified"] += result is not None and result["dominates"]
        elapsed = time.perf_counter() - started
        self.stats["seconds"] += elapsed
        self.stats["maximum_seconds"] = max(self.stats["maximum_seconds"], elapsed)
        index = int(np.searchsorted(self.edges, elapsed))
        self.latency["all"][index] += 1
        kind = ("ineligible" if result is None else
                "cold_board" if self.stats["board_misses"] > misses else "cached_board")
        self.latency[kind][index] += 1
        return result

    def diagnostics(self):
        return dict(**self.stats, board_cache_entries=len(self.boards),
                    certificate_cache_entries=len(self.certificates),
                    board_array_bytes=sum(h.nbytes + c.nbytes for h, c in self.boards.values()),
                    latency_bin_upper_seconds=[*map(float, self.edges), None],
                    latency_counts={k: v.tolist() for k, v in self.latency.items()},
                    scope="All policy queries, including hypothetical responder queries; not gameplay activation rate")


class ExactCallSafeguard(PolicyAdapter):
    """An adapter so all hypothetical LBR/strategic queries see the same safeguard."""
    def __init__(self, game, base, *, verifier=None):
        if not isinstance(base, PolicyAdapter):
            base = PolicyAdapter(base)
        super().__init__(base, batch_size=base.batch_size, metadata=dict(
            base.metadata, deployment="exact_terminal_call_v1"))
        self.base = base
        self.verifier = verifier or TerminalCallVerifier(game)
        self.reference = self  # inherited scalar/batch parity checks the deployed policy

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        raw = self.base.action_probabilities(state, player)
        if raw.get(0, 0.) > 0:
            proof = self.verifier.certificate(state, player)
            if proof is not None and proof["dominates"]:
                raw = dict(raw)
                raw[1] += raw[0]
                raw[0] = 0.
        return raw

    def batch_probabilities(self, states, player):
        states = list(states)
        rows = self.base.batch_probabilities(states, player).copy()
        for i, state in enumerate(states):
            if rows[i, 0] > 0:
                proof = self.verifier.certificate(state, player)
                if proof is not None and proof["dominates"]:
                    rows[i, 1] += rows[i, 0]
                    rows[i, 0] = 0.
        return rows


def deploy(game, adapter, name):
    if name is None:
        return adapter
    if name != "exact_terminal_call_v1":
        raise ValueError(f"Unknown deployment safeguard: {name}")
    return ExactCallSafeguard(game, adapter)
