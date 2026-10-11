"""Reuse exact card features across a SINGLE cold-board solve.

The frozen Exp16 encoder's [104:138] slice contains player, round and betting
history. All remaining entries depend only on own cards/public board. Encoding
these repeatedly at every public node dominates a subsecond search. Reusing
them is not an abstraction, cached policy table or cross-hand search cache.
Other encoders use the original adapter path unchanged.
"""
import numpy as np
import torch

from ..best_response.adapters import _distributions
from ..best_response.flop import make_state, validate_game
from .features import card_frames


class RangePolicyQuery:
    def __init__(self, game, adapter, space, hero, deadline, *, implementation="vectorized"):
        if implementation not in ("vectorized", "legacy"):
            raise ValueError("Unknown range encoder implementation")
        self.implementation = implementation
        self.game, self.adapter, self.space = game, adapter, space
        self.deck = validate_game(game)
        self.hero, self.deadline, self.frames = hero, deadline, {}
        encoder = getattr(adapter.native, "feature_encoder", None)
        self.fast = (adapter.backend == "native_logits" and encoder is not None
                     and type(encoder).__name__ == "FHPHandBoardFeatureEncoder"
                     and encoder.encoder_id == "fhp_lossless_hand_board_v2"
                     and encoder.policy_size == 213 and encoder.policy_layout.card_size == 104
                     and int(game.get_parameters()["numRanks"]) == 13
                     and int(game.get_parameters()["numSuits"]) == 4)

    def __call__(self, board, pre, post=()):
        self.deadline.check()
        if not self.fast:
            result = []
            for offset in range(0, len(self.space.hands), self.adapter.batch_size):
                self.deadline.check()
                states = [make_state(self.game, self.deck, h, self.hero,
                                     board=board, preflop=pre, flop=post)
                          for h in self.space.hands[offset:offset + self.adapter.batch_size]]
                result.append(self.adapter.batch_probabilities(states, self.hero))
            self.deadline.check()
            return np.concatenate(result)
        encoder = self.adapter.native.feature_encoder
        phase = bool(board)
        representative = make_state(self.game, self.deck, self.space.hands[0], self.hero,
                                    board=board, preflop=pre, flop=post)
        if phase not in self.frames:
            with self.deadline.measure("card_features"):
                if self.implementation == "vectorized":
                    self.frames[phase] = card_frames(self.space.hands, board)
                else:
                    rows = []
                    for h in self.space.hands:
                        self.deadline.check()
                        state = make_state(self.game, self.deck, h, self.hero,
                                           board=board, preflop=pre, flop=post)
                        rows.append(encoder.information_state(state, self.hero))
                    self.frames[phase] = np.asarray(rows, dtype=np.float32)
            self.deadline.check()
        inputs = self.frames[phase].copy()
        inputs[:, 104:138] = encoder.information_state(representative, self.hero)[104:138]
        mask = np.asarray(representative.legal_actions_mask(self.hero), dtype=bool)
        outputs = []
        for offset in range(0, len(inputs), self.adapter.batch_size):
            self.deadline.check()
            with self.deadline.measure("policy_inference"), torch.inference_mode():
                logits = self.adapter.native.model(torch.from_numpy(inputs[offset:offset + self.adapter.batch_size]))
                logits = logits.masked_fill(~torch.from_numpy(mask), -torch.inf)
                values = torch.softmax(logits, dim=-1).cpu().numpy()
            outputs.append(_distributions(values, np.broadcast_to(mask, values.shape)))
        self.deadline.check()
        return np.concatenate(outputs)
