"""exp1_br_validation: no trained model, paid job or full-deck traversal required."""

import hashlib
import importlib.metadata
import time

import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response, cfr

from ..game import FHP_GAME_PARAMETERS
from .flop import FlopBestResponse
from .reference import exact_small_game_response, scalar_flop_response


class FixedPolicy(policy.Policy):
    """Stable information-set-only controls, including nonuniform hidden-hand ranges."""
    def __init__(self, game, kind="random"):
        super().__init__(game, list(range(game.num_players())))
        self.kind = kind

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        legal = state.legal_actions(player)
        if self.kind in ("fold", "call", "raise"):
            preferred = {"fold": 0, "call": 1, "raise": 2}[self.kind]
            return {preferred if preferred in legal else min(legal): 1.}
        if self.kind == "uniform":
            values = np.ones(len(legal))
        else:
            key = state.information_state_string(player).encode()
            digest = hashlib.sha256(key).digest()
            values = np.asarray([1 + digest[a] for a in legal], dtype=float)
        values /= values.sum()
        return dict(zip(legal, map(float, values)))


def run_validation():
    started = time.perf_counter()
    records = []
    for name in ("kuhn_poker", "leduc_poker"):
        game = pyspiel.load_game(name)
        solver = cfr.CFRPlusSolver(game)
        for _ in range(20):
            solver.evaluate_and_update_policy()
        controls = [(kind, FixedPolicy(game, kind)) for kind in ("uniform", "random")]
        controls.append(("cfr_plus_20", solver.average_policy()))
        for label, target in controls:
            for player in (0, 1):
                actual = exact_small_game_response(game, target, player)
                expected = best_response.BestResponsePolicy(game, player, target).value(game.new_initial_state())
                error = abs(actual["value"] - expected)
                if error > 1e-9:
                    raise AssertionError(f"OpenSpiel BR differential failure: {name}, {label}, {error}")
                records.append(dict(game=name, policy=label, player=player, error=error, **actual))
    game = pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))
    for kind in ("uniform", "random", "fold", "call", "raise"):
        target = FixedPolicy(game, kind)
        engine = FlopBestResponse(game, target)
        for preflop in ((1, 1), (2, 1), (2, 2, 2, 1)):
            for player in (0, 1):
                result = engine.solve((0, 3, 6), preflop, player)
                errors = []
                for index, hand in enumerate(result.hands):
                    reference = scalar_flop_response(game, target, result.board, preflop, player, hand)
                    errors.append(abs(reference["weighted_value"] - result.weighted_values[index]))
                    np.testing.assert_allclose(reference["opponent_mass"], result.opponent_mass[index], atol=1e-10)
                if max(errors) > 1e-8:
                    raise AssertionError(f"Reduced FHP differential failure: {kind}, {max(errors)}")
                records.append(dict(game="reduced_fhp", policy=kind, preflop=list(preflop),
                                    player=player, hands=len(result.hands), error=max(errors)))
    return dict(experiment="exp1_br_validation", status="succeeded", exact_flop_validated=True,
                full_game_fhp_exploitability_implemented=False, records=records,
                max_absolute_error=max(row["error"] for row in records),
                open_spiel=importlib.metadata.version("open_spiel"),
                elapsed_seconds=time.perf_counter() - started)
