"""Coupled duplicate games with exact terminal integration for rare-event precision.

Both policies have identical paths until an altered terminal decision. We can
therefore score BOTH actual sampled games on that shared path, and additionally
integrate the final action's randomness analytically. No importance weighting,
selected rare-position sampling, opponent-card access by the policy, or replay
of an opponent's decisions under a changed nonterminal history is involved.
"""
import numpy as np

from .policy import TerminalCallVerifier
from .experiment import stream_seed
from ..game import MILLI_BIG_BLINDS_PER_CHIP


# Each row is an independent duplicate pair, not an individual hand.
COLUMNS = ("baseline_mbb", "guarded_mbb", "sampled_delta_mbb", "integrated_delta_mbb",
           "activated_hands", "eligible_hands", "fold_mass", "hero_decisions")


def paired_matches(game, base, opponent, task, spec, *, verifier=None):
    verifier = verifier or TerminalCallVerifier(game)
    n = task["count"]
    output = np.zeros((n, len(COLUMNS)), dtype=np.float64)
    # Bound the number of native game objects and neural input rows.
    for offset in range(0, n, spec["batch_size"]):
        width = min(spec["batch_size"], n-offset)
        games = []
        for local in range(width):
            pair = task["start"] + offset + local
            master = np.random.default_rng(stream_seed(spec, task["seed"], task["opponent"], pair))
            chance, action = map(int, master.integers(0, 2**63-1, size=2))
            for seat in (0, 1):
                games.append(dict(state=game.new_initial_state(), seat=seat, row=offset+local,
                                  chance=np.random.default_rng(chance), action=np.random.default_rng(action)))
        while games:
            active = []
            for item in games:
                state = item["state"]
                while state.is_chance_node():
                    actions, probabilities = zip(*state.chance_outcomes())
                    state.apply_action(int(item["chance"].choice(actions, p=probabilities)))
                if state.is_terminal():
                    value = .5 * state.returns()[item["seat"]] * MILLI_BIG_BLINDS_PER_CHIP
                    output[item["row"], :2] += value
                else:
                    active.append(item)
            games = []
            # Freeze membership BEFORE advancing any mutable state: advancing a
            # player-0 game can otherwise accidentally put it in player 1's batch
            # in this same pass (or turn it terminal underneath a later group).
            groups = {(player, hero_turn): [g for g in active if g["state"].current_player() == player
                       and (g["seat"] == player) == hero_turn]
                      for player in (0, 1) for hero_turn in (False, True)}
            for player in (0, 1):
                for hero_turn in (False, True):
                    group = groups[player, hero_turn]
                    if not group:
                        continue
                    adapter = base if hero_turn else opponent
                    rows = adapter.batch_probabilities([g["state"] for g in group], player)
                    for item, probabilities in zip(group, rows):
                        state, index = item["state"], item["row"]
                        uniform = item["action"].random()
                        chosen = min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), 2)
                        if chosen not in state.legal_actions():
                            raise AssertionError("Invalid policy probabilities in paired match")
                        proof = None
                        if hero_turn:
                            output[index, 7] += 1
                            proof = verifier.certificate(state, player) if probabilities[0] > 0 else None
                            output[index, 5] += proof is not None
                        if hero_turn and proof is not None and proof["dominates"] and probabilities[0] > 0:
                            # Access to the actual deal is confined to evaluator scoring,
                            # AFTER the observable-information certificate is complete.
                            call, fold = state.child(1).returns()[player], state.child(0).returns()[player]
                            if call < fold:
                                raise AssertionError("Verifier disagrees with OpenSpiel on a reached deal")
                            baseline = fold if chosen == 0 else call
                            scale = .5 * MILLI_BIG_BLINDS_PER_CHIP
                            output[index, 0] += scale * baseline
                            output[index, 1] += scale * call
                            output[index, 3] += scale * probabilities[0] * (call-fold)
                            output[index, 4] += 1
                            output[index, 6] += probabilities[0]
                        else:
                            state.apply_action(chosen)
                            games.append(item)
    output[:, 2] = output[:, 1] - output[:, 0]
    if np.any(output[:, 2:4] < -1e-10):
        raise AssertionError("A terminal-only coupled substitution cannot lose against a fixed opponent")
    return output, verifier.diagnostics()
