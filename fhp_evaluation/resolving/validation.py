"""Mandatory correctness checks; no tuning using gameplay outcomes."""
import numpy as np
import pyspiel

from ..best_response.adapters import PolicyAdapter
from ..best_response.flop import FlopBestResponse, make_state, validate_game
from ..best_response.validation import FixedPolicy
from ..duplicate import play_hand as original_play
from ..game import FHP_GAME_PARAMETERS
from .policy import ResolvingPolicy, play_hand
from .solver import Deadline, prepare, response_values, certify, solve


def algebra_checks():
    game = pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS, numRanks=4, numSuits=2))
    adapter = PolicyAdapter(FixedPolicy(game, "random"))
    records = []
    for hero in (0, 1):
        for pre in ((1, 1), (2, 1), (2, 2, 2, 1)):
            p, candidate, stats = solve(game, adapter, (0, 3, 6), pre, hero, seconds=None, iterations=100)
            bound = response_values(p, p.blueprint, Deadline())
            reference = FlopBestResponse(game, adapter).solve((0, 3, 6), pre, 1 - hero,
                                                             opponent_reach=p.reach)
            error = float(np.max(np.abs(bound - reference.weighted_values / 100)))
            if error > 1e-10 or candidate is None or not certify(p, candidate, bound, Deadline())["passed"]:
                raise AssertionError("Exact response differential/candidate certificate failed")
            records.append(dict(hero=hero, pre=pre, error_bb=error, accepted=stats["accepted"]))
    opponent = PolicyAdapter(FixedPolicy(game, "uniform"))
    for seat in (0, 1):
        wrapper = ResolvingPolicy(game, adapter, seconds=0)
        for seed in range(20):
            original = (adapter, opponent) if seat == 0 else (opponent, adapter)
            revised = (wrapper, opponent) if seat == 0 else (opponent, wrapper)
            expected = original_play(game, original, chance_seed=seed, action_seed=seed + 100)
            actual, _ = play_hand(game, revised, chance_seed=seed, action_seed=seed + 100)
            if actual != expected:
                raise AssertionError("Zero-budget simulator parity failed")
    return dict(passed=True, records=records, zero_budget_hands=40)


def native_checks(game, adapter):
    """Compare feature reuse against untouched native inference at EVERY node.

    Also compare own-information-identical states with different hidden cards.
    No value derived from hidden opponent cards is an input to the online solve.
    """
    deck, board, errors, hidden_errors = validate_game(game), (16, 20, 32), [], []
    for hero in (0, 1):
        for pre in ((1, 1), (2, 2, 2, 1)):
            p = prepare(game, adapter, board, pre, hero, Deadline())
            indices = (0, len(p.space.hands) // 2, len(p.space.hands) - 1)
            for i, rows in p.blueprint.items():
                n = p.nodes[i]
                states = [make_state(game, deck, p.space.hands[j], hero, board=board,
                                     preflop=pre, flop=n.path) for j in indices]
                untouched = adapter.batch_probabilities(states, hero)
                errors.append(float(np.max(np.abs(rows[list(indices)] - untouched[:, [a for a, _ in n.children]]))))
                adapter.check_scalar_parity(states)
            own = p.space.hands[indices[1]]
            remaining = [c for c in deck if c not in (*own, *board)]
            states = []
            for other in (remaining[:2], remaining[-2:]):
                state = game.new_initial_state()
                cards = (*own, *other) if hero == 0 else (*other, *own)
                for a in (*cards, *pre, *board):
                    state.apply_action(int(a))
                # Root actor may not be hero; a check is a legal route to hero.
                if state.current_player() != hero:
                    state.apply_action(1)
                states.append(state)
            if states[0].information_state_string(hero) != states[1].information_state_string(hero):
                raise AssertionError("Invalid hidden-card invariance fixture")
            probs = adapter.batch_probabilities(states, hero)
            hidden_errors.append(float(np.max(np.abs(probs[0] - probs[1]))))
    if max(errors) > 3e-6 or max(hidden_errors) > 1e-12:
        raise AssertionError("Native feature parity or hidden-card invariance failed")
    return dict(passed=True, maximum_reuse_probability_error=max(errors),
                maximum_hidden_card_probability_error=max(hidden_errors))
