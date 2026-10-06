"""Exact local-deviation and remaining-round gaps against a FIXED opponent.

All hidden hands are integrated BEFORE maximising. Local Q uses the candidate's
unchanged continuation; BR Q is never substituted for it. Unsupported ranges
are NaN internally and become null/skipped observations in reports, never uniform.
"""

from dataclasses import dataclass

import numpy as np

from ..game import BIG_BLIND_CHIPS
from .tables import key, reach_preflop


@dataclass
class Decision:
    path: tuple
    player: int
    probabilities: np.ndarray
    q_bb: np.ndarray
    local_gap_bb: np.ndarray
    response_gap_bb: np.ndarray
    opponent_mass: np.ndarray
    equity: np.ndarray
    global_unbeatable: np.ndarray
    certain_loss: np.ndarray
    pot_bb: float
    call_cost_bb: float
    threshold: float | None
    legal: tuple
    raises_remaining: int


def normalise(values, mass):
    out = np.full_like(values, np.nan, dtype=np.float64)
    np.divide(values, mass, out=out, where=mass > 0)
    return out


def audit_entry(context, candidate, opponent, pre, player, *, range_tilt=0.):
    """Yield one decision per public path, covering all hero hands simultaneously.

    Optional exponential tilt is an explicitly synthetic prior stress test, based
    on showdown rank against a uniform range. It is not a fitted opponent model.
    """
    if not np.isfinite(range_tilt):
        raise ValueError("Finite range tilt required")
    space, nodes = context.space, context.trees[tuple(pre)]
    uniform_mass = space.mass(np.ones(len(space.hands)))
    uniform_edge = normalise(space.showdown(np.ones(len(space.hands)), 1, 1), uniform_mass)
    # Structural unbeatable test independent of the specified opponent support.
    from .metrics import win_tie_loss
    _, _, losses = win_tie_loss(space, np.ones(len(space.hands)))
    unbeatable = losses == 0
    weights = reach_preflop(context, opponent, tuple(pre), 1 - player)
    if range_tilt:
        weights *= np.exp(range_tilt * uniform_edge)
    records = []

    def walk(index, reach):
        node = nodes[index]
        support = space.mass((reach > 0).astype(float))
        mass = np.where(support > 0, np.maximum(space.mass(reach), 0.), 0.)
        if np.any((support > 0) & (mass <= 0)):
            raise FloatingPointError("Positive range support lost to numerical cancellation")
        if node.player == -1:
            value = (node.fold_returns[player] * mass if node.fold_returns is not None else
                     space.showdown(reach, node.spent[player], node.spent[1 - player]))
            return value, value
        if node.player != player:
            probs = opponent[key(pre, node.path)]
            children = [walk(child, reach * probs[:, a]) for a, child in node.children]
            return (sum((x[0] for x in children), np.zeros_like(reach)),
                    sum((x[1] for x in children), np.zeros_like(reach)))
        children = [walk(child, reach) for _, child in node.children]
        legal = tuple(a for a, _ in node.children)
        q = np.stack([v[0] for v in children], axis=1)
        br_q = np.stack([v[1] for v in children], axis=1)
        probs = candidate[key(pre, node.path)]
        current = np.sum(probs[:, legal] * q, axis=1)
        optimal = np.max(br_q, axis=1)
        q_full = np.full((len(reach), 3), np.nan)
        q_full[:, legal] = normalise(q, mass[:, None]) / BIG_BLIND_CHIPS
        local = normalise(np.max(q, axis=1) - current, mass) / BIG_BLIND_CHIPS
        remaining = normalise(optimal - current, mass) / BIG_BLIND_CHIPS
        if np.any(local < -1e-9) or np.any(remaining < -1e-9):
            raise AssertionError("Negative decision gap")
        wins, ties, losses = win_tie_loss(space, reach)
        equity = np.clip(normalise(wins + .5 * ties, mass), 0, 1)
        support_wins, support_ties, _ = win_tie_loss(space, (reach > 0).astype(float))
        call_cost = max(node.spent) - node.spent[player]
        call_child = next((nodes[c] for a, c in node.children if a == 1), None)
        final_call = (call_cost > 0 and set(legal) == {0, 1} and
                      call_child is not None and call_child.player == -1 and
                      call_child.fold_returns is None)
        threshold = call_cost / (sum(node.spent) + call_cost) if final_call else None
        records.append(Decision(node.path, player, probs, q_full, np.maximum(local, 0),
                                np.maximum(remaining, 0), mass, equity, unbeatable,
                                (support_wins == 0) & (support_ties == 0) & (mass > 0),
                                sum(node.spent) / BIG_BLIND_CHIPS, call_cost / BIG_BLIND_CHIPS,
                                threshold, legal, 3 - node.path.count(2)))
        return current, optimal

    value, br_value = walk(0, weights)
    mass = np.where(space.mass((weights > 0).astype(float)) > 0, np.maximum(space.mass(weights), 0.), 0.)
    return dict(decisions=sorted(records, key=lambda r: (len(r.path), r.path)),
                value_bb=normalise(value, mass) / BIG_BLIND_CHIPS,
                best_response_bb=normalise(br_value, mass) / BIG_BLIND_CHIPS,
                root_opponent_mass=mass)
