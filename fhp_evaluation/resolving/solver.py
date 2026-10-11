"""Range-vector regret-matching+ on a blueprint opt-out gadget.

Only the public flop/preflop history and the blueprint enter the solve. The
opponent is free to best respond; its actual policy/cards never enter it.
Every published continuation passes a separate exact (floating-point) BR check
for EACH opponent root hand, with the unchanged hero preflop counterfactual
reach. This is a local numerical certificate, not measured whole-game NashConv.
"""
from dataclasses import dataclass
import time

import numpy as np

from ..best_response.flop import HandSpace, compile_tree, make_state, validate_game
from ..game import BIG_BLIND_CHIPS
from ..strategic_audit.tables import preflop_nodes
from .range_policy import RangePolicyQuery


class DeadlineExceeded(Exception):
    pass


class Deadline:
    def __init__(self, seconds=None, *, clock=time.perf_counter):
        self.clock = clock
        self.start = clock()
        self.end = float("inf") if seconds is None else self.start + seconds

    def check(self):
        if self.clock() >= self.end:
            raise DeadlineExceeded


@dataclass
class Problem:
    space: object
    nodes: tuple
    hero: int
    reach: np.ndarray
    blueprint: dict


def prepare(game, adapter, board, pre, hero, deadline):
    deadline.check()
    deck = validate_game(game)
    space = HandSpace(game, deck, board)
    root = make_state(game, deck, space.hands[0], hero, board=board, preflop=pre)
    nodes = compile_tree(root)
    pre_nodes, entries = preflop_nodes(game)
    if pre not in entries or hero not in (0, 1):
        raise ValueError("Invalid public flop entry")

    query = RangePolicyQuery(game, adapter, space, hero, deadline)

    reach = np.ones(len(space.hands))
    for i, a in enumerate(pre):
        if pre_nodes[pre[:i]][0] == hero:
            reach *= query((), pre[:i])[:, a]
    if reach.sum() <= 0:
        raise ValueError("Blueprint has no hero support at this entry")
    reach /= reach.sum()
    blueprint = {i: query(board, pre, node.path)[:, [a for a, _ in node.children]]
                 for i, node in enumerate(nodes) if node.player == hero}
    return Problem(space, nodes, hero, reach, blueprint)


def terminal(problem, node, player, other_reach):
    if node.fold_returns is not None:
        return node.fold_returns[player] / BIG_BLIND_CHIPS * problem.space.mass(other_reach)
    return problem.space.showdown(other_reach, node.spent[player],
                                  node.spent[1 - player]) / BIG_BLIND_CHIPS


def response_values(problem, strategy, deadline):
    """Independent response recursion; max AFTER integration, never per deal."""
    for i, original in problem.blueprint.items():
        p = strategy[i]
        if (p.shape != original.shape or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError("Malformed continuation distribution")

    def walk(i, reach):
        deadline.check()
        node = problem.nodes[i]
        if node.player < 0:
            return terminal(problem, node, 1 - problem.hero, reach)
        if node.player == problem.hero:
            return sum((walk(child, reach * strategy[i][:, j])
                        for j, (_, child) in enumerate(node.children)), np.zeros_like(reach))
        return np.max(np.stack([walk(child, reach) for _, child in node.children]), axis=0)
    return walk(0, problem.reach)


def certify(problem, strategy, bound, deadline, tolerance_bb=1e-8):
    mass = problem.space.mass(problem.reach)
    # Do not silently discard positive but ill-conditioned opponent types.
    if np.any((mass > 0) & (mass < 1e-12)) or np.any(mass < -1e-12):
        raise ValueError("Ill-conditioned root counterfactual mass")
    values = response_values(problem, strategy, deadline)
    supported = mass > 0
    if not supported.any():
        raise ValueError("No compatible counterfactual support")
    delta = (values[supported] - bound[supported]) / mass[supported]
    return dict(passed=bool(np.max(delta) <= tolerance_bb),
                max_increase_bb=float(np.max(delta)), mean_change_bb=float(np.mean(delta)),
                supported_opponent_hands=int(supported.sum()), tolerance_bb=tolerance_bb)


def matching(regrets):
    positive = np.maximum(regrets, 0)
    totals = positive.sum(axis=1, keepdims=True)
    return np.divide(positive, totals, out=np.full_like(positive, 1 / positive.shape[1]),
                     where=totals > 0)


class Gadget:
    """Simultaneous regret-matching+; linear own-reach-weighted hero average.

    Chance mass is proportional to hero blueprint reach over compatible deals.
    Opponent root opt-out utility is its blueprint counterfactual BR value.
    No guarantee relies on a finite number of these optimisation steps: the
    independent per-hand certificate is the publication gate.
    """
    def __init__(self, problem, bound):
        self.p, self.bound, self.iteration = problem, bound, 0
        k = len(problem.reach)
        self.regrets = {i: (problem.blueprint[i] * .001 if n.player == problem.hero
                           else np.zeros((k, len(n.children))))
                        for i, n in enumerate(problem.nodes) if n.player >= 0}
        self.gadget_regrets = np.tile([0., .001], (k, 1))
        self.average = {i: np.zeros_like(v) for i, v in problem.blueprint.items()}

    def step(self, deadline):
        self.iteration += 1
        p = self.p
        policies = {i: matching(v) for i, v in self.regrets.items()}
        gadget = matching(self.gadget_regrets)
        deltas = {}

        def walk(i, r0, r1):
            deadline.check()
            n = p.nodes[i]
            if n.player < 0:
                return terminal(p, n, 0, r1), terminal(p, n, 1, r0)
            strategy = policies[i]
            children = [walk(child, r0 * strategy[:, j] if n.player == 0 else r0,
                             r1 * strategy[:, j] if n.player == 1 else r1)
                        for j, (_, child) in enumerate(n.children)]
            q = np.stack([c[n.player] for c in children], axis=1)
            own = np.sum(q * strategy, axis=1)
            other = sum((c[1 - n.player] for c in children), np.zeros_like(own))
            deltas[i] = q - own[:, None]
            if n.player == p.hero:
                own_reach = r0 if p.hero == 0 else r1
                self.average[i] += self.iteration * own_reach[:, None] * strategy
            return (own, other) if n.player == 0 else (other, own)

        opponent_reach = gadget[:, 1] / len(p.reach)
        r = (p.reach, opponent_reach) if p.hero == 0 else (opponent_reach, p.reach)
        values = walk(0, *r)
        q = np.stack((self.bound, values[1 - p.hero]), axis=1)
        self.gadget_regrets = np.maximum(0., self.gadget_regrets + q - (q * gadget).sum(1)[:, None])
        for i, delta in deltas.items():
            self.regrets[i] = np.maximum(0., self.regrets[i] + delta)

    def strategy(self):
        return {i: np.divide(v, v.sum(1)[:, None], out=self.p.blueprint[i].copy(),
                             where=v.sum(1)[:, None] > 0) for i, v in self.average.items()}


def solve(game, adapter, board, pre, hero, *, seconds=.5, iterations=10000, clock=time.perf_counter):
    """Cold online work only. No board/solution cache is shared across hands.

    Cooperative deadline checks bound Python work; native kernels can overrun.
    Overruns are measured and gate cloud rollout, not hidden as a hard guarantee.
    The last fully checked candidate survives interruption of a later iteration.
    """
    if seconds is not None and (not np.isfinite(seconds) or seconds < 0):
        raise ValueError("Invalid search budget")
    deadline = Deadline(seconds, clock=clock)
    result, problem, best_score = None, None, 0.
    stats = dict(iterations=0, selected_iteration=None, checks=0, accepted=False, preparation_seconds=0.,
                 preparation_completed=False, rejected_checks=0, best_rejected_max_increase_bb=None,
                 max_increase_bb=0., mean_change_bb=0., strategy_change=0., reason="no_verified_improvement")
    try:
        problem = prepare(game, adapter, tuple(board), tuple(pre), hero, deadline)
        bound = response_values(problem, problem.blueprint, deadline)
        stats["preparation_seconds"] = clock() - deadline.start
        stats["preparation_completed"] = True
        gadget = Gadget(problem, bound)
        last_step = 0.
        for _ in range(iterations):
            if deadline.end - clock() < max(.002, 2 * last_step):
                break
            started = clock()
            gadget.step(deadline)
            stats["iterations"] = gadget.iteration
            candidate = gadget.strategy()
            certificate = certify(problem, candidate, bound, deadline)
            stats["checks"] += 1
            if not certificate["passed"]:
                stats["rejected_checks"] += 1
                previous = stats["best_rejected_max_increase_bb"]
                stats["best_rejected_max_increase_bb"] = min(
                    float("inf") if previous is None else previous, certificate["max_increase_bb"])
            if certificate["passed"] and certificate["mean_change_bb"] < best_score - 1e-10:
                change = max(float(np.max(np.abs(v - problem.blueprint[i]))) for i, v in candidate.items())
                if change > 1e-6:
                    deadline.check()
                    result, best_score = candidate, certificate["mean_change_bb"]
                    stats.update(accepted=True, selected_iteration=gadget.iteration, reason="verified", strategy_change=change,
                                 max_increase_bb=certificate["max_increase_bb"], mean_change_bb=best_score)
            last_step = clock() - started
    except DeadlineExceeded:
        if result is None:
            stats["reason"] = "deadline"
    stats["elapsed_seconds"] = clock() - deadline.start
    if not stats["preparation_completed"]:
        stats["preparation_seconds"] = stats["elapsed_seconds"]
    stats["overrun_seconds"] = max(0., stats["elapsed_seconds"] - seconds) if seconds is not None else 0.
    # A late result is never deployed, even if its strategy passed the certificate.
    if seconds is not None and stats["elapsed_seconds"] > seconds:
        result = None
        stats.update(accepted=False, reason="deadline_overrun")
    return problem, result, stats
