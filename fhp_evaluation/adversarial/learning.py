"""Batched CPU Double-DQN attackers and on-policy policy-gradient repair.

Only acting-player information states enter networks. Hidden simulator state
is used to play the game, never as a feature, critic input or training label
other than the ordinary terminal return. Returns are in big blinds (gamma=1).
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..game import BIG_BLIND_CHIPS


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def masked_logs(model, x, masks):
    if not masks.any(dim=-1).all():
        raise ValueError("No legal action in policy query")
    logits = model(x)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite network output")
    return torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)


class NetworkPolicy:
    def __init__(self, model, encoder, *, greedy=False):
        self.model, self.feature_encoder, self.greedy = model, encoder, greedy

    def batch(self, states):
        x = np.asarray([self.feature_encoder.information_state(s, s.current_player())
                        for s in states], dtype=np.float32)
        mask = np.asarray([s.legal_actions_mask() for s in states], dtype=bool)
        with torch.inference_mode():
            logs = masked_logs(self.model, torch.from_numpy(x), torch.from_numpy(mask))
            if self.greedy:
                values = F.one_hot(logs.argmax(-1), 3).to(torch.float32).numpy()
            else:
                values = logs.exp().numpy()
        return x, mask, values

    def action_probabilities(self, state, player_id=None):
        if player_id is not None and state.current_player() != player_id:
            raise ValueError("Policy queries require the acting player's information")
        probabilities = self.batch([state])[2][0]
        return {a: float(probabilities[a]) for a in state.legal_actions()}


def attacker_model(width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 256),
                             nn.ReLU(), nn.Linear(256, 3))


@dataclass
class Episode:
    rows: list
    reward: float


def collect(game, learner, opponents, seats, rng, *, epsilon=0, reward_scale=BIG_BLIND_CHIPS):
    """Ragged hands with batched neural queries, no Python replay-row objects retained."""
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Positive finite reward scale required")
    count = len(seats)
    if len(opponents) != count or any(s not in (0, 1) for s in seats):
        raise ValueError("One legal learning seat and opponent per hand required")
    states = [game.new_initial_state() for _ in seats]
    rows = [[] for _ in seats]
    while True:
        groups = {}
        active = False
        for index, state in enumerate(states):
            while state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probabilities)))
            if state.is_terminal():
                continue
            active = True
            is_learner = state.current_player() == seats[index]
            actor = learner if is_learner else opponents[index]
            groups.setdefault((id(actor), is_learner), (actor, []))[1].append(index)
        if not active:
            break
        for (_, is_learner), (actor, indices) in groups.items():
            x, masks, probabilities = actor.batch([states[i] for i in indices])
            if is_learner and epsilon:
                probabilities = (1 - epsilon) * probabilities + epsilon * masks / masks.sum(1)[:, None]
            # One independent uniform per hand. CDF sampling avoids a Python
            # torch call / numpy.choice distribution setup at every decision.
            probabilities = np.asarray(probabilities, dtype=np.float64)
            probabilities[~masks] = 0
            probabilities /= probabilities.sum(1)[:, None]
            cumulative = np.cumsum(probabilities, axis=1)
            last_legal = 2 - np.argmax(masks[:, ::-1], axis=1)
            cumulative[np.arange(3)[None, :] >= last_legal[:, None]] = 1
            actions = (rng.random(len(indices))[:, None] >= cumulative).sum(1)
            for j, i in enumerate(indices):
                action = int(actions[j])
                if not masks[j, action]:
                    raise ValueError("Sampled an illegal action")
                if is_learner:
                    rows[i].append((x[j], masks[j], action))
                states[i].apply_action(action)
    return [Episode(row, float(state.returns()[seat]) / reward_scale)
            for row, state, seat in zip(rows, states, seats)]


class TransitionReplay:
    """Contiguous float32/bool/int64 circular storage; vectorised minibatch gathers."""
    def __init__(self, capacity, width):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.x = np.empty((capacity, width), np.float32)
        self.next_x = np.empty_like(self.x)
        self.next_mask = np.empty((capacity, 3), bool)
        self.action = np.empty(capacity, np.int64)
        self.reward = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)

    def append(self, episodes):
        for episode in episodes:
            for j, (x, _, action) in enumerate(episode.rows):
                i = self.cursor
                done = j + 1 == len(episode.rows)
                self.x[i], self.action[i], self.done[i] = x, action, done
                self.reward[i] = episode.reward if done else 0
                if done:
                    self.next_x[i].fill(0)
                    self.next_mask[i].fill(False)
                else:
                    self.next_x[i], self.next_mask[i], _ = episode.rows[j + 1]
                self.cursor = (i + 1) % self.capacity
                self.size = min(self.capacity, self.size + 1)

    def sample(self, rng, size):
        indices = rng.integers(0, self.size, size=size)
        return tuple(torch.from_numpy(v[indices]) for v in
                     (self.x, self.action, self.reward, self.next_x, self.next_mask, self.done))


def double_dqn_targets(online, target, next_x, next_mask, rewards, dones):
    with torch.no_grad():
        actions = online(next_x).masked_fill(~next_mask, -1e9).argmax(1)
        bootstrap = target(next_x).gather(1, actions[:, None]).squeeze(1)
        return rewards + torch.where(dones, torch.zeros_like(bootstrap), bootstrap)


def step(optimizer, model, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step()


def attacker_loss(predicted, desired, kind="smooth_l1"):
    """Half-MSE matches Smooth L1's local curvature (beta=1), not its tails."""
    if kind == "smooth_l1":
        return F.smooth_l1_loss(predicted, desired)
    if kind == "half_mse":
        return .5 * F.mse_loss(predicted, desired)
    raise ValueError(f"Unknown attacker loss: {kind}")


def train_attacker(game, victim, seat, seed, cfg, progress=lambda **kw: None, *,
                   checkpoint_seconds=(), on_checkpoint=None, exploration_seconds=None,
                   loss_kind="smooth_l1", reward_scale=BIG_BLIND_CHIPS):
    """Original Exp21 learner, with optional observational Exp26 snapshots.

    Snapshot I/O is excluded from the active clock. No evaluation, RNG draws or
    fitting occurs in the snapshot callback. The default Exp21 exploration
    schedule remains tied to cfg.attack_seconds; Exp26 fixes it to 900 seconds.
    Checkpoints are taken at completed collection/update-wave boundaries.
    """
    if loss_kind not in ("smooth_l1", "half_mse"):
        raise ValueError("Unknown attacker loss")
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Positive finite reward scale required")
    checkpoints = list(checkpoint_seconds)
    if (checkpoints != sorted(set(checkpoints)) or any(t < 0 or t > cfg.attack_seconds for t in checkpoints)
            or (bool(checkpoints) != (on_checkpoint is not None))):
        raise ValueError("Ordered, unique in-budget snapshots require a callback")
    exploration_seconds = cfg.attack_seconds if exploration_seconds is None else exploration_seconds
    if exploration_seconds <= 0:
        raise ValueError("Positive exploration horizon required")
    rng = np.random.default_rng(seed)
    width = victim.feature_encoder.policy_layout.total_size
    model = attacker_model(width, seed)
    initial = deepcopy(model)
    target = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.attack_lr)
    actor = NetworkPolicy(model, victim.feature_encoder, greedy=True)
    replay = TransitionReplay(cfg.replay_capacity, width)
    started, waves, updates, hands = time.monotonic(), 0, 0, 0
    excluded_seconds = 0.
    losses = []

    def elapsed():
        return time.monotonic() - started - excluded_seconds

    def snapshot():
        nonlocal excluded_seconds
        active = elapsed()
        while checkpoints and checkpoints[0] <= active:
            nominal = checkpoints.pop(0)
            stamp = time.monotonic()
            on_checkpoint(nominal, dict(model=model.state_dict(), target=target.state_dict(),
                optimizer=optimizer.state_dict(), rng=deepcopy(rng.bit_generator.state), seat=seat,
                seed=seed, width=width, replay_retained=False,
                continuation="Playable attacker only; transition replay not retained, not exact training resume"),
                dict(active_seconds=active, nominal_seconds=nominal, hands=hands, updates=updates,
                     replay_rows=replay.size, waves=waves, hash=model_hash(model),
                     mean_last_loss=float(np.mean(losses[-100:])) if losses else None))
            excluded_seconds += time.monotonic() - stamp

    if checkpoints:
        snapshot()
    while (waves == 0 or elapsed() < cfg.attack_seconds) and hands < cfg.episode_limit:
        fraction = min(1., elapsed() / exploration_seconds)
        epsilon = 1. - .95 * min(1., fraction / .8)
        n = min(cfg.environments, cfg.episode_limit - hands)
        episodes = collect(game, actor, [victim] * n, [seat] * n, rng, epsilon=epsilon,
                           reward_scale=reward_scale)
        replay.append(episodes)
        hands += n
        waves += 1
        if replay.size >= cfg.warmup:
            for _ in range(cfg.dqn_updates_per_wave):
                x, a, r, nx, nm, done = replay.sample(rng, cfg.batch_size)
                desired = double_dqn_targets(model, target, nx, nm, r, done)
                predicted = model(x).gather(1, a[:, None]).squeeze(1)
                loss = attacker_loss(predicted, desired, loss_kind)
                step(optimizer, model, loss)
                losses.append(float(loss.detach()))
                updates += 1
                if updates % cfg.target_sync_updates == 0:
                    target.load_state_dict(model.state_dict())
        progress(kind="attacker", seat=seat, hands=hands, updates=updates,
                 elapsed_seconds=elapsed(), replay_rows=replay.size)
        if checkpoints:
            snapshot()
    if checkpoints:
        raise RuntimeError("Episode limit prevented the fixed snapshot schedule from completing")
    if updates == 0:
        raise RuntimeError("Attacker budget ended before any network update")
    state = dict(model=model.state_dict(), target=target.state_dict(), optimizer=optimizer.state_dict(),
                 rng=rng.bit_generator.state, seat=seat, seed=seed, width=width,
                 replay_retained=False, continuation="new replay required; not exact training resume")
    metrics = dict(seconds=time.monotonic() - started, hands=hands, updates=updates,
                   mean_last_loss=float(np.mean(losses[-100:])), hash=model_hash(model),
                   initial_hash=model_hash(initial), seat=seat, seed=seed,
                   active_seconds=elapsed(), snapshot_seconds=excluded_seconds,
                   exploration_seconds=exploration_seconds, loss_kind=loss_kind,
                   reward_scale=reward_scale)
    return actor, NetworkPolicy(initial, victim.feature_encoder, greedy=True), state, metrics


def replay_loss(model, replay, rng, size):
    x, y, mask = replay.sample(rng, size)
    return -(y * masked_logs(model, x, mask)).sum(1).mean() * replay.scale


def repair_loss(model, reference, episodes, baseline, *, preserve):
    rows = [(row, ep.reward) for ep in episodes for row in ep.rows]
    if not rows:
        return None, None
    x = torch.from_numpy(np.stack([row[0] for row, _ in rows]))
    masks = torch.from_numpy(np.stack([row[1] for row, _ in rows]))
    actions = torch.tensor([row[2] for row, _ in rows], dtype=torch.long)
    returns = torch.tensor([r for _, r in rows], dtype=torch.float32)
    logs = masked_logs(model, x, masks)
    # Sum per trajectory, then average over ALL hands, including no-decision
    # hands. Averaging over decision rows would reweight long betting paths.
    pg = -(logs.gather(1, actions[:, None]).squeeze(1) * (returns - baseline)).sum() / len(episodes)
    kl = pg.new_zeros(())
    if preserve:
        with torch.no_grad():
            anchor = masked_logs(reference, x, masks)
        kl = (anchor.exp() * (anchor - logs)).sum(1).mean()
    return pg, kl


def repair(game, student, reference, replay, pool, optimizer, seed, cfg, *, preserve,
           progress=lambda **kw: None):
    rng = np.random.default_rng(seed)
    started, updates, hands, baseline = time.monotonic(), 0, 0, 0.
    while (updates == 0 or time.monotonic() - started < cfg.repair_seconds) and hands < cfg.episode_limit:
        n = min(cfg.environments, cfg.episode_limit - hands)
        seats = [(hands + i) % 2 for i in range(n)]
        # Uniform historical population, sampled once per hand, not per action.
        opponents = [pool[1 - seat][int(rng.integers(len(pool[1 - seat])))] for seat in seats]
        episodes = collect(game, student, opponents, seats, rng)
        pg, kl = repair_loss(student.model, reference.model, episodes, baseline, preserve=preserve)
        hands += n
        if pg is not None:
            ce = replay_loss(student.model, replay, rng, cfg.replay_batch_size) if preserve else pg.new_zeros(())
            loss = pg + cfg.replay_coefficient * ce + cfg.kl_coefficient * kl
            step(optimizer, student.model, loss)
            updates += 1
        # Previous-batch baseline: no action-dependent / same-batch centring bias.
        baseline = .9 * baseline + .1 * float(np.mean([e.reward for e in episodes]))
        progress(kind="repair", updates=updates, hands=hands,
                 elapsed_seconds=time.monotonic() - started, preserve=preserve)
    if not updates:
        raise RuntimeError("No policy-gradient updates")
    return dict(seconds=time.monotonic() - started, updates=updates, hands=hands,
                rng=rng.bit_generator.state, baseline=baseline)


def ordinary_fit(student, replay, seed, cfg, seconds, progress=lambda **kw: None):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(student.model.parameters(), lr=cfg.policy_lr)
    started, updates = time.monotonic(), 0
    while updates == 0 or time.monotonic() - started < seconds:
        loss = replay_loss(student.model, replay, rng, cfg.replay_batch_size)
        step(optimizer, student.model, loss)
        updates += 1
        progress(kind="ordinary", updates=updates, elapsed_seconds=time.monotonic() - started)
    return dict(seconds=time.monotonic() - started, matched_budget_seconds=seconds, updates=updates,
                rng=rng.bit_generator.state), optimizer
