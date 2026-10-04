"""Validated, bounded batches over native frozen behavioural policies.

Native repos are explicit dependencies, not copied implementations. UCV and VR
both own a package called vr_deep_cfr; refuse a collision instead of silently
using whichever solver Python happened to import first. Run families in separate
processes (the profiling/comparison runner does this automatically).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ..loaders import LoadedCheckpointPolicy, sha256_file


def _distributions(values, masks):
    values = np.asarray(values, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if values.shape != masks.shape or not np.isfinite(values).all():
        raise ValueError("Invalid policy batch shape or non-finite probabilities")
    if np.any(values < 0) or np.any(values[~masks] != 0):
        raise ValueError("Negative probability or probability on an illegal action")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0) or not np.allclose(totals, 1., atol=2e-6, rtol=2e-6):
        raise ValueError("Policy distributions must have unit legal mass")
    return values / totals


class PolicyAdapter:
    """Scalar OpenSpiel interface plus an explicit batched behavioural interface."""

    def __init__(self, native, *, batch_size=1024, metadata=None, backend="scalar", reference=None):
        if int(batch_size) < 1:
            raise ValueError("Positive batch_size required")
        if hasattr(native, "selected_iterations") or hasattr(native, "begin_episode"):
            raise ValueError("Best response requires a behavioural mixture, not a sampled model")
        self.native, self.batch_size, self.backend = native, int(batch_size), backend
        self.reference = native if reference is None else reference
        self.metadata = dict(metadata or {})
        self.stats = dict(rows=0, batches=0, query_seconds=0.)

    def batch_probabilities(self, states, player):
        states = list(states)
        if not states:
            return np.empty((0, 3), dtype=np.float64)
        outputs = []
        for offset in range(0, len(states), self.batch_size):
            batch = states[offset:offset + self.batch_size]
            if any(s.is_terminal() or s.is_chance_node() or s.current_player() != player for s in batch):
                raise ValueError("Action queries require the acting player")
            masks = np.asarray([s.legal_actions_mask(player) for s in batch], dtype=bool)
            started = time.perf_counter()
            if self.backend == "sd_cfr":
                _, values = self.native.batch_reach_and_probabilities(batch, player)
            elif self.backend in ("native_logits", "raw_mlp"):
                encoder = getattr(self.native, "feature_encoder", None)
                inputs = np.asarray([
                    s.information_state_tensor(player) if encoder is None
                    else encoder.information_state(s, player) for s in batch
                ], dtype=np.float32)
                with torch.inference_mode():
                    logits = (self.native._logits(inputs) if self.backend == "raw_mlp"
                              else self.native.model(torch.from_numpy(inputs)))
                    if logits.shape != masks.shape or not torch.isfinite(logits).all():
                        raise ValueError("Invalid policy logits")
                    logits = logits.masked_fill(~torch.from_numpy(masks), -torch.inf)
                    values = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                values = np.zeros(masks.shape, dtype=np.float64)
                for row, state in enumerate(batch):
                    raw = self.native.action_probabilities(state, player)
                    if set(raw) - set(state.legal_actions(player)):
                        raise ValueError("Illegal action in scalar policy")
                    for action, probability in raw.items():
                        values[row, action] = probability
            outputs.append(_distributions(values, masks))
            self.stats["query_seconds"] += time.perf_counter() - started
            self.stats["batches"] += 1
            self.stats["rows"] += len(batch)
        return np.concatenate(outputs)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        row = self.batch_probabilities([state], player)[0]
        return {a: float(row[a]) for a in state.legal_actions(player)}

    def check_scalar_parity(self, states, tolerance=3e-6):
        maximum = 0.
        for player in (0, 1):
            selected = [s for s in states if s.current_player() == player]
            actual = self.batch_probabilities(selected, player)
            for state, row in zip(selected, actual):
                raw = self.reference.action_probabilities(state, player)
                if set(raw) - set(state.legal_actions(player)):
                    raise ValueError("Illegal action in scalar reference policy")
                expected = np.asarray([raw.get(a, 0.) for a in range(3)], dtype=np.float64)
                expected = _distributions(expected[None, :],
                                          [state.legal_actions_mask(player)])[0]
                maximum = max(maximum, float(np.max(np.abs(row - expected))))
        if maximum > tolerance:
            raise ValueError(f"Native/batched policy mismatch: {maximum} > {tolerance}")
        return dict(max_absolute_error=maximum, tolerance=tolerance, passed=True)

    def preflop_policy(self, game, config):
        if self.backend == "sd_cfr":
            from deep_cfr_poker.sd_cfr_lbr import ExactSDCFRLocalBestResponsePolicy
            return ExactSDCFRLocalBestResponsePolicy(game, self.native, config=config)
        from ..lbr import LocalBestResponsePolicy
        return LocalBestResponsePolicy(game, self, config=config)


def _native_root(root, packages):
    root = Path(root).resolve()
    for package in packages:
        if not (root / package / "__init__.py").is_file():
            raise ValueError(f"Native repository missing {package}: {root}")
        loaded = sys.modules.get(package)
        if loaded is not None:
            origin = Path(loaded.__file__).resolve()
            if root not in origin.parents:
                raise RuntimeError(f"Package collision for {package}: {origin}. Use a fresh process.")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    for package in packages:
        origin = Path(importlib.util.find_spec(package).origin).resolve()
        if root not in origin.parents:
            raise RuntimeError(f"Wrong native package origin: {origin}")
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(native_repo=str(root), native_commit=git("rev-parse", "HEAD"),
                native_dirty=bool(git("status", "--porcelain")))


def load_target(game, path, *, family, repo_root=None, batch_size=1024, model_batch_size=32):
    """Load trusted local checkpoints only; pickle/Torch formats can execute code."""
    path = Path(path).resolve()
    metadata = dict(family=family, checkpoint=str(path), checkpoint_sha256=sha256_file(path))
    reference = None
    if family == "raw_mlp":
        native = LoadedCheckpointPolicy(game, path)
        payload = native.payload
        if (payload.get("feature_encoder") is not None
                or payload.get("policy_model", {}).get("type", "mlp_v1") != "mlp_v1"
                or native.layers[0][0].shape[1] != game.information_state_tensor_size()):
            raise ValueError("raw_mlp is only for unencoded plain MLP checkpoints")
        backend = "raw_mlp"
    else:
        if repo_root is None:
            raise ValueError("Native checkpoint families require --repo-root")
        packages = {"ucv": ("fhp_escher", "vr_deep_cfr"),
                    "vr_deep": ("fhp_vr_deep", "vr_deep_cfr"),
                    "sd_cfr": ("deep_cfr_poker",)}
        if family not in packages:
            raise ValueError(f"Unknown policy family: {family}")
        metadata.update(_native_root(repo_root, packages[family]))
        # Native model construction is permitted but must not consume simulator RNGs.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                if family == "ucv":
                    from fhp_escher.checkpointing import LoadedFHPPolicy
                    native = LoadedFHPPolicy(game, path)
                elif family == "vr_deep":
                    from vr_deep_cfr.policy_snapshots import LoadedVRPolicy
                    native = LoadedVRPolicy(game, path)
                else:
                    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, DiskBehaviouralPolicy
                    from deep_cfr_poker.sd_cfr_lbr import BatchedDiskBehaviouralPolicy
                    reader = DiskArchiveReader(path, game, verify=True)
                    native = BatchedDiskBehaviouralPolicy(
                        reader, game, model_batch_size=model_batch_size,
                        state_batch_size=batch_size, cache_size=256, device="cpu")
                    reference = DiskBehaviouralPolicy(reader, game, model_batch_size=model_batch_size,
                                                      cache_size=256)
                    metadata.update(models_per_player=reader.count, weighting="uniform",
                                    archive_integrity_verified=True)
        finally:
            np.random.set_state(numpy_state)
        backend = "sd_cfr" if family == "sd_cfr" else "native_logits"
    return PolicyAdapter(native, batch_size=batch_size, backend=backend, metadata=metadata,
                         reference=reference)
