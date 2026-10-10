"""Predeclared Experiment 21 pilot. Smoke budgets are not scientific results."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json

SOURCE_RUN = "exp10-features-20261001-161740"
ARMS = ("unchanged", "ordinary", "adversarial", "preserved")
MANIFEST_HASHES = (
    "34fa347573184a3b2983c2e31250013a0b39e13593810700824623dedbdf9672",
    "da904251276543a42d0dfc038a3d09454a7f267f38588121dd25dd2cce9e232e",
    "b5dd5fd5cd73249affac0b91eaf971d60b59a28d6f24108c7075e484d46c94db",
)


@dataclass(frozen=True)
class Config:
    version: int = 1
    smoke: bool = False
    threads: int = 2
    rounds: int = 3
    attackers_per_seat: int = 2
    attack_seconds: float = 900
    repair_seconds: float = 300
    episode_limit: int = 10_000_000
    environments: int = 32
    replay_capacity: int = 100_000
    warmup: int = 1024
    batch_size: int = 256
    replay_batch_size: int = 2048
    dqn_updates_per_wave: int = 4
    target_sync_updates: int = 250
    attack_lr: float = 0.0003
    policy_lr: float = 0.0001
    replay_coefficient: float = 1.0
    kl_coefficient: float = 1.0
    selection_pairs: int = 1000
    gate_pairs: int = 2000
    test_pairs: int = 5000
    br_pairs: int = 500
    preflop_rollouts: int = 4096

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def configuration(smoke=False):
    base = Config()
    return replace(base, smoke=True, rounds=1, attack_seconds=0.15, repair_seconds=0.1,
                   episode_limit=16, environments=4, replay_capacity=128, warmup=4,
                   batch_size=4, replay_batch_size=8, dqn_updates_per_wave=1,
                   target_sync_updates=2, selection_pairs=4, gate_pairs=4,
                   test_pairs=4, br_pairs=2, preflop_rollouts=8) if smoke else base


def stream(source_seed, purpose, *coordinates):
    """Stable independent namespaces; do not use Python's salted hash()."""
    message = json.dumps(["fhp-exp21-v1", int(source_seed), purpose, *coordinates])
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") % (2**63 - 1)
