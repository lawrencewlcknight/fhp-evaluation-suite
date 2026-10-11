"""Stdlib-only, frozen Exp26 design. Negative findings still complete this study."""
from dataclasses import replace
import hashlib
import json

if __package__:
    from .config import configuration
else:
    # Batch preparation must remain stdlib-only; the package __init__ loads ML.
    import importlib.util
    from pathlib import Path
    import sys
    _spec = importlib.util.spec_from_file_location("exp26_base_config", Path(__file__).with_name("config.py"))
    _base = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
    configuration = _base.configuration


def protocol(smoke=False):
    cfg = configuration(smoke)
    seconds = [0., .05, .15, .30, .60] if smoke else [0., 300., 900., 1800., 3600.]
    # Preserve Exp21's first-15-minute exploration schedule during extension.
    learner = replace(cfg, attack_seconds=seconds[-1], episode_limit=10_000_000)
    return dict(protocol="fhp_exp26_attacker_diagnostic_v1", source_seed=0, smoke=smoke,
                attacker_repetitions=2, checkpoints_seconds=seconds,
                learner=learner.to_dict(), exploration_seconds=.15 if smoke else 900.,
                curve_pairs=4 if smoke else 2000, final_pairs=8 if smoke else 10000,
                calibration_hands_per_seat=8 if smoke else 512,
                processes=4, threads=2, training_worker_seconds=180 if smoke else 5400,
                evaluation_worker_seconds=300 if smoke else 10800,
                stopping_rule="Fixed schedule; no payoff-dependent extension or optional stopping",
                selection="Lock best repetition at 15m and 60m separately using development deals only",
                warning="One source seed; conditional Monte Carlo intervals, not exact exploitability or replication across trained defenders")


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def stream(purpose, *coordinates):
    raw = json.dumps(["fhp-exp26-v1", purpose, *coordinates]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
