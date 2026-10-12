"""Frozen, stdlib-only Experiment 27 protocol; no Exp21 gate overrides."""
from dataclasses import replace
import hashlib
import json

if __package__:
    from .config import configuration
else:
    import importlib.util
    from pathlib import Path
    import sys
    _s = importlib.util.spec_from_file_location("exp27_base_config", Path(__file__).with_name("config.py"))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_s.name] = _m
    _s.loader.exec_module(_m)
    configuration = _m.configuration

ARMS = ("smooth_l1", "half_mse")
CONTROLS = ("leduc_uniform", "fhp_fold", "fhp_call")


def stream(purpose, *coordinates):
    raw = json.dumps(["fhp-exp27-v1", purpose, *coordinates]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def protocol(smoke=False):
    return dict(protocol="fhp_exp27_attacker_development_v1", smoke=smoke, arms=list(ARMS),
                controls=list(CONTROLS), fold_control_probability=.8,
                repetitions=2, threads=2, processes=4,
                control_seconds=.6 if smoke else 300., attack_seconds=.6 if smoke else 3600.,
                development_pairs=4 if smoke else 2000, final_pairs=8 if smoke else 10000,
                control_pairs=8 if smoke else 2000, calibration_hands_per_seat=8 if smoke else 512,
                training_worker_seconds=180 if smoke else 5400,
                assessment_worker_seconds=600 if smoke else 3600,
                replication_seeds=[1, 2], primary_source_seed=0,
                selection="Among control-eligible arms, lock highest seed-0 development payoff; ties prefer smooth_l1",
                qualification="Independent payoff CI95 lower > 0 AND gain-over-own-initial CI95 lower > 0",
                control_gate="Each weak FHP control: positive pooled held-out payoff lower CI; Leduc: each repetition positive exact payoff, positive mean learning gain",
                primary_contrast="half_mse minus smooth_l1 on seed 0; conditional paired-deal interval",
                loss_scale="half MSE matches beta=1 Smooth L1 curvature near zero; gradient norm clip 10 unchanged",
                warning="Development study: conditional deal-level intervals, not exact FHP exploitability or a general multi-seed treatment-effect claim")


def cell_protocol(spec, case, arm, source_seed=0):
    if arm not in ARMS or case not in (*CONTROLS, "exp10") or source_seed not in (0, 1, 2):
        raise ValueError("Invalid development cell")
    if case != "exp10" and source_seed != 0:
        raise ValueError("Controls do not use multiple defender seeds")
    seconds = spec["attack_seconds"] if case == "exp10" else spec["control_seconds"]
    cfg = replace(configuration(spec["smoke"]), attack_seconds=seconds, episode_limit=10_000_000)
    return dict(spec, case=case, arm=arm, source_seed=source_seed, learner=cfg.to_dict(),
                checkpoints_seconds=[0., seconds / 4, seconds] if case == "exp10" else [0., seconds],
                exploration_seconds=(.15 if spec["smoke"] else 900.) if case == "exp10" else seconds,
                reward_scale=1. if case == "leduc_uniform" else 100.)


def replication_allowed(result):
    """Never turn execution-smoke scores into scientific authorisation."""
    return (result.get("smoke") is False and result.get("qualified") is True
            and result.get("selected_arm") in ARMS)
