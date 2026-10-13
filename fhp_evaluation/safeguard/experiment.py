"""Prespecified Experiment 34 design; cloud preparation needs only the stdlib."""
import hashlib
import importlib.util
from pathlib import Path

_source = importlib.util.spec_from_file_location(
    "safeguard_exp16_registry", Path(__file__).resolve().parents[1] / "strategic_audit/exp16.py")
_registry = importlib.util.module_from_spec(_source)
_source.loader.exec_module(_registry)
RULES = _registry.RULES


def required_sources(mode):
    if mode not in ("smoke", "full"):
        raise ValueError("Mode must be smoke or full")
    rows = _registry.checkpoint_registry("development")
    return {k: v for k, v in rows.items() if mode == "full" or v["training_seed"] == 0}


def config(mode):
    full = mode == "full"
    return dict(protocol="ucv-exp34-exact-terminal-call-v1", mode=mode,
                seeds=[0, 1, 2] if full else [0], checkpoint_registry=required_sources(mode),
                primary_pairs=1_000_000 if full else 32,
                panel_pairs=20_000 if full else 8, response_pairs=2000 if full else 2,
                shard_pairs=2000 if full else 8, response_shard_pairs=25 if full else 2,
                processes=4, torch_threads=4, batch_size=256, rollout_samples=4096 if full else 16,
                stream_seed=202610130034, response_seed=202610140034,
                strategic_stage="development" if full else "pilot",
                session_seconds=46*3600 if full else 1800,
                # Advisory estimate only; the unchanged Batch/session limits bound cost.
                gameplay_planning_hours=30., projection_margin=2., profile_shards_per_matchup=2)


def opponents(spec, seed):
    labels = [f"exp16_seed{s}_48h" for s in spec["seeds"]]
    labels += [f"exp16_seed{seed}_{h}h" for h in (24, 36, 42)]
    labels += [f"exp9_seed{s}_24h" for s in spec["seeds"]]
    return labels + list(RULES) + [f"{method}_{arm}" for method in ("lbr", "full_flop")
                                  for arm in ("unmodified", "guarded")]


def count(spec, seed, opponent):
    if opponent == f"exp16_seed{seed}_48h":
        return spec["primary_pairs"]
    return spec["response_pairs"] if is_response(opponent) else spec["panel_pairs"]


def is_response(opponent):
    return opponent.startswith(("lbr_", "full_flop_"))


def tasks(spec):
    for seed in spec["seeds"]:
        for opponent in opponents(spec, seed):
            n = count(spec, seed, opponent)
            step = spec["response_shard_pairs"] if is_response(opponent) else spec["shard_pairs"]
            for start in range(0, n, step):
                yield dict(seed=seed, opponent=opponent, start=start, count=min(step, n-start),
                           id=f"s{seed}_{opponent}_{start:07d}")


def profiling_tasks(spec):
    """First two production shards per matchup, retained in the final sample.

    Selection is fixed before play and independent of outcomes. Use normal shard
    sizes (including response shards) so inference batching is representative.
    Interleave the repeats rather than timing one matchup twice immediately.
    """
    selected = []
    for task in tasks(spec):
        step = spec["response_shard_pairs"] if is_response(task["opponent"]) else spec["shard_pairs"]
        repeat = task["start"] // step
        if repeat < spec["profile_shards_per_matchup"]:
            selected.append((repeat, task))
    return [task for _, task in sorted(selected, key=lambda item: item[0])]


def stream_seed(spec, seed, opponent, pair):
    # Index-addressable stream: worker count/shard boundaries/recovery do not change deals.
    key = f"{spec['protocol']}:{spec['stream_seed']}:{seed}:{opponent}:{pair}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
