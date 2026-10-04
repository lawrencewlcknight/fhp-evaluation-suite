"""Frozen-policy cohort evaluation using the UCV retrospective deal protocol.

No training or policy distillation. Native loaders are explicit import paths;
the suite owns opponents, LBR, duplicate play, uncertainty and durable tasks.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from .duplicate import play_hand
from .game import MILLI_BIG_BLINDS_PER_CHIP
from .lbr import LBRConfig, LocalBestResponsePolicy
from .rule_agents import published_rule_agents
from .statistics import sample_summary


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe_json(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def portable(value):
    """Local relocation is allowed; scientific identity and file hashes are not."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def initialise_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@lru_cache(maxsize=40)
def load_policy(loader, path, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ValueError(f"Policy checksum mismatch: {path}")
    module, function = loader.split(":")
    return getattr(importlib.import_module(module), function)(path)


def duplicate_samples(game, a, b, count, seed):
    """Split chance/action arrays exactly as in the UCV retrospective runners."""
    rng = np.random.default_rng(seed)
    chance = rng.integers(0, 2**63 - 1, size=count, dtype=np.int64)
    action = rng.integers(0, 2**63 - 1, size=count, dtype=np.int64)
    values = []
    for c, s in zip(chance, action):
        left = play_hand(game, (a, b), chance_seed=int(c), action_seed=int(s))[0]
        right = play_hand(game, (b, a), chance_seed=int(c), action_seed=int(s))[1]
        values.append(0.5 * (left + right) * MILLI_BIG_BLINDS_PER_CHIP)
    return values


def score_task(task):
    start = time.monotonic()
    row = task["a"]
    game, a = load_policy(task["loader"], row["path"], row["sha256"])
    if task["kind"] == "rule":
        b = published_rule_agents(game)[task["opponent"]]
    elif task["kind"] == "lbr":
        b = a
        a = LocalBestResponsePolicy(game, b, config=LBRConfig(
            preflop_rollout_samples=task["rollouts"], seed=task["lbr_seed"]))
    elif task["kind"] in ("direct", "temporal"):
        row = task["b"]
        _, b = load_policy(task["loader"], row["path"], row["sha256"])
    else:
        raise ValueError(f"Unknown task kind: {task['kind']}")
    values = duplicate_samples(game, a, b, task["deals"], task["evaluation_seed"])
    return dict(task=portable(task), summary=sample_summary(values),
                # Only LBR is sharded. Raw small shards permit exact pooling.
                paired_mbb=values if task["kind"] == "lbr" else None,
                elapsed_seconds=time.monotonic() - start)


def execute_tasks(tasks, output, identity, *, workers, seconds, scorer=score_task):
    """Hard-bounded spawned workers, atomic results, fail-closed resume identity."""
    if workers < 1 or seconds <= 0 or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Invalid worker count, deadline, or duplicate tasks")
    output = Path(output)
    contract = dict(identity=identity, tasks=portable(tasks))
    signature = digest(contract)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError(f"Changed evaluator, policies or protocol: {output}")
    write_json(manifest, dict(signature=signature, contract=contract))
    results, pending = {}, []
    cache = output / "task_results"
    cache.mkdir(exist_ok=True)
    for task in tasks:
        path = cache / (task["id"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["signature"] != signature or saved["result_sha256"] != digest(saved["result"])
                    or saved["result"]["task"] != portable(task)):
                raise ValueError(f"Invalid cached result: {path}")
            results[task["id"]] = saved["result"]
        else:
            pending.append(task)
    if pending:
        deadline = time.monotonic() + seconds
        pool = ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                   mp_context=mp.get_context("spawn"), initializer=initialise_worker)
        try:
            futures = {pool.submit(scorer, task): task for task in pending}
            for future in as_completed(futures, timeout=max(.001, deadline - time.monotonic())):
                # BrokenProcessPool reports an OOM/killed worker immediately;
                # multiprocessing.Pool can silently replace it and lose its task.
                result = future.result()
                task_id = result["task"]["id"]
                write_json(cache / (task_id + ".json"), dict(signature=signature,
                           result_sha256=digest(result), result=result))
                results[task_id] = result
                print(f"{output.name}: {len(results)}/{len(tasks)} completed", flush=True)
        except BaseException:
            # Python 3.11 has no public terminate_workers(). Explicitly terminate
            # children before shutdown so a timed-out LBR cannot block cleanup.
            for process in tuple(pool._processes.values()):
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return [results[t["id"]] for t in tasks]


def cost_projection(probes, tasks, workers):
    """Slowest observed rate per class, twofold margin, and a serial-tail bound."""
    rates = defaultdict(list)
    def key(task):
        # LBR is profiled separately for every target, not extrapolated raw->encoded.
        return (task["kind"], task["a"]["id"] if task["kind"] == "lbr"
                else task.get("opponent", ""))
    for row in probes:
        rates[key(row["task"])].append(row["elapsed_seconds"] / row["task"]["deals"])
    costs = []
    for task in tasks:
        if not rates[key(task)]:
            raise ValueError(f"Unprofiled task class: {key(task)}")
        costs.append(max(rates[key(task)]) * task["deals"])
    return 2.0 * (sum(costs) / workers + max(costs, default=0)) + 120


def collapse(results):
    groups = defaultdict(list)
    for result in results:
        task = result["task"]
        groups[task["match_id"]].append(result)
    rows = []
    for match_id, selected in sorted(groups.items()):
        task = selected[0]["task"]
        if task["kind"] == "lbr":
            values = [v for r in sorted(selected, key=lambda r: r["task"]["shard"])
                      for v in r["paired_mbb"]]
            summary = sample_summary(values)
        elif len(selected) == 1:
            summary = selected[0]["summary"]
        else:
            raise ValueError("Only LBR may be sharded")
        a, b = task["a"], task.get("b", {})
        rows.append(dict(match_id=match_id, kind=task["kind"], experiment=a["experiment"],
                         opponent=task.get("opponent", b.get("experiment")),
                         seed=a["seed"], hours=a["hours"], earlier_hours=b.get("hours"),
                         nodes=a["nodes"], opponent_nodes=b.get("nodes"),
                         **{f"match_{k}": v for k, v in summary.items()}))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    keys = ("kind", "experiment", "opponent", "hours", "earlier_hours")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, selected in groups.items():
        if len({r["seed"] for r in selected}) != len(selected):
            raise ValueError("Duplicate training-seed observations")
        result.append(dict(zip(keys, key), nodes_mean=float(np.mean([r["nodes"] for r in selected])),
                           **{f"seed_{k}": v for k, v in
                              sample_summary([r["match_mean"] for r in selected]).items()}))
    return result


def paired_differences(rows):
    lookup = {(r["kind"], r["experiment"], r["opponent"], r["hours"], r["seed"]): r
              for r in rows if r["kind"] in ("rule", "lbr")}
    differences = []
    for (kind, experiment, opponent, hours, seed), row in lookup.items():
        for earlier in range(1, int(experiment[-1])):
            other = lookup.get((kind, f"exp{earlier}", opponent, hours, seed))
            if other:
                differences.append(dict(row, kind=kind + "_difference", opponent=opponent,
                    experiment=f"{experiment}-exp{earlier}", match_mean=row["match_mean"]-other["match_mean"],
                    # Paired seed differences, not invented match-level CIs.
                    match_std=None, match_se=None, match_ci95_low=None, match_ci95_high=None))
    return differences
