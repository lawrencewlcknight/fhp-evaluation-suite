"""Three-seed Experiment 9 exact-flop/LBR production evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from ..statistics import sample_summary
from .runner import identity, write_json


EXPERIMENT = "exp5_ucv_exp9_br_production_3seed"
PROTOCOL_VERSION = 1
SOURCE_RUN = "exp9-cache24-20261001-132550"
TRAINING_SEEDS = (0, 1, 2)
SHARDS_PER_SEED = 4
PAIRS_PER_SHARD = 500
TOTAL_PAIRS_PER_SEED = SHARDS_PER_SEED * PAIRS_PER_SHARD
ROLLOUTS = 4096
SEARCH_SEED = 731
THREADS = 2
# These fresh streams are shared across training seeds shard-for-shard. They are
# deliberately distinct from the pilot's 2026100402 stream.
SHARD_EVALUATION_SEEDS = (202610050101, 202610050102, 202610050103, 202610050104)
CHECKPOINTS = {
    0: {
        "relative": ("workers/task_000_cached_parallel_structured_ucv_escher_seed_0/"
                     "checkpoints/cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl"),
        "sha256": "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449",
    },
    1: {
        "relative": ("workers/task_001_cached_parallel_structured_ucv_escher_seed_1/"
                     "checkpoints/cached_parallel_structured_ucv_escher_seed_1_time_24h.pkl"),
        "sha256": "ef5f0db716bd2652cd06817ff0b4d4b6e5d1307c5063b982c69b768d1b3b7caa",
    },
    2: {
        "relative": ("workers/task_002_cached_parallel_structured_ucv_escher_seed_2/"
                     "checkpoints/cached_parallel_structured_ucv_escher_seed_2_time_24h.pkl"),
        "sha256": "d678f805ad9c9665545ba91164a9129c5bb60fab09c90b8478cbcaa35b9def7b",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def task_schedule():
    return tuple((seed, shard) for seed in TRAINING_SEEDS for shard in range(SHARDS_PER_SEED))


def task_name(task_index):
    seed, shard = task_schedule()[task_index]
    return f"task_{task_index:03d}_seed_{seed}_shard_{shard}"


def contract(*, smoke=False):
    return {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "source_run": SOURCE_RUN,
        "training_seeds": list(TRAINING_SEEDS),
        "shards_per_seed": SHARDS_PER_SEED,
        "pairs_per_shard": 2 if smoke else PAIRS_PER_SHARD,
        "evaluation_seeds": list(SHARD_EVALUATION_SEEDS),
        "preflop_rollouts": 8 if smoke else ROLLOUTS,
        "search_seed": SEARCH_SEED,
        "threads": THREADS,
        "smoke": bool(smoke),
        "common_random_numbers": "same shard deal/action stream for all training seeds and both responders",
        "primary_inference_unit": "training_seed",
        "primary_estimand": "unweighted mean seed-level paired exact-flop-minus-LBR response payoff",
        "exact_full_game": False,
    }


def _finite_summary(values):
    result = sample_summary(np.asarray(values, dtype=float))
    # Production summaries always have n >= 2, so JSON never needs NaN.
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite production summary")
    return {key: float(value) if key != "n" else int(value) for key, value in result.items()}


def run_shard(task_index, checkpoint, repo_root, output_dir, source_bundle_sha256, *, smoke=False):
    from ..loaders import sha256_file
    from .worker import execute

    schedule = task_schedule()
    if not 0 <= task_index < len(schedule):
        raise ValueError(f"task-index must be in [0, {len(schedule) - 1}]")
    if (len(source_bundle_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_bundle_sha256)):
        raise ValueError("A 64-character source bundle SHA-256 is required")
    training_seed, shard = schedule[task_index]
    checkpoint, repo_root, output = Path(checkpoint).resolve(), Path(repo_root).resolve(), Path(output_dir).resolve()
    expected = CHECKPOINTS[training_seed]
    output.mkdir(parents=True, exist_ok=True)
    success_path = output / "SUCCESS.json"
    shard_contract = contract(smoke=smoke)
    shard_identity = {
        "task_index": task_index,
        "task_name": task_name(task_index),
        "training_seed": training_seed,
        "shard_index": shard,
        "evaluation_seed": SHARD_EVALUATION_SEEDS[shard],
        "checkpoint_relative": expected["relative"],
        "checkpoint_sha256": expected["sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "contract": shard_contract,
    }
    if success_path.is_file():
        previous = json.loads(success_path.read_text())
        if previous.get("identity") != shard_identity:
            raise ValueError("Existing successful shard has a different immutable identity")
        result_path = output / "result.json"
        if (previous.get("completed_pairs") != shard_contract["pairs_per_shard"]
                or not result_path.is_file()):
            raise ValueError("Successful shard marker is missing its complete result")
        previous_result = json.loads(result_path.read_text())
        if (previous_result.get("task_index") != task_index
                or previous_result.get("training_seed") != training_seed
                or previous_result.get("shard_index") != shard
                or previous_result.get("num_deal_pairs") != shard_contract["pairs_per_shard"]
                or previous_result.get("source_bundle_sha256") != source_bundle_sha256):
            raise ValueError("Successful shard result does not match its marker")
        return previous
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for training seed {training_seed}")
    manifest_path = output / "manifest.json"
    if (output / "partial.json").is_file() and not manifest_path.is_file():
        raise ValueError("Partial shard is missing its immutable manifest")
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest.get("identity") != shard_identity:
            raise ValueError("Existing partial shard has a different immutable identity")
    manifest = {
        "identity": shard_identity,
        "status": "running",
        "started_utc": (previous_manifest.get("started_utc", utc_now())
                        if manifest_path.is_file() else utc_now()),
        "resumed_utc": utc_now() if manifest_path.is_file() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    write_json(manifest_path, manifest)
    spec = {
        "mode": "compare",
        "family": "ucv",
        "checkpoint": str(checkpoint),
        "repo_root": str(repo_root),
        "batch_size": 1024,
        "model_batch_size": 32,
        "seed": SHARD_EVALUATION_SEEDS[shard],
        "search_seed": SEARCH_SEED,
        "threads": THREADS,
        "deals": 2 if smoke else PAIRS_PER_SHARD,
        "rollouts": 8 if smoke else ROLLOUTS,
        "resume_partial": True,
    }
    started = time.monotonic()
    try:
        evaluation_identity = identity(spec)
        write_json(output / "spec.json", spec)
        result = execute(spec, progress_path=output / "partial.json")
        result.update(experiment=EXPERIMENT, protocol_version=PROTOCOL_VERSION,
                      training_seed=training_seed, shard_index=shard, task_index=task_index,
                      source_bundle_sha256=source_bundle_sha256,
                      evaluation_identity=evaluation_identity)
        write_json(output / "result.json", result)
        manifest.update(status="succeeded", completed_utc=utc_now(),
                        elapsed_seconds=time.monotonic() - started,
                        completed_pairs=result["num_deal_pairs"])
        write_json(manifest_path, manifest)
        success = {"identity": shard_identity, "status": "succeeded",
                   "completed_utc": manifest["completed_utc"],
                   "completed_pairs": result["num_deal_pairs"]}
        write_json(success_path, success)
        return success
    except BaseException as error:
        manifest.update(status="failed", completed_utc=utc_now(),
                        elapsed_seconds=time.monotonic() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(manifest_path, manifest)
        raise


def _write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(workers_root, output_dir, source_bundle_sha256, *, smoke=False):
    workers_root, output = Path(workers_root).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty aggregate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expected_contract = contract(smoke=smoke)
    per_seed = {seed: {"lbr": [], "full_flop": []} for seed in TRAINING_SEEDS}
    worker_rows = []
    for task_index, (seed, shard) in enumerate(task_schedule()):
        directory = workers_root / task_name(task_index)
        success = json.loads((directory / "SUCCESS.json").read_text())
        expected_identity = {
            "task_index": task_index,
            "task_name": task_name(task_index),
            "training_seed": seed,
            "shard_index": shard,
            "evaluation_seed": SHARD_EVALUATION_SEEDS[shard],
            "checkpoint_relative": CHECKPOINTS[seed]["relative"],
            "checkpoint_sha256": CHECKPOINTS[seed]["sha256"],
            "source_bundle_sha256": source_bundle_sha256,
            "contract": expected_contract,
        }
        if success.get("identity") != expected_identity or success.get("status") != "succeeded":
            raise ValueError(f"Shard identity/status mismatch: {directory}")
        result = json.loads((directory / "result.json").read_text())
        if (result.get("training_seed"), result.get("shard_index"), result.get("task_index")) != (seed, shard, task_index):
            raise ValueError(f"Shard result metadata mismatch: {directory}")
        if (result.get("protocol_version") != PROTOCOL_VERSION
                or result.get("source_bundle_sha256") != source_bundle_sha256
                or result.get("evaluation_seed") != SHARD_EVALUATION_SEEDS[shard]
                or result.get("search_seed") != SEARCH_SEED
                or result.get("preflop_rollouts") != expected_contract["preflop_rollouts"]
                or result.get("evaluation_identity", {}).get("checkpoint_sha256") != CHECKPOINTS[seed]["sha256"]):
            raise ValueError(f"Shard result protocol/source mismatch: {directory}")
        values = result.get("pair_payoffs_mbb", {})
        seats = result.get("seat_payoffs_chips", {})
        requested = expected_contract["pairs_per_shard"]
        if any(len(values.get(label, ())) != requested for label in ("lbr", "full_flop")):
            raise ValueError(f"Incomplete raw pair scores: {directory}")
        if any(len(seats.get(label, ())) != 2
               or any(len(row) != requested for row in seats[label])
               for label in ("lbr", "full_flop")):
            raise ValueError(f"Incomplete raw seat scores: {directory}")
        if not all(np.isfinite(values[label]).all() for label in ("lbr", "full_flop")):
            raise ValueError(f"Non-finite raw pair scores: {directory}")
        per_seed[seed]["lbr"].extend(values["lbr"])
        per_seed[seed]["full_flop"].extend(values["full_flop"])
        worker_rows.append({"task_index": task_index, "training_seed": seed, "shard_index": shard,
                            "evaluation_seed": SHARD_EVALUATION_SEEDS[shard],
                            "pairs": requested, "checkpoint_sha256": CHECKPOINTS[seed]["sha256"]})
    seed_rows, seed_deltas = [], []
    pooled_lbr, pooled_flop, pooled_delta = [], [], []
    for seed in TRAINING_SEEDS:
        lbr = np.asarray(per_seed[seed]["lbr"], dtype=float)
        flop = np.asarray(per_seed[seed]["full_flop"], dtype=float)
        delta = flop - lbr
        lbr_summary, flop_summary, delta_summary = map(_finite_summary, (lbr, flop, delta))
        seed_deltas.append(delta_summary["mean"])
        pooled_lbr.extend(lbr.tolist())
        pooled_flop.extend(flop.tolist())
        pooled_delta.extend(delta.tolist())
        seed_rows.append({
            "training_seed": seed, "pairs": len(delta),
            "lbr_mean_mbb_per_hand": lbr_summary["mean"],
            "lbr_ci95_low": lbr_summary["ci95_low"], "lbr_ci95_high": lbr_summary["ci95_high"],
            "exact_flop_mean_mbb_per_hand": flop_summary["mean"],
            "exact_flop_ci95_low": flop_summary["ci95_low"], "exact_flop_ci95_high": flop_summary["ci95_high"],
            "paired_delta_mean_mbb_per_hand": delta_summary["mean"],
            "paired_delta_ci95_low": delta_summary["ci95_low"],
            "paired_delta_ci95_high": delta_summary["ci95_high"],
        })
    primary = _finite_summary(seed_deltas)
    pooled = {
        "lbr": _finite_summary(pooled_lbr),
        "full_flop": _finite_summary(pooled_flop),
        "paired_improvement": _finite_summary(pooled_delta),
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "completed_utc": utc_now(),
        "contract": expected_contract,
        "source_bundle_sha256": source_bundle_sha256,
        "workers": worker_rows,
        "seed_results": seed_rows,
        "primary_training_seed_inference": {
            **primary,
            "units": "mbb_per_hand",
            "quantity": "paired_exact_flop_minus_lbr_response_payoff",
            "ci_method": "Student_t_over_three_training_seed_means",
        },
        "secondary_pair_level_conditional_summary": {
            **pooled,
            "warning": "Conditional Monte Carlo precision only; duplicate pairs are not independent training seeds.",
        },
        "interpretation": ("Both response scores are whole-game exploitability lower-bound estimates. "
                           "The flop continuation is exact but preflop remains LBR; this is not exact exploitability."),
    }
    write_json(output / "aggregate_summary.json", summary)
    _write_csv(output / "seed_summary.csv", seed_rows)
    primary_mean = primary["mean"]
    report = "# Experiment 5 production result\n\n"
    report += f"Status: complete. Evaluated {len(pooled_delta):,} fresh duplicate pairs ({2 * len(pooled_delta):,} hands) over three frozen training seeds.\n\n"
    report += "| Training seed | Pairs | LBR mean | Exact-flop mean | Paired delta | 95% pair-level CI for delta |\n|---:|---:|---:|---:|---:|---:|\n"
    for row in seed_rows:
        report += (f"| {row['training_seed']} | {row['pairs']:,} | {row['lbr_mean_mbb_per_hand']:.1f} | "
                   f"{row['exact_flop_mean_mbb_per_hand']:.1f} | {row['paired_delta_mean_mbb_per_hand']:.1f} | "
                   f"[{row['paired_delta_ci95_low']:.1f}, {row['paired_delta_ci95_high']:.1f}] |\n")
    report += (f"\nPrimary result: unweighted mean seed-level paired delta **{primary_mean:.1f} mbb/hand** "
               f"(95% Student-t interval across 3 seeds [{primary['ci95_low']:.1f}, {primary['ci95_high']:.1f}]).\n\n"
               "The pooled pair-level interval is secondary and conditional on these three trained policies. "
               "Neither response is exact full-game exploitability because preflop remains approximate LBR.\n")
    (output / "report.md").write_text(report)
    write_json(output / "SUCCESS.json", {"experiment": EXPERIMENT, "status": "complete",
                                          "completed_utc": summary["completed_utc"],
                                          "source_bundle_sha256": source_bundle_sha256})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task-index", required=True, type=int)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--repo-root", required=True)
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--source-bundle-sha256", required=True)
    worker.add_argument("--smoke", action="store_true")
    reducer = sub.add_parser("aggregate")
    reducer.add_argument("--workers-root", required=True)
    reducer.add_argument("--output-dir", required=True)
    reducer.add_argument("--source-bundle-sha256", required=True)
    reducer.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        run_shard(args.task_index, args.checkpoint, args.repo_root, args.output_dir,
                  args.source_bundle_sha256, smoke=args.smoke)
    else:
        aggregate(args.workers_root, args.output_dir, args.source_bundle_sha256, smoke=args.smoke)


if __name__ == "__main__":
    main()
