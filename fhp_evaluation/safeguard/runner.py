"""Experiment 34: resumable, source-pinned paired evaluation without retraining."""
import argparse
from collections import defaultdict
from concurrent.futures import as_completed
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from ..best_response.adapters import PolicyAdapter, load_target
from ..best_response.flop import FullFlopResponsePolicy
from ..best_response.runner import source_hash, write_json
from ..cohort import digest
from ..game import load_fhp_game
from ..lbr import LBRConfig, LocalBestResponsePolicy
from ..loaders import sha256_file
from ..resolving.runner import worker_pool, initialise
from ..rule_agents import published_rule_agents
from ..strategic_audit.workflow import exclusive
from .experiment import config, count, is_response, profiling_tasks, tasks
from .matches import COLUMNS, paired_matches
from .policy import ExactCallSafeguard
from .report import aggregate, describe, write_report
from .validation import validate_native, validate_verifier


@lru_cache(maxsize=15)
def load(label, input_root, native_root, frozen):
    spec = json.loads(frozen)
    path = Path(input_root) / f"{label}.pkl"
    if sha256_file(path) != spec["checkpoint_registry"][label]["sha256"]:
        raise ValueError(f"Source checkpoint hash mismatch: {label}")
    game = load_fhp_game()
    return game, load_target(game, path, family="ucv", repo_root=native_root,
                             batch_size=spec["batch_size"])


def score_task(task, input_root, native_root, spec, directory, fingerprint):
    started = time.perf_counter()
    frozen = json.dumps(spec, sort_keys=True)
    game, base = load(f"exp16_seed{task['seed']}_48h", input_root, native_root, frozen)
    name = task["opponent"]
    response_guard = None
    if is_response(name):
        response_guard = ExactCallSafeguard(game, base)
        target = response_guard if name.endswith("_guarded") else base
        cfg = LBRConfig(seed=spec["response_seed"] + task["seed"],
                        preflop_rollout_samples=spec["rollout_samples"])
        native = (FullFlopResponsePolicy(game, target, config=cfg) if name.startswith("full_flop_")
                  else LocalBestResponsePolicy(game, target, config=cfg))
        opponent = PolicyAdapter(native)
    elif name.startswith("exp"):
        _, opponent = load(name, input_root, native_root, frozen)
    else:
        opponent = PolicyAdapter(published_rule_agents(game)[name])
    rows, timing = paired_matches(game, base, opponent, task, spec)
    metadata = dict(task=task, identity=fingerprint, columns=COLUMNS,
                    seconds=time.perf_counter()-started, deployment_verifier=timing,
                    response_verifier=response_guard.verifier.diagnostics() if response_guard else None,
                    policy=base.metadata)
    path = Path(directory) / task["id"]
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "pairs.tmp.npz"
    np.savez_compressed(temporary, rows=rows)
    temporary.replace(path / "pairs.npz")
    write_json(path / "result.json", metadata)
    write_json(path / "SUCCESS.json", dict(identity=fingerprint,
        result_sha256=sha256_file(path / "result.json"), pairs_sha256=sha256_file(path / "pairs.npz")))
    return dict(id=task["id"], seconds=metadata["seconds"], count=task["count"],
                seed=task["seed"], opponent=name)


def completed(directory, task, fingerprint):
    path = Path(directory) / task["id"]
    if not (path / "SUCCESS.json").exists():
        return False
    marker = json.loads((path / "SUCCESS.json").read_text())
    row = json.loads((path / "result.json").read_text())
    if (marker["identity"] != fingerprint or row["identity"] != fingerprint or row["task"] != task
            or marker["result_sha256"] != sha256_file(path / "result.json")
            or marker["pairs_sha256"] != sha256_file(path / "pairs.npz")):
        raise ValueError(f"Changed completed shard: {path}")
    return True


def run_tasks(schedule, args, spec, folder, fingerprint, deadline):
    pending = [r for r in schedule if not completed(folder, r, fingerprint)]
    if not pending:
        return
    with worker_pool(spec) as pool:
        futures = [pool.submit(score_task, row, str(args.input_root), str(args.native_root),
                               spec, str(folder), fingerprint) for row in pending]
        for future in as_completed(futures, timeout=max(1., deadline-time.monotonic())):
            row = future.result()
            write_json(args.output / "status.json", dict(status="running", last_completed=row))
            print(json.dumps(dict(event="shard_completed", **row)), flush=True)


def profile(args, spec, fingerprint, deadline):
    started = time.monotonic()
    schedule = profiling_tasks(spec)
    # These are real production shards, not disposable timing-only games. The
    # main task pass and subsequent recoveries verify and skip them as usual.
    folder = args.output / "task_results"
    run_tasks(schedule, args, spec, folder, fingerprint, deadline)
    records = [json.loads((folder / row["id"] / "result.json").read_text()) for row in schedule]
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"]["seed"], record["task"]["opponent"])].append(record)
    matchups = []
    for (seed, opponent), rows in grouped.items():
        pairs = sum(r["task"]["count"] for r in rows)
        seconds = sum(r["seconds"] for r in rows)
        rates = [r["seconds"] / r["task"]["count"] for r in rows]
        matchups.append(dict(seed=seed, opponent=opponent, measured_shards=len(rows), measured_pairs=pairs,
            measured_worker_seconds=seconds, worker_seconds_per_pair=seconds/pairs,
            min_shard_seconds_per_pair=min(rates), max_shard_seconds_per_pair=max(rates),
            projected_worker_seconds=seconds/pairs*count(spec, seed, opponent)))
    unpadded = sum(r["projected_worker_seconds"] for r in matchups) / spec["processes"] / 3600
    projected = spec["projection_margin"] * unpadded
    within_target = projected <= spec["gameplay_planning_hours"]
    result = dict(projected_gameplay_hours=projected, unpadded_gameplay_hours=unpadded,
                  safety_multiplier=spec["projection_margin"], planning_gameplay_hours=spec["gameplay_planning_hours"],
                  within_planning_target=within_target, advisory_only=True, decision="continue",
                  measured_pairs_retained=sum(r["measured_pairs"] for r in matchups),
                  profiling_invocation_seconds=time.monotonic()-started,
                  session_hours_remaining=max(0., deadline-time.monotonic())/3600,
                  note="Advisory forecast from repeated production shards, retained in the fixed final sample. "
                       "Includes response probes and worker model loading; excludes strategic audit/setup. "
                       "Repeat ranges are diagnostics, not confidence intervals. No sample-size reduction or "
                       "forecast-based abort; actual session/Batch limits still apply.",
                  matchups=matchups, records=records)
    write_json(args.output / "analysis/profile.json", result)
    print(json.dumps(dict(event="runtime_forecast", decision="continue", advisory_only=True,
        level="info" if within_target else "warning", projected_gameplay_hours=projected,
        unpadded_gameplay_hours=unpadded, planning_gameplay_hours=spec["gameplay_planning_hours"],
        measured_pairs_retained=result["measured_pairs_retained"],
        message="Continuing the unchanged evaluation sample within the existing wall-clock cap.")), flush=True)
    return result


def strategic_spec(spec, input_root, native_root):
    from ..strategic_audit import PROTOCOL
    from .experiment import RULES
    policies = {r: dict(family="rule", name=r) for r in RULES}
    candidates = []
    for label, row in spec["checkpoint_registry"].items():
        if row["cohort"] != "exp9" and row["hours"] != 48:
            continue
        item = dict(family="ucv", checkpoint=str(Path(input_root)/f"{label}.pkl"),
                    sha256=row["sha256"], repo_root=str(native_root), hours=row["hours"],
                    training_seed=row["training_seed"], cohort="unmodified", require_suit_invariance=True)
        policies[label] = item
        if row["cohort"] == "exp16":
            guarded = label + "_guarded"
            policies[guarded] = dict(item, cohort="guarded", deployment="exact_terminal_call_v1")
            candidates += [label, guarded]
    return dict(protocol=PROTOCOL, policies=policies, candidates=candidates,
                opponents=list(RULES)+[f"exp9_seed{s}_24h" for s in spec["seeds"]],
                reference_policy="exp9_seed0_24h", range_tilts=[0, -2, 2])


def run_strategic(args, spec, deadline):
    from ..strategic_audit.exp16 import LIBRARY_SEED
    from ..strategic_audit.library import build_library
    from ..strategic_audit.workflow import run
    root = args.output / "analysis/strategic"
    library = build_library(LIBRARY_SEED, pilot=spec["mode"] == "smoke",
                            reference_deals=8 if spec["mode"] == "smoke" else 64)
    remaining = deadline-time.monotonic()
    if remaining < 60:
        raise TimeoutError("Session budget exhausted before strategic audit; recover this run")
    path = run(strategic_spec(spec, args.input_root, args.native_root), library, root,
               include_reference=True, resume=(root / "manifest.json").exists(), threads=4,
               max_seconds=remaining, max_rss_mb=48000)
    result = json.loads(path.read_text())
    paired = {}
    for label, row in result["comparisons"].items():
        if "48h_guarded__" not in label:
            continue
        baseline = result["comparisons"][label.replace("48h_guarded__", "48h__")]
        paired[label] = dict(guarded_root_gap_bb=row["mean_root_response_gap_bb"],
                            unmodified_root_gap_bb=baseline["mean_root_response_gap_bb"],
                            difference_bb=row["mean_root_response_gap_bb"]-baseline["mean_root_response_gap_bb"])
    write_json(args.output / "analysis/strategic_comparison.json", dict(
        comparisons=paired, scope="Conditional audited positions, NOT whole-game exploitability",
        assessment_evaluated=False, diagnostic_library="Existing development library; includes known failures"))


def aggregate_tasks(args, spec, fingerprint):
    grouped = defaultdict(list)
    timing = {}
    for row in tasks(spec):
        if not completed(args.output / "task_results", row, fingerprint):
            raise ValueError(f"Missing complete shard: {row['id']}")
        grouped[(row["seed"], row["opponent"])].append(row)
        details = json.loads((args.output / "task_results" / row["id"] / "result.json").read_text())
        for scope in ("deployment_verifier", "response_verifier"):
            stats = details.get(scope)
            if stats is None:
                continue
            total = timing.setdefault(scope, dict(seconds=0., queries=0, board_misses=0, maximum_seconds=0.,
                peak_board_array_bytes=0, latency_counts={k: np.zeros(66, dtype=np.int64) for k in stats["latency_counts"]},
                latency_bin_upper_seconds=stats["latency_bin_upper_seconds"]))
            for key in ("seconds", "queries", "board_misses"):
                total[key] += stats[key]
            total["maximum_seconds"] = max(total["maximum_seconds"], stats["maximum_seconds"])
            total["peak_board_array_bytes"] = max(total["peak_board_array_bytes"], stats["board_array_bytes"])
            for key, values in stats["latency_counts"].items():
                total["latency_counts"][key] += values
    records = {}
    for (seed, opponent), group in grouped.items():
        arrays = []
        for row in sorted(group, key=lambda r: r["start"]):
            with np.load(args.output / "task_results" / row["id"] / "pairs.npz", allow_pickle=False) as data:
                if data["rows"].shape != (row["count"], len(COLUMNS)):
                    raise ValueError("Invalid shard dimensions")
                arrays.append(data["rows"])
        combined = np.concatenate(arrays)
        if len(combined) != count(spec, seed, opponent):
            raise ValueError("Incomplete fixed sample")
        records[f"{seed}/{opponent}"] = describe(combined)
    result = aggregate(records, spec)
    for total in timing.values():
        total["mean_extra_seconds_per_verifier_query"] = total["seconds"]/max(1,total["queries"])
        quantiles = {}
        for kind, histogram in total["latency_counts"].items():
            quantiles[kind] = {}
            for label, q in (("p50", .5), ("p95", .95), ("p99", .99)):
                index = int(np.searchsorted(np.cumsum(histogram), q*histogram.sum()))
                quantiles[kind][label+"_upper_seconds"] = total["latency_bin_upper_seconds"][index] if histogram.sum() else None
        total["quantiles_from_histogram"] = quantiles
        total["latency_counts"] = {k: v.tolist() for k,v in total["latency_counts"].items()}
    write_json(args.output / "analysis/latency.json", dict(verifier_overhead=timing,
        note="Incremental verifier time, not end-to-end network inference. Cold-board and cached queries separated; "
             "query totals may include repeated/hypothetical states. Natural activation rates are in summary.json."))
    write_json(args.output / "analysis/summary.json", result)
    write_report(args.output / "analysis/report.md", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    for key in ("input_root", "native_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    spec = config(args.mode)
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise RuntimeError("Production evaluation requires OpenSpiel 1.6.3")
    provenance = dict(spec=spec, source_sha256=source_hash(Path(__file__).resolve().parents[1]),
        native_packages={name: source_hash(args.native_root/name) for name in ("fhp_escher", "vr_deep_cfr")},
        runtime={name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "open_spiel")})
    fingerprint = digest(provenance)
    deadline = time.monotonic()+spec["session_seconds"]
    initialise(spec["torch_threads"])
    with exclusive(args.output):
        manifest = args.output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError("Recovery requires identical source, configuration and runtime")
        write_json(manifest, provenance)
        (args.output / "analysis").mkdir(exist_ok=True)
        write_json(args.output / "analysis/deployment_manifest.json", dict(
            wrapper="fhp_evaluation.safeguard.policy.ExactCallSafeguard",
            source_identity=fingerprint, base_checkpoints={name: row for name, row in spec["checkpoint_registry"].items()
                                                        if row["cohort"] == "exp16" and row["hours"] == 48},
            neural_weights_changed=False, safeguard="exact_terminal_call_v1",
            requirement="Load the trusted native checkpoint, verify its SHA-256, then wrap its behavioural adapter. "
                        "This small manifest plus the unchanged playable checkpoint defines the deployed model."))
        try:
            frozen = json.dumps(spec, sort_keys=True)
            native = {}
            for label in spec["checkpoint_registry"]:
                game, adapter = load(label, str(args.input_root), str(args.native_root), frozen)
                if label.endswith("_48h"):
                    native[label] = validate_native(game, adapter)
            write_json(args.output / "analysis/validation.json", dict(verifier=validate_verifier(), native=native))
            if spec["mode"] == "full":
                profile(args, spec, fingerprint, deadline)
            run_tasks(list(tasks(spec)), args, spec, args.output / "task_results", fingerprint, deadline)
            aggregate_tasks(args, spec, fingerprint)
            run_strategic(args, spec, deadline)
            write_json(args.output / "analysis/SUCCESS.json", dict(
                status="succeeded", identity=fingerprint, mode=spec["mode"],
                components=["paired_head_to_head", "rules", "temporal", "fixed_neural_panel",
                            "lbr_cross_probes", "full_flop_cross_probes", "strategic_audit"],
                trained_attackers=False, weights_changed=False))
            write_json(args.output / "status.json", dict(status="succeeded", identity=fingerprint))
        except BaseException as error:
            write_json(args.output / "status.json", dict(status="incomplete", identity=fingerprint,
                                                        error=repr(error)))
            raise


if __name__ == "__main__":
    main()
