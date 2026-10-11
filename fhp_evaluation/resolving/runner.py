"""Durable paired gameplay; 4 processes x 4 Torch threads, one 16-vCPU VM."""
import argparse
import csv
import io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

from ..best_response.adapters import load_target
from ..best_response.flop import make_state, validate_game
from ..cohort import digest, sha256, write_json
from ..game import load_fhp_game, MILLI_BIG_BLINDS_PER_CHIP
from ..rule_agents import published_rule_agents
from ..statistics import sample_summary
from ..strategic_audit.tables import preflop_nodes
from .experiment import config, source_hash, stream_seed, tasks, required_sources
from .policy import ResolvingPolicy, play_hand
from .validation import algebra_checks, native_checks


def initialise(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


@contextmanager
def worker_pool(spec):
    # The native checkout also vendors an older fhp_evaluation package. Keep
    # this suite first when spawn reconstructs sys.path, even after a caller
    # loaded a native policy in the parent process.
    evaluator_root = str(Path(__file__).resolve().parents[2])
    sys.path[:] = [evaluator_root, *(p for p in sys.path if p != evaluator_root)]
    pool = ProcessPoolExecutor(max_workers=spec["processes"], mp_context=mp.get_context("spawn"),
                               initializer=initialise, initargs=(spec["torch_threads"],))
    try:
        yield pool
    except BaseException:
        # Python 3.11 lacks terminate_workers(). Do not finish hundreds of
        # already-queued shards after a feasibility gate or a task has failed.
        for process in (pool._processes or {}).values():
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


@lru_cache(maxsize=9)
def load(label, input_root, native_root, spec_json):
    spec = json.loads(spec_json)
    path = Path(input_root) / f"{label}.pkl"
    if sha256(path) != spec["checkpoint_registry"][label]["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {label}")
    game = load_fhp_game()
    return game, load_target(game, path, family="ucv", repo_root=native_root, batch_size=spec["batch_size"])


def timing_summary(traces):
    if not traces:
        return dict(attempts=0, acceptance_rate=0., overrun_rate=0., maximum_seconds=0., reasons={})
    values = np.asarray([t["total_seconds"] for t in traces])
    decisions = sum(t.get("flop_decisions", 0) for t in traces)
    stages = sorted({name for t in traces for name in t.get("stage_seconds", {})})
    return dict(attempts=len(traces), acceptance_rate=float(np.mean([t["accepted"] for t in traces])),
                preparation_completed=sum(t.get("preparation_completed", False) for t in traces),
                mean_preparation_seconds=float(np.mean([t.get("preparation_seconds", 0.) for t in traces])),
                mean_stage_seconds={name: float(np.mean([t.get("stage_seconds", {}).get(name, 0.) for t in traces])) for name in stages},
                flop_decisions=decisions,
                changed_decision_rate=sum(t.get("changed_flop_decisions", 0) for t in traces) / decisions if decisions else None,
                mean_decision_total_variation=sum(t.get("policy_total_variation_sum", 0.) for t in traces) / decisions if decisions else None,
                overrun_rate=float(np.mean(values > np.asarray([t["budget_seconds"] for t in traces]))),
                maximum_seconds=float(values.max()), median_seconds=float(np.median(values)),
                p95_seconds=float(np.quantile(values, .95)), p99_seconds=float(np.quantile(values, .99)),
                mean_iterations=float(np.mean([t["iterations"] for t in traces])),
                reasons=dict(Counter(t["reason"] for t in traces)))


def enforce_gate(summary, spec):
    if (summary["attempts"] == 0 or summary["acceptance_rate"] < spec["minimum_acceptance_rate"]
            or summary["overrun_rate"] > spec["maximum_overrun_rate"]
            or summary["maximum_seconds"] > spec["maximum_latency_seconds"]):
        raise RuntimeError("500ms feasibility gate failed; inspect timing/acceptance diagnostics. "
                           "Do not interpret mostly-fallback play as a negative search result or raise the budget silently.")


def preflight_task(row, input_root, native_root, spec):
    seed, index = row
    game, adapter = load(f"exp16_seed{seed}_48h", input_root, native_root, json.dumps(spec, sort_keys=True))
    # Separate positions from gameplay; outcomes are never used for tuning.
    rng = np.random.default_rng(spec["preflight_seed"] + seed * 1000 + index)
    board = tuple(sorted(map(int, rng.choice(validate_game(game), 3, replace=False))))
    _, entries = preflop_nodes(game)
    pre = sorted(entries)[index % len(entries)]
    hero = index % 2
    own = tuple(c for c in validate_game(game) if c not in board)[:2]
    state = make_state(game, validate_game(game), own, hero, board=board, preflop=pre)
    wrapper = ResolvingPolicy(game, adapter, seconds=spec["search_seconds"], iterations=spec["max_iterations"])
    wrapper.observe_flop_entry(state, hero)
    return dict(seed=seed, index=index, board=board, pre=pre, hero=hero, **wrapper.stats)


def parity_task(seed, input_root, native_root, spec):
    game, adapter = load(f"exp16_seed{seed}_48h", input_root, native_root, json.dumps(spec, sort_keys=True))
    return dict(seed=seed, **native_checks(game, adapter))


def profile_task(row, input_root, native_root, spec):
    from .profiling import compare_preparation
    seed, index = row
    game, adapter = load(f"exp16_seed{seed}_48h", input_root, native_root, json.dumps(spec, sort_keys=True))
    return compare_preparation(game, adapter, seed, index, spec)


def deal_stream(spec, seed, opponent):
    rng = np.random.default_rng(stream_seed(spec, seed, opponent))
    n = spec["pairs_per_match"]
    return rng.integers(0, 2**63 - 1, size=n, dtype=np.int64), rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)


def score_task(task, input_root, native_root, spec):
    started = time.perf_counter()
    text = json.dumps(spec, sort_keys=True)
    seed, opponent_name = task["seed"], task["opponent"]
    game, blueprint = load(f"exp16_seed{seed}_48h", input_root, native_root, text)
    search = ResolvingPolicy(game, blueprint, seconds=spec["search_seconds"], iterations=spec["max_iterations"])
    if opponent_name == "blueprint":
        opponent = blueprint
    elif opponent_name in spec["rules"]:
        opponent = published_rule_agents(game)[opponent_name]
    else:
        hour = 48 if opponent_name == "exp8" else 24
        _, opponent = load(f"{opponent_name}_seed{seed}_{hour}h", input_root, native_root, text)
    chance, action = deal_stream(spec, seed, opponent_name)
    samples, traces, baseline_seconds, search_seconds = [], [], 0., 0.
    for index in range(task["start"], task["start"] + task["count"]):
        scores = {"search": [], "baseline": []}
        # Alternate execution order to avoid systematic warmup/frequency bias.
        modes = ("search", "baseline") if index % 2 else ("baseline", "search")
        for mode in modes:
            if mode == "baseline" and opponent_name == "blueprint":
                continue  # Direct search-v-blueprint EV is the primary estimand.
            candidate = search if mode == "search" else blueprint
            tick = time.perf_counter()
            for seat in (0, 1):
                policies = (candidate, opponent) if seat == 0 else (opponent, candidate)
                try:
                    returns, measurements = play_hand(game, policies, chance_seed=chance[index], action_seed=action[index])
                except Exception as exc:
                    raise RuntimeError(f"{task['id']}: pair {index}, mode {mode}, seat {seat}") from exc
                scores[mode].append(returns[seat] * MILLI_BIG_BLINDS_PER_CHIP)
                traces.extend(dict(pair_index=index, seat=seat, **m) for m in measurements)
            if mode == "search":
                search_seconds += time.perf_counter() - tick
            else:
                baseline_seconds += time.perf_counter() - tick
        samples.append(dict(pair_index=index, search_seats=scores["search"],
                            baseline_seats=scores["baseline"] or None))
    return dict(task=task, samples=samples, traces=traces, timings=timing_summary(traces),
                search_gameplay_seconds=search_seconds, baseline_gameplay_seconds=baseline_seconds,
                elapsed_seconds=time.perf_counter() - started)


def sign_shard(result, identity):
    return dict(identity=identity, payload=result, payload_sha256=digest(result))


def read_shard(path, task, identity):
    row = json.loads(Path(path).read_text())
    if (row.get("identity") != identity or digest(row["payload"]) != row.get("payload_sha256")
            or row["payload"]["task"] != task):
        raise ValueError(f"Incompatible or corrupted completed shard: {path}")
    samples = row["payload"]["samples"]
    if [s["pair_index"] for s in samples] != list(range(task["start"], task["start"] + task["count"])):
        raise ValueError(f"Missing or duplicate deals: {path}")
    return row["payload"]


def summarise(results, spec):
    matches, seed_timings = [], {}
    for seed in spec["training_seeds"]:
        seed_rows = [r for r in results if r["task"]["seed"] == seed]
        seed_timings[str(seed)] = timing_summary([t for r in seed_rows for t in r["traces"]])
        for opponent in ["blueprint", *spec["rules"], "exp8", "exp9"]:
            rows = [r for r in seed_rows if r["task"]["opponent"] == opponent]
            samples = sorted([s for r in rows for s in r["samples"]], key=lambda x: x["pair_index"])
            if [s["pair_index"] for s in samples] != list(range(spec["pairs_per_match"])):
                raise ValueError("Incomplete/duplicated matchup; refusing final summary")
            searched = np.asarray([s["search_seats"] for s in samples])
            row = dict(seed=seed, opponent=opponent, search=sample_summary(searched.mean(1)),
                       search_seat_means=searched.mean(0).tolist(),
                       search_gameplay_seconds=sum(r["search_gameplay_seconds"] for r in rows),
                       baseline_gameplay_seconds=sum(r["baseline_gameplay_seconds"] for r in rows))
            if opponent != "blueprint":
                baseline = np.asarray([s["baseline_seats"] for s in samples])
                row.update(baseline=sample_summary(baseline.mean(1)),
                           paired_uplift=sample_summary((searched - baseline).mean(1)))
            matches.append(row)
    primary = [r["search"]["mean"] for r in matches if r["opponent"] == "blueprint"]
    panel = [float(np.mean([r["paired_uplift"]["mean"] for r in matches
                           if r["seed"] == seed and r["opponent"] != "blueprint"]))
             for seed in spec["training_seeds"]]
    return dict(units="milli-big-blinds per hand; positive favours search", matches=matches,
                primary_across_training_seeds=dict(seed_values=primary, **sample_summary(primary)),
                external_panel_uplift_across_training_seeds=dict(seed_values=panel, **sample_summary(panel)),
                timing_by_seed=seed_timings, inference="Three training trajectories, not opponents/hands, are the replication unit. "
                "Within-match intervals condition on frozen policies and measure Monte Carlo uncertainty. "
                "Across-seed t intervals are descriptive with n=3; secondary contrasts are exploratory. "
                "No whole-game exploitability or general robustness claim follows from winning this panel.")


def write_report(analysis, result):
    """Compact outputs survive an analysis-only download excluding task_results."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["training_seed", "opponent", "search_mbb_per_hand", "baseline_mbb_per_hand",
                     "paired_uplift_mbb_per_hand", "paired_mc_ci95_low", "paired_mc_ci95_high"])
    for row in result["matches"]:
        uplift = row.get("paired_uplift", {})
        writer.writerow([row["seed"], row["opponent"], row["search"]["mean"],
                         row.get("baseline", {}).get("mean"), uplift.get("mean"),
                         uplift.get("ci95_low"), uplift.get("ci95_high")])
    text = ["# Experiment 23: paired flop re-solving", "", result["inference"], "",
            "Values below are mbb/hand; positive favours search. Across-seed intervals are descriptive, n=3.", ""]
    for label, key in (("Search versus own blueprint (primary)", "primary_across_training_seeds"),
                       ("External-panel paired uplift (secondary)", "external_panel_uplift_across_training_seeds")):
        row = result[key]
        text += [f"## {label}", "", f"Mean {row['mean']:.3f}; 95% t interval "
                 f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]. Seed means: {row['seed_values']}.", ""]
    text += ["## Deployment diagnostics", ""]
    for seed, row in result["timing_by_seed"].items():
        text += [f"- Seed {seed}: {row['attempts']} flop entries; verified-change acceptance "
                 f"{row['acceptance_rate']:.1%}; latency overruns {row['overrun_rate']:.1%}; "
                 f"median/p95/p99 {row.get('median_seconds', 0):.4f}/"
                 f"{row.get('p95_seconds', 0):.4f}/{row.get('p99_seconds', 0):.4f} seconds."]
    text += ["", "Consult SUCCESS.json: a complete gameplay report can still fail the final latency/acceptance gates.",
             "Neither direct-play gains nor local continuation certificates establish whole-game exploitability.", ""]
    for name, content in (("matchups.csv", stream.getvalue()), ("report.md", "\n".join(text))):
        temporary = analysis / (name + ".tmp")
        temporary.write_text(content)
        os.replace(temporary, analysis / name)


def execute(input_root, native_root, output, *, preflight_only=False):
    spec = config()
    output = Path(output)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((analysis / name).exists() for name in ("SUCCESS.json", "FEASIBILITY_SUCCESS.json")):
            raise ValueError("Already completed; refusing to overwrite a completed experiment")
        runtime = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "open_spiel", "scipy")}
        contract = dict(spec=spec, evaluator_source=source_hash(Path(__file__).resolve().parents[1]),
                        mode="feasibility" if preflight_only else "gameplay",
                        native_source={name: source_hash(Path(native_root) / name) for name in ("fhp_escher", "vr_deep_cfr")},
                        runtime=runtime)
        identity = digest(contract)
        manifest = analysis / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError("Output identity differs; use a new run ID")
        write_json(manifest, contract)
        write_json(output / "status.json", dict(status="preflight", identity=identity))
        try:
            for label, record in required_sources(contract["mode"]).items():
                if sha256(Path(input_root) / f"{label}.pkl") != record["sha256"]:
                    raise ValueError(f"Checkpoint mismatch: {label}")
            write_json(analysis / "algebra_checks.json", algebra_checks())
            with worker_pool(spec) as pool:
                parity = list(pool.map(parity_task, spec["training_seeds"],
                                       [input_root] * 3, [native_root] * 3, [spec] * 3))
                write_json(analysis / "native_checks.json", parity)
                if preflight_only:
                    positions = [(seed, index) for index in range(3) for seed in spec["training_seeds"]]
                    profiles = list(pool.map(profile_task, positions, [input_root] * len(positions),
                                             [native_root] * len(positions), [spec] * len(positions)))
                    write_json(analysis / "preparation_profile.json", dict(
                        records=profiles, all_fixed_work_checks_passed=all(p["passed"] for p in profiles),
                        note="Uncapped fixed-work profiling, not gameplay latency. Native feature caches disabled; "
                             "legacy/vectorized order alternated. Gate below retains the unchanged 500ms budget."))
                # Four concurrently loaded workers, as in gameplay. Rechecked on recovery.
                checks = [(seed, index) for index in range(spec["preflight_positions_per_seed"])
                          for seed in spec["training_seeds"]]
                probes = list(pool.map(preflight_task, checks, [input_root] * len(checks),
                                       [native_root] * len(checks), [spec] * len(checks)))
                summaries = {str(seed): timing_summary([p for p in probes if p["seed"] == seed])
                             for seed in spec["training_seeds"]}
                write_json(analysis / "preflight.json", dict(probes=probes, timing_by_seed=summaries))
                for seed, row in summaries.items():
                    try:
                        enforce_gate(row, spec)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Seed {seed}: accepted {row['acceptance_rate']:.1%}, "
                                           f"overall deadline overruns {row['overrun_rate']:.1%}. {exc}") from exc
                if preflight_only:
                    write_json(analysis / "FEASIBILITY_SUCCESS.json", dict(status="preflight_passed", identity=identity,
                                                                          gameplay_performed=False))
                    write_json(output / "status.json", dict(status="preflight_passed", identity=identity))
                    return
                plan, results, pending = tasks(spec), [], {}
                for task in plan:
                    path = analysis / "task_results" / (task["id"] + ".json")
                    if path.exists():
                        results.append(read_shard(path, task, identity))
                    else:
                        future = pool.submit(score_task, task, input_root, native_root, spec)
                        pending[future] = task
                for future in as_completed(pending):
                    task = pending[future]
                    result = future.result()
                    write_json(analysis / "task_results" / (task["id"] + ".json"), sign_shard(result, identity))
                    results.append(result)
                    write_json(output / "status.json", dict(status="gameplay", completed=len(results), total=len(plan),
                                                             last_task=task["id"], identity=identity))
                    print(f"Completed {len(results)}/{len(plan)}: {task['id']}", flush=True)
                result = summarise(results, spec)
                write_json(analysis / "summary.json", result)
                write_report(analysis, result)
                for row in result["timing_by_seed"].values():
                    enforce_gate(row, spec)
                write_json(analysis / "SUCCESS.json", dict(status="succeeded", identity=identity, tasks=len(plan),
                                                            search_hands=240000, additional_baseline_hands=210000))
                write_json(output / "status.json", dict(status="succeeded", identity=identity))
        except BaseException as exc:
            write_json(output / "status.json", dict(status="failed", identity=identity, error=repr(exc)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true", help="Fixed-work profiling and 72 timed probes only; never launch gameplay")
    args = parser.parse_args()
    execute(str(Path(args.input_root).resolve()), str(Path(args.native_root).resolve()),
            args.output, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
