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
from .experiment import config, source_hash, stream_seed, tasks, required_sources, opponents, pair_count, workload
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
    rejected = [t["best_rejected_max_increase_bb"] for t in traces
                if not t["accepted"] and t.get("best_rejected_max_increase_bb") is not None]
    violation = (dict(n=len(rejected), minimum=float(np.min(rejected)), median=float(np.median(rejected)),
                      p95=float(np.quantile(rejected, .95)), maximum=float(np.max(rejected))) if rejected else None)
    return dict(attempts=len(traces), accepted=sum(bool(t["accepted"]) for t in traces),
                acceptance_rate=float(np.mean([t["accepted"] for t in traces])),
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
                minimum_iterations=min(t["iterations"] for t in traces),
                maximum_iterations=max(t["iterations"] for t in traces),
                mean_checks=float(np.mean([t.get("checks", 0) for t in traces])),
                failed_probe_best_violation_bb=violation,
                reasons=dict(Counter(t["reason"] for t in traces)))


def enforce_gate(summary, spec):
    metrics = [summary[k] for k in ("attempts", "acceptance_rate", "overrun_rate", "maximum_seconds")]
    if (not np.isfinite(metrics).all() or summary["attempts"] <= 0
            or not 0 <= summary["acceptance_rate"] <= 1 or not 0 <= summary["overrun_rate"] <= 1
            or summary["maximum_seconds"] < 0
            or summary["acceptance_rate"] < spec["minimum_acceptance_rate"]
            or summary["overrun_rate"] > spec["maximum_overrun_rate"]
            or summary["maximum_seconds"] > spec["maximum_latency_seconds"]):
        raise RuntimeError(f"{spec['search_seconds']:g}s feasibility gate failed "
                           f"(acceptance >= {spec['minimum_acceptance_rate']:.0%}, "
                           f"overruns <= {spec['maximum_overrun_rate']:.0%}, "
                           f"maximum latency <= {spec['maximum_latency_seconds']:g}s). "
                           "Do not interpret mostly-fallback play as a negative search result or raise the budget silently.")


def assess_gates(summaries, spec):
    """Retain every seed's failure, rather than hiding all but the first one."""
    seeds = {}
    for seed, row in summaries.items():
        try:
            enforce_gate(row, spec)
            seeds[seed] = dict(passed=True, error=None)
        except RuntimeError as exc:
            seeds[seed] = dict(passed=False, error=str(exc))
    return dict(passed=bool(seeds) and all(s["passed"] for s in seeds.values()), seeds=seeds,
                search_seconds=spec["search_seconds"], maximum_latency_seconds=spec["maximum_latency_seconds"],
                minimum_acceptance_rate=spec["minimum_acceptance_rate"],
                maximum_overrun_rate=spec["maximum_overrun_rate"])


def write_preflight_report(analysis, summaries, gates):
    text = ["# Experiment 23: flop-search feasibility", "",
            f"Search budget: {gates['search_seconds']:g} seconds per public-flop entry. "
            f"Overall gate: {'PASSED' if gates['passed'] else 'FAILED'}.", "",
            "This report measures verified-change coverage and cost, not playing strength or whole-game exploitability.", "",
            "| Seed | Accepted / probes | Rate | Mean iterations | Preparation complete | Overruns | Median / p95 / max seconds | Gate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for seed, row in summaries.items():
        text.append(f"| {seed} | {row['accepted']} / {row['attempts']} | {row['acceptance_rate']:.1%} | "
                    f"{row['mean_iterations']:.1f} | {row['preparation_completed']} / {row['attempts']} | "
                    f"{row['overrun_rate']:.1%} | {row['median_seconds']:.4f} / {row['p95_seconds']:.4f} / "
                    f"{row['maximum_seconds']:.4f} | {'PASS' if gates['seeds'][seed]['passed'] else 'FAIL'} |")
    text += ["", "## Limits and interpretation", "",
             f"Each seed must accept at least {gates['minimum_acceptance_rate']:.0%} of probes. "
             f"At most {gates['maximum_overrun_rate']:.0%} may overrun the search budget; "
             f"none may exceed the separate {gates['maximum_latency_seconds']:g}s pathological-latency guard.",
             "Late strategies are never deployed. The per-opponent-hand certificate tolerance remains 1e-8 BB; "
             "the strategy-change threshold remains 1e-6.", "",
             "All seeds and public probe positions match the earlier 500ms test. More computation alone is changed. "
             "Acceptance does not imply a head-to-head gain. A failed gate is a completed negative feasibility "
             "result, but the job intentionally exits nonzero and never starts gameplay in feasibility mode.", "",
             "preflight.json retains every position, stage timings, iteration counts and the smallest maximum "
             "per-hand violation found among rejected candidates; these are not whole-game exploitability values.", ""]
    for seed, row in summaries.items():
        violation = row.get("failed_probe_best_violation_bb")
        if violation:
            text.append(f"- Seed {seed}: among {violation['n']} failed probes with rejected candidates, "
                        f"best maximum per-hand violation median/p95/max = {violation['median']:.6g} / "
                        f"{violation['p95']:.6g} / {violation['maximum']:.6g} BB.")
    temporary = analysis / "feasibility_report.md.tmp"
    temporary.write_text("\n".join(text) + "\n")
    os.replace(temporary, analysis / "feasibility_report.md")


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
    n = pair_count(spec, opponent)
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


def validate_samples(result, task):
    if result["task"] != task:
        raise ValueError("Unexpected shard task")
    samples = result["samples"]
    if [s["pair_index"] for s in samples] != list(range(task["start"], task["start"] + task["count"])):
        raise ValueError("Missing, duplicate or out-of-order deals")
    searched = np.asarray([s["search_seats"] for s in samples], dtype=float)
    if searched.shape != (task["count"], 2) or not np.isfinite(searched).all():
        raise ValueError("Search samples require two finite seat payoffs per duplicate pair")
    if task["opponent"] == "blueprint":
        if any(s["baseline_seats"] is not None for s in samples):
            raise ValueError("Direct-play primary must not contain an external-panel control")
    else:
        baseline = np.asarray([s["baseline_seats"] for s in samples], dtype=float)
        if baseline.shape != searched.shape or not np.isfinite(baseline).all():
            raise ValueError("External-panel samples require matched finite controls in both seats")


def read_shard(path, task, identity):
    row = json.loads(Path(path).read_text())
    if (row.get("identity") != identity or digest(row["payload"]) != row.get("payload_sha256")
            or row["payload"]["task"] != task):
        raise ValueError(f"Incompatible or corrupted completed shard: {path}")
    validate_samples(row["payload"], task)
    return row["payload"]


def conditional_mean(components):
    """Equal-weight mean of independent match streams, holding models fixed.

    This approximate MC interval measures deal/action noise, NOT uncertainty
    over new training seeds. Keep it separate from the across-model t interval.
    """
    mean = float(np.mean([c["mean"] for c in components]))
    se = float(np.sqrt(sum(c["se"] ** 2 for c in components)) / len(components))
    return dict(mean=mean, se=se, ci95_low=mean - 1.959963984540054 * se,
                ci95_high=mean + 1.959963984540054 * se,
                independent_match_streams=len(components), duplicate_pairs=sum(c["n"] for c in components),
                interpretation="Equal-weight conditional Monte Carlo normal-approximation interval for "
                               "this frozen checkpoint cohort; not an interval over new trained models.")


def summarise(results, spec):
    expected = {t["id"]: t for t in tasks(spec)}
    ids = [r["task"]["id"] for r in results]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Incomplete/duplicated/unexpected cohort; refusing final summary")
    for result in results:
        validate_samples(result, expected[result["task"]["id"]])
    matches, seed_timings = [], {}
    for seed in spec["training_seeds"]:
        seed_rows = [r for r in results if r["task"]["seed"] == seed]
        seed_timings[str(seed)] = timing_summary([t for r in seed_rows for t in r["traces"]])
        for opponent in opponents(spec):
            rows = [r for r in seed_rows if r["task"]["opponent"] == opponent]
            samples = sorted([s for r in rows for s in r["samples"]], key=lambda x: x["pair_index"])
            if [s["pair_index"] for s in samples] != list(range(pair_count(spec, opponent))):
                raise ValueError("Incomplete/duplicated matchup; refusing final summary")
            searched = np.asarray([s["search_seats"] for s in samples])
            row = dict(seed=seed, opponent=opponent, search=sample_summary(searched.mean(1)),
                       search_seat_means=searched.mean(0).tolist(),
                       search_gameplay_seconds=sum(r["search_gameplay_seconds"] for r in rows),
                       baseline_gameplay_seconds=sum(r["baseline_gameplay_seconds"] for r in rows))
            if opponent != "blueprint":
                baseline = np.asarray([s["baseline_seats"] for s in samples])
                row.update(baseline=sample_summary(baseline.mean(1)),
                           baseline_seat_means=baseline.mean(0).tolist(),
                           paired_uplift=sample_summary((searched - baseline).mean(1)))
            matches.append(row)
    primary = [r["search"]["mean"] for r in matches if r["opponent"] == "blueprint"]
    panel = [float(np.mean([r["paired_uplift"]["mean"] for r in matches
                           if r["seed"] == seed and r["opponent"] != "blueprint"]))
             for seed in spec["training_seeds"]]
    return dict(units="milli-big-blinds per hand; positive favours search", matches=matches,
                protocol=spec["protocol"], search_seconds=spec["search_seconds"], workload=workload(spec),
                primary_across_training_seeds=dict(seed_values=primary, **sample_summary(primary)),
                external_panel_uplift_across_training_seeds=dict(seed_values=panel, **sample_summary(panel)),
                primary_frozen_cohort_mc=conditional_mean([r["search"] for r in matches if r["opponent"] == "blueprint"]),
                external_panel_frozen_cohort_mc=conditional_mean([r["paired_uplift"] for r in matches if r["opponent"] != "blueprint"]),
                measured_worker_seconds=dict(search=sum(r["search_gameplay_seconds"] for r in results),
                                             control=sum(r["baseline_gameplay_seconds"] for r in results)),
                timing_by_seed=seed_timings, inference="Three training trajectories, not opponents/hands, are the replication unit. "
                "Within-match intervals condition on frozen policies and measure Monte Carlo uncertainty. "
                "Across-seed t intervals are descriptive with n=3; secondary contrasts are exploratory. "
                "No whole-game exploitability or general robustness claim follows from winning this panel.")


def write_report(analysis, result):
    """Compact outputs survive an analysis-only download excluding task_results."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["training_seed", "opponent", "duplicate_pairs", "search_mbb_per_hand",
                     "search_mc_ci95_low", "search_mc_ci95_high", "baseline_mbb_per_hand",
                     "paired_uplift_mbb_per_hand", "paired_mc_ci95_low", "paired_mc_ci95_high"])
    for row in result["matches"]:
        uplift = row.get("paired_uplift", {})
        writer.writerow([row["seed"], row["opponent"], row["search"]["n"], row["search"]["mean"],
                         row["search"]["ci95_low"], row["search"]["ci95_high"],
                         row.get("baseline", {}).get("mean"), uplift.get("mean"),
                         uplift.get("ci95_low"), uplift.get("ci95_high")])
    text = ["# Experiment 23: paired flop re-solving", "", result["inference"], "",
            f"Protocol: {result['protocol']}; search allowance: {result['search_seconds']:g}s; "
            f"fixed completed sample: {result['workload']['total_hands']:,} hands.", "",
            "Values below are mbb/hand; positive favours search. Across-seed intervals are descriptive, n=3.", ""]
    for label, key in (("Search versus own blueprint (primary)", "primary_across_training_seeds"),
                       ("External-panel paired uplift (secondary)", "external_panel_uplift_across_training_seeds")):
        row = result[key]
        text += [f"## {label}", "", f"Mean {row['mean']:.3f}; 95% t interval "
                 f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]. Seed means: {row['seed_values']}.", ""]
    text += ["## Conditional Monte Carlo precision for these frozen models", ""]
    for label, key in (("Primary", "primary_frozen_cohort_mc"), ("External-panel uplift", "external_panel_frozen_cohort_mc")):
        row = result[key]
        text += [f"- {label}: mean {row['mean']:.3f}; approximate 95% MC interval "
                 f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] from {row['duplicate_pairs']:,} duplicate pairs."]
    text += ["", result["primary_frozen_cohort_mc"]["interpretation"], "",
             "## Model-seed and opponent results", "",
             "| Seed | Opponent | Pairs | Search payoff | Control payoff | Primary payoff / paired uplift | 95% MC interval |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["matches"]:
        estimate = row.get("paired_uplift", row["search"])
        control = f"{row['baseline']['mean']:.3f}" if "baseline" in row else "—"
        text += [f"| {row['seed']} | {row['opponent']} | {row['search']['n']} | {row['search']['mean']:.3f} | "
                 f"{control} | {estimate['mean']:.3f} | [{estimate['ci95_low']:.3f}, {estimate['ci95_high']:.3f}] |"]
    text += [""]
    text += ["## Deployment diagnostics", ""]
    for seed, row in result["timing_by_seed"].items():
        text += [f"- Seed {seed}: {row['attempts']} flop entries; verified-change acceptance "
                 f"{row['acceptance_rate']:.1%}; latency overruns {row['overrun_rate']:.1%}; "
                 f"median/p95/p99 {row.get('median_seconds', 0):.4f}/"
                 f"{row.get('p95_seconds', 0):.4f}/{row.get('p99_seconds', 0):.4f} seconds; "
                 f"changed visited-decision rate {row.get('changed_decision_rate')}; "
                 f"mean decision total-variation distance {row.get('mean_decision_total_variation')}."]
    text += ["", f"Summed worker gameplay seconds (not wall time): search "
             f"{result['measured_worker_seconds']['search']:.1f}; controls "
             f"{result['measured_worker_seconds']['control']:.1f}. Consult Batch duration and bootstrap logs "
             "for setup, diagnostics, uploads and interrupted-attempt overhead."]
    text += ["", "Consult SUCCESS.json: a complete gameplay report can still fail the final latency/acceptance gates.",
             "Neither direct-play gains nor local continuation certificates establish whole-game exploitability.", ""]
    for name, content in (("matchups.csv", stream.getvalue()), ("report.md", "\n".join(text))):
        temporary = analysis / (name + ".tmp")
        temporary.write_text(content)
        os.replace(temporary, analysis / name)


def run_gameplay(pool, input_root, native_root, output, spec, identity):
    """Complete signed shards, or leave a resumable, explicitly incomplete run."""
    output = Path(output)
    analysis = output / "analysis"
    plan, results, pending = tasks(spec), [], {}
    started = time.monotonic()

    def progress(state, last_task=None):
        counts = Counter((r["task"]["seed"], r["task"]["opponent"]) for r in results)
        pairs = Counter()
        for r in results:
            pairs[r["task"]["seed"], r["task"]["opponent"]] += r["task"]["count"]
        row = dict(status=state, identity=identity, completed=len(results), total=len(plan), last_task=last_task,
                   attempt_gameplay_seconds=time.monotonic() - started,
                   gameplay_budget_seconds=spec["gameplay_budget_seconds"],
                   matches=[dict(seed=s, opponent=o, completed_pairs=pairs[s, o],
                                 required_pairs=pair_count(spec, o), completed_shards=counts[s, o])
                            for s in spec["training_seeds"] for o in opponents(spec)])
        write_json(analysis / "progress.json", row)
        write_json(output / "status.json", row)

    for task in plan:
        path = analysis / "task_results" / (task["id"] + ".json")
        if path.exists():
            results.append(read_shard(path, task, identity))
        else:
            pending[pool.submit(score_task, task, input_root, native_root, spec)] = task
    progress("gameplay")
    print(f"Gameplay: {len(results)}/{len(plan)} signed shards reused; "
          f"{spec['gameplay_budget_seconds'] / 3600:g}h attempt budget; no payoff-based stopping.", flush=True)
    try:
        remaining = max(0., spec["gameplay_budget_seconds"] - (time.monotonic() - started))
        for future in as_completed(pending, timeout=remaining):
            task = pending[future]
            result = future.result()
            validate_samples(result, task)
            write_json(analysis / "task_results" / (task["id"] + ".json"), sign_shard(result, identity))
            results.append(result)
            progress("gameplay", task["id"])
            print(f"Completed {len(results)}/{len(plan)}: {task['id']}", flush=True)
    except TimeoutError:
        progress("budget_exhausted")
        # worker_pool terminates running/queued workers on exit. Existing signed
        # shards survive; neither truncated samples nor partial CIs are published.
        raise TimeoutError("Gameplay budget exhausted before the fixed sample completed. "
                           "Resume the same pinned run; no final result or SUCCESS marker was produced.") from None
    progress("aggregating")
    return results


def execute(input_root, native_root, output, *, preflight_only=False):
    attempt_started = time.monotonic()
    spec = config("feasibility" if preflight_only else "gameplay")
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
        if not preflight_only:
            write_json(analysis / "workload.json", workload(spec))
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
                             f"legacy/vectorized order alternated. Timed probes use {spec['search_seconds']:g}s."))
                    if not all(p["passed"] for p in profiles):
                        raise RuntimeError("Fixed-work preparation parity failed")
                # Four concurrently loaded workers, as in gameplay. Rechecked on recovery.
                checks = [(seed, index) for index in range(spec["preflight_positions_per_seed"])
                          for seed in spec["training_seeds"]]
                probes = list(pool.map(preflight_task, checks, [input_root] * len(checks),
                                       [native_root] * len(checks), [spec] * len(checks)))
                summaries = {str(seed): timing_summary([p for p in probes if p["seed"] == seed])
                             for seed in spec["training_seeds"]}
                write_json(analysis / "preflight.json", dict(probes=probes, timing_by_seed=summaries))
                gates = assess_gates(summaries, spec)
                write_json(analysis / "feasibility_gates.json", gates)
                write_preflight_report(analysis, summaries, gates)
                for seed, row in summaries.items():
                    print(f"Seed {seed}: {row['accepted']}/{row['attempts']} accepted; "
                          f"{row['mean_iterations']:.1f} mean iterations; "
                          f"{row['overrun_rate']:.1%} overruns; gate "
                          f"{'PASS' if gates['seeds'][seed]['passed'] else 'FAIL'}", flush=True)
                if not gates["passed"]:
                    failures = [f"Seed {seed}: accepted {summaries[seed]['acceptance_rate']:.1%}, "
                                f"overall deadline overruns {summaries[seed]['overrun_rate']:.1%}"
                                for seed, gate in gates["seeds"].items() if not gate["passed"]]
                    raise RuntimeError(f"{spec['search_seconds']:g}s feasibility gate failed. " +
                                       "; ".join(failures) + ". Inspect analysis/feasibility_gates.json and feasibility_report.md.")
                if preflight_only:
                    write_json(analysis / "FEASIBILITY_SUCCESS.json", dict(status="preflight_passed", identity=identity,
                                                                          gameplay_performed=False))
                    write_json(output / "status.json", dict(status="preflight_passed", identity=identity))
                    return
                # Exercise all nine actual policies and paired-control paths
                # before committing hours. Separate streams; no payoff gate and
                # no smoke sample enters the final fixed-sample analysis.
                smoke_spec = dict(spec, evaluation_seed=spec["evaluation_seed"] + 1000000,
                                  pairs_per_opponent={o: 2 for o in opponents(spec)})
                smoke_plan = tasks(smoke_spec)
                smoke = list(pool.map(score_task, smoke_plan, [input_root] * len(smoke_plan),
                                      [native_root] * len(smoke_plan), [smoke_spec] * len(smoke_plan)))
                for row, task in zip(smoke, smoke_plan):
                    validate_samples(row, task)
                write_json(analysis / "gameplay_smoke.json", dict(passed=True, results=smoke,
                            evaluation_seed=smoke_spec["evaluation_seed"], workload=workload(smoke_spec),
                            note="Separate diagnostic deals; no payoff threshold; excluded from final estimates."))
                results = run_gameplay(pool, input_root, native_root, output, spec, identity)
                result = summarise(results, spec)
                result["current_attempt_seconds_through_aggregation"] = time.monotonic() - attempt_started
                write_json(analysis / "summary.json", result)
                write_report(analysis, result)
                final_gates = assess_gates(result["timing_by_seed"], spec)
                write_json(analysis / "gameplay_gates.json", final_gates)
                if not final_gates["passed"]:
                    raise RuntimeError("Complete fixed-sample gameplay failed final engineering gates; "
                                       "inspect gameplay_gates.json. Payoffs did not control stopping.")
                write_json(analysis / "SUCCESS.json", dict(status="succeeded", identity=identity,
                                                            tasks=len(tasks(spec)), **workload(spec)))
                progress = json.loads((analysis / "progress.json").read_text())
                write_json(analysis / "progress.json", dict(progress, status="succeeded"))
                write_json(output / "status.json", dict(status="succeeded", identity=identity))
        except BaseException as exc:
            write_json(output / "status.json", dict(status="budget_exhausted" if isinstance(exc, TimeoutError) else "failed",
                                                    identity=identity, error=repr(exc)))
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
