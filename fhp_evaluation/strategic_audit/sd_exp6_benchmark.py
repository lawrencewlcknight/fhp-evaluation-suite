"""Execution-only SD6 benchmark. Never migrates caches or resumes the scientific audit."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import statistics
import time

from . import sd_exp6 as audit

EXPERIMENT = "sd6_execution_benchmark_v1"
SOURCE_RUN = "fhp-sd6-audit-dev-20261013-104108"
ALLOCATIONS = ((4, 4), (8, 2), (8, 1))
REPEATS = 2
CELL_SECONDS = 3 * 3600


def rows(progress):
    return progress.get("tasks") or [r for wave in progress["waves"] for r in wave["tasks"]]


def simulate(durations, workers=4):
    slots = [0.] * workers
    for duration in durations:
        heapq.heapreplace(slots, slots[0] + duration)
    return max(slots)


def make_plan(progress):
    """Stratify on execution cost only, never on policy quality or assessment states."""
    if progress.get("scheduler") or progress.get("config", {}).get("workers") != 4:
        raise ValueError("Expected the original four-worker wave timing report")
    tasks = rows(progress)
    success = [r for r in tasks if r["status"] == "succeeded"]
    if len({(r["policy"], tuple(r["board"])) for r in tasks}) != len(tasks):
        raise ValueError("Duplicate timing tasks")
    by_hour = {h: {tuple(r["board"]): r["elapsed_seconds"] for r in success
                   if r["policy"] == f"sd_exp6_seed0_{h}h"} for h in (24, 48)}
    if len(by_hour[24]) != 64 or len(by_hour[48]) < 16:
        raise ValueError("Insufficient original seed-0 timing coverage")
    if any(not 0 < s < float("inf") for v in by_hour.values() for s in v.values()):
        raise ValueError("Invalid task duration")
    paired = sorted(set(by_hour[24]) & set(by_hour[48]), key=lambda b: (by_hour[48][b], b))
    chosen = [paired[round(q * (len(paired) - 1))] for q in (0, 1/3, 2/3, 1)]
    ratio = statistics.median(by_hour[48][b] / by_hour[24][b] for b in paired)
    completed = {(r["policy"], tuple(r["board"])) for r in success}
    future = []
    for label, record in audit.checkpoint_registry("development").items():
        h = record["hours"]
        scale = record["completed_iterations"] / audit.HASHES[0][int(h == 48)][0]
        for board, seconds24 in sorted(by_hour[24].items()):
            if (label, board) not in completed:
                future.append(by_hour[h].get(board, seconds24 * ratio) * scale)
    wall = sum(w["elapsed_seconds"] for w in progress["waves"])
    busy = sum(r["elapsed_seconds"] for r in tasks)
    forecast = simulate(future) / 3600
    return dict(experiment=EXPERIMENT, source_run=SOURCE_RUN, original_identity=progress["identity"],
        saved_successful_exports=len(success), allocations=[list(a) for a in ALLOCATIONS],
        repeats=REPEATS, cell_seconds=CELL_SECONDS, boards=[list(b) for b in chosen],
        selection="Seed-0 48h successful-board cost ranks: min, one-third, two-thirds, max; paired 24h/48h.",
        failed_boards=[r for r in tasks if r["status"] != "succeeded"],
        measured=dict(wall_hours=wall / 3600, worker_hours=busy / 3600,
                      wave_worker_slot_idle_fraction=1 - busy / (4 * wall),
                      observed_48_to_24_median_ratio=ratio,
                      hours={str(h): dict(count=len(v), mean_minutes=statistics.mean(v.values()) / 60,
                              min_minutes=min(v.values()) / 60, max_minutes=max(v.values()) / 60)
                             for h, v in by_hour.items()}),
        forecast=dict(remaining_sd_exports=len(future), rolling4_hours=forecast,
                      sensitivity_hours=[forecast * .8, forecast * 1.3],
                      assumed_reusable_exports=len(success), downstream_hours=None,
                      note="Scheduling simulation, not measured speedup or confidence interval. Holds task durations "
                           "at observed 4x4 contention; scales other seeds by model count. Excludes rules, scoring, "
                           "setup, transfer, repair and cache migration. Cache reuse is not yet authorised or implemented."))


def validate_plan(plan):
    if (plan.get("experiment") != EXPERIMENT or plan.get("source_run") != SOURCE_RUN
            or plan.get("allocations") != [list(a) for a in ALLOCATIONS]
            or plan.get("repeats") != REPEATS or plan.get("cell_seconds") != CELL_SECONDS):
        raise ValueError("Unexpected benchmark contract")
    boards = plan.get("boards", [])
    if (len(boards) != 4 or len({tuple(b) for b in boards}) != 4
            or any(len(b) != 3 or b != sorted(set(b))
                   or any(type(c) is not int or not 0 <= c < 52 for c in b) for b in boards)):
        raise ValueError("Four distinct valid boards required")


def compare_tables(reference, candidate, policies, boards):
    import numpy as np
    from .library import board_key
    largest = 0.
    for label in policies:
        for board in boards:
            rel = Path("tables") / label / board_key(board) / "tables.npz"
            with np.load(reference / rel, allow_pickle=False) as a, np.load(candidate / rel, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise ValueError("Benchmark output schemas differ")
                for key in a.files:
                    if a[key].shape != b[key].shape or not np.isfinite(b[key]).all():
                        raise ValueError("Invalid benchmark probabilities")
                    largest = max(largest, float(np.max(np.abs(a[key] - b[key]))))
    return dict(max_abs_difference=largest, tolerance=3e-6, passed=largest <= 3e-6)


def summary(cells, plan):
    groups = {}
    for workers, threads in ALLOCATIONS:
        label = f"w{workers}-t{threads}"
        group = [c for c in cells if c["allocation"] == label]
        valid = [c for c in group if c.get("status") == "complete" and c.get("parity", {}).get("passed")]
        tasks = [t for c in valid for t in c.get("tasks", [])]
        by_horizon = {}
        for hour in (24, 48):
            times = [t["elapsed_seconds"] for t in tasks if t["policy"].endswith(f"_{hour}h")
                     and t.get("elapsed_seconds") is not None]
            if times:
                by_horizon[str(hour)] = dict(count=len(times), mean_seconds=statistics.mean(times),
                                            min_seconds=min(times), max_seconds=max(times))
        groups[label] = dict(completed_repetitions=len(valid), eligible=len(valid) == REPEATS,
                            wall_seconds=[c["elapsed_seconds"] for c in valid],
                            mean_seconds=statistics.mean(c["elapsed_seconds"] for c in valid) if valid else None,
                            per_horizon_worker_timings=by_horizon,
                            peak_worker_rss_bytes=max((t.get("peak_rss_bytes") or 0 for t in tasks), default=0))
    baseline = groups["w4-t4"]
    eligible = {k: v for k, v in groups.items() if v["eligible"]}
    winner = min(eligible, key=lambda k: eligible[k]["mean_seconds"]) if eligible and baseline["eligible"] else None
    if winner:
        for g in groups.values():
            g["speedup_vs_rolling4x4"] = baseline["mean_seconds"] / g["mean_seconds"] if g["eligible"] else None
    return dict(experiment=EXPERIMENT, groups=groups, provisional_fastest=winner,
                audit_restarted=False, failed_board_resolved=False, saved_exports_modified=False,
                revised_plan=plan, cells=cells,
                caveats=["Execution benchmark, not policy-strength evidence; four cost-stratified boards, seed 0 only.",
                         "Two repetitions with reversed allocation order; shared OS cache is not flushed.",
                         "Same complete archives, hand tables, batch sizes and correctness checks in every cell.",
                         "48 full board exports total. Current wave scheduler is simulated from saved timings, not rerun.",
                         "Do not scale the full audit estimate by one speedup without checking 24h/48h and board tails.",
                         "No recommendation to restart until the original parity failure and cache compatibility are resolved."])


def benchmark(plan, input_root, native_root, output):
    from ..best_response.runner import write_json
    from .board_export import export_boards
    from .library import board_schedule
    from .workflow import completed, digest, exclusive, policy_identity
    validate_plan(plan)
    allowed = {tuple(b) for b in board_schedule(audit.make_library("development"), "development")}
    if not {tuple(b) for b in plan["boards"]} <= allowed:
        raise ValueError("Benchmark may only use existing development boards")
    output = Path(output)
    with exclusive(output):
        if (output / "benchmark_manifest.json").exists():
            raise FileExistsError("Use a new benchmark run; repetitions must never hit an old timing cache")
        spec = audit.make_spec("pilot", input_root, native_root)
        policies = {k: v for k, v in spec["policies"].items() if v["family"] == "sd_cfr"}
        identities = {k: policy_identity(v) for k, v in policies.items()}
        write_json(output / "benchmark_manifest.json", dict(plan=plan, policies=identities))
        cells, reference = [], None
        for repetition in range(REPEATS):
            order = ALLOCATIONS if repetition == 0 else tuple(reversed(ALLOCATIONS))
            for workers, threads in order:
                label = f"w{workers}-t{threads}"
                directory = output / "cells" / f"rep{repetition}-{label}"
                provenance = dict(spec=dict(policies=policies), policies=identities, threads=threads)
                fingerprint = digest(dict(provenance=provenance, plan=plan, repetition=repetition))
                config = dict(workers=workers, cpu_limit=16, task_seconds=10800, session_seconds=CELL_SECONDS)
                started = time.monotonic()
                cell = dict(allocation=label, repetition=repetition, workers=workers, threads=threads)
                try:
                    report = export_boards(provenance, plan["boards"], directory, fingerprint, config,
                        deadline=started + CELL_SECONDS, max_rss_mb=52000, completed=completed)
                    cell.update(status="complete", tasks=report["tasks"])
                except (RuntimeError, ValueError) as error:
                    cell.update(status="incomplete", error=str(error))
                cell["elapsed_seconds"] = time.monotonic() - started
                cell["directory"] = str(directory.relative_to(output))
                if cell["status"] == "complete":
                    if reference is None and label == "w4-t4":
                        reference = directory
                    if reference is not None:
                        cell["parity"] = compare_tables(reference, directory, policies, plan["boards"])
                cells.append(cell)
                write_json(output / "analysis/benchmark_summary.json", summary(cells, plan))
        # If the first baseline failed but the second passed, compare earlier complete cells too.
        if reference is not None:
            for cell in cells:
                if cell["status"] == "complete" and "parity" not in cell:
                    cell["parity"] = compare_tables(reference, output / cell["directory"], policies, plan["boards"])
        result = summary(cells, plan)
        write_json(output / "analysis/benchmark_summary.json", result)
        write_json(output / "status.json", dict(status="benchmark_complete" if all(
            g["eligible"] for g in result["groups"].values()) else "benchmark_incomplete", audit_restarted=False))
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run"))
    p.add_argument("--progress", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--input-root", type=Path)
    p.add_argument("--native-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.action == "plan":
        raw = args.progress.read_bytes()
        plan = make_plan(json.loads(raw))
        plan["progress_sha256"] = hashlib.sha256(raw).hexdigest()
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
    else:
        if not all((args.input_root, args.native_root, args.output)):
            p.error("run requires input-root, native-root and output")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        from ..best_response.runner import write_json
        if args.download:
            write_json(args.output / "archive_sources.json", audit.download_archives("pilot", args.input_root))
        result = benchmark(plan, args.input_root, args.native_root, args.output)
        print(json.dumps({"provisional_fastest": result["provisional_fastest"], "audit_restarted": False}))


if __name__ == "__main__":
    main()
