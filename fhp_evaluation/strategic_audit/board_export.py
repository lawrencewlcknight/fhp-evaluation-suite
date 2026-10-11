"""Board-sized, bounded native exports; the policy and scoring remain unchanged.

Each task uses the existing isolated worker with exactly one board. Only the
coordinator owns the directory lock; tasks never share writable board paths.
No completed marker means no reusable board. No automatic retries are made.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time

from ..best_response.runner import run_bounded, write_json
from .library import board_key


class AuditDeferred(RuntimeError):
    """A bounded invocation stopped; verified board caches can be resumed."""


def validate_config(config, threads, max_rss_mb):
    if set(config) != {"workers", "task_seconds", "session_seconds", "cpu_limit"}:
        raise ValueError("Exact board-export resource contract required")
    if any(type(config[k]) is not int or config[k] <= 0 for k in config):
        raise ValueError("Board-export settings must be positive integers")
    if config["workers"] * threads > config["cpu_limit"]:
        raise ValueError("Export workers times threads exceeds the CPU allocation")
    if config["task_seconds"] > config["session_seconds"] or max_rss_mb < config["workers"]:
        raise ValueError("Invalid per-task time or aggregate worker memory budget")
    return dict(config)


def next_attempt(root):
    root = Path(root)
    numbers = [int(p.name.removeprefix("attempt_")) for p in root.glob("attempt_*")
               if p.is_dir() and p.name.removeprefix("attempt_").isdigit()]
    return root / f"attempt_{max(numbers, default=-1) + 1:03d}"


def export_boards(provenance, boards, directory, fingerprint, config, *, deadline,
                  max_rss_mb, completed):
    """Run deterministic waves and publish timing/progress after every wave.

    The first wave is a real-archive parallel-throughput measurement, not a
    synthetic speedup claim. Benchmark the same frozen panel in a fresh run
    with workers=1/2/4 to compare allocations. Never infer strength from timings.
    """
    directory = Path(directory)
    policies = provenance["policies"]
    spec = provenance["spec"]["policies"]
    order = sorted(policies, key=lambda label: (spec[label]["family"] != "sd_cfr", label))
    tasks = [(label, board) for label in order for board in boards
             if not completed(directory / "tables" / label / board_key(board),
                              fingerprint, "tables.npz")]
    total = len(policies) * len(boards)
    initial_cached = total - len(tasks)
    segments = directory / "export_segments"
    segments.mkdir(parents=True, exist_ok=True)
    segment = next_attempt(segments)
    segment.mkdir()
    started = time.monotonic()
    report = dict(identity=fingerprint, config=config, status="running",
                  total_boards=total, initial_cached_boards=initial_cached,
                  completed_boards=initial_cached, remaining_boards=len(tasks), waves=[],
                  per_worker_rss_limit_mb=max_rss_mb // config["workers"],
                  worker_memory_budget_mb=max_rss_mb,
                  note="Board-task timing, including startup and archive validation. "
                       "First wave measures actual parallel throughput, not a controlled serial speedup.")
    cancelled = threading.Event()
    def publish(status):
        report.update(status=status, elapsed_seconds=time.monotonic() - started)
        write_json(segment / "progress.json", report)
        write_json(directory / "export_progress.json", report)

    def execute(task, seconds):
        label, board = task
        cache = directory / "tables" / label
        attempt = next_attempt(directory / "workers" / label / board_key(board))
        worker_spec = dict(operation="tables", policy=spec[label], boards=[board],
                           cache=str(cache), identity=fingerprint,
                           threads=provenance["threads"], expected_policy_identity=policies[label])
        result = run_bounded(worker_spec, attempt, max_seconds=seconds,
                             cancel_event=cancelled,
                             max_rss_mb=report["per_worker_rss_limit_mb"],
                             command=[sys.executable, "-m", "fhp_evaluation.strategic_audit.worker",
                                      str(attempt / "spec.json")])
        return dict(policy=label, board=list(board), attempt=str(attempt.relative_to(directory)),
                    status=result["status"], elapsed_seconds=result.get("elapsed_seconds"),
                    peak_rss_bytes=result.get("peak_rss_bytes"), max_seconds=seconds)

    publish("running")
    while tasks:
        wave = tasks[:config["workers"]]
        families = {spec[label]["family"] for label, _ in wave}
        remaining = deadline - time.monotonic()
        # Do not spend the last few minutes starting a large native archive.
        # Reserve the slowest observed comparable family plus 20%; an expensive
        # SD wave must not prevent short single-network exports from starting.
        observed = [r["elapsed_seconds"] for w in report["waves"] for r in w["tasks"]
                    if r["status"] == "succeeded" and r["elapsed_seconds"] is not None
                    and spec[r["policy"]]["family"] in families]
        minimum = min(config["task_seconds"], max(300., 1.2 * max(observed, default=0.)))
        if remaining < minimum:
            publish("deferred_time_budget")
            raise AuditDeferred("Board export reached its invocation budget; resume with identical inputs.")
        wave_started = time.monotonic()
        seconds = min(float(config["task_seconds"]), remaining)
        try:
            with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
                futures = [pool.submit(execute, task, seconds) for task in wave]
                try:
                    results = [future.result() for future in futures]
                except BaseException:
                    cancelled.set()  # Stop only this wave's owned process groups.
                    raise
            for row in results:
                path = directory / "tables" / row["policy"] / board_key(row["board"])
                if row["status"] == "succeeded" and not completed(path, fingerprint, "tables.npz"):
                    row["status"] = "missing_verified_board"
        except BaseException:
            publish("failed")
            raise
        elapsed = time.monotonic() - wave_started
        succeeded = sum(r["status"] == "succeeded" for r in results)
        report["waves"].append(dict(tasks=results, elapsed_seconds=elapsed,
                                   completed_boards=succeeded,
                                   boards_per_hour=succeeded * 3600 / elapsed if elapsed > 0 else None,
                                   # Sum of separate peaks is a conservative upper bound,
                                   # not a measured simultaneous process-tree peak.
                                   sum_worker_peak_rss_bytes=sum(r["peak_rss_bytes"] or 0 for r in results)))
        report["completed_boards"] += succeeded
        report["remaining_boards"] -= succeeded
        failures = [r for r in results if r["status"] != "succeeded"]
        if failures:
            if all(r["status"] == "time_limit" and r["max_seconds"] < config["task_seconds"]
                   for r in failures):
                publish("deferred_time_budget")
                raise AuditDeferred("Invocation budget expired during export; completed boards are reusable.")
            publish("failed")
            raise RuntimeError(f"Board export failed: {failures}. See the named task run.json/stderr.log.")
        tasks = tasks[len(wave):]
        publish("running")
    publish("complete")
    return report
