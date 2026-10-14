"""Board-sized, bounded native exports; the policy and scoring remain unchanged.

Each task uses the existing isolated worker with exactly one board. Only the
coordinator owns the directory lock; tasks never share writable board paths.
No completed marker means no reusable board. No automatic retries are made.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    """Keep slots occupied; checkpoint each completion, never retry in place."""
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
                  scheduler="rolling_v1", tasks=[], failures=[],
                  per_worker_rss_limit_mb=max_rss_mb // config["workers"],
                  worker_memory_budget_mb=max_rss_mb,
                  note="Board-task timing, including startup and archive validation. "
                       "Rolling scheduling; waves is a legacy completion-event view, not concurrent waves.")
    cancelled = threading.Event()
    def publish(status):
        report.update(status=status, elapsed_seconds=time.monotonic() - started)
        write_json(segment / "progress.json", report)
        write_json(directory / "export_progress.json", report)

    def execute(task, seconds):
        task_started = time.monotonic() - started
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
                    peak_rss_bytes=result.get("peak_rss_bytes"), max_seconds=seconds,
                    started_seconds=task_started, finished_seconds=time.monotonic() - started)

    publish("running")
    pending = {}
    pool = ThreadPoolExecutor(max_workers=config["workers"])
    try:
        while tasks or pending:
            while tasks and len(pending) < config["workers"]:
                remaining = deadline - time.monotonic()
                family = spec[tasks[0][0]]["family"]
                observed = [r["elapsed_seconds"] for r in report["tasks"]
                            if r["status"] == "succeeded" and r["elapsed_seconds"] is not None
                            and spec[r["policy"]]["family"] == family]
                minimum = min(config["task_seconds"], max(300., 1.2 * max(observed, default=0.)))
                if remaining < minimum:
                    break
                task = tasks.pop(0)
                future = pool.submit(execute, task, min(float(config["task_seconds"]), remaining))
                pending[future] = task
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                row = future.result()
                path = directory / "tables" / row["policy"] / board_key(row["board"])
                if row["status"] == "succeeded" and not completed(path, fingerprint, "tables.npz"):
                    row["status"] = "missing_verified_board"
                report["tasks"].append(row)
                succeeded = int(row["status"] == "succeeded")
                report["completed_boards"] += succeeded
                report["remaining_boards"] -= succeeded
                if not succeeded:
                    report["failures"].append(row)
                # Legacy readers flatten waves; these are individual completions now.
                report["waves"].append(dict(tasks=[row], completed_boards=succeeded,
                    elapsed_seconds=row["elapsed_seconds"],
                    boards_per_hour=3600 / row["elapsed_seconds"] if succeeded and row["elapsed_seconds"] else None))
                publish("running_with_failures" if report["failures"] else "running")
    except BaseException:
        cancelled.set()
        publish("failed")
        raise
    finally:
        pool.shutdown(wait=True)
    failures = report["failures"]
    hard_failures = [r for r in failures if not (r["status"] == "time_limit"
                     and r["max_seconds"] < config["task_seconds"])]
    if hard_failures:
        publish("failed")
        raise RuntimeError(f"Board export failed: {hard_failures}. See the named task run.json/stderr.log.")
    if tasks or failures:
        publish("deferred_time_budget")
        raise AuditDeferred("Invocation budget expired; resume verified completed boards with identical inputs.")
    publish("complete")
    return report
