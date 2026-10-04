"""Bounded Experiment 9 seed-0 pilot; called by the Batch worker, never trains."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from ..loaders import sha256_file
from .runner import run_bounded, write_json


EXPERIMENT = "exp4_ucv_exp9_br_pilot"
SOURCE_RUN = "exp9-cache24-20261001-132550"
CHECKPOINT_RELATIVE = ("workers/task_000_cached_parallel_structured_ucv_escher_seed_0/"
                       "checkpoints/cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl")
CHECKPOINT_SHA256 = "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def select_threads(profiles):
    candidates = [r for r in profiles if r["status"] == "succeeded"]
    if not candidates:
        raise RuntimeError("No profile completed within its resource limits")
    # Selection uses timing alone, never poker payoffs. Ties prefer fewer threads.
    best = min(candidates, key=lambda r: (r["result"]["median_seconds_per_board"],
                                          r["result"]["threads"]))
    return int(best["result"]["threads"])


def run_pilot(checkpoint, repo_root, output, *, expected_sha256=CHECKPOINT_SHA256,
              smoke=False, runner=run_bounded):
    output, checkpoint, repo_root = map(lambda p: Path(p).resolve(), (output, checkpoint, repo_root))
    if (output / "pilot_manifest.json").exists():
        raise FileExistsError("Pilot outputs exist; use a new run/output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = dict(experiment=EXPERIMENT, status="running", started_utc=utc_now(),
                    source_run=SOURCE_RUN, training_seed=0, checkpoint_id="time_24h",
                    expected_checkpoint_sha256=expected_sha256,
                    scope="evaluation_only_pilot_not_a_final_policy_quality_estimate",
                    exact_full_game=False, smoke=smoke, stages=[],
                    python=platform.python_version(), platform=platform.platform(),
                    max_rss_mb_per_worker=8192, retries=0)

    def persist():
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "pilot_manifest.json", manifest)

    def stage(name, spec, seconds):
        manifest["active_stage"] = name
        persist()
        print(json.dumps(dict(event="pilot_stage_started", stage=name)), flush=True)
        result = runner(spec, output / name, max_seconds=seconds, max_rss_mb=8192)
        manifest["stages"].append(dict(name=name, status=result["status"],
                                        elapsed_seconds=result["elapsed_seconds"],
                                        peak_rss_bytes=result.get("peak_rss_bytes")))
        persist()
        print(json.dumps(dict(event="pilot_stage_finished", stage=name, status=result["status"])), flush=True)
        return result

    def require_success(record, name):
        if record["status"] != "succeeded":
            raise RuntimeError(f"{name} did not succeed: {record['status']}; see stage diagnostics")

    persist()
    try:
        actual = sha256_file(checkpoint)
        manifest["checkpoint_sha256"] = actual
        if actual != expected_sha256:
            raise ValueError("Checkpoint SHA-256 mismatch; refusing a different model")
        base = dict(family="ucv", checkpoint=str(checkpoint), repo_root=str(repo_root),
                    batch_size=1024, model_batch_size=32, seed=20261004, threads=1)
        require_success(stage("validation", dict(mode="validate", threads=1), 120), "Validation")
        require_success(stage("smoke", dict(base, mode="compare", deals=2, rollouts=8,
                                             seed=2026100401, search_seed=731), 120), "Native smoke")
        profiles = []
        for threads in (1, 2, 4, 8):
            record = stage(f"profile_threads_{threads}", dict(base, mode="profile", threads=threads,
                          boards=1 if smoke else 4, repeats=1 if smoke else 3), 120 if smoke else 600)
            profiles.append(record)
            # Timeout is a useful feasibility observation; malformed policies,
            # OOMs or monitoring failures must not be treated as timing results.
            if record["status"] not in ("succeeded", "time_limit"):
                require_success(record, f"Profile with {threads} threads")
        selected = select_threads(profiles)
        manifest.update(selected_threads=selected, selection_basis="lowest_successful_median_board_time",
                        profile_timeouts=[r["request"]["identity"]["spec"]["threads"]
                                          for r in profiles if r["status"] == "time_limit"])
        persist()
        comparison = stage("comparison", dict(base, mode="compare", threads=selected,
                             deals=2 if smoke else 25, rollouts=8 if smoke else 4096,
                             seed=2026100402, search_seed=731), 120 if smoke else 1800)
        require_success(comparison, "Paired comparison")
        manifest.update(status="succeeded", completed_utc=utc_now(), active_stage=None,
                        interpretation="Whole-game exploitability lower-bound estimate only; preflop is approximate.")
        persist()
        write_json(output / "SUCCESS.json", dict(experiment=EXPERIMENT, completed_utc=manifest["completed_utc"],
                                                 checkpoint_sha256=actual, selected_threads=selected,
                                                 computational_stages_complete=True))
        return manifest
    except BaseException as error:
        manifest.update(status="failed", completed_utc=utc_now(),
                        error_type=type(error).__name__, error=str(error))
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_pilot(args.checkpoint, args.repo_root, args.output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
