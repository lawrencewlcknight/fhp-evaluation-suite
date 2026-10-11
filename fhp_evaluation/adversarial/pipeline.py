"""One-VM Exp21 orchestration, with unchanged science and explicit stage barriers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

from ..best_response.runner import run_bounded, write_json
from ..loaders import sha256_file
from .config import ARMS, configuration

LIMITS = {"qualify": 7200, "train": 50400, "evaluate": 21600, "aggregate": 900}


class QualificationStopped(RuntimeError):
    """A valid measurement did not qualify the attackers; do not spend repair time."""


def checked_result(directory, stage, seed, smoke, bundle, arm=None):
    directory = Path(directory)
    marker = json.loads((directory / "SUCCESS.json").read_text())
    if marker.get("result_sha256") != sha256_file(directory / "result.json"):
        raise ValueError(f"Result checksum mismatch: {directory}")
    row = json.loads((directory / "result.json").read_text())
    expected = dict(phase=stage, seed=seed, config_hash=configuration(smoke).fingerprint(),
                    bundle_sha256=bundle, status="smoke_complete" if smoke else "complete")
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incomplete or mixed stage result: {directory}")
    if any(marker.get(key) != expected[key] for key in ("phase", "seed", "config_hash")):
        raise ValueError(f"Mixed success marker: {directory}")
    if arm is not None and row.get("arm") != arm:
        raise ValueError(f"Wrong evaluation arm: {directory}")
    return row


def task(stage, index, smoke, root, inputs, native_repo, bucket, bundle, cancel):
    seed = index // 4 if stage == "evaluate" else index
    arm = ARMS[index % 4] if stage == "evaluate" else None
    output = root / stage / f"task_{index}" / "analysis"
    measurement = output / "measurement"
    command = [sys.executable, "-m", "fhp_evaluation.adversarial.workflow", stage,
               "--seed", str(seed), "--source", str(inputs / f"seed_{seed}"),
               "--native-repo", str(native_repo), "--bucket", bucket,
               "--output", str(measurement), "--worker"]
    if smoke:
        command.append("--smoke")
    if stage == "train":
        prior = root / "qualify" / f"task_{seed}" / "analysis"
        checked_result(prior, "qualify", seed, smoke, bundle)
        command += ["--gate", str(prior / "result.json"),
                    "--replay", str(prior / "measurement" / "replay")]
    elif stage == "evaluate":
        prior = root / "train" / f"task_{seed}" / "analysis"
        checked_result(prior, "train", seed, smoke, bundle)
        command += ["--training", str(prior), "--arm", arm]
    record = run_bounded(dict(mode="exp21", phase=stage, seed=seed, arm=arm,
                              smoke=smoke, threads=2), measurement,
                         max_seconds=900 if smoke else LIMITS[stage],
                         max_rss_mb=13000 if stage == "evaluate" else 56000,
                         command=command, cancel_event=cancel)
    if record["status"] != "succeeded":
        raise RuntimeError(f"{stage} task {index}: {record['status']}; see {measurement}/run.json")
    row = checked_result(measurement, stage, seed, smoke, bundle, arm)
    if stage == "train":
        for name in ARMS:
            path = measurement / f"{name}.pkl"
            if sha256_file(path) != row["policy_hashes"][name]:
                raise ValueError(f"Modified trained policy: {path}")
            shutil.copyfile(path, output / path.name)
    for name in ("result.json", "SUCCESS.json"):
        shutil.copyfile(measurement / name, output / name)
    return row


def group(stage, smoke, root, inputs, native_repo, bucket, bundle, progress):
    count = (1 if smoke else 3) * (4 if stage == "evaluate" else 1)
    cancel = threading.Event()
    rows = {}
    progress(stage, 0, count)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {}
        try:
            for index in range(count):
                future = pool.submit(task, stage, index, smoke, root, inputs, native_repo,
                                     bucket, bundle, cancel)
                futures[future] = index
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
                progress(stage, len(rows), count)
        except BaseException:
            cancel.set()
            for future in futures:
                future.cancel()
            raise
    return rows


def aggregate_stage(smoke, root, inputs, bundle):
    count = 4 if smoke else 12
    staged = inputs / "aggregation"
    staged.mkdir(parents=True, exist_ok=False)
    for index in range(count):
        source = root / "evaluate" / f"task_{index}" / "analysis"
        checked_result(source, "evaluate", index // 4, smoke, bundle, ARMS[index % 4])
        destination = staged / f"task_{index}"
        destination.mkdir()
        shutil.copyfile(source / "result.json", destination / "result.json")
    diagnostics = root / "aggregation_diagnostics"
    command = [sys.executable, "-m", "fhp_evaluation.adversarial.pipeline", "aggregate-worker",
               "--input", str(staged), "--output", str(root / "analysis"),
               "--result", str(diagnostics / "result.json")]
    if smoke:
        command.append("--smoke")
    record = run_bounded(dict(mode="exp21", phase="aggregate", smoke=smoke, threads=2),
                         diagnostics, max_seconds=LIMITS["aggregate"], max_rss_mb=13000,
                         command=command)
    if record["status"] != "succeeded":
        raise RuntimeError(f"Aggregation failed: {record['status']}")
    report = json.loads((root / "analysis" / "summary.json").read_text())
    marker = json.loads((root / "analysis" / "SUCCESS.json").read_text())
    expected = "smoke_complete" if smoke else "complete"
    if any(row.get("status") != expected or row.get("bundle_sha256") != bundle
           for row in (report, marker)):
        raise ValueError("Invalid aggregate result or source bundle")


def run(output, inputs, native_repo, bucket, *, smoke_only=False):
    output, inputs, native_repo = map(lambda p: Path(p).resolve(), (output, inputs, native_repo))
    bundle = os.environ.get("FHP_BUNDLE_SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle or ""):
        raise ValueError("An immutable FHP_BUNDLE_SHA256 is required")
    if any((output / name).exists() for name in
           ("STATUS.json", "smoke", "qualify", "train", "evaluate", "analysis", "aggregation_diagnostics")):
        raise FileExistsError("Pipeline output is occupied; no automatic resume or overwrite")
    started = time.monotonic()
    state = dict(status="running", complete=False, scientific_complete=False,
                 smoke_only=smoke_only, bundle_sha256=bundle, completed_stages=[])

    def progress(stage, completed, total):
        state.update(phase=stage, completed_tasks=completed, total_tasks=total,
                     elapsed_seconds=time.monotonic() - started)
        write_json(output / "STATUS.json", state)
        print(json.dumps(state), flush=True)

    try:
        for smoke in ([True] if smoke_only else [True, False]):
            state["mode"] = "smoke" if smoke else "full"
            root = output / "smoke" if smoke else output
            source = inputs / state["mode"]
            for stage in ("qualify", "train", "evaluate"):
                rows = group(stage, smoke, root, source, native_repo, bucket, bundle, progress)
                state["completed_stages"].append(f"{state['mode']}/{stage}")
                if stage == "qualify" and not smoke:
                    rejected = [seed for seed, row in rows.items() if row.get("gate_passed") is not True]
                    if rejected:
                        raise QualificationStopped(f"Attacker qualification inconclusive for seeds {sorted(rejected)}; "
                                                   "no full training or evaluation started")
            progress("aggregate", 0, 1)
            aggregate_stage(smoke, root, source, bundle)
            state["completed_stages"].append(f"{state['mode']}/aggregate")
            progress("aggregate", 1, 1)
        state.update(status="complete", complete=True, scientific_complete=not smoke_only)
        progress("complete", 1, 1)
    except BaseException as error:
        state.update(status="inconclusive_qualification" if isinstance(error, QualificationStopped) else "failed",
                     error=f"{type(error).__name__}: {error}")
        progress(state.get("phase", "initialization"), state.get("completed_tasks", 0),
                 state.get("total_tasks", 0))
        raise
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    whole = commands.add_parser("run")
    for name in ("output", "inputs", "native-repo"):
        whole.add_argument("--" + name, type=Path, required=True)
    whole.add_argument("--bucket", required=True)
    whole.add_argument("--smoke-only", action="store_true")
    worker = commands.add_parser("aggregate-worker")
    for name in ("input", "output", "result"):
        worker.add_argument("--" + name, type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "aggregate-worker":
        from .aggregate import aggregate
        report = aggregate(args.input, args.output, args.smoke)
        write_json(args.result, dict(status=report["status"], bundle_sha256=report["bundle_sha256"]))
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Pipeline received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        run(args.output, args.inputs, args.native_repo, args.bucket, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()
