#!/usr/bin/env python3
"""Prepare/submit only the bounded SD6 execution benchmark, never the full audit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sd_audit_batch", ROOT / "gcp/sd_exp6_strategic_audit_batch.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE_RUN = "fhp-sd6-audit-dev-20261013-104108"
FIELDS = ("run_id", "project", "region", "bucket", "service_account")


def check_plan(plan):
    # This preparation path deliberately needs only stdlib, not OpenSpiel/Torch.
    if (plan.get("experiment") != "sd6_execution_benchmark_v1" or plan.get("source_run") != SOURCE_RUN
            or plan.get("allocations") != [[4, 4], [8, 2], [8, 1]]
            or plan.get("repeats") != 2 or plan.get("cell_seconds") != 10800
            or len(plan.get("boards", [])) != 4
            or not re.fullmatch(r"[0-9a-f]{64}", plan.get("progress_sha256", ""))):
        raise ValueError("Unexpected benchmark plan")


def build_job(args, bundle_hash, plan_hash):
    scripts = {name: (ROOT / name).read_text() for name in base.EXTRA_FILES[1:]}
    name = base.EXTRA_FILES[1]
    original = scripts[name]
    cut = original.index('"$WORK/venv/bin/python" -m fhp_evaluation.strategic_audit.sd_exp6 ')
    scripts[name] = original[:cut] + '''timeout 180 gcloud storage cp "$FHP_DESTINATION/inputs/plan.json" "$WORK/plan.json"
printf '%s  %s\\n' "$FHP_PLAN_SHA256" "$WORK/plan.json" | sha256sum -c -
"$WORK/venv/bin/python" -m fhp_evaluation.strategic_audit.sd_exp6_benchmark run \\
  --plan "$WORK/plan.json" --input-root "$WORK/input" --native-root "$WORK/source/native" \\
  --output "$OUTPUT" --download
'''
    job = base.build_job(args, bundle_hash, scripts)
    task = job["taskGroups"][0]["taskSpec"]
    task["maxRunDuration"] = "72000s"  # 20h, including setup/transfers/final upload.
    task["runnables"][0]["timeout"] = "70800s"
    task["runnables"][0]["displayName"] = "sd6-execution-benchmark"
    task["environment"]["variables"]["FHP_PLAN_SHA256"] = plan_hash
    job["labels"] = dict(workload="fhp-sd6-benchmark", stage="timing")
    return job


def preflight(args, request, directory):
    base.read("gcloud", "iam", "service-accounts", "describe", args.service_account,
              "--project", args.project, "--format=value(email)")
    if base.list_jobs(args, args.run_id + "-"):
        raise ValueError("Benchmark job already exists; never resubmit paid measurements")
    objects = json.loads(base.read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                                  "--exhaustive", "--limit=1", "--format=json"))
    if not isinstance(objects, list) or objects:
        raise ValueError("Benchmark requires a new, empty output prefix")
    plan = json.loads((directory / "plan.json").read_text())
    check_plan(plan)
    progress = base.read("gcloud", "storage", "cat", f"{args.bucket}/{SOURCE_RUN}/export_progress.json")
    if hashlib.sha256(progress.encode()).hexdigest() != plan["progress_sha256"]:
        raise ValueError("Source timings changed; regenerate and review plan")
    # Verify pinned full-history manifests before paying for a VM.
    for record in base.experiment.checkpoint_registry("pilot").values():
        raw = base.read("gcloud", "storage", "cat",
                        f"{base.experiment.SOURCE_BUCKET}/{record['run']}/{record['relative']}")
        if hashlib.sha256(raw.encode()).hexdigest() != record["sha256"]:
            raise ValueError("Checkpoint manifest checksum mismatch")
        base.experiment.archive_chunks(json.loads(raw), record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "submit"))
    p.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    p.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    p.add_argument("--region", default=os.environ.get("REGION"))
    p.add_argument("--bucket", default=os.environ.get("BUCKET"))
    p.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    p.add_argument("--native-repo", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--run-dir", type=Path)
    a = p.parse_args()
    if not (a.run_id or "").startswith("fhp-sd6-audit-bench-"):
        p.error("Use a separate fhp-sd6-audit-bench- run ID")
    a.stage, a.previous_run_id, a.previous_reviewed, a.skip_pilot = "profile", None, False, False
    base.validate(a, preparing=a.action == "prepare")
    if a.bucket != base.experiment.SOURCE_BUCKET:
        raise ValueError("Use the SD-CFR results bucket")
    directory = (a.run_dir or ROOT / "outputs/batch" / a.run_id).resolve()
    if a.action == "prepare":
        if a.plan is None:
            p.error("prepare requires --plan")
        plan_bytes = a.plan.read_bytes()
        check_plan(json.loads(plan_bytes))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "plan.json").write_bytes(plan_bytes)
        source = base.common.make_bundle(ROOT, a.native_repo, directory / "source.tar.gz",
            extra_files=(*base.EXTRA_FILES, "gcp/sd_exp6_execution_benchmark_batch.py"),
            native_packages=("deep_cfr_poker",))
        job = build_job(a, base.common.sha256(directory / "source.tar.gz"),
                       base.common.sha256(directory / "plan.json"))
        (directory / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        request = {key: getattr(a, key) for key in FIELDS}
        request.update(source=source, prepared_files_sha256={n: base.common.sha256(directory / n)
                       for n in ("source.tar.gz", "plan.json", "job.json")})
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {directory}. No cloud writes; full audit remains stopped.")
        return
    request = json.loads((directory / "request.json").read_text())
    if any(request[key] != getattr(a, key) for key in FIELDS):
        raise ValueError("Submission differs from prepared request")
    if any(base.common.sha256(directory / n) != h for n, h in request["prepared_files_sha256"].items()):
        raise ValueError("Prepared benchmark files changed")
    preflight(a, request, directory)
    for name in ("source.tar.gz", "plan.json", "request.json", "job.json"):
        subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(directory / name),
                        f"{a.bucket}/{a.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", a.run_id + "-benchmark", "--project", a.project,
                    "--location", a.region, "--config", str(directory / "job.json")], check=True, timeout=180)
    print(f"Submitted {a.run_id}-benchmark; no automatic retries or full-audit continuation.")


if __name__ == "__main__":
    main()
