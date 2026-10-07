#!/usr/bin/env python3
"""Prepare/submit a staged, frozen-policy Exp16 strategic audit (stdlib only)."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("br_bundle", ROOT / "gcp/exp5_ucv_exp9_br_production_batch.py")
experiment = load_module("strategic_exp16", ROOT / "fhp_evaluation/strategic_audit/exp16.py")
EXTRA_FILES = ("gcp/exp6_ucv_exp16_strategic_audit_batch.py",
               "gcp/run_exp6_ucv_exp16_strategic_audit.sh",
               "gcp/finalize_exp6_ucv_exp16_strategic_audit.sh")


def validate(args, *, preparing=False):
    if not re.fullmatch(r"fhp-strat16-[a-z0-9][a-z0-9-]{0,29}", args.run_id or ""):
        raise ValueError("RUN_ID must start fhp-strat16- and be at most 42 lowercase characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be one bucket without a prefix")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com", args.service_account or ""):
        raise ValueError("Set SA_EMAIL")
    if preparing and (not args.native_repo or not Path(args.native_repo).is_dir()):
        raise ValueError("Provide --native-repo pointing to the UCV checkout")
    if args.stage == "development":
        if not args.pilot_reviewed or not re.fullmatch(r"fhp-strat16-[a-z0-9][a-z0-9-]{0,29}", args.pilot_run_id or ""):
            raise ValueError("Development requires --pilot-run-id and explicit --pilot-reviewed")
        if args.pilot_run_id == args.run_id:
            raise ValueError("Pilot and development must use separate run IDs")
    elif args.pilot_reviewed or args.pilot_run_id:
        raise ValueError("Pilot review flags apply only to development")


def build_job(args, bundle_hash):
    destination = f"{args.bucket}/{args.run_id}"
    duration = 21600 if args.stage == "pilot" else 86400
    env = dict(FHP_BUNDLE_URI=f"{destination}/inputs/source.tar.gz", FHP_BUNDLE_SHA256=bundle_hash,
               FHP_DESTINATION=destination, FHP_SOURCE_BUCKET=args.bucket, FHP_STAGE=args.stage,
               PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="")
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1, taskSpec=dict(
        computeResource=dict(cpuMilli=16000, memoryMib=56000), maxRetryCount=0,
        maxRunDuration=f"{duration}s", environment=dict(variables=env), runnables=[
            dict(displayName="strategic-audit", timeout=f"{duration - 1200}s",
                 script=dict(text=(ROOT / EXTRA_FILES[1]).read_text())),
            dict(displayName="final-upload", alwaysRun=True, timeout="900s",
                 script=dict(text=(ROOT / EXTRA_FILES[2]).read_text()))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account), instances=[dict(policy=dict(
            machineType="n2-standard-16", provisioningModel="STANDARD",
            bootDisk=dict(sizeGb=100, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="fhp-strat16", stage=args.stage))


def read(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout


def list_jobs(args, prefix):
    rows = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                           "--location", args.region, "--filter", f"name:{prefix}", "--format=json"))
    if not isinstance(rows, list):
        raise ValueError("Invalid Batch listing")
    return [r for r in rows if r.get("name", "").rsplit("/", 1)[-1].startswith(prefix)]


def cloud_preflight(args, request, job_id, recovery=False):
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    jobs = list_jobs(args, args.run_id + "-")
    if any(r.get("status", {}).get("state") not in ("SUCCEEDED", "FAILED") for r in jobs):
        raise ValueError("Another attempt is active or its state is unknown")
    if any(r["name"].rsplit("/", 1)[-1] == job_id for r in jobs):
        raise ValueError("Batch job name already exists")
    objects = json.loads(read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                              "--exhaustive", "--limit=1", "--format=json"))
    if not isinstance(objects, list) or (not recovery and objects) or (recovery and not objects):
        raise ValueError("New runs require an empty namespace; recovery requires an existing namespace")
    if recovery:
        previous = json.loads(read("gcloud", "storage", "cat", f"{args.bucket}/{args.run_id}/inputs/request.json"))
        if previous != request:
            raise ValueError("Recovery differs from the immutable submitted request")
        if any(r.get("status", {}).get("state") == "SUCCEEDED" for r in jobs):
            raise ValueError("Audit already succeeded; do not submit redundant recovery")
    for item in experiment.checkpoint_registry(args.stage).values():
        uri = f"{args.bucket}/{item['run']}/{item['relative']}"
        meta = json.loads(read("gcloud", "storage", "objects", "describe", uri, "--format=json"))
        size = meta.get("size", meta.get("sizeBytes", 0))
        if not 0 < int(size) <= 256 * 1024**2:
            raise ValueError(f"Unexpected playable-checkpoint size: {uri}")
    if args.stage == "development":
        prefix = f"{args.bucket}/{args.pilot_run_id}"
        success = json.loads(read("gcloud", "storage", "cat", prefix + "/analysis/SUCCESS.json"))
        source = json.loads(read("gcloud", "storage", "cat", prefix + "/inputs/request.json"))
        if success.get("status") != "succeeded" or success.get("stage") != "pilot":
            raise ValueError("A completed real-model pilot is required")
        if success.get("checkpoint_sha256") != {k: v["sha256"] for k, v in experiment.checkpoint_registry("pilot").items()}:
            raise ValueError("Pilot used different checkpoints")
        if source["source"]["files"] != request["source"]["files"]:
            raise ValueError("Source changed since pilot; rerun the pilot with this implementation")
        prior_jobs = list_jobs(args, args.pilot_run_id + "-")
        if not any(r.get("status", {}).get("state") == "SUCCEEDED" for r in prior_jobs):
            raise ValueError("Pilot Batch job must have succeeded, including final upload")


def verify_request(args, run_dir):
    request = json.loads((run_dir / "request.json").read_text())
    for name in ("run_id", "project", "region", "bucket", "service_account", "stage", "pilot_run_id", "pilot_reviewed"):
        if request[name] != getattr(args, name):
            raise ValueError(f"Arguments differ from prepared request: {name}")
    for name in ("source.tar.gz", "job.json"):
        if common.sha256(run_dir / name) != request["prepared_files_sha256"][name]:
            raise ValueError(f"Prepared file changed: {name}")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "recover"))
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--native-repo", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pilot-run-id")
    parser.add_argument("--pilot-reviewed", action="store_true")
    parser.add_argument("--recovery-tag")
    args = parser.parse_args()
    validate(args, preparing=args.action == "prepare")
    directory = (args.run_dir or ROOT / "outputs/batch" / args.run_id).resolve()
    if args.action == "prepare":
        directory.mkdir(parents=True, exist_ok=False)
        source = common.make_bundle(ROOT, args.native_repo, directory / "source.tar.gz", extra_files=EXTRA_FILES)
        bundle_hash = common.sha256(directory / "source.tar.gz")
        (directory / "job.json").write_text(json.dumps(build_job(args, bundle_hash), indent=2) + "\n")
        request = {key: getattr(args, key) for key in (
            "run_id", "project", "region", "bucket", "service_account", "stage", "pilot_run_id", "pilot_reviewed")}
        request.update(source=source, bundle_sha256=bundle_hash,
                       prepared_files_sha256={p: common.sha256(directory / p) for p in ("source.tar.gz", "job.json")})
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {args.stage} in {directory}; no cloud calls or paid jobs.")
        return
    request = verify_request(args, directory)
    if args.action == "recover":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,9}", args.recovery_tag or ""):
            raise ValueError("Recovery needs a unique 1-10-character lowercase --recovery-tag")
        suffix = "r-" + args.recovery_tag
    else:
        if args.recovery_tag:
            raise ValueError("Recovery tag only applies to recover")
        suffix = "audit"
    job_id = f"{args.run_id}-{suffix}"
    cloud_preflight(args, request, job_id, recovery=args.action == "recover")
    if args.action == "submit":
        for name in ("source.tar.gz", "job.json", "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(directory / name),
                            f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", job_id, "--project", args.project,
                    "--location", args.region, "--config", str(directory / "job.json")], check=True, timeout=180)
    print(f"Submitted {job_id}. This does not launch any subsequent stage.")


if __name__ == "__main__":
    main()
