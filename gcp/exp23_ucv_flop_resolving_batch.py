#!/usr/bin/env python3
"""Source-pinned, single-VM UCV Experiment 23 launcher (stdlib-only)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
configuration_spec = importlib.util.spec_from_file_location("exp23_config", ROOT / "fhp_evaluation/resolving/experiment.py")
configuration = importlib.util.module_from_spec(configuration_spec)
configuration_spec.loader.exec_module(configuration)
config, checkpoint_registry = configuration.config, configuration.checkpoint_registry

spec = importlib.util.spec_from_file_location("bundle_common", ROOT / "gcp/exp5_ucv_exp9_br_production_batch.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
EXTRA_FILES = ("gcp/exp23_ucv_flop_resolving_batch.py", "gcp/run_exp23_ucv_flop_resolving.sh",
               "gcp/finalize_exp23_ucv_flop_resolving.sh")


def validate(args):
    if not re.fullmatch(r"exp23-search-[a-z0-9][a-z0-9-]{0,28}", args.run_id or ""):
        raise ValueError("Use a fresh exp23-search-* RUN_ID (at most 42 characters)")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be a bucket, without subdirectories")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com", args.service_account or ""):
        raise ValueError("Set SA_EMAIL to an existing Batch runner service account")
    if not (args.native_repo / "fhp_escher/checkpointing.py").is_file():
        raise ValueError("Set --native-repo to the UCV checkout")


def build_job(args, bundle_hash):
    destination = f"{args.bucket}/{args.run_id}"
    env = dict(FHP_BUNDLE_URI=f"{destination}/inputs/source.tar.gz", FHP_BUNDLE_SHA256=bundle_hash,
               FHP_DESTINATION=destination, FHP_SOURCE_BUCKET=args.bucket,
               PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="")
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1, taskSpec=dict(
        computeResource=dict(cpuMilli=16000, memoryMib=56000), maxRetryCount=0,
        maxRunDuration="172800s", environment=dict(variables=env), runnables=[
            dict(displayName="paired-search-gameplay", timeout="171600s",
                 script=dict(text=(ROOT / EXTRA_FILES[1]).read_text())),
            dict(displayName="final-upload", alwaysRun=True, timeout="900s",
                 script=dict(text=(ROOT / EXTRA_FILES[2]).read_text()))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account), instances=[dict(policy=dict(
            machineType="n2-standard-16", provisioningModel="STANDARD",
            bootDisk=dict(sizeGb=100, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(workload="ucv-exp23-search"))


def read(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=180).stdout


def preflight(args, request, job_id):
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    rows = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                           "--location", args.region, "--filter", f"name:{args.run_id}-", "--format=json"))
    if not isinstance(rows, list):
        raise ValueError("Invalid Batch listing")
    jobs = [r for r in rows if r.get("name", "").rsplit("/", 1)[-1].startswith(args.run_id + "-")]
    if any(r.get("status", {}).get("state") not in ("SUCCEEDED", "FAILED") for r in jobs):
        raise ValueError("Another attempt is active/unknown; refusing concurrent writers")
    if any(r["name"].rsplit("/", 1)[-1] == job_id for r in jobs):
        raise ValueError("Job name already exists")
    objects = json.loads(read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                              "--exhaustive", "--limit=1", "--format=json"))
    recovery = args.action == "recover"
    if not isinstance(objects, list) or (objects and not recovery) or (recovery and not objects):
        raise ValueError("New runs require an empty namespace; recovery requires an existing run")
    if recovery:
        old = json.loads(read("gcloud", "storage", "cat", f"{args.bucket}/{args.run_id}/inputs/request.json"))
        if old != request or any(r.get("status", {}).get("state") == "SUCCEEDED" for r in jobs):
            raise ValueError("Recovery must match the immutable failed run, not a completed run")
    for record in checkpoint_registry().values():
        uri = f"{args.bucket}/{record['run']}/{record['relative']}"
        meta = json.loads(read("gcloud", "storage", "objects", "describe", uri, "--format=json"))
        if not 0 < int(meta.get("size", meta.get("sizeBytes", 0))) <= 256 * 1024**2:
            raise ValueError(f"Missing/oversized playable checkpoint: {uri}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "recover"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--native-repo", type=Path, default=ROOT.parent / "fhp_ucv_escher/fhp-ucv-escher-experiments")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--recovery-tag")
    args = parser.parse_args()
    validate(args)
    directory = args.run_dir or ROOT / "outputs/batch" / args.run_id
    fields = ("project", "region", "bucket", "service_account", "run_id")
    if args.action == "prepare":
        directory.mkdir(parents=True, exist_ok=False)
        source = common.make_bundle(ROOT, args.native_repo, directory / "source.tar.gz", extra_files=EXTRA_FILES)
        bundle_hash = common.sha256(directory / "source.tar.gz")
        (directory / "job.json").write_text(json.dumps(build_job(args, bundle_hash), indent=2) + "\n")
        request = {f: getattr(args, f) for f in fields}
        request.update(source=source, experiment=config(), bundle_sha256=bundle_hash,
                       prepared_files_sha256={p: common.sha256(directory / p) for p in ("source.tar.gz", "job.json")})
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {directory}; no cloud calls or paid jobs. Source is bundled by hash, not REPO_REF.")
        return
    request = json.loads((directory / "request.json").read_text())
    if any(request[f] != getattr(args, f) for f in fields):
        raise ValueError("Arguments differ from the prepared request")
    for name, sha in request["prepared_files_sha256"].items():
        if common.sha256(directory / name) != sha:
            raise ValueError(f"Prepared input changed: {name}")
    if args.action == "recover":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,9}", args.recovery_tag or ""):
            raise ValueError("Recovery needs a new --recovery-tag of 1-10 lowercase characters")
        suffix = "r-" + args.recovery_tag
    else:
        if args.recovery_tag:
            raise ValueError("Recovery tags only apply to recover")
        suffix = "evaluate"
    job_id = args.run_id + "-" + suffix
    preflight(args, request, job_id)
    if args.action == "submit":
        for name in ("source.tar.gz", "job.json", "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(directory / name),
                            f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", job_id, "--project", args.project,
                    "--location", args.region, "--config", str(directory / "job.json")], check=True, timeout=180)
    print(f"Submitted {job_id}. One VM runs checks then gameplay automatically. No training.")


if __name__ == "__main__":
    main()
