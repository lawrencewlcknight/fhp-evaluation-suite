#!/usr/bin/env python3
"""Prepare/submit a gated SD-CFR Exp6 audit. Prepare is offline and stdlib-only."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUN_PATTERN = r"fhp-sd6-audit-[a-z0-9](?:[a-z0-9-]{0,26}[a-z0-9])?"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("sd_audit_bundle", ROOT / "gcp/exp5_ucv_exp9_br_production_batch.py")
experiment = load_module("sd_audit_config", ROOT / "fhp_evaluation/strategic_audit/sd_exp6.py")
EXTRA_FILES = ("gcp/sd_exp6_strategic_audit_batch.py", "gcp/run_sd_exp6_strategic_audit.sh",
               "gcp/finalize_sd_exp6_strategic_audit.sh")
REQUEST_FIELDS = ("run_id", "project", "region", "bucket", "service_account", "stage",
                  "previous_run_id", "previous_reviewed", "skip_pilot")


def validate(args, *, preparing=False):
    if args.skip_pilot and args.stage != "development":
        raise ValueError("--skip-pilot is only valid for development after a reviewed profile")
    if not re.fullmatch(RUN_PATTERN, args.run_id or ""):
        raise ValueError("RUN_ID must start fhp-sd6-audit- and contain at most 41 lowercase characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be one output bucket without a prefix")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"(?:[a-z0-9-]+@[a-z0-9-]+\.iam|[0-9]+-compute@developer)\.gserviceaccount\.com",
                        args.service_account or ""):
        raise ValueError("Set SA_EMAIL")
    if preparing and (not args.native_repo or
                      not (Path(args.native_repo) / "deep_cfr_poker/__init__.py").is_file()):
        raise ValueError("Provide --native-repo pointing to the SD-CFR checkout")
    experiment.checkpoint_registry(args.stage)
    if args.stage == "profile":
        if args.previous_run_id or args.previous_reviewed:
            raise ValueError("The initial profile has no previous-stage review flags")
    elif (not args.previous_reviewed or not re.fullmatch(RUN_PATTERN, args.previous_run_id or "") or
          args.previous_run_id == args.run_id):
        raise ValueError("Next stage requires a distinct --previous-run-id and explicit --previous-reviewed")


def frozen_profile(args):
    """Reuse the profiled payload verbatim; never bundle unrelated current edits.

    Prepare is still offline. Submit authenticates this exact request and bundle
    against the successful remote profile before any upload or paid submission.
    """
    directory = args.profile_run_dir or ROOT / "outputs/batch" / args.previous_run_id
    request_bytes = (directory / "request.json").read_bytes()
    request = json.loads(request_bytes)
    if (request.get("experiment") != experiment.EXPERIMENT or request.get("stage") != "profile" or
            request.get("run_id") != args.previous_run_id):
        raise ValueError("--skip-pilot requires the prepared inputs of the reviewed profile")
    for name in ("project", "region", "bucket", "service_account"):
        if request.get(name) != getattr(args, name):
            raise ValueError(f"Profile and development differ: {name}")
    bundle = (directory / "source.tar.gz").read_bytes()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    if (bundle_hash != request.get("bundle_sha256") or
            bundle_hash != request.get("prepared_files_sha256", {}).get("source.tar.gz")):
        raise ValueError("Frozen profile bundle failed its checksum")
    source = request["source"]
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = archive.getnames()
        if (len(names) != len(set(names)) or set(names) != set(source["files"]) | {"source_manifest.json"} or
                any(not member.isfile() for member in archive.getmembers())):
            raise ValueError("Unexpected frozen profile bundle contents")
        for name, digest in source["files"].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError(f"Frozen profile source checksum failed: {name}")
        if json.loads(archive.extractfile("source_manifest.json").read()) != source:
            raise ValueError("Frozen profile source manifest differs from its request")
        scripts = {name: archive.extractfile("evaluator/" + name).read().decode()
                   for name in EXTRA_FILES[1:]}
    # The local launcher must still describe the frozen checkpoint/resource contract.
    config = "fhp_evaluation/strategic_audit/sd_exp6.py"
    if source["files"].get("evaluator/" + config) != common.sha256(ROOT / config):
        raise ValueError("Audit configuration changed since profiling; review before direct development")
    return bundle, source, scripts, hashlib.sha256(request_bytes).hexdigest()


def build_job(args, bundle_hash, scripts=None):
    scripts = scripts or {name: (ROOT / name).read_text() for name in EXTRA_FILES[1:]}
    duration = experiment.JOB_HOURS[args.stage] * 3600
    destination = f"{args.bucket}/{args.run_id}"
    env = dict(FHP_BUNDLE_URI=f"{destination}/inputs/source.tar.gz", FHP_BUNDLE_SHA256=bundle_hash,
               FHP_DESTINATION=destination, FHP_STAGE=args.stage, PYTHONUNBUFFERED="1",
               PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1, taskSpec=dict(
        computeResource=dict(cpuMilli=16000, memoryMib=56000), maxRetryCount=0,
        maxRunDuration=f"{duration}s", environment=dict(variables=env), runnables=[
            dict(displayName="sd-exp6-audit", timeout=f"{duration - 1200}s",
                 script=dict(text=scripts[EXTRA_FILES[1]])),
            dict(displayName="final-upload", alwaysRun=True, timeout="900s",
                 script=dict(text=scripts[EXTRA_FILES[2]]))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account), instances=[dict(policy=dict(
            machineType="n2-standard-16", provisioningModel="STANDARD",
            bootDisk=dict(sizeGb=200, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="fhp-sd6-audit", stage=args.stage))


def read(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout


def list_jobs(args, prefix):
    rows = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                           "--location", args.region, "--filter", f"name:{prefix}", "--format=json"))
    if not isinstance(rows, list):
        raise ValueError("Invalid Batch listing")
    return [r for r in rows if r.get("name", "").rsplit("/", 1)[-1].startswith(prefix)]


def verify_previous(args, request):
    if args.stage == "profile":
        return
    previous_stage = "profile" if args.stage == "pilot" or args.skip_pilot else "pilot"
    prefix = f"{args.bucket}/{args.previous_run_id}"
    success = json.loads(read("gcloud", "storage", "cat", prefix + "/analysis/SUCCESS.json"))
    source_text = read("gcloud", "storage", "cat", prefix + "/inputs/request.json")
    source = json.loads(source_text)
    if args.skip_pilot and (
            hashlib.sha256(source_text.encode()).hexdigest() != request.get("profile_request_sha256") or
            source.get("bundle_sha256") != request.get("bundle_sha256")):
        raise ValueError("Direct development must reuse the exact successful profile request and bundle")
    checkpoints = experiment.checkpoint_registry(previous_stage)
    if (success.get("experiment") != experiment.EXPERIMENT or success.get("status") != "succeeded" or
            success.get("stage") != previous_stage or success.get("full_archive") is not True or
            success.get("assessment_evaluated") is not False or source.get("stage") != previous_stage or
            success.get("checkpoint_sha256") != {k: v["sha256"] for k, v in checkpoints.items()}):
        raise ValueError(f"A completed full-archive {previous_stage} of these checkpoints is required")
    if source.get("source", {}).get("files") != request["source"]["files"]:
        raise ValueError("Source changed since the previous stage; rerun the staged feasibility checks")
    runtime_text = read("gcloud", "storage", "cat", prefix + "/analysis/runtime_summary.json")
    if hashlib.sha256(runtime_text.encode()).hexdigest() != success.get("report_sha256", {}).get("runtime_summary.json"):
        raise ValueError("Previous runtime report failed its checksum")
    runtime = json.loads(runtime_text)
    rows = runtime.get("full_archive_exports", [])
    if (runtime.get("experiment") != experiment.EXPERIMENT or runtime.get("stage") != previous_stage or
            runtime.get("execution") != experiment.execution_config(previous_stage) or
            len(rows) != len(checkpoints) * experiment.BOARD_COUNTS[previous_stage] or
            len({(r["policy"], tuple(r["board"])) for r in rows}) != len(rows) or
            any(sum(r["policy"] == label for r in rows) != experiment.BOARD_COUNTS[previous_stage]
                for label in checkpoints) or
            any(r["policy"] not in checkpoints or
                r["models_per_player"] != checkpoints[r["policy"]]["completed_iterations"] for r in rows)):
        raise ValueError("Previous stage did not verify the complete archive workload")
    prior_jobs = list_jobs(args, args.previous_run_id + "-")
    if (not any(r.get("status", {}).get("state") == "SUCCEEDED" for r in prior_jobs) or
            any(r.get("status", {}).get("state") not in ("SUCCEEDED", "FAILED") for r in prior_jobs)):
        raise ValueError("Previous Batch stage must have finished successfully, including final upload")


def cloud_preflight(args, request, job_id, *, recovery=False):
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
    for record in experiment.checkpoint_registry(args.stage).values():
        uri = f"{experiment.SOURCE_BUCKET}/{record['run']}/{record['relative']}"
        meta = json.loads(read("gcloud", "storage", "objects", "describe", uri, "--format=json"))
        if not 0 < int(meta.get("size", meta.get("sizeBytes", 0))) <= 4 * 1024**2:
            raise ValueError(f"Unexpected archive manifest size: {uri}")
        raw = read("gcloud", "storage", "cat", uri)
        if hashlib.sha256(raw.encode()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source manifest checksum mismatch: {uri}")
        experiment.archive_chunks(json.loads(raw), record)
    verify_previous(args, request)


def verify_request(args, directory):
    request = json.loads((directory / "request.json").read_text())
    for name in REQUEST_FIELDS:
        value = request.get(name, False) if name == "skip_pilot" else request[name]
        if value != getattr(args, name):
            raise ValueError(f"Arguments differ from prepared request: {name}")
    for name in ("source.tar.gz", "job.json"):
        if common.sha256(directory / name) != request["prepared_files_sha256"][name]:
            raise ValueError(f"Prepared file changed: {name}")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "recover"))
    parser.add_argument("--stage", choices=experiment.STAGES, default="profile")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--native-repo", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--previous-run-id")
    parser.add_argument("--previous-reviewed", action="store_true")
    parser.add_argument("--skip-pilot", action="store_true",
                        help="Explicitly proceed profile -> development using the exact profiled source bundle")
    parser.add_argument("--profile-run-dir", type=Path,
                        help="Local prepared profile directory for --skip-pilot prepare; default outputs/batch/<previous-run-id>")
    parser.add_argument("--recovery-tag")
    args = parser.parse_args()
    if args.profile_run_dir and (not args.skip_pilot or args.action != "prepare"):
        parser.error("--profile-run-dir is only for prepare --skip-pilot")
    validate(args, preparing=args.action == "prepare")
    directory = (args.run_dir or ROOT / "outputs/batch" / args.run_id).resolve()
    if args.action == "prepare":
        if args.recovery_tag:
            raise ValueError("Recovery tag applies only to recover")
        frozen = frozen_profile(args) if args.skip_pilot else None
        directory.mkdir(parents=True, exist_ok=False)
        if frozen:
            bundle, source, scripts, profile_request_hash = frozen
            (directory / "source.tar.gz").write_bytes(bundle)
        else:
            source = common.make_bundle(ROOT, args.native_repo, directory / "source.tar.gz",
                                        extra_files=EXTRA_FILES, native_packages=("deep_cfr_poker",))
            scripts = None
        bundle_hash = common.sha256(directory / "source.tar.gz")
        (directory / "job.json").write_text(json.dumps(build_job(args, bundle_hash, scripts), indent=2) + "\n")
        request = {key: getattr(args, key) for key in REQUEST_FIELDS}
        request.update(experiment=experiment.EXPERIMENT, source=source, bundle_sha256=bundle_hash,
                       prepared_files_sha256={p: common.sha256(directory / p) for p in ("source.tar.gz", "job.json")})
        if frozen:
            request["profile_request_sha256"] = profile_request_hash
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {args.stage} in {directory}; no cloud calls or paid jobs.")
        if frozen:
            print("Pilot explicitly skipped. Exact profiled source reused; parallel throughput remains unmeasured.")
        return
    request = verify_request(args, directory)
    if args.action == "recover":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,9}", args.recovery_tag or ""):
            raise ValueError("Recovery requires a unique 1-10-character lowercase --recovery-tag")
        suffix = "r-" + args.recovery_tag
    else:
        if args.recovery_tag:
            raise ValueError("Recovery tag applies only to recover")
        suffix = "audit"
    job_id = f"{args.run_id}-{suffix}"
    cloud_preflight(args, request, job_id, recovery=args.action == "recover")
    if args.action == "submit":
        for name in ("source.tar.gz", "job.json", "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(directory / name),
                            f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", job_id, "--project", args.project,
                    "--location", args.region, "--config", str(directory / "job.json")], check=True, timeout=180)
    print(f"Submitted {job_id}. No retries or later stages are launched automatically.")


if __name__ == "__main__":
    main()
