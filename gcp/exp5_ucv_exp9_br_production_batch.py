#!/usr/bin/env python3
"""Prepare and submit the three-seed Experiment 9 response evaluation."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = "exp9-cache24-20261001-132550"
CHECKPOINTS = {
    0: ("workers/task_000_cached_parallel_structured_ucv_escher_seed_0/checkpoints/"
        "cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl",
        "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449"),
    1: ("workers/task_001_cached_parallel_structured_ucv_escher_seed_1/checkpoints/"
        "cached_parallel_structured_ucv_escher_seed_1_time_24h.pkl",
        "ef5f0db716bd2652cd06817ff0b4d4b6e5d1307c5063b982c69b768d1b3b7caa"),
    2: ("workers/task_002_cached_parallel_structured_ucv_escher_seed_2/checkpoints/"
        "cached_parallel_structured_ucv_escher_seed_2_time_24h.pkl",
        "d678f805ad9c9665545ba91164a9129c5bb60fab09c90b8478cbcaa35b9def7b"),
}
TASK_COUNT = 12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(root):
    def read(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": read("rev-parse", "HEAD"), "dirty": bool(read("status", "--porcelain"))}


def make_bundle(evaluation_root, native_root, destination, *, extra_files=()):
    evaluation_root, native_root = Path(evaluation_root).resolve(), Path(native_root).resolve()
    candidates = {}
    for source_root, prefix, packages in ((evaluation_root, "evaluator", ("fhp_evaluation",)),
                                          (native_root, "native", ("fhp_escher", "vr_deep_cfr"))):
        for package in packages:
            package_root = source_root / package
            if package_root.is_symlink() or not (package_root / "__init__.py").is_file():
                raise ValueError(f"Missing or symlinked source package: {package_root}")
            for path in sorted(package_root.rglob("*.py")):
                candidates[f"{prefix}/{path.relative_to(source_root).as_posix()}"] = path
    for name in ("pyproject.toml", "gcp/exp5_ucv_exp9_br_production_batch.py",
                 "gcp/run_exp5_ucv_exp9_br_production.sh",
                 "gcp/finalize_exp5_ucv_exp9_br_production.sh",
                 "gcp/aggregate_exp5_ucv_exp9_br_production.sh",
                 "gcp/requirements-br-pilot.txt", *extra_files):
        candidates[f"evaluator/{name}"] = evaluation_root / name
    files, size = {}, 0
    for name, path in candidates.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing missing or symlinked source: {path}")
        if path.stat().st_size > 4 * 1024**2:
            raise ValueError(f"Unexpectedly large source file: {path}")
        content = path.read_bytes()
        size += len(content)
        if size > 16 * 1024**2:
            raise ValueError("Source bundle exceeds 16 MiB")
        files[name] = content
    manifest = {"schema_version": 1, "source_only": True,
                "evaluator_git": git_provenance(evaluation_root), "native_git": git_provenance(native_root),
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}}
    files["source_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with Path(destination).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
    return manifest


def validate(args, *, preparing=False):
    if not re.fullmatch(r"fhp-br-exp9-prod-[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?", args.run_id or ""):
        raise ValueError("RUN_ID must start fhp-br-exp9-prod- and be at most 51 lowercase characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be one bucket without a prefix")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"(?:[a-z0-9-]+@[a-z0-9-]+\.iam|[0-9]+-compute@developer)\.gserviceaccount\.com",
                        args.service_account or ""):
        raise ValueError("Set SA_EMAIL")
    if preparing and (not args.native_repo or not Path(args.native_repo).is_dir()):
        raise ValueError("--native-repo must identify the UCV training checkout")


def variables(args, bundle_sha256):
    destination = f"{args.bucket}/{args.run_id}"
    result = {"FHP_BUNDLE_URI": f"{destination}/inputs/source.tar.gz",
              "FHP_BUNDLE_SHA256": bundle_sha256, "FHP_DESTINATION": destination,
              "FHP_SMOKE": "1" if args.smoke else "0", "PYTHONUNBUFFERED": "1",
              "PYTHONFAULTHANDLER": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    for seed, (relative, digest) in CHECKPOINTS.items():
        result[f"FHP_CHECKPOINT_URI_{seed}"] = f"{args.bucket}/{SOURCE_RUN}/{relative}"
        result[f"FHP_CHECKPOINT_SHA256_{seed}"] = digest
    return result


def build_job(args, bundle_sha256, stage):
    env = variables(args, bundle_sha256)
    if stage == "workers":
        group = {"taskCount": TASK_COUNT, "parallelism": TASK_COUNT, "taskCountPerNode": 1,
                 "taskSpec": {"computeResource": {"cpuMilli": 2000, "memoryMib": 7000},
                              "maxRetryCount": 0, "maxRunDuration": "14400s",
                              "environment": {"variables": env}, "runnables": [
                                  {"displayName": "production-shard", "timeout": "13200s",
                                   "script": {"text": (ROOT / "gcp/run_exp5_ucv_exp9_br_production.sh").read_text()}},
                                  {"displayName": "final-upload", "alwaysRun": True, "timeout": "900s",
                                   "script": {"text": (ROOT / "gcp/finalize_exp5_ucv_exp9_br_production.sh").read_text()}},
                              ]}}
        duration, machine, disk = "workers", "n2-standard-2", 30
    elif stage == "aggregate":
        group = {"taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
                 "taskSpec": {"computeResource": {"cpuMilli": 2000, "memoryMib": 7000},
                              "maxRetryCount": 0, "maxRunDuration": "3600s",
                              "environment": {"variables": env}, "runnables": [
                                  {"displayName": "aggregate", "timeout": "3300s",
                                   "script": {"text": (ROOT / "gcp/aggregate_exp5_ucv_exp9_br_production.sh").read_text()}},
                              ]}}
        duration, machine, disk = "aggregate", "n2-standard-2", 30
    else:
        raise ValueError(stage)
    return {"taskGroups": [group],
            "allocationPolicy": {"serviceAccount": {"email": args.service_account},
                                 "instances": [{"policy": {"machineType": machine,
                                    "provisioningModel": "STANDARD",
                                    "bootDisk": {"sizeGb": disk, "type": "pd-balanced"}}}]},
            "logsPolicy": {"destination": "CLOUD_LOGGING"},
            "labels": {"workload": "fhp-br-exp9-prod", "stage": duration,
                       "mode": "smoke" if args.smoke else "production"}}


def _read(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout


def cloud_preflight(args, stage, job_id, *, recovery=False):
    _read("gcloud", "iam", "service-accounts", "describe", args.service_account,
          "--project", args.project, "--format=value(email)")
    if recovery or stage == "aggregate":
        attempts = json.loads(_read("gcloud", "batch", "jobs", "list", "--project", args.project,
                                    "--location", args.region, "--filter", f"name:{args.run_id}-", "--format=json"))
        if not isinstance(attempts, list):
            raise ValueError("Invalid Batch attempt listing")
        run_prefix = f"projects/{args.project}/locations/{args.region}/jobs/{args.run_id}-"
        attempts = [job for job in attempts if str(job.get("name", "")).startswith(run_prefix)]
        if any(job.get("status", {}).get("state") not in ("SUCCEEDED", "FAILED") for job in attempts):
            raise ValueError("A prior attempt is active or has an unknown state; refusing concurrent output writes")
        if stage == "aggregate" and not any(
                job.get("status", {}).get("state") == "SUCCEEDED"
                and (job["name"] == run_prefix + "workers" or job["name"].startswith(run_prefix + "r-"))
                for job in attempts):
            raise ValueError("Aggregation requires a successful workers/recovery Batch job including final uploads")
    if stage == "workers":
        for relative, _ in CHECKPOINTS.values():
            metadata = json.loads(_read("gcloud", "storage", "objects", "describe",
                f"{args.bucket}/{SOURCE_RUN}/{relative}", "--format=json"))
            size = metadata.get("size", metadata.get("sizeBytes"))
            if size is None or not 0 < int(size) <= 256 * 1024**2:
                raise ValueError("Unexpected checkpoint size")
        prefix = (f"{args.bucket}/{args.run_id}/workers/**" if recovery
                  else f"{args.bucket}/{args.run_id}/**")
        objects = json.loads(_read("gcloud", "storage", "objects", "list",
            prefix, "--exhaustive", "--limit=1", "--format=json"))
        if not isinstance(objects, list):
            raise ValueError("Output object listing was invalid")
        if recovery and not objects:
            raise ValueError("Recovery requires an existing production namespace")
        if not recovery and objects:
            raise ValueError("Output namespace is occupied")
    else:
        objects = json.loads(_read("gcloud", "storage", "objects", "list",
            f"{args.bucket}/{args.run_id}/workers/**/SUCCESS.json", "--exhaustive", "--format=json"))
        if not isinstance(objects, list) or len(objects) != TASK_COUNT:
            raise ValueError(f"Expected exactly {TASK_COUNT} successful worker markers before aggregation")
        analysis = json.loads(_read("gcloud", "storage", "objects", "list",
            f"{args.bucket}/{args.run_id}/analysis/**", "--exhaustive", "--limit=1", "--format=json"))
        if not isinstance(analysis, list) or analysis:
            raise ValueError("Analysis namespace is occupied or object listing was invalid")
    jobs = json.loads(_read("gcloud", "batch", "jobs", "list", "--project", args.project,
                            "--location", args.region, "--filter", f"name:{job_id}", "--format=json"))
    if not isinstance(jobs, list) or jobs:
        raise ValueError(f"Batch job name is occupied or listing invalid: {job_id}")


def _request_matches(args, request):
    expected = {"run_id": args.run_id, "project": args.project, "region": args.region,
                "bucket": args.bucket, "service_account": args.service_account, "smoke": args.smoke}
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Submission arguments differ from the prepared request")


def verify_prepared_files(run_dir, request):
    """Do not silently submit an edited job or a re-generated source bundle."""
    for name in ("source.tar.gz", "workers_job.json", "aggregate_job.json"):
        expected = request.get("prepared_files_sha256", {}).get(name)
        if not expected or sha256(Path(run_dir) / name) != expected:
            raise ValueError(f"Prepared input checksum mismatch or legacy request: {name}; prepare a new run ID")
    if request["prepared_files_sha256"]["source.tar.gz"] != request["bundle_sha256"]:
        raise ValueError("Prepared source bundle identity differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit-workers", "submit-recovery", "submit-aggregate"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--native-repo", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recovery-tag", help="unique lowercase tag for an explicitly reviewed recovery job")
    args = parser.parse_args()
    validate(args, preparing=args.action == "prepare")
    run_dir = (args.run_dir or ROOT / "outputs" / "batch" / args.run_id).resolve()
    if args.action == "prepare":
        run_dir.mkdir(parents=True, exist_ok=False)
        bundle = run_dir / "source.tar.gz"
        provenance = make_bundle(ROOT, args.native_repo, bundle)
        bundle_hash = sha256(bundle)
        for stage in ("workers", "aggregate"):
            (run_dir / f"{stage}_job.json").write_text(json.dumps(build_job(args, bundle_hash, stage), indent=2) + "\n")
        request = {"run_id": args.run_id, "project": args.project, "region": args.region,
                   "bucket": args.bucket, "service_account": args.service_account, "smoke": args.smoke,
                   "bundle_sha256": bundle_hash, "source": provenance,
                   "checkpoint_sha256": {str(k): v[1] for k, v in CHECKPOINTS.items()},
                   "prepared_files_sha256": {name: sha256(run_dir / name) for name in (
                       "source.tar.gz", "workers_job.json", "aggregate_job.json")}}
        (run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared immutable production request in {run_dir}; no cloud calls made.")
        return
    request = json.loads((run_dir / "request.json").read_text())
    _request_matches(args, request)
    bundle = run_dir / "source.tar.gz"
    verify_prepared_files(run_dir, request)
    recovery = args.action == "submit-recovery"
    stage = "workers" if args.action in ("submit-workers", "submit-recovery") else "aggregate"
    if recovery:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,11}[a-z0-9])?", args.recovery_tag or ""):
            raise ValueError("--recovery-tag must be 1-13 lowercase letters, digits or internal hyphens")
        job_id = f"{args.run_id}-r-{args.recovery_tag}"
    else:
        if args.recovery_tag:
            raise ValueError("--recovery-tag is only valid with submit-recovery")
        job_id = f"{args.run_id}-{stage}"
    if len(job_id) > 63:
        raise ValueError("Batch job ID exceeds 63 characters")
    cloud_preflight(args, stage, job_id, recovery=recovery)
    if stage == "workers" and not recovery:
        inputs = f"{args.bucket}/{args.run_id}/inputs"
        for path in (bundle, run_dir / "workers_job.json", run_dir / "aggregate_job.json", run_dir / "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(path),
                            f"{inputs}/{path.name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", job_id, "--project", args.project,
                    "--location", args.region, "--config", str(run_dir / f"{stage}_job.json")],
                   check=True, timeout=180)
    print(f"Submitted {job_id}.")


if __name__ == "__main__":
    main()
