#!/usr/bin/env python3
"""Prepare or explicitly submit the one-VM Exp9 best-response pilot (stdlib only)."""

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
CHECKPOINT_RELATIVE = ("workers/task_000_cached_parallel_structured_ucv_escher_seed_0/"
                       "checkpoints/cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl")
CHECKPOINT_SHA256 = "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449"


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
    return dict(commit=read("rev-parse", "HEAD"), dirty=bool(read("status", "--porcelain")))


def make_bundle(evaluation_root, native_root, destination):
    """Allowlist source only; never archive checkpoints, credentials, .git or results."""
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
    for name in ("pyproject.toml", "gcp/exp4_ucv_exp9_br_pilot_batch.py",
                 "gcp/run_exp4_ucv_exp9_br_pilot.sh", "gcp/finalize_exp4_ucv_exp9_br_pilot.sh",
                 "gcp/requirements-br-pilot.txt"):
        candidates[f"evaluator/{name}"] = evaluation_root / name
    if len(candidates) > 2048:
        raise ValueError("Unexpectedly large source tree")
    files, size = {}, 0
    for name, path in candidates.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing missing or symlinked source: {path}")
        if path.stat().st_size > 4 * 1024**2:
            raise ValueError(f"Unexpectedly large source file: {path}")
        content = path.read_bytes()
        size += len(content)
        if size > 16 * 1024**2:
            raise ValueError("Source bundle exceeds the 16 MiB allowlist budget")
        files[name] = content
    manifest = dict(schema_version=1, source_only=True,
                    evaluator_git=git_provenance(evaluation_root), native_git=git_provenance(native_root),
                    files={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())})
    files["source_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with Path(destination).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
    return manifest


def validate(args):
    if not re.fullmatch(r"fhp-br-exp9-[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?", args.run_id or ""):
        raise ValueError("RUN_ID must start fhp-br-exp9- and be at most 52 lowercase characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID to your project ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION to a GCP region, e.g. europe-west1")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be a bucket name or gs://bucket, without a prefix")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"(?:[a-z0-9-]+@[a-z0-9-]+\.iam|[0-9]+-compute@developer)\.gserviceaccount\.com",
                        args.service_account or ""):
        raise ValueError("Set SA_EMAIL to the Batch runner service account")
    if not Path(args.native_repo).is_dir():
        raise ValueError("--native-repo must identify the UCV training checkout")


def build_job(args, bundle_sha256):
    destination = f"{args.bucket}/{args.run_id}"
    variables = dict(FHP_BUNDLE_URI=f"{destination}/inputs/source.tar.gz",
                     FHP_BUNDLE_SHA256=bundle_sha256,
                     FHP_CHECKPOINT_URI=f"{args.bucket}/{SOURCE_RUN}/{CHECKPOINT_RELATIVE}",
                     FHP_CHECKPOINT_SHA256=CHECKPOINT_SHA256,
                     FHP_DESTINATION=destination, FHP_SMOKE="1" if args.smoke else "0",
                     PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1",
                     CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1,
        taskSpec=dict(computeResource=dict(cpuMilli=8000, memoryMib=30000),
                      maxRetryCount=0, maxRunDuration="7200s", environment=dict(variables=variables),
                      runnables=[
                          dict(displayName="bounded-br-pilot", timeout="6300s",
                               script=dict(text=(ROOT / "gcp/run_exp4_ucv_exp9_br_pilot.sh").read_text())),
                          dict(displayName="upload-diagnostics", alwaysRun=True, timeout="600s",
                               script=dict(text=(ROOT / "gcp/finalize_exp4_ucv_exp9_br_pilot.sh").read_text()))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
                              instances=[dict(policy=dict(machineType="n2-standard-8",
                                  provisioningModel="STANDARD", bootDisk=dict(sizeGb=50, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="fhp-br-exp9", stage="smoke" if args.smoke else "pilot"))


def cloud_preflight(args):
    """Fail closed on API/auth errors and refuse occupied output namespaces."""
    def read(*command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    metadata = json.loads(read("gcloud", "storage", "objects", "describe",
        f"{args.bucket}/{SOURCE_RUN}/{CHECKPOINT_RELATIVE}", "--format=json"))
    # Some CLI versions use size, others sizeBytes. The worker always verifies SHA-256.
    size = metadata.get("size", metadata.get("sizeBytes"))
    if size is not None and not 0 < int(size) <= 256 * 1024**2:
        raise ValueError("Unexpected checkpoint size; refusing a large or empty download")
    # Use gcloud's authenticated, TLS-verified transport for every cloud check.
    # Python.org macOS runtimes can lack a usable default certificate store even
    # when gcloud works. Do not extract tokens or disable certificate verification.
    # Unlike `storage ls`/`--stat`, objects list returns [] for no matches. Any
    # command failure still propagates; it must never be mistaken for emptiness.
    objects = json.loads(read("gcloud", "storage", "objects", "list",
                             f"{args.bucket}/{args.run_id}/**",
                             "--exhaustive", "--limit=1", "--format=json"))
    if not isinstance(objects, list):
        raise ValueError("Unexpected object listing; refusing submission")
    if objects:
        raise ValueError("Output namespace already exists; use a new RUN_ID")
    jobs = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                          "--location", args.region, "--filter", f"name:{args.run_id}", "--format=json"))
    if jobs:
        raise ValueError("A Batch job already uses this RUN_ID")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "submit"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--native-repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="same cloud path, smaller execution-only workloads")
    args = parser.parse_args()
    validate(args)
    output = args.output_dir or ROOT / "outputs" / "batch" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "source.tar.gz"
    provenance = make_bundle(ROOT, args.native_repo, bundle)
    bundle_hash = sha256(bundle)
    job = build_job(args, bundle_hash)
    config = output / "job.json"
    config.write_text(json.dumps(job, indent=2) + "\n")
    request = dict(run_id=args.run_id, project=args.project, region=args.region,
                   destination=f"{args.bucket}/{args.run_id}", bundle_sha256=bundle_hash,
                   checkpoint_sha256=CHECKPOINT_SHA256, source=provenance, smoke=args.smoke)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(f"Prepared {config.resolve()} (source bundle {bundle.stat().st_size} bytes)", flush=True)
    if args.action == "dry-run":
        print("No cloud calls, uploads or jobs submitted. Review job.json and request.json.")
        return
    cloud_preflight(args)
    inputs = f"{args.bucket}/{args.run_id}/inputs"
    # Generation precondition prevents accidental overwrite/racing submissions.
    for path in (bundle, config, output / "request.json"):
        subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(path),
                        f"{inputs}/{path.name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", args.run_id, "--project", args.project,
                    "--location", args.region, "--config", str(config)], check=True, timeout=120)
    print(f"Submitted {args.run_id}. Results: {args.bucket}/{args.run_id}/analysis/")


if __name__ == "__main__":
    main()
