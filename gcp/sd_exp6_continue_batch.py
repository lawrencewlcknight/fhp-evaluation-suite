#!/usr/bin/env python3
"""Prepare/submit a new SD6 continuation, preserving the stopped run's caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import sd_exp6_strategic_audit_batch as base

ROOT = base.ROOT
continuation = base.load_module("sd_continuation_config", ROOT / "fhp_evaluation/strategic_audit/sd_continuation.py")


def preflight(args, request, job_id, recovery):
    read = base.read
    state = read("gcloud", "batch", "jobs", "describe", continuation.SOURCE_RUN + "-audit",
                 "--project", args.project, "--location", args.region, "--format=value(status.state)").strip()
    if state != "FAILED":
        raise ValueError("The pinned source job must be stopped and failed")
    raw = read("gcloud", "storage", "cat", continuation.SOURCE_URI + "/manifest.json")
    if hashlib.sha256(raw.encode()).hexdigest() != continuation.MANIFEST_SHA256:
        raise ValueError("Pinned source manifest has changed")
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    jobs = base.list_jobs(args, args.run_id + "-")
    if any(r.get("status", {}).get("state") != "FAILED" for r in jobs):
        raise ValueError("Continuation already succeeded, is active, or has unknown status")
    if any(r["name"].rsplit("/", 1)[-1] == job_id for r in jobs):
        raise ValueError("Job name already exists; use a new recovery tag")
    objects = json.loads(read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                             "--exhaustive", "--limit=1", "--format=json"))
    if not isinstance(objects, list) or bool(objects) != recovery:
        raise ValueError("Submit requires an empty namespace; recover requires an existing one")
    if recovery:
        original = json.loads(read("gcloud", "storage", "cat", f"{args.bucket}/{args.run_id}/inputs/request.json"))
        if original != request:
            raise ValueError("Recovery must use the exact prepared continuation request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "recover"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL"))
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--recovery-tag")
    args = parser.parse_args()
    args.stage, args.previous_run_id, args.previous_reviewed, args.skip_pilot = (
        "development", continuation.SOURCE_RUN, True, False)
    base.validate(args, preparing=args.action == "prepare")
    if args.project != "clever-overview-399515" or args.region != "europe-west1":
        raise ValueError("This continuation is pinned to the original project and region")
    directory = (args.run_dir or ROOT / "outputs/batch" / args.run_id).resolve()
    if args.action == "prepare":
        if args.recovery_tag:
            parser.error("Recovery tag is for recover only")
        directory.mkdir(parents=True, exist_ok=False)
        source = base.common.make_bundle(ROOT, args.native_repo, directory / "source.tar.gz",
            extra_files=(*base.EXTRA_FILES, "gcp/sd_exp6_continue_batch.py"), native_packages=("deep_cfr_poker",))
        bundle_hash = base.common.sha256(directory / "source.tar.gz")
        job = base.build_job(args, bundle_hash)
        job["taskGroups"][0]["taskSpec"]["environment"]["variables"]["FHP_IMPORT_SOURCE"] = continuation.SOURCE_URI
        (directory / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        request = {key: getattr(args, key) for key in base.REQUEST_FIELDS}
        request.update(experiment=base.experiment.EXPERIMENT, source=source, bundle_sha256=bundle_hash,
                       continuation=dict(source_uri=continuation.SOURCE_URI,
                                         manifest_sha256=continuation.MANIFEST_SHA256),
                       prepared_files_sha256={n: base.common.sha256(directory / n)
                                              for n in ("source.tar.gz", "job.json")})
        (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {directory}; no job submitted. The VM will import 115 verified boards.")
        return
    request = base.verify_request(args, directory)
    if request.get("continuation") != dict(source_uri=continuation.SOURCE_URI,
                                          manifest_sha256=continuation.MANIFEST_SHA256):
        raise ValueError("Not the prepared pinned continuation")
    if args.action == "recover":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,9}", args.recovery_tag or ""):
            parser.error("Use a unique 1-10 character --recovery-tag")
        suffix = "r-" + args.recovery_tag
    else:
        if args.recovery_tag:
            parser.error("Recovery tag is for recover only")
        suffix = "audit"
    job_id = args.run_id + "-" + suffix
    preflight(args, request, job_id, args.action == "recover")
    if args.action == "submit":
        for name in ("source.tar.gz", "job.json", "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(directory / name),
                            f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", job_id, "--project", args.project,
                    "--location", args.region, "--config", str(directory / "job.json")], check=True, timeout=180)
    print(f"Submitted {job_id}; original run untouched, no automatic retries.")


if __name__ == "__main__":
    main()
