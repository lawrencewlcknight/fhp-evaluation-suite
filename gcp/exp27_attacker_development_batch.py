#!/usr/bin/env python3
"""Stdlib-only Exp27 launcher: one job, controls -> ablation -> gated replication."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gcp"))
import exp26_attacker_diagnostic_batch as prior
from exp5_ucv_exp9_br_production_batch import make_bundle, sha256

_s = importlib.util.spec_from_file_location("exp27_protocol", ROOT / "fhp_evaluation/adversarial/development_config.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
protocol = _m.protocol


def validate(args):
    if not re.fullmatch(r"exp27-attack-[a-z0-9][a-z0-9-]{0,32}", args.run_id or ""):
        raise ValueError("Use a fresh exp27-attack-* RUN_ID (at most 46 characters)")
    original = args.run_id
    args.run_id = "exp26-attack-validation"
    try:
        prior.validate(args)
    finally:
        args.run_id = original


def build_job(args, bundle_hash):
    job = prior.build_job(args, bundle_hash)
    task = job["taskGroups"][0]["taskSpec"]
    task["maxRunDuration"] = "5400s" if args.smoke else "28800s"
    main, final = task["runnables"]
    main["timeout"] = "3600s" if args.smoke else "27000s"
    for runnable in (main, final):
        text = runnable["script"]["text"].replace("/workspace/fhp-exp26", "/workspace/fhp-exp27")
        text = text.replace("adversarial.diagnostic run", "adversarial.development run")
        if not args.smoke:
            text = text.replace('17400 "$PYTHON"', '25200 "$PYTHON"')
        runnable["script"]["text"] = text
    main["displayName"] = "attacker-development"
    job["labels"] = dict(experiment="exp27-attacker-development", mode="smoke" if args.smoke else "full")
    return job


def prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source = make_bundle(ROOT, args.native_repo, args.run_dir / "source.tar.gz", extra_files=(
        "gcp/exp21_adversarial_preservation_batch.py", "gcp/exp26_attacker_diagnostic_batch.py",
        "gcp/exp27_attacker_development_batch.py"))
    bundle = sha256(args.run_dir / "source.tar.gz")
    (args.run_dir / "job.json").write_text(json.dumps(build_job(args, bundle), indent=2) + "\n")
    request = {k: getattr(args, k) for k in ("project", "region", "bucket", "service_account", "run_id", "smoke")}
    request.update(protocol=protocol(args.smoke), source=source, bundle_sha256=bundle,
                   files={n: sha256(args.run_dir / n) for n in ("source.tar.gz", "job.json")})
    (args.run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(f"Prepared {args.run_dir}; no cloud calls. One job, no UCV repair, replication only after qualification.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "submit", "run"))
    for name, variable in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                           ("service-account", "SA_EMAIL"), ("run-id", "RUN_ID")):
        p.add_argument("--" + name, default=os.environ.get(variable))
    p.add_argument("--native-repo", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    validate(args)
    if args.action == "prepare":
        prepare(args)
        return
    if args.action == "run" and not args.run_dir.exists():
        prepare(args)
    request = json.loads((args.run_dir / "request.json").read_text())
    for key in ("project", "region", "bucket", "service_account", "run_id", "smoke"):
        if request.get(key) != getattr(args, key):
            raise ValueError(f"Prepared request differs: {key}")
    if request.get("protocol") != protocol(args.smoke):
        raise ValueError("Obsolete protocol; prepare a fresh run")
    if set(request.get("files", {})) != {"source.tar.gz", "job.json"}:
        raise ValueError("Invalid prepared file manifest")
    for name, digest in request["files"].items():
        if sha256(args.run_dir / name) != digest:
            raise ValueError("Prepared input changed; prepare a fresh run")
    if request["files"]["source.tar.gz"] != request.get("bundle_sha256"):
        raise ValueError("Source bundle identity differs")
    read = prior.base.read
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    objects = json.loads(read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                              "--exhaustive", "--limit=1", "--format=json"))
    jobs = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                           "--location", args.region, "--filter", f"name:{args.run_id}-", "--format=json"))
    if not isinstance(objects, list) or not isinstance(jobs, list) or objects or jobs:
        raise ValueError("Occupied cloud namespace; no automatic overwrite/retry")
    for name in (*request["files"], "request.json"):
        subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(args.run_dir / name),
                        f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", f"{args.run_id}-run", "--project", args.project,
                    "--location", args.region, "--config", str(args.run_dir / "job.json")], check=True, timeout=180)
    print(f"Submitted {args.run_id}-run. All permitted stages run in this one job; no policy repair.")


if __name__ == "__main__":
    main()
