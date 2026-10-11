#!/usr/bin/env python3
"""Stdlib-only Exp26 launcher: one correctness-first, attacker-only Batch job."""
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
import exp21_adversarial_preservation_batch as base
from exp5_ucv_exp9_br_production_batch import make_bundle, sha256

_spec = importlib.util.spec_from_file_location("exp26_protocol", ROOT / "fhp_evaluation/adversarial/diagnostic_config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
protocol = _config.protocol


def validate(args):
    if not re.fullmatch(r"exp26-attack-[a-z0-9][a-z0-9-]{0,32}", args.run_id or ""):
        raise ValueError("Use a fresh exp26-attack-* RUN_ID (at most 46 characters)")
    # Reuse the checked project/bucket/account contract, not Exp21 run semantics.
    original = args.run_id
    args.run_id = "exp21-adv-validation"
    try:
        base.validate(args)
    finally:
        args.run_id = original
    if not (args.native_repo / "fhp_escher/checkpointing.py").is_file():
        raise ValueError("Missing native UCV repository")


def build_job(args, bundle_hash):
    job = base.build_pipeline_job(args, bundle_hash)
    spec = job["taskGroups"][0]["taskSpec"]
    spec["computeResource"] = dict(cpuMilli=8000, memoryMib=28000)
    spec["maxRunDuration"] = "5400s" if args.smoke else "21600s"
    main, final = spec["runnables"]
    main["timeout"] = "3600s" if args.smoke else "19800s"
    final["timeout"] = "1200s"
    for runnable in (main, final):
        text = runnable["script"]["text"].replace("/workspace/fhp-exp21", "/workspace/fhp-exp26")
        text = text.replace('for marker in smoke/analysis/SUCCESS.json analysis/SUCCESS.json;',
                            'for marker in analysis/SUCCESS.json;')
        text = text.replace('(s.get("status")=="complete" and s.get("scientific_complete") is True)',
                            '(s.get("status")=="complete")')
        runnable["script"]["text"] = text
    script = main["script"]["text"]
    script = script.replace('ARGS+=(--smoke-only)', 'ARGS+=(--smoke)')
    script = script.replace("stage sequential_pipeline", "stage attacker_diagnostic")
    script = script.replace("90000 \"$PYTHON\" -m fhp_evaluation.adversarial.pipeline run",
                            ("1800" if args.smoke else "17400") + ' "$PYTHON" -m fhp_evaluation.adversarial.diagnostic run')
    main["script"]["text"] = script
    main["displayName"] = "attacker-diagnostic"
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    policy.update(machineType="n2-standard-8", bootDisk=dict(sizeGb=60, type="pd-balanced"))
    job["labels"] = dict(experiment="exp26-attacker-diagnostic", mode="smoke" if args.smoke else "full")
    return job


def prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source = make_bundle(ROOT, args.native_repo, args.run_dir / "source.tar.gz",
        extra_files=("gcp/exp21_adversarial_preservation_batch.py", "gcp/exp26_attacker_diagnostic_batch.py"))
    bundle = sha256(args.run_dir / "source.tar.gz")
    (args.run_dir / "job.json").write_text(json.dumps(build_job(args, bundle), indent=2) + "\n")
    request = {k: getattr(args, k) for k in ("project", "region", "bucket", "service_account", "run_id", "smoke")}
    request.update(protocol=protocol(args.smoke), source=source, bundle_sha256=bundle,
                   files={name: sha256(args.run_dir / name) for name in ("source.tar.gz", "job.json")})
    (args.run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(f"Prepared {args.run_dir}; no cloud calls. One seed, no UCV repair, no historical replay download.")


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
        raise ValueError("Obsolete prepared protocol; choose a fresh run ID")
    for name, digest in request["files"].items():
        if sha256(args.run_dir / name) != digest:
            raise ValueError("Prepared input changed; choose a fresh run ID")
    base.read("gcloud", "iam", "service-accounts", "describe", args.service_account,
              "--project", args.project, "--format=value(email)")
    objects = json.loads(base.read("gcloud", "storage", "objects", "list", f"{args.bucket}/{args.run_id}/**",
                                  "--exhaustive", "--limit=1", "--format=json"))
    jobs = json.loads(base.read("gcloud", "batch", "jobs", "list", "--project", args.project,
                               "--location", args.region, "--filter", f"name:{args.run_id}-", "--format=json"))
    if not isinstance(objects, list) or not isinstance(jobs, list) or objects or jobs:
        raise ValueError("Output namespace/job already exists; never overwrite or retry automatically")
    for name in (*request["files"], "request.json"):
        subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(args.run_dir / name),
                        f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", f"{args.run_id}-run", "--project", args.project,
                    "--location", args.region, "--config", str(args.run_dir / "job.json")], check=True, timeout=180)
    print(f"Submitted {args.run_id}-run. Only attacker diagnostics; no automatic policy repair or follow-up jobs.")


if __name__ == "__main__":
    main()
