#!/usr/bin/env python3
"""Source-bundled, explicitly cost-gated UCV Experiment 21 (no automatic launches)."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_ucv_exp9_br_production_batch import make_bundle, sha256

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("qualify", "train", "evaluate", "aggregate")


def read(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout


def validate(args):
    if not re.fullmatch(r"exp21-adv-[a-z0-9](?:[a-z0-9-]{0,35}[a-z0-9])?", args.run_id or ""):
        raise ValueError("RUN_ID must start exp21-adv- and be at most 47 lowercase characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project or ""):
        raise ValueError("Set PROJECT_ID")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", args.region or ""):
        raise ValueError("Set REGION")
    bucket = (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket):
        raise ValueError("BUCKET must be a bucket without a path")
    args.bucket = "gs://" + bucket
    if not re.fullmatch(r"(?:[a-z0-9-]+@[a-z0-9-]+\.iam|[0-9]+-compute@developer)\.gserviceaccount\.com",
                        args.service_account or ""):
        raise ValueError("Set SA_EMAIL")


def task_setup(stage):
    text = '''FHP_RUN_DESTINATION="$FHP_DESTINATION"
FHP_TASK="${BATCH_TASK_INDEX:-0}"
FHP_SEED="$FHP_TASK"
FHP_ARM=""
'''
    if stage == "evaluate":
        text += '''FHP_SEED="$((FHP_TASK / 4))"
FHP_ARMS=(unchanged ordinary adversarial preserved)
FHP_ARM="${FHP_ARMS[$((FHP_TASK % 4))]}"
'''
    if stage != "aggregate":
        text += f'FHP_DESTINATION="$FHP_RUN_DESTINATION/{stage}/task_$FHP_TASK"\n'
    return text


def main_script(stage):
    # Reuse the bootstrap which fixed the earlier Python/SSL/PATH failures.
    script = (ROOT / "gcp/run_exp4_ucv_exp9_br_pilot.sh").read_text()
    start = script.index("stage checkpoint_download\n")
    stop = script.index("# Batch script runnables", start)
    script = script[:start] + script[stop:]
    script = script[:script.index("ARGS=()\n")]
    script = script.replace("/workspace/fhp-br-exp9", "/workspace/fhp-exp21")
    script = script.replace("WORK=/workspace/fhp-exp21", task_setup(stage) + "WORK=/workspace/fhp-exp21")
    script += '''ARGS=()
if [[ "$FHP_SMOKE" == 1 ]]; then ARGS+=(--smoke); fi
'''
    if stage == "aggregate":
        script += '''stage stage_inputs
COUNT=12
if [[ "$FHP_SMOKE" == 1 ]]; then COUNT=4; fi
for ((i=0; i<COUNT; i++)); do
  mkdir -p "$WORK/input/task_$i"
  gcloud storage cp "$FHP_RUN_DESTINATION/evaluate/task_$i/analysis/result.json" "$WORK/input/task_$i/result.json"
done
stage aggregation
"$PYTHON" -m fhp_evaluation.adversarial.aggregate --input "$WORK/input" --output "$OUTPUT" "${ARGS[@]}"
stage complete
'''
        return script
    if stage == "train":
        script += '''stage qualification_download
gcloud storage cp "$FHP_RUN_DESTINATION/qualify/task_$FHP_SEED/analysis/result.json" "$WORK/input/gate.json"
mkdir -p "$WORK/input/replay"
for name in manifest.json x.npy y.npy mask.npy cdf.npy; do
  gcloud storage cp "$FHP_RUN_DESTINATION/qualify/task_$FHP_SEED/analysis/measurement/replay/$name" "$WORK/input/replay/$name"
done
ARGS+=(--gate "$WORK/input/gate.json" --replay "$WORK/input/replay")
'''
    elif stage == "evaluate":
        script += '''stage candidate_download
mkdir -p "$WORK/input/training"
for name in result.json unchanged.pkl ordinary.pkl adversarial.pkl preserved.pkl; do
  gcloud storage cp "$FHP_RUN_DESTINATION/train/task_$FHP_SEED/analysis/$name" "$WORK/input/training/$name"
done
ARGS+=(--training "$WORK/input/training" --arm "$FHP_ARM")
'''
    script += f'''stage {stage}
"$PYTHON" -m fhp_evaluation.adversarial.workflow {stage} --seed "$FHP_SEED" \\
  --source "$WORK/input/source" --bucket "$FHP_SOURCE_BUCKET" \\
  --native-repo "$WORK/source/native" --output "$OUTPUT/measurement" "${{ARGS[@]}}"
# Export small index/final policies to the stable stage root; detailed logs and
# intermediate opponent/policy checkpoints remain under measurement/.
cp "$OUTPUT/measurement/result.json" "$OUTPUT/result.json"
cp "$OUTPUT/measurement/SUCCESS.json" "$OUTPUT/SUCCESS.json"
'''
    if stage == "train":
        script += '''for arm in unchanged ordinary adversarial preserved; do
  cp "$OUTPUT/measurement/$arm.pkl" "$OUTPUT/$arm.pkl"
done
'''
    script += "stage complete\n"
    return script


def build_job(args, bundle, stage):
    seeds = 1 if args.smoke else 3
    tasks = seeds * 4 if stage == "evaluate" else 1 if stage == "aggregate" else seeds
    cpus = 4 if stage in ("evaluate", "aggregate") else 16
    seconds = 900 if args.smoke else dict(qualify=7200, train=50400, evaluate=21600, aggregate=900)[stage]
    env = dict(FHP_BUNDLE_URI=f"{args.bucket}/{args.run_id}/inputs/source.tar.gz",
               FHP_BUNDLE_SHA256=bundle, FHP_DESTINATION=f"{args.bucket}/{args.run_id}",
               FHP_SOURCE_BUCKET=args.bucket, FHP_SMOKE="1" if args.smoke else "0",
               PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    final = (ROOT / "gcp/finalize_exp4_ucv_exp9_br_pilot.sh").read_text()
    final = final.replace("/workspace/fhp-br-exp9", "/workspace/fhp-exp21")
    final = final.replace("OUTPUT=/workspace/fhp-exp21/output", task_setup(stage) + "OUTPUT=/workspace/fhp-exp21/output")
    return dict(taskGroups=[dict(taskCount=tasks, parallelism=tasks, taskCountPerNode=1,
        taskSpec=dict(computeResource=dict(cpuMilli=cpus * 1000, memoryMib=14000 if cpus == 4 else 60000),
                      maxRetryCount=0, maxRunDuration=f"{seconds + 2400}s",
                      environment=dict(variables=env), runnables=[
                          dict(displayName=stage, timeout=f"{seconds + 1500}s", script=dict(text=main_script(stage))),
                          dict(displayName="final-upload", alwaysRun=True, timeout="600s", script=dict(text=final))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account), instances=[dict(policy=dict(
            machineType=f"n2-standard-{cpus}", provisioningModel="STANDARD",
            bootDisk=dict(sizeGb=50, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(experiment="exp21-adversarial", stage=stage))


def remote_json(uri):
    return json.loads(read("gcloud", "storage", "cat", uri))


def require_job(args, run_id, stage):
    job = json.loads(read("gcloud", "batch", "jobs", "describe", f"{run_id}-{stage}",
                         "--project", args.project, "--location", args.region, "--format=json"))
    if job.get("status", {}).get("state") != "SUCCEEDED":
        raise ValueError(f"Require successful {run_id}-{stage}, including final upload")


def preflight(args, request, stage):
    base = f"{args.bucket}/{args.run_id}"
    read("gcloud", "iam", "service-accounts", "describe", args.service_account,
         "--project", args.project, "--format=value(email)")
    prefix = base if stage == "qualify" else f"{base}/{'analysis' if stage == 'aggregate' else stage}"
    listing = json.loads(read("gcloud", "storage", "objects", "list", f"{prefix}/**",
                             "--exhaustive", "--limit=1", "--format=json"))
    if not isinstance(listing, list) or listing:
        raise ValueError("Output namespace occupied/invalid; never overwrite a stage")
    jobs = json.loads(read("gcloud", "batch", "jobs", "list", "--project", args.project,
                     "--location", args.region, "--filter", f"name:{args.run_id}-{stage}", "--format=json"))
    if not isinstance(jobs, list) or jobs:
        raise ValueError("Job name already used")
    if stage == "qualify":
        if not args.smoke:
            if not re.fullmatch(r"exp21-adv-[a-z0-9-]+", args.smoke_run_id or ""):
                raise ValueError("Production qualification requires SMOKE_RUN_ID")
            previous = remote_json(f"{args.bucket}/{args.smoke_run_id}/inputs/request.json")
            if previous.get("smoke") is not True or previous["bundle_sha256"] != request["bundle_sha256"]:
                raise ValueError("Require a complete smoke of this identical source bundle")
            for previous_stage in STAGES:
                require_job(args, args.smoke_run_id, previous_stage)
    else:
        remote = remote_json(f"{base}/inputs/request.json")
        if remote != request:
            raise ValueError("Cloud and local frozen requests differ")
        predecessor = dict(train="qualify", evaluate="train", aggregate="evaluate")[stage]
        require_job(args, args.run_id, predecessor)
        tasks = (1 if args.smoke else 3) * (4 if predecessor == "evaluate" else 1)
        for task in range(tasks):
            uri = f"{base}/{predecessor}/task_{task}/analysis"
            marker = remote_json(f"{uri}/SUCCESS.json")
            raw = read("gcloud", "storage", "cat", f"{uri}/result.json")
            import hashlib
            if hashlib.sha256(raw.encode()).hexdigest() != marker["result_sha256"]:
                raise ValueError("Prior result checksum mismatch")
            result = json.loads(raw)
            if result.get("bundle_sha256") != request["bundle_sha256"]:
                raise ValueError("Prior result used a different source bundle")
            if stage == "train" and not args.smoke and result.get("gate_passed") is not True:
                raise ValueError(f"Seed {task} failed attacker qualification. Stop and review; no repair submitted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", *STAGES))
    for name, env in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                      ("service-account", "SA_EMAIL"), ("run-id", "RUN_ID"), ("smoke-run-id", "SMOKE_RUN_ID")):
        parser.add_argument("--" + name, default=os.environ.get(env))
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    validate(args)
    if args.action == "prepare":
        args.run_dir.mkdir(parents=True, exist_ok=False)
        bundle = args.run_dir / "source.tar.gz"
        provenance = make_bundle(ROOT, args.native_repo, bundle, extra_files=(
            "gcp/exp21_adversarial_preservation_batch.py", "gcp/run_exp4_ucv_exp9_br_pilot.sh",
            "gcp/finalize_exp4_ucv_exp9_br_pilot.sh"))
        digest = sha256(bundle)
        for stage in STAGES:
            (args.run_dir / f"{stage}_job.json").write_text(json.dumps(build_job(args, digest, stage), indent=2) + "\n")
        files = ["source.tar.gz", *(f"{s}_job.json" for s in STAGES)]
        request = dict(run_id=args.run_id, project=args.project, region=args.region, bucket=args.bucket,
                       service_account=args.service_account, smoke=args.smoke, source=provenance,
                       bundle_sha256=digest, files={name: sha256(args.run_dir / name) for name in files})
        (args.run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Prepared {args.run_dir}; no cloud calls or jobs submitted.")
        return
    request = json.loads((args.run_dir / "request.json").read_text())
    for key in ("run_id", "project", "region", "bucket", "service_account", "smoke"):
        if request[key] != getattr(args, key):
            raise ValueError(f"Prepared request differs: {key}")
    for name, digest in request["files"].items():
        if sha256(args.run_dir / name) != digest:
            raise ValueError("Prepared files have changed; prepare a new run")
    preflight(args, request, args.action)
    if args.action == "qualify":
        for name in (*request["files"], "request.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0", str(args.run_dir / name),
                            f"{args.bucket}/{args.run_id}/inputs/{name}"], check=True, timeout=300)
    subprocess.run(["gcloud", "batch", "jobs", "submit", f"{args.run_id}-{args.action}",
                    "--project", args.project, "--location", args.region,
                    "--config", str(args.run_dir / f"{args.action}_job.json")], check=True, timeout=180)
    print(f"Submitted {args.run_id}-{args.action}. No later phase is automatically launched.")


if __name__ == "__main__":
    main()
