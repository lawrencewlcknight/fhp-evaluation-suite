# GCP Batch: Experiment 9 specialised best-response pilot

Evaluation-suite experiment **`exp4_ucv_exp9_br_pilot`** evaluates the frozen
UCV-Escher **training Experiment 9**, seed 0, at 24 hours. It does not train a
model, evaluate other seeds, or compute exact full-game exploitability.

## Pilot contract

| Setting | Value |
| --- | --- |
| Source run | `exp9-cache24-20261001-132550` |
| Policy | `cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl` |
| Machine | One `n2-standard-8`, standard provisioning, CPU only |
| Task allocation | 8 vCPUs, 30,000 MiB requested memory, 50 GB boot disk |
| Overall task ceiling | 2 hours, including setup; no automatic retries |
| Per-evaluation-worker memory watchdog | 8,192 MiB process-tree RSS |
| Validation | Independent correctness suite, 120-second cap |
| Native execution smoke | 2 duplicate pairs, 8 preflop rollouts, 120-second cap |
| Profiling | 1, 2, 4, 8 computation threads, sequentially on the same VM |
| Profile sample | 4 identical sampled boards per configuration, 3 repeats; separate excluded warmup |
| Each profile cap | 600 seconds |
| Pilot comparison | 25 duplicate pairs, both seats, 4,096 preflop rollouts |
| Comparison cap | 1,800 seconds |

Each profile covers all seven preflop entry lines and both responder seats.
The lowest successful median board time selects the comparison's thread count;
ties prefer fewer threads. This selection never uses poker payoffs. A timed-out
profile is recorded and excluded from selection. Any other profile failure stops
the pilot, as do validation or native-smoke failures. If no profile succeeds,
the comparison is not attempted.

The paired comparison runs old LBR and the new exact-flop response on the same
deals and reports each response's winnings and the paired difference in
milli-big-blinds per hand. Higher responder winnings mean a higher exploitability
lower bound, not a stronger target model. Preflop remains approximate. This small
pilot assesses feasibility and variance; its intervals are not a Nash certificate
or a final thesis-quality estimate. Fresh evaluation deals and all three training
seeds should be used for the subsequent production study.

## Prerequisites

- A logged-in `gcloud` CLI, Python 3.10–3.12, and the existing Batch configuration.
- `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL` set as for previous experiments.
  `BUCKET` accepts either `gs://bucket-name` or `bucket-name`, without a prefix.
- Permission to submit Batch jobs and use `SA_EMAIL`; bucket list/read/write access.
  The runner account needs Batch agent/logging permissions and bucket read/write
  access. The VM needs outbound access for its small inference environment setup.
- The local UCV checkout used for the native checkpoint loader.

The launcher builds a **source-only snapshot**, with per-file hashes and Git
provenance where available, instead of requiring a GitHub clone or pushed
evaluation commit. Only
the evaluator, the two native inference packages, and explicitly listed launcher
files are included. No `.git`, credentials, checkpoint weights, replay buffers,
training states or result directories are uploaded from the laptop. Review
`request.json` for the complete source-file allowlist. Source code must be trusted.

The VM downloads only the one policy checkpoint directly from the existing run
in `$BUCKET`. Its SHA-256 is checked before unpickling and again before evaluation:

```text
9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449
```

This hash is the one recorded by `fhp-eval79-20261002-164912` for the 24-hour,
seed-0 Experiment 9 policy. There is no production CLI override for another
checkpoint or hash.

## 1. Prepare and inspect without cloud calls

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp-evaluation-suite
export FHP_NATIVE_REPO=/Users/lawrenceknight/Documents/deep_cfr_v3/fhp_ucv_escher/fhp-ucv-escher-experiments
export RUN_ID="fhp-br-exp9-$(date -u +%Y%m%d-%H%M%S)"

python3 gcp/exp4_ucv_exp9_br_pilot_batch.py dry-run \
  --native-repo "$FHP_NATIVE_REPO" \
  --output-dir "outputs/batch/${RUN_ID}-preview"
```

Inspect `job.json` and `request.json` in that directory. Dry-run makes **no cloud
calls**, uploads nothing and submits nothing. The normal `submit` command below
creates a fresh snapshot, so do not change source code between preview and submit
if you want to submit exactly the reviewed source.

## 2. Cloud execution smoke (required before the first pilot or after setup changes)

This uses the same real Experiment 9 checkpoint and the same bootstrap/upload
path, but reduces each profile to one board/one repeat and the final comparison
to two pairs/eight rollouts. It is not a performance measurement. Use a distinct
run ID from the main pilot:

```bash
export RUN_ID="fhp-br-exp9-smoke-$(date -u +%Y%m%d-%H%M%S)"
python3 gcp/exp4_ucv_exp9_br_pilot_batch.py submit \
  --native-repo "$FHP_NATIVE_REPO" --smoke
```

Wait for `SUCCEEDED` and inspect its diagnostics before starting the main pilot.
No later job is automatically launched by a smoke job.

## 3. Submit the pilot

After the smoke succeeds, set a new pilot ID:

```bash
export RUN_ID="fhp-br-exp9-$(date -u +%Y%m%d-%H%M%S)"
python3 gcp/exp4_ucv_exp9_br_pilot_batch.py submit \
  --native-repo "$FHP_NATIVE_REPO"
```

Submission checks the runner account, source object metadata, an unused output
prefix and an unused job name. Authentication/network failures stop submission.
Source uploads use generation preconditions to refuse overwrites. If submission
fails after uploads, use a new `RUN_ID`; no automatic retry or cleanup is attempted.
Local output directories are also never overwritten. You can disconnect the
laptop after successful submission.

The runtime uses Python 3.11.13, CPU Torch 2.7.0, OpenSpiel 1.6.3 and the pinned
dependencies in `gcp/requirements-br-pilot.txt`. Ray and the training stack are
not required. Resolved dependencies and Python version are saved with results.

### Recovering from a local TLS preflight error

The launcher now uses `gcloud` for every cloud preflight request, including
checking whether the output prefix is unused. This avoids relying on the local
Python installation's separate certificate store; TLS verification remains
enabled. Authentication, network and malformed-listing errors still stop submission.

If an earlier version failed with Python's `CERTIFICATE_VERIFY_FAILED` before
source upload, it did not submit a Batch job. Keep the local preparation files
for diagnosis and retry with a **new `RUN_ID`**, because existing local output
directories are deliberately not overwritten. No system certificate changes or
TLS-verification bypass are needed for this corrected launcher.

### Recovering from the VM Python setup failure

Run `fhp-br-exp9-20261004-104050` failed before evaluation when Python could
not determine its base executable while creating the bootstrap virtual environment.
The corrected scripts export a standard Linux `PATH` (retaining existing extra
entries) and invoke `/usr/bin/python3 -I` explicitly for bootstrap. The pinned
Python 3.11.13 evaluation runtime and all experiment settings remain unchanged.

`bootstrap.log` now records named setup stages, the resolved Python executable,
base executable and version, and the stage/line/exit code on failure. No complete
environment dump or credentials are logged. `main_exit.json` also identifies the
main stage reached. Both main and final-upload runnables initialise their PATH.

Use a **new `RUN_ID`** and run the cloud smoke above first; do not reuse the failed
job ID or overwrite its diagnostics. Local tests do not replace verification on
the actual Batch image. A successful cloud smoke is a manual prerequisite, not an
automatic launcher-enforced gate. No pilot is submitted automatically.

## 4. Monitor

```bash
gcloud batch jobs describe "$RUN_ID" \
  --project "$PROJECT_ID" --location "$REGION" \
  --format='yaml(status)'
```

Stage-start/stage-end events and bootstrap errors are in Cloud Logging. Detailed
worker logs, resource samples and partial results are copied to
`$BUCKET/$RUN_ID/analysis` approximately once per minute.

The 2-hour limit is a task runtime ceiling, not a guarantee about queueing time
or a prediction that the pilot needs two hours. The main runnable has a shorter
105-minute timeout and the final upload runnable has ten minutes, leaving room
inside the overall deadline. Batch's `alwaysRun` upload is eligible after a main
runnable failure, but cannot override the overall task deadline or recover a lost
VM. See the [Batch runnable contract](https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs#runnable).

## 5. Download and inspect

```bash
FHP_BUCKET_ROOT="gs://${BUCKET#gs://}"
FHP_BUCKET_ROOT="${FHP_BUCKET_ROOT%/}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_BUCKET_ROOT/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

Important outputs:

- `pilot_manifest.json`: stages, selected threads, timings and overall computation status;
- `comparison/result.json`: complete paired pilot estimates and raw deal-pair scores;
- `profile_threads_*/result.json`: completed profiling measurements;
- each stage's `run.json`, `resources.jsonl`, `stdout.log`, `stderr.log`;
- `partial.json`: completed boards or deal pairs if a worker is interrupted;
- `bootstrap.log`, `main_exit.json`, `final_system_diagnostics.txt`, `pip_freeze.txt`;
- `source_manifest.json` and `checkpoint_identity.json`: immutable source/model provenance;
- `SUCCESS.json`: all computation stages finished; also require Batch `SUCCEEDED`
  to establish that the mandatory final upload completed.

Partial files are **diagnostics, not final estimates**; they intentionally omit
confidence intervals. A comparison timeout marks the pilot failed/incomplete and
does not emit `SUCCESS.json`. Completed stages and partial pairs remain available
for cost diagnosis. There is no automatic resume or pooling of timed-out samples.

RSS is polled every 0.1 seconds, not enforced as a hard allocation ceiling. A VM
or process killed abruptly may lack its final diagnostics; inspect periodic
uploads and Batch/Cloud Logging events as well. Upload errors are not silently
reported as successful job completion.

## Production follow-up — `exp5_ucv_exp9_br_production_3seed`

Experiment 5 freezes the 24-hour policies from Experiment 9 training seeds 0,
1 and 2. For each policy it evaluates 2,000 fresh duplicate pairs (4 shards of
500), with both seats per pair, 4,096 preflop rollouts and the pilot-selected two
CPU threads. There are 6,000 pairs/12,000 hands in total. Corresponding shards
across training seeds use identical deal/action streams; LBR and exact-flop use
the same stream within every shard. The four streams are distinct from the pilot.

The worker job has 12 parallel `n2-standard-2` tasks, a four-hour task ceiling,
standard provisioning and no automatic retries. Each pair is atomically saved;
the active shard is uploaded every minute. A restored shard resumes only when
its protocol, source bundle, checkpoint hash and random stream identity match.

The separately cost-gated aggregate job can only be submitted after all 12
`SUCCESS.json` markers exist. Its primary estimate is the unweighted mean of the
three seed-level paired exact-flop-minus-LBR differences, with a Student-*t*
interval over training seeds. The pooled pair-level interval is secondary and
conditional on the frozen policies. Both response scores remain whole-game
exploitability lower bounds because preflop is still LBR.

### 1. Prepare an immutable request (no cloud calls)

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp-evaluation-suite
export FHP_NATIVE_REPO=/Users/lawrenceknight/Documents/deep_cfr_v3/fhp_ucv_escher/fhp-ucv-escher-experiments
export RUN_ID="fhp-br-exp9-prod-$(date -u +%Y%m%d-%H%M%S)"

python3 gcp/exp5_ucv_exp9_br_production_batch.py prepare \
  --native-repo "$FHP_NATIVE_REPO"
```

Review `outputs/batch/$RUN_ID/request.json`, `workers_job.json`,
`aggregate_job.json`, and the source manifest inside `source.tar.gz`. Preparation
does not contact GCP. Submission verifies the exact prepared bundle hash; source
changes require a new run ID and a new preparation.

For the first infrastructure check, prepare a distinct ID with `--smoke`; repeat
`--smoke` on both submission commands. Smoke uses two pairs per shard and eight
rollouts and is not an estimate.

### 2. Submit and monitor the workers

```bash
python3 gcp/exp5_ucv_exp9_br_production_batch.py submit-workers

gcloud batch jobs describe "${RUN_ID}-workers" \
  --project "$PROJECT_ID" --location "$REGION" --format='yaml(status)'
```

The launcher verifies all three source checkpoints, refuses an occupied output
prefix/job name, then uploads the immutable request with generation
preconditions. The fixed checkpoint SHA-256 values are checked before loading.

If a task is interrupted, its last uploaded `partial.json` is retained. Do not
edit it. After diagnosing the failure, explicitly submit a recovery attempt with
a unique tag (completed shards validate their identity and exit; incomplete
shards continue at their next unsaved pair):

```bash
python3 gcp/exp5_ucv_exp9_br_production_batch.py submit-recovery \
  --recovery-tag r1
```

Recovery requires the existing namespace and refuses an occupied recovery job
name. It reuses the reviewed local request/bundle and does not overwrite cloud
inputs. Every task fails closed if restored metadata differs from its immutable
source/checkpoint/protocol identity.

### 3. Submit aggregation only after all workers succeed

```bash
python3 gcp/exp5_ucv_exp9_br_production_batch.py submit-aggregate

gcloud batch jobs describe "${RUN_ID}-aggregate" \
  --project "$PROJECT_ID" --location "$REGION" --format='yaml(status)'
```

Preflight requires exactly 12 worker success markers and an empty analysis
prefix. Important outputs are `analysis/aggregate_summary.json`,
`analysis/seed_summary.csv`, `analysis/report.md`, and `analysis/SUCCESS.json`.

### 4. Download final analysis and raw shards

```bash
FHP_BUCKET_ROOT="gs://${BUCKET#gs://}"
FHP_BUCKET_ROOT="${FHP_BUCKET_ROOT%/}"
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage rsync --recursive \
  "$FHP_BUCKET_ROOT/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_BUCKET_ROOT/$RUN_ID/workers" \
  "cloud_outputs/$RUN_ID/workers"
```

Raw per-pair response scores and separate seat payoffs remain in every shard's
`result.json`; the aggregate does not discard or rewrite them.
