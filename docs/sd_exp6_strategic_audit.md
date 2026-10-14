# SD-CFR Experiment 6: staged strategic-position audit

## Question and frozen design

Does the selected 48-hour SD-CFR configuration show smaller strategic response
gaps than its own 24-hour policies, and which final-round decisions remain
problematic? This is **evaluation only**, using the completed Experiment 6 run
`sdcfr6-48h-20261002-161544`. It neither selects a favourable training seed nor
compares SD-CFR with another learning algorithm.

The evaluated policy is the **complete uniform historical strategy mixture**
with exact own-reach weighting at each information set. There is no archive
thinning, model subsampling, final-network substitution or average-policy fit.
The native SD-CFR adapter is retained unchanged. Each manifest and every
referenced chunk must pass its checksum; iteration prefixes, array shapes,
encoder/game identity, legal actions, native scalar/batched parity, private-card
invariance, card-order invariance and suit invariance are checked. Completion
also requires every exported board to report the full expected model count.

The pinned archive-manifest SHA-256 values and counts are recorded in
`fhp_evaluation/strategic_audit/sd_exp6.py`. Their provenance is the completed
SD-CFR internal league's `analysis/evaluation_manifest.json`, run
`fhp-sdcfr-h2h-20261005-110840`, source commit
`bb689252d6b322adb3de6102d095a49c3ed87250`.

| Training seed | 24h historical models per player | 48h historical models per player |
|---|---:|---:|
| 0 | 17,690 | 35,535 |
| 1 | 17,739 | 35,822 |
| 2 | 18,547 | 37,210 |

All inputs are fetched from the fixed SD-CFR results bucket
`gs://clever-overview-399515-fhp-deep-cfr-results`. The only downloads are the
pinned archive manifests and their complete chunks. Shared 24h/48h chunks are
downloaded once per seed. No replay, optimizer or training-state files are used.
Source code is bundled from the local evaluation suite and the native
`deep_cfr_poker` package; no cloud Git checkout or training command is run.

## Three separately submitted stages

| Stage | Policies | Development boards | Full-archive board exports | Batch safety cap |
|---|---|---:|---:|---:|
| `profile` | Seed 0, 48h | 1 prespecified board | 1 | 6h |
| `pilot` | Seed 0, 24h and 48h | 8, covering all six textures | 16 | 24h |
| `development` | Seeds 0/1/2, 24h and 48h | 64 | 384 | 72h |

**The caps are spending/runtime bounds, not expected runtimes or promises that
the stages will finish.** Every stage uses one `n2-standard-16` VM, 56,000 MiB
Batch memory allocation and a 200 GB balanced boot disk. The complete VM is
billed while running, even during the single-worker profile. No automatic
retries, continuation controller or automatic next-stage submission is enabled.

The profile exports one real full-size archive, then evaluates that board
against the same five rule opponents; it is not a tiny-model smoke test and
does not support a strength conclusion. Profile execution is serial with four
Torch threads. Pilot/development allow four isolated board workers, each with
four threads and a 13,000 MiB process-tree RSS limit. This leaves CPU and memory
bounded and measures actual parallel throughput in the pilot. Each board export
has a three-hour limit; invocation limits are 4h/22h/70h respectively, leaving
two hours within the Batch cap for setup, transfer and final uploads. Overall
Batch deadlines still apply if recovery transfer or setup consumes that reserve.

Review `analysis/runtime_summary.json`, `export_progress.json`, bootstrap logs
and peak RSS **before** advancing. A single-board time is not representative of
every texture; do not assume fourfold parallel speedup. In particular, verify
that the full-size profile leaves room within the pilot's per-worker memory
cap. Use the pilot's observed concurrent waves, including 48h policies, to
budget the 384 development exports plus rule exports and analysis. If the
workload is impractical, stop and revise the execution plan explicitly; never
silently shorten the strategy archive or relax the scientific panel.

By default, the next-stage submission requires both `--previous-reviewed` and a
completed predecessor of the correct stage. An explicit `--skip-pilot` option
allows development directly after a successful reviewed profile, accepting that
parallel throughput has not been measured; see the direct-launch instructions below.
Cloud preflight verifies checkpoint identity,
the complete archive workload, runtime-report checksum, identical bundled source
files, and successful Batch final upload. Code changes require rerunning the
feasibility stages with that code. Keep the predecessor jobs until the next
stage has been submitted, because their completion state is part of the gate.

## Metrics and interpretation

The fixed opponents are CandidStatistician, LooseAggressive, LoosePassive,
TightPassive and TightAggressive, using the suite's existing corrected rule
definitions. No UCV or VR policies are opponents or reference policies.

The established library seed `20261006` supplies nested profile/pilot/development
board sets. Its 32 assessment suit classes stay locked throughout; there is no
assessment stage in this launcher. We omit reference-deal visitation sampling,
so this audit describes a stratified diagnostic panel, **not natural play
frequencies**.

The primary quantity is the **48h-minus-24h change in remaining-round root
response gap**, in big blinds per supported audited root, against unmodified
opponent ranges. Within each opponent, the engine averages equally over
supported roots. Within each training seed, the five opponent means have equal
weight; paired differences are then summarised across the three training seeds.
We verify identical board/root-count support across checkpoints and seeds.
Boards, private hands, positions and opponents are not independent policy-level
replicates. Three-seed 95% Student-t intervals are descriptive, pointwise,
unadjusted for multiple comparisons and low-powered; the one-seed pilot has no
across-seed interval. Synthetic range tilts -2/+2 are sensitivity analyses, not
additional primary tests.

Detailed reports retain decision-level dominated-action diagnostics, local
one-decision EV gaps, remaining-round gaps, board textures and error cases.
A smaller root response gap means the policy gives up less value relative to
a remaining-round best response **to that fixed opponent**. This is not exact
whole-game exploitability, and a missed opportunity to exploit a rule opponent
need not be an equilibrium error. Any proposed learning/representation change
motivated by these findings needs separate fresh-seed training validation and
an appropriate locked assessment protocol.

## Launch from the SD-CFR repository

Both local checkouts must contain this implementation. The source-only bundle
records file hashes and Git provenance, including dirty status; pushing is not
required for the bundle to run, but commit the intended versions for permanent
provenance. The source bundle excludes result archives and thesis material.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp_deep_cfr/fhp-deep-cfr-experiments

export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export PROFILE_RUN_ID="fhp-sd6-audit-prof-$(date -u '+%Y%m%d-%H%M%S')"

# Offline preparation only; inspect the job and immutable request before submit.
bash gcp/run_sd_exp6_strategic_audit.sh prepare \
  --stage profile --run-id "$PROFILE_RUN_ID"

# This command submits the paid, bounded full-archive profile only.
bash gcp/run_sd_exp6_strategic_audit.sh submit \
  --stage profile --run-id "$PROFILE_RUN_ID"
```

The wrapper selects the sibling evaluation suite at
`../../fhp-evaluation-suite`; set `FHP_EVALUATION_REPO` to its absolute path for
a different layout. It needs only Python's standard library locally and
authenticated `gcloud` for submission. The configured service account needs
read access to the SD-CFR source bucket and write access to the output bucket,
in addition to the project's existing Batch/Logging permissions. Preflight
reads as your local identity; it cannot guarantee the VM's IAM permissions.

Prepared files are in the **evaluation suite's**
`outputs/batch/<run-id>/{source.tar.gz,job.json,request.json}`. Keep them for
recovery. For a direct launch from the evaluation suite, replace the wrapper
with `python3 gcp/sd_exp6_strategic_audit_batch.py` and supply
`--native-repo ../fhp_deep_cfr/fhp-deep-cfr-experiments` when preparing.

### Pilot: only after reviewing a successful profile

Retain the actual `PROFILE_RUN_ID`; do not generate a new predecessor ID.

```bash
export PILOT_RUN_ID="fhp-sd6-audit-pilot-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_sd_exp6_strategic_audit.sh prepare --stage pilot \
  --run-id "$PILOT_RUN_ID" --previous-run-id "$PROFILE_RUN_ID" --previous-reviewed
bash gcp/run_sd_exp6_strategic_audit.sh submit --stage pilot \
  --run-id "$PILOT_RUN_ID" --previous-run-id "$PROFILE_RUN_ID" --previous-reviewed
```

### Development: only after reviewing a successful pilot

```bash
export DEV_RUN_ID="fhp-sd6-audit-dev-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_sd_exp6_strategic_audit.sh prepare --stage development \
  --run-id "$DEV_RUN_ID" --previous-run-id "$PILOT_RUN_ID" --previous-reviewed
bash gcp/run_sd_exp6_strategic_audit.sh submit --stage development \
  --run-id "$DEV_RUN_ID" --previous-run-id "$PILOT_RUN_ID" --previous-reviewed
```

### Explicit direct development: omit the pilot after a successful profile

Use this only when deliberately accepting the unmeasured parallel-throughput
risk. It still runs **all 64 boards, all three seeds, both 24h/48h checkpoints,
complete archives and five fixed opponents**, with the unchanged 72h Batch cap.
It does not unlock assessment boards or submit a pilot.

The direct path reuses the successful profile's **exact source archive and VM
scripts**, not the current working tree. Local unrelated changes therefore do
not enter this experiment. Preparation needs the profile's original
`outputs/batch/<profile-run-id>/{request.json,source.tar.gz}` in the evaluation
suite; use `--profile-run-dir /absolute/path` on **prepare only** if those files
are elsewhere. Submit verifies the frozen request/bundle against the remote
successful profile, along with the normal full-archive, runtime and Batch gates.
Keep that profile job until the development submission has passed preflight.

From the SD-CFR root, using the environment above:

```bash
export PROFILE_RUN_ID="fhp-sd6-audit-prof-20261012-164550"
export DEV_RUN_ID="fhp-sd6-audit-dev-$(date -u '+%Y%m%d-%H%M%S')"

bash gcp/run_sd_exp6_strategic_audit.sh prepare --stage development \
  --run-id "$DEV_RUN_ID" --previous-run-id "$PROFILE_RUN_ID" \
  --previous-reviewed --skip-pilot &&
bash gcp/run_sd_exp6_strategic_audit.sh submit --stage development \
  --run-id "$DEV_RUN_ID" --previous-run-id "$PROFILE_RUN_ID" \
  --previous-reviewed --skip-pilot
```

The skipped-pilot choice is recorded in the immutable request and must also be
repeated on `recover`, together with the same predecessor and review flag.

**Historical estimate, superseded by the failed development run's timings:**
see [the execution benchmark and revised plan](sd_exp6_execution_benchmark.md)
before any recovery. The successful 2026-10-12 profile took
1,576.95 seconds (26.28 minutes) for one full 35,535-model board worker. Scaling
by the actual six archive sizes and 64 boards gives about 128 worker-hours,
or 32 hours if four workers achieve ideal concurrent throughput. A practical
planning allowance is **40–60 hours**, including contention, transfers, setup,
rule exports and analysis. This is an extrapolation from one board, not a
measured interval or completion guarantee. Different textures or poor parallel
scaling could exceed the 72h cap. The 70h evaluation-session budget leaves a
two-hour reserve; verified completed tables are recoverable if a bounded run
does not finish, but recovery is manual and incurs another paid invocation.

## Progress, results and recovery

For the October 2026 numerical failure and the locally tested query-grouping
improvement, see [numerical repair and query reuse](sd_exp6_numerical_repair.md).
The old frozen recovery request does not automatically incorporate those code
changes; use the explicit cache-preserving continuation command documented
there, not an unchanged retry.

```bash
gcloud batch jobs describe "$PROFILE_RUN_ID-audit" \
  --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'

# Read only; the progress object appears once board exports begin.
gcloud storage cat "$BUCKET/$PROFILE_RUN_ID/export_progress.json"

# Replace PROFILE_RUN_ID with the completed pilot/development ID as appropriate.
mkdir -p "cloud_outputs/$PROFILE_RUN_ID"
gcloud storage rsync --recursive \
  --exclude='(^inputs/|^tables/|^board_results/|^workers/)' \
  "$BUCKET/$PROFILE_RUN_ID" "cloud_outputs/$PROFILE_RUN_ID"
```

The compact download keeps analysis reports, library/spec/provenance, oracle
validation, archive-source inventory, timings, resource logs and completion
markers. It omits large reusable tables and worker logs. Those remain in GCS;
download the omitted paths when diagnosing a worker or preserving an offline
recovery copy. `analysis/SUCCESS.json` is written only after all required
exports, analysis, full-archive verification and reporting finish. Also check
that Batch itself succeeded, including final upload.

For profile, start with `analysis/runtime_summary.json`; one-board diagnostic
values are not the results panel. For pilot/development, start with
`analysis/sd_exp6_report.html` and `analysis/temporal_summary.json`; detailed
cases are in `analysis/report.html` and `analysis/summary.json`.

Every minute, verified tables and logs are uploaded. A failed or time-deferred
job does **not** count as a completed audit. Recovery is explicit, uses the
same prepared files/run namespace and resumes checksummed completed boards:

```bash
# Example for a failed/deferred profile. Inspect the failure before retrying.
bash gcp/run_sd_exp6_strategic_audit.sh recover --stage profile \
  --run-id "$PROFILE_RUN_ID" --recovery-tag "retry1"
```

For later stages, also repeat exactly the original `--previous-run-id` and
`--previous-reviewed` arguments. An active/unknown previous attempt, changed
request, damaged cache or already successful job blocks recovery. Recovery
allocates another paid VM with another bounded invocation; it is never
automatic. A per-board time/memory failure generally needs investigation, not
repeated identical retries. Do not delete remote tables or input manifests
while recovery or later-stage approval is pending.

## Local verification

With the suite's pinned dependencies and pytest installed:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp-evaluation-suite
export FHP_SD_REPO="/Users/lawrenceknight/Documents/deep_cfr_v3/fhp_deep_cfr/fhp-deep-cfr-experiments"
python3 -m pytest -q tests/test_sd_exp6_strategic_audit.py \
  tests/test_exp16_strategic_audit.py tests/test_strategic_board_export.py
```

Native integration uses a complete three-iteration fixture to exercise real
SD-CFR policy export, the shared numerical audit, checksummed recovery and
completion reporting. It validates mechanics, not production timing or policy
quality; the full 35,535-model cloud profile remains the required first stage.
