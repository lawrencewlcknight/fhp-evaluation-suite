# SD-CFR Experiment 6: execution benchmark before recovery

**Historical optional proposal, not the current restart plan.** The cloud
benchmark was cancelled at the user's request. The current approach uses the
locally tested numerical repair and query reuse, then continues the saved audit
directly; see [restart instructions](sd_exp6_numerical_repair.md). None of the
continuation commands below or in that document launches this benchmark.

This is a runtime experiment, not a policy-strength experiment. It does not
restart `fhp-sd6-audit-dev-20261013-104108`, repair its numerical-parity failure,
rewrite its provenance, or modify its saved 115 board tables. The original
scalar/batched, privacy, card-order and suit checks remain unchanged.

## Revised execution plan from saved measurements

The original `export_progress.json` has SHA-256
`d20c694d40c0a0d01fd03b0db6dfc5f795f76a38ccec744b154cf88d51e23840`.
It reports 64 successful seed-0 24h boards and 51 successful seed-0 48h boards.
Their mean worker times are 32.77 and 62.38 minutes. Export elapsed time was
25.28 hours; roughly 13.0% of available worker-slot time was idle within waves.
This is scheduling occupancy, **not measured CPU utilisation**.

The revised scheduler refills a free slot immediately, writes progress after
each completion, and records an unsuccessful board while continuing other
independent work. It never retries a board within the same invocation or
declares an incomplete panel successful. Deadlines and memory limits remain.
Coordinator/integrity errors still stop execution. The legacy `waves` field
in new reports contains individual completion events; use `tasks`, task start
and finish offsets, and overall elapsed time for rolling-throughput analysis.

A deterministic scheduling simulation estimates **54.49 hours for the 269
remaining SD board exports** at four rolling workers, assuming their current
individual speeds persist. Missing 48h board times use their observed 24h
times multiplied by the median paired ratio (1.8733); other seed times scale
by their pinned archive model counts. A deliberately broad -20%/+30%
sensitivity range is 43.60–70.84 hours. This is **not a confidence interval**.
Rule-policy exports, strategic scoring, setup, transfers, numerical repair and
cache migration are excluded; downstream duration is unknown. Do not present
this as a full-job completion forecast or a measured scheduler speedup.

## Controlled benchmark

One `n2-standard-16`, CPU only, STANDARD provisioning, 200 GB disk. Full seed-0
24h and 48h archives (17,690 and 35,535 networks per player), unchanged batching
and weighting. No truncation or model subsampling. The four boards are selected
only by successful 48h worker duration at ranks 0, 1/3, 2/3 and 1:

- `[16,40,48]`
- `[9,10,18]`
- `[1,34,42]`
- `[3,20,26]`

Each configuration exports all hands and histories on all four boards at both
horizons, twice: **48 full exports total**, isolated output paths for every
repetition. Allocation order is 4×4, 8×2, 8×1, then reversed:

| Workers | Torch threads per worker | Maximum computation threads |
|---|---|---|
| 4 | 4 | 16 |
| 8 | 2 | 16 |
| 8 | 1 | 8 |

This tests thread contention as well as parallel allocation; it is not a pure
worker-count ablation. Archive transfers occur once before measurements.
Every task starts an isolated process and includes startup, archive validation,
correctness checks and complete table construction. Record individual task
times, cell elapsed time and peak RSS. OS/filesystem caches are not flushed;
reversing the allocation order reduces but does not eliminate cache-order bias.

Each of six cells is bounded at three hours, with a **20-hour total Batch cap**
including setup and final uploads. A failed/timed-out cell is incomplete, not
a fast result. Other cells continue. Benchmark artifacts upload every minute
and on exit. No automatic retries or subsequent full audit are submitted.
The original failing board `[23,24,42]` is explicitly listed in the plan but
excluded from the timing panel because it has no valid completed 48h timing.
Its correctness failure remains a separate blocker to full-audit recovery.

All successful candidate tables are compared against a completed rolling-4×4
reference with the unchanged absolute tolerance `3e-6`. Only configurations
with two complete, parity-passing repetitions are eligible. Even then the
fastest is provisional: inspect both training horizons, slow-board tails and
repeat variation before choosing an allocation. This panel is not a claim
about all boards or all seeds, and it does not measure downstream scoring.

## Prepare and submit

Run from the evaluation-suite repository. The launcher bundles current local
source with hashes, so a Git push is not required. Existing unrelated edits
are not committed or overwritten.

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export RUN_ID="fhp-sd6-audit-bench-$(date -u +%Y%m%d-%H%M%S)"

mkdir -p outputs/batch/fhp-sd6-audit-dev-20261013-104108
gcloud storage cp \
  "$BUCKET/fhp-sd6-audit-dev-20261013-104108/export_progress.json" \
  outputs/batch/fhp-sd6-audit-dev-20261013-104108/export_progress.json

python3 -m fhp_evaluation.strategic_audit.sd_exp6_benchmark plan \
  --progress outputs/batch/fhp-sd6-audit-dev-20261013-104108/export_progress.json \
  --plan outputs/batch/sd6-execution-plan.json

python3 gcp/sd_exp6_execution_benchmark_batch.py prepare \
  --native-repo ../fhp_deep_cfr/fhp-deep-cfr-experiments \
  --plan outputs/batch/sd6-execution-plan.json

# Submits only this paid execution benchmark, once.
python3 gcp/sd_exp6_execution_benchmark_batch.py submit
```

Download the small report (leave tables/logs in GCS unless needed):

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Review `analysis/benchmark_summary.json`. A Batch exit alone does not establish
that every configuration qualified; check each group's `eligible` field.
Never automatically relaunch a benchmark submission after an uncertain network
response: first inspect its exact Batch job ID and uploaded immutable inputs.

## Before full-audit recovery

1. Diagnose and resolve the original scalar/batched discrepancy, without
   weakening private-information invariance checks.
2. Review benchmark completeness, parity, measured throughput and memory.
3. Validate source-compatible reuse of all 115 saved exports explicitly.
   Source hashes have changed; this benchmark intentionally does **not** relabel
   old success markers or bypass the existing identity checks.
4. Budget downstream scoring separately and arrange resumable bounded
   invocations if the remaining work exceeds one Batch job's time allocation.
5. Only then approve a full-audit restart. No benchmark outcome triggers one.
