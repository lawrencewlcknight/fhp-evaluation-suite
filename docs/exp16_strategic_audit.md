# Evaluation Experiment 6: Experiment 16 strategic-position audit

This is an evaluation-suite experiment, **not a new UCV training experiment**.
It evaluates the existing Exp10-to-Exp16 continuation without fitting any model.

## Frozen design

| Stage | Training policies | Audited boards | Independent reference deals |
| --- | --- | --- | --- |
| Pilot | Seed 0 at 24h and 48h | 8 development boards | 8 |
| Development | Seeds 0, 1, 2 at 24h, 36h, 42h, 48h | 64 development boards | 64 |

The full library also contains 32 assessment boards. **Neither stage evaluates
them**, including through reference sampling; the experiment launcher provides
no assessment option. Development is not an untouched confirmation test: it
includes the pilot seed and boards. A later confirmation requires a separately
frozen question and explicit assessment access through the underlying tool.

All candidates face the same eight frozen opponents: the five published rule
agents (with the suite's documented LooseAggressive correction) and all three
Experiment 9 24h policies. The fixed Exp9 seed-0 policy generates each candidate's
identical reference-position distribution against each opponent. Both seats,
all final-round decision histories, blockers, ties and Bayesian opponent ranges
are handled by the existing exact final-round audit. Reference deals provide a
small supplementary frequency-weighted diagnostic, not precise population estimates.

Sources are pinned in `fhp_evaluation/strategic_audit/exp16.py`:

- Exp16: `exp16-feat48-20261004-182051`.
- Exp9 opponents: `exp9-cache24-20261001-132550`.
- Exp16 24h checkpoints are the imported Exp10 ancestors, not extra training runs.
- All 15 possible native checkpoints have explicit SHA256 hashes from the
  completed source evaluations. The pilot stages only its five needed files.
- The native Exp16 hand/board encoder is preserved; this is not a raw-tensor
  approximation. The current native inference source is bundled and hashed,
  rather than silently checking out a changing remote branch. Historical training
  code is not rerun. Native/batched parity, hidden-card privacy, card-order and suit
  invariance checks run on every exported board.

## Scientific endpoints

The primary endpoint is **48h minus 24h in the remaining-round root response
gap**, separately against every opponent and averaged with equal weight over the
eight opponents **within each training seed first**. Negative changes mean less
foregone value against that fixed opponent. The evaluator verifies common board
and root support across checkpoints before forming differences.

Root means weight supported hand/history/seat roots equally within each frozen
board suite. These are stratified diagnostics, not the natural probability of
encountering these roots in play. Root gap is the difference between the optimal
remaining-round response to a fixed opponent and the candidate's own remaining
play. It is **not** how much a best response can exploit the candidate, and is
**not whole-game exploitability**.

Secondary contrasts are 48h minus 36h and 48h minus 42h. The existing detailed
report supplies local decision gaps, fold/call/raise distributions, strong-hand
value, pot-odds, kicker/board/seat/history categories and the largest opportunity
losses. Tilt 0 is primary; tilts -2 and +2 are explicitly synthetic range
sensitivity checks. Do not promote whichever tilt or subgroup looks best.

The three training trajectories, not boards, decisions or opponents, are the
inferential units. Reports include every seed's change and descriptive
pointwise 95% t intervals (df=2), without significance claims or multiplicity
correction. The pilot has no training-population interval. Its purpose is
correctness, runtime, coverage and interpretability, not winner selection.

### What would make this tool useful?

Review the pilot before authorising development:

1. All exact-value, native-policy and privacy/invariance checks pass; unsupported
   opponent ranges are reported rather than filled with invented uniform ranges.
2. Inspect concrete high-gap hands and their action values. Explanations must be
   conditional on opponent ranges and legal fixed-limit actions—not simplistic
   prescriptions to always raise a strong hand or fold a weak one.
3. Check that there is sufficient supported coverage for all opponents, and that
   most conclusions are not isolated to one texture or synthetic range tilt.
4. Use `runtime_summary.json`, native export diagnostics and board timings to
   check feasibility. Budget caps below are safety limits, **not runtime estimates**.

After development, compare the audit's findings with Exp16's existing temporal
and external-panel head-to-head results. Agreement is corroborative; disagreement
can reveal genuine policy trade-offs. **Do not require 48h to beat 24h on this
audit for the tool to count as useful**, or tune the board panel to reproduce the
known head-to-head ranking. With three seeds and a selected opponent panel,
results remain diagnostic development evidence, not proof of equilibrium convergence.

## Cloud resources and retention

Each stage runs on **one `n2-standard-16` VM**, with four inference/BLAS threads,
52 GiB process-tree RSS limit, a 100 GB boot disk, and no GPU or Ray training.
Policy exports and numerical analysis execute as isolated, bounded workers;
the 16-vCPU reservation is not 16 independent training runs. Start with the pilot
to measure utilisation before considering cheaper scheduling for future studies.
Batch caps are six hours for the pilot and 24 hours for development, including
bootstrap/upload allowance. No automatic retries or automatic next-stage launch.

The VM installs Python 3.11.16, CPU Torch 2.7.0, NumPy 1.26.4 and OpenSpiel 1.6.3
with the suite's existing pinned inference requirements. No local ML environment
is needed to prepare or submit; the local launcher uses the standard library.

Only playable `.pkl` policies are downloaded. No full training states or replay
reservoirs are loaded or generated. Outputs retain analysis, diagnostic logs,
metadata and compressed behavioural-table/board caches for resumable evaluation.
These caches can exceed the small analysis download; they are evaluation caches,
not resumable training states. Existing cloud objects are never deleted.

## Launch the pilot

Run from `fhp-evaluation-suite`, after these changes have been pushed/pulled.
`prepare` is local only; `submit` starts the paid VM. Use the runner service
account already authorised to read and write the FHP results bucket.

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export UCV_REPO="/Users/lawrenceknight/Documents/deep_cfr_v3/fhp_ucv_escher/fhp-ucv-escher-experiments"
export PILOT_RUN_ID="fhp-strat16-pilot-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$PILOT_RUN_ID"

python3 gcp/exp6_ucv_exp16_strategic_audit_batch.py prepare \
  --native-repo "$UCV_REPO"
python3 gcp/exp6_ucv_exp16_strategic_audit_batch.py submit

gcloud batch jobs describe "$RUN_ID-audit" \
  --project "$PROJECT_ID" --location "$REGION"
```

Checks run before submission: service account exists, every required playable
checkpoint exists with a reasonable size, no conflicting active job, unused job
name, empty output namespace, immutable prepared source/job checksums. Each
checkpoint's actual SHA256 is checked on the VM **before loading its pickle**.
No IAM bindings are changed automatically. Once submitted, the laptop may close.

## Download and review

```bash
for experiment_run in "$PILOT_RUN_ID"
do
  mkdir -p "cloud_outputs/$experiment_run/analysis"
  gcloud storage rsync --recursive \
    "$BUCKET/$experiment_run/analysis" \
    "cloud_outputs/$experiment_run/analysis" || break
done
```

Open `analysis/exp16_report.html` for paired temporal results and the linked
`report.html` for strategic detail. `temporal_summary.json` contains all
per-opponent, per-seed and sensitivity contrasts; `summary.json` contains the
underlying diagnostic groups and cases. `runtime_summary.json` provides elapsed
worker time and peak RSS. A successful run has `analysis/SUCCESS.json`.
Bootstrap/worker logs remain at the cloud run root if failure diagnosis is needed.

## Separately approve development

After inspecting the pilot, preserve its run ID and prepare a **new** output run.
The launcher verifies a successful pilot Batch job and matching source-file and
checkpoint identities. It does not accept a synthetic control smoke test in its
place. `--pilot-reviewed` is the user's explicit review acknowledgement, not a
claim that the later policy won.

```bash
export RUN_ID="fhp-strat16-dev-$(date -u '+%Y%m%d-%H%M%S')"
python3 gcp/exp6_ucv_exp16_strategic_audit_batch.py prepare \
  --stage development --native-repo "$UCV_REPO" \
  --pilot-run-id "$PILOT_RUN_ID" --pilot-reviewed
python3 gcp/exp6_ucv_exp16_strategic_audit_batch.py submit \
  --stage development --pilot-run-id "$PILOT_RUN_ID" --pilot-reviewed
```

Source files must be unchanged since the pilot. Merely committing the same
contents is allowed; altering evaluator/native source requires a fresh pilot.
The local request and source archive are saved under `outputs/batch/$RUN_ID`.
Do not run `prepare` again for the same ID.

## Interrupted-run recovery

After the previous attempt has failed or been stopped, use the **same run ID,
same prepared directory and stage flags**, with a new recovery tag:

```bash
python3 gcp/exp6_ucv_exp16_strategic_audit_batch.py recover \
  --recovery-tag retry1
```

For development also pass `--stage development --pilot-run-id "$PILOT_RUN_ID"
--pilot-reviewed`. The cloud request must exactly match the locally prepared
request; active/unknown attempts and already-successful jobs block recovery.
The VM restores successful board caches to the same absolute paths. Both the
code/runtime fingerprint and cached checksums must match; mixed-source recovery
is rejected. Partial results are not published as a successful complete audit.
If no source request reached GCS at all, use a fresh run ID.

## Tests

Run `python -m pytest -q` in the pinned evaluation environment. Set
`FHP_UCV_REPO` to include native Exp16 hand/board MLP and structured-policy tests;
`FHP_VR_REPO` and `FHP_SD_REPO` enable the other native regressions.
The new tests cover checkpoint selection, assessment isolation, equal-panel
weighting, paired seed inference, missing/mixed inputs, source-only bundles,
offline preparation, cloud gates and shell syntax. No test submits cloud jobs.
