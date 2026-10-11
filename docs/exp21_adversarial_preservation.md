# UCV Experiment 21 — adversarial policy improvement with preservation

This is a **pilot intervention on deployed policies**, not another UCV solver
run, an exact best response, or a true-average-strategy compression experiment.
It asks whether sample-trained opponents can guide improvement of an existing
FHP policy, and whether replay/blueprint preservation prevents collateral damage.
There is no assumption that adversarial training improves equilibrium quality.

## Frozen source and arms

Use the original Experiment 10 run `exp10-features-20261001-161740`, seeds 0, 1,
2, at 24 active hours. The historical checkpoint manifests are pinned by SHA-256
in `fhp_evaluation/adversarial/config.py`. Never modify these source objects.
Student architecture, exact-card/hand-board encoding and deployment size stay
unchanged: 213 policy inputs, 64-wide branches and a 192×192 trunk.

Each source seed supplies four arms starting from identical policy weights:

| Arm | Post-training |
| --- | --- |
| `unchanged` | Frozen source policy |
| `ordinary` | Additional grouped, iteration-weighted soft-target CE |
| `adversarial` | On-policy payoff gradient against a retained attacker population |
| `preserved` | Same payoff gradient + original replay CE + KL to the frozen source |

All updated policies use Adam at `1e-4`, gradient clipping at 10 and warm-start
weights. The ordinary arm is **additional fitting**, not a repeat of Experiment
10's reset-at-checkpoint fitting at `0.003`. It receives as much wall time as the
preserved arm's attack discovery, repair and persistence **plus qualification**.
Source preparation included in qualification makes this control deliberately
generous. The unregularised arm has the same fixed attacker/repair time budgets;
its exact elapsed time can differ. Report elapsed times and update counts, not
only nominal budgets. Qualification and independent evaluation costs are recorded.

Replay groups retain original gamma=2 iteration mass. Mass-proportional IID
group sampling with the original constant loss scale is unbiased for the same
weighted grouped CE objective; it is not identical to the historical minibatch
stream. The replay covers sampled states, **not** the true CFR average at every
information set. KL preserves the *source neural policy*, not an exact teacher.

## Stage 1: independently qualify the attackers

Per source seed, train two fresh Double-DQN attackers for each seat, 15 minutes
each (one hour total). They use only the acting player's existing lossless
information-state encoder. Architecture: 256×256 ReLU, three Q outputs; Adam
`3e-4`; gamma=1; terminal rewards in big blinds; replay capacity 100,000;
1,024-transition warmup; minibatch 256; four updates per 32-hand collection wave;
target sync every 250 updates. Epsilon falls from 1 to .05 over 80% of the budget.
Illegal next actions are masked in Double-DQN action selection; terminal targets
do not bootstrap. CPU queries are batched; replay uses contiguous arrays.

The two paired-seat candidates are ranked on 1,000 development duplicate pairs.
The selected response is then assessed on **2,000 different pairs**. Both its
payoff against the source and its paired improvement over its corresponding
untrained network must have positive lower 95% Student-t confidence bounds.
No repeated testing until significance and no optional early stopping. This is
evidence of a useful attacker, **not** evidence of a close-to-optimal response.
Failure means *inconclusive, stop and review*, not proof that the source is safe.

The three-seed repair phase requires every source seed to pass. In the one-job
workflow, an inconclusive gate stops the job before full training: pipeline
`STATUS.json` reports `inconclusive_qualification`, and Batch reports `FAILED`.
For legacy separate-stage jobs, a successful qualification job can still report
`gate_passed: false`; it completed the measurement, not the scientific gate.
Execution-only smoke can exercise later stages but is explicitly barred from
qualifying a production run.

Historical Exp10 states are approximately 5.4 GB monolithic files. Qualification
downloads and verifies one per source seed, extracts only policy replay and saves
a hashed grouped cache. Training reuses this cache; evaluation needs policies
only. The one-job workflow reuses the local verified source inputs across phases,
with separate smoke/full input directories. No critic training or full-game
traversal is performed.

## Stage 2: fixed-budget policy repair

For each adversarial arm, perform three rounds. Each round freezes the current
student, trains four fresh attackers (two per seat, 15 minutes each), then allows
five minutes of student repair. Keep all earlier opponents. Sample a uniform
population member once per hand; never change the opponent within a hand.
Student seat alternates, with 32 simultaneous hands. The payoff loss is the sum
of log probabilities within each trajectory, averaged over hands, with a
previous-batch moving return baseline. No privileged-state critic is introduced.

The preserved loss is:

`payoff_policy_gradient + 1.0 * original_replay_CE + 1.0 * KL(source || student)`.

CE uses batches of 2,048 grouped examples. KL is measured at repair decision
states. Both coefficients are fixed pilot choices, not selected using final test
performance; replay CE and KL are a package in this first pilot. A positive result
would require ablations of these components. Preservation is a heuristic, not a
guarantee of non-increasing exploitability. Endpoints are the fixed final round,
not the best observed checkpoint.

Saved outputs include native-playable policy files for all arms, every round's
policy, attacker networks, optimizer states, RNG metadata and detailed timings.
They are **not full resumable UCV solver checkpoints**. Attacker transition replay
is not retained; exact interruption/resume of this pilot is not implemented.
No automatic retries or recovery loop silently spends another budget.

## Stage 3: independent evaluation

For each of the 12 final policies (four arms × three seeds), separately train
fresh attackers using new training randomness and select on development deals.
Each resulting paired-seat challenger then plays **all four arms** from the same
source seed. Separate common-random-number panels are used for pool selection
(1,000 pairs) and final scoring (5,000 pairs). Aggregation selects each defender's
strongest challenger on validation payoffs, never on its reported test payoffs.

Also run, per candidate:

- 5,000 duplicated pairs directly against its unchanged source;
- 5,000 pairs against each of the five existing rule agents;
- 5,000 pairs against each of the three frozen Exp10 reference policies;
- paired standard LBR/full-flop-response scoring on 500 pairs, with 4,096
  preflop rollouts and two computation threads, using the shared evaluation suite.

The pilot's frozen neural panel is UCV-only. Cross-family VR-Deep/SD-CFR comparisons
can use the emitted native policies in the existing retrospective evaluations;
they are not silently included in this runtime budget. Full-flop response keeps
the existing approximate preflop response, so its whole-game score remains an
estimated lower bound, not exact exploitability.

Primary contrast: **preserved minus ordinary learned-response payoff** (negative
is favourable). Secondary contrasts: preserved minus adversarial/unchanged,
full-flop response, head-to-head and the external panel. All arms and diagnostics
are retained even when unfavourable. No broad significance claim from multiple
secondary comparisons. A lower approximate response score does not establish a
lower true maximum: report gains as resistance to these tested attackers.

`summary.json` reports per-source-seed means, a Student-t interval over the three
seed means conditional on shared deals, and a separate Monte Carlo interval over
deal means conditional on the trained models. These are **not joint uncertainty
intervals**; correlated matches are not treated as independent training seeds.
One attacker-training realisation per source/arm is a pilot limitation. Small
effects may need more deals, more source seeds and attacker-training repetitions.

## Cost, jobs and diagnostics

The recommended `run` action submits one Batch task on a single
`n2-standard-48` VM with a 200-GB balanced boot disk and zero automatic retries.
It runs all four execution-smoke phases, then full qualification, training,
evaluation and aggregation, with a barrier after every phase. No cloud controller
or child jobs are submitted; the laptop is not needed after submission.

Qualification and training retain three concurrent workers, one per source seed.
Each uses two Torch threads, batching but **not Ray**. Qualification has a two-hour
process ceiling. Training has a 14-hour process ceiling; nominal adversarial work
is 3.25 hours per updated adversarial arm, plus the generous ordinary-fitting
control. Evaluation retains 12 concurrent workers, each with a six-hour process
ceiling and two Torch threads. Aggregation has a 15-minute ceiling. Every smoke
worker has a 15-minute ceiling. RSS caps remain 56,000 MiB per qualification/
training worker and 13,000 MiB per evaluation/aggregation worker.

The entire Batch task has a **26-hour cap**, including separately bounded setup
and final upload. Budget ~12–24 elapsed hours plus provisioning if gates pass;
this is a planning estimate, not a benchmark of this combined job or a throughput
guarantee. All 48 vCPUs stay allocated during smoke, aggregation and phase tails,
so billed VM time does not scale down between phases. The scientific budgets and
worker concurrency are unchanged. The legacy separate-stage implementation still
uses three `n2-standard-16` qualification/training VMs, twelve `n2-standard-4`
evaluation VMs and one `n2-standard-4` aggregation VM.

Each worker has a process-tree RSS/wall-time watchdog, recorded stdout/stderr,
10-second training progress, periodic cloud uploads and an independent final
upload runnable. On OOM/time limit/error, partial files remain diagnostic-only;
failed stages cannot unlock the next phase. Cloud Logging and final cgroup/kernel
diagnostics supplement process RSS. No monitoring can survive complete VM loss.

Source-only bundles pin both checkouts and all source file hashes, avoiding native
package-name collisions. Python 3.11.13, CPU Torch 2.7.0, NumPy 1.26.4 and OpenSpiel
1.6.3 are installed using the established Batch bootstrap. Policies are loaded
only from trusted hash-checked experiment artefacts (pickle is not a safe format
for untrusted files). Preparing or submitting never requires a GitHub login.

## Running from the UCV repository root

Have `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL` and an up-to-date sibling
`fhp-evaluation-suite` checkout. Optionally set `FHP_EVAL_REPO` explicitly.
Use a fresh run ID and one command for the entire sequence:

```bash
export RUN_ID="exp21-adv-full-$(date -u +%Y%m%d-%H%M%S)"
unset FHP_BATCH_OUTPUT_DIR
bash gcp/run_exp21_adversarial_preservation.sh run
```

`run` creates the immutable source bundle and `run_job.json` under
`outputs/batch/$RUN_ID`, checks that the cloud destination and job name are unused,
uploads inputs and submits `$RUN_ID-run`. Keep this directory unchanged. No push,
`REPO_REF`, `SMOKE_RUN_ID` or subsequent submission is required. The integrated
smoke exercises qualification, repair, all four evaluation arms and aggregation
with the identical bundle before any full work starts. The old qualification-only
smoke is not a substitute. Source bundling includes local edits in both checkouts.

Optional: `prepare-run` creates that bundle offline, then `run` with the same ID
submits it. `run-smoke` on a fresh ID runs only the integrated smoke (not scientific
qualification). These actions are not interchangeable with old separately
prepared runs. A new run ID is required for a new bundle; no automatic resume or
retries are implemented.

```bash
bash gcp/run_exp21_adversarial_preservation.sh status
gcloud storage cat "$BUCKET/$RUN_ID/STATUS.json"
```

`STATUS.json` appears after bootstrap and is periodically uploaded; it records
mode, phase, completed tasks/stages and failure details. A full result requires
Batch `SUCCEEDED`, `STATUS.json` with `scientific_complete: true`, and
`analysis/SUCCESS.json`. Overall success markers are excluded from periodic
uploads and published only after a successful final sync. Smoke analysis is
isolated under `smoke/analysis`; full aggregate analysis remains under `analysis/`.
Failure cancels other active workers in that phase and never starts successors.
If an attacker-strength gate fails, stop and review rather than repeatedly testing
under new IDs. This is an inconclusive scientific qualification, not a software
failure or evidence of low exploitability.

## Legacy separately submitted phases

The following workflow remains supported for existing separately prepared runs.
It is **not needed** for the recommended single-job workflow above.
No job is submitted by legacy `prepare`; subsequent stages each require an
explicit command. Keep the prepared directory unchanged.

```bash
export RUN_ID="exp21-adv-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp21_adversarial_preservation.sh prepare-smoke
bash gcp/run_exp21_adversarial_preservation.sh qualify-smoke
```

Wait for qualification `SUCCEEDED`, then run each next command **only after the
previous job succeeds**:

```bash
bash gcp/run_exp21_adversarial_preservation.sh train-smoke
# Wait for SUCCEEDED.
bash gcp/run_exp21_adversarial_preservation.sh evaluate-smoke
# Wait for SUCCEEDED.
bash gcp/run_exp21_adversarial_preservation.sh aggregate-smoke
```

The smoke reads seed 0's real historical state, so download/setup/grouping are
not instantaneous. Learning/scoring budgets alone are tiny; it tests plumbing,
not attack strength. Full qualification requires all four smoke jobs to succeed
with the **identical source bundle**. Code changes require a new smoke.

```bash
# Preserve SMOKE_RUN_ID from the completed smoke above.
export RUN_ID="exp21-adv-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp21_adversarial_preservation.sh prepare
bash gcp/run_exp21_adversarial_preservation.sh qualify
bash gcp/run_exp21_adversarial_preservation.sh status
```

Inspect qualification results, then continue only if all three gates pass:

```bash
bash gcp/run_exp21_adversarial_preservation.sh train
# Wait for SUCCEEDED before evaluation.
bash gcp/run_exp21_adversarial_preservation.sh evaluate
# Wait for SUCCEEDED before aggregation.
bash gcp/run_exp21_adversarial_preservation.sh aggregate
```

Every stage refuses an occupied output namespace and has zero automatic retries.
An inconclusive qualification should prompt analysis, not repeated reruns under
new IDs until a gate passes. A failed stage needs diagnosis before a new attempt.

## Downloading results (either workflow)

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Full per-seed results, checkpoints and diagnostics are under
`$RUN_ID/{qualify,train,evaluate}/task_N/analysis/`. Final policy files are
`$RUN_ID/train/task_SEED/analysis/{unchanged,ordinary,adversarial,preserved}.pkl`.
Download those explicitly if needed; aggregate analysis does not duplicate
multi-GB replay caches.
