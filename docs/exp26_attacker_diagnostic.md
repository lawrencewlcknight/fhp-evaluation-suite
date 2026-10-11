# Experiment 26 — diagnose and extend the Experiment 21 attackers

## Motivation and scope

`exp21-adv-full-20261011-220953-run` completed its execution smoke and all three
qualification measurements, but its trained attackers still lost to the frozen
Experiment 10 defenders. Their mean payoffs were -184.75, -175.625 and -198.5
mbb/hand; all three 95% intervals were negative. Gains over initial attackers
were positive. The pipeline correctly stopped before full policy repair.

This development study asks whether the attacker implementation is internally
consistent and whether extending its training improves held-out play. It does
**not** relax Experiment 21's gate, refit a UCV policy or establish exact
exploitability. A completed diagnostic with losing attackers is a valid negative
result, not a Batch execution failure. Exp21's scientific configuration remains
unchanged; its learner gains only opt-in observational snapshot support.

## Frozen protocol

- Defender: the original Exp10 24-hour **seed 0**, with the existing pinned
  manifest and checkpoint hashes. Download only its playable policy, not the
  5.4-GB training state or policy replay. Source weights remain immutable.
- Four fresh attackers: two independently initialised networks per seat. Use
  the same attacker seeds as Exp21 seed-0 qualification, the same 213-input
  encoder, 256x256 Double-DQN, two Torch threads, optimizer, replay capacity,
  batch size, target sync and collection settings. Different concurrent load
  and instrumentation mean equal wall time need not reproduce historical weights.
- Save each attacker at **0, 5, 15, 30 and 60 active training minutes**. Snapshots
  occur after complete collection/update waves. Persist actual elapsed time,
  episode counts, update counts, replay occupancy, hashes and loss diagnostics.
- Keep exploration decay tied to the original 900-second horizon: epsilon
  reaches 0.05 after 12 minutes and remains there through the extension. Do not
  stretch exploration decay to 60 minutes. Preserve replay, Adam and target
  state continuously across the 15-to-60-minute extension.
- Snapshot serialization/checksum time is excluded from active training time
  and reported separately. Evaluation runs **after all fitting**, never inside
  the training loop or as a feedback signal. No payoff-dependent stopping.
- One `n2-standard-8`, 32 GB nominal RAM, 60-GB disk. Four concurrent training
  processes, two Torch threads each; assessment is a separate process. No Ray,
  automatic retries, paid child jobs or policy-repair stages. Each process has
  a 6,000-MiB RSS watchdog. The full Batch job has a **six-hour safety cap**.
  Planning allowance: roughly 1.5–3 hours including evaluation/setup, not a
  measured cloud runtime guarantee. Four one-hour trainings overlap.

## Fail-fast correctness and behavioural diagnostics

Before long training, test real OpenSpiel FHP hands for:

1. Known fold rewards in chips, both seat signs, zero-sum returns and conversion
   to big blinds. Calling-only showdowns exercise terminal returns as well.
2. Reconstruction of production-collected histories with scalar policy queries,
   actor-only information-state features, next-own-decision transitions and
   terminal rewards. Terminal rows must not bootstrap.
3. The collection wrapper's action probabilities versus the independent native
   checkpoint loader on a batch of visited states, including both players.
4. Hidden-opponent-card invariance for each seat and explicit masked Double-DQN
   target algebra, including illegal high-valued actions and terminal states.
5. An unchanged source-model hash throughout diagnostics and fitting.

These tests are not a proof of statistical consistency of function-approximation
RL or of attacker optimality. The evaluation also checks reloaded attacker
snapshots against the collector. Behavioural outputs include greedy action
counts by seat/street and Q-versus-realised-return calibration on independent
greedy-policy rollouts. Q diagnostics are descriptive, not training targets;
multiple decision rows from one hand are correlated.

## Development curves, selection and independent final assessment

Form two paired-seat candidates by pairing seat-0 repetition i with seat-1
repetition i. For **each repetition at each snapshot**, evaluate **2,000
duplicate deal pairs** on the same development chance/action stream. Save both
seat payoffs, aggregate payoff, paired gain over the corresponding initial
attacker, and action frequencies. This is 10 candidate evaluations / 40,000
hands. Per-seat scores include positional/blind asymmetry: do not interpret a
single seat's raw payoff as exploitability.

After all curves are complete, choose the best repetition at 15 minutes and at
60 minutes **separately**, using development mean payoff only. Break ties by
the lower repetition index. Save an immutable selection record before using
any final-assessment deals. The 60-minute winner's earlier 30-minute and initial
snapshots are fixed secondary comparators; the 15-minute comparator is its own
development-selected winner, which can be a different repetition.

Evaluate those four fixed candidates on **10,000 fresh duplicate pairs each**
(80,000 further hands), sharing chance/action seeds to support paired inference.
Development, final assessment, calibration and training use distinct namespaces.
There is no rerunning qualification until an interval becomes favourable.

Primary comparison: **60-minute minus 15-minute attacker payoff**, conditional
on the frozen defender and selected attackers. Secondary comparisons: 60 versus
30 minutes, 60 versus its initial weights, absolute attacker payoff, and seat
breakdowns. Report 95% Student-t MC intervals over duplicate-pair means. Shared
hands are not independent training seeds; development curves/secondary
comparisons are exploratory, not simultaneous confidence bands. Two attacker
repetitions and one defender seed cannot support broad generalisation claims.

Training runs to the fixed endpoint even if an intermediate attacker appears
weak. Correctness, source mismatch, non-finite values, corrupted snapshots or
resource limits stop the diagnostic with preserved logs. Complete negative or
inconclusive scientific results receive `SUCCESS.json` with
`policy_repair_authorised: false`. Never use this marker to bypass Exp21's gate.

## Outputs, reproducibility and costs

- `analysis/report.md` and `analysis/summary.json`: all curves, final contrasts,
  selected repetitions, per-seat/action diagnostics, calibration, correctness
  checks, runtime versions, training manifests and total costs.
- `assessment/selection_lock.json`: choices persisted before final assessment.
- `assessment/development_curves.json`, `final_samples.json`: incremental
  samples, diagnostic-only until overall completion; complete samples also
  appear in the final summary.
- `train/seatS_repR/checkpoints/*.pt`: independently playable attacker snapshots
  with Q network, target, Adam and RNG metadata. **Transition replay is not
  retained: these are not exact-resume training checkpoints.** The current
  experiment never claims exact resumption from the failed Exp21 attackers.
- `preflight/`, `train/`, `assessment/`: process-tree RSS/time samples,
  `run.json`, stdout/stderr and progress events. Root `STATUS.json`, bootstrap
  logs and final system diagnostics survive through periodic/final uploads.
- Summed worker seconds include parallel overlap and must not be mistaken for
  wall time. Batch duration adds setup, provisioning and final uploads.

Only checksum-verified trusted checkpoints are loaded. The launcher pins a
source-only bundle of both local checkouts. Neither `REPO_REF` nor the failed
run's `SMOKE_RUN_ID` selects this code. Fresh run/output IDs are required; no
automatic resume is offered. Keep failed-run artifacts for scientific records.

## Run from the UCV-Escher repository root

With `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` already set and both checkouts
updated:

```bash
# Optional new-code execution smoke; all scores are non-scientific.
export RUN_ID="exp26-attack-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp26_attacker_diagnostic.sh run-smoke

# After reviewing the smoke, use a NEW namespace for the full diagnostic.
export RUN_ID="exp26-attack-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp26_attacker_diagnostic.sh run

bash gcp/run_exp26_attacker_diagnostic.sh status
```

These are separate submissions; do not paste both launches without first
reviewing the smoke. The full diagnostic always repeats correctness checks
before its one-hour attacker runs. `prepare` / `prepare-smoke` create an offline
preview; `submit` / `submit-smoke` submit the exact prepared bundle. `run` also
accepts an already prepared matching directory. `EXP26_RUN_DIR` overrides the
default `outputs/batch/$RUN_ID`; `FHP_EVAL_REPO` overrides the sibling suite path.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Do not repeat the original full Exp21 job until this diagnostic has been
reviewed. Positive findings should be replicated on the other source seeds
before authorising costly policy repair; negative findings motivate an
attacker-learning investigation, not a claim that the defender is safe.
