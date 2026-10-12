# Experiment 27 — attacker learning capability and loss ablation

## Question and relationship to Experiment 21

Experiment 26's selected one-hour attacker improved by 28.7 mbb/hand over the
15-minute version, but still lost 130.5 mbb/hand to the frozen Exp10 seed-0
defender. Execution checks passed. This study tests learning capability and a
specific fitting-objective hypothesis before authorising expensive policy repair.

One invocation runs learning controls, an equal-time loss comparison and, only
if independently qualified, replication on the other two Exp10 defender seeds.
**No defender is retrained, no Exp21 gate is relaxed, and no Exp27 result can be
used as an Exp21 qualification artifact.** Even a replicated result requires a
separately reviewed revision and submission of Experiment 21.

## Frozen objectives and matching

The two arms are:

| Arm | Loss for sampled Double-DQN targets |
| --- | --- |
| `smooth_l1` | Original beta=1 Smooth L1 |
| `half_mse` | One half of squared error |

The factor one half matches the original loss's small-error curvature. Large
errors remain fully quadratic, so their gradient influence intentionally differs.
Both retain gradient-norm clipping at 10. Architecture (256x256), Adam learning
rate, replay size, warm-up, minibatch size, target synchronisation, sampling and
thread count remain unchanged from the full Exp21/26 learner. Existing experiments
continue to default to Smooth L1 and FHP returns in big blinds.

Each cell trains two repetitions per seat: four networks, four concurrent
processes, two CPU threads each. The two arms start from identical weights and
RNG seeds for corresponding case/defender/seat/repetition. They run in separate
four-worker blocks on the same VM. Replay and minibatches naturally diverge as
policies diverge: this is an **online equal-time comparison**, not a frozen-data
or equal-update comparison. Actual hands, updates and all overhead are retained.
Common random seeds do not imply identical trajectories. Initial model hashes
are auditable in every training manifest.

A preflight enumerates a finite stochastic target: -1 with probability .8 and
+4 with probability .2. The expectation is zero; half-MSE minimises there while
Smooth L1 minimises at -.75. This demonstrates a possible objective mismatch,
**not evidence that this mechanism explains the observed FHP failure**.

## Stage A: learning-capability controls

Both objectives get five active training minutes per network on each control:

1. **Leduc against uniform legal actions.** Use OpenSpiel's lossless information
   tensor, with returns in Leduc chips, not FHP big blinds. OpenSpiel
   `BestResponsePolicy` with zero pruning calculates an exact response value
   for each seat. Exhaustive policy-value evaluation measures every initial and
   final learned attacker, its improvement, remaining gap and fraction of the
   exact response value. Exact tree enumeration is restricted to Leduc.
2. **FHP fold-prone opponent.** Fold with probability .8 and call with .2 whenever
   folding is legal; otherwise check/call. Never inspect hidden cards. The .2
   continuation is essential: an always-folding first player would end every
   seat-1 training hand before the learner makes any decision.
3. **FHP calling-only opponent.** Always check/call, irrespective of cards.

Weak FHP controls use the same 213-feature encoder as Exp10. Each initial and
final repetition receives 2,000 held-out duplicate pairs, shared across arms and
repetitions. For the pooled control interval, average the two fixed repetitions
within each deal first; do not count those shared deals twice.

An objective is eligible for selection if both FHP controls have pooled positive
payoff lower 95% bounds, and both Leduc repetitions have positive exact mean
payoff with a positive mean gain over their initial policies. This is a modest
capability/resource criterion, **not a requirement or claim of reaching the exact
Leduc response**. Exact remaining gaps are always reported. Failure means no
demonstrated capability under this budget, not an algorithmic impossibility.

If neither objective passes, publish the full negative diagnostic and finish
successfully without the expensive Exp10 stages. If either passes, train both
arms in Stage B so the comparison retains the original baseline. An ineligible
arm remains a measurement/control, but cannot become the selected candidate.

## Stage B: original Experiment 10 seed 0

Download only the pinned seed-0 24h policy and manifest, not its large training
state or replay. Repeat Exp26 correctness checks (rewards, transitions, illegal
actions, hidden-card invariance and native/collection inference parity).

Train fresh attackers for **60 active minutes** per network and objective, with
snapshots at **0, 15 and 60 minutes**. Exploration still reaches .05 at 12 minutes;
do not stretch the original 900-second horizon. Keep replay, optimizer and target
continuous within each run. Snapshot serialisation is excluded from active time
and charged separately; all evaluation occurs after fitting.

Development: 2,000 duplicate pairs per objective/repetition/snapshot, on a common
panel. Select each arm's best final repetition and the best eligible objective
using only development mean payoff. Ties prefer Smooth L1, then repetition zero.
Write `selection_lock.json` before accessing the independent final stream.

Final assessment: each arm's locked final attacker and its own initial policy
receive **10,000 fresh duplicate pairs**, with common chance/action seeds across
candidates. Primary contrast is half-MSE minus Smooth L1. Also report absolute
payoffs, gains over initial policies, both seats, action frequencies and descriptive
Q/return calibration. Training, control assessment, development, final and
calibration streams use separate Exp27 namespaces.

The **locked** objective qualifies for replication only when both its held-out
payoff and paired gain over initial have positive lower 95% confidence bounds.
Do not switch to the other arm after looking at the final results, keep extending
training until it passes, or recycle the final panel for tuning.

## Stage C: conditional replication

Only after Stage B qualifies, download the pinned Exp10 seed-1 and seed-2 policies.
Repeat correctness checks, and train **only the locked objective**, from fresh
weights, for one hour per network on each defender. Two repetitions per seat,
development-only repetition selection and independent 10,000-pair final tests
are repeated unchanged. Both defender replications run even if seed 1 is negative,
so a partial replication cannot be selectively reported.

If all three defenders qualify, report `replicated_candidate_requires_human_review`.
Otherwise report the appropriate negative/inconclusive outcome. This replicates
attacker usefulness, **not the treatment effect of MSE versus Smooth L1**: that
would require both objectives on all source seeds. No meta-analytic interval treats
deal pairs as independent training seeds. No exact FHP exploitability or Nash
convergence claim is made. Failed attackers do not establish defender safety.

## Execution, costs and artifacts

One `n2-standard-8`, 32 GB nominal RAM, 60-GB disk, four two-thread fitting workers
at most. Each worker has a 6,000-MiB RSS watchdog; stdout, stderr, resource time
series, phase status, immutable inputs and runtime versions are retained. No
automatic retries, child Batch jobs, checkpoint reuse or overwrite. The coordinator
has a seven-hour limit; the full Batch job an **eight-hour hard safety cap**.

Planning estimates, not runtime guarantees:

- Controls only: around 30 minutes of fitting plus setup/assessment.
- Through seed-0 comparison: around 2.5 hours of fitting plus setup/assessment.
- With both conditional replications: around 4.5 hours of fitting; allow roughly
  5–6 hours overall. All counts and summed worker seconds are reported separately
  from coordinator wall time and cloud setup/upload.

`analysis/report.md`, `summary.json` and `SUCCESS.json` are sufficient for review.
The summary includes all curves, raw paired results, exact-control measurements,
training manifests, preflights and costs. Per-cell logs and `.pt` snapshots remain
under `cells/`; locks under `assessments/` and the root. Snapshots contain networks,
target, Adam and RNG metadata **but not replay**, so are playable artifacts, not
exact-resume training checkpoints. Do not claim continuation of Exp26.

Negative scientific outcomes are successful diagnostics, with
`policy_repair_authorised: false`. Corruption, non-finite fitting, correctness
failure or resource-limit violations remain execution failures and block later
stages. On failure, logs and already completed worker artifacts remain available.

## Run from the UCV repository root

Set the usual `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL`; update both local
repositories. The launcher pins a source bundle of both checkouts; `REPO_REF`
and any previous `SMOKE_RUN_ID` are not used.

```bash
export RUN_ID="exp27-attack-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp27_attacker_development.sh run-smoke
bash gcp/run_exp27_attacker_development.sh status
```

Smoke uses tiny training/deal budgets and explicitly exercises all replication
branches regardless of tiny-sample scores. It can never scientifically qualify
an attacker or set `ready_for_exp21_design_review` true. After reviewing smoke
success, submit separately with a fresh ID:

```bash
export RUN_ID="exp27-attack-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp27_attacker_development.sh run
bash gcp/run_exp27_attacker_development.sh status
```

`prepare` and `prepare-smoke` are offline previews; `submit` and `submit-smoke`
submit an unchanged prepared bundle. `EXP27_RUN_DIR` can override the prepared
directory; `FHP_EVAL_REPO` can identify a nonstandard suite checkout.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
