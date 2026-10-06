# Specialised FHP best-response experiments

This is the first implementation stage: native policy adapters, an exact final-
round response engine, independent correctness checks, bounded CPU profiling,
and paired comparisons with the existing LBR. No training algorithms are changed.

## What is (and is not) exact

`FullFlopResponsePolicy` uses **unchanged LBR decisions preflop** and an exact
best response for the entire remaining betting tree once the flop is known.
It integrates every compatible opponent hand with its action-history likelihood.
There is no hand-strength bucketing, card subsampling, suit-isomorphism merging,
learned continuation value, or future-card sampling in the flop engine.

The resulting whole-game score is an **exploitability lower-bound estimate**,
not exact full-game exploitability. A sampling confidence interval describes
uncertainty in this response's winnings, not the gap to a true best response.
Small scores do not certify proximity to Nash equilibrium. Individual sample
scores may be negative; we preserve them rather than clipping away uncertainty.

Full preflop integration over all 22,100 flops, distributed board-shard reduction,
learned responses and certified upper bounds are **not implemented** in this stage.
Do not interpret a profile over selected boards as an exploitability estimate.

## Install

From the evaluation-suite checkout, using Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

Workers require the repository's pinned `open_spiel==1.6.3`. Checkpoint formats
are trusted-input formats: Python pickle and Torch loading can execute code.
Use only checkpoints you trust.

Native checkpoints require the corresponding training checkout and its runtime
dependencies. Supply its absolute path with `--repo-root`; this reuses the actual
encoder and architecture rather than reimplementing their semantics. Native code
and checkpoint hashes are recorded with every run. Do not alter a checkpoint or
native checkout while its evaluation is running.

## Experiment 1 — `exp1_br_validation`

```bash
python -m fhp_evaluation.cli br-validate \
  --output-dir outputs/exp1_br_validation \
  --max-seconds 120 --max-rss-mb 2048
```

Checks an independent information-set best-response reference against OpenSpiel
on Kuhn and Leduc (uniform, nonuniform, and CFR+-trained policies). Separately,
it checks the vector flop engine against explicit OpenSpiel terminal returns on
eight-card FHP, across both seats, five control policies and three betting lines.
This is a correctness experiment, not a full-deck scalability claim.

The broader regression suite includes full-deck spot checks, hidden-card
invariance, payoff/blocker arithmetic, batching, cache bounds and watchdog tests:

```bash
python -m pytest -q
```

Enable native integration tests by setting `FHP_UCV_REPO`, `FHP_VR_REPO` and
`FHP_SD_REPO` to the absolute paths of their training checkouts:

```bash
python -m pytest -q tests/test_best_response_native.py
```

These tests use fresh processes and generated fixture checkpoints. They compare
native scalar versus batched outputs for UCV raw/encoded/structured/residual-card
architectures, VR-Deep raw/encoded snapshots, and raw/encoded SD-CFR mixtures.

## Experiment 2 — `exp2_br_feasibility`

Cheap execution smoke, without loading a trained checkpoint:

```bash
python -m fhp_evaluation.cli br-profile \
  --family control --control uniform \
  --threads 1 2 --boards 1 --repeats 1 \
  --output-dir outputs/exp2_br_control_smoke \
  --max-seconds 60 --max-rss-mb 2048
```

Real UCV checkpoint, assuming `FHP_CHECKPOINT` and `FHP_NATIVE_REPO` have been set:

```bash
python -m fhp_evaluation.cli br-profile "$FHP_CHECKPOINT" \
  --family ucv --repo-root "$FHP_NATIVE_REPO" \
  --threads 1 2 4 8 --boards 4 --repeats 3 \
  --batch-size 1024 \
  --output-dir outputs/exp2_br_ucv_profile \
  --max-seconds 600 --max-rss-mb 8192
```

For VR-Deep change the family to `vr_deep` and point to its native repository and
policy snapshot. For SD-CFR use `sd_cfr`, a chunked-uniform archive manifest,
and its native repository. **All referenced archive chunks are required** and
their hashes are checked. `--model-batch-size 32` bounds SD-CFR model batching;
no historical models are discarded. Other SD-CFR archive formats are not silently
converted or approximated.

Each thread configuration runs **sequentially in a separate process**, with the
same board sample, one excluded warmup, and repeated complete flop sweeps over
all seven preflop entry lines and both responder seats. The thread argument is
computation threads, not Ray workers. Each process has its own runtime/RSS cap;
four configurations can therefore consume up to four times `--max-seconds`.

Outputs contain per-board times, policy-query time/rows, peak RSS, native-scalar
parity results and target-loading time. A projected full-board runtime is a
planning estimate only: it excludes full preflop integration and scheduling and
is not a validated cloud runtime forecast. Profile SD-CFR separately: its full
historical mixture may be much more expensive than a single policy network.

Use `--resume` with exactly the same arguments to reuse successful configuration
shards. Changed inputs, code or limits are rejected. Failed/interrupted shards
currently require a **new output directory**; there is no mid-board recovery.

## Experiment 3 — `exp3_br_checkpoint_comparison`

Run only after the correctness and real-checkpoint feasibility gates:

```bash
python -m fhp_evaluation.cli br-compare "$FHP_CHECKPOINT" \
  --family ucv --repo-root "$FHP_NATIVE_REPO" \
  --threads 1 --deals 1000 --rollouts 4096 \
  --seed 20261004 --search-seed 731 \
  --output-dir outputs/exp3_br_ucv_comparison \
  --max-seconds 3600 --max-rss-mb 8192
```

This is a **capped request**, not a promise that 1,000 pairs fit within one hour.
Use `--deals 2 --rollouts 8` for execution smoke only. Choose the final deal count
from pilot variance/precision and cost, not simply the smoke defaults.

The run evaluates existing LBR and full-flop response on identical independent
duplicate-deal pairs. It reports both winnings and the **paired difference**,
with pair-level Student-t intervals, separate seat payoffs and raw pair scores.
Units are milli-big-blinds per hand; the game uses a 100-chip big blind. Evaluate
training seeds separately and retain seed-level results; thousands of deals are
not thousands of independent training seeds. Use fresh evaluation seeds for final
reporting after any evaluator-setting selection.

Run each checkpoint/family in a separate output directory and process. Never
pool individual historical SD-CFR networks' exploitabilities: that is not the
exploitability of the deployed mixture.

## Resource safety and outputs

Each process directory has:

- `spec.json`: requested settings;
- `run.json`: status, source/checkpoint identity, exit code, elapsed time and peak RSS;
- `resources.jsonl`: process-tree RSS sampled every 0.1 seconds;
- `stdout.log`: progress events, including completed boards or deal pairs;
- `stderr.log`: tracebacks and native runtime messages;
- `result.json`: successful output only.

Possible failures include `time_limit`, `memory_limit`, `monitor_error` and
`failed`. The parent terminates the worker's own process group at a limit, with
a two-second grace period before forced shutdown. The RSS limit is a polled
watchdog, **not an OS allocation cap**: short-lived spikes may exceed it. Do not
set it close to a VM's physical limit. If monitoring is denied, execution stops
rather than silently running without protection.

Native UCV and VR repositories both contain a package named `vr_deep_cfr`.
The loader refuses conflicting package origins. The CLI's fresh workers avoid
this collision; do not load both families directly into one Python process.

No cloud jobs are submitted by these commands. They can run on an existing CPU
VM with the same environment and staged native repositories/checkpoints.
[The separate GCP Batch launcher](gcp_batch_experiments.md) implements both the
bounded Experiment 9 seed-0 pilot (`exp4_ucv_exp9_br_pilot`) and its frozen,
three-seed production follow-up (`exp5_ucv_exp9_br_production_3seed`). The latter
uses twelve deterministic/resumable shards and reports policy-level uncertainty
over training seeds separately from conditional deal-level Monte Carlo precision.
Matching deals are averaged across the three frozen models before estimating
the latter interval, preserving common-random-number covariance. Both intervals
are conditional, not a joint bound over all sources of randomness. A small
response payoff does not certify proximity to Nash equilibrium.

## Design references

- Johanson et al. (2011), [Accelerating Best Response Calculation in Large Extensive Games](https://poker.cs.ualberta.ca/publications/ijcai2011_accelerated_best_response.pdf): public-state ranges and blocker-aware terminal evaluation.
- Rudolph et al. (ICLR 2026), [Reevaluating Policy Gradient Methods for Imperfect-Information Games](https://www.mit.edu/~gfarina/2026/iclr26_reevaluating/iclr26_reevaluating.pdf): compact exact evaluation; its benchmark package does not implement FHP.
- Timbers et al. (IJCAI 2022), [Approximate Exploitability: Learning a Best Response](https://www.ijcai.org/proceedings/2022/0484.pdf): a possible later learned-response extension, not implemented here.

The numerical kernel is independently implemented for this repository. No
reported speedup from another game's solver is assumed to transfer to FHP.
