# UCV Experiment 34: exact terminal-call deployment safeguard

This is **evaluation only**, on the three SHA-256-pinned 48-hour Experiment 16
policies (`exp16-feat48-20261004-182051`). Experiment 16 is the chosen frozen
long-horizon hand/board-feature baseline, not a claim of a universally established
best policy. No weights, replay, training schedule, or source checkpoints change.

## Intervention and claim

At a final-round decision with exactly fold/call legal, require that calling
ends the hand and costs chips. Enumerate every one of the 1,081 holdings compatible
with **our** two cards and the public flop. Calculate net call/fold payoffs from
public contributions and exact five-card ranks. Transfer all fold probability to
call only when every call payoff is at least its fold payoff, with **zero safety
tolerance**. Otherwise preserve the probabilities. Ties are evaluated explicitly.
The deployed verifier never reads actual enemy cards or actual showdown returns.

Independent OpenSpiel enumeration checks 21,620 call/fold payoff comparisons,
including both seats, the known straight-flush and quads failures, nut-straight
ties, and a non-nut control. Suit permutations, dealing order, private-information
invariance, native/batched parity and literal-game coupling have regression tests.

The no-harm statement is structural: conditional on correct certification and
unchanged earlier actions, this terminal-only substitution cannot reduce expected
payoff against any **fixed** opponent, since each changed terminal action is no
worse for every possible hidden hand. It is not a learned improvement, a general
strong-hand heuristic, a guarantee about an arbitrary search algorithm, or a
statement that finite independently sampled matches cannot lose.

## Frozen evaluation plan

Full mode uses all three seeds and a common-random-number duplicate-deal protocol.

| Component | Fixed number per candidate training seed |
|---|---:|
| Safeguarded vs its own unchanged 48h version (primary) | 1,000,000 duplicate pairs |
| Each of the other two unchanged 48h seeds | 20,000 pairs |
| Each own 24h, 36h and 42h predecessor | 20,000 pairs |
| Each of the five published rule agents | 20,000 pairs |
| Each of the three frozen Experiment 9 24h policies | 20,000 pairs |
| LBR targeting unchanged / safeguarded policy | 2,000 pairs **each** |
| Full-flop responder targeting unchanged / safeguarded policy | 2,000 pairs **each** |

Every opponent is played against **both** versions. LBR uses the suite's existing
4,096 preflop rollouts. Full-flop response uses the existing exact range-vector
flop solver with LBR preflop. Hypothetical queries, including range construction,
see the safeguard when the guarded target is specified. Both target-aware probes
are cross-played against both versions, avoiding a comparison that silently
changes the opponent along with the policy. Their scores are approximate-response
payoff estimates, **not exact whole-game exploitability**. New learned DQN attacker
training (the separate opt-in Experiment 27 research procedure) is not included.

The primary test alone represents six million hands for each arm. Across all
gameplay components there are 3,804,000 duplicate pairs: 7,608,000 hands per arm.
The two arms share the same path until a changed terminal action. The evaluator
scores both literal sampled outcomes there, avoiding duplicate simulation of an
identical prefix. It also integrates the final action randomness analytically:
`original_fold_probability * (actual_call_payoff - actual_fold_payoff)`.
The actual opponent cards enter **scoring only after certification**, never the
decision. This is an unbiased lower-variance estimate of the same whole-game
fixed-opponent gain under natural visitation, not selected rare-hand performance.

Activation frequency, transferred fold mass, conditional gain, sampled gain and
integrated gain are kept separately. Pair-level uncertainty and the three
training-seed summaries are not conflated. A zero-activation sample is flagged;
an exact one-sided bound on pair activation is reported, not a claim of zero
population effect. There is no payoff- or significance-based stopping rule.

## Entire strategic-position diagnostic

Both arms receive the established 64-development-board, 64-reference-deal audit,
all public histories, eight fixed opponents, and range tilts 0/-2/+2. Exported
behavioural tables contain the safeguard, so response gaps evaluate the deployed
policy. The 32 assessment board classes stay locked. This is the existing
development diagnostic, **not** a newly untouched confirmation set. Ordinary
whole-game deals are independently sampled without selecting known failures.
The position audit must not be presented as whole-game exploitability.

## Runtime, storage and recovery

One `n2-standard-16` VM, four processes with four Torch threads each, CPU only.
Full mode first times **two actual production shards per seed/opponent**: 2,000
duplicate pairs per ordinary shard and 25 per response shard, using the production
batch size and rollout budget. These 168,600 pairs are saved in `task_results/`,
included exactly once in the prespecified final sample, and skipped by subsequent
gameplay/recovery. There is no separate disposable timing sample. Shard selection
is fixed before play; no payoff or significance result changes the evaluation.

The projection pools the two timings within each matchup before extrapolating
its fixed sample size. `analysis/profile.json` records the raw estimate, a 2x
padded estimate, per-matchup repeat ranges, measured pairs retained, and remaining
session time. The ranges describe timing variability, **not confidence intervals**.
Cold model-loading costs are included; cloud setup and the strategic audit are
not. The **30-hour padded gameplay planning target is advisory**: exceeding it
logs a warning and continues, without reducing samples or dropping components.
It is neither an additional timeout nor a guarantee that the suite will finish.

The Batch hard limit remains **48 hours per attempt**, not an ETA or an unlimited
spending authorization. The runner retains its 46-hour session budget and outer
watchdog/upload headroom; the strategic audit can use the remaining time rather
than being cut off by a separate 18-hour worker limit. Correctness, checksum,
memory and concurrent-writer checks remain mandatory. Completed gameplay shards
and strategic board caches can be recovered with the same source/configuration/
runtime/checkpoint hashes. A real timeout can still leave an incomplete run;
recovery explicitly launches another paid attempt, reusing completed artifacts.
Incomplete output is never labelled a successful full-suite evaluation.

The verifier uses bounded board-rank and information-state caches; there is no
full-game tree or unbounded list of positions. Cold-board and cached-query latency
histograms, maxima, timing forecasts and cache memory are recorded. Ineligible
decisions have a separate histogram, avoiding dilution of eligible-query latency.
Query timings
from hypothetical responder calculations are explicitly separated from natural
gameplay activation counts. Playable source policies are retained in GCS and only
read; no full training states are downloaded or emitted. New outputs are compressed
pair results, portable policy tables for the finite diagnostic boards and reports.

Results are complete only when `analysis/SUCCESS.json` exists. `analysis/summary.json`
and `report.md` cover gameplay, `strategic_comparison.json` and `strategic/analysis/`
cover the strategic audit. `latency.json` aggregates verifier overhead and
`deployment_manifest.json` pins the wrapper and original playable checkpoints;
the latter together define the deployed model without duplicating weights.
Validation/profiling and per-shard timing are retained.
Partial gameplay summaries are not a completed full-suite experiment.

## Run from the UCV repository

The sibling `fhp-evaluation-suite` must include this implementation. The launcher
bundles both codebases by hash; `REPO_REF` is not used. Existing GCP project,
region, bucket and runner service-account variables must be set.

```bash
export EXP34_MODE=smoke
export RUN_ID="exp34-safeguard-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp34_terminal_call_safeguard.sh run
```

Smoke uses seed 0 and tiny gameplay samples but exercises every component,
including the eight-board strategic pilot. Review success/correctness/timings;
it is not a performance conclusion. Then use a **new** run ID:

```bash
export EXP34_MODE=full
export RUN_ID="exp34-safeguard-full-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp34_terminal_call_safeguard.sh run
```

`prepare` is entirely local. `submit` submits an already prepared bundle. `recover`
requires the original run directory, unchanged inputs and a fresh `RECOVERY_TAG`;
it refuses completed jobs or concurrent writers. Any code/design change needs a
new run ID. Neither this implementation nor its local tests launch paid jobs.

### Migrating the failed 12 October attempt

`exp34-safeguard-full-20261012-205730-evaluate` stopped after about 8m23s when the
old 16-pair forecast was 28.0532 hours against a rigid 28-hour gate (the unpadded
estimate was 14.0266 hours). Its validation and timing artifacts remain in GCS;
no main gameplay shards or strategic-audit results were produced. That elapsed
compute cannot be reclaimed, but no trained models or long evaluation were lost.
The old immutable source bundle would fail the same gate again. Use a **fresh
run ID and preparation** for the revised code, not `recover` or `submit` against
that old bundle. Do not copy its separate timing games into the final sample:
they used a different deal stream. New runs retain profiling work as normal,
checksum-verified gameplay shards instead.
