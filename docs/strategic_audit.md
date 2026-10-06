# FHP strategic-position audit v1

This is a **diagnostic**, not exact whole-game exploitability and not a quiz with
universal preferred poker actions. It changes no training algorithm. It supports
native UCV, VR-Deep and the complete own-reach-weighted SD-CFR behavioural mixture.
No historical SD-CFR models are discarded. No cloud job is submitted by these
commands. Run locally for validation, then on a staged CPU VM for trained models.

## Scientific contract

FHP is two-player fixed-limit poker with two private cards and a final three-card
public board. There is no turn/river, future draw equity or arbitrary bet sizing.
Every final-round action branch and compatible hidden hand is integrated exactly
(up to floating-point arithmetic). Optimisation occurs **after** hidden hands
are integrated, so it never grants the player knowledge of the opponent's cards.

Three different measurements are deliberately not conflated:

1. **Provable errors:** folding an unbeatable hand at a final call-or-fold
   decision. The unbeatable condition covers every compatible opponent hand,
   independently of the chosen opponent's support. Calling with a hand that
   always loses is classified separately as a *range-conditioned* error.
2. **Local decision gap:** maximum action value minus the deployed action
   distribution's value, keeping the candidate's own subsequent decisions fixed.
3. **Remaining-round response gap:** an optimal continuation against the frozen
   opponent minus the deployed continuation. Root values are reported separately
   from per-decision gaps. Do not sum these gaps along a trajectory.

All values use big blinds (100 chips) per audited decision, not winnings per
whole-game hand. Action values include sunk contributions; value **differences**
are decision costs. Terminal call equity includes half the tie probability, and
its threshold is call cost / (pot before call + call cost). This shortcut is
applied only when the legal choice is fold/call and calling ends at showdown.
Checking the nuts and bluffing weak hands are NOT blanket failures.

Best-response opportunities against a particular opponent are not necessarily
errors in equilibrium play: the tested policies are not informed of opponent
identity. This audit cannot certify convergence to equilibrium. Use independent
head-to-head and exploiter evaluation for claims about overall playing strength.

## Frozen coverage and holdout

`strategic-library` freezes 96 non-isomorphic boards: 24 each unpaired monotone,
two-tone and rainbow; 8 each paired two-tone, paired rainbow and trips rainbow.
These are the six feasible suit/pairing strata. Rank configurations are sampled
deterministically without replacement within strata. Each selected suit class
receives a seeded random suit-label assignment, avoiding systematic preference
for low-index suits when testing raw-input networks. All seven legal preflop
entry histories, both seats, every reachable public final-round decision and all
1,176 private hands per board are audited. Card blockers remove incompatible
opponent hands. Coverage does not imply exhaustive full-deck board evaluation.

The board classes split into **64 development and 32 locked assessment boards**.
Assessment requires a separate output directory and `--unlock-assessment`.
Inspect development results, freeze model selection, then unlock once. Reusing
assessment results to tune candidates consumes that holdout. This is a holdout
from audit-driven selection, not a claim about unseen training cards. Suit
permutations do not create additional independent samples.

The balanced score gives each supported information-set observation equal
weight. It is not a natural visitation distribution. Overlapping category tags
(made hand/rank, pair kicker, over/underpair, board-pair interaction, pot, seat,
remaining raises and full history) must not be added as disjoint partitions.
Unsupported opponent ranges are counted and excluded, never replaced with a
uniform range. The same opponent and positions are used for every candidate.
The core includes counterfactual decisions that a candidate might never reach
under its own policy; these are not presented as observed whole-game losses.

`--include-reference` also evaluates a fixed visitation view. The library records
independent seven-card deals and action seeds; a frozen reference policy plays
the fixed opponent panel in both seats. Candidates cannot select the visited
positions. Both seats share the same action stream and swap private hands.
Preflop folds contribute no flop position. Sampling is uniform over deals
**conditional on excluding assessment board classes**, not over board classes.
Future sampled board cards are never included in preflop network inputs.

The default 64 deal pairs per opponent are a **small diagnostic sample**, not
sufficient evidence for precise population estimates. Increase `--reference-deals`
before freezing a production library after measuring inference costs. Each extra
board needs probability tables. No Monte Carlo confidence interval is fabricated
from the correlated decisions. Seed-level mean/SE summaries are descriptive and
conditional on the fixed suite/panel; three seeds have limited inferential power.

## Opponents and sensitivity

The provided Exp9/10 spec uses the existing five corrected published rule agents
plus three frozen Exp9 24h policies. Exp9 seed0 generates the reference-position
distribution. Its own comparisons with that anchor are self-play diagnostics,
not independent opponent confirmation. Keep the panel frozen across candidates.

Opponent ranges use its own observed action likelihoods and compatible cards;
the candidate's own reach is not inserted into the hidden-opponent posterior.
`range_tilts: [0, -2, 2]` adds synthetic weak-/strong-range stress tests: multiply
opponent reach by exp(tilt * uniform showdown edge), then normalise separately
for each compatible hero hand. This is a stipulated prior perturbation, **not** a
learned range or another observed opponent. Continuation remains the frozen
opponent policy. Main conclusions use tilt 0; report sensitivity separately.

## Validate and run an eight-board smoke (no checkpoints needed)

From the evaluation-suite checkout, in its normal Python environment:

```bash
python -m pip install -e . pytest
python -m pytest -q tests/test_strategic_audit.py

python -m fhp_evaluation.cli strategic-validate --output strategic_validation.json
python -m fhp_evaluation.cli strategic-library \
  --pilot --reference-deals 1 --output strategic_pilot_library.json
python -m fhp_evaluation.cli strategic-spec --smoke --output strategic_smoke_spec.json
python -m fhp_evaluation.cli strategic-run \
  --spec strategic_smoke_spec.json --library strategic_pilot_library.json \
  --include-reference --output-dir outputs/strategic_smoke \
  --threads 2 --max-seconds 300 --max-rss-mb 4096
```

The pilot contains eight development boards plus the optional independent
reference deal's board. It is execution/correctness evidence, not policy-quality
evidence. Generated library/spec/validation files refuse to overwrite existing
files. Native checkpoints are trusted pickle/Torch inputs: do not load untrusted
checkpoints. Runtime and source revisions are recorded.

## First trained-model comparison: Exp9 versus Exp10

Stage the native UCV repo and the six **playable policy** checkpoints preserving
their `workers/.../checkpoints/...seed_N_time_24h.pkl` paths beneath the run roots.
Full training states are unnecessary. The source runs are
`exp9-cache24-20261001-132550` and `exp10-features-20261001-161740`.
Set absolute paths appropriate to the VM:

```bash
export UCV_REPO="/absolute/path/to/fhp-ucv-escher-experiments"
export EXP9_ROOT="/absolute/path/to/exp9-cache24-20261001-132550"
export EXP10_ROOT="/absolute/path/to/exp10-features-20261001-161740"

python -m fhp_evaluation.cli strategic-spec \
  --ucv-repo "$UCV_REPO" --exp9-root "$EXP9_ROOT" --exp10-root "$EXP10_ROOT" \
  --output strategic_exp9_exp10.json
python -m fhp_evaluation.cli strategic-run \
  --spec strategic_exp9_exp10.json --library strategic_pilot_library.json \
  --output-dir outputs/strategic_exp9_exp10_pilot \
  --threads 2 --max-seconds 7200 --max-rss-mb 8192
```

The generator checks there is exactly one checkpoint per seed/run and pins its
SHA-256. Verify the staged source identities; it cannot infer whether a mislabeled
directory contains the intended scientific experiment. The pilot times each
native policy export and board analysis. Profile SD-CFR separately before adding
it. The time cap is **per policy-export/analysis attempt**, not the whole audit.
Caps are safety limits, not runtime forecasts. Production Exp9/10 with eight
opponents and three range settings performs substantially more work than smoke.

After the real-policy pilot, freeze the full library and run development:

```bash
python -m fhp_evaluation.cli strategic-library \
  --reference-deals 64 --output strategic_library_v1.json
python -m fhp_evaluation.cli strategic-run \
  --spec strategic_exp9_exp10.json --library strategic_library_v1.json \
  --include-reference --output-dir outputs/strategic_exp9_exp10_development \
  --threads 2 --max-seconds 43200 --max-rss-mb 8192
```

Use the same command with `--resume` after interruption. Successful checksummed
board tables/results are reused; the incomplete board is recomputed. A new worker
attempt retains old failure logs. Changed inputs, runtime versions, code or thread
count require a new output directory; increasing resource caps is allowed.
An exclusive output lock prevents concurrent writers. Native families run in
fresh processes to avoid the UCV/VR `vr_deep_cfr` package collision.

After model selection is frozen, use a new directory, omit `--include-reference`
and add `--partition assessment --unlock-assessment`. This explicit action is
recorded in the immutable manifest. No need to retrain the algorithm.

## Other checkpoints and algorithm families

The JSON spec is deliberately explicit. Each candidate has a unique
`cohort`, `training_seed` and `hours` combination, plus an id, native family,
absolute checkpoint and repo paths, and checksum. Add 36h/48h policies as new
entries with the same cohort/seed and their actual hours; training-trajectory
plots then appear automatically. Do not label seed-number matches across
independent training studies as paired experiments without further evidence.

For `vr_deep`, use its native repo and playable snapshot. For `sd_cfr`, use the
chunked-uniform archive manifest and **all its referenced chunks**, preserving
relative paths. Every chunk is hashed and the native reader verifies integrity.
Full mixture probabilities are cached as float64, not sampled historical models.
Fields `batch_size` (default 1024) and `model_batch_size` (SD default 32) bound
inference. `require_suit_invariance` is true for the supplied canonical UCV
comparison; leave it false for a raw network whose learned suit symmetry is a
diagnostic rather than an architectural guarantee.

## Outputs, cost and limitations

- `manifest.json`: inputs, partitions, policies, runtime and source identities.
- `status.json`: overall completion/failure, not just worker exit status.
- `tables/POLICY/BOARD/`: compressed portable action probabilities, per-board
  parity/privacy/order/suit checks, timing and success checksum.
- `board_results/`: resumable aggregate statistics and deterministic top cases.
- `analysis/summary.json`: all summaries, per-board statistics, source provenance,
  seed-level mean/SE, reference view and implementation checks.
- `analysis/report.html`: self-contained report with inline SVG charts.
- `analysis/*.svg`: standalone gap, pot-odds and checkpoint-trajectory figures.
- `workers/`: bounded-process logs, memory monitoring and per-attempt timing.

Download `analysis/` for interpretation. Keep tables remotely if further audits
or recovery are needed. No replay buffers, new training states, full pairwise
private-hand payoff matrices or millions of per-position JSON records are saved.
Only the largest 20 local-gap cases per comparison are retained; the probability
tables and manifest reproduce all other decisions. Rare high-gap cases can be
overrepresented in these examples, so read them beside coverage/weighted results.

The implementation reuses compiled public trees and blocker-aware vector kernels,
shares tables across opponents/candidates, caches per-board hand tags and batches
native inference. Published rule-agent flop equities are computed once per board
by the exact vector kernel; their native action thresholds and bluff logic remain
unchanged, with independent scalar parity checks. It does not yet share preflop inference across different boards,
or distribute board shards across VMs. Inference exports are sequential and
bounded in native model memory. One `n2-standard-16` can run the workflow with
conservative thread/RSS limits; whole-production runtime needs a trained-model
pilot. The default representative sample is intentionally small and adjustable.

Suit checks cover all 24 permutations of selected legal states per board; privacy
and card-order checks cover both seats. They are strong checks, not an exhaustive
proof for every information set. Independent reduced-deck validation covers all
audited decisions and hands for multiple policies, with full-deck spot checks in
the test suite. A failed check blocks table publication rather than silently
changing the policy or smoothing its probabilities.
