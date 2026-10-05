# FHP evaluation suite

This repository is the single evaluation implementation shared by
`fhp-ucv-escher`, `fhp-vr-deep`, and `fhp_deep_cfr`. It provides:

- both-seat duplicate-deal matches with pair-level Student-*t* confidence intervals;
- the five hand-strength agents used by Xu et al. (2026);
- a corrected LooseAggressive definition with increasing bands `(-300, -100)`;
- a poker-specific Local Best Response (LBR) lower-bound probe;
- collision-free checkpoint loading for all three policy formats.

The released DeepPDCFR code contains `LooseAggressive=(-100,-300)`. That
ordering makes its middle (call) branch unreachable. The suite treats it as a
transposition error and uses `(-300,-100)`. `StrengthBands` rejects reversed
thresholds and a regression test fixes the intended weak/call/strong partition.

## Specialised best-response experiments

The new `br-validate`, `br-profile` and `br-compare` commands implement
`exp1_br_validation`, `exp2_br_feasibility` and `exp3_br_checkpoint_comparison`.
See [the step-by-step guide](docs/best_response_experiments.md).

The response is exact **after the flop**, with the existing LBR decisions
preflop. Its whole-game score remains a lower-bound estimate, not exact
full-game exploitability. Native UCV, VR-Deep and full-mixture SD-CFR adapters,
bounded CPU batches, resource watchdogs and independent validation are included.
Full preflop integration is not part of this first stage. A bounded,
single-VM Experiment 9 pilot is available: see
[GCP Batch instructions](docs/gcp_batch_experiments.md).
The follow-on `exp5_ucv_exp9_br_production_3seed` freezes all three 24-hour
Experiment 9 checkpoints and evaluates 2,000 fresh duplicate pairs per training
seed in twelve resumable shards. Its primary interval treats the three training
seeds—not the 6,000 deal pairs—as the independent policy-level observations.

## Install and run

From any of the three nested training repositories:

```bash
python -m pip install -e ../../fhp-evaluation-suite
fhp-evaluate benchmark path/to/policy_checkpoint --deals 10000 --seed 2026
fhp-evaluate lbr path/to/policy_checkpoint --deals 1000 --seed 2026
```

LBR output is labelled `lbr_lower_bound`. It is the payoff of a legal,
one-step approximate response, not exact exploitability. Opponent ranges are
updated by Bayes' rule from the frozen policy's action probabilities. Flop
equity is exact; pre-flop board completion is deterministic Monte Carlo, and
the rollout count is stored in every result.

## Validation contract

Run `python -m pytest`. The tests cover card ordering, exact showdown equity,
all threshold bands, the LooseAggressive correction, policy legality,
duplicate reproducibility/antisymmetry/self-play cancellation, checkpoint
loader parity, Bayesian LBR information-set invariance, and exploitation of
controlled always-fold and always-call targets.

The pre-flop lookup is reduced from `random_board_wp.npy` distributed with the
DeepPDCFR code artefact (SHA-256
`b685e2d65cf3d772478a0ff2893993c16e6586ef9b1d7b791229862b74fb7a77`).
The 169-class reduction was independently generated and checked to be
suit-isomorphic. Flop equity does not depend on the unavailable upstream
`flop_matrix.npy`; it is independently enumerated over every legal opponent
hand.

The dated acceptance results and differential-loader checks are recorded in
[`VALIDATION.md`](VALIDATION.md).

## Frozen-policy cohort runner

`fhp_evaluation.cohort` and `cohort_report` support native policy adapters,
UCV-compatible retrospective duplicate schedules, bounded process workers,
checksum-guarded task recovery, timing forecasts, and seed-level reports.
The VR-Deep repository's `retrospective_exp1_exp2_exp3_evaluation` defines its
frozen sources and GCP launcher. Main comparisons finish before its independently
cost-gated nine-final-policy LBR stage; no policy training occurs in this suite.
