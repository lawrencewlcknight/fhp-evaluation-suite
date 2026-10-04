# Validation record

## Experiment 9 Batch pilot — 4 October 2026

After adding `exp4_ucv_exp9_br_pilot`, the full suite passed **56 tests** with
all three native integration paths enabled. New checks cover checkpoint identity,
fixed pilot budgets and thread selection, stop-on-failure behaviour, partial
result persistence, source-only packaging, overwrite protection, cloud preflight
failure handling, and a dry-run that makes no cloud calls.

Both Batch shell scripts passed `bash -n`. The generated job JSON decoded
successfully using the installed Google Cloud SDK's Batch v1 `Job` schema.
This checks local schema compatibility, not cloud IAM, quotas or VM availability.

A source bundle built from the actual evaluator and UCV checkout was extracted
to a separate directory and executed in fresh processes. Validation, native
smoke, all four thread profiles and the paired-comparison smoke succeeded.
This acceptance used the already-downloaded Experiment 2 checkpoint recorded
below, with an explicit test-only hash override. **It did not evaluate Experiment
9**: that policy is not downloaded locally. The production launcher fixes
Experiment 9's seed-0 24-hour checkpoint URI and independently recorded SHA-256,
with no command-line hash override.

No GCP job was submitted, no source bundle was uploaded, and the Linux bootstrap
has not yet run on a GCP VM. The documented cloud-smoke command is the next
deployment validation step. See [the Batch run guide](docs/gcp_batch_experiments.md).

## Specialised best response — 4 October 2026

Validated with OpenSpiel **1.6.3**, NumPy 1.26.4 and PyTorch 2.7.0 in an isolated
Python environment. The complete evaluation-suite regression run passed:
**37 tests**, including all three optional native-repository integrations
(`FHP_UCV_REPO`, `FHP_VR_REPO`, `FHP_SD_REPO` set).

The independent correctness experiment completed 42 checks. Its maximum
absolute discrepancy was `2.2737367544323206e-13`:

- the explicit information-set reference agrees with OpenSpiel best responses
  on Kuhn and Leduc for uniform, nonuniform and CFR+-trained target policies;
- the range-vector flop engine agrees with independent enumeration of OpenSpiel
  terminal returns on reduced-deck FHP for both seats, five control policies
  and three preflop betting lines;
- additional regression tests cover full-52-card spot checks, blockers,
  zero-reach branches, hidden-opponent-card invariance and bounded caches.

Generated native checkpoint fixtures passed scalar-versus-batched checks for
UCV raw, encoded, structured and residual-card models (dense, DeepSets and
attention); VR-Deep raw/encoded policies; and raw/encoded SD-CFR historical
mixtures. The SD-CFR check compares the batched full reach-weighted mixture
against its independent native scalar implementation, not a sampled network.
These fixtures validate loading and inference, not learned policy quality.

The watchdog-backed commands also completed end to end:

| Local acceptance run | Result |
| --- | --- |
| `br-validate` | Passed, 42 checks |
| `br-profile`, uniform control, 1 and 2 computation threads | Both passed |
| `br-profile`, saved UCV experiment 2 checkpoint, 1 thread | Passed |
| `br-compare`, uniform control, 2 duplicate pairs, 8 preflop rollouts | Passed |

The saved UCV checkpoint had SHA-256
`bc459776f9e6dcba9efd0b3f185471ad26d7462a6e3aa9c99fb149f9d3973b55`.
Its maximum scalar/batched probability discrepancy was `1.2113618552689331e-07`.
One measured full-board sweep (all seven preflop entry lines, both seats) took
10.07 seconds, with approximately 269 MiB peak process-tree RSS. This is one
local execution smoke measurement, **not** a representative speed benchmark or
a forecast for GCP. No paid cloud jobs were launched.

Failure-path tests passed for runtime limits, memory limits, worker exceptions,
worker launch failure and successful-shard resume identity checks. The watchdog
samples RSS; it does not provide a hard operating-system allocation limit.

**Interpretation:** only the remaining flop betting response is exact. Preflop
still uses LBR. The whole-game result is an exploitability lower-bound estimate;
its sampling interval does not bound the gap to the true best response. Neither
exact full-game FHP exploitability nor a certified upper bound is implemented.
The tiny comparison run above is an execution test, not a policy-quality result.
See [the experiment guide](docs/best_response_experiments.md) for reproducible
commands, resource limits and the next-stage boundaries.

## Original suite — 30 August 2026

Validated on 30 August 2026 with Python 3.12.2, OpenSpiel 1.6.3, NumPy
1.26.4, and PyTorch 2.7.0.

## Automated suites

| Repository | Result |
| --- | ---: |
| `fhp-evaluation-suite` | 18 passed |
| `fhp-ucv-escher-experiments` | 23 passed |
| `fhp-vr-deep-experiments` | 13 passed |
| `fhp-deep-cfr-experiments` | 11 passed |

The evaluator tests cover independent five-card ranking, exact flop equity,
the 169-class published pre-flop table, every rule-agent threshold path, the
LooseAggressive correction, duplicate-match invariants, both supported network
state-dict layouts, Bayesian range replay, LBR information-set invariance, and
controlled target policies.

## Loader differential checks

- A real UCV-ESCHER final policy checkpoint was queried with both its native
  loader and the neutral loader at 214 reachable decision states. Maximum
  absolute action-probability error: `0.0`.
- A VR-Deep policy snapshot made with the native network and snapshot schema
  was compared at a reachable FHP state. Maximum error:
  `9.934107481068821e-09`.
- A Deep CFR policy snapshot made with the native network and snapshot schema
  was compared at a reachable FHP state. Maximum error: `0.0`.

The VR difference is below single-precision rounding tolerance and comes from
the neutral loader normalising only the indexed legal logits, whereas the
native path masks the full vector before softmax.

## End-to-end acceptance

The command-line suite loaded the real UCV-ESCHER checkpoint and completed:

- all five corrected rule-agent duplicate matches;
- a both-seat LBR duplicate match;
- strict JSON output with checkpoint SHA-256 provenance.

Small deal counts and 32 pre-flop rollouts were used only for this execution
smoke test; they are not analysis settings.

## Interpretation limits

The test suite establishes implementation consistency, legality, reproducible
coupling, and controlled-case behaviour. It does not establish that LBR is a
tight approximation to an exact best response. Every LBR result therefore
retains the label `lbr_lower_bound` and its complete rollout configuration.

The supplied thesis fragments were checked for balanced braces, unique
bibliography keys, and complete citation-key resolution. A TeX engine was not
available in the workspace, so full-document typesetting remains a downstream
check.
