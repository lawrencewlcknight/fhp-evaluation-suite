# SD6 numerical repair and query reuse (15 October 2026)

This is an evaluation-only change. It does not retrain policies, subsample the
historical strategy archive, change own-reach weights, or weaken tolerances.
No cloud job was launched to validate this change.

## Failure and repair

The development job `fhp-sd6-audit-dev-20261013-104108-audit` failed on the
seed-0 48-hour archive, board `[23, 24, 42]`, at the native/batched parity gate:
maximum absolute probability difference `1.881912142642772e-05`, tolerance
`3e-06`. Different neural matrix-multiplication batch shapes can introduce FP32
rounding differences; regret matching can amplify small differences near zero.
This is the suspected mechanism, not an exact reproduction of the cloud CPU.

The SD6 specification now enables an explicit FP64 numerical retry. Only a
scalar/batched probability mismatch triggers it. Both the scalar reference and
batched exporter promote the same archived FP32 weights to FP64, retaining
the FP32-rounded input encoding. The full checks run again at the original
tolerance, and the entire affected board is exported with the repaired adapter.
Privacy, card-order, required suit-invariance and non-finite failures are not
silently accepted. A failed FP64 check still rejects the board.

An additional gate compares actual exported rows to the scalar reference at
the existing probe histories. This catches a discrepancy caused by the larger
export batch shape even if the preliminary probe passed. Such a discrepancy
also triggers a complete FP64 rebuild before a success marker can be written.
The check covers both seats, one hand and the first preflop entry's decision
histories; it is not an exhaustive parity proof. Retry details are recorded in
each board's `diagnostics.json`.

## Local full-archive numerical evidence

The pinned archive contains 35,535 models per player in 280 chunks. All chunks
were verified against manifest sizes and SHA256 hashes. Manifest checksum:
`f1d567fdb134ed5b1ad14398ffa62745e46176070b55131ddca04def07815fe3`.

Using the failed board's original eight probe histories on the local Apple
CPU, at one computation thread:

| Inference | Largest scalar/batched probability difference |
| --- | ---: |
| FP32 | 2.22159736357197e-7 |
| FP64 | 1.1102230246251565e-16 |

Thus the cloud-specific failure did not recur locally in FP32. The FP64 result
supports the precision repair but is not a guarantee about every untested
board or the cloud machine. Each production board retains its own gates.
The scalar FP32-to-FP64 change on these eight probes was at most
`9.410520307451264e-8`; promotion does change numerical arithmetic, not the
underlying strategy-mixture definition.
The subsequent full-archive FP64 check completed in 168.6 seconds, with zero
hidden-card, card-order and suit-renaming probability differences on its probe
set. Suit invariance was recorded diagnostically in that run; its measured
maximum was zero (also satisfying the production requirement). The reproduction
command now explicitly requires suit invariance as well.

The local reproduction commands (no Batch submission) are:

```bash
python -m fhp_evaluation.strategic_audit.sd_numerics download \
  --root outputs/diagnostics/sd6-numerics/archive

python -m fhp_evaluation.strategic_audit.sd_numerics probe \
  --root outputs/diagnostics/sd6-numerics/archive \
  --native-repo "$FHP_SD_REPO" \
  --output outputs/diagnostics/sd6-numerics/parity-float64-all-checks.json \
  --precision float64 --all-checks
```

The download is approximately 4 GB and reads the existing archive from GCS.
Use the suite's pinned OpenSpiel 1.6.3 environment and the updated native repo.

## Speed improvement: hand-major query grouping

Previously, table export queried one betting history across all hands before
moving to the next history. That repeated inference for shared earlier
decisions. Grouping a hand's histories in a bounded query batch lets the
existing SD input-row deduplication reuse those decisions.

The new `hand_major` order retains every table, hand, model and weighting term.
Other callers retain the original `schema` default; SD6 explicitly opts in.
There is no unbounded cross-board cache or approximate policy compression.

A local complete-board code-path test used 1,176 hands, 64 histories and only
the first 128 historical models, at one computation thread. It ran layouts in
the order schema / hand-major / hand-major / schema:

| Quantity | Schema | Hand-major |
| --- | ---: | ---: |
| Network-input rows | 134,116 | 70,368 |
| Model-batch calls | 512 | 296 |
| Mean elapsed seconds (two runs) | 52.51 | 25.58 |

This is 47.5% fewer input rows and approximately 2.05x local speedup. Maximum
table probability difference was `9.66838689109295e-8`, below the unchanged
parity tolerance. Background activity and CPU warm-up can affect these timings;
the operation-count reduction is stronger evidence than the elapsed-time ratio.
This is **not** a full-archive or GCP runtime forecast. FP64 retries, setup,
analysis and uploads also contribute to total runtime.

## Tests and recovery boundary

Regression tests cover unchanged default precision, FP64 scalar/batched parity
for raw and structured encodings, fresh precision-specific caches, full-table
layout equivalence, unchanged safety gates, and failure-before-success-marking
when either preliminary or exported-row parity remains invalid. The native
full-archive fixture integration also passed; that fixture is small and is not
a production speed benchmark.

Verification totals: 84 evaluation regression tests passed (one skipped, two
deselected), five targeted native inference tests passed, and the separately
run native integration test passed.

The original 115 verified exports and all remote artifacts remain untouched.
The old recovery request freezes source identities; the old `recover` command
does not deploy this repair. Use the explicit continuation below. It validates
the pinned original manifest, the scientific specification, archive hashes,
runtime versions, table schemas and both table/diagnostic checksums. It copies
verified boards into a new run identity, preserving the old manifest and every
original completion marker in the migration record. No old analysis is reused.
Both source namespaces remain distinct. A later recovery of the continuation
must use its own unchanged prepared request.

## Restart with the repaired code (user-submitted)

From the SD-CFR repository root, with both repositories updated:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export RUN_ID="fhp-sd6-audit-fix-$(date -u +%Y%m%d-%H%M%S)"

bash gcp/continue_sd_exp6_strategic_audit.sh prepare &&
bash gcp/continue_sd_exp6_strategic_audit.sh submit
```

`prepare` is offline. `submit` is the explicit paid job action. The job uses
`n2-standard-16`, four workers with four threads each, and the existing 72-hour
safety cap. It imports 115 completed SD board exports, calculates the remaining
269, exports the rule opponents and runs the original analysis. It does not run
the proposed scheduling benchmark. Watch `$BUCKET/$RUN_ID/cache_import.json`
for the migration ledger and `export_progress.json` for live progress.

Do not delete the original failed job or its GCS files before continuation
preflight and migration succeed. If this new invocation is interrupted, retain
its `RUN_ID` and prepared files, inspect the reason, then explicitly use:

```bash
bash gcp/continue_sd_exp6_strategic_audit.sh recover --recovery-tag retry1
```
