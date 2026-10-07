"""Evaluation experiment 6: frozen Exp16 temporal strategic-position audit.

The checkpoint registry is transcribed from the completed Exp16 evaluation
manifest. Its 24h policies are imported Exp10 policies, not new training seeds.
This module's configuration helpers deliberately use only the standard library.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

EXP16_RUN = "exp16-feat48-20261004-182051"
EXP9_RUN = "exp9-cache24-20261001-132550"
LIBRARY_SEED = 20261006
HOURS = (24, 36, 42, 48)
RULES = ("candid_statistician", "loose_aggressive", "loose_passive",
         "tight_passive", "tight_aggressive")
HASHES = {
    0: (
        "2dabc42248cb33bfd351ad38eb9d97633acf2e030bf501b051e4d6469d3b01d9",
        "275337358eea0009fde8ea473eacb0ceddbb9157d6237c73324bc5121c0d0ce9",
        "7d3ca232b276c4be9125c3f626dafa230138153c4e5a54c6189385ebcbefbdb0",
        "5ca65a3998b3e7d7b18cbe975a804328ebcb4cfaede467c97e7a5ceb25cfb8e1"),
    1: (
        "d7960186bedc4d2456a09dd82f7cf8b863a0a67486b896f77552b30fb63abfa0",
        "a74991941da33d99b4aaea840904e6757c42546d73f843da5bbfd08408cb103a",
        "45d5214271066c87e71fb3426b56abf8115790b8d9978ddb51abfe0f5a91d621",
        "c863a95ed21eb8315b688404dab92e46ca0800f2abe7ee62d94a0fd8958a7ab3"),
    2: (
        "99950075ebf28a0811c49775280bebc7ff3ab1a1b2c12e472006bf262c6b02fe",
        "35d668d104253d7bb825e5cc7c8abe8f210e64107922dc341a71f540ee26ed17",
        "3bafaaf4f69ce5c760c1e90aed5c0d1a7d7c26f1d12c5b293b9f13542114370e",
        "bf553194dc176bd01ec0fca3f04caf2fa70a238c6aaad404c4175892dc76bf14"),
}
ANCHOR_HASHES = (
    "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449",
    "ef5f0db716bd2652cd06817ff0b4d4b6e5d1307c5063b982c69b768d1b3b7caa",
    "d678f805ad9c9665545ba91164a9129c5bb60fab09c90b8478cbcaa35b9def7b")


def checkpoint_registry(stage):
    if stage not in ("pilot", "development"):
        raise ValueError("Only pilot/development are authorised; assessment remains locked")
    records = {}
    for seed in ((0,) if stage == "pilot" else range(3)):
        for hour, digest in zip(HOURS, HASHES[seed]):
            if stage == "pilot" and hour not in (24, 48):
                continue
            name = f"hand_board_cached_parallel_ucv_escher_seed_{seed}"
            records[f"exp16_seed{seed}_{hour}h"] = dict(
                run=EXP16_RUN, relative=f"workers/task_{seed:03d}_{name}/checkpoints/{name}_time_{hour:02d}h.pkl",
                sha256=digest, training_seed=seed, hours=hour, cohort="exp16")
    for seed, digest in enumerate(ANCHOR_HASHES):
        name = f"cached_parallel_structured_ucv_escher_seed_{seed}"
        records[f"exp9_seed{seed}_24h"] = dict(
            run=EXP9_RUN, relative=f"workers/task_{seed:03d}_{name}/checkpoints/{name}_time_24h.pkl",
            sha256=digest, training_seed=seed, hours=24, cohort="exp9")
    return records


def make_spec(stage, input_root, native_root):
    from . import PROTOCOL
    policies = {name: dict(family="rule", name=name) for name in RULES}
    for label, record in checkpoint_registry(stage).items():
        policies[label] = dict(family="ucv", checkpoint=str(Path(input_root).resolve() / f"{label}.pkl"),
                              sha256=record["sha256"], repo_root=str(Path(native_root).resolve()),
                              cohort=record["cohort"], training_seed=record["training_seed"],
                              hours=record["hours"], require_suit_invariance=True)
    return dict(protocol=PROTOCOL, policies=policies,
                candidates=[p for p in policies if p.startswith("exp16_")],
                opponents=list(RULES) + [f"exp9_seed{s}_24h" for s in range(3)],
                reference_policy="exp9_seed0_24h", range_tilts=[0, -2, 2],
                note="24h is the imported Exp10 policy. Eight fixed opponents, not eight independent training replicates.")


def seed_summary(values):
    """Uncertainty across training trajectories, never across positions/opponents."""
    rows = [float(v) for v in values.values()]
    if not rows or not all(math.isfinite(v) for v in rows):
        raise ValueError("Missing or non-finite seed values")
    se = statistics.stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else None
    # Only the prespecified one-seed pilot or three-seed development is permitted.
    if len(rows) not in (1, 3):
        raise ValueError("Incomplete seed cohort")
    mean = statistics.mean(rows)
    return dict(n=len(rows), seed_values=values, mean=mean, standard_error=se,
                descriptive_95pct_t_interval=[mean - 4.3026527299 * se, mean + 4.3026527299 * se]
                if se is not None else None)


def temporal_summary(result, stage):
    """Fixed support, root-gap primary estimand; equal weight to each opponent."""
    spec = result["provenance"]["spec"]
    expected = make_spec(stage, "/unused", "/unused")
    for name in ("candidates", "opponents", "range_tilts"):
        if spec[name] != expected[name]:
            raise ValueError(f"Unexpected {name} panel")
    seeds = (0,) if stage == "pilot" else (0, 1, 2)
    hours = (24, 48) if stage == "pilot" else HOURS
    endpoints, paired = {}, {}
    for tilt in spec["range_tilts"]:
        by_hour = {}
        for hour in hours:
            by_hour[hour] = {}
            for opponent in spec["opponents"]:
                values = {}
                for seed in seeds:
                    label = f"exp16_seed{seed}_{hour}h__vs__{opponent}__tilt_{tilt:g}"
                    row = result["comparisons"][label]
                    base = result["comparisons"][f"exp16_seed{seed}_24h__vs__{opponent}__tilt_{tilt:g}"]
                    def support(r):
                        return sorted((tuple(b["board"]), b["root_count"]) for b in r["boards"] if b["core"])
                    if support(row) != support(base) or not sum(n for _, n in support(row)):
                        raise ValueError("Checkpoint root support differs or is empty")
                    values[str(seed)] = row["mean_root_response_gap_bb"]
                by_hour[hour][opponent] = values
            by_hour[hour]["equal_weight_panel"] = {
                str(seed): statistics.mean(by_hour[hour][opp][str(seed)] for opp in spec["opponents"])
                for seed in seeds}
            endpoints[f"{hour}h_tilt_{tilt:g}"] = {
                opp: seed_summary(values) for opp, values in by_hour[hour].items()}
        for earlier in hours[:-1]:
            paired[f"48_minus_{earlier}h_tilt_{tilt:g}"] = {
                opp: seed_summary({str(seed): by_hour[48][opp][str(seed)] - by_hour[earlier][opp][str(seed)]
                                   for seed in seeds}) for opp in by_hour[48]}
    return dict(stage=stage, metric="remaining-round root response gap (BB per supported audited root)",
                primary="48_minus_24h_tilt_0", lower_is_better=True, endpoints=endpoints, paired_changes=paired,
                caveats=["Conditional on the fixed stratified boards, public histories and opponent panel; not exploitability.",
                         "Training seeds are inferential units. Three seeds provide limited inferential power.",
                         "Intervals are descriptive, pointwise and unadjusted for multiple comparisons.",
                         "Range tilts -2 and +2 are synthetic sensitivity checks; zero tilt is primary.",
                         "The 24h policies are Exp10 ancestors, not additional independent trajectories.",
                         "Audit improvements need not reproduce the ranking from head-to-head play."])


def write_temporal_report(output, result, stage):
    from ..best_response.runner import write_json
    from .report import svg_curve
    output = Path(output)
    summary = temporal_summary(result, stage)
    write_json(output / "temporal_summary.json", summary)
    parts = ["<!doctype html><meta charset='utf-8'><title>Experiment 16 strategic audit</title>",
             "<h1>Experiment 16: strategic quality through time</h1>",
             "<p>Negative 48h-minus-earlier differences indicate smaller remaining-round response gaps. "
             "This is not whole-game exploitability. All values are in big blinds per audited root.</p>",
             '<p><a href="report.html">Detailed positions, local gaps and implementation checks</a></p>',
             "<h2>Primary: 48h minus 24h, unmodified opponent ranges</h2><table>",
             "<tr><th>Opponent</th><th>Mean change</th><th>Seed changes</th><th>Descriptive 95% interval</th></tr>"]
    for opponent, row in summary["paired_changes"][summary["primary"]].items():
        parts.append(f"<tr><td>{html.escape(opponent)}</td><td>{row['mean']:.6f}</td>"
                     f"<td>{html.escape(str(row['seed_values']))}</td>"
                     f"<td>{html.escape(str(row['descriptive_95pct_t_interval']))}</td></tr>")
    parts.append("</table><h2>Equal-weight fixed-panel trajectory</h2>")
    hours = (24, 48) if stage == "pilot" else HOURS
    points = [(h, summary["endpoints"][f"{h}h_tilt_0"]["equal_weight_panel"]["mean"]) for h in hours]
    (output / "panel_root_gap.svg").write_text(svg_curve(points, "Fixed-panel root response gap", "Active training hours", "BB per audited root"))
    parts.append('<img src="panel_root_gap.svg" alt="Fixed-panel root response gap through time"><ul>')
    parts.extend(f"<li>{html.escape(note)}</li>" for note in summary["caveats"])
    parts.append("</ul>")
    (output / "exp16_report.html").write_text("\n".join(parts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("pilot", "development"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    from ..best_response.runner import write_json
    from .library import build_library
    from .workflow import run
    from .validation import validate
    args.output.mkdir(parents=True, exist_ok=True)
    spec = make_spec(args.stage, args.input_root, args.native_root)
    library = build_library(LIBRARY_SEED, pilot=args.stage == "pilot",
                            reference_deals=8 if args.stage == "pilot" else 64)
    for name, value in (("spec.json", spec), ("library.json", library)):
        path = args.output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing changed frozen input: {path}")
        write_json(path, value)
    write_json(args.output / "oracle_validation.json", validate())
    summary_path = run(spec, library, args.output, include_reference=True,
                       resume=(args.output / "manifest.json").exists(), threads=args.threads,
                       max_seconds=10800 if args.stage == "pilot" else 75600, max_rss_mb=52000)
    result = json.loads(summary_path.read_text())
    write_temporal_report(args.output / "analysis", result, args.stage)
    attempts = {str(p.relative_to(args.output)): json.loads(p.read_text())
                for p in sorted((args.output / "workers").glob("*/attempt_*/run.json"))}
    write_json(args.output / "analysis" / "runtime_summary.json", dict(
        attempts={k: {name: r.get(name) for name in ("status", "elapsed_seconds", "peak_rss_bytes")}
                  for k, r in attempts.items()},
        summed_worker_seconds=sum(r.get("elapsed_seconds", 0) for r in attempts.values()),
        peak_worker_rss_bytes=max((r.get("peak_rss_bytes", 0) for r in attempts.values()), default=0),
        note="Worker durations include all restored attempts; exclude VM setup and transfer. Use the pilot to budget development."))
    write_json(args.output / "analysis" / "SUCCESS.json", dict(
        status="succeeded", stage=args.stage, assessment_evaluated=False,
        checkpoint_sha256={k: v["sha256"] for k, v in checkpoint_registry(args.stage).items()},
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
