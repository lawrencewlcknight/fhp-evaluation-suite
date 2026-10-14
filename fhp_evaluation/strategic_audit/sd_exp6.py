"""Frozen SD-CFR Experiment 6 strategic audit: profile, pilot, development.

Configuration and archive-transfer helpers are stdlib-only so the Batch
launcher can prepare jobs without importing the inference runtime.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

EXPERIMENT = "sd_exp6_strategic_audit_v1"
SOURCE_RUN = "sdcfr6-48h-20261002-161544"
SOURCE_BUCKET = "gs://clever-overview-399515-fhp-deep-cfr-results"
LIBRARY_SEED = 20261006  # Keep the established development/assessment split.
RULES = ("candid_statistician", "loose_aggressive", "loose_passive",
         "tight_passive", "tight_aggressive")
STAGES = ("profile", "pilot", "development")
BOARD_COUNTS = dict(profile=1, pilot=8, development=64)
JOB_HOURS = dict(profile=6, pilot=24, development=72)  # Safety caps, not forecasts.
THREADS = 4
MAX_RSS_MB = 52000
# Completed internal H2H evaluation_manifest.json, sources for Exp6 at 24/48h.
# These are archive-manifest hashes, not final-network or replay checkpoints.
HASHES = {
    0: ((17690, "823affdbfa055f3d582bee58a75e0ee7d7af4c088c1585532d0f0adc736d57fb"),
        (35535, "f1d567fdb134ed5b1ad14398ffa62745e46176070b55131ddca04def07815fe3")),
    1: ((17739, "2e239e2ed160546bc4afb92dab1c92516c2eab436729ace78fb6f29212ab8316"),
        (35822, "1ee27c51dbf259afa359b4d155c7bcafba9de101efa2ce9ed0b58df372808ffa")),
    2: ((18547, "4bd3d9a8b0cdab88bb2dc222f7c696a02dff7de8824124185234081eda53452a"),
        (37210, "accab98c7c6cfe9eaa4efb3aba311514e56f4692c5fb4a5436fb50ef99edfa7d")),
}


def checkpoint_registry(stage):
    if stage not in STAGES:
        raise ValueError("Only profile/pilot/development are authorised; assessment remains locked")
    records = {}
    for seed in ((0, 1, 2) if stage == "development" else (0,)):
        for hour, (count, digest) in zip((24, 48), HASHES[seed]):
            if stage == "profile" and hour != 48:
                continue
            records[f"sd_exp6_seed{seed}_{hour}h"] = dict(
                run=SOURCE_RUN,
                relative=f"workers/task_{seed:03d}_parallel_structured_uniform_sd_cfr_48h_seed_{seed}"
                         f"/archive/time_{hour:02d}h.json",
                local=f"seed_{seed}/time_{hour:02d}h.json", sha256=digest,
                completed_iterations=count, training_seed=seed, hours=hour, cohort="sd_exp6")
    return records


def execution_config(stage):
    checkpoint_registry(stage)
    return dict(workers=1 if stage == "profile" else 4, task_seconds=10800,
                session_seconds=(JOB_HOURS[stage] - 2) * 3600, cpu_limit=16)


def make_spec(stage, input_root, native_root):
    from . import PROTOCOL
    policies = {name: dict(family="rule", name=name) for name in RULES}
    for label, item in checkpoint_registry(stage).items():
        policies[label] = dict(
            family="sd_cfr", checkpoint=str(Path(input_root).resolve() / item["local"]),
            sha256=item["sha256"], repo_root=str(Path(native_root).resolve()),
            cohort=item["cohort"], training_seed=item["training_seed"], hours=item["hours"],
            require_suit_invariance=True, batch_size=1024, model_batch_size=32,
            sd_float64_fallback=True, sd_query_order="hand_major")
    return dict(protocol=PROTOCOL, policies=policies, candidates=list(checkpoint_registry(stage)),
                opponents=list(RULES), range_tilts=[0, -2, 2],
                note="SD-CFR Exp6 only; full uniform archives with exact own-reach weighting. "
                     "Five fixed rule opponents; no cross-algorithm or reference-visitation panel.")


def make_library(stage):
    from .library import build_library, validate_library
    checkpoint_registry(stage)
    library = build_library(LIBRARY_SEED, pilot=stage != "development", reference_deals=0)
    if stage == "profile":
        library["boards"] = library["boards"][:1]  # Prespecified, not selected by results.
        library["partition_counts"] = {"development": 1}
    validate_library(library)
    return library


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_chunks(manifest, record):
    """Reject truncation/path traversal before downloading any chunk objects."""
    if (manifest.get("format") != "sd_cfr_chunked_uniform_v1" or
            manifest.get("weighting") != "uniform" or
            manifest.get("completed_iterations") != record["completed_iterations"] or
            manifest.get("num_players") != 2 or manifest.get("num_actions") != 3 or
            manifest.get("embedding_size") != 183):
        raise ValueError("Wrong or incomplete full-history archive contract")
    expected, seen = 1, set()
    chunks = manifest.get("chunks", [])
    for chunk in chunks:
        name = chunk["path"]
        first, last = chunk["first_iteration"], chunk["last_iteration"]
        if (not re.fullmatch(r"chunk_[0-9]{8}_[0-9]{8}\.npy", name) or name in seen or
                type(first) is not int or type(last) is not int or first != expected or last < first or
                name != f"chunk_{first:08d}_{last:08d}.npy" or
                not re.fullmatch(r"[0-9a-f]{64}", chunk["sha256"]) or
                type(chunk["size_bytes"]) is not int or not 0 < chunk["size_bytes"] <= 512 * 1024**2):
            raise ValueError("Unsafe chunk or missing/overlapping historical strategies")
        seen.add(name)
        expected = last + 1
    if not chunks or expected != record["completed_iterations"] + 1:
        raise ValueError("Incomplete historical strategy prefix")
    return chunks


def download_archives(stage, input_root, *, timeout_seconds=3600):
    """Download only pinned manifests and their complete chunks, deduplicated per seed.

    Fixed source bucket and exact object names: never recursive training downloads.
    A partial transfer is not promoted, and existing files are verified before reuse.
    """
    root = Path(input_root).resolve()
    deadline = time.monotonic() + timeout_seconds
    registry, contracts, seen = checkpoint_registry(stage), {}, {}

    def fetch(uri, target, digest, size=None):
        if target.exists():
            if sha256(target) != digest or (size is not None and target.stat().st_size != size):
                raise ValueError(f"Existing archive input failed integrity: {target}")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Archive download exceeded its bounded transfer budget")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        subprocess.run(["gcloud", "storage", "cp", uri, str(temporary)], check=True,
                       timeout=min(600, remaining))
        if sha256(temporary) != digest or (size is not None and temporary.stat().st_size != size):
            raise ValueError(f"Downloaded archive input failed integrity: {uri}")
        temporary.replace(target)

    # Pin every manifest before following any embedded chunk paths.
    for label, record in registry.items():
        path = root / record["local"]
        fetch(f"{SOURCE_BUCKET}/{record['run']}/{record['relative']}", path, record["sha256"])
        chunks = archive_chunks(json.loads(path.read_text()), record)
        contracts[label] = chunks
        for chunk in chunks:
            target = path.parent / chunk["path"]
            if target in seen and seen[target] != chunk:
                raise ValueError("24h/48h archives disagree about a shared historical chunk")
            seen[target] = chunk
    for seed in {r["training_seed"] for r in registry.values()}:
        earlier = contracts.get(f"sd_exp6_seed{seed}_24h")
        later = contracts[f"sd_exp6_seed{seed}_48h"]
        if earlier is not None and earlier != later[:len(earlier)]:
            raise ValueError("24h is not the exact historical prefix of the 48h archive")
    if sum(c["size_bytes"] for c in seen.values()) > 100 * 1024**3:
        raise ValueError("Archive download exceeds the 100 GiB disk safety budget")
    transferred = set()
    for label, record in registry.items():
        for chunk in contracts[label]:
            target = (root / record["local"]).parent / chunk["path"]
            if target in transferred:
                continue
            uri = f"{SOURCE_BUCKET}/{record['run']}/{Path(record['relative']).parent}/{chunk['path']}"
            fetch(uri, target, chunk["sha256"], chunk["size_bytes"])
            transferred.add(target)
    return dict(checkpoints=registry, unique_chunks=len(seen),
                archive_bytes=sum(c["size_bytes"] for c in seen.values()),
                full_archive=True, weighting="uniform", source_bucket=SOURCE_BUCKET)


def temporal_summary(result, stage):
    """Equal opponent weights within seed; paired 48h-24h differences across seeds."""
    from .exp16 import seed_summary
    if stage not in ("pilot", "development"):
        raise ValueError("The one-board profile is a feasibility measurement, not a results panel")
    if result.get("status") != "succeeded":
        raise ValueError("An incomplete numerical audit cannot be summarised")
    expected = make_spec(stage, "/unused", "/unused")
    spec = result["provenance"]["spec"]
    for key in ("candidates", "opponents", "range_tilts"):
        if spec[key] != expected[key]:
            raise ValueError(f"Unexpected {key} panel")
    for label in expected["candidates"]:
        for key in ("family", "sha256", "cohort", "training_seed", "hours"):
            if spec["policies"][label][key] != expected["policies"][label][key]:
                raise ValueError("Unexpected checkpoint identity")
    seeds = (0,) if stage == "pilot" else (0, 1, 2)
    endpoints, paired = {}, {}
    for tilt in expected["range_tilts"]:
        values = {24: {}, 48: {}}
        for opponent in RULES:
            common_support = None
            for hour in (24, 48):
                values[hour][opponent] = {}
                for seed in seeds:
                    row = result["comparisons"][f"sd_exp6_seed{seed}_{hour}h__vs__{opponent}__tilt_{tilt:g}"]
                    support = sorted((tuple(b["board"]), b["root_count"]) for b in row["boards"] if b["core"])
                    if (len(support) != BOARD_COUNTS[stage] or
                            len({board for board, _ in support}) != len(support) or
                            any(type(n) is not int or n < 0 for _, n in support) or
                            not sum(n for _, n in support) or
                            (common_support is not None and support != common_support)):
                        raise ValueError("Incomplete or unmatched checkpoint root support")
                    common_support = support
                    value = row["mean_root_response_gap_bb"]
                    if not math.isfinite(value):
                        raise ValueError("Non-finite root gap")
                    values[hour][opponent][str(seed)] = value
        for hour in (24, 48):
            values[hour]["equal_weight_panel"] = {
                str(s): statistics.mean(values[hour][o][str(s)] for o in RULES) for s in seeds}
            endpoints[f"{hour}h_tilt_{tilt:g}"] = {o: seed_summary(v) for o, v in values[hour].items()}
        paired[f"48_minus_24h_tilt_{tilt:g}"] = {
            o: seed_summary({str(s): values[48][o][str(s)] - values[24][o][str(s)] for s in seeds})
            for o in values[48]}
    return dict(experiment=EXPERIMENT, stage=stage,
                metric="remaining-round root response gap (BB per supported audited root)",
                primary="48_minus_24h_tilt_0", lower_is_better=True,
                endpoints=endpoints, paired_changes=paired, caveats=[
                    "Conditional on fixed stratified boards, supported public histories and five rule opponents.",
                    "Not whole-game exploitability: missed response opportunities against a rule opponent are not necessarily equilibrium errors.",
                    "Training seeds, not boards, hands or opponents, are the policy-level replication units.",
                    "The one-seed pilot is descriptive only; three-seed t intervals are descriptive, pointwise and low-powered.",
                    "Zero tilt is primary; -2 and +2 are synthetic opponent-range sensitivity checks.",
                    "24h and 48h are paired prefixes of the same training trajectories, not independent models.",
                    "Development evidence may motivate a new training specification; it does not validate a retrained algorithm.",
                    "No assessment boards, natural-visitation sample, cross-algorithm comparison or new training."])


def write_report(output, result, stage):
    from ..best_response.runner import write_json
    from .report import svg_curve
    output = Path(output)
    summary = temporal_summary(result, stage)
    write_json(output / "temporal_summary.json", summary)
    parts = ["<!doctype html><meta charset='utf-8'><title>SD-CFR Experiment 6 audit</title>",
             "<h1>SD-CFR Experiment 6: 24h to 48h strategic audit</h1>",
             "<p>Negative changes mean smaller remaining-round response gaps, not certified exploitability reduction.</p>",
             '<p><a href="report.html">Detailed positions, local errors, textures and implementation checks</a></p>',
             "<h2>Primary: 48h minus 24h, untilted opponent ranges</h2><table>",
             "<tr><th>Opponent</th><th>Mean change (BB/root)</th><th>Seed changes</th><th>Descriptive 95% interval</th></tr>"]
    for opponent, row in summary["paired_changes"][summary["primary"]].items():
        parts.append(f"<tr><td>{html.escape(opponent)}</td><td>{row['mean']:.6f}</td>"
                     f"<td>{html.escape(str(row['seed_values']))}</td>"
                     f"<td>{html.escape(str(row['descriptive_95pct_t_interval']))}</td></tr>")
    points = [(h, summary["endpoints"][f"{h}h_tilt_0"]["equal_weight_panel"]["mean"]) for h in (24, 48)]
    (output / "panel_root_gap.svg").write_text(svg_curve(points, "SD-CFR fixed-panel root response gap",
                                                        "Active training hours", "BB per audited root"))
    parts += ['</table><img src="panel_root_gap.svg" alt="SD-CFR root gap through time"><ul>']
    parts += [f"<li>{html.escape(note)}</li>" for note in summary["caveats"]]
    (output / "sd_exp6_report.html").write_text("\n".join(parts + ["</ul>"]))
    return summary


def write_runtime_report(output, stage, library):
    """Verify every full-archive export before publishing a stage-level success."""
    from ..best_response.runner import write_json
    from .library import board_schedule, board_key
    from .workflow import completed, digest
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    fingerprint = digest(manifest)
    rows = []
    for label, record in checkpoint_registry(stage).items():
        for board in board_schedule(library, "development"):
            path = output / "tables" / label / board_key(board)
            if not completed(path, fingerprint, "tables.npz"):
                raise ValueError("Missing verified full-archive board export")
            diagnostic = json.loads((path / "diagnostics.json").read_text())
            adapter = diagnostic["adapter"]
            if (adapter.get("models_per_player") != record["completed_iterations"] or
                    adapter.get("weighting") != "uniform" or
                    adapter.get("archive_integrity_verified") is not True or
                    adapter.get("checkpoint_sha256") != record["sha256"]):
                raise ValueError("Export was not the pinned complete uniform archive")
            rows.append(dict(policy=label, board=board, models_per_player=adapter["models_per_player"],
                             query_seconds=diagnostic["stats"]["query_seconds"],
                             export_seconds=diagnostic["elapsed_seconds"]))
    attempts = {str(p.relative_to(output)): json.loads(p.read_text())
                for p in sorted((output / "workers").rglob("run.json"))}
    report = dict(experiment=EXPERIMENT, stage=stage, full_archive_exports=rows,
                  execution=execution_config(stage), job_cap_hours=JOB_HOURS[stage],
                  attempts={k: {n: v.get(n) for n in ("status", "elapsed_seconds", "peak_rss_bytes")}
                            for k, v in attempts.items()},
                  summed_worker_seconds=sum(r.get("elapsed_seconds", 0) for r in attempts.values()),
                  peak_worker_rss_bytes=max((r.get("peak_rss_bytes", 0) for r in attempts.values()), default=0),
                  next_stage_workload=dict(pilot_full_archive_board_exports=16,
                                           development_full_archive_board_exports=384),
                  note="Profile: one 48h archive on one board, serial. Pilot: four concurrent board workers. "
                       "Review actual elapsed time and memory before approving the next stage; no assumed parallel speedup. "
                       "Counts exclude rule exports, analysis, archive checks, setup and transfers. Caps are not forecasts.")
    write_json(output / "analysis/runtime_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-sd-cache", type=Path,
                        help="Explicit continuation of the pinned stopped development audit")
    parser.add_argument("--download", action="store_true", help="Fetch only complete pinned archives from the fixed source bucket")
    args = parser.parse_args()
    if args.import_sd_cache and args.stage != "development":
        parser.error("The cache continuation is development-only")
    from ..best_response.runner import write_json
    from .validation import validate
    from .workflow import run
    args.output.mkdir(parents=True, exist_ok=True)
    if args.download:
        write_json(args.output / "archive_sources.json", download_archives(args.stage, args.input_root))
    spec, library = make_spec(args.stage, args.input_root, args.native_root), make_library(args.stage)
    for name, value in (("spec.json", spec), ("library.json", library)):
        path = args.output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing changed frozen input: {path}")
        write_json(path, value)
    write_json(args.output / "oracle_validation.json", validate())
    execution = execution_config(args.stage)
    summary_path = run(spec, library, args.output, include_reference=False,
                       resume=(args.output / "manifest.json").exists(), threads=THREADS,
                       max_seconds=execution["session_seconds"], max_rss_mb=MAX_RSS_MB,
                       board_exports=execution, import_sd_cache=args.import_sd_cache)
    result = json.loads(summary_path.read_text())
    if result.get("status") != "succeeded" or result.get("exact_full_game_exploitability") is not False:
        raise ValueError("Missing or incompatible completed numerical audit")
    if args.stage != "profile":
        write_report(args.output / "analysis", result, args.stage)
    write_runtime_report(args.output, args.stage, library)
    reports = [summary_path, args.output / "analysis/runtime_summary.json"]
    if args.stage != "profile":
        reports.append(args.output / "analysis/temporal_summary.json")
    write_json(args.output / "analysis/SUCCESS.json", dict(
        experiment=EXPERIMENT, status="succeeded", stage=args.stage, assessment_evaluated=False,
        full_archive=True, checkpoint_sha256={k: v["sha256"] for k, v in checkpoint_registry(args.stage).items()},
        report_sha256={p.name: sha256(p) for p in reports}))


if __name__ == "__main__":
    main()
