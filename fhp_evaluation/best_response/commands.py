"""Safe entry points for the three best-response evaluation experiments."""

import argparse
import json
from pathlib import Path

from .runner import run_bounded, write_json


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def register(subparsers):
    for name, help_text in [
        ("validate", "exp1: independent small-game and exact flop validation"),
        ("profile", "exp2: bounded real-checkpoint CPU feasibility measurements"),
        ("compare", "exp3: paired LBR versus full-flop lower-bound evaluation"),
    ]:
        parser = subparsers.add_parser("br-" + name, help=help_text)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--max-seconds", type=positive, default=300)
        parser.add_argument("--max-rss-mb", type=positive, default=4096)
        parser.add_argument("--resume", action="store_true", help="reuse identical successful process shards only")
        if name == "validate":
            continue
        parser.add_argument("policy", nargs="?", help="trusted checkpoint, or SD-CFR manifest with all chunks")
        parser.add_argument("--family", choices=["ucv", "vr_deep", "sd_cfr", "raw_mlp", "control"], required=True)
        parser.add_argument("--repo-root", help="native training checkout; required except raw_mlp/control")
        parser.add_argument("--control", choices=["random", "uniform", "fold", "call", "raise"], default="random")
        parser.add_argument("--batch-size", type=positive, default=1024)
        parser.add_argument("--model-batch-size", type=positive, default=32)
        parser.add_argument("--seed", type=int, default=20261004)
        if name == "profile":
            parser.add_argument("--threads", type=positive, nargs="+", default=[1, 4, 8])
            parser.add_argument("--boards", type=positive, default=2)
            parser.add_argument("--repeats", type=positive, default=2)
        else:
            parser.add_argument("--threads", type=positive, default=1)
            parser.add_argument("--deals", type=positive, default=100)
            parser.add_argument("--rollouts", type=positive, default=4096)
            parser.add_argument("--search-seed", type=int, default=731)


def dispatch(args):
    mode = args.command.removeprefix("br-")
    spec = dict(mode=mode, threads=1)
    if mode != "validate":
        if args.family != "control" and not args.policy:
            raise ValueError("A checkpoint is required unless --family control")
        if args.family == "control" and args.policy:
            raise ValueError("Control profiling must not silently ignore a checkpoint")
        if args.family in ("ucv", "vr_deep", "sd_cfr") and not args.repo_root:
            raise ValueError("Native checkpoint family requires --repo-root")
        spec.update(family=args.family, checkpoint=str(Path(args.policy).resolve()) if args.policy else None,
                    repo_root=str(Path(args.repo_root).resolve()) if args.repo_root else None,
                    batch_size=args.batch_size, model_batch_size=args.model_batch_size,
                    seed=args.seed, control=args.control)
    directory = Path(args.output_dir)
    kwargs = dict(max_seconds=args.max_seconds, max_rss_mb=args.max_rss_mb, resume=args.resume)
    if mode == "profile":
        if args.boards >= 22100:
            raise ValueError("Reserve one distinct warmup board; --boards must be < 22100")
        if len(set(args.threads)) != len(args.threads):
            raise ValueError("Thread counts must be distinct")
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for threads in args.threads:
            record = run_bounded(dict(spec, threads=threads, boards=args.boards, repeats=args.repeats),
                                 directory / f"threads_{threads}", **kwargs)
            records.append(record)
            print(json.dumps(dict(threads=threads, status=record["status"],
                                  elapsed_seconds=record["elapsed_seconds"])))
        result = dict(experiment="exp2_br_feasibility", runs=records)
        write_json(directory / "summary.json", result)
    else:
        if mode == "compare":
            if args.deals < 2:
                raise ValueError("At least two independent deal pairs are needed for a confidence interval")
            spec.update(threads=args.threads, deals=args.deals, rollouts=args.rollouts, search_seed=args.search_seed)
        records = [run_bounded(spec, directory, **kwargs)]
        print(json.dumps(dict(status=records[0]["status"], output=str(directory.resolve()))))
    if any(row["status"] != "succeeded" for row in records):
        raise SystemExit(1)
