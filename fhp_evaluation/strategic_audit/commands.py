"""Explicit generation, validation and execution; never submits paid jobs."""

import json
from pathlib import Path

from ..best_response.runner import write_json
from ..loaders import sha256_file
from ..rule_agents import PUBLISHED_AGENT_NAMES
from . import PROTOCOL


def register(subparsers):
    library = subparsers.add_parser("strategic-library", help="freeze a 96-board library or eight-board pilot")
    library.add_argument("--output", required=True)
    library.add_argument("--seed", type=int, default=20261006)
    library.add_argument("--pilot", action="store_true")
    library.add_argument("--reference-deals", type=int, default=64)
    config = subparsers.add_parser("strategic-spec", help="create a control smoke spec or pinned Exp9/10 comparison")
    config.add_argument("--output", required=True)
    config.add_argument("--smoke", action="store_true")
    config.add_argument("--ucv-repo")
    config.add_argument("--exp9-root")
    config.add_argument("--exp10-root")
    validate = subparsers.add_parser("strategic-validate", help="independent reduced-deck decision-value validation")
    validate.add_argument("--output", required=True)
    run = subparsers.add_parser("strategic-run", help="bounded resumable strategic audit of frozen policies")
    run.add_argument("--spec", required=True)
    run.add_argument("--library", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--partition", choices=("development", "assessment"), default="development")
    run.add_argument("--unlock-assessment", action="store_true")
    run.add_argument("--include-reference", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--max-seconds", type=int, default=7200, help="per native-policy export / analysis attempt")
    run.add_argument("--max-rss-mb", type=int, default=8192)


def create_spec(*, smoke=False, ucv_repo=None, exp9_root=None, exp10_root=None):
    if smoke:
        if any((ucv_repo, exp9_root, exp10_root)):
            raise ValueError("Smoke mode must not silently ignore real checkpoints")
        policies = {"uniform": dict(family="control", name="uniform", cohort="uniform_control",
                                     training_seed=0, hours=0),
                    "caller": dict(family="control", name="call", cohort="caller_control",
                                    training_seed=0, hours=0)}
        return dict(protocol=PROTOCOL, policies=policies, candidates=["uniform"], opponents=["caller"],
                    reference_policy="uniform", range_tilts=[0])
    if not ucv_repo or not exp9_root or not exp10_root:
        raise ValueError("Supply native UCV repo and staged Exp9/Exp10 run roots, or use --smoke")
    policies = {name: dict(family="rule", name=name) for name in PUBLISHED_AGENT_NAMES}
    candidates, anchors = [], []
    for experiment, root in ((9, Path(exp9_root)), (10, Path(exp10_root))):
        for seed in range(3):
            paths = sorted(root.glob(f"workers/*seed_{seed}/checkpoints/*seed_{seed}_time_24h.pkl"))
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one 24h seed-{seed} checkpoint under {root}, found {len(paths)}")
            label = f"exp{experiment}_seed{seed}_24h"
            policies[label] = dict(family="ucv", checkpoint=str(paths[0].resolve()), sha256=sha256_file(paths[0]),
                                   repo_root=str(Path(ucv_repo).resolve()), cohort=f"exp{experiment}",
                                   training_seed=seed, hours=24, require_suit_invariance=True)
            candidates.append(label)
            if experiment == 9:
                anchors.append(label)
    return dict(protocol=PROTOCOL, policies=policies, candidates=candidates,
                opponents=list(PUBLISHED_AGENT_NAMES) + anchors,
                reference_policy=anchors[0], range_tilts=[0, -2, 2],
                note="Exp9 seed0 generates the shared reference-position distribution; self-anchor results are labelled, not held-out evidence")


def dispatch(args):
    if args.command != "strategic-run" and Path(args.output).exists():
        raise FileExistsError("Refusing to replace an existing frozen artifact")
    if args.command == "strategic-library":
        from .library import build_library
        write_json(args.output, build_library(args.seed, pilot=args.pilot, reference_deals=args.reference_deals))
    elif args.command == "strategic-spec":
        from .workflow import validate_spec
        result = create_spec(smoke=args.smoke, ucv_repo=args.ucv_repo, exp9_root=args.exp9_root, exp10_root=args.exp10_root)
        validate_spec(result)
        write_json(args.output, result)
    elif args.command == "strategic-validate":
        from .validation import validate
        write_json(args.output, validate())
    else:
        from .workflow import run
        result = run(json.loads(Path(args.spec).read_text()), json.loads(Path(args.library).read_text()),
                     args.output_dir, partition=args.partition, include_reference=args.include_reference,
                     unlock_assessment=args.unlock_assessment, resume=args.resume, threads=args.threads,
                     max_seconds=args.max_seconds, max_rss_mb=args.max_rss_mb)
        print(result)
