"""Experiment 27: learning controls, matched loss ablation, gated replication.

All gates route resource allocation only. No worker trains a UCV defender and
no result is accepted as an Exp21 qualification artifact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import torch

from ..best_response.adapters import load_target
from ..best_response.runner import run_bounded, write_json
from ..game import load_fhp_game
from ..loaders import sha256_file
from ..statistics import sample_summary
from . import diagnostic as diagnostic
from .development_config import ARMS, CONTROLS, cell_protocol, protocol, replication_allowed, stream
from .development_controls import ControlPolicy, exact_oracle, exact_score, leduc_reference, stochastic_loss_check
from .diagnostic_checks import correctness_checks
from .learning import NetworkPolicy, model_hash
from .source import fetch_source, load_source, safe_path
from .workflow import Events

MODULE = "fhp_evaluation.adversarial.development"


def cell_root(root, case, arm, seed):
    return Path(root) / "cells" / case / f"seed{seed}" / arm


def reference_for(args):
    if args.case == "leduc_uniform":
        game, reference = leduc_reference()
        target = None
    else:
        game = load_fhp_game()
        source = args.inputs / f"source{args.seed}"
        metadata = json.loads((source / "source.json").read_text())
        target = load_target(game, safe_path(source, metadata["checkpoint"]["path"]),
                             family="ucv", repo_root=args.native_repo)
        load_source(source, args.seed, target, need_replay=False)
        reference = (NetworkPolicy(target.native.model, target.native.feature_encoder) if args.case == "exp10"
                     else ControlPolicy(target.native.feature_encoder, args.case))
    if args.case == "exp10":
        source_hash = target.metadata["checkpoint_sha256"]
    else:
        source_hash = hashlib.sha256(json.dumps(dict(case=args.case, game=str(game),
            width=reference.feature_encoder.policy_layout.total_size), sort_keys=True).encode()).hexdigest()
    return game, target, reference, source_hash


def policies(root, spec, reference, case, arm, seed, rep, seconds, source_hash):
    cell = cell_protocol(spec, case, arm, seed)
    directory = cell_root(root, case, arm, seed)
    return [diagnostic.load_attacker(directory, cell, reference, seat, rep, seconds, source_hash)[0]
            for seat in (0, 1)]


def pooled(rows):
    """Average same-deal payoffs across fixed attacker repetitions BEFORE CI."""
    panels = np.asarray([r["seat_payoffs_mbb"] for r in rows])
    if panels.ndim != 3 or panels.shape[-1] != 2:
        raise ValueError("Expected repetition x deal x seat panels")
    return sample_summary(panels.mean(axis=(0, 2)))


def control_assessment(args, game, reference, spec, source_hash):
    seconds = spec["control_seconds"]
    records = []
    oracle = exact_oracle(game, reference) if args.case == "leduc_uniform" else None
    for rep in range(spec["repetitions"]):
        initial, trained = [policies(args.root, spec, reference, args.case, args.arm, 0, rep, t, source_hash)
                            for t in (0., seconds)]
        if oracle:
            before, after = exact_score(game, initial, reference), exact_score(game, trained, reference)
            if any(v > b + 1e-8 for v, b in zip(after["per_seat_chips"], oracle["per_seat_chips"])):
                raise ValueError("Learned policy exceeds exact best response; evaluation inconsistency")
            records.append(dict(repetition=rep, initial=before, final=after,
                gain_chips=after["mean_chips"]-before["mean_chips"],
                gap_to_exact_br_chips=oracle["mean_chips"]-after["mean_chips"],
                fraction_of_exact_br=after["mean_chips"]/oracle["mean_chips"]))
        else:
            panel = stream("control-assessment", args.case)
            before, after = [diagnostic.score(game, p, reference, spec["control_pairs"], panel) for p in (initial, trained)]
            records.append(dict(repetition=rep, initial=before, final=after,
                                gain=diagnostic.paired_difference(after, before)))
    if oracle:
        passed = all(r["final"]["mean_chips"] > 0 for r in records) and np.mean([r["gain_chips"] for r in records]) > 0
        summary = dict(oracle=oracle, units="Leduc chips/hand; exact enumeration, not mbb/hand")
    else:
        summary = dict(payoff_mbb=pooled([r["final"] for r in records]),
                       units="FHP mbb/hand; conditional paired-deal CI after averaging repetitions")
        passed = summary["payoff_mbb"]["ci95_low"] > 0
    return dict(records=records, capability_passed=bool(passed) if not spec["smoke"] else False, **summary)


def lock_selection(curves, eligible, seconds):
    candidates = [r for r in curves if r["arm"] in eligible and r["nominal_seconds"] == seconds]
    if not candidates:
        raise ValueError("No eligible development candidate")
    return max(candidates, key=lambda r: (r["payoff_mbb"]["mean"], -ARMS.index(r["arm"]), -r["repetition"]))


def qualifies(final, initial):
    return (final["payoff_mbb"]["ci95_low"] > 0
            and diagnostic.paired_difference(final, initial)["payoff_mbb"]["ci95_low"] > 0)


def fhp_assessment(args, game, reference, spec, source_hash):
    arms = list(ARMS) if args.seed == 0 else [args.arm]
    eligible = args.eligible.split(",")
    if any(a not in arms for a in eligible) or not eligible:
        raise ValueError("Invalid eligible arms")
    events = Events(args.output / "events.jsonl")
    curves = []
    for arm in arms:
        for rep in range(spec["repetitions"]):
            for seconds in cell_protocol(spec, "exp10", arm, args.seed)["checkpoints_seconds"]:
                pair = policies(args.root, spec, reference, "exp10", arm, args.seed, rep, seconds, source_hash)
                row = dict(arm=arm, repetition=rep, nominal_seconds=seconds,
                    **diagnostic.score(game, pair, reference, spec["development_pairs"],
                                       stream("development", args.seed), events))
                curves.append(row)
                write_json(args.output / "development_curves.json", dict(complete=False, curves=curves))
    endpoint = spec["attack_seconds"]
    selected = lock_selection(curves, eligible, endpoint)
    # Each arm gets a development-selected repetition for the primary contrast.
    choices = {arm: lock_selection(curves, [arm], endpoint)["repetition"] for arm in arms}
    lock = dict(selected_arm=selected["arm"], repetitions=choices, eligible=eligible,
                source_seed=args.seed, source_policy_sha256=source_hash,
                development_seed=stream("development", args.seed), **diagnostic.identity(spec))
    write_json(args.output / "selection_lock.json", lock)
    # No final data are read before the objective and repetitions are locked.
    final, initial = {}, {}
    for arm, rep in choices.items():
        for seconds, destination in ((0., initial), (endpoint, final)):
            pair = policies(args.root, spec, reference, "exp10", arm, args.seed, rep, seconds, source_hash)
            destination[arm] = diagnostic.score(game, pair, reference, spec["final_pairs"],
                                                stream("final", args.seed), events)
        write_json(args.output / "final_samples.json", dict(complete=False, initial=initial, final=final))
    chosen = selected["arm"]
    gains = {arm: diagnostic.paired_difference(final[arm], initial[arm]) for arm in arms}
    contrast = diagnostic.paired_difference(final["half_mse"], final["smooth_l1"]) if len(arms) == 2 else None
    calibration_spec = dict(spec)  # diagnostic.calibration takes this single count.
    calibration = diagnostic.calibration(game,
        policies(args.root, spec, reference, "exp10", chosen, args.seed, choices[chosen], endpoint, source_hash),
        reference, calibration_spec, seed_for_seat=lambda seat: stream("calibration", args.seed, seat))
    return dict(curves=curves, lock=lock, final=final, initial=initial, gains=gains,
                primary_contrast=contrast, selected_arm=chosen, calibration=calibration,
                qualified=bool(qualifies(final[chosen], initial[chosen])) if not spec["smoke"] else False,
                warning="No post-test switch to the other arm; replication is conditional on the locked candidate")


def worker(args):
    spec = protocol(args.smoke)
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise ValueError("Exp27 requires OpenSpiel 1.6.3")
    started = time.monotonic()
    if args.phase == "preflight":
        fetch_source(args.bucket, args.seed, args.inputs / f"source{args.seed}", include_replay=False)
    game, target, reference, source_hash = reference_for(args)
    frozen = model_hash(reference.model)
    if args.phase == "preflight":
        result = correctness_checks(game, target)
        if args.seed == 0:
            leduc, uniform = leduc_reference()
            result.update(loss_diagnostic=stochastic_loss_check(), leduc_oracle=exact_oracle(leduc, uniform))
    elif args.phase == "train":
        cell = cell_protocol(spec, args.case, args.arm, args.seed)
        result = diagnostic.train_worker(game, reference, cell, args.seat, args.repetition, args.output, source_hash,
            training_seed=stream("training", args.case, args.seed, args.seat, args.repetition),
            loss_kind=args.arm, reward_scale=cell["reward_scale"])
    elif args.phase == "control-assessment":
        result = control_assessment(args, game, reference, spec, source_hash)
    else:
        result = fhp_assessment(args, game, reference, spec, source_hash)
    if frozen != model_hash(reference.model):
        raise ValueError("Frozen reference was changed")
    result.update(status="smoke_complete" if args.smoke else "complete", smoke=args.smoke,
                  phase=args.phase, case=args.case, arm=args.arm, source_seed=args.seed,
                  source_policy_sha256=source_hash, elapsed_seconds=time.monotonic()-started,
                  runtime=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
                               open_spiel=importlib.metadata.version("open_spiel"), game=str(game)),
                  **diagnostic.identity(spec))
    write_json(args.output / "result.json", result)


def render_report(summary):
    lines = ["# Experiment 27 — attacker learning capability and loss ablation", "",
             f"Outcome: **{summary['outcome']}**. No UCV policy was changed; Experiment 21 remains separately gated.",
             "", summary["spec"]["warning"], "", "## Learning controls", "",
             "| Objective | Control | Capability criterion | Measurement |", "| --- | --- | --- | --- |"]
    for row in summary["controls"]:
        if row["case"] == "leduc_uniform":
            measurement = "; ".join(f"rep {r['repetition']}: {r['final']['mean_chips']:.4f} chips, BR gap {r['gap_to_exact_br_chips']:.4f}"
                                    for r in row["records"])
        else:
            q = row["payoff_mbb"]
            measurement = f"{q['mean']:.2f} [{q['ci95_low']:.2f}, {q['ci95_high']:.2f}] mbb/hand"
        lines.append(f"| {row['arm']} | {row['case']} | {row['capability_passed']} | {measurement} |")
    lines += ["", "## Frozen Experiment 10 assessments", "",
              "Positive favours the attacker; 95% intervals are conditional on the fitted models.", "",
              "| Source seed | Objective | Payoff mbb/hand | Gain over initial | Locked / qualified |",
              "| --- | --- | --- | --- | --- |"]
    for result in summary["assessments"]:
        for arm, final in result["final"].items():
            q, g = final["payoff_mbb"], result["gains"][arm]["payoff_mbb"]
            lines.append(f"| {result['source_seed']} | {arm} | {q['mean']:.2f} [{q['ci95_low']:.2f}, {q['ci95_high']:.2f}] | "
                         f"{g['mean']:.2f} [{g['ci95_low']:.2f}, {g['ci95_high']:.2f}] | "
                         f"{arm == result['selected_arm']} / {result['qualified'] if arm == result['selected_arm'] else 'not gated'} |")
        if result["primary_contrast"]:
            q = result["primary_contrast"]["payoff_mbb"]
            lines += ["", f"Seed-0 primary half-MSE minus Smooth L1: {q['mean']:.2f} [{q['ci95_low']:.2f}, {q['ci95_high']:.2f}] mbb/hand."]
    lines += ["", "Replication evaluates only the locked objective, so it is not a replicated two-objective treatment-effect test.",
              "No exact FHP exploitability or convergence claim is supported. Losing/inconclusive results still complete this diagnostic.",
              "", f"Coordinator wall seconds: {summary['costs']['wall_seconds']:.1f}; summed stage seconds (overlapping): {summary['costs']['summed_worker_seconds']:.1f}.",
              "Provisioning, installation and final cloud upload are additional. Full training counts, snapshots, costs and raw paired samples are in summary.json."]
    return "\n".join(lines) + "\n"


def run(args):
    spec = protocol(args.smoke)
    ident = diagnostic.identity(spec)
    if any((args.output / p).exists() for p in ("STATUS.json", "analysis", "cells", "preflight", "assessments")):
        raise FileExistsError("Use a fresh output namespace; no automatic resume")
    started, cancel, records = time.monotonic(), threading.Event(), []
    summary = dict(spec=spec, controls=[], assessments=[], preflights=[], policy_repair_authorised=False, **ident)

    def status(phase, **extra):
        write_json(args.output / "STATUS.json", dict(experiment=27, phase=phase,
            elapsed_seconds=time.monotonic()-started, **ident, **extra))

    def execute(phase, case="exp10", arm="smooth_l1", seed=0, seat=0, repetition=0, eligible=None):
        cell = cell_root(args.output, case, arm, seed)
        directory = (args.output / "preflight" / f"seed{seed}" if phase == "preflight" else
                     cell / "train" / f"seat{seat}_rep{repetition}" if phase == "train" else
                     cell / "assessment" if phase == "control-assessment" else
                     args.output / "assessments" / f"seed{seed}")
        command = [sys.executable, "-m", MODULE, "worker", "--phase", phase, "--case", case,
            "--arm", arm, "--seed", str(seed), "--seat", str(seat), "--repetition", str(repetition),
            "--root", str(args.output), "--output", str(directory), "--inputs", str(args.inputs),
            "--native-repo", str(args.native_repo), "--bucket", args.bucket,
            "--eligible", ",".join(eligible or [arm])]
        if args.smoke:
            command.append("--smoke")
        limits = 900 if phase == "preflight" else spec["training_worker_seconds"] if phase == "train" else spec["assessment_worker_seconds"]
        r = run_bounded(dict(mode="exp27", phase=phase, case=case, arm=arm, source_seed=seed,
                            seat=seat, repetition=repetition, threads=spec["threads"], **ident), directory,
                        max_seconds=limits, max_rss_mb=6000, command=command, cancel_event=cancel)
        if r["status"] != "succeeded":
            raise RuntimeError(f"{phase}/{case}/{arm}/seed{seed}/seat{seat}/rep{repetition}: {r['status']}; see {directory}/run.json")
        result = r["result"]
        expected = dict(phase=phase, case=case, arm=arm, source_seed=seed, smoke=args.smoke,
                        status="smoke_complete" if args.smoke else "complete", **ident)
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("Mixed, incomplete or untrusted worker result")
        return r

    def record(r):
        records.append(r)
        return r["result"]

    def train_block(case, arm, seed=0):
        status("train", status="running", case=case, arm=arm, seed=seed)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute, "train", case, arm, seed, seat, rep)
                       for seat in (0, 1) for rep in range(2)]
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                cancel.set()
                raise

    def preflight(seed):
        status("preflight", status="running", seed=seed)
        result = record(execute("preflight", seed=seed))
        if result.get("passed") is not True:
            raise ValueError("Correctness preflight failed")
        summary["preflights"].append(result)

    def finish(outcome):
        summary.update(outcome=outcome, status="smoke_complete" if args.smoke else "complete",
                       ready_for_exp21_design_review=outcome == "replicated_candidate_requires_human_review",
                       stages=[{k: r[k] for k in ("elapsed_seconds", "peak_rss_bytes", "request")} for r in records],
                       training=[r["result"] for r in records if r["result"]["phase"] == "train"],
                       costs=dict(wall_seconds=time.monotonic()-started,
                                  summed_worker_seconds=sum(r["elapsed_seconds"] for r in records)))
        analysis = args.output / "analysis"
        write_json(analysis / "summary.json", summary)
        (analysis / "report.md").write_text(render_report(summary))
        write_json(analysis / "SUCCESS.json", dict(status=summary["status"], outcome=outcome,
            policy_repair_authorised=False, summary_sha256=sha256_file(analysis / "summary.json"), **ident))
        status("complete", status="complete", outcome=outcome, policy_repair_authorised=False)

    try:
        preflight(0)
        for case in CONTROLS:
            # Separate blocks keep each objective on the same four-worker geometry.
            for arm in ARMS:
                train_block(case, arm)
                summary["controls"].append(record(execute("control-assessment", case, arm)))
        eligible = [arm for arm in ARMS if all(r["capability_passed"] for r in summary["controls"] if r["arm"] == arm)]
        if args.smoke:
            eligible = list(ARMS)  # Exercise orchestration only, never scientific qualification.
        write_json(args.output / "control_gate.json", dict(eligible=eligible, smoke_only=args.smoke, **ident))
        if not eligible:
            finish("no_arm_passed_learning_controls")
            return
        for arm in ARMS:
            train_block("exp10", arm)
        status("assessment", status="running", seed=0)
        first = record(execute("assessment", eligible=eligible))
        summary["assessments"].append(first)
        # The replicated objective was locked using development data only.
        if not replication_allowed(first) and not args.smoke:
            finish("locked_seed0_attacker_did_not_qualify")
            return
        chosen = first["selected_arm"]
        write_json(args.output / "replication_lock.json", dict(arm=chosen, source_seed=0,
                   smoke_only=args.smoke, qualification=first["qualified"], **ident))
        for seed in spec["replication_seeds"]:
            preflight(seed)
            train_block("exp10", chosen, seed)
            status("assessment", status="running", seed=seed)
            summary["assessments"].append(record(execute("assessment", arm=chosen, seed=seed, eligible=[chosen])))
        finish("execution_smoke_only" if args.smoke else
               "replicated_candidate_requires_human_review" if all(r["qualified"] for r in summary["assessments"])
               else "attacker_strength_not_replicated_across_all_seeds")
    except BaseException as error:
        cancel.set()
        status("stopped", status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "worker"))
    for name in ("output", "inputs", "native-repo"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--bucket", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--phase", choices=("preflight", "train", "control-assessment", "assessment"))
    p.add_argument("--case", choices=(*CONTROLS, "exp10"), default="exp10")
    p.add_argument("--arm", choices=ARMS, default=ARMS[0])
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--seat", type=int, choices=(0, 1), default=0)
    p.add_argument("--repetition", type=int, choices=(0, 1), default=0)
    p.add_argument("--eligible", default=",".join(ARMS))
    args = p.parse_args()
    for name in ("root", "output", "inputs", "native_repo"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.action == "worker":
        if args.phase is None or args.root is None:
            p.error("Worker requires --phase and --root")
        if args.phase == "preflight" and args.case != "exp10":
            p.error("Preflight requires exp10")
        worker(args)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Exp27 received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        run(args)


if __name__ == "__main__":
    main()
