"""Exp26: correctness-first, fixed-budget attacker learning curves; never repair."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import numpy as np
import torch

from ..best_response.adapters import load_target
from ..best_response.runner import run_bounded, write_json
from ..duplicate import play_hand
from ..game import load_fhp_game, MILLI_BIG_BLINDS_PER_CHIP
from ..loaders import sha256_file
from ..statistics import sample_summary
from .config import Config, stream as training_stream
from .diagnostic_config import protocol, fingerprint, stream
from .diagnostic_checks import correctness_checks, verify_collector
from .learning import NetworkPolicy, attacker_model, collect, model_hash, train_attacker
from .source import fetch_source, load_source, safe_path
from .workflow import Events, SeatPolicies, save_torch

MODULE = "fhp_evaluation.adversarial.diagnostic"


def identity(spec):
    bundle = os.environ.get("FHP_BUNDLE_SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle or ""):
        raise ValueError("Pinned FHP_BUNDLE_SHA256 required")
    return dict(protocol_hash=fingerprint(spec), bundle_sha256=bundle)


def load_reference(source, native_repo):
    game = load_fhp_game()
    metadata = json.loads((Path(source) / "source.json").read_text())
    path = safe_path(source, metadata["checkpoint"]["path"])
    target = load_target(game, path, family="ucv", repo_root=native_repo)
    load_source(source, 0, target, need_replay=False)
    return game, target, NetworkPolicy(target.native.model, target.native.feature_encoder)


def train_worker(game, reference, spec, seat, repetition, output, source_hash, *,
                 training_seed=None, loss_kind="smooth_l1", reward_scale=100.):
    records = []
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=False)
    events = Events(output / "events.jsonl")
    cfg = Config(**spec["learner"])
    seed = (training_stream(0, "qualification-train", seat, repetition)
            if training_seed is None else training_seed)
    before = model_hash(reference.model)

    def snapshot(nominal, state, metrics):
        path = checkpoints / f"checkpoint_{nominal:g}s.pt"
        state.update(identity=identity(spec), source_policy_sha256=source_hash,
                     repetition=repetition, nominal_seconds=nominal, metrics=metrics)
        save_torch(path, state)
        record = dict(path=str(path.relative_to(output)), sha256=sha256_file(path), **metrics)
        records.append(record)
        write_json(output / "checkpoint_manifest.json", dict(checkpoints=records, complete=False))
        events(force=True, kind="snapshot", **metrics)

    _, _, _, metrics = train_attacker(game, reference, seat, seed, cfg, events,
        checkpoint_seconds=spec["checkpoints_seconds"], on_checkpoint=snapshot,
        exploration_seconds=spec["exploration_seconds"], loss_kind=loss_kind,
        reward_scale=reward_scale)
    if [r["nominal_seconds"] for r in records] != spec["checkpoints_seconds"]:
        raise ValueError("Incomplete fixed training schedule")
    if model_hash(reference.model) != before:
        raise ValueError("Training modified frozen defender")
    manifest = dict(checkpoints=records, complete=True, seat=seat, repetition=repetition,
                    training_seed=seed, source_policy_sha256=source_hash, metrics=metrics, **identity(spec))
    write_json(output / "checkpoint_manifest.json", manifest)
    return manifest


def load_attacker(root, spec, reference, seat, repetition, seconds, source_hash):
    directory = Path(root) / "train" / f"seat{seat}_rep{repetition}"
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    if (not manifest.get("complete") or manifest.get("seat") != seat
            or manifest.get("repetition") != repetition or manifest.get("source_policy_sha256") != source_hash
            or any(manifest.get(k) != v for k, v in identity(spec).items())):
        raise ValueError("Attacker manifest provenance mismatch")
    rows = [r for r in manifest["checkpoints"] if r["nominal_seconds"] == seconds]
    if len(rows) != 1:
        raise ValueError("Missing/ambiguous attacker snapshot")
    path = safe_path(directory, rows[0]["path"])
    if sha256_file(path) != rows[0]["sha256"]:
        raise ValueError("Attacker snapshot checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("identity") != identity(spec) or state.get("source_policy_sha256") != source_hash
            or state.get("seat") != seat or state.get("repetition") != repetition
            or state.get("nominal_seconds") != seconds or state.get("metrics") != {k: rows[0][k] for k in state["metrics"]}):
        raise ValueError("Attacker payload provenance mismatch")
    model = attacker_model(reference.feature_encoder.policy_layout.total_size, 0)
    model.load_state_dict(state["model"])
    if model_hash(model) != rows[0]["hash"]:
        raise ValueError("Attacker model hash mismatch")
    model.eval()
    return NetworkPolicy(model, reference.feature_encoder, greedy=True), rows[0]


class Behaviour:
    """Observe only visited actions; does not alter the returned distribution."""
    def __init__(self, policy):
        self.policy = policy
        self.counts = np.zeros((2, 2, 3), dtype=np.int64)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        values = self.policy.action_probabilities(state, player)
        # OpenSpiel's public board count, used only for descriptive grouping.
        street = int("Public:" in state.information_state_string(player) and
                     state.information_state_string(player).split("Public:", 1)[1].split("]", 1)[0].strip() != "")
        action = max(values, key=values.get)
        self.counts[player, street, action] += 1
        return values


def score(game, policies, reference, count, seed, events=None):
    actor = Behaviour(SeatPolicies(policies))
    rng = np.random.default_rng(seed)
    seats = np.empty((count, 2), dtype=float)
    for index in range(count):
        chance, action = map(int, rng.integers(0, 2**63 - 1, size=2))
        seats[index, 0] = play_hand(game, (actor, reference), chance_seed=chance, action_seed=action)[0]
        seats[index, 1] = play_hand(game, (reference, actor), chance_seed=chance, action_seed=action)[1]
        if events:
            events(kind="evaluation", pairs=index + 1, total=count)
    seats *= MILLI_BIG_BLINDS_PER_CHIP
    return dict(payoff_mbb=sample_summary(seats.mean(1)),
                per_seat_mbb=[sample_summary(seats[:, s]) for s in (0, 1)],
                seat_payoffs_mbb=seats.tolist(), action_counts_by_seat_street=actor.counts.tolist())


def paired_difference(left, right):
    a, b = (np.asarray(v["seat_payoffs_mbb"]) for v in (left, right))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Paired comparisons require matching two-seat deal panels")
    return dict(payoff_mbb=sample_summary((a - b).mean(1)),
                per_seat_mbb=[sample_summary((a - b)[:, s]) for s in (0, 1)])


def choose(curves, seconds):
    candidates = [r for r in curves if r["nominal_seconds"] == seconds]
    return max(candidates, key=lambda r: (r["payoff_mbb"]["mean"], -r["repetition"]))["repetition"]


def calibration(game, policies, reference, spec, *, seed_for_seat=None):
    """Q versus realised greedy-policy returns; descriptive, not a BR target."""
    results = []
    for seat, policy in enumerate(policies):
        rng = np.random.default_rng(stream("calibration", seat) if seed_for_seat is None else seed_for_seat(seat))
        predicted, realised = [], []
        for start in range(0, spec["calibration_hands_per_seat"], 32):
            count = min(32, spec["calibration_hands_per_seat"] - start)
            eps = collect(game, policy, [reference] * count, [seat] * count, rng)
            for episode in eps:
                if not episode.rows:
                    continue
                x = torch.from_numpy(np.stack([r[0] for r in episode.rows]))
                with torch.no_grad():
                    q = policy.model(x).numpy()
                predicted.extend(q[j, row[2]].item() for j, row in enumerate(episode.rows))
                realised.extend([episode.reward] * len(episode.rows))
        delta = np.asarray(predicted) - realised
        if not len(delta):
            results.append(dict(seat=seat, decision_rows=0,
                                caveat="No attacker decisions in this calibration sample; Q calibration unavailable"))
            continue
        results.append(dict(seat=seat, decision_rows=len(delta), mean_predicted_bb=float(np.mean(predicted)),
                            mean_realised_bb=float(np.mean(realised)), mean_error_bb=float(delta.mean()),
                            rmse_bb=float(np.sqrt(np.mean(delta**2))),
                            caveat="Correlated on-policy decision rows, not independent replicates or exact Q labels"))
    return results


def evaluate_worker(game, reference, spec, root, output, source_hash):
    events = Events(output / "events.jsonl")
    curves = []

    def policies(repetition, seconds):
        return [load_attacker(root, spec, reference, seat, repetition, seconds, source_hash)[0] for seat in (0, 1)]

    for repetition in range(spec["attacker_repetitions"]):
        initial = None
        for seconds in spec["checkpoints_seconds"]:
            pair = policies(repetition, seconds)
            for seat, policy in enumerate(pair):
                verify_collector(game, policy, reference, stream("collector-check", repetition, seconds, seat))
            row = dict(repetition=repetition, nominal_seconds=seconds,
                       **score(game, pair, reference, spec["curve_pairs"], stream("development"), events))
            if initial is None:
                initial = row
            row["improvement_over_initial"] = paired_difference(row, initial)
            curves.append(row)
            write_json(output / "development_curves.json", dict(complete=False, curves=curves))
    times = spec["checkpoints_seconds"]
    selection = dict(baseline_15m_repetition=choose(curves, times[2]),
                     extended_60m_repetition=choose(curves, times[-1]),
                     rule=spec["selection"], development_seed=stream("development"),
                     source_policy_sha256=source_hash, **identity(spec))
    # Persist the selection lock BEFORE touching the independent final stream.
    write_json(output / "selection_lock.json", selection)
    entries = {"initial": (selection["extended_60m_repetition"], times[0]),
               "baseline_15m": (selection["baseline_15m_repetition"], times[2]),
               "extended_30m": (selection["extended_60m_repetition"], times[3]),
               "extended_60m": (selection["extended_60m_repetition"], times[4])}
    final = {}
    for label, (repetition, seconds) in entries.items():
        final[label] = dict(repetition=repetition, nominal_seconds=seconds,
            **score(game, policies(repetition, seconds), reference, spec["final_pairs"], stream("final"), events))
        write_json(output / "final_samples.json", dict(complete=False, candidates=final))
    contrasts = {"60m_minus_15m": paired_difference(final["extended_60m"], final["baseline_15m"]),
                 "60m_minus_30m": paired_difference(final["extended_60m"], final["extended_30m"]),
                 "60m_minus_initial": paired_difference(final["extended_60m"], final["initial"])}
    payoff = final["extended_60m"]["payoff_mbb"]
    if spec["smoke"]:
        interpretation = "execution_smoke_only"
    elif payoff["ci95_high"] < 0:
        interpretation = "extended_attacker_still_loses_review_learner_before_repair"
    elif payoff["ci95_low"] <= 0:
        interpretation = "absolute_attacker_strength_not_established"
    elif contrasts["60m_minus_15m"]["payoff_mbb"]["ci95_low"] > 0:
        interpretation = "longer_training_promising_replicate_before_policy_repair"
    else:
        interpretation = "positive_attacker_payoff_but_extension_benefit_not_established"
    return dict(development_curves=curves, selection=selection, final=final, paired_contrasts=contrasts,
                calibration=calibration(game, policies(selection["extended_60m_repetition"], times[-1]), reference, spec),
                interpretation=interpretation, policy_repair_authorised=False,
                warning=spec["warning"], final_seed=stream("final"))


def worker(args):
    spec = protocol(args.smoke)
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise ValueError("Exp26 requires OpenSpiel 1.6.3")
    source = args.inputs / "source"
    if args.phase == "preflight":
        fetch_source(args.bucket, 0, source, include_replay=False)
    game, target, reference = load_reference(source, args.native_repo)
    before = model_hash(reference.model)
    source_hash = target.metadata["checkpoint_sha256"]
    started = time.monotonic()
    if args.phase == "preflight":
        result = correctness_checks(game, target)
    elif args.phase == "train":
        result = train_worker(game, reference, spec, args.seat, args.repetition, args.output, source_hash)
    else:
        result = evaluate_worker(game, reference, spec, args.root, args.output, source_hash)
    if model_hash(reference.model) != before:
        raise ValueError("Frozen source changed")
    result.update(source_policy_sha256=source_hash, spec=spec, phase=args.phase,
                  elapsed_seconds=time.monotonic() - started,
                  runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                               open_spiel=importlib.metadata.version("open_spiel"), threads=torch.get_num_threads(),
                               game=str(game)),
                  status="smoke_complete" if args.smoke else "complete", **identity(spec))
    write_json(args.output / "result.json", result)


def write_report(path, result, costs):
    lines = ["# Experiment 26 — attacker learning diagnostic", "", result["warning"], "",
             "Positive mbb/hand favours the attacker. Intervals are conditional 95% Monte Carlo t intervals.",
             "No repaired UCV policy or exact exploitability value is produced.", "",
             "## Development learning curves", "",
             "| Rep | Minutes | Payoff | 95% interval | Seat 0 | Seat 1 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in result["development_curves"]:
        q = r["payoff_mbb"]
        lines.append(f"| {r['repetition']} | {r['nominal_seconds']/60:g} | {q['mean']:.2f} | "
                     f"[{q['ci95_low']:.2f}, {q['ci95_high']:.2f}] | "
                     f"{r['per_seat_mbb'][0]['mean']:.2f} | {r['per_seat_mbb'][1]['mean']:.2f} |")
    lines += ["", "Seat payoffs include blind/position asymmetry; they are not standalone exploitability measures.",
              "Development curves are exploratory and share deals; individual intervals are not simultaneous bands.",
              "", "## Independent final assessment", "",
              "| Candidate | Rep | Payoff | 95% interval |", "| --- | --- | --- | --- |"]
    for label, r in result["final"].items():
        q = r["payoff_mbb"]
        lines.append(f"| {label} | {r['repetition']} | {q['mean']:.2f} | [{q['ci95_low']:.2f}, {q['ci95_high']:.2f}] |")
    lines += ["", "Primary contrast: 60m minus independently development-selected 15m; positive favours extension."]
    for label, r in result["paired_contrasts"].items():
        q = r["payoff_mbb"]
        lines.append(f"- {label}: {q['mean']:.2f} [{q['ci95_low']:.2f}, {q['ci95_high']:.2f}].")
    lines += ["", f"Decision aid: {result['interpretation']}. No automatic downstream run is authorised.", "",
              "## Costs", "", f"Coordinator wall seconds through assessment: {costs['wall_seconds']:.1f}.",
              f"Sum of watchdog stage elapsed seconds (parallel workers overlap): {costs['summed_worker_seconds']:.1f}.",
              "Per-checkpoint active time, hands, updates and snapshot overhead are in training manifests.",
              "Batch duration additionally includes VM provisioning, installation and upload."]
    path.write_text("\n".join(lines) + "\n")


def run(args):
    spec = protocol(args.smoke)
    ident = identity(spec)
    if any((args.output / name).exists() for name in ("STATUS.json", "analysis", "preflight", "train", "assessment")):
        raise FileExistsError("Occupied diagnostic output; use a fresh run, no automatic resume")
    start = time.monotonic()
    records, cancel = [], threading.Event()

    def status(phase, **extra):
        write_json(args.output / "STATUS.json", dict(phase=phase, elapsed_seconds=time.monotonic()-start,
                   experiment=26, smoke=args.smoke, **ident, **extra))

    def execute(phase, seat=0, repetition=0):
        directory = args.output / (f"train/seat{seat}_rep{repetition}" if phase == "train" else phase)
        command = [sys.executable, "-m", MODULE, "worker", "--phase", phase,
                   "--seat", str(seat), "--repetition", str(repetition), "--root", str(args.output),
                   "--inputs", str(args.inputs), "--native-repo", str(args.native_repo),
                   "--bucket", args.bucket, "--output", str(directory)]
        if args.smoke:
            command.append("--smoke")
        seconds = 900 if phase == "preflight" else spec["training_worker_seconds"] if phase == "train" else spec["evaluation_worker_seconds"]
        r = run_bounded(dict(mode="exp26", phase=phase, seat=seat, repetition=repetition,
                             threads=spec["threads"], **ident), directory,
                        max_seconds=seconds, max_rss_mb=6000, command=command, cancel_event=cancel)
        if r["status"] != "succeeded":
            raise RuntimeError(f"{phase} seat{seat}/rep{repetition}: {r['status']}; inspect {directory}/run.json")
        result = r["result"]
        if any(result.get(k) != v for k, v in ident.items()) or result["status"] != ("smoke_complete" if args.smoke else "complete"):
            raise ValueError("Mixed/incomplete diagnostic result")
        return r

    try:
        status("preflight", status="running")
        preflight = execute("preflight")
        if preflight["result"].get("passed") is not True:
            raise ValueError("Correctness checks did not pass")
        records.append(preflight)
        status("train", status="running", completed=0, total=4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute, "train", seat, rep) for seat in (0, 1) for rep in range(2)]
            try:
                for i, future in enumerate(as_completed(futures)):
                    records.append(future.result())
                    status("train", status="running", completed=i+1, total=4)
            except BaseException:
                cancel.set()
                raise
        status("assessment", status="running")
        assessment = execute("assessment")
        records.append(assessment)
        result = assessment["result"]
        costs = dict(wall_seconds=time.monotonic()-start,
                     summed_worker_seconds=sum(r["elapsed_seconds"] for r in records),
                     peak_worker_rss_bytes=max(r["peak_rss_bytes"] for r in records))
        analysis = args.output / "analysis"
        summary = dict(result, costs=costs, preflight=preflight["result"],
                       training=[r["result"] for r in records if r["result"]["phase"] == "train"])
        write_json(analysis / "summary.json", summary)
        write_report(analysis / "report.md", result, costs)
        write_json(analysis / "SUCCESS.json", dict(status="smoke_complete" if args.smoke else "complete",
                   summary_sha256=sha256_file(analysis / "summary.json"), policy_repair_authorised=False, **ident))
        status("complete", status="complete", policy_repair_authorised=False)
    except BaseException as error:
        cancel.set()
        status("stopped", status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "worker"))
    for name in ("output", "inputs", "native-repo"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--phase", choices=("preflight", "train", "assessment"))
    p.add_argument("--root", type=Path)
    p.add_argument("--seat", type=int, choices=(0, 1), default=0)
    p.add_argument("--repetition", type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    # macOS /tmp -> /private/tmp and relative local invocations must agree with
    # safe_path's resolved checkpoint paths before source fetch/validation.
    for name in ("output", "inputs", "native_repo", "root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.action == "worker":
        if args.phase is None or args.root is None:
            p.error("worker needs --phase and --root")
        worker(args)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Exp26 received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        run(args)


if __name__ == "__main__":
    main()
