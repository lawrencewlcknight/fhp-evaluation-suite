"""Explicit qualification, training and locked evaluation phases for UCV Exp21."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import torch

from ..best_response.adapters import load_target
from ..best_response.runner import run_bounded, write_json
from ..best_response.worker import compare
from ..duplicate import play_hand
from ..game import load_fhp_game, MILLI_BIG_BLINDS_PER_CHIP
from ..loaders import sha256_file
from ..rule_agents import published_rule_agents
from ..statistics import sample_summary
from .config import ARMS, configuration, stream
from .learning import NetworkPolicy, model_hash, ordinary_fit, repair, train_attacker
from .source import Replay, fetch_panel, fetch_source, load_source, safe_path


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Events:
    def __init__(self, path):
        self.path, self.last = Path(path), 0.

    def __call__(self, *, force=False, **values):
        now = time.monotonic()
        if force or now - self.last >= 10:
            record = dict(unix_seconds=time.time(), **values)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            self.last = now


class SeatPolicies:
    """A legal fixed behavioural response, one information-state network per seat."""
    def __init__(self, policies):
        self.policies = policies

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        return self.policies[player].action_probabilities(state, player)


def pairs(game, a, b, count, seed, progress=None):
    rng = np.random.default_rng(seed)
    values = []
    for index in range(count):
        chance, action = (int(rng.integers(0, 2**63 - 1)) for _ in range(2))
        p0 = play_hand(game, (a, b), chance_seed=chance, action_seed=action)[0]
        p1 = play_hand(game, (b, a), chance_seed=chance, action_seed=action)[1]
        values.append(.5 * (p0 + p1) * MILLI_BIG_BLINDS_PER_CHIP)
        if progress is not None:
            progress(kind="duplicate_play", completed=index + 1, total=count)
    return np.asarray(values, dtype=np.float64)


def population(game, victim, cfg, source_seed, namespace, output, events):
    pool, initial, metrics = [[], []], [[], []], []
    for seat in (0, 1):
        for index in range(cfg.attackers_per_seat):
            events(force=True, kind="attacker_started", namespace=namespace, seat=seat, index=index)
            policy, before, state, timing = train_attacker(
                game, victim, seat, stream(source_seed, namespace, seat, index), cfg, events)
            save_torch(output / f"attacker_{seat}_{index}.pt", state)
            pool[seat].append(policy)
            initial[seat].append(before)
            metrics.append(timing)
    return pool, initial, metrics


def choose_attacker(game, pool, victim, cfg, seed, events):
    # Each candidate combines independently trained networks for the two seats.
    candidates = [SeatPolicies([pool[0][i], pool[1][i]]) for i in range(cfg.attackers_per_seat)]
    scores = [float(pairs(game, p, victim, cfg.selection_pairs, seed, events).mean()) for p in candidates]
    selected = int(np.argmax(scores))
    return candidates[selected], selected, scores


def qualification(game, reference, cfg, seed, output, events):
    pool, initial, timings = population(game, reference, cfg, seed, "qualification-train", output, events)
    selected, index, scores = choose_attacker(
        game, pool, reference, cfg, stream(seed, "qualification-selection"), events)
    testing_seed = stream(seed, "qualification-test")
    trained = pairs(game, selected, reference, cfg.gate_pairs, testing_seed, events)
    untrained = pairs(game, SeatPolicies([initial[0][index], initial[1][index]]),
                      reference, cfg.gate_pairs, testing_seed, events)
    gain, payoff = sample_summary(trained - untrained), sample_summary(trained)
    # No optional stopping, no retries until a favourable CI. Both predeclared
    # tests must pass on a separate fixed panel, after candidate selection.
    passed = gain["ci95_low"] > 0 and payoff["ci95_low"] > 0
    return dict(gate_passed=bool(passed) if not cfg.smoke else False,
                smoke_execution_only=cfg.smoke, selected_attacker=index,
                selection_means_mbb=scores, payoff_mbb=payoff, improvement_mbb=gain,
                pair_payoffs_mbb=trained.tolist(), initial_pair_payoffs_mbb=untrained.tolist(),
                timings=timings, requirement="positive held-out payoff AND gain over initial attacker",
                decision="smoke_only" if cfg.smoke else "proceed" if passed else "inconclusive_stop")


def save_policy(reference, student, path, arm, cfg, source_hash, extra=None):
    payload = deepcopy(reference.native.checkpoint)
    payload.update(experiment_id=21, experiment_name="exp21_fhp_adversarial_preservation",
                   algorithm_id=f"exp21_{arm}", algorithm="UCV-ESCHER post-trained policy",
                   policy_state_dict={k: v.detach().cpu().clone() for k, v in student.model.state_dict().items()})
    payload["post_training"] = dict(arm=arm, config=cfg.to_dict(), source_sha256=source_hash,
                                   original_training_hours=24, exact_cfr_average=False,
                                   ucv_solver_resumable=False, **(extra or {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def train(game, target, replay, cfg, seed, output, events, qualification_seconds=0):
    reference = NetworkPolicy(deepcopy(target.native.model), target.native.feature_encoder)
    initial_hash = model_hash(reference.model)
    records = {}
    source_hash = target.metadata["checkpoint_sha256"]
    save_policy(target, reference, output / "unchanged.pkl", "unchanged", cfg, source_hash)
    for arm in ("adversarial", "preserved"):
        student = NetworkPolicy(deepcopy(reference.model), reference.feature_encoder)
        optimizer = torch.optim.Adam(student.model.parameters(), lr=cfg.policy_lr)
        pool, rounds = [[], []], []
        start = time.monotonic()
        for round_id in range(cfg.rounds):
            events(force=True, kind="round_started", arm=arm, round=round_id)
            # Matched initial attacker RNGs across the two arms; later policies
            # differ, as expected. Networks observe a frozen victim each round.
            victim = NetworkPolicy(deepcopy(student.model), student.feature_encoder)
            directory = output / arm / f"round_{round_id}"
            new, _, timings = population(game, victim, cfg, seed, f"repair-train-{round_id}", directory, events)
            for seat in (0, 1):
                pool[seat].extend(new[seat])
            timing = repair(game, student, reference, replay, pool, optimizer,
                            stream(seed, "repair-sampling", round_id), cfg,
                            preserve=arm == "preserved", progress=events)
            rounds.append(dict(round=round_id, attackers=timings, repair=timing,
                               pool_size_per_seat=[len(v) for v in pool]))
            save_policy(target, student, directory / "policy.pkl", arm, cfg, source_hash,
                        extra=dict(round=round_id))
            save_torch(directory / "repair_state.pt", dict(optimizer=optimizer.state_dict(),
                       model=student.model.state_dict(), reference_hash=initial_hash,
                       sampling=timing, continuation="round boundary; see retained opponent population"))
        records[arm] = dict(seconds=time.monotonic() - start, rounds=rounds, initial_hash=initial_hash,
                            final_hash=model_hash(student.model))
        save_policy(target, student, output / f"{arm}.pkl", arm, cfg, source_hash)
        if model_hash(reference.model) != initial_hash:
            raise RuntimeError("Frozen reference was modified")
    ordinary = NetworkPolicy(deepcopy(reference.model), reference.feature_encoder)
    # Matched to the PRIMARY preserved arm's end-to-end repair wall time PLUS
    # qualification cost, including source preparation and attack validation.
    # This deliberately gives the ordinary-fitting control a generous budget.
    timing, optimizer = ordinary_fit(ordinary, replay, stream(seed, "ordinary-fitting"), cfg,
                                     records["preserved"]["seconds"] + qualification_seconds, events)
    records["ordinary"] = dict(**timing, initial_hash=initial_hash, final_hash=model_hash(ordinary.model))
    save_policy(target, ordinary, output / "ordinary.pkl", "ordinary", cfg, source_hash)
    save_torch(output / "ordinary_state.pt", dict(model=ordinary.model.state_dict(),
               optimizer=optimizer.state_dict(), rng=timing["rng"]))
    return dict(arms=records, initial_hash=initial_hash,
                qualification_seconds_charged_to_control=qualification_seconds,
                policy_hashes={arm: sha256_file(output / f"{arm}.pkl") for arm in ARMS},
                endpoint_selection="fixed final round; no best-on-test selection",
                compute_control="ordinary matches preserved repair plus qualification end-to-end seconds")


def evaluate(game, policies, cfg, seed, arm, output, events, panel=None):
    victim = policies[arm]
    network = NetworkPolicy(victim.native.model, victim.native.feature_encoder)
    pool, initial, timings = population(game, network, cfg, seed, f"heldout-train-{arm}", output, events)
    attacker, index, own_scores = choose_attacker(
        game, pool, victim, cfg, stream(seed, f"heldout-own-selection-{arm}"), events)
    matrix = {}
    for defender in ARMS:
        # All final challengers cross-play every final candidate. Pool selection
        # and scoring use separate streams, common across arms AND source seeds.
        validation = pairs(game, attacker, policies[defender], cfg.selection_pairs,
                           stream(0, "locked-pool-selection"), events)
        testing = pairs(game, attacker, policies[defender], cfg.test_pairs,
                        stream(0, "locked-pool-test"), events)
        matrix[defender] = dict(selection_mean_mbb=float(validation.mean()),
                               pair_payoffs_mbb=testing.tolist(), payoff_mbb=sample_summary(testing))
    untrained = pairs(game, SeatPolicies([initial[0][index], initial[1][index]]), victim,
                      cfg.test_pairs, stream(0, "locked-pool-test"), events)
    gain = sample_summary(np.asarray(matrix[arm]["pair_payoffs_mbb"]) - untrained)
    strength = dict(learning_gain_mbb=gain, own_payoff_mbb=matrix[arm]["payoff_mbb"],
                    useful_attack_demonstrated=bool(gain["ci95_low"] > 0 and
                                                    matrix[arm]["payoff_mbb"]["ci95_low"] > 0))
    h2h = pairs(game, victim, policies["unchanged"], cfg.test_pairs, stream(0, "locked-h2h"), events)
    rule_results = {}
    for name, opponent in published_rule_agents(game).items():
        scores = pairs(game, victim, opponent, cfg.test_pairs, stream(0, f"locked-rule-{name}"), events)
        rule_results[name] = dict(summary=sample_summary(scores), pair_payoffs_mbb=scores.tolist())
    panel_results = {}
    for name, opponent in (panel or {}).items():
        scores = pairs(game, victim, opponent, cfg.test_pairs, stream(0, f"locked-panel-{name}"), events)
        panel_results[name] = dict(summary=sample_summary(scores), pair_payoffs_mbb=scores.tolist(),
                                   source=opponent.metadata)
    events(force=True, kind="specialised_response_started", arm=arm, pairs=cfg.br_pairs)
    response = compare(game, victim, dict(deals=cfg.br_pairs, rollouts=cfg.preflop_rollouts,
                       search_seed=stream(0, "locked-br-search") % (2**31 - 1),
                       seed=stream(0, "locked-br-test"), record_pair_timings=True),
                       progress_path=output / "br_partial.json")
    return dict(arm=arm, selected_attacker=index, own_selection_means_mbb=own_scores,
                attacker_timings=timings, challenger_matrix=matrix, attacker_strength=strength,
                h2h_vs_unchanged=dict(summary=sample_summary(h2h), pair_payoffs_mbb=h2h.tolist()),
                rule_panel=rule_results, frozen_ucv_panel=panel_results, specialised_response=response,
                units="mbb_per_hand", exact_exploitability=False,
                interpretation="Lower scores against tested responders are not an upper bound on exploitability")


def require_gate(gate, cfg, seed, source_hash, bundle_hash):
    if (gate.get("phase") != "qualify" or gate.get("seed") != seed
            or gate.get("config_hash") != cfg.fingerprint()
            or gate.get("source_policy_sha256") != source_hash
            or gate.get("bundle_sha256") != bundle_hash):
        raise ValueError("Qualification provenance differs from requested experiment")
    if cfg.smoke:
        if not gate.get("smoke_execution_only"):
            raise ValueError("Expected smoke-only gate")
    elif gate.get("gate_passed") is not True:
        raise ValueError("Attacker strength gate did not pass; do not submit policy repair")


def execute(args):
    cfg = configuration(args.smoke)
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise RuntimeError("Production/smoke worker requires pinned OpenSpiel 1.6.3")
    torch.set_num_threads(cfg.threads)
    torch.set_num_interop_threads(1)
    output = Path(args.output).resolve()
    events = Events(output / "events.jsonl")
    started = time.monotonic()
    game = load_fhp_game()
    source = Path(args.source).resolve()
    if not source.exists():
        if not args.bucket:
            raise ValueError("Missing local source; --bucket required to fetch pinned source")
        fetch_source(args.bucket, args.seed, source, include_replay=args.phase == "qualify")
    source_metadata = json.loads((source / "source.json").read_text())
    checkpoint = safe_path(source, source_metadata["checkpoint"]["path"])
    target = load_target(game, checkpoint, family="ucv", repo_root=args.native_repo)
    # Validate the source before ANY training, including qualification. Only the
    # selected policy replay is grouped, never critic buffers or full game trees.
    replay, _ = load_source(source, args.seed, target, need_replay=args.phase == "qualify")
    bundle_hash = os.environ.get("FHP_BUNDLE_SHA256", "local")
    common = dict(experiment="exp21_fhp_adversarial_preservation", seed=args.seed,
                  phase=args.phase, config=cfg.to_dict(), config_hash=cfg.fingerprint(),
                  source_policy_sha256=target.metadata["checkpoint_sha256"],
                  source=source_metadata, bundle_sha256=bundle_hash,
                  runtime=dict(python=platform.python_version(), platform=platform.platform(),
                               torch=str(torch.__version__), numpy=np.__version__,
                               open_spiel=importlib.metadata.version("open_spiel"),
                               torch_threads=torch.get_num_threads(), game=str(game)))
    write_json(output / "protocol.json", common)
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    if args.phase == "qualify":
        replay_hash = replay.save(output / "replay", target.metadata["checkpoint_sha256"])
        result = qualification(game, reference, cfg, args.seed, output, events)
        result.update(replay_manifest_sha256=replay_hash,
                      grouped_replay=dict(rows=replay.rows, groups=len(replay.x), objective_scale=replay.scale))
    elif args.phase == "train":
        gate = json.loads(Path(args.gate).read_text())
        require_gate(gate, cfg, args.seed, target.metadata["checkpoint_sha256"], bundle_hash)
        replay = Replay.load(args.replay, gate["replay_manifest_sha256"], target.metadata["checkpoint_sha256"])
        result = train(game, target, replay, cfg, args.seed, output, events,
                       qualification_seconds=0 if cfg.smoke else gate["elapsed_seconds"])
    else:
        training = Path(args.training)
        manifest = json.loads((training / "result.json").read_text())
        if (manifest["phase"] != "train" or manifest["config_hash"] != cfg.fingerprint()
                or manifest["seed"] != args.seed or manifest["bundle_sha256"] != bundle_hash
                or manifest["source_policy_sha256"] != target.metadata["checkpoint_sha256"]):
            raise ValueError("Training/evaluation provenance mismatch")
        policies = {}
        for arm in ARMS:
            path = training / f"{arm}.pkl"
            if sha256_file(path) != manifest["policy_hashes"][arm]:
                raise ValueError("Modified final candidate")
            policies[arm] = load_target(game, path, family="ucv", repo_root=args.native_repo)
        panel = {name: load_target(game, path, family="ucv", repo_root=args.native_repo)
                 for name, path in fetch_panel(args.bucket, output / "panel_sources").items()} if args.bucket else {}
        result = evaluate(game, policies, cfg, args.seed, args.arm, output, events, panel)
    result.update(common, elapsed_seconds=time.monotonic() - started,
                  status="smoke_complete" if cfg.smoke else "complete")
    write_json(output / "result.json", result)
    write_json(output / "SUCCESS.json", dict(result_sha256=sha256_file(output / "result.json"),
               phase=args.phase, config_hash=cfg.fingerprint(), seed=args.seed,
               gate_passed=result.get("gate_passed")))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qualify", "train", "evaluate"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase == "train" and (args.gate is None or args.replay is None) or
            args.phase == "evaluate" and (args.training is None or args.arm is None)):
        parser.error("Training needs --gate and --replay; evaluation needs --training and --arm")
    if args.worker:
        execute(args)
    else:
        seconds = 900 if args.smoke else dict(qualify=7200, train=50400, evaluate=21600)[args.phase]
        command = [sys.executable, "-m", __name__, *(sys.argv[1:] if argv is None else argv), "--worker"]
        # __name__ is __main__ under -m; explicitly pin the importable worker.
        command[2] = "fhp_evaluation.adversarial.workflow"
        result = run_bounded(dict(mode="exp21", phase=args.phase, seed=args.seed,
                             smoke=args.smoke, threads=2), args.output, max_seconds=seconds,
                             max_rss_mb=56000 if args.phase != "evaluate" else 13000, command=command)
        if result["status"] != "succeeded":
            raise SystemExit(f"Phase failed: {result['status']}; inspect {args.output}/run.json")


if __name__ == "__main__":
    main()
