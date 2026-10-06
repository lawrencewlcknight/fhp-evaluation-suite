"""Internal worker. Use the watchdog-backed br-* CLI commands, not this directly."""

import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ..duplicate import play_hand
from ..game import load_fhp_game, MILLI_BIG_BLINDS_PER_CHIP
from ..lbr import LBRConfig
from ..statistics import sample_summary
from .adapters import load_target, PolicyAdapter
from .flop import FlopBestResponse, FullFlopResponsePolicy, make_state
from .runner import write_json
from .validation import run_validation, FixedPolicy


def event(kind, **values):
    print(json.dumps(dict(event=kind, **values), allow_nan=False), flush=True)


def preflop_prefixes(game):
    state = game.new_initial_state()
    for _ in range(4):
        state.apply_action(state.legal_actions()[0])
    prefixes = []
    def walk(cursor, path):
        if cursor.is_terminal():
            return
        if cursor.is_chance_node():
            prefixes.append(path)
        else:
            for action in cursor.legal_actions():
                walk(cursor.child(action), path + (action,))
    walk(state, ())
    return prefixes


def profile(game, target, spec, *, progress_path=None):
    engine = FlopBestResponse(game, target)
    boards = list(combinations(engine.deck, 3))
    rng = np.random.default_rng(spec["seed"])
    indices = rng.choice(len(boards), size=spec["boards"] + 1, replace=False)
    prefixes = preflop_prefixes(game)
    # Fixed warmup, explicitly excluded from steady-state measurements.
    engine.solve(boards[indices[0]], prefixes[0], 0)
    records = []
    for repeat in range(spec["repeats"]):
        for index in indices[1:]:
            started = time.perf_counter()
            before = dict(target.stats)
            for prefix in prefixes:
                for player in (0, 1):
                    engine.solve(boards[index], prefix, player)
            row = dict(repeat=repeat, board=list(boards[index]),
                       elapsed_seconds=time.perf_counter() - started,
                       policy_rows=target.stats["rows"] - before["rows"],
                       policy_query_seconds=target.stats["query_seconds"] - before["query_seconds"])
            records.append(row)
            if progress_path is not None:
                write_json(progress_path, dict(complete=False, records=records,
                                               scope="partial_timing_diagnostics_only"))
            event("board_completed", **row)
    timings = [row["elapsed_seconds"] for row in records]
    return dict(experiment="exp2_br_feasibility", scope="timing_only_not_exploitability",
                threads=spec["threads"], repeats=spec["repeats"], boards=spec["boards"],
                board_seed=spec["seed"], warmup_boards=1, preflop_prefixes=len(prefixes),
                records=records, median_seconds_per_board=float(np.median(timings)),
                projected_all_board_hours=float(np.mean(timings) * len(boards) / 3600),
                projection_warning="Planning estimate only; excludes full-game integration, loading and scheduling.")


def summary(values):
    result = sample_summary(np.asarray(values, dtype=float))
    return {key: float(value) for key, value in result.items()}


def compare(game, target, spec, *, progress_path=None):
    config = LBRConfig(seed=spec["search_seed"], preflop_rollout_samples=spec["rollouts"])
    baseline = target.preflop_policy(game, config)
    improved = FullFlopResponsePolicy(game, target, config=config)
    rng = np.random.default_rng(spec["seed"])
    values = {"lbr": [], "full_flop": []}
    seats = {"lbr": [[], []], "full_flop": [[], []]}
    pair_seconds = []
    start = 0
    progress_identity = {
        "schema_version": 1,
        "evaluation_seed": int(spec["seed"]),
        "search_seed": int(spec["search_seed"]),
        "preflop_rollouts": int(spec["rollouts"]),
        "requested_pairs": int(spec["deals"]),
        "record_pair_timings": bool(spec.get("record_pair_timings", False)),
    }
    if progress_path is not None and spec.get("resume_partial") and Path(progress_path).is_file():
        previous = json.loads(Path(progress_path).read_text())
        if previous.get("progress_identity") != progress_identity:
            raise ValueError("Partial comparison identity does not match this request")
        start = int(previous.get("completed_pairs", -1))
        previous_values = previous.get("pair_payoffs_mbb")
        previous_seats = previous.get("seat_payoffs_chips")
        if not 0 <= start <= spec["deals"]:
            raise ValueError("Invalid completed-pair count in partial comparison")
        if (set(previous_values or ()) != set(values)
                or set(previous_seats or ()) != set(seats)
                or any(len(previous_values[label]) != start for label in values)
                or any(len(previous_seats[label]) != 2
                       or any(len(row) != start for row in previous_seats[label])
                       for label in seats)):
            raise ValueError("Malformed partial comparison payload")
        values = {label: list(previous_values[label]) for label in values}
        seats = {label: [list(row) for row in previous_seats[label]] for label in seats}
        if spec.get("record_pair_timings"):
            pair_seconds = previous.get("pair_elapsed_seconds", [])
            if (len(pair_seconds) != start or not np.isfinite(pair_seconds).all()
                    or any(seconds < 0 for seconds in pair_seconds)):
                raise ValueError("Malformed partial pair timings")
        for _ in range(start):
            rng.integers(0, 2**63 - 1)
            rng.integers(0, 2**63 - 1)
        event("comparison_resumed", completed=start, total=spec["deals"])
    for index in range(start, spec["deals"]):
        pair_started = time.perf_counter()
        chance, action = (int(rng.integers(0, 2**63 - 1)) for _ in range(2))
        for label, responder in (("lbr", baseline), ("full_flop", improved)):
            if len(baseline._action_cache) >= 2048:
                baseline._action_cache.clear()
            p0 = play_hand(game, (responder, target), chance_seed=chance, action_seed=action)[0]
            p1 = play_hand(game, (target, responder), chance_seed=chance, action_seed=action)[1]
            seats[label][0].append(p0)
            seats[label][1].append(p1)
            values[label].append(.5 * (p0 + p1) * MILLI_BIG_BLINDS_PER_CHIP)
        if spec.get("record_pair_timings"):
            pair_seconds.append(time.perf_counter() - pair_started)
        if progress_path is not None:
            # Preserve completed pairs if the watchdog interrupts the worker.
            # No CI: time-limited partial samples are not final inferential results.
            write_json(progress_path, dict(complete=False, completed_pairs=index + 1,
                                           requested_pairs=spec["deals"], pair_payoffs_mbb=values,
                                           seat_payoffs_chips=seats, evaluation_seed=spec["seed"],
                                           progress_identity=progress_identity,
                                           pair_elapsed_seconds=pair_seconds,
                                           scope="partial_payoff_diagnostics_not_a_final_estimate"))
        event("deal_pair_completed", completed=index + 1, total=spec["deals"])
    return dict(experiment="exp3_br_checkpoint_comparison", metric="exploitability_lower_bound_estimate",
                exact_full_game=False, units="mbb_per_hand", num_deal_pairs=spec["deals"],
                evaluation_seed=spec["seed"], search_seed=spec["search_seed"], preflop_rollouts=spec["rollouts"],
                lbr=summary(values["lbr"]), full_flop=summary(values["full_flop"]),
                paired_improvement=summary(np.subtract(values["full_flop"], values["lbr"])),
                pair_payoffs_mbb=values, seat_payoffs_chips=seats,
                pair_elapsed_seconds=pair_seconds,
                sampling_ci_method="Student_t_over_independent_duplicate_pairs",
                response_approximation_error="not_bounded_by_the_sampling_confidence_interval",
                interpretation="CIs measure payoff sampling uncertainty, not distance to a true best response.")


def execute(spec, *, progress_path=None):
    version = importlib.metadata.version("open_spiel")
    if version != "1.6.3":
        raise RuntimeError(f"Pinned OpenSpiel 1.6.3 required; found {version}")
    torch.set_num_threads(spec.get("threads", 1))
    torch.set_num_interop_threads(1)
    if spec["mode"] == "validate":
        return run_validation()
    game = load_fhp_game()
    started = time.perf_counter()
    event("loading_target")
    if spec["family"] == "control":
        target = PolicyAdapter(FixedPolicy(game, spec.get("control", "random")),
                               batch_size=spec["batch_size"], metadata=dict(family="control"))
    else:
        target = load_target(game, spec["checkpoint"], family=spec["family"],
                             repo_root=spec.get("repo_root"), batch_size=spec["batch_size"],
                             model_batch_size=spec["model_batch_size"])
    load_seconds = time.perf_counter() - started
    # Raw, raised, capped and flop states: always run a scalar-v-batch admission gate.
    deck = tuple(game.new_initial_state().legal_actions())
    states = []
    for pre, board, post in [((), (), ()), ((2,), (), ()), ((2, 2, 2), (), ()),
                             ((1, 1), (16, 20, 32), ()), ((1, 1), (16, 20, 32), (2,))]:
        for player in (0, 1):
            state = make_state(game, deck, (0, 4), player, board=board, preflop=pre, flop=post)
            states.append(state)
    parity = target.check_scalar_parity(states)
    event("policy_validated", **parity)
    result = (profile(game, target, spec, progress_path=progress_path) if spec["mode"] == "profile"
              else compare(game, target, spec, progress_path=progress_path))
    result.update(target=target.metadata, policy_parity=parity, target_load_seconds=load_seconds,
                  policy_stats=target.stats, open_spiel=version, torch=torch.__version__,
                  game=str(game), full_game_fhp_exploitability_implemented=False)
    return result


def main():
    path = Path(sys.argv[1])
    try:
        result = execute(json.loads(path.read_text()), progress_path=path.parent / "partial.json")
        write_json(path.parent / "result.json", result)
        event("completed")
    except Exception as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
