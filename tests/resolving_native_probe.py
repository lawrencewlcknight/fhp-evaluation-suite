"""Separate process: exercise real Exp16 encoder/network/checkpoint loader.

Creates random, NON-uniform production-sized weights, not trained-policy results.
"""
import json
from itertools import combinations, permutations
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from fhp_evaluation.best_response.adapters import load_target
from fhp_evaluation.best_response.flop import HandSpace, make_state, validate_game
from fhp_evaluation.game import load_fhp_game, FHP_GAME_PARAMETERS
from fhp_evaluation.resolving.range_policy import RangePolicyQuery
from fhp_evaluation.resolving.solver import Deadline, solve
from fhp_evaluation.resolving.features import card_frames
from fhp_evaluation.resolving.validation import native_checks
from fhp_evaluation.resolving.runner import score_task, worker_pool
from fhp_evaluation.resolving.experiment import config
from fhp_evaluation.loaders import sha256_file
from fhp_evaluation.strategic_audit.tables import preflop_nodes

def main():
    native_root, output = sys.argv[1:]
    sys.path.insert(0, native_root)
    from fhp_escher.features import StructuredFHPMLP
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder, HAND_POLICY_LAYOUT

    torch.set_num_threads(4)
    torch.manual_seed(237)
    model = StructuredFHPMLP(HAND_POLICY_LAYOUT, [192, 192], 3, branch_width=64)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.05 * torch.randn_like(p))
    encoder = FHPHandBoardFeatureEncoder()
    payload = dict(type="fhp_ucv_escher_policy_checkpoint", version=1,
                   game=dict(parameters=dict(FHP_GAME_PARAMETERS)), input_size=213, num_actions=3,
                   policy_network_layers=[192, 192], feature_encoder=encoder.metadata(),
                   policy_model=model.checkpoint_metadata(), policy_state_dict=model.state_dict())
    path = Path(output) / "exp16_seed0_48h.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    game = load_fhp_game()
    adapter = load_target(game, path, family="ucv", repo_root=native_root)
    checks = native_checks(game, adapter)
    # Every input element, not just a few policy predictions. Cover paired,
    # trips, monotone, wheel, straight-flush and rank-boundary boards, all
    # compatible private hands and both players. Preflop uses the full deck.
    boards = [(), (0, 1, 2), (0, 4, 8), (0, 4, 48), (36, 40, 44),
              (16, 20, 32), (40, 41, 48), (1, 22, 51)]
    feature_rows = 0
    for board in boards:
        hands = np.asarray(list(combinations([c for c in range(52) if c not in board], 2)))
        frames = card_frames(hands, board)
        for hero in (0, 1):
            history = (1, 1) if board else (() if hero == 0 else (1,))
            states = [make_state(game, validate_game(game), h, hero, board=board, preflop=history) for h in hands]
            expected = np.asarray([encoder.information_state(s, hero) for s in states])
            actual = frames.copy()
            actual[:, 104:138] = expected[0, 104:138]
            np.testing.assert_array_equal(actual, expected)
            feature_rows += len(hands)
    # Suit renaming cannot change canonical features.
    for permutation in permutations(range(4)):
        rename = np.asarray(permutation)
        h = np.asarray([[0, 51], [22, 23], [9, 29]])
        b = np.asarray([16, 20, 32])
        np.testing.assert_array_equal(card_frames(h, b), card_frames(h // 4 * 4 + rename[h % 4], b // 4 * 4 + rename[b % 4]))
    checks['bitwise_feature_rows'] = feature_rows
    # Fixed-work differential: faster preparation must not change solving.
    old_p, old_strategy, old_stats = solve(game, adapter, (16, 20, 32), (1, 1), 0,
                                         seconds=None, iterations=12, query_implementation='legacy')
    new_p, new_strategy, new_stats = solve(game, adapter, (16, 20, 32), (1, 1), 0,
                                         seconds=None, iterations=12)
    np.testing.assert_array_equal(old_p.reach, new_p.reach)
    for i in old_p.blueprint:
        np.testing.assert_array_equal(old_p.blueprint[i], new_p.blueprint[i])
    assert old_stats['accepted'] == new_stats['accepted']
    if old_strategy is not None:
        for i in old_strategy:
            np.testing.assert_array_equal(old_strategy[i], new_strategy[i])
    # Full-range pre-flop parity across history contexts, including reuse.
    space = HandSpace(game, validate_game(game), (16, 20, 32))
    pre, _ = preflop_nodes(game)
    errors = []
    for hero in (0, 1):
        query = RangePolicyQuery(game, adapter, space, hero, Deadline())
        for history, (player, _) in pre.items():
            if player != hero:
                continue
            actual = query((), history)
            states = [make_state(game, validate_game(game), h, hero, preflop=history) for h in space.hands]
            expected = adapter.batch_probabilities(states, hero)
            errors.append(float(np.max(np.abs(actual - expected))))
    assert max(errors) < 1e-7, errors
    checks["preflop_full_range_max_error"] = max(errors)
    checks["timing_only_not_quality_evidence"] = []
    for board in ((17, 21, 33), (8, 9, 40)):
        _, _, stats = solve(game, adapter, board, (1, 1), 0, seconds=.5)
        checks["timing_only_not_quality_evidence"].append(stats)
    spec = config()
    spec["checkpoint_registry"]["exp16_seed0_48h"]["sha256"] = sha256_file(path)
    # Production-shaped synthetic opponents exercise both learned-policy paths.
    # These are test fixtures, not substitutes for the pinned trained models.
    for label in ("exp8_seed0_48h", "exp9_seed0_24h"):
        other_path = path.with_name(label + ".pkl")
        other_path.write_bytes(path.read_bytes())
        spec["checkpoint_registry"][label]["sha256"] = sha256_file(other_path)
    spec["search_seconds"] = 0.
    task = dict(id="native-zero-budget", seed=0, opponent="exp8", start=0, count=3)
    control = score_task(task, output, native_root, spec)
    assert all(s["search_seats"] == s["baseline_seats"] for s in control["samples"])
    checks["native_zero_budget_duplicate_pairs"] = len(control["samples"])
    spec["search_seconds"] = 2.
    task.update(id="native-gameplay-smoke", opponent="blueprint", count=3)
    # Exercise spawn after native loading in the parent, which otherwise can
    # shadow the suite with the native checkout's older evaluation package.
    with worker_pool(spec) as pool:
        futures = [pool.submit(score_task, dict(task, opponent=o), output, native_root, spec)
                   for o in ("blueprint", "exp8", "exp9", "blueprint")]
        for future in futures:
            played = future.result(timeout=45)
            assert len(played["samples"]) == 3
            assert np.isfinite([s["search_seats"] for s in played["samples"]]).all()
    checks["native_spawned_gameplay_smoke_pairs"] = 12
    (Path(output) / "native_probe.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
