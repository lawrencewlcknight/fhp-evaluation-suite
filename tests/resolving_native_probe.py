"""Separate process: exercise real Exp16 encoder/network/checkpoint loader.

Creates random, NON-uniform production-sized weights, not trained-policy results.
"""
import json
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
    spec["search_seconds"] = 0.
    task = dict(id="native-zero-budget", seed=0, opponent="candid_statistician", start=0, count=3)
    control = score_task(task, output, native_root, spec)
    assert all(s["search_seats"] == s["baseline_seats"] for s in control["samples"])
    checks["native_zero_budget_duplicate_pairs"] = len(control["samples"])
    spec["search_seconds"] = .5
    task.update(id="native-gameplay-smoke", opponent="blueprint", count=3)
    # Exercise spawn after native loading in the parent, which otherwise can
    # shadow the suite with the native checkout's older evaluation package.
    with worker_pool(spec) as pool:
        futures = [pool.submit(score_task, task, output, native_root, spec) for _ in range(4)]
        for future in futures:
            played = future.result(timeout=45)
            assert len(played["samples"]) == 3
            assert np.isfinite([s["search_seats"] for s in played["samples"]]).all()
    checks["native_spawned_gameplay_smoke_pairs"] = 12
    (Path(output) / "native_probe.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
