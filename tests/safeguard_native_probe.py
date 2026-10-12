"""Manual local integration check using one trusted, pinned Exp16 native checkpoint."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from fhp_evaluation.best_response.adapters import PolicyAdapter, load_target
from fhp_evaluation.best_response.validation import FixedPolicy
from fhp_evaluation.game import load_fhp_game
from fhp_evaluation.loaders import sha256_file
from fhp_evaluation.safeguard.experiment import config
from fhp_evaluation.safeguard.matches import paired_matches
from fhp_evaluation.safeguard.runner import score_task
from fhp_evaluation.safeguard.validation import validate_native
from fhp_evaluation.strategic_audit.worker import export
from fhp_evaluation.strategic_audit.workflow import policy_identity
from fhp_evaluation.strategic_audit.tables import BoardContext, load_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    args = parser.parse_args()
    spec = config("smoke")
    expected = spec["checkpoint_registry"]["exp16_seed0_48h"]["sha256"]
    assert sha256_file(args.checkpoint) == expected
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    game = load_fhp_game()
    adapter = load_target(game, args.checkpoint, family="ucv", repo_root=args.native_root, batch_size=256)
    result = validate_native(game, adapter)
    rows, timing = paired_matches(game, adapter, PolicyAdapter(FixedPolicy(game,"raise")),
                                  dict(seed=0,opponent="raise",start=0,count=256),spec)
    assert np.isfinite(rows).all() and np.all(rows[:,2:4] >= 0)
    with tempfile.TemporaryDirectory(prefix="fhp-safeguard-responses-") as tmp:
        for method in ("lbr", "full_flop"):
            for arm in ("unmodified", "guarded"):
                name = f"{method}_{arm}"
                score_task(dict(seed=0,opponent=name,start=0,count=2,id=name),
                           str(args.checkpoint.parent),str(args.native_root),spec,tmp,"native-probe")
    board = (6,10,50)
    item = dict(family="ucv",checkpoint=str(args.checkpoint.resolve()), repo_root=str(args.native_root.resolve()),
                sha256=expected, deployment="exact_terminal_call_v1", require_suit_invariance=True)
    with tempfile.TemporaryDirectory(prefix="fhp-safeguard-native-") as tmp:
        export(dict(policy=item,boards=[board],cache=tmp,identity="native-probe",
                    expected_policy_identity=policy_identity(item)))
        context = BoardContext(game,board)
        tables = load_tables(Path(tmp)/"6-10-50/tables.npz",context)
        index = context.space.lookup[(2,14)]
        assert tables["flop_12221_1222"][index,0] == 0
        assert tables["flop_12221_1222"][index,1] == 1
    print(json.dumps(dict(native_passed=result["passed"],hidden_card_error=result["hidden_card_max_error"],
        gameplay_pairs=len(rows),response_cross_probes_passed=4,strategic_export_passed=True,
        verifier_maximum_seconds=timing["maximum_seconds"],board_misses=timing["board_misses"])))


if __name__ == "__main__":
    main()
