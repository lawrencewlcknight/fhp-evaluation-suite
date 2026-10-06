"""One native family per process. Successful board caches survive interrupted runs."""

import json
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ..best_response.adapters import PolicyAdapter, load_target
from ..best_response.runner import write_json
from ..best_response.validation import FixedPolicy
from ..game import load_fhp_game
from ..rule_agents import published_rule_agents
from .checks import check_adapter
from .library import board_key
from .tables import BoardContext, build_tables, accelerate_rule
from .workflow import analyse, completed, mark, policy_identity


def export(spec):
    game = load_fhp_game()
    item = spec["policy"]
    if policy_identity(item) != spec["expected_policy_identity"]:
        raise ValueError("Policy changed after the audit was frozen")
    if item["family"] == "rule":
        adapter = PolicyAdapter(published_rule_agents(game)[item["name"]])
    elif item["family"] == "control":
        adapter = PolicyAdapter(FixedPolicy(game, item["name"]))
    else:
        adapter = load_target(game, item["checkpoint"], family=item["family"], repo_root=item.get("repo_root"),
                              batch_size=item.get("batch_size", 1024), model_batch_size=item.get("model_batch_size", 32))
    for board in spec["boards"]:
        output = Path(spec["cache"]) / board_key(board)
        if completed(output, spec["identity"], "tables.npz"):
            continue
        started = time.perf_counter()
        context = BoardContext(game, board)
        board_adapter = accelerate_rule(context, adapter) if item["family"] == "rule" else adapter
        checks = check_adapter(context, board_adapter, require_suit_invariance=item.get("require_suit_invariance", False))
        tables = build_tables(context, board_adapter)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "tables.tmp.npz"
        np.savez_compressed(temporary, **tables)
        temporary.replace(output / "tables.npz")
        write_json(output / "diagnostics.json", dict(checks=checks, adapter=board_adapter.metadata,
                    stats=board_adapter.stats,
                    stats_scope="this_board" if item["family"] == "rule" else "cumulative_current_attempt",
                    elapsed_seconds=time.perf_counter() - started))
        mark(output, spec["identity"], "tables.npz")
        print(json.dumps(dict(board=board, status="succeeded")), flush=True)
    if policy_identity(item) != spec["expected_policy_identity"]:
        raise ValueError("Policy changed during table export; do not use these results")
    return dict(status="succeeded", boards=len(spec["boards"]),
                inference_diagnostics=str(Path(spec["cache"]).resolve()))


def main():
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise RuntimeError("Pinned OpenSpiel 1.6.3 required")
    path = Path(sys.argv[1])
    spec = json.loads(path.read_text())
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    result = export(spec) if spec["operation"] == "tables" else analyse(spec["root"], spec["identity"])
    write_json(path.parent / "result.json", result)


if __name__ == "__main__":
    main()
