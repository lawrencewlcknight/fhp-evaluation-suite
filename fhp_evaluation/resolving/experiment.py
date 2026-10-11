"""UCV Experiment 23: prespecified paired flop re-solving gameplay study.

Configuration stays stdlib-only for local preparation without training packages.
No fitting, new training trajectories, LBR or cross-algorithm comparisons.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

PROTOCOL = "fhp_exp23_flop_resolving_v1"
# Load the historical registry without importing the evaluation package (whose
# public __init__ imports NumPy/OpenSpiel). Laptop Batch preparation is stdlib-only.
_module_spec = importlib.util.spec_from_file_location(
    "exp23_historical_registry", Path(__file__).resolve().parents[1] / "strategic_audit/exp16.py")
_historical = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_historical)
previous_registry, RULES = _historical.checkpoint_registry, _historical.RULES

EXP8_HASHES = (
    "5d432c54978f563e59157bc7aa325bd9d770545e65399afa546a3711bf971fc1",
    "9e66cd726ac1eee8ecf06dc10f2a1f88595bf0e3a33530207ba1a459af2c6d60",
    "8f7696e829ed8d84a5f107a8df519b70841b9c4fb7e64bc3f7fed160dd08de92")


def checkpoint_registry():
    result = {k: v for k, v in previous_registry("development").items()
              if k.startswith("exp9_") or k.endswith("48h")}
    for seed, sha in enumerate(EXP8_HASHES):
        name = f"parallel_structured_ucv_escher_48h_seed_{seed}"
        result[f"exp8_seed{seed}_48h"] = dict(
            run="exp8-par48-20261001-005208",
            relative=f"workers/task_{seed:03d}_{name}/checkpoints/{name}_time_48h.pkl",
            sha256=sha, training_seed=seed, hours=48, cohort="exp8")
    return result


def config():
    return dict(protocol=PROTOCOL, experiment=23, training_seeds=[0, 1, 2],
                search_seconds=.5, max_iterations=10000, pairs_per_match=5000,
                pairs_per_shard=250, processes=4, torch_threads=4, batch_size=1024,
                evaluation_seed=20261011, preflight_seed=20261012,
                preflight_positions_per_seed=24, minimum_acceptance_rate=.10,
                maximum_overrun_rate=.05, maximum_latency_seconds=1.,
                policy_shift_threshold=1e-6, certificate_tolerance_bb=1e-8,
                primary="search_vs_own_blueprint_mbb_per_hand",
                secondary="equal_weight_external_panel_paired_uplift_mbb_per_hand",
                rules=list(RULES), learned_opponents=["exp8_48h_same_label", "exp9_24h_same_label"],
                checkpoint_registry=checkpoint_registry(),
                warning="Local numerical BR certificates are not whole-game exploitability measurements.")


def tasks(spec=None):
    spec = config() if spec is None else spec
    rows = []
    for seed in spec["training_seeds"]:
        for opponent in ["blueprint", *spec["rules"], "exp8", "exp9"]:
            for start in range(0, spec["pairs_per_match"], spec["pairs_per_shard"]):
                rows.append(dict(id=f"seed{seed}-{opponent}-{start:05d}", seed=seed,
                                 opponent=opponent, start=start,
                                 count=min(spec["pairs_per_shard"], spec["pairs_per_match"] - start)))
    return rows


def stream_seed(spec, seed, opponent):
    # Task scheduling, sharding, search RNG and resume cannot change the deals.
    label = f"{PROTOCOL}:{spec['evaluation_seed']}:{seed}:{opponent}"
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def source_hash(root):
    entries = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(root).rglob("*.py"))}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
