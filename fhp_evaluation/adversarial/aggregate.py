"""Predeclared paired contrasts, with seed and deal uncertainty kept separate."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..best_response.runner import write_json
from ..statistics import sample_summary
from .config import ARMS, configuration


def uncertainty(rows, smoke=False):
    matrix = np.asarray(rows, dtype=float)
    result = dict(mean_mbb=float(matrix.mean()), seeds=len(matrix), pairs_per_seed=matrix.shape[1],
                  seed_means_mbb=matrix.mean(1).tolist(),
                  paired_deal_ci_conditional_on_models=sample_summary(matrix.mean(0)))
    if len(matrix) > 1:
        result["seed_ci_conditional_on_shared_deals"] = sample_summary(matrix.mean(1))
    result["joint_training_and_sampling_ci"] = None
    result["smoke_execution_only"] = smoke
    return result


def aggregate(root, output, smoke=False):
    root, output = Path(root), Path(output)
    cfg = configuration(smoke)
    seeds = [0] if smoke else [0, 1, 2]
    records = {}
    bundle = None
    for seed in seeds:
        for arm_id, arm in enumerate(ARMS):
            row = json.loads((root / f"task_{seed * 4 + arm_id}" / "result.json").read_text())
            if (row["config_hash"] != cfg.fingerprint() or row["seed"] != seed
                    or row["arm"] != arm or row["phase"] != "evaluate"
                    or row["status"] != ("smoke_complete" if smoke else "complete")):
                raise ValueError("Incomplete or mixed evaluation records")
            if bundle is not None and bundle != row["bundle_sha256"]:
                raise ValueError("Mixed source bundles")
            bundle = row["bundle_sha256"]
            records[seed, arm] = row
    # Select the challenger with the greatest validation payoff for each policy;
    # report ONLY its independently scored test payoff. Never max test scores.
    responses, selected = {}, {}
    for arm in ARMS:
        values, names = [], []
        for seed in seeds:
            chosen = max(ARMS, key=lambda a: records[seed, a]["challenger_matrix"][arm]["selection_mean_mbb"])
            scores = records[seed, chosen]["challenger_matrix"][arm]["pair_payoffs_mbb"]
            if len(scores) != cfg.test_pairs:
                raise ValueError("Partial evaluation panel")
            values.append(scores)
            names.append(chosen)
        responses[arm], selected[arm] = np.asarray(values), names
    result = dict(experiment="exp21_fhp_adversarial_preservation", config=cfg.to_dict(),
                  bundle_sha256=bundle, status="smoke_complete" if smoke else "complete",
                  exact_exploitability=False, selected_challengers=selected, arms={}, contrasts={},
                  primary_contrast="preserved minus ordinary; negative response payoff difference is favourable",
                  warning="Approximate attack lower bounds cannot certify decreased true exploitability.")
    for arm in ARMS:
        result["arms"][arm] = dict(
            own_attacker_strength=[records[s, arm].get("attacker_strength") for s in seeds],
            learned_response=uncertainty(responses[arm], smoke),
            h2h_vs_unchanged=uncertainty([records[s, arm]["h2h_vs_unchanged"]["pair_payoffs_mbb"] for s in seeds], smoke),
            lbr=uncertainty([records[s, arm]["specialised_response"]["pair_payoffs_mbb"]["lbr"] for s in seeds], smoke),
            full_flop=uncertainty([records[s, arm]["specialised_response"]["pair_payoffs_mbb"]["full_flop"] for s in seeds], smoke),
            rule_panel={name: uncertainty([records[s, arm]["rule_panel"][name]["pair_payoffs_mbb"] for s in seeds], smoke)
                        for name in records[seeds[0], arm]["rule_panel"]},
            frozen_ucv_panel={name: uncertainty([records[s, arm]["frozen_ucv_panel"][name]["pair_payoffs_mbb"] for s in seeds], smoke)
                              for name in records[seeds[0], arm].get("frozen_ucv_panel", {})})
    for left, right in (("preserved", "ordinary"), ("preserved", "adversarial"), ("preserved", "unchanged")):
        result["contrasts"][f"{left}_minus_{right}"] = dict(
            learned_response=uncertainty(responses[left] - responses[right], smoke),
            full_flop=uncertainty([
                np.asarray(records[s, left]["specialised_response"]["pair_payoffs_mbb"]["full_flop"])
                - np.asarray(records[s, right]["specialised_response"]["pair_payoffs_mbb"]["full_flop"])
                for s in seeds], smoke))
    write_json(output / "summary.json", result)
    write_json(output / "SUCCESS.json", dict(status=result["status"], bundle_sha256=bundle))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    aggregate(args.input, args.output, args.smoke)


if __name__ == "__main__":
    main()
