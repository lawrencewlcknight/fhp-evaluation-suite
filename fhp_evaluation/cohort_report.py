"""Seed-level tables and figures for frozen-policy cohort comparisons."""
import csv
from pathlib import Path

import numpy as np

from .cohort import aggregate, collapse, paired_differences, write_json


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(results, output):
    output = Path(output)
    rows = collapse(results)
    means = aggregate(rows)
    differences = paired_differences(rows)
    for name, data in (("by_seed", rows), ("aggregate", means),
                       ("paired_differences_by_seed", differences),
                       ("paired_differences_aggregate", aggregate(differences))):
        write_json(output / f"{name}.json", data)
        write_csv(output / f"{name}.csv", data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in ("rule", "lbr", "direct"):
        opponents = sorted({r["opponent"] for r in means if r["kind"] == kind})
        if not opponents:
            continue
        for axis in ("hours", "nodes_mean"):
            fig, axes = plt.subplots(1, len(opponents), figsize=(5 * len(opponents), 4), squeeze=False)
            for ax, opponent in zip(axes[0], opponents):
                for experiment in ("exp1", "exp2", "exp3"):
                    data = sorted([r for r in means if r["kind"] == kind and r["opponent"] == opponent
                                   and r["experiment"] == experiment], key=lambda r: r[axis])
                    if not data:
                        continue
                    y = np.array([r["seed_mean"] for r in data])
                    error = np.array([[v-r["seed_ci95_low"], r["seed_ci95_high"]-v]
                                      for v, r in zip(y, data)]).T
                    error = np.where(np.isfinite(error), error, 0)
                    ax.errorbar([r[axis] for r in data], y, yerr=error, marker="o", label=experiment)
                ax.axhline(0, color="grey", linewidth=.6)
                ax.set(title=f"{kind}: {opponent}", xlabel="Active training hours" if axis == "hours"
                       else "Mean training nodes (not matched)", ylabel="mbb/hand")
                ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{kind}_by_{axis}.png", dpi=140)
            plt.close(fig)
    direct = [r for r in means if r["kind"] == "direct" and r["hours"] == 24]
    if direct:
        matrix = np.zeros((3, 3))
        for row in direct:
            i, j = int(row["experiment"][-1])-1, int(row["opponent"][-1])-1
            matrix[i, j], matrix[j, i] = row["seed_mean"], -row["seed_mean"]
        fig, ax = plt.subplots(figsize=(5, 4))
        limit = max(1, abs(matrix).max())
        im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                        color="white" if abs(matrix[i,j]) > .5 * limit else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["Exp1", "Exp2", "Exp3"],
               yticklabels=["Exp1", "Exp2", "Exp3"], title="24h: row payoff versus column")
        fig.colorbar(im, ax=ax, label="mbb/hand")
        fig.tight_layout()
        fig.savefig(output / "final_head_to_head.png", dpi=140)
        plt.close(fig)
    temporal = [r for r in means if r["kind"] == "temporal"]
    if temporal:
        hours = [6, 12, 18, 24]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        limit = max(1, max(abs(r["seed_mean"]) for r in temporal))
        for e, ax in enumerate(axes, 1):
            matrix = np.full((4, 4), np.nan)
            for r in temporal:
                if r["experiment"] == f"exp{e}":
                    matrix[hours.index(r["hours"]), hours.index(r["earlier_hours"])] = r["seed_mean"]
            ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
            for i, j in zip(*np.where(np.isfinite(matrix))):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                        color="white" if abs(matrix[i,j]) > .5 * limit else "black")
            ax.set(xticks=range(4), yticks=range(4), xticklabels=hours, yticklabels=hours,
                   title=f"Exp{e}: later payoff (mbb/hand)", xlabel="Earlier hours", ylabel="Later hours")
        fig.tight_layout()
        fig.savefig(output / "temporal_head_to_head.png", dpi=140)
        plt.close(fig)
    (output / "analysis_summary.md").write_text(
        "# Frozen-policy comparison\n\n"
        "Primary comparisons: 24h Exp2 vs Exp1 and Exp3 vs Exp2; Exp3 vs Exp1 is secondary.\n\n"
        "All values are mbb/hand. Higher rule/direct/temporal payoffs are better. "
        "LBR reports the responder's payoff, so lower is better. Paired differences are newer minus older.\n\n"
        "Tables separate within-match duplicate-pair uncertainty from across-training-seed "
        "Student-t intervals (three seeds). Intervals are exploratory, not multiplicity-adjusted. "
        "Matched seed labels are not identical training trajectories. Equal active hours are not equal "
        "compute cost. Node plots are descriptive, not matched-node experiments. "
        "LBR is a sampled lower-bound diagnostic, not exact exploitability; "
        "head-to-head superiority does not certify Nash convergence.\n")
    return dict(matches=len(rows), units="mbb/hand", training_seeds=sorted({r["seed"] for r in rows}))
