"""Pair uncertainty and (separately) the three independent training-seed units."""
import numpy as np
from scipy.stats import beta, t

from ..statistics import sample_summary
from .matches import COLUMNS


def describe(rows):
    rows = np.asarray(rows)
    n = len(rows)
    if rows.shape != (n, len(COLUMNS)) or n < 2 or not np.isfinite(rows).all():
        raise ValueError("Expected at least two finite duplicate pairs")
    triggered = int(np.count_nonzero(rows[:, 4]))
    return dict(pairs=n, hands=2*n,
                scores={name: sample_summary(rows[:, i]) for i, name in enumerate(COLUMNS[:4])},
                activated_hands=int(rows[:, 4].sum()), eligible_hands=int(rows[:, 5].sum()),
                activated_pairs=triggered, mean_transferred_fold_probability_per_hand=float(rows[:, 6].sum()/(2*n)),
                conditional_mean_integrated_gain_bb_per_activated_hand=
                    float(2*rows[:, 3].sum()/rows[:, 4].sum()/1000) if triggered else None,
                eligibility_scope="Terminal call/fold decisions with nonzero original fold probability",
                hero_decisions=int(rows[:, 7].sum()),
                pair_activation_probability_95_upper=float(beta.ppf(.95, triggered+1, n-triggered))
                if triggered < n else 1.,
                zero_event_warning=("No activation observed: zero empirical gain/SE is NOT proof of no population benefit. "
                                    "Use the one-sided exact pair-activation probability bound.") if not triggered else None)


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) == 1:
        return dict(n=1, mean=mean, seed_means=values.tolist(), se=None, ci95=None)
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    margin = float(t.ppf(.975, len(values)-1)*se)
    return dict(n=len(values), mean=mean, seed_means=values.tolist(), se=se,
                ci95=[mean-margin, mean+margin],
                inference="Descriptive training-seed t interval; only three seeds; not a large confirmatory trial")


def aggregate(records, spec):
    by_seed = {}
    for seed in spec["seeds"]:
        opponent = f"exp16_seed{seed}_48h"
        by_seed[str(seed)] = records[f"{seed}/{opponent}"]
    primary = {metric: seed_summary([r["scores"][metric]["mean"] for r in by_seed.values()])
               for metric in COLUMNS[:4]}
    panels = {}
    from .experiment import RULES
    for name, labels in (
            ("rules", lambda s: list(RULES)),
            ("frozen_exp9", lambda s: [f"exp9_seed{v}_24h" for v in spec["seeds"]]),
            ("own_predecessors", lambda s: [f"exp16_seed{s}_{h}h" for h in (24,36,42)]),
            *((f"{kind}_{arm}", lambda s, k=kind, a=arm: [f"{k}_{a}"])
              for kind in ("lbr", "full_flop") for arm in ("unmodified", "guarded"))):
        panels[name] = {metric: seed_summary([
            np.mean([records[f"{seed}/{opponent}"]["scores"][metric]["mean"] for opponent in labels(seed)])
            for seed in spec["seeds"]]) for metric in COLUMNS[:4]}
    return dict(primary=primary, primary_by_seed=by_seed, comparisons=records, panels=panels,
                units="milli-big-blinds per hand", neural_weights_changed=False,
                note="Integrated delta averages over the final action randomness on naturally sampled paths. "
                     "It estimates the same fixed-opponent gain as sampled delta; it is not a rare-position audit score.",
                response_note="Four probes (LBR/full-flop, each targeting both arms) are cross-played against both policies. "
                              "These are approximate-response payoff estimates, not exact exploitability.")


def write_report(path, result):
    lines = ["# Experiment 34: exact terminal-call safeguard", "",
             "Frozen Experiment 16 48h policies; no training or weight changes.", "",
             "| Seed | Guarded vs original (mbb/h) | Integrated gain (mbb/h) | Activated hands / hands |",
             "|---|---:|---:|---:|"]
    for seed, row in result["primary_by_seed"].items():
        lines.append(f"| {seed} | {row['scores']['guarded_mbb']['mean']:.6g} | "
                     f"{row['scores']['integrated_delta_mbb']['mean']:.6g} | {row['activated_hands']} / {row['hands']} |")
    lines += ["", "## Evaluation panels", "",
              "Equal opponent weight within each training seed; means below average those seed summaries.", "",
              "| Panel | Original (mbb/h) | Guarded (mbb/h) | Integrated improvement (mbb/h) |",
              "|---|---:|---:|---:|"]
    for label, panel in result["panels"].items():
        lines.append(f"| {label} | {panel['baseline_mbb']['mean']:.6g} | "
                     f"{panel['guarded_mbb']['mean']:.6g} | {panel['integrated_delta_mbb']['mean']:.6g} |")
    lines += ["", result["note"], "", result["response_note"], "",
              "The structural no-harm statement applies to expected payoff against any fixed opponent, "
              "conditional on the verifier and terminal-only implementation being correct. It is not a claim "
              "that the neural policy learned these decisions. Rare-event zero samples are explicitly flagged.", "",
              "See summary.json for pair-level intervals and separate training-seed summaries, strategic/ for "
              "the full position audit, validation.json for independent verifier checks, and task timing records "
              "for cold/cached latency. No adaptive early stopping on significance or favourable outcomes.", ""]
    path.write_text("\n".join(lines))
