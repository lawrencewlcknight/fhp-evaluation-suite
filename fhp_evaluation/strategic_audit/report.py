"""Small additive summaries, selected cases and dependency-free SVG/HTML reporting."""

from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np

from ..best_response.runner import write_json
from ..cards import card_name
from .metrics import hand_tags
from .tables import key


class Accumulator:
    def __init__(self):
        self.groups = {}
        self.cases = []

    def add(self, name, gap, response, probs, *, weights=None):
        gap, response, probs = np.asarray(gap), np.asarray(response), np.asarray(probs)
        valid = np.isfinite(gap)
        w = np.ones(len(gap)) if weights is None else np.asarray(weights)
        valid &= w > 0
        row = self.groups.setdefault(name, dict(count=0, unsupported=0, weight=0., local_sum=0.,
                    response_sum=0., fold_sum=0., call_sum=0., raise_sum=0., max_local=0.))
        row["unsupported"] += int(np.sum(~np.isfinite(gap)))
        row["count"] += int(valid.sum())
        if not valid.any():
            return
        w = w[valid]
        row["weight"] += float(w.sum())
        row["local_sum"] += float(np.dot(w, gap[valid]))
        row["response_sum"] += float(np.dot(w, response[valid]))
        for a, label in enumerate(("fold", "call", "raise")):
            row[label + "_sum"] += float(np.dot(w, probs[valid, a]))
        row["max_local"] = max(row["max_local"], float(gap[valid].max()))

    def merge(self, groups):
        for name, row in groups.items():
            if name not in self.groups:
                self.groups[name] = dict(row)
            else:
                for k, v in row.items():
                    self.groups[name][k] = max(self.groups[name][k], v) if k == "max_local" else self.groups[name][k] + v

    def summaries(self):
        result = {}
        for name, row in self.groups.items():
            w = row["weight"]
            result[name] = dict(count=row["count"], unsupported=row["unsupported"],
                               mean_local_gap_bb=row["local_sum"] / w if w else None,
                               mean_response_gap_bb=row["response_sum"] / w if w else None,
                               max_local_gap_bb=row["max_local"] if w else None,
                               probabilities=[row[k + "_sum"] / w if w else None
                                              for k in ("fold", "call", "raise")])
        return result


def summarise_decisions(context, pre, result, accumulator, *, top=20):
    tags = hand_tags(context)
    if not hasattr(context, "_tag_indices"):
        indices = defaultdict(list)
        for i, labels in enumerate(tags):
            for label in labels:
                indices[label].append(i)
        context._tag_indices = {name: np.asarray(rows, dtype=int) for name, rows in indices.items()}
    all_rows = np.arange(len(tags))
    for decision in result["decisions"]:
        d = decision
        names = dict(context._tag_indices)
        for label in ("all", f"seat_{d.player}", f"pot_{d.pot_bb:g}bb",
                      f"history_{''.join(map(str, pre))}/{''.join(map(str, d.path))}",
                      f"raises_remaining_{d.raises_remaining}"):
            names[label] = all_rows
        equity_bins = np.minimum(np.floor(d.equity * 10), 9)
        for value in np.unique(equity_bins[np.isfinite(equity_bins)]):
            names[f"range_equity_decile_{int(value)}"] = np.flatnonzero(equity_bins == value)
        if d.threshold is not None:
            # Half-percent bins, no near-indifferent hard-label penalty.
            bins = np.floor((d.equity - d.threshold) / .005) * .005
            for value in np.unique(bins[np.isfinite(bins)]):
                names[f"odds_margin_{value:.3f}"] = np.flatnonzero(bins == value)
            names["proof_unbeatable_final_call"] = np.flatnonzero(d.global_unbeatable & (d.opponent_mass > 0))
            names["range_certain_loss_final_call"] = np.flatnonzero(d.certain_loss)
        for name, indices in names.items():
            accumulator.add(name, d.local_gap_bb[indices], d.response_gap_bb[indices], d.probabilities[indices])
        valid = np.flatnonzero(np.isfinite(d.local_gap_bb))
        for i in sorted(valid, key=lambda j: (-d.local_gap_bb[j], j))[:top]:
            q = [float(v) if np.isfinite(v) else None for v in d.q_bb[i]]
            accumulator.cases.append(dict(board=list(context.space.board), hand=list(map(int, context.space.hands[i])),
                preflop=list(pre), flop=list(d.path), seat=d.player, local_gap_bb=float(d.local_gap_bb[i]),
                response_gap_bb=float(d.response_gap_bb[i]), probabilities=d.probabilities[i].tolist(),
                action_values_bb=q, equity=float(d.equity[i]), pot_bb=d.pot_bb,
                call_cost_bb=d.call_cost_bb, pot_odds_threshold=d.threshold, tags=tags[i]))
        accumulator.cases.sort(key=lambda c: (-c["local_gap_bb"], c["board"], c["hand"], c["preflop"], c["flop"]))
        del accumulator.cases[top:]


def svg_bars(rows, title):
    height = 60 + 32 * len(rows)
    maximum = max([float(v or 0) for _, v in rows] + [1e-12])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="12" y="24" font-size="16">{html.escape(title)}</text>']
    for i, (label, value) in enumerate(rows):
        y = 50 + i * 32
        length = 380 * (value or 0) / maximum
        parts.extend([f'<text x="12" y="{y+15}" font-size="12">{html.escape(label[:65])}</text>',
                      f'<rect x="450" y="{y}" width="{length:.3f}" height="20" fill="#3578ac"/>',
                      f'<text x="{455+length:.3f}" y="{y+15}" font-size="12">{value:.5f}</text>' if value is not None else ""])
    return "\n".join(parts + ["</svg>"])


def svg_curve(points, title, xlabel, ylabel):
    """Small standalone scientific plot, no optional plotting dependency."""
    points = [(float(x), float(y)) for x, y in points if y is not None]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="290" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="22" font-size="14">{html.escape(title)}</text>',
             '<path d="M60 40 V240 H670" fill="none" stroke="black"/>',
             f'<text x="220" y="285" font-size="12">{html.escape(xlabel)}</text>',
             f'<text x="65" y="38" font-size="12">{html.escape(ylabel)}</text>']
    if points:
        points.sort()
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(0., min(y for _, y in points)), max(y for _, y in points)
        xmax = max(xmax, xmin + 1e-9)
        ymax = max(ymax, ymin + 1e-9)
        xy = [(60 + 600 * (x-xmin)/(xmax-xmin), 240 - 190*(y-ymin)/(ymax-ymin)) for x, y in points]
        parts.append('<polyline fill="none" stroke="#3578ac" points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in xy) + '"/>')
        for x, y in xy:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#3578ac"/>')
        parts += [f'<text x="60" y="260" font-size="11">{xmin:.4g}</text>',
                  f'<text x="630" y="260" font-size="11">{xmax:.4g}</text>',
                  f'<text x="5" y="55" font-size="11">{ymax:.4g}</text>',
                  f'<text x="5" y="240" font-size="11">{ymin:.4g}</text>']
    return "\n".join(parts + ["</svg>"])


def write_report(directory, result):
    directory = Path(directory)
    write_json(directory / "summary.json", result)
    rows = [(name, r["summary"].get("all", {}).get("mean_local_gap_bb")) for name, r in result["comparisons"].items()]
    (directory / "decision_gaps.svg").write_text(svg_bars(rows, "Mean local decision gap (BB per audited decision)"))
    chunks = ["<!doctype html><meta charset='utf-8'><title>FHP strategic audit</title>",
              "<style>body{font:16px system-ui;max-width:1100px;margin:35px auto}td,th{padding:6px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}pre{white-space:pre-wrap}svg{max-width:100%}</style>",
              "<h1>FHP strategic-position audit</h1>",
              "<p>Conditional diagnostics, not whole-game exploitability or a universal optimal-action test. "
              "Local gaps hold the candidate continuation fixed. Remaining-round gaps optimise its continuation. "
              "Do not sum gaps along a trajectory. Unsupported ranges are excluded and counted. "
              "The balanced view weights feasible information-set observations equally; it is not a visitation distribution. "
              "Synthetic tilts are stress tests. Exact values have no deal-level sampling interval.</p>",
              (directory / "decision_gaps.svg").read_text()]
    for name, r in result["comparisons"].items():
        chunks.append(f"<h2>{html.escape(name)}</h2><table><tr><th>Category</th><th>N</th><th>Unsupported</th><th>Local gap BB</th><th>Remaining gap BB</th><th>Fold / call / raise</th></tr>")
        for label, row in sorted(r["summary"].items()):
            chunks.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                label, row["count"], row["unsupported"], row["mean_local_gap_bb"], row["mean_response_gap_bb"],
                row["probabilities"])) + "</tr>")
        chunks.append("</table><h3>Largest local gaps (deterministic selection)</h3>")
        odds = [(float(label.removeprefix("odds_margin_")), row["probabilities"][1])
                for label, row in r["summary"].items() if label.startswith("odds_margin_")]
        chart = svg_curve(odds, name + ": terminal calls", "Equity minus pot-odds threshold (0.005 bins)", "Call probability")
        (directory / f"{name}_pot_odds.svg").write_text(chart)
        chunks.extend([chart, "<p>These are descriptive binned averages, not independent samples. "
                       "Root remaining-round gap: " + html.escape(str(r["mean_root_response_gap_bb"])) + " BB.</p>",
                       "<h3>Frozen reference-distribution view</h3><pre>" + html.escape(json.dumps(r["reference_summary"], indent=2)) + "</pre>"])
        equity_profile = [(float(label.removeprefix("range_equity_decile_")) / 10 + .05, row["probabilities"][2])
                          for label, row in r["summary"].items() if label.startswith("range_equity_decile_")]
        chart = svg_curve(equity_profile, name + ": aggression by equity", "Range-conditioned equity bin midpoint", "Bet/raise probability")
        (directory / f"{name}_aggression.svg").write_text(chart)
        chunks.append(chart)
        for case in r["cases"]:
            case = dict(case, cards=" ".join(card_name(c) for c in case["hand"]),
                        board_cards=" ".join(card_name(c) for c in case["board"]))
            chunks.append("<pre>" + html.escape(json.dumps(case, indent=2)) + "</pre>")
    chunks.append("<h2>Training-seed summaries</h2><pre>" + html.escape(json.dumps(result["cohort_summaries"], indent=2)) + "</pre>")
    for name, points in result.get("training_trajectories", {}).items():
        chart = svg_curve(points, name, "Active training hours", "Mean local gap (BB/decision)")
        (directory / f"{name}_training.svg").write_text(chart)
        chunks.append(chart)
    chunks.append("<h2>Native correctness and symmetry diagnostics</h2><pre>" + html.escape(json.dumps(result["implementation_checks"], indent=2)) + "</pre>")
    chunks.append("<h2>Provenance and coverage</h2><pre>" + html.escape(json.dumps(result["provenance"], indent=2)) + "</pre>")
    (directory / "report.html").write_text("\n".join(chunks))
