"""Immutable inputs, isolated native inference and resumable board-sized analysis."""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from ..best_response.runner import identity, run_bounded, source_hash, write_json
from ..game import load_fhp_game
from ..loaders import sha256_file
from ..rule_agents import PUBLISHED_AGENT_NAMES
from . import PROTOCOL
from .engine import audit_entry
from .library import board_key, board_schedule, texture, validate_library
from .report import Accumulator, summarise_decisions, write_report
from .tables import BoardContext, key, load_tables


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_spec(spec):
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported audit protocol")
    policies = spec["policies"]
    if not policies or not spec.get("candidates") or not spec.get("opponents"):
        raise ValueError("Policies, candidates and opponents are required")
    for label, item in policies.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Policy ids must be filesystem-safe")
        family = item["family"]
        if family not in ("ucv", "vr_deep", "sd_cfr", "raw_mlp", "control", "rule"):
            raise ValueError("Unknown policy family")
        if family == "rule" and item.get("name") not in PUBLISHED_AGENT_NAMES:
            raise ValueError("Unknown published rule agent")
        if family == "control" and item.get("name") not in ("uniform", "random", "fold", "call", "raise"):
            raise ValueError("Unknown control policy")
        if family not in ("rule", "control"):
            if not Path(item["checkpoint"]).is_absolute() or not Path(item["checkpoint"]).is_file():
                raise ValueError("Provide an existing absolute checkpoint path")
            if sha256_file(item["checkpoint"]) != item["sha256"]:
                raise ValueError("Checkpoint does not match its pinned sha256")
            if family != "raw_mlp" and (not Path(item["repo_root"]).is_absolute()
                                        or not Path(item["repo_root"]).is_dir()):
                raise ValueError("Existing absolute native repo_root required")
        if label in spec["candidates"] and not all(k in item for k in ("cohort", "training_seed", "hours")):
            raise ValueError("Candidate cohort, training_seed and hours are required (null allowed for controls)")
        if label in spec["candidates"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item["cohort"]):
                raise ValueError("Cohort ids must be filesystem-safe")
            if type(item["training_seed"]) is not int or item["training_seed"] < 0:
                raise ValueError("Nonnegative integer training_seed required")
            hours = item["hours"]
            if hours is not None and (not isinstance(hours, (int, float)) or not np.isfinite(hours) or hours < 0):
                raise ValueError("Finite nonnegative hours or null required")
    for group in (spec["candidates"], spec["opponents"]):
        if len(group) != len(set(group)) or not set(group) <= set(policies):
            raise ValueError("Unknown or duplicate policy in panel")
    tilts = spec.get("range_tilts", [0, -2, 2])
    if 0 not in tilts or len(tilts) != len(set(tilts)) or any(not np.isfinite(t) or abs(t) > 10 for t in tilts):
        raise ValueError("Distinct finite range tilts in [-10,10], including 0, required")
    if len({f"{t:g}" for t in tilts}) != len(tilts):
        raise ValueError("Range tilt labels would collide")
    if spec.get("reference_policy") is not None and spec["reference_policy"] not in policies:
        raise ValueError("Unknown frozen reference policy")
    # No repeated seed treated as an independent training run at a fixed budget.
    units = [(policies[c]["cohort"], policies[c]["training_seed"], policies[c]["hours"])
             for c in spec["candidates"]]
    if len(units) != len(set(units)):
        raise ValueError("Duplicate cohort/seed/time inferential units")


def policy_identity(item):
    if item["family"] in ("rule", "control"):
        return dict(item)
    value = identity(dict(family=item["family"], checkpoint=item["checkpoint"], repo_root=item.get("repo_root")))
    # Include all native Python packages, not just the adapter entry point.
    if item.get("repo_root"):
        root = Path(item["repo_root"])
        value["native_packages"] = {p.name: source_hash(p) for p in sorted(root.iterdir())
                                     if p.is_dir() and (p / "__init__.py").is_file()}
    if item["family"] == "sd_cfr":
        manifest = json.loads(Path(item["checkpoint"]).read_text())
        value["chunk_hashes"] = {c["path"]: sha256_file(Path(item["checkpoint"]).parent / c["path"])
                                  for c in manifest["chunks"]}
    return dict(spec=item, native=value)


def completed(path, fingerprint, filename):
    path = Path(path)
    if not (path / "SUCCESS.json").exists():
        return False
    marker = json.loads((path / "SUCCESS.json").read_text())
    if marker["identity"] != fingerprint or marker["sha256"] != sha256_file(path / filename):
        raise ValueError(f"Cached artifact identity/checksum mismatch: {path}")
    if filename == "tables.npz" and marker.get("diagnostics_sha256") != sha256_file(path / "diagnostics.json"):
        raise ValueError(f"Cached diagnostics checksum mismatch: {path}")
    return True


def mark(path, fingerprint, filename):
    marker = dict(identity=fingerprint, sha256=sha256_file(Path(path) / filename))
    if filename == "tables.npz":
        marker["diagnostics_sha256"] = sha256_file(Path(path) / "diagnostics.json")
    write_json(Path(path) / "SUCCESS.json", marker)


@contextmanager
def exclusive(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another audit process is using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(spec, library, directory, *, partition="development", include_reference=False,
        unlock_assessment=False, resume=False, threads=2, max_seconds=7200, max_rss_mb=8192):
    if importlib.metadata.version("open_spiel") != "1.6.3":
        raise RuntimeError("Pinned OpenSpiel 1.6.3 required for production audit runs")
    validate_spec(spec)
    validate_library(library)
    if partition == "assessment" and not unlock_assessment:
        raise ValueError("Assessment access requires --unlock-assessment; freeze model selection first")
    if include_reference and not spec.get("reference_policy"):
        raise ValueError("Freeze a reference_policy before sampling representative positions")
    if threads < 1 or max_seconds <= 0 or max_rss_mb <= 0:
        raise ValueError("Positive resource settings required")
    boards = board_schedule(library, partition, include_reference)
    if not boards:
        raise ValueError("No boards in selected partition")
    policies = spec["policies"]
    used = set(spec["candidates"]) | set(spec["opponents"])
    if include_reference:
        used.add(spec["reference_policy"])
    provenance = dict(protocol=PROTOCOL, spec=spec, library=library, partition=partition,
                      include_reference=include_reference, threads=threads,
                      evaluator_source_sha256=source_hash(Path(__file__).resolve().parents[1]),
                      packages={p: importlib.metadata.version(p) for p in ("numpy", "torch", "open_spiel", "scipy")},
                      policies={p: policy_identity(policies[p]) for p in sorted(used)})
    fingerprint = digest(provenance)
    directory = Path(directory).resolve()
    with exclusive(directory):
        manifest = directory / "manifest.json"
        if manifest.exists():
            if not resume or json.loads(manifest.read_text()) != provenance:
                raise ValueError("Existing audit requires --resume with identical inputs, code and runtime")
        else:
            if resume:
                raise ValueError("Cannot resume an audit without a manifest")
            write_json(manifest, provenance)
        write_json(directory / "status.json", dict(status="running", identity=fingerprint))
        for label in sorted(used):
            cache = directory / "tables" / label
            if all(completed(cache / board_key(b), fingerprint, "tables.npz") for b in boards):
                continue
            attempts = directory / "workers" / label
            attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):03d}"
            worker_spec = dict(operation="tables", policy=policies[label], boards=boards,
                               cache=str(cache), identity=fingerprint, threads=threads,
                               expected_policy_identity=provenance["policies"][label])
            result = run_bounded(worker_spec, attempt, max_seconds=max_seconds, max_rss_mb=max_rss_mb,
                                 command=[sys.executable, "-m", "fhp_evaluation.strategic_audit.worker", str(attempt / "spec.json")])
            if result["status"] != "succeeded":
                write_json(directory / "status.json", dict(status=result["status"], worker=str(attempt)))
                raise RuntimeError(f"Policy export failed; see {attempt / 'stderr.log'}. Successful boards can be resumed.")
        # Numerical analysis is also monitored, with board-level durable progress.
        attempts = directory / "workers" / "analysis"
        attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):03d}"
        result = run_bounded(dict(operation="analysis", root=str(directory), identity=fingerprint, threads=threads),
                             attempt, max_seconds=max_seconds, max_rss_mb=max_rss_mb,
                             command=[sys.executable, "-m", "fhp_evaluation.strategic_audit.worker", str(attempt / "spec.json")])
        if result["status"] != "succeeded":
            write_json(directory / "status.json", dict(status=result["status"], worker=str(attempt)))
            raise RuntimeError(f"Analysis failed; see {attempt / 'stderr.log'}")
        write_json(directory / "status.json", dict(status="succeeded", identity=fingerprint,
                                                   report=str(directory / "analysis" / "report.html")))
    return directory / "analysis" / "summary.json"


def sampled_positions(context, deal, reference, opponent):
    """Both seats on a shared deal; public histories generated only by frozen policies.

    Future board/private cards are sampled upfront but never passed to a player's
    information state preflop. Tables queried here were exported by those states.
    """
    cards = deal["cards"]
    for seat in (0, 1):
        # Swap the private hands with the focal seat (duplicate-deal convention).
        hands = [tuple(sorted(cards[:2])), tuple(sorted(cards[2:4]))]
        if seat == 1:
            hands.reverse()
        indices = [context.space.lookup[h] for h in hands]
        rng = np.random.default_rng(deal["action_seed"])
        pre = ()
        while pre in context.pre_nodes:
            player = context.pre_nodes[pre][0]
            probabilities = (reference if player == seat else opponent)[key(pre)][indices[player]]
            action = int(rng.choice(3, p=probabilities))
            pre += (action,)
        if pre not in context.trees:
            continue  # preflop fold: no final-round position
        nodes, index = context.trees[pre], 0
        while nodes[index].player >= 0:
            node = nodes[index]
            if node.player == seat:
                yield pre, seat, node.path, indices[seat]
            probabilities = (reference if node.player == seat else opponent)[key(pre, node.path)][indices[node.player]]
            action = int(rng.choice(3, p=probabilities))
            index = dict(node.children)[action]


def analyse(directory, fingerprint):
    directory = Path(directory)
    provenance = json.loads((directory / "manifest.json").read_text())
    if digest(provenance) != fingerprint:
        raise ValueError("Analysis identity mismatch")
    if source_hash(Path(__file__).resolve().parents[1]) != provenance["evaluator_source_sha256"]:
        raise ValueError("Evaluator source changed during the audit")
    spec, library = provenance["spec"], provenance["library"]
    partition = provenance["partition"]
    boards = board_schedule(library, partition, provenance["include_reference"])
    core_boards = {tuple(sorted(r["board"])) for r in library["boards"] if r["partition"] == partition}
    game = load_fhp_game()
    combined, cases, per_board = {}, defaultdict(list), defaultdict(list)
    for board in boards:
        shard = directory / "board_results" / board_key(board)
        if not completed(shard, fingerprint, "result.json"):
            started = time.perf_counter()
            context = BoardContext(game, board)
            tables = {}
            def table(label):
                # Bounded within this board; policies are only small float tables.
                if label not in tables:
                    path = directory / "tables" / label / board_key(board)
                    if not completed(path, fingerprint, "tables.npz"):
                        raise ValueError("Missing verified table")
                    tables[label] = load_tables(path / "tables.npz", context)
                return tables[label]
            reference_positions = {}
            if provenance["include_reference"]:
                for opp in spec["opponents"]:
                    reference_positions[opp] = [position for deal in library["reference_deals"]
                        if tuple(sorted(deal["cards"][4:])) == board
                        for position in sampled_positions(context, deal, table(spec["reference_policy"]), table(opp))]
            outputs = {}
            for candidate in spec["candidates"]:
                for opp in spec["opponents"]:
                    for tilt in spec.get("range_tilts", [0, -2, 2]):
                        if board not in core_boards and tilt != 0:
                            continue
                        acc, sampled = Accumulator(), Accumulator()
                        roots = []
                        for pre in context.entries:
                            for seat in (0, 1):
                                result = audit_entry(context, table(candidate), table(opp), pre, seat, range_tilt=tilt)
                                if board in core_boards:
                                    summarise_decisions(context, pre, result, acc)
                                    gaps = result["best_response_bb"] - result["value_bb"]
                                    roots.extend(gaps[np.isfinite(gaps)].tolist())
                                if tilt == 0:
                                    indexed = {d.path: d for d in result["decisions"]}
                                    for rpre, rseat, path, i in reference_positions.get(opp, []):
                                        if rpre == pre and rseat == seat:
                                            d = indexed[path]
                                            sampled.add("all", d.local_gap_bb[[i]], d.response_gap_bb[[i]], d.probabilities[[i]])
                        label = f"{candidate}__vs__{opp}__tilt_{tilt:g}"
                        outputs[label] = dict(groups=acc.groups, cases=acc.cases,
                            reference_groups=sampled.groups, candidate=candidate, opponent=opp, range_tilt=tilt,
                            root_gap_sum=float(sum(roots)), root_count=len(roots))
            shard.mkdir(parents=True, exist_ok=True)
            write_json(shard / "result.json", dict(board=list(board), texture=texture(board), comparisons=outputs,
                                                   elapsed_seconds=time.perf_counter() - started))
            mark(shard, fingerprint, "result.json")
            print(json.dumps(dict(board=list(board), status="succeeded")), flush=True)
        data = json.loads((shard / "result.json").read_text())
        for name, row in data["comparisons"].items():
            accum, sampled = combined.setdefault(name, (Accumulator(), Accumulator()))
            accum.merge(row["groups"])
            sampled.merge(row["reference_groups"])
            cases[name].extend(row["cases"])
            cases[name].sort(key=lambda c: (-c["local_gap_bb"], c["board"], c["hand"]))
            del cases[name][20:]
            per_board[name].append(dict(board=list(board), core=board in core_boards,
                local_sum=row["groups"].get("all", {}).get("local_sum", 0.),
                count=row["groups"].get("all", {}).get("weight", 0.),
                root_gap_sum=row["root_gap_sum"], root_count=row["root_count"]))
    comparisons = {}
    for name, (accum, sampled) in combined.items():
        count = sum(b["root_count"] for b in per_board[name])
        comparisons[name] = dict(summary=accum.summaries(), reference_summary=sampled.summaries(),
            cases=cases[name], boards=per_board[name],
            mean_root_response_gap_bb=sum(b["root_gap_sum"] for b in per_board[name]) / count if count else None)
    result = dict(protocol=PROTOCOL, status="succeeded", exact_full_game_exploitability=False,
                  comparisons=comparisons, provenance=provenance,
                  inferential_unit="training seed; do not treat information sets as independent training replicates")
    result["cohort_summaries"] = cohort_summaries(spec, comparisons)
    trajectories = defaultdict(list)
    for candidate in spec["candidates"]:
        item = spec["policies"][candidate]
        for opp in spec["opponents"]:
            name = f"{candidate}__vs__{opp}__tilt_0"
            value = comparisons[name]["summary"].get("all", {}).get("mean_local_gap_bb")
            if item["hours"] is not None and value is not None:
                trajectories[f"{item['cohort']}_seed{item['training_seed']}__vs__{opp}"].append([item["hours"], value])
    result["training_trajectories"] = dict(trajectories)
    result["implementation_checks"] = {label: [json.loads((directory / "tables" / label / board_key(b) / "diagnostics.json").read_text())["checks"]
                                               for b in boards] for label in provenance["policies"]}
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_report(output, result)
    return dict(status="succeeded", report=str(output / "report.html"), comparisons=len(comparisons))


def cohort_summaries(spec, comparisons):
    """Descriptive seed-level uncertainty, conditional on the frozen board/panel suite."""
    groups = defaultdict(list)
    for candidate in spec["candidates"]:
        item = spec["policies"][candidate]
        for opp in spec["opponents"]:
            for tilt in spec.get("range_tilts", [0, -2, 2]):
                name = f"{candidate}__vs__{opp}__tilt_{tilt:g}"
                value = comparisons[name]["summary"].get("all", {}).get("mean_local_gap_bb")
                if value is not None:
                    group = f"{item['cohort']}__{item['hours']}h__vs__{opp}__tilt_{tilt:g}"
                    groups[group].append(dict(seed=item["training_seed"], candidate=candidate, value=value))
    return {name: dict(n=len(rows), seed_values=rows, mean=float(np.mean([r["value"] for r in rows])),
                       standard_error=float(np.std([r["value"] for r in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
                       note="descriptive; small seed cohorts, selected positions, no multiplicity-adjusted claim")
            for name, rows in groups.items()}
