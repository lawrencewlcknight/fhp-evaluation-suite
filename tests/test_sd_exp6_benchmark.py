import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fhp_evaluation.strategic_audit import sd_exp6_benchmark as b


def progress():
    boards = [[i // 40, 10 + i // 8, 25 + i % 8] for i in range(64)]
    waves = []
    for hour, count in ((24, 64), (48, 51)):
        for start in range(0, count, 4):
            tasks = [dict(policy=f"sd_exp6_seed0_{hour}h", board=boards[i], status="succeeded",
                          elapsed_seconds=(100 + i) * hour / 24) for i in range(start, min(start + 4, count))]
            waves.append(dict(tasks=tasks, elapsed_seconds=max(t["elapsed_seconds"] for t in tasks)))
    return dict(identity="test", config=dict(workers=4), waves=waves)


def test_plan_counts_quantiles_and_remaining_work():
    plan = b.make_plan(progress())
    b.validate_plan(plan)
    assert plan["saved_successful_exports"] == 115
    assert plan["forecast"]["remaining_sd_exports"] == 269
    assert plan["measured"]["observed_48_to_24_median_ratio"] == 2
    assert plan["forecast"]["downstream_hours"] is None
    assert plan["repeats"] == 2 and len(plan["boards"]) == 4
    assert b.simulate([10, 1, 1, 1], 2) == 10
    assert b.make_plan(progress()) == plan


def test_invalid_plan_and_timings_rejected():
    plan = b.make_plan(progress())
    with pytest.raises(ValueError):
        b.validate_plan(dict(plan, boards=[[0, 0, 1]] * 4))
    with pytest.raises(ValueError):
        b.validate_plan(dict(plan, allocations=[[16, 16]]))
    p = progress()
    p["waves"][0]["tasks"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        b.make_plan(p)


def test_only_complete_twice_and_parity_passed_can_win():
    cells = [dict(allocation=f"w{w}-t{t}", status="complete", elapsed_seconds=10 / w,
                  parity=dict(passed=True)) for w, t in b.ALLOCATIONS for _ in range(2)]
    out = b.summary(cells, {})
    assert out["groups"]["w8-t2"]["speedup_vs_rolling4x4"] == 2
    cells[0]["parity"]["passed"] = False
    assert b.summary(cells, {})["provisional_fastest"] is None
    assert not out["audit_restarted"] and not out["failed_board_resolved"]


def test_table_comparison_retains_strict_tolerance(tmp_path):
    for label, delta in (("ref", 0), ("ok", 1e-7), ("bad", 1e-4)):
        path = tmp_path / label / "tables/p/0-1-2"
        path.mkdir(parents=True)
        np.savez(path / "tables.npz", policy=np.array([[.5 + delta, .5 - delta]]))
    assert b.compare_tables(tmp_path / "ref", tmp_path / "ok", ["p"], [[0, 1, 2]])["passed"]
    assert not b.compare_tables(tmp_path / "ref", tmp_path / "bad", ["p"], [[0, 1, 2]])["passed"]


def test_batch_is_bounded_source_only_and_never_resumes_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("benchmark_batch", root / "gcp/sd_exp6_execution_benchmark_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    args = argparse.Namespace(stage="profile", bucket="gs://bucket", run_id="fhp-sd6-audit-bench-test",
                              service_account="runner@project.iam.gserviceaccount.com")
    job = batch.build_job(args, "a" * 64, "b" * 64)
    task = job["taskGroups"][0]["taskSpec"]
    assert task["maxRunDuration"] == "72000s" and task["maxRetryCount"] == 0
    script = task["runnables"][0]["script"]["text"]
    assert "sd_exp6_benchmark run" in script
    assert "strategic_audit.sd_exp6 \\\n" not in script
    assert "FHP_PLAN_SHA256" in script and "--download" in script
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    # Parse the exact rendered shell, rather than just testing string snippets.
    import subprocess
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_benchmark_runs_matched_cells_reverse_order_without_reuse(tmp_path, monkeypatch):
    from fhp_evaluation.strategic_audit import board_export, workflow, library
    from fhp_evaluation.best_response.runner import write_json
    plan = b.make_plan(progress())
    policies = {f"sd_exp6_seed0_{h}h": dict(family="sd_cfr") for h in (24, 48)}
    monkeypatch.setattr(b.audit, "make_spec", lambda *a: dict(policies=policies))
    monkeypatch.setattr(b.audit, "make_library", lambda *a: {})
    monkeypatch.setattr(library, "board_schedule", lambda *a: plan["boards"])
    monkeypatch.setattr(workflow, "policy_identity", lambda item: item)
    seen = []
    def export(provenance, boards, directory, fingerprint, config, **kwargs):
        assert boards == plan["boards"] and config["workers"] * provenance["threads"] <= 16
        assert not directory.exists()
        seen.append((config["workers"], provenance["threads"]))
        tasks = []
        for label in policies:
            for board in boards:
                path = directory / "tables" / label / library.board_key(board)
                path.mkdir(parents=True)
                np.savez(path / "tables.npz", p=np.array([[.4, .6]]))
                tasks.append(dict(policy=label, board=board))
        return dict(tasks=tasks)
    monkeypatch.setattr(board_export, "export_boards", export)
    result = b.benchmark(plan, tmp_path / "inputs", tmp_path / "native", tmp_path / "output")
    assert seen == list(b.ALLOCATIONS) + list(reversed(b.ALLOCATIONS))
    assert len(result["cells"]) == 6 and all(g["eligible"] for g in result["groups"].values())
    assert all(len(c["tasks"]) == 8 for c in result["cells"])
    assert json.loads((tmp_path / "output/status.json").read_text())["status"] == "benchmark_complete"
    with pytest.raises(FileExistsError):
        b.benchmark(plan, tmp_path / "inputs", tmp_path / "native", tmp_path / "output")
