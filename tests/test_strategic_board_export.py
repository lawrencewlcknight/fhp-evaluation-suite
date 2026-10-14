import json
import threading
import time

import numpy as np
import pytest

from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.strategic_audit import board_export as b
from fhp_evaluation.strategic_audit import workflow as w


def inputs():
    policies = {"sd": {"family": "sd_cfr"}, "vr": {"family": "vr_deep"}}
    return dict(spec=dict(policies=policies), policies=policies, threads=4)


def config(workers=4):
    return dict(workers=workers, task_seconds=20, session_seconds=100, cpu_limit=16)


def cache(path, fingerprint):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "tables.npz", probabilities=np.array([.25, .75]))
    write_json(path / "diagnostics.json", dict(checked=True))
    w.mark(path, fingerprint, "tables.npz")


def test_resource_contract_rejects_oversubscription():
    assert b.validate_config(config(), 4, 52000) == config()
    with pytest.raises(ValueError, match="CPU"):
        b.validate_config(config(4), 8, 52000)
    for values in (dict(config(), workers=0), dict(config(), task_seconds=101),
                   dict(config(), cpu_limit=True), dict(config(), unexpected=1)):
        with pytest.raises(ValueError):
            b.validate_config(values, 4, 52000)


def test_parallel_unique_boards_memory_budget_resume_and_timings(tmp_path, monkeypatch):
    boards = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    active, peak, seen, specs = 0, 0, set(), []
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    # Use cache path from worker spec, not assumptions about log layout.
    def run(spec, directory, **kwargs):
        nonlocal active, peak
        with lock:
            task = (spec["cache"], tuple(spec["boards"][0]))
            assert task not in seen
            seen.add(task)
            specs.append(spec)
            active += 1
            peak = max(peak, active)
        assert len(spec["boards"]) == 1 and spec["threads"] == 4
        assert kwargs["max_rss_mb"] == 13000 and kwargs["max_seconds"] <= 20
        barrier.wait(timeout=5)
        from pathlib import Path
        cache(Path(spec["cache"]) / b.board_key(spec["boards"][0]), "f")
        with lock:
            active -= 1
        return dict(status="succeeded", elapsed_seconds=2, peak_rss_bytes=1024)
    monkeypatch.setattr(b, "run_bounded", run)
    report = b.export_boards(inputs(), boards, tmp_path, "f", config(),
                             deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    assert peak == 4 and len(seen) == 8
    assert all(s["policy"]["family"] == "sd_cfr" for s in specs[:4])
    assert report["completed_boards"] == 8 and report["remaining_boards"] == 0
    assert len(report["tasks"]) == 8 and report["scheduler"] == "rolling_v1"
    assert report["waves"][0]["boards_per_hour"] > 0
    monkeypatch.setattr(b, "run_bounded", lambda *a, **k: pytest.fail("repeated a completed board"))
    report = b.export_boards(inputs(), boards, tmp_path, "f", config(),
                             deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    assert report["initial_cached_boards"] == 8 and report["waves"] == []


def test_deadline_defers_without_launch_or_fake_success(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "run_bounded", lambda *a, **k: pytest.fail("launched past budget"))
    with pytest.raises(b.AuditDeferred):
        b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", config(),
                        deadline=time.monotonic()-1, max_rss_mb=52000, completed=w.completed)
    assert json.loads((tmp_path / "export_progress.json").read_text())["status"] == "deferred_time_budget"
    assert not (tmp_path / "analysis/SUCCESS.json").exists()


@pytest.mark.parametrize("status", ["time_limit", "memory_limit", "failed", "succeeded"])
def test_worker_failures_and_missing_markers_never_satisfy_completion(tmp_path, monkeypatch, status):
    monkeypatch.setattr(b, "run_bounded", lambda *a, **k:
                        dict(status=status, elapsed_seconds=20, peak_rss_bytes=1024))
    with pytest.raises(RuntimeError, match="Board export failed"):
        b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", config(),
                        deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    assert json.loads((tmp_path / "export_progress.json").read_text())["status"] == "failed"


def test_corrupt_cached_board_is_rejected(tmp_path):
    path = tmp_path / "tables/sd/0-1-2"
    cache(path, "f")
    (path / "diagnostics.json").write_text("changed")
    with pytest.raises(ValueError, match="diagnostics"):
        b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", config(),
                        deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)


def test_partial_wave_resume_only_retries_missing_board(tmp_path, monkeypatch):
    from pathlib import Path
    done = []
    def run(spec, directory, **kwargs):
        directory.mkdir(parents=True)
        board = spec["boards"][0]
        if spec["policy"]["family"] == "vr_deep":
            return dict(status="time_limit", elapsed_seconds=20, peak_rss_bytes=10)
        cache(Path(spec["cache"]) / b.board_key(board), "f")
        done.append(directory)
        return dict(status="succeeded", elapsed_seconds=1, peak_rss_bytes=10)
    monkeypatch.setattr(b, "run_bounded", run)
    with pytest.raises(RuntimeError, match="Board export failed"):
        b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", config(2),
                        deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    def retry(spec, directory, **kwargs):
        assert spec["policy"]["family"] == "vr_deep"
        assert directory.name == "attempt_001"
        cache(Path(spec["cache"]) / b.board_key(spec["boards"][0]), "f")
        return dict(status="succeeded", elapsed_seconds=1, peak_rss_bytes=10)
    monkeypatch.setattr(b, "run_bounded", retry)
    report = b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", config(2),
                             deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    assert report["initial_cached_boards"] == 1 and report["completed_boards"] == 2


def test_workflow_freezes_execution_settings_and_defers_analysis(tmp_path, monkeypatch):
    from fhp_evaluation.strategic_audit.commands import create_spec
    from fhp_evaluation.strategic_audit.library import build_library
    spec, library = create_spec(smoke=True), build_library(pilot=True, reference_deals=0)
    # Keep the manifest real, isolate expensive gameplay.
    monkeypatch.setattr(b, "export_boards", lambda *a, **kw: None)
    monkeypatch.setattr(w, "run_bounded", lambda *a, **kw: dict(status="time_limit"))
    kwargs = dict(board_exports=config(), max_seconds=90, threads=4)
    with pytest.raises(b.AuditDeferred, match="Analysis"):
        w.run(spec, library, tmp_path, **kwargs)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "deferred_time_budget"
    with pytest.raises(ValueError, match="identical inputs"):
        w.run(spec, library, tmp_path, resume=True, board_exports=config(2), threads=4)


@pytest.mark.slow
def test_real_isolated_board_exports_match_serial_tables(tmp_path):
    """Real OpenSpiel, full-deck queries and subprocesses; not a cloud speed claim."""
    from fhp_evaluation.game import load_fhp_game
    from fhp_evaluation.best_response.adapters import PolicyAdapter
    from fhp_evaluation.best_response.validation import FixedPolicy
    from fhp_evaluation.strategic_audit.tables import BoardContext, build_tables, load_tables
    policy = dict(family="control", name="uniform")
    provenance = dict(spec=dict(policies=dict(control=policy)),
                      policies=dict(control=w.policy_identity(policy)), threads=1)
    boards = [[0, 5, 10], [16, 20, 32]]
    limits = dict(workers=2, task_seconds=300, session_seconds=600, cpu_limit=2)
    report = b.export_boards(provenance, boards, tmp_path, "real", limits,
                             deadline=time.monotonic()+600, max_rss_mb=4096, completed=w.completed)
    assert report["completed_boards"] == 2
    game = load_fhp_game()
    for board in boards:
        context = BoardContext(game, board)
        expected = build_tables(context, PolicyAdapter(FixedPolicy(game, "uniform")))
        actual = load_tables(tmp_path / "tables/control" / b.board_key(board) / "tables.npz", context)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_bounded_worker_cancellation_stops_owned_process(tmp_path):
    import sys
    from fhp_evaluation.best_response.runner import run_bounded
    event = threading.Event()
    event.set()
    result = run_bounded(dict(threads=1), tmp_path / "cancel", max_seconds=30,
                         cancel_event=event, command=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert result["status"] == "cancelled"
    assert result["elapsed_seconds"] < 10
    assert result["returncode"] != 0


def test_inflight_session_timeout_is_deferred_not_board_failure(tmp_path, monkeypatch):
    limits = dict(workers=1, task_seconds=7200, session_seconds=16200, cpu_limit=4)
    monkeypatch.setattr(b, "run_bounded", lambda *a, **kw:
                        dict(status="time_limit", elapsed_seconds=400, peak_rss_bytes=100))
    with pytest.raises(b.AuditDeferred, match="Invocation budget expired"):
        b.export_boards(inputs(), [[0, 1, 2]], tmp_path, "f", limits,
                        deadline=time.monotonic()+400, max_rss_mb=52000, completed=w.completed)
    assert json.loads((tmp_path / "export_progress.json").read_text())["status"] == "deferred_time_budget"


def test_no_analysis_until_all_board_exports_complete(tmp_path, monkeypatch):
    from fhp_evaluation.strategic_audit.commands import create_spec
    from fhp_evaluation.strategic_audit.library import build_library
    def defer(*args, **kwargs):
        raise b.AuditDeferred("partial exports")
    monkeypatch.setattr(b, "export_boards", defer)
    monkeypatch.setattr(w, "run_bounded", lambda *a, **kw: pytest.fail("analysis ran on partial panel"))
    with pytest.raises(b.AuditDeferred, match="partial exports"):
        w.run(create_spec(smoke=True), build_library(pilot=True, reference_deals=0), tmp_path,
              board_exports=config(), threads=4)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "deferred_time_budget"


def test_rolling_refills_before_slowest_task_finishes(tmp_path, monkeypatch):
    from pathlib import Path
    slow_started, third_started = threading.Event(), threading.Event()
    boards = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    provenance = dict(spec=dict(policies=dict(sd={"family": "sd_cfr"})),
                      policies=dict(sd={"family": "sd_cfr"}), threads=1)
    def run(spec, directory, **kwargs):
        board = spec["boards"][0]
        if board == boards[0]:
            slow_started.set()
            assert third_started.wait(timeout=5), "idle slot was not refilled"
        elif board == boards[1]:
            assert slow_started.wait(timeout=5)
        else:
            third_started.set()
        cache(Path(spec["cache"]) / b.board_key(board), "f")
        return dict(status="succeeded", elapsed_seconds=1, peak_rss_bytes=10)
    monkeypatch.setattr(b, "run_bounded", run)
    report = b.export_boards(provenance, boards, tmp_path, "f", config(2),
                             deadline=time.monotonic()+100, max_rss_mb=1000, completed=w.completed)
    assert report["completed_boards"] == 3


def test_failed_board_does_not_prevent_other_boards_and_is_not_retried(tmp_path, monkeypatch):
    from pathlib import Path
    seen = []
    def run(spec, directory, **kwargs):
        seen.append((spec["cache"], spec["boards"]))
        if len(seen) == 1:
            return dict(status="failed", elapsed_seconds=1, peak_rss_bytes=10)
        cache(Path(spec["cache"]) / b.board_key(spec["boards"][0]), "f")
        return dict(status="succeeded", elapsed_seconds=1, peak_rss_bytes=10)
    monkeypatch.setattr(b, "run_bounded", run)
    with pytest.raises(RuntimeError, match="Board export failed"):
        b.export_boards(inputs(), [[0, 1, 2], [3, 4, 5]], tmp_path, "f", config(1),
                        deadline=time.monotonic()+100, max_rss_mb=52000, completed=w.completed)
    report = json.loads((tmp_path / "export_progress.json").read_text())
    assert len(seen) == 4 and report["completed_boards"] == 3
    assert report["remaining_boards"] == 1 and len(report["failures"]) == 1
    assert report["status"] == "failed"
