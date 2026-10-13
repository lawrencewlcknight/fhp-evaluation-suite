import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.loaders import sha256_file
from fhp_evaluation.safeguard.experiment import config, tasks, profiling_tasks, opponents, is_response
from fhp_evaluation.safeguard import runner
from fhp_evaluation.safeguard.runner import completed, aggregate_tasks
from fhp_evaluation.safeguard.policy import TerminalCallVerifier
from fhp_evaluation.game import load_fhp_game
from fhp_evaluation.strategic_audit import PROTOCOL
from fhp_evaluation.strategic_audit.library import texture
from fhp_evaluation.strategic_audit.workflow import run

ROOT = Path(__file__).resolve().parents[1]


def test_guarded_strategic_workflow_and_recovery(tmp_path):
    spec = dict(protocol=PROTOCOL, policies={
        "base": dict(family="control",name="uniform",cohort="base",training_seed=0,hours=48),
        "guard": dict(family="control",name="uniform",cohort="guard",training_seed=0,hours=48,
                      deployment="exact_terminal_call_v1")},
        candidates=["base","guard"],opponents=["base"],reference_policy="base",range_tilts=[0])
    board = [6,10,50]
    library = dict(protocol=PROTOCOL,boards=[dict(board=board,texture=texture(board),partition="development")],
                   assessment_board_classes=[],reference_deals=[])
    path = run(spec,library,tmp_path,threads=1,max_seconds=90,max_rss_mb=4096)
    result = json.loads(path.read_text())
    base = result["comparisons"]["base__vs__base__tilt_0"]["mean_root_response_gap_bb"]
    guarded = result["comparisons"]["guard__vs__base__tilt_0"]["mean_root_response_gap_bb"]
    assert guarded < base
    assert run(spec,library,tmp_path,threads=1,max_seconds=90,max_rss_mb=4096,resume=True) == path


def test_offline_cloud_preparation_bundles_both_repos(tmp_path):
    native = tmp_path / "native"
    for name in ("fhp_escher", "vr_deep_cfr"):
        (native/name).mkdir(parents=True)
        (native/name/"__init__.py").write_text("")
    (native/"fhp_escher/checkpointing.py").write_text("# test fixture\n")
    for mode in ("smoke","full"):
        target = tmp_path/mode
        result = subprocess.run([sys.executable,str(ROOT/"gcp/exp34_ucv_terminal_safeguard_batch.py"),
            "prepare","--project","clever-overview-399515","--region","europe-west1",
            "--bucket","gs://clever-overview-399515-fhp-escher-results",
            "--service-account","fhp-runner@clever-overview-399515.iam.gserviceaccount.com",
            "--run-id",f"exp34-safeguard-test-{mode}","--mode",mode,
            "--native-repo",str(native),"--run-dir",str(target)],capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        request = json.loads((target/"request.json").read_text())
        assert request["experiment"] == config(mode)
        assert "evaluator/fhp_evaluation/safeguard/policy.py" in request["source"]["files"]
        assert "native/fhp_escher/checkpointing.py" in request["source"]["files"]
        job = json.loads((target/"job.json").read_text())
        task = job["taskGroups"][0]["taskSpec"]
        assert task["environment"]["variables"]["FHP_SAFEGUARD_MODE"] == mode
        assert task["maxRetryCount"] == 0
        assert task["runnables"][-1]["alwaysRun"]
        assert "fhp_evaluation.safeguard.runner" in task["runnables"][0]["script"]["text"]
        assert task["maxRunDuration"] == ("172800s" if mode == "full" else "7200s")
        assert request["experiment"]["session_seconds"] == (46*3600 if mode == "full" else 1800)
        assert int(task["runnables"][0]["timeout"][:-1]) + int(task["runnables"][1]["timeout"][:-1]) <= int(task["maxRunDuration"][:-1])


def test_profile_is_repeated_production_work_not_an_additional_sample():
    spec = config("full")
    full = list(tasks(spec))
    selected = profiling_tasks(spec)
    counts = Counter((r["seed"], r["opponent"]) for r in selected)
    assert len(counts) == sum(len(opponents(spec, seed)) for seed in spec["seeds"])
    assert set(counts.values()) == {2}
    assert len({r["id"] for r in selected}) == len(selected)
    assert all(row in full for row in selected)
    assert all(row["count"] == (25 if is_response(row["opponent"]) else 2000) for row in selected)
    assert sum(r["count"] for r in selected) == 168_600
    assert sum(r["count"] for r in full) == 3_804_000
    # First repeat covers every matchup before the second repeat begins.
    assert all(row["start"] == 0 for row in selected[:len(counts)])
    assert all(row["start"] > 0 for row in selected[len(counts):])


def test_slow_forecast_continues_and_reuses_verified_production_shards(tmp_path, monkeypatch, capsys):
    spec = dict(config("smoke"), mode="full", primary_pairs=10, panel_pairs=8, response_pairs=5,
                shard_pairs=3, response_shard_pairs=2, processes=2)
    args = SimpleNamespace(input_root=tmp_path/"input", native_root=tmp_path/"native", output=tmp_path)
    fingerprint = "same-source-and-config"
    scored = []

    def score(task, input_root, native_root, supplied_spec, directory, identity):
        assert supplied_spec == spec  # No substituted timing-only deal stream.
        scored.append(task["id"])
        path = Path(directory)/task["id"]
        path.mkdir(parents=True)
        # Two distinct (deliberately very slow) timing measurements per matchup.
        seconds = task["count"] * (10000 if task["start"] == 0 else 20000)
        np.savez_compressed(path/"pairs.npz", rows=np.zeros((task["count"], 8)))
        write_json(path/"result.json", dict(task=task, identity=identity, seconds=seconds,
                                          deployment_verifier=None))
        write_json(path/"SUCCESS.json", dict(identity=identity,
            result_sha256=sha256_file(path/"result.json"), pairs_sha256=sha256_file(path/"pairs.npz")))
        return dict(id=task["id"], seconds=seconds, count=task["count"], seed=task["seed"], opponent=task["opponent"])

    monkeypatch.setattr(runner, "score_task", score)
    monkeypatch.setattr(runner, "worker_pool", lambda cfg: ThreadPoolExecutor(max_workers=cfg["processes"]))
    deadline = time.monotonic()+60
    result = runner.profile(args, spec, fingerprint, deadline)
    assert result["decision"] == "continue" and result["advisory_only"]
    assert not result["within_planning_target"]
    total_pairs = sum(row["count"] for row in tasks(spec))
    # Pool repeat timings within each matchup; do not count its full workload twice.
    assert result["unpadded_gameplay_hours"] == pytest.approx(15000*total_pairs/2/3600)
    assert result["projected_gameplay_hours"] == pytest.approx(result["unpadded_gameplay_hours"]*2)
    assert all(row["measured_shards"] == 2 for row in result["matchups"])
    assert all(row["min_shard_seconds_per_pair"] == 10000 for row in result["matchups"])
    assert all(row["max_shard_seconds_per_pair"] == 20000 for row in result["matchups"])
    assert '"level": "warning"' in capsys.readouterr().out
    assert json.loads((tmp_path/"analysis/profile.json").read_text()) == result
    profiled_ids = set(scored)
    assert profiled_ids == {row["id"] for row in profiling_tasks(spec)}
    # Recover after profiling: neither the profile nor the main pass repeats completed work.
    runner.profile(args, spec, fingerprint, deadline)
    assert len(scored) == len(profiled_ids)
    runner.run_tasks(list(tasks(spec)), args, spec, tmp_path/"task_results", fingerprint, deadline)
    assert set(scored) == {row["id"] for row in tasks(spec)}
    assert len(scored) == len(set(scored))
    aggregate_tasks(args, spec, fingerprint)
    assert (tmp_path/"analysis/summary.json").is_file()
    # An advisory runtime policy must not bypass evidence-integrity checks.
    first = profiling_tasks(spec)[0]
    (tmp_path/"task_results"/first["id"]/"pairs.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Changed completed shard"):
        runner.profile(args, spec, fingerprint, deadline)


def test_strategic_audit_can_use_remaining_session_not_an_18_hour_subcap(tmp_path, monkeypatch):
    from fhp_evaluation.strategic_audit import library, workflow
    args = SimpleNamespace(output=tmp_path, input_root=tmp_path/"input", native_root=tmp_path/"native")
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.)
    monkeypatch.setattr(library, "build_library", lambda *a, **kw: {})
    calls = []

    def strategic_run(spec, boards, root, **kwargs):
        calls.append(kwargs)
        path = root/"analysis/summary.json"
        write_json(path, dict(comparisons={}))
        return path

    monkeypatch.setattr(workflow, "run", strategic_run)
    runner.run_strategic(args, config("full"), 100.+24*3600)
    assert calls[0]["max_seconds"] == 24*3600
    assert calls[0]["max_rss_mb"] == 48000
    assert (tmp_path/"analysis/strategic_comparison.json").is_file()
    with pytest.raises(TimeoutError, match="Session budget exhausted"):
        runner.run_strategic(args, config("full"), 130.)
    assert len(calls) == 1


def test_partial_shard_not_treated_as_complete(tmp_path):
    task = dict(id="test",seed=0,opponent="base",start=0,count=5)
    (tmp_path/"test").mkdir()
    (tmp_path/"test/result.json").write_text("{}")
    assert not completed(tmp_path,task,"identity")


def test_complete_aggregation_and_checksum_admission(tmp_path):
    spec = config("smoke")
    fingerprint = "test-identity"
    timing = TerminalCallVerifier(load_fhp_game()).diagnostics()
    for task in tasks(spec):
        path = tmp_path/"task_results"/task["id"]
        path.mkdir(parents=True)
        np.savez_compressed(path/"pairs.npz",rows=np.zeros((task["count"],8)))
        write_json(path/"result.json",dict(task=task,identity=fingerprint,deployment_verifier=timing))
        write_json(path/"SUCCESS.json",dict(identity=fingerprint,
            result_sha256=sha256_file(path/"result.json"),pairs_sha256=sha256_file(path/"pairs.npz")))
    aggregate_tasks(SimpleNamespace(output=tmp_path),spec,fingerprint)
    summary = json.loads((tmp_path/"analysis/summary.json").read_text())
    assert summary["primary"]["integrated_delta_mbb"]["n"] == 1
    assert summary["primary_by_seed"]["0"]["zero_event_warning"]
    assert len(summary["panels"]) == 7
    latency = json.loads((tmp_path/"analysis/latency.json").read_text())
    assert "ineligible" in latency["verifier_overhead"]["deployment_verifier"]["latency_counts"]
    task = next(iter(tasks(spec)))
    (tmp_path/"task_results"/task["id"]/"result.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        completed(tmp_path/"task_results",task,fingerprint)
