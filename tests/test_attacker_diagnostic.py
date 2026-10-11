"""Exp26 independent-panel, frozen-source and original-learner regression tests."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_adversarial import native_root, target
from fhp_evaluation.adversarial.config import configuration
from fhp_evaluation.adversarial.diagnostic_config import protocol, stream
from fhp_evaluation.adversarial.diagnostic_checks import correctness_checks
from fhp_evaluation.adversarial import diagnostic as d
from fhp_evaluation.adversarial.learning import NetworkPolicy, model_hash, train_attacker

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_disjoint_streams():
    p = protocol()
    assert p["checkpoints_seconds"] == [0, 300, 900, 1800, 3600]
    assert p["exploration_seconds"] == 900
    assert p["final_pairs"] == 10000 and p["source_seed"] == 0
    base = configuration().to_dict()
    base["attack_seconds"] = 3600
    assert p["learner"] == base
    assert len({stream(s) for s in ("development", "final", "calibration")}) == 3
    assert configuration().attack_seconds == 900  # No Exp21 budget change.


def test_cli_normalises_symlinked_paths(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    captured = []
    monkeypatch.setattr(d, "run", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["diagnostic", "run", "--output", str(link / "out"),
                       "--inputs", str(link / "in"), "--native-repo", str(link / "native"), "--bucket", "gs://test"])
    d.main()
    assert captured[0].inputs == real / "in" and captured[0].output == real / "out"


def test_real_game_correctness_checks(game, target):
    torch.set_num_threads(2)
    result = correctness_checks(game, target)
    assert result["passed"] and result["native_parity_states"] > 30
    assert result["native_parity_max_error"] < 3e-6


def test_snapshot_hook_does_not_change_fixed_work_training(game, target, monkeypatch):
    from fhp_evaluation.adversarial import learning
    # End at the identical fixed hand count, making elapsed-time sampling irrelevant.
    monkeypatch.setattr(learning.time, "monotonic", lambda: 0.)
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    cfg = replace(configuration(True), episode_limit=32)
    standard = train_attacker(game, reference, 0, 12345, cfg)
    observations = []
    with_snapshots = train_attacker(game, reference, 0, 12345, cfg,
        checkpoint_seconds=[0.], on_checkpoint=lambda t, state, metrics: observations.append(deepcopy(state)))
    assert len(observations) == 1
    assert model_hash(standard[0].model) == model_hash(with_snapshots[0].model)
    assert standard[3]["hands"] == with_snapshots[3]["hands"] == 32
    assert standard[2]["rng"] == with_snapshots[2]["rng"]
    for key in standard[2]["target"]:
        assert torch.equal(standard[2]["target"][key], with_snapshots[2]["target"][key])


def test_snapshot_validation_and_early_episode_limit(game, target):
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    cfg = replace(configuration(True), episode_limit=4, attack_seconds=1000)
    with pytest.raises(ValueError, match="snapshots"):
        train_attacker(game, reference, 0, 123, cfg, checkpoint_seconds=[2, 1], on_checkpoint=lambda *a: None)
    with pytest.raises(RuntimeError, match="Episode limit"):
        train_attacker(game, reference, 0, 123, cfg, checkpoint_seconds=[0, 1000], on_checkpoint=lambda *a: None)


def test_seat_scoring_matches_existing_duplicate_evaluator(game, target):
    from fhp_evaluation.adversarial.workflow import pairs, SeatPolicies
    from fhp_evaluation.adversarial.learning import attacker_model
    ref = NetworkPolicy(target.native.model, target.native.feature_encoder)
    actors = [NetworkPolicy(attacker_model(213, s), ref.feature_encoder, greedy=True) for s in (31, 32)]
    result = d.score(game, actors, ref, 16, 126)
    expected = pairs(game, SeatPolicies(actors), ref, 16, 126)
    np.testing.assert_array_equal(np.asarray(result["seat_payoffs_mbb"]).mean(1), expected)
    assert result["payoff_mbb"]["mean"] == expected.mean()
    difference = d.paired_difference(result, result)
    assert difference["payoff_mbb"]["mean"] == difference["payoff_mbb"]["se"] == 0


def test_real_smoke_learning_and_locked_evaluation(game, target, tmp_path, monkeypatch):
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "a" * 64)
    torch.set_num_threads(2)
    spec = protocol(True)
    ref = NetworkPolicy(target.native.model, target.native.feature_encoder)
    before = model_hash(ref.model)
    source_hash = target.metadata["checkpoint_sha256"]
    for seat in (0, 1):
        for rep in range(2):
            out = tmp_path / "train" / f"seat{seat}_rep{rep}"
            out.mkdir(parents=True)
            manifest = d.train_worker(game, ref, spec, seat, rep, out, source_hash)
            assert manifest["complete"] and len(manifest["checkpoints"]) == 5
            assert manifest["metrics"]["updates"] > 0
            assert manifest["metrics"]["active_seconds"] >= .6
            assert manifest["metrics"]["snapshot_seconds"] > 0
    output = tmp_path / "assessment"
    output.mkdir()
    original = d.score
    def checked_score(game, actors, ref, count, seed, events=None):
        if seed == stream("final"):
            assert (output / "selection_lock.json").exists()
        return original(game, actors, ref, count, seed, events)
    monkeypatch.setattr(d, "score", checked_score)
    result = d.evaluate_worker(game, ref, spec, tmp_path, output, source_hash)
    assert len(result["development_curves"]) == 10
    assert set(result["final"]) == {"initial", "baseline_15m", "extended_30m", "extended_60m"}
    assert result["interpretation"] == "execution_smoke_only"
    assert not result["policy_repair_authorised"]
    assert model_hash(ref.model) == before
    d.write_report(output / "report.md", result, dict(wall_seconds=1, summed_worker_seconds=1))
    assert "Independent final" in (output / "report.md").read_text()
    p = tmp_path / "train/seat0_rep0/checkpoints/checkpoint_0s.pt"
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        d.load_attacker(tmp_path, spec, ref, 0, 0, 0., source_hash)


def test_selection_cannot_use_final_and_ties_are_stable():
    rows = [dict(nominal_seconds=900, repetition=i, payoff_mbb=dict(mean=v)) for i, v in [(1, 20), (0, 20)]]
    rows += [dict(nominal_seconds=3600, repetition=0, payoff_mbb=dict(mean=1000))]
    assert d.choose(rows, 900) == 0


def batch_module():
    path = ROOT / "gcp/exp26_attacker_diagnostic_batch.py"
    spec = importlib.util.spec_from_file_location("exp26_batch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("smoke", [False, True])
def test_batch_and_offline_prepare(tmp_path, smoke):
    from argparse import Namespace
    module = batch_module()
    native = ROOT.parent / "fhp_ucv_escher/fhp-ucv-escher-experiments"
    args = Namespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
        service_account="runner@test-project.iam.gserviceaccount.com", run_id="exp26-attack-test",
        smoke=smoke, native_repo=native, run_dir=tmp_path / "prepared")
    module.validate(args)
    job = module.build_job(args, "a" * 64)
    task = job["taskGroups"][0]
    assert task["taskCount"] == task["parallelism"] == 1
    assert task["taskSpec"]["maxRetryCount"] == 0
    assert task["taskSpec"]["computeResource"]["cpuMilli"] == 8000
    assert task["taskSpec"]["maxRunDuration"] == ("5400s" if smoke else "21600s")
    for runnable in task["taskSpec"]["runnables"]:
        script = runnable["script"]["text"]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "gcloud batch" not in script
        assert "/workspace/fhp-exp21" not in script
        assert '"scientific_complete"' not in script
    main = task["taskSpec"]["runnables"][0]["script"]["text"]
    assert "adversarial.diagnostic run" in main and "adversarial.pipeline" not in main
    assert "--smoke-only" not in main
    # -S prevents ML/site packages from loading in the laptop-side preparer.
    command = [sys.executable, "-S", str(ROOT / "gcp/exp26_attacker_diagnostic_batch.py"), "prepare",
               "--project", args.project, "--region", args.region, "--bucket", args.bucket,
               "--service-account", args.service_account, "--run-id", args.run_id,
               "--native-repo", str(native), "--run-dir", str(args.run_dir)]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, capture_output=True, text=True)
    request = json.loads((args.run_dir / "request.json").read_text())
    assert request["protocol"] == protocol(smoke)
    assert request["files"]["source.tar.gz"] == request["bundle_sha256"]


def test_failed_preflight_never_trains(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "b" * 64)
    calls = []
    def fail(spec, directory, **kw):
        calls.append(spec["phase"])
        return dict(status="failed")
    monkeypatch.setattr(d, "run_bounded", fail)
    args = Namespace(smoke=True, output=tmp_path / "out", inputs=tmp_path / "in",
                     native_repo=tmp_path / "native", bucket="gs://test-bucket")
    with pytest.raises(RuntimeError, match="preflight"):
        d.run(args)
    assert calls == ["preflight"]
    assert json.loads((args.output / "STATUS.json").read_text())["status"] == "failed"
    assert not (args.output / "analysis/SUCCESS.json").exists()


def test_complete_negative_result_is_success_not_policy_repair(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "c" * 64)
    calls = []
    def completed(spec, directory, **kw):
        phase = spec["phase"]
        calls.append(phase)
        row = dict(status="complete", phase=phase, **d.identity(protocol()))
        if phase == "preflight":
            row["passed"] = True
        elif phase == "assessment":
            assert calls.count("train") == 4
            row.update(interpretation="extended_attacker_still_loses_review_learner_before_repair",
                       policy_repair_authorised=False)
        return dict(status="succeeded", result=row, elapsed_seconds=1., peak_rss_bytes=100)
    monkeypatch.setattr(d, "run_bounded", completed)
    monkeypatch.setattr(d, "write_report", lambda *args: None)
    args = Namespace(smoke=False, output=tmp_path / "out", inputs=tmp_path / "in",
                     native_repo=tmp_path / "native", bucket="gs://test-bucket")
    d.run(args)
    marker = json.loads((args.output / "analysis/SUCCESS.json").read_text())
    assert marker["status"] == "complete" and marker["policy_repair_authorised"] is False
    assert calls[0] == "preflight" and calls[-1] == "assessment" and len(calls) == 6
    with pytest.raises(FileExistsError):
        d.run(args)


def test_train_failure_cancels_peers_and_skips_assessment(tmp_path, monkeypatch):
    from argparse import Namespace
    import threading
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "d" * 64)
    barrier = threading.Barrier(4, timeout=5)
    calls = []
    def bounded(spec, directory, **kw):
        calls.append(spec["phase"])
        if spec["phase"] == "preflight":
            return dict(status="succeeded", result=dict(status="smoke_complete", passed=True,
                        phase="preflight", **d.identity(protocol(True))))
        assert spec["phase"] == "train"
        barrier.wait()
        if spec["seat"] == spec["repetition"] == 0:
            return dict(status="failed")
        assert kw["cancel_event"].wait(timeout=5)
        return dict(status="cancelled")
    monkeypatch.setattr(d, "run_bounded", bounded)
    args = Namespace(smoke=True, output=tmp_path / "out", inputs=tmp_path / "in",
                     native_repo=tmp_path / "native", bucket="gs://test-bucket")
    with pytest.raises(RuntimeError, match="train"):
        d.run(args)
    assert "assessment" not in calls
    assert not (args.output / "analysis/SUCCESS.json").exists()


def test_ucv_root_wrapper_prepares_both_modes(tmp_path):
    import os
    native = ROOT.parent / "fhp_ucv_escher/fhp-ucv-escher-experiments"
    wrapper = native / "gcp/run_exp26_attacker_diagnostic.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    for smoke in (False, True):
        directory = tmp_path / str(smoke)
        env = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1", BUCKET="gs://test-bucket",
                   SA_EMAIL="runner@test-project.iam.gserviceaccount.com", RUN_ID="exp26-attack-wrapper",
                   FHP_EVAL_REPO=str(ROOT), EXP26_RUN_DIR=str(directory), PYTHON=sys.executable)
        subprocess.run(["bash", str(wrapper), "prepare-smoke" if smoke else "prepare"],
                       env=env, cwd=native, check=True, capture_output=True, text=True)
        assert json.loads((directory / "request.json").read_text())["smoke"] is smoke
