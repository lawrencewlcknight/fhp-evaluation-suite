"""Exp27 loss algebra, exact controls, gating, provenance and Batch integration."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_adversarial import native_root, target
from fhp_evaluation.adversarial import development as d
from fhp_evaluation.adversarial.config import configuration
from fhp_evaluation.adversarial.development_config import ARMS, CONTROLS, cell_protocol, protocol, replication_allowed, stream
from fhp_evaluation.adversarial.development_controls import ControlPolicy, exact_oracle, exact_score, leduc_reference, stochastic_loss_check
from fhp_evaluation.adversarial.learning import NetworkPolicy, attacker_loss, collect, model_hash, train_attacker

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_separation_and_unchanged_defaults():
    p = protocol()
    assert p["control_seconds"] == 300 and p["attack_seconds"] == 3600
    assert p["replication_seeds"] == [1, 2] and p["final_pairs"] == 10000
    a, b = (cell_protocol(p, "exp10", arm) for arm in ARMS)
    assert a["learner"] == b["learner"]
    assert a["exploration_seconds"] == 900 and a["reward_scale"] == 100
    assert cell_protocol(p, "leduc_uniform", ARMS[0])["reward_scale"] == 1
    assert configuration().attack_seconds == 900
    assert len({stream(x, 0) for x in ("training", "development", "final", "calibration", "control-assessment")}) == 5
    assert not replication_allowed(dict(smoke=True, qualified=True, selected_arm=ARMS[0]))
    assert not replication_allowed(dict(smoke=False, qualified=False, selected_arm=ARMS[0]))
    assert replication_allowed(dict(smoke=False, qualified=True, selected_arm=ARMS[1]))


def test_loss_expectations_and_local_gradient_matching():
    assert stochastic_loss_check()["passed"]
    x = torch.tensor([-.2, .3], requires_grad=True)
    y = torch.zeros_like(x)
    gradients = [torch.autograd.grad(attacker_loss(x, y, kind), x, retain_graph=True)[0] for kind in ARMS]
    assert torch.equal(*gradients)
    assert attacker_loss(torch.tensor([5.]), torch.tensor([0.]), "half_mse") > attacker_loss(torch.tensor([5.]), torch.tensor([0.]))
    with pytest.raises(ValueError):
        attacker_loss(x, y, "unknown")


def test_reward_scaling_and_exact_leduc_control():
    game, ref = leduc_reference()
    a = collect(game, ref, [ref]*16, [i%2 for i in range(16)], np.random.default_rng(5), reward_scale=1)
    b = collect(game, ref, [ref]*16, [i%2 for i in range(16)], np.random.default_rng(5), reward_scale=100)
    np.testing.assert_allclose([e.reward for e in a], np.asarray([e.reward for e in b])*100)
    oracle = exact_oracle(game, ref)
    assert oracle["per_seat_chips"] == pytest.approx([2.0875, 2.6597222222222223])
    assert exact_score(game, [ref, ref], ref)["mean_chips"] == pytest.approx(0.)
    assert oracle["mean_chips"] > 0
    with pytest.raises(ValueError, match="scale"):
        collect(game, ref, [ref], [0], np.random.default_rng(5), reward_scale=0)


def test_fold_prone_control_allows_both_seats_to_learn(game, target):
    ref = ControlPolicy(target.native.feature_encoder, "fhp_fold")
    actor = ControlPolicy(target.native.feature_encoder, "fhp_call")
    episodes = collect(game, actor, [ref]*128, [1]*128, np.random.default_rng(27))
    assert any(e.rows for e in episodes) and any(not e.rows for e in episodes)


def test_default_training_equals_explicit_original_loss(game, target, monkeypatch):
    from fhp_evaluation.adversarial import learning
    monkeypatch.setattr(learning.time, "monotonic", lambda: 0.)
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    cfg = replace(configuration(True), episode_limit=32)
    a = train_attacker(game, reference, 0, 27, cfg)
    b = train_attacker(game, reference, 0, 27, cfg, loss_kind="smooth_l1", reward_scale=100)
    assert model_hash(a[0].model) == model_hash(b[0].model)
    assert a[2]["rng"] == b[2]["rng"]


def small_training(root, game, reference, spec, case, arm, source_hash, seed=0):
    for seat in (0, 1):
        for rep in range(2):
            out = d.cell_root(root, case, arm, seed) / "train" / f"seat{seat}_rep{rep}"
            out.mkdir(parents=True)
            cell = cell_protocol(spec, case, arm, seed)
            d.diagnostic.train_worker(game, reference, cell, seat, rep, out, source_hash,
                training_seed=stream("training", case, seed, seat, rep), loss_kind=arm,
                reward_scale=cell["reward_scale"])


def test_real_smoke_controls_and_frozen_seed0_comparison(game, target, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "a"*64)
    spec = protocol(True)
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    original_hash = model_hash(reference.model)
    source_hash = target.metadata["checkpoint_sha256"]
    for case in CONTROLS:
        cg, ref = leduc_reference() if case == "leduc_uniform" else (game, ControlPolicy(reference.feature_encoder, case))
        for arm in ARMS:
            small_training(tmp_path, cg, ref, spec, case, arm, "test-control")
            output = d.cell_root(tmp_path, case, arm, 0) / "assessment"
            output.mkdir()
            args = Namespace(case=case, arm=arm, root=tmp_path, output=output)
            result = d.control_assessment(args, cg, ref, spec, "test-control")
            assert not result["capability_passed"] and len(result["records"]) == 2
    for arm in ARMS:
        small_training(tmp_path, game, reference, spec, "exp10", arm, source_hash)
    # Initial weights match between objectives for each seat/repetition.
    for seat in (0, 1):
        for rep in range(2):
            models = [d.policies(tmp_path, spec, reference, "exp10", arm, 0, rep, 0., source_hash)[seat] for arm in ARMS]
            assert model_hash(models[0].model) == model_hash(models[1].model)
    output = tmp_path / "assessment0"
    output.mkdir()
    args = Namespace(seed=0, arm=ARMS[0], eligible=",".join(ARMS), root=tmp_path, output=output)
    original = d.diagnostic.score
    def scoring(game, actors, ref, count, seed, events=None):
        if seed == stream("final", 0):
            assert (output / "selection_lock.json").exists()
        return original(game, actors, ref, count, seed, events)
    monkeypatch.setattr(d.diagnostic, "score", scoring)
    result = d.fhp_assessment(args, game, reference, spec, source_hash)
    assert len(result["curves"]) == 12 and result["primary_contrast"] is not None
    assert not result["qualified"] and model_hash(reference.model) == original_hash
    # Loading an attacker as the wrong objective must not pass the hash lock.
    with pytest.raises(ValueError, match="provenance"):
        d.diagnostic.load_attacker(d.cell_root(tmp_path, "exp10", ARMS[0], 0),
            cell_protocol(spec, "exp10", ARMS[1]), reference, 0, 0, .6, source_hash)


def test_selection_uses_development_and_eligibility_only():
    rows = [dict(arm=arm, repetition=rep, nominal_seconds=t, payoff_mbb=dict(mean=mean))
            for arm, rep, t, mean in [(ARMS[0], 1, 3600, 3), (ARMS[0], 0, 3600, 3),
                                      (ARMS[1], 0, 3600, 20), (ARMS[0], 0, 900, 9000)]]
    assert d.lock_selection(rows, list(ARMS), 3600)["arm"] == ARMS[1]
    chosen = d.lock_selection(rows, [ARMS[0]], 3600)
    assert chosen["arm"] == ARMS[0] and chosen["repetition"] == 0


def test_pooled_deals_not_pseudoreplicated():
    rows = [dict(seat_payoffs_mbb=[[v, v], [v+2, v+2], [v+4, v+4]]) for v in (0, 100)]
    result = d.pooled(rows)
    assert result["n"] == 3 and result["mean"] == 52


@pytest.mark.parametrize("controls_pass,qualifications,outcome", [
    (False, {}, "no_arm_passed_learning_controls"),
    (True, {0: False}, "locked_seed0_attacker_did_not_qualify"),
    (True, {0: True, 1: True, 2: False}, "attacker_strength_not_replicated_across_all_seeds"),
    (True, {0: True, 1: True, 2: True}, "replicated_candidate_requires_human_review"),
])
def test_pipeline_conditional_replication(tmp_path, monkeypatch, controls_pass, qualifications, outcome):
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "b"*64)
    calls = []
    def bounded(request, directory, **kwargs):
        calls.append(deepcopy(request))
        phase, seed = request["phase"], request["source_seed"]
        row = dict(request, smoke=False, status="complete")
        if phase == "preflight":
            row["passed"] = True
        elif phase == "control-assessment":
            row["capability_passed"] = controls_pass
        elif phase == "assessment":
            row.update(selected_arm=ARMS[1], qualified=qualifications[seed])
            if seed:
                assert (args.output / "replication_lock.json").exists()
        return dict(status="succeeded", result=row, elapsed_seconds=1, peak_rss_bytes=10, request=request)
    monkeypatch.setattr(d, "run_bounded", bounded)
    monkeypatch.setattr(d, "render_report", lambda s: s["outcome"])
    args = Namespace(smoke=False, output=tmp_path / "out", inputs=tmp_path / "in", native_repo=tmp_path, bucket="gs://test")
    d.run(args)
    result = json.loads((args.output / "analysis/summary.json").read_text())
    assert result["outcome"] == outcome and result["policy_repair_authorised"] is False
    seeds = {r["source_seed"] for r in calls if r["phase"] == "assessment"}
    assert seeds == (set() if not controls_pass else {0} if not qualifications[0] else {0, 1, 2})
    assert all(r["arm"] == ARMS[1] for r in calls if r["phase"] == "train" and r["source_seed"] > 0)
    with pytest.raises(FileExistsError):
        d.run(args)


def test_failed_correctness_stops_all_learning(tmp_path, monkeypatch):
    monkeypatch.setenv("FHP_BUNDLE_SHA256", "c"*64)
    calls = []
    def failure(request, directory, **kwargs):
        calls.append(request["phase"])
        return dict(status="failed")
    monkeypatch.setattr(d, "run_bounded", failure)
    args = Namespace(smoke=False, output=tmp_path / "out", inputs=tmp_path / "in", native_repo=tmp_path, bucket="gs://test")
    with pytest.raises(RuntimeError, match="preflight"):
        d.run(args)
    assert calls == ["preflight"] and not (args.output / "analysis/SUCCESS.json").exists()


@pytest.mark.parametrize("smoke", [False, True])
def test_batch_offline_bundle_and_ucv_wrapper(tmp_path, smoke):
    path = ROOT / "gcp/exp27_attacker_development_batch.py"
    mspec = importlib.util.spec_from_file_location("exp27_test_batch", path)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    native = ROOT.parent / "fhp_ucv_escher/fhp-ucv-escher-experiments"
    args = Namespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
        service_account="runner@test-project.iam.gserviceaccount.com", run_id="exp27-attack-test",
        smoke=smoke, native_repo=native, run_dir=tmp_path / "prepared")
    module.validate(args)
    job = module.build_job(args, "a"*64)
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    task = group["taskSpec"]
    assert task["maxRetryCount"] == 0 and task["computeResource"]["cpuMilli"] == 8000
    assert task["maxRunDuration"] == ("5400s" if smoke else "28800s")
    for runnable in task["runnables"]:
        text = runnable["script"]["text"]
        subprocess.run(["bash", "-n"], input=text, text=True, check=True)
        assert "gcloud batch" not in text and "/workspace/fhp-exp26" not in text
    script = task["runnables"][0]["script"]["text"]
    assert "adversarial.development run" in script and "adversarial.pipeline" not in script
    assert ('1800 "$PYTHON"' if smoke else '25200 "$PYTHON"') in script
    command = [sys.executable, "-S", str(path), "prepare", "--native-repo", str(native),
               "--run-dir", str(args.run_dir), "--project", args.project, "--region", args.region,
               "--bucket", args.bucket, "--service-account", args.service_account, "--run-id", args.run_id]
    subprocess.run(command + (["--smoke"] if smoke else []), capture_output=True, text=True, check=True)
    req = json.loads((args.run_dir / "request.json").read_text())
    assert req["protocol"] == protocol(smoke) and req["files"]["source.tar.gz"] == req["bundle_sha256"]
    env = dict(os.environ, PROJECT_ID=args.project, REGION=args.region, BUCKET=args.bucket,
               SA_EMAIL=args.service_account, RUN_ID=args.run_id, FHP_EVAL_REPO=str(ROOT),
               EXP27_RUN_DIR=str(tmp_path / "wrapper"), PYTHON=sys.executable)
    wrapper = native / "gcp/run_exp27_attacker_development.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", str(wrapper), "prepare-smoke" if smoke else "prepare"],
                   cwd=native, env=env, capture_output=True, text=True, check=True)
    assert json.loads((tmp_path / "wrapper/request.json").read_text())["smoke"] == smoke
