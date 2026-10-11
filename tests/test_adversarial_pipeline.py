"""Stage barriers, provenance, cancellation and bounded aggregation; no cloud calls."""

import json
import threading

import pytest

from fhp_evaluation.adversarial import pipeline
from fhp_evaluation.adversarial.config import ARMS, configuration
from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.loaders import sha256_file

BUNDLE = "a" * 64


def result(directory, stage, seed=0, smoke=False, arm=None, **extra):
    row = dict(phase=stage, seed=seed, config_hash=configuration(smoke).fingerprint(),
               bundle_sha256=BUNDLE, status="smoke_complete" if smoke else "complete", **extra)
    if arm is not None:
        row["arm"] = arm
    write_json(directory / "result.json", row)
    write_json(directory / "SUCCESS.json", dict(result_sha256=sha256_file(directory / "result.json"),
               phase=stage, seed=seed, config_hash=row["config_hash"]))
    return row


@pytest.mark.parametrize("smoke_only", [False, True])
def test_order_barriers_and_completion(monkeypatch, tmp_path, smoke_only):
    monkeypatch.setenv("FHP_BUNDLE_SHA256", BUNDLE)
    calls = []
    def group(stage, smoke, root, inputs, native, bucket, bundle, progress):
        calls.append((smoke, stage))
        assert root == (tmp_path / "out/smoke" if smoke else tmp_path / "out")
        assert inputs == tmp_path / "input" / ("smoke" if smoke else "full")
        count = (1 if smoke else 3) * (4 if stage == "evaluate" else 1)
        progress(stage, count, count)
        return {i: {"gate_passed": not smoke} for i in range(count)}
    monkeypatch.setattr(pipeline, "group", group)
    monkeypatch.setattr(pipeline, "aggregate_stage", lambda smoke, *args: calls.append((smoke, "aggregate")))
    state = pipeline.run(tmp_path / "out", tmp_path / "input", tmp_path / "native", "gs://test",
                         smoke_only=smoke_only)
    expected = [(smoke, stage) for smoke in ([True] if smoke_only else [True, False])
                for stage in ("qualify", "train", "evaluate", "aggregate")]
    assert calls == expected
    assert state["complete"] and state["scientific_complete"] == (not smoke_only)
    assert state == json.loads((tmp_path / "out/STATUS.json").read_text())
    with pytest.raises(FileExistsError):
        pipeline.run(tmp_path / "out", tmp_path / "input", tmp_path / "native", "gs://test")


@pytest.mark.parametrize("failure", ["smoke_error", "full_error", "gate", "aggregate"])
def test_failed_or_inconclusive_phases_never_unlock_successors(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("FHP_BUNDLE_SHA256", BUNDLE)
    calls = []
    def group(stage, smoke, root, inputs, native, bucket, bundle, progress):
        calls.append((smoke, stage))
        progress(stage, 0, 1 if smoke else 3)
        if (failure == "smoke_error" and smoke and stage == "train" or
                failure == "full_error" and not smoke and stage == "train"):
            raise RuntimeError("worker failed")
        return {i: {"gate_passed": not (failure == "gate" and i == 1)} for i in range(1 if smoke else 3)}
    def aggregate(smoke, *args):
        calls.append((smoke, "aggregate"))
        if failure == "aggregate" and smoke:
            raise RuntimeError("aggregate failed")
    monkeypatch.setattr(pipeline, "group", group)
    monkeypatch.setattr(pipeline, "aggregate_stage", aggregate)
    with pytest.raises(RuntimeError):
        pipeline.run(tmp_path / "out", tmp_path / "input", tmp_path / "native", "gs://test")
    state = json.loads((tmp_path / "out/STATUS.json").read_text())
    assert not state["complete"] and not state["scientific_complete"]
    assert state["status"] == ("inconclusive_qualification" if failure == "gate" else "failed")
    if failure in ("smoke_error", "aggregate"):
        assert not any(not smoke for smoke, stage in calls)
    elif failure == "gate":
        assert (False, "train") not in calls
    else:
        assert (False, "evaluate") not in calls


@pytest.mark.parametrize("smoke,stage,count", [(True, "qualify", 1), (True, "evaluate", 4),
                                               (False, "train", 3), (False, "evaluate", 12)])
def test_group_counts_and_waits_for_every_worker(monkeypatch, tmp_path, smoke, stage, count):
    barrier = threading.Barrier(count, timeout=5)
    def task(stage, index, *args):
        barrier.wait()
        return index
    monkeypatch.setattr(pipeline, "task", task)
    progress = []
    rows = pipeline.group(stage, smoke, tmp_path, tmp_path, tmp_path, "gs://test", BUNDLE,
                          lambda *args: progress.append(args))
    assert rows == dict(enumerate(range(count)))
    assert progress[0] == (stage, 0, count) and progress[-1] == (stage, count, count)


def test_worker_error_cancels_siblings(monkeypatch, tmp_path):
    started = threading.Barrier(3, timeout=5)
    stopped = []
    def task(stage, index, smoke, root, inputs, native, bucket, bundle, cancel):
        started.wait()
        if index == 0:
            raise RuntimeError("first worker failed")
        assert cancel.wait(timeout=5)
        stopped.append(index)
    monkeypatch.setattr(pipeline, "task", task)
    with pytest.raises(RuntimeError, match="first worker"):
        pipeline.group("train", False, tmp_path, tmp_path, tmp_path, "gs://test", BUNDLE, lambda *args: None)
    assert sorted(stopped) == [1, 2]


@pytest.mark.parametrize("stage,index", [("qualify", 2), ("train", 2), ("evaluate", 11)])
def test_task_handoffs_exports_and_watchdog(monkeypatch, tmp_path, stage, index):
    seed = index // 4 if stage == "evaluate" else index
    arm = ARMS[index % 4] if stage == "evaluate" else None
    prior_stage = "qualify" if stage == "train" else "train"
    prior = tmp_path / prior_stage / f"task_{seed}" / "analysis"
    if stage != "qualify":
        result(prior, prior_stage, seed)
    cancel = threading.Event()
    def bounded(spec, directory, **kwargs):
        assert kwargs["cancel_event"] is cancel
        assert kwargs["max_seconds"] == pipeline.LIMITS[stage]
        assert kwargs["max_rss_mb"] == (13000 if stage == "evaluate" else 56000)
        command = kwargs["command"]
        assert command[2] == "fhp_evaluation.adversarial.workflow" and "--worker" in command
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--source") + 1] == str(tmp_path / "input" / f"seed_{seed}")
        if stage == "train":
            assert command[command.index("--gate") + 1] == str(prior / "result.json")
            assert command[command.index("--replay") + 1] == str(prior / "measurement/replay")
        if stage == "evaluate":
            assert command[command.index("--training") + 1] == str(prior)
            assert command[command.index("--arm") + 1] == "preserved"
        extra = {}
        directory.mkdir(parents=True)
        if stage == "train":
            hashes = {}
            for name in ARMS:
                path = directory / f"{name}.pkl"
                path.write_bytes(name.encode())
                hashes[name] = sha256_file(path)
            extra["policy_hashes"] = hashes
        result(directory, stage, seed, arm=arm, **extra)
        return {"status": "succeeded"}
    monkeypatch.setattr(pipeline, "run_bounded", bounded)
    pipeline.task(stage, index, False, tmp_path, tmp_path / "input", tmp_path / "native",
                  "gs://test", BUNDLE, cancel)
    output = tmp_path / stage / f"task_{index}" / "analysis"
    pipeline.checked_result(output, stage, seed, False, BUNDLE, arm)
    if stage == "train":
        assert all((output / f"{name}.pkl").read_bytes() == name.encode() for name in ARMS)


@pytest.mark.parametrize("mutation", ["checksum", "seed", "config_hash", "bundle_sha256", "status", "arm", "marker"])
def test_result_validation_rejects_corruption_or_mixed_provenance(tmp_path, mutation):
    extra = {mutation: "wrong"} if mutation not in ("checksum", "marker") else {}
    row = result(tmp_path, "evaluate", arm="unchanged")
    row.update(extra)
    write_json(tmp_path / "result.json", row)
    marker = json.loads((tmp_path / "SUCCESS.json").read_text())
    marker["result_sha256"] = "wrong" if mutation == "checksum" else sha256_file(tmp_path / "result.json")
    if mutation == "marker":
        marker["seed"] = 2
    write_json(tmp_path / "SUCCESS.json", marker)
    with pytest.raises(ValueError):
        pipeline.checked_result(tmp_path, "evaluate", 0, False, BUNDLE, "unchanged")


def test_real_bounded_aggregate_worker(tmp_path):
    """Exercise CLI, subprocess watchdog and real aggregator with small synthetic data."""
    scores = [1.0] * configuration(True).test_pairs
    for index, arm in enumerate(ARMS):
        directory = tmp_path / "evaluate" / f"task_{index}" / "analysis"
        result(directory, "evaluate", smoke=True, arm=arm,
               challenger_matrix={a: dict(selection_mean_mbb=1., pair_payoffs_mbb=scores) for a in ARMS},
               h2h_vs_unchanged=dict(pair_payoffs_mbb=scores),
               specialised_response=dict(pair_payoffs_mbb=dict(lbr=scores, full_flop=scores)),
               rule_panel={}, frozen_ucv_panel={})
    pipeline.aggregate_stage(True, tmp_path, tmp_path / "input", BUNDLE)
    assert json.loads((tmp_path / "analysis/SUCCESS.json").read_text())["status"] == "smoke_complete"
    assert json.loads((tmp_path / "aggregation_diagnostics/run.json").read_text())["status"] == "succeeded"
