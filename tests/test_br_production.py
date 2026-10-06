import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from fhp_evaluation.best_response import production
from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.loaders import sha256_file
from fhp_evaluation.game import MILLI_BIG_BLINDS_PER_CHIP


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "br_production_batch", ROOT / "gcp/exp5_ucv_exp9_br_production_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)
BUNDLE_SHA = "a" * 64


def builder_args(native, *, smoke=False):
    return argparse.Namespace(run_id="fhp-br-exp9-prod-test", project="clever-overview-399515",
                              region="europe-west1", bucket="gs://example-bucket",
                              service_account="batch@example-project.iam.gserviceaccount.com",
                              native_repo=native, smoke=smoke)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "native"
    for package in ("fhp_escher", "vr_deep_cfr"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("# source\n")
    (root / "secret.json").write_text("DO NOT UPLOAD")
    return root


def test_frozen_schedule_and_checkpoint_contract():
    assert production.task_schedule() == tuple((seed, shard) for seed in (0, 1, 2) for shard in range(4))
    assert production.TOTAL_PAIRS_PER_SEED == 2000
    assert production.ROLLOUTS == 4096
    assert production.THREADS == 2
    assert 2026100402 not in production.SHARD_EVALUATION_SEEDS
    assert batch.TASK_COUNT == len(production.task_schedule()) == 12
    assert batch.SOURCE_RUN == production.SOURCE_RUN
    for seed, (relative, digest) in batch.CHECKPOINTS.items():
        assert production.CHECKPOINTS[seed] == {"relative": relative, "sha256": digest}


def test_batch_jobs_are_bounded_parallel_and_syntax_checked(native):
    args = builder_args(native)
    batch.validate(args, preparing=True)
    workers = batch.build_job(args, BUNDLE_SHA, "workers")
    group = workers["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 12
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "14400s"
    assert group["taskSpec"]["computeResource"] == {"cpuMilli": 2000, "memoryMib": 7000}
    assert workers["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-2"
    variables = group["taskSpec"]["environment"]["variables"]
    for seed in production.TRAINING_SEEDS:
        assert variables[f"FHP_CHECKPOINT_SHA256_{seed}"] == production.CHECKPOINTS[seed]["sha256"]
    aggregate = batch.build_job(args, BUNDLE_SHA, "aggregate")
    assert aggregate["taskGroups"][0]["taskCount"] == 1
    for job in (workers, aggregate):
        for runnable in job["taskGroups"][0]["taskSpec"]["runnables"]:
            subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)


def test_source_bundle_is_allowlisted(native, tmp_path):
    destination = tmp_path / "source.tar.gz"
    manifest = batch.make_bundle(ROOT, native, destination)
    with tarfile.open(destination) as archive:
        names = archive.getnames()
        assert "evaluator/fhp_evaluation/best_response/production.py" in names
        assert "evaluator/gcp/run_exp5_ucv_exp9_br_production.sh" in names
        assert not any("secret" in name or "/.git/" in name for name in names)
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest


def test_recovery_preflight_requires_existing_worker_output(native, monkeypatch):
    args = builder_args(native)
    seen = []
    def read(*command):
        seen.append(command)
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            return '{"size":303508}'
        if command[:4] == ("gcloud", "storage", "objects", "list"):
            return '[{"name":"partial"}]'
        if command[:4] == ("gcloud", "batch", "jobs", "list"):
            return '[]'
        return ""
    monkeypatch.setattr(batch, "_read", read)
    batch.cloud_preflight(args, "workers", "fhp-br-exp9-prod-test-r-r1", recovery=True)
    patterns = [call[4] for call in seen if call[:4] == ("gcloud", "storage", "objects", "list")]
    assert patterns == ["gs://example-bucket/fhp-br-exp9-prod-test/workers/**"]


def _write_complete_shard(root, task_index, smoke=False):
    seed, shard = production.task_schedule()[task_index]
    count = production.contract(smoke=smoke)["pairs_per_shard"]
    directory = root / production.task_name(task_index)
    identity = {
        "task_index": task_index, "task_name": production.task_name(task_index),
        "training_seed": seed, "shard_index": shard,
        "evaluation_seed": production.SHARD_EVALUATION_SEEDS[shard],
        "checkpoint_relative": production.CHECKPOINTS[seed]["relative"],
        "checkpoint_sha256": production.CHECKPOINTS[seed]["sha256"],
        "source_bundle_sha256": BUNDLE_SHA, "contract": production.contract(smoke=smoke),
    }
    base = seed * 10 + shard
    write_json(directory / "result.json", {
        "training_seed": seed, "shard_index": shard, "task_index": task_index,
        "protocol_version": production.PROTOCOL_VERSION,
        "source_bundle_sha256": BUNDLE_SHA,
        "evaluation_seed": production.SHARD_EVALUATION_SEEDS[shard],
        "search_seed": production.SEARCH_SEED,
        "preflop_rollouts": production.contract(smoke=smoke)["preflop_rollouts"],
        "num_deal_pairs": count,
        "pair_elapsed_seconds": [10.] * count,
        "peak_rss_bytes": 350 * 1024**2,
        "evaluation_identity": {"checkpoint_sha256": production.CHECKPOINTS[seed]["sha256"]},
        "pair_payoffs_mbb": {"lbr": [float(base)] * count,
                             "full_flop": [float(base + seed + 1)] * count},
        "seat_payoffs_chips": {"lbr": [[base / MILLI_BIG_BLINDS_PER_CHIP] * count] * 2,
                               "full_flop": [[(base + seed + 1) / MILLI_BIG_BLINDS_PER_CHIP] * count] * 2},
    })
    write_json(directory / "SUCCESS.json", {"identity": identity, "status": "succeeded",
               "completed_pairs": count, "result_sha256": sha256_file(directory / "result.json")})


def test_aggregation_uses_training_seeds_as_primary_inference_unit(tmp_path):
    workers = tmp_path / "workers"
    for task_index in range(12):
        _write_complete_shard(workers, task_index, smoke=True)
    result = production.aggregate(workers, tmp_path / "analysis", BUNDLE_SHA, smoke=True)
    primary = result["primary_training_seed_inference"]
    assert primary["n"] == 3
    assert primary["mean"] == pytest.approx(2.0)
    assert primary["ci_method"] == "Student_t_over_three_training_seed_means"
    assert result["secondary_pair_level_conditional_summary"]["paired_improvement"]["n"] == 8
    assert result["total_hands_both_responders"] == 96
    assert result["comparison_worker_seconds"] == 240
    assert (tmp_path / "analysis" / "shard_timings.csv").is_file()
    assert (tmp_path / "analysis" / "seed_summary.csv").is_file()
    assert (tmp_path / "analysis" / "report.md").is_file()
    assert (tmp_path / "analysis" / "SUCCESS.json").is_file()


def test_aggregation_refuses_missing_or_mixed_shards(tmp_path):
    workers = tmp_path / "workers"
    for task_index in range(12):
        _write_complete_shard(workers, task_index, smoke=True)
    marker = workers / production.task_name(3) / "SUCCESS.json"
    value = json.loads(marker.read_text())
    value["identity"]["source_bundle_sha256"] = "b" * 64
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        production.aggregate(workers, tmp_path / "analysis", BUNDLE_SHA, smoke=True)


def test_conditional_interval_preserves_common_deal_covariance(tmp_path):
    workers = tmp_path / "workers"
    for task_index in range(12):
        _write_complete_shard(workers, task_index, smoke=True)
        directory = workers / production.task_name(task_index)
        result = json.loads((directory / "result.json").read_text())
        # Perfectly correlated payoff changes across models: tripling the model
        # count must not shrink conditional Monte Carlo SE by sqrt(3).
        shard = result["shard_index"]
        values = [float(2 * shard), float(2 * shard + 1)]
        result["pair_payoffs_mbb"] = {"lbr": [0., 0.], "full_flop": values}
        result["seat_payoffs_chips"] = {"lbr": [[0., 0.], [0., 0.]],
            "full_flop": [[v / MILLI_BIG_BLINDS_PER_CHIP for v in values]] * 2}
        write_json(directory / "result.json", result)
        marker = json.loads((directory / "SUCCESS.json").read_text())
        marker["result_sha256"] = sha256_file(directory / "result.json")
        write_json(directory / "SUCCESS.json", marker)
    summary = production.aggregate(workers, tmp_path / "analysis", BUNDLE_SHA, smoke=True)
    expected = production._finite_summary(list(range(8)))
    observed = summary["secondary_pair_level_conditional_summary"]["paired_improvement"]
    assert observed == expected


@pytest.mark.parametrize("change", ["checksum", "counters", "pair_seat", "timing", "missing"])
def test_reject_corrupt_or_incomplete_shards(tmp_path, change):
    workers = tmp_path / "workers"
    for index in range(12):
        _write_complete_shard(workers, index, smoke=True)
    directory = workers / production.task_name(0)
    result = json.loads((directory / "result.json").read_text())
    if change == "missing":
        (directory / "SUCCESS.json").unlink()
    else:
        if change in ("checksum", "pair_seat"): result["pair_payoffs_mbb"]["full_flop"][0] += 1
        elif change == "counters": result["num_deal_pairs"] = 1
        else: result["pair_elapsed_seconds"] = [1.]
        write_json(directory / "result.json", result)
        if change != "checksum":
            marker = json.loads((directory / "SUCCESS.json").read_text())
            marker["result_sha256"] = sha256_file(directory / "result.json")
            write_json(directory / "SUCCESS.json", marker)
    output = tmp_path / "analysis"
    with pytest.raises((ValueError, FileNotFoundError)):
        production.aggregate(workers, output, BUNDLE_SHA, smoke=True)
    assert not (output / "SUCCESS.json").exists()


@pytest.mark.parametrize("stage,recovery", [("workers", True), ("aggregate", False)])
@pytest.mark.parametrize("state", ["RUNNING", "QUEUED", "SCHEDULED", "UNKNOWN"])
def test_refuse_concurrent_attempts(native, monkeypatch, stage, recovery, state):
    args = builder_args(native)
    def read(*command):
        if command[:3] == ("gcloud", "iam", "service-accounts"):
            return args.service_account
        assert command[:4] == ("gcloud", "batch", "jobs", "list")
        return json.dumps([{"name": f"projects/{args.project}/locations/{args.region}/jobs/{args.run_id}-workers",
                            "status": {"state": state}}])
    monkeypatch.setattr(batch, "_read", read)
    with pytest.raises(ValueError, match="concurrent"):
        batch.cloud_preflight(args, stage, args.run_id + "-r-r1", recovery=recovery)


def test_aggregate_requires_completed_batch_uploads(native, monkeypatch):
    args = builder_args(native)
    monkeypatch.setattr(batch, "_read", lambda *command: "[]")
    with pytest.raises(ValueError, match="successful workers"):
        batch.cloud_preflight(args, "aggregate", args.run_id + "-aggregate")


@pytest.mark.parametrize("changed", ["source.tar.gz", "workers_job.json", "aggregate_job.json"])
def test_prepared_files_are_immutable(tmp_path, changed):
    names = ("source.tar.gz", "workers_job.json", "aggregate_job.json")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    request = {"prepared_files_sha256": {name: batch.sha256(tmp_path / name) for name in names},
               "bundle_sha256": batch.sha256(tmp_path / "source.tar.gz")}
    batch.verify_prepared_files(tmp_path, request)
    (tmp_path / changed).write_bytes(b"edited")
    with pytest.raises(ValueError, match="checksum"):
        batch.verify_prepared_files(tmp_path, request)


def test_aggregate_runtime_installs_torch():
    script = (ROOT / "gcp/aggregate_exp5_ucv_exp9_br_production.sh").read_text()
    assert "torch==2.7.0+cpu" in script and "python install 3.11.13" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_shard_completion_reuse_and_result_integrity(tmp_path, monkeypatch):
    from fhp_evaluation.best_response import worker
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"synthetic transport fixture; no pickle loading")
    digest = sha256_file(checkpoint)
    monkeypatch.setitem(production.CHECKPOINTS, 0, {"relative": "fixture.pkl", "sha256": digest})
    monkeypatch.setattr(production, "identity", lambda spec: {"checkpoint_sha256": digest})
    calls = []
    def execute(spec, **kwargs):
        calls.append(spec)
        assert spec["deals"] == 2 and spec["threads"] == 2 and spec["record_pair_timings"]
        return {"num_deal_pairs": 2, "pair_elapsed_seconds": [1., 2.]}
    monkeypatch.setattr(worker, "execute", execute)
    output = tmp_path / "worker"
    marker = production.run_shard(0, checkpoint, tmp_path, output, BUNDLE_SHA, smoke=True)
    assert marker["result_sha256"] == sha256_file(output / "result.json")
    assert marker["completed_pairs"] == 2
    assert production.run_shard(0, checkpoint, tmp_path, output, BUNDLE_SHA, smoke=True) == marker
    assert len(calls) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["peak_rss_bytes"] > 0
    result["pair_elapsed_seconds"][0] += 1
    write_json(output / "result.json", result)
    with pytest.raises(ValueError, match="checksum"):
        production.run_shard(0, checkpoint, tmp_path, output, BUNDLE_SHA, smoke=True)


def test_failed_shard_has_no_success_marker(tmp_path, monkeypatch):
    from fhp_evaluation.best_response import worker
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"synthetic transport fixture")
    monkeypatch.setitem(production.CHECKPOINTS, 0, {"relative": "fixture.pkl", "sha256": sha256_file(checkpoint)})
    monkeypatch.setattr(production, "identity", lambda spec: {})
    def execute(*args, **kwargs):
        raise RuntimeError("fixture evaluation failure")
    monkeypatch.setattr(worker, "execute", execute)
    output = tmp_path / "worker"
    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        production.run_shard(0, checkpoint, tmp_path, output, BUNDLE_SHA, smoke=True)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "SUCCESS.json").exists()
