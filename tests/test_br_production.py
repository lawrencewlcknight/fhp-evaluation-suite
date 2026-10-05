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
    write_json(directory / "SUCCESS.json", {"identity": identity, "status": "succeeded"})
    base = seed * 10 + shard
    write_json(directory / "result.json", {
        "training_seed": seed, "shard_index": shard, "task_index": task_index,
        "protocol_version": production.PROTOCOL_VERSION,
        "source_bundle_sha256": BUNDLE_SHA,
        "evaluation_seed": production.SHARD_EVALUATION_SEEDS[shard],
        "search_seed": production.SEARCH_SEED,
        "preflop_rollouts": production.contract(smoke=smoke)["preflop_rollouts"],
        "evaluation_identity": {"checkpoint_sha256": production.CHECKPOINTS[seed]["sha256"]},
        "pair_payoffs_mbb": {"lbr": [float(base)] * count,
                             "full_flop": [float(base + seed + 1)] * count},
        "seat_payoffs_chips": {"lbr": [[0.0] * count, [0.0] * count],
                               "full_flop": [[0.0] * count, [0.0] * count]},
    })


def test_aggregation_uses_training_seeds_as_primary_inference_unit(tmp_path):
    workers = tmp_path / "workers"
    for task_index in range(12):
        _write_complete_shard(workers, task_index, smoke=True)
    result = production.aggregate(workers, tmp_path / "analysis", BUNDLE_SHA, smoke=True)
    primary = result["primary_training_seed_inference"]
    assert primary["n"] == 3
    assert primary["mean"] == pytest.approx(2.0)
    assert primary["ci_method"] == "Student_t_over_three_training_seed_means"
    assert result["secondary_pair_level_conditional_summary"]["paired_improvement"]["n"] == 24
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
