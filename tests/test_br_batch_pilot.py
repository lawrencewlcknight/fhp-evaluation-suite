import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from fhp_evaluation.best_response import pilot


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("br_batch", ROOT / "gcp/exp4_ucv_exp9_br_pilot_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def args(native):
    return argparse.Namespace(run_id="fhp-br-exp9-test", project="clever-overview-399515",
                              region="europe-west1", bucket="gs://example-bucket",
                              service_account="batch@example-project.iam.gserviceaccount.com",
                              native_repo=native, smoke=False)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "native"
    for package in ("fhp_escher", "vr_deep_cfr"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("# source\n")
    (root / "secret.json").write_text("DO NOT UPLOAD")
    (root / "checkpoint.pkl").write_bytes(b"NOT SOURCE")
    return root


def test_batch_job_bounds_and_checkpoint_identity(native):
    config = args(native)
    batch.validate(config)
    job = batch.build_job(config, "0" * 64)
    task = job["taskGroups"][0]
    assert task["taskCount"] == task["parallelism"] == task["taskCountPerNode"] == 1
    spec = task["taskSpec"]
    assert spec["maxRetryCount"] == 0
    assert spec["maxRunDuration"] == "7200s"
    assert spec["runnables"][0]["timeout"] == "6300s"
    assert spec["runnables"][1]["alwaysRun"]
    assert sum(int(r["timeout"][:-1]) for r in spec["runnables"]) < 7200
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    variables = spec["environment"]["variables"]
    assert variables["FHP_CHECKPOINT_SHA256"] == pilot.CHECKPOINT_SHA256 == batch.CHECKPOINT_SHA256
    assert batch.CHECKPOINT_RELATIVE == pilot.CHECKPOINT_RELATIVE
    assert variables["FHP_CHECKPOINT_URI"].endswith(pilot.CHECKPOINT_RELATIVE)
    for runnable in spec["runnables"]:
        subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)


@pytest.mark.parametrize("key,value", [("run_id", "exp9-cache24-20261001-132550"),
    ("run_id", "fhp-br-exp9-bad;echo"), ("run_id", "fhp-br-exp9-trailing-"), ("bucket", "gs://bucket/prefix"),
    ("region", "europe-west1-a"), ("service_account", "bad@example.com")])
def test_reject_unsafe_configuration(native, key, value):
    config = args(native)
    setattr(config, key, value)
    with pytest.raises(ValueError):
        batch.validate(config)


def test_bundle_allowlist_hashes_and_symlink_rejection(native, tmp_path):
    destination = tmp_path / "source.tar.gz"
    manifest = batch.make_bundle(ROOT, native, destination)
    with tarfile.open(destination) as archive:
        assert not any("secret" in name or "checkpoint.pkl" in name or "/.git/" in name
                       for name in archive.getnames())
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    (native / "fhp_escher" / "leak.py").symlink_to(native / "secret.json")
    with pytest.raises(ValueError, match="symlinked"):
        batch.make_bundle(ROOT, native, tmp_path / "other.tar.gz")


def test_dry_run_no_cloud_calls_and_refuses_overwrite(native, tmp_path, monkeypatch):
    output = tmp_path / "dry"
    monkeypatch.setattr(batch, "cloud_preflight", lambda _: pytest.fail("dry-run made cloud calls"))
    monkeypatch.setattr(sys, "argv", ["builder", "dry-run", "--native-repo", str(native),
        "--project", "clever-overview-399515", "--region", "europe-west1",
        "--bucket", "example-bucket", "--service-account", "batch@example-project.iam.gserviceaccount.com",
        "--run-id", "fhp-br-exp9-test", "--output-dir", str(output)])
    batch.main()
    assert (output / "job.json").is_file()
    assert (output / "request.json").is_file()
    with pytest.raises(FileExistsError):
        batch.main()


@pytest.mark.parametrize("occupied,denied", [(False, False), (True, False), (False, True)])
def test_cloud_preflight_fails_closed(native, monkeypatch, occupied, denied):
    calls = []
    def fake_run(command, **_):
        calls.append(command)
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            return SimpleNamespace(stdout=json.dumps(dict(size=1000)))
        if command[:4] == ("gcloud", "storage", "objects", "list"):
            assert command[4] == "gs://example-bucket/fhp-br-exp9-test/**"
            assert "--exhaustive" in command and "--limit=1" in command
            assert "--format=json" in command
            if denied:
                raise subprocess.CalledProcessError(1, command, stderr="403 denied")
            return SimpleNamespace(stdout=json.dumps([dict(name="existing")] if occupied else []))
        return SimpleNamespace(stdout="[]")
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    if denied:
        with pytest.raises(subprocess.CalledProcessError):
            batch.cloud_preflight(args(native))
    elif occupied:
        with pytest.raises(ValueError, match="namespace"):
            batch.cloud_preflight(args(native))
    else:
        batch.cloud_preflight(args(native))
    assert not any("submit" in c or "cp" in c for c in calls)
    assert not any("print-access-token" in c for c in calls)


@pytest.mark.parametrize("response", ["", "null", "{}", "not-json"])
def test_namespace_check_rejects_invalid_output(native, monkeypatch, response):
    def fake_run(command, **_):
        if command[:4] == ("gcloud", "storage", "objects", "list"):
            return SimpleNamespace(stdout=response)
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            return SimpleNamespace(stdout='{"size":1000}')
        return SimpleNamespace(stdout="[]")
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    with pytest.raises(ValueError):
        batch.cloud_preflight(args(native))


def test_namespace_check_does_not_use_python_https(native, monkeypatch):
    import urllib.request
    def forbidden(*_, **__):
        pytest.fail("Preflight bypassed gcloud's TLS/authentication transport")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(batch.subprocess, "run", lambda command, **_: SimpleNamespace(
        stdout='{"size":1000}' if command[:4] == ("gcloud", "storage", "objects", "describe") else "[]"))
    batch.cloud_preflight(args(native))


def fake_worker(seen, statuses=None):
    def run(spec, directory, **limits):
        directory = Path(directory)
        seen.append((directory.name, spec, limits))
        record = dict(status=(statuses or {}).get(directory.name, "succeeded"), elapsed_seconds=1.,
                      peak_rss_bytes=100, request=dict(identity=dict(spec=spec)),
                      result=dict(threads=spec["threads"], median_seconds_per_board=1 / spec["threads"]))
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record))
        return record
    return run


def test_pilot_schedule_and_selection(tmp_path):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"fixture")
    seen = []
    result = pilot.run_pilot(checkpoint, tmp_path, tmp_path / "out",
        expected_sha256=batch.sha256(checkpoint), runner=fake_worker(seen, {"profile_threads_4": "time_limit"}))
    assert result["selected_threads"] == 8
    assert result["profile_timeouts"] == [4]
    assert [row[0] for row in seen] == ["validation", "smoke", "profile_threads_1", "profile_threads_2",
                                        "profile_threads_4", "profile_threads_8", "comparison"]
    assert seen[-1][1]["deals"] == 25
    assert seen[-1][1]["rollouts"] == 4096
    assert seen[-1][2]["max_seconds"] == 1800
    assert all(row[2]["max_rss_mb"] == 8192 for row in seen)
    assert (tmp_path / "out" / "SUCCESS.json").is_file()


@pytest.mark.parametrize("failure_stage,status", [("validation", "failed"), ("smoke", "time_limit"),
    ("profile_threads_1", "memory_limit"), ("comparison", "time_limit")])
def test_pilot_failure_does_not_produce_success(tmp_path, failure_stage, status):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"fixture")
    seen = []
    with pytest.raises(RuntimeError):
        pilot.run_pilot(checkpoint, tmp_path, tmp_path / "out", expected_sha256=batch.sha256(checkpoint),
                        runner=fake_worker(seen, {failure_stage: status}))
    assert seen[-1][0] == failure_stage
    assert not (tmp_path / "out" / "SUCCESS.json").exists()
    assert json.loads((tmp_path / "out" / "pilot_manifest.json").read_text())["status"] == "failed"


def test_checkpoint_mismatch_stops_before_any_work(tmp_path):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"wrong")
    seen = []
    with pytest.raises(ValueError, match="SHA-256"):
        pilot.run_pilot(checkpoint, tmp_path, tmp_path / "out", runner=fake_worker(seen))
    assert not seen


def test_all_profiles_timed_out():
    with pytest.raises(RuntimeError, match="No profile"):
        pilot.select_threads([dict(status="time_limit")])
