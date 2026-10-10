"""Offline job preparation, source isolation and phase safety."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp21_batch", ROOT / "gcp/exp21_adversarial_preservation_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def arguments(tmp_path, smoke=False):
    return argparse.Namespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                              service_account="batch@test-project.iam.gserviceaccount.com",
                              run_id="exp21-adv-test", smoke=smoke, smoke_run_id=None,
                              native_repo=tmp_path / "native", run_dir=tmp_path / "prepared")


@pytest.mark.parametrize("stage,count,cpus", [("qualify", 3, 16), ("train", 3, 16), ("evaluate", 12, 4), ("aggregate", 1, 4)])
def test_generated_jobs_and_shell(tmp_path, stage, count, cpus):
    args = arguments(tmp_path)
    job = batch.build_job(args, "a" * 64, stage)
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == count
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == f"n2-standard-{cpus}"
    for runnable in group["taskSpec"]["runnables"]:
        subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "best_response.pilot" not in script
    assert "FHP_CHECKPOINT_URI" not in script
    assert "/workspace/fhp-br-exp9" not in script
    assert group["taskSpec"]["runnables"][1]["alwaysRun"]
    if stage == "train":
        assert "--replay" in script and "qualification_download" in script
    if stage == "evaluate":
        assert "FHP_TASK / 4" in script and "FHP_TASK % 4" in script


def test_full_qualification_requires_smoke(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    def fake(*command):
        return "[]" if "list" in command else "{}"
    monkeypatch.setattr(batch, "read", fake)
    with pytest.raises(ValueError, match="SMOKE_RUN_ID"):
        batch.preflight(args, {"bundle_sha256": "abc"}, "qualify")


def test_train_denied_for_inconclusive_gate(monkeypatch, tmp_path):
    import hashlib
    args = arguments(tmp_path)
    request = {"bundle_sha256": "abc"}
    row = json.dumps(dict(bundle_sha256="abc", gate_passed=False))
    def fake(*command):
        if "list" in command:
            return "[]"
        if "describe" in command and "jobs" in command:
            return '{"status":{"state":"SUCCEEDED"}}'
        if "cat" in command:
            if command[-1].endswith("request.json"):
                return json.dumps(request)
            if command[-1].endswith("SUCCESS.json"):
                return json.dumps(dict(result_sha256=hashlib.sha256(row.encode()).hexdigest()))
            return row
        return "{}"
    monkeypatch.setattr(batch, "read", fake)
    with pytest.raises(ValueError, match="failed attacker qualification"):
        batch.preflight(args, request, "train")


def test_bundle_allowlist_contains_new_code_not_results(tmp_path):
    native = tmp_path / "native"
    for package in ("fhp_escher", "vr_deep_cfr"):
        (native / package).mkdir(parents=True)
        (native / package / "__init__.py").write_text("")
    (native / "secret.pkl").write_bytes(b"not source")
    archive = tmp_path / "bundle.tar.gz"
    batch.make_bundle(ROOT, native, archive, extra_files=("gcp/exp21_adversarial_preservation_batch.py",))
    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert "evaluator/fhp_evaluation/adversarial/learning.py" in names
    assert not any("secret" in n or "cloud_outputs" in n for n in names)
