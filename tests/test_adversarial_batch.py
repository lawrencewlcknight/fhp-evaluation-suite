"""Offline job preparation, source isolation and phase safety."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
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
    assert "evaluator/fhp_evaluation/adversarial/pipeline.py" in names
    assert not any("secret" in n or "cloud_outputs" in n for n in names)


@pytest.mark.parametrize("smoke", [False, True])
def test_single_job_resources_shell_and_no_child_submissions(tmp_path, smoke):
    job = batch.build_pipeline_job(arguments(tmp_path, smoke), "a" * 64)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    assert group["taskCount"] == group["parallelism"] == group["taskCountPerNode"] == 1
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "93600s"
    assert spec["computeResource"] == dict(cpuMilli=48000, memoryMib=180000)
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-48" and policy["bootDisk"]["sizeGb"] == 200
    assert spec["environment"]["variables"]["FHP_SMOKE"] == str(int(smoke))
    main, final = spec["runnables"]
    assert final["alwaysRun"]
    for runnable in (main, final):
        script = runnable["script"]["text"]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "gcloud batch" not in script
        assert "FHP_TASK" not in script and "FHP_RUN_DESTINATION" not in script
        assert '"$OUTPUT" "$FHP_DESTINATION"' in script
        assert '"$FHP_DESTINATION/analysis"' not in script
        assert "scientific_complete" in script and "sync_outputs 1" in script
    assert "fhp_evaluation.adversarial.pipeline run" in main["script"]["text"]
    assert "--smoke-only" in main["script"]["text"]
    assert "final_system_diagnostics" in final["script"]["text"]


def test_single_job_preflight_needs_fresh_namespace_not_external_smoke(monkeypatch, tmp_path):
    calls = []
    def fake(*command):
        calls.append(command)
        return "[]" if "list" in command else "{}"
    monkeypatch.setattr(batch, "read", fake)
    batch.preflight(arguments(tmp_path), {"bundle_sha256": "a" * 64}, "run")
    assert not any("cat" in command for command in calls)
    assert any("gs://test-bucket/exp21-adv-test/**" in command for command in calls)
    monkeypatch.setattr(batch, "read", lambda *command: '[{"name":"existing"}]' if "list" in command else "{}")
    with pytest.raises(ValueError, match="occupied"):
        batch.preflight(arguments(tmp_path), {}, "run")


def test_run_prepares_uploads_and_submits_once(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    for package in ("fhp_escher", "vr_deep_cfr"):
        (args.native_repo / package).mkdir(parents=True)
        (args.native_repo / package / "__init__.py").write_text("")
    original_run = subprocess.run
    calls = []
    def fake_run(command, **kwargs):
        if command[0] == "gcloud":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)
    monkeypatch.setattr(batch, "read", lambda *command: "[]" if "list" in command else "{}")
    monkeypatch.setattr(subprocess, "run", fake_run)
    argv = ["run", "--project", args.project, "--region", args.region, "--bucket", args.bucket,
            "--service-account", args.service_account, "--run-id", args.run_id,
            "--native-repo", str(args.native_repo), "--run-dir", str(args.run_dir)]
    batch.main(argv)
    request = json.loads((args.run_dir / "request.json").read_text())
    assert request["layout"] == "single_job_v1" and not request["smoke"]
    assert set(request["files"]) == {"source.tar.gz", "run_job.json"}
    submissions = [c for c in calls if c[1:4] == ["batch", "jobs", "submit"]]
    assert len(submissions) == 1 and submissions[0][4] == "exp21-adv-test-run"
    uploads = [c for c in calls if c[1:3] == ["storage", "cp"]]
    assert len(uploads) == 3 and all("--if-generation-match=0" in c for c in uploads)
    (args.run_dir / "run_job.json").write_text("{}")
    with pytest.raises(ValueError, match="files have changed"):
        batch.main(argv)
    assert len([c for c in calls if "submit" in c]) == 1
    with pytest.raises(ValueError, match="layout differs"):
        batch.main(["qualify", *argv[1:]])


@pytest.mark.parametrize("final,status,scientific,smoke_done,sync_exit,markers", [
    (False, "complete", True, True, 0, []),
    (True, "failed", False, False, 0, []),
    (True, "inconclusive_qualification", False, True, 0, ["smoke/analysis/SUCCESS.json"]),
    (True, "complete", False, True, 0, ["smoke/analysis/SUCCESS.json"]),
    (True, "complete", True, True, 0, ["smoke/analysis/SUCCESS.json", "analysis/SUCCESS.json"]),
    (True, "complete", True, True, 1, []),
])
def test_success_markers_published_only_after_final_sync(tmp_path, final, status, scientific,
                                                       smoke_done, sync_exit, markers):
    for name in ("analysis", "smoke/analysis"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "SUCCESS.json").write_text("{}")
    (tmp_path / "STATUS.json").write_text(json.dumps(dict(
        status=status, scientific_complete=scientific,
        completed_stages=["smoke/aggregate"] if smoke_done else [])))
    # Stub only external commands; exercise the generated Bash function and its
    # actual Python publication guard. This test makes no GCS writes.
    script = f'''set -Eeuo pipefail
WORK={shlex.quote(str(tmp_path))}
OUTPUT="$WORK"
FHP_DESTINATION=gs://test/exp21
flock() {{ shift 3; "$@"; }}
timeout() {{ shift; "$@"; }}
gcloud() {{
  printf '%s\\n' "$*" >> "$WORK/calls.txt"
  if [[ "$2" == rsync ]]; then return {sync_exit}; fi
}}
''' + batch.pipeline_upload_function() + f"\nsync_outputs {int(final)}\n"
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (run.returncode == 0) == (sync_exit == 0), run.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 1 + len(markers)
    for call, marker in zip(calls[1:], markers):
        assert call.endswith(f"gs://test/exp21/{marker}")
