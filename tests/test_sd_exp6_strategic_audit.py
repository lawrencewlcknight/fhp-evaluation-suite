"""Offline SD-CFR audit contracts, full-archive transfers and staged launch gates."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from fhp_evaluation.strategic_audit import sd_exp6 as experiment
from fhp_evaluation.strategic_audit.library import board_schedule, canonical_board
from fhp_evaluation.strategic_audit.workflow import validate_spec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sd_audit_batch", ROOT / "gcp/sd_exp6_strategic_audit_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def args(native=None, stage="profile"):
    return argparse.Namespace(run_id="fhp-sd6-audit-test", project="clever-overview-399515",
                              region="europe-west1", bucket="gs://example-bucket",
                              service_account="runner@example-project.iam.gserviceaccount.com",
                              native_repo=native, stage=stage, previous_run_id=None, previous_reviewed=False,
                              skip_pilot=False, profile_run_dir=None)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "native"
    (root / "deep_cfr_poker").mkdir(parents=True)
    (root / "deep_cfr_poker/__init__.py").write_text("# source only\n")
    (root / "secret.json").write_text("do not upload")
    (root / "deep_cfr_poker/replay.pkl").write_bytes(b"not source")
    return root


@pytest.fixture(scope="module")
def libraries():
    return {stage: experiment.make_library(stage) for stage in experiment.STAGES}


def synthetic_result(stage="development"):
    spec = experiment.make_spec(stage, "/input", "/native")
    comparisons = {}
    for label in spec["candidates"]:
        meta = spec["policies"][label]
        for i, opponent in enumerate(spec["opponents"]):
            value = 10 * i + (48 - meta["hours"]) / 100 * (meta["training_seed"] + 1)
            for tilt in spec["range_tilts"]:
                comparisons[f"{label}__vs__{opponent}__tilt_{tilt:g}"] = dict(
                    mean_root_response_gap_bb=value,
                    boards=[dict(board=[j, j + 1, j + 2], core=True, root_count=i + 1)
                            for j in range(experiment.BOARD_COUNTS[stage])])
    return dict(status="succeeded", provenance=dict(spec=spec), comparisons=comparisons)


def test_prespecified_archives_and_sd_only_panel():
    profile, pilot, development = [experiment.checkpoint_registry(s) for s in experiment.STAGES]
    assert (len(profile), len(pilot), len(development)) == (1, 2, 6)
    assert set(profile) <= set(pilot) <= set(development)
    assert list(profile) == ["sd_exp6_seed0_48h"]
    assert {r["training_seed"] for r in development.values()} == {0, 1, 2}
    assert {r["hours"] for r in development.values()} == {24, 48}
    assert [r["completed_iterations"] for r in development.values()] == [17690, 35535, 17739, 35822, 18547, 37210]
    for stage in experiment.STAGES:
        config = experiment.make_spec(stage, "/input", "/native")
        assert len(config["opponents"]) == 5 and config["opponents"] == list(experiment.RULES)
        assert set(p["family"] for p in config["policies"].values()) == {"sd_cfr", "rule"}
        assert "reference_policy" not in config
        for label in config["candidates"]:
            assert config["policies"][label]["require_suit_invariance"]
            assert config["policies"][label]["sha256"] == development[label]["sha256"]
        assert "training_state" not in json.dumps(experiment.checkpoint_registry(stage))
    with pytest.raises(ValueError, match="locked"):
        experiment.checkpoint_registry("assessment")


def test_real_manifest_hash_is_required(native, tmp_path):
    config = experiment.make_spec("profile", tmp_path, native)
    path = Path(config["policies"]["sd_exp6_seed0_48h"]["checkpoint"])
    path.parent.mkdir()
    path.write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        validate_spec(config)


def test_nested_boards_and_locked_assessment(libraries):
    boards = {s: set(board_schedule(lib, "development")) for s, lib in libraries.items()}
    assert [len(boards[s]) for s in experiment.STAGES] == [1, 8, 64]
    assert boards["profile"] <= boards["pilot"] <= boards["development"]
    locked = {canonical_board(b) for b in libraries["development"]["assessment_board_classes"]}
    assert len(locked) == 32
    for lib in libraries.values():
        assert lib["reference_deals"] == []
        assert not locked.intersection(canonical_board(b) for b in board_schedule(lib, "development"))


def test_paired_seed_summary_not_position_or_opponent_pseudoreplication(tmp_path):
    summary = experiment.write_report(tmp_path, synthetic_result(), "development")
    primary = summary["paired_changes"]["48_minus_24h_tilt_0"]["equal_weight_panel"]
    assert primary["n"] == 3
    assert primary["mean"] == pytest.approx(-.48)
    assert list(primary["seed_values"].values()) == pytest.approx([-.24, -.48, -.72])
    assert primary["standard_error"] == pytest.approx(.24 / 3**.5)
    assert summary["endpoints"]["48h_tilt_0"]["equal_weight_panel"]["mean"] == 20
    assert len(summary["paired_changes"]) == 3
    assert (tmp_path / "sd_exp6_report.html").exists()
    assert (tmp_path / "panel_root_gap.svg").exists()
    pilot = experiment.temporal_summary(synthetic_result("pilot"), "pilot")
    assert pilot["paired_changes"][pilot["primary"]]["equal_weight_panel"]["descriptive_95pct_t_interval"] is None
    with pytest.raises(ValueError, match="feasibility"):
        experiment.temporal_summary({}, "profile")


@pytest.mark.parametrize("corruption", ["support", "seed_support", "board_count", "duplicate_board",
                                      "missing", "nonfinite", "panel", "checkpoint", "status"])
def test_summary_rejects_incomplete_or_changed_inputs(corruption):
    result = synthetic_result()
    key = "sd_exp6_seed0_48h__vs__candid_statistician__tilt_0"
    row = result["comparisons"][key]
    if corruption == "support":
        row["boards"][0]["root_count"] = 7
    elif corruption == "seed_support":
        for hour in (24, 48):
            result["comparisons"][f"sd_exp6_seed1_{hour}h__vs__candid_statistician__tilt_0"]["boards"][0]["root_count"] = 7
    elif corruption == "board_count":
        row["boards"].pop()
    elif corruption == "duplicate_board":
        row["boards"][1] = row["boards"][0]
    elif corruption == "missing":
        del result["comparisons"][key]
    elif corruption == "panel":
        result["provenance"]["spec"]["opponents"].pop()
    elif corruption == "checkpoint":
        result["provenance"]["spec"]["policies"]["sd_exp6_seed0_48h"]["sha256"] = "x" * 64
    elif corruption == "status":
        result["status"] = "failed"
    else:
        row["mean_root_response_gap_bb"] = float("nan")
    with pytest.raises((KeyError, ValueError)):
        experiment.temporal_summary(result, "development")


def tiny_manifest(count=2):
    chunks = [dict(path=f"chunk_{i:08d}_{i:08d}.npy", first_iteration=i, last_iteration=i,
                   sha256=hashlib.sha256(bytes([i])).hexdigest(), size_bytes=1) for i in range(1, count + 1)]
    return dict(format="sd_cfr_chunked_uniform_v1", weighting="uniform", completed_iterations=count,
                num_players=2, num_actions=3, embedding_size=183, chunks=chunks)


@pytest.mark.parametrize("corruption", ["missing", "gap", "overlap", "path", "size", "weighting", "count"])
def test_full_history_contract_rejects_corrupt_or_thinned_archives(corruption):
    manifest = tiny_manifest()
    record = dict(completed_iterations=2)
    assert experiment.archive_chunks(manifest, record) == manifest["chunks"]
    if corruption == "missing":
        manifest["chunks"].pop()
    elif corruption == "gap":
        manifest["chunks"][0]["first_iteration"] = 2
    elif corruption == "overlap":
        manifest["chunks"][1]["first_iteration"] = 1
    elif corruption == "path":
        manifest["chunks"][0]["path"] = "../../secret.npy"
    elif corruption == "size":
        manifest["chunks"][0]["size_bytes"] = 2**40
    elif corruption == "weighting":
        manifest["weighting"] = "linear"
    else:
        manifest["completed_iterations"] = 1
    with pytest.raises(ValueError):
        experiment.archive_chunks(manifest, record)


def test_transfer_complete_prefix_deduplicates_chunks_and_detects_tampering(tmp_path, monkeypatch):
    registry, payloads = {}, {}
    for hour, count in ((24, 1), (48, 2)):
        data = json.dumps(tiny_manifest(count)).encode()
        label = f"sd_exp6_seed0_{hour}h"
        registry[label] = dict(local=f"seed_0/time_{hour}h.json", run="source", relative=f"archive/time_{hour}h.json",
                               sha256=hashlib.sha256(data).hexdigest(), training_seed=0, completed_iterations=count)
        payloads[f"time_{hour}h.json"] = data
    payloads.update({f"chunk_{i:08d}_{i:08d}.npy": bytes([i]) for i in (1, 2)})
    calls = []
    def download(command, **kwargs):
        assert command[:3] == ["gcloud", "storage", "cp"]
        assert "--recursive" not in command and 0 < kwargs["timeout"] <= 600
        calls.append(command[3])
        Path(command[4]).write_bytes(payloads[command[3].rsplit("/", 1)[-1]])
    monkeypatch.setattr(experiment, "checkpoint_registry", lambda _: registry)
    monkeypatch.setattr(experiment.subprocess, "run", download)
    report = experiment.download_archives("pilot", tmp_path)
    assert report["unique_chunks"] == 2 and report["archive_bytes"] == 2
    assert len(calls) == 4  # two manifests + two unique chunks, not two archive trees
    experiment.download_archives("pilot", tmp_path)
    assert len(calls) == 4
    (tmp_path / "seed_0/chunk_00000001_00000001.npy").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="integrity"):
        experiment.download_archives("pilot", tmp_path)


def test_transfer_does_not_follow_an_unpinned_manifest(tmp_path, monkeypatch):
    registry = experiment.checkpoint_registry("profile")
    monkeypatch.setattr(experiment.subprocess, "run", lambda command, **kw: Path(command[4]).write_text("{}"))
    with pytest.raises(ValueError, match="integrity"):
        experiment.download_archives("profile", tmp_path)
    assert not (tmp_path / next(iter(registry.values()))["local"]).exists()
    assert not list(tmp_path.rglob("*.npy"))


def test_recovery_identity_uses_content_not_download_timestamps(native, tmp_path):
    from fhp_evaluation.strategic_audit.workflow import policy_identity
    path = tmp_path / "policy.json"
    manifest = tiny_manifest()
    path.write_text(json.dumps(manifest))
    for i, chunk in enumerate(manifest["chunks"], 1):
        (tmp_path / chunk["path"]).write_bytes(bytes([i]))
    item = dict(family="sd_cfr", checkpoint=str(path), sha256=experiment.sha256(path), repo_root=str(native))
    before = policy_identity(item)
    for chunk in manifest["chunks"]:
        os.utime(tmp_path / chunk["path"], (2000000000, 2000000000))
    assert policy_identity(item) == before
    assert "archive_file_stamps" not in before["native"]
    (tmp_path / manifest["chunks"][0]["path"]).write_bytes(b"changed")
    assert policy_identity(item) != before


@pytest.mark.parametrize("stage", ["pilot", "development"])
def test_stages_require_explicit_review(native, stage):
    config = args(native, stage)
    with pytest.raises(ValueError, match="previous-reviewed"):
        batch.validate(config, preparing=True)
    config.previous_run_id, config.previous_reviewed = "fhp-sd6-audit-prior", True
    batch.validate(config, preparing=True)
    config.previous_run_id = config.run_id
    with pytest.raises(ValueError, match="distinct"):
        batch.validate(config)


def test_inference_only_job_caps_and_shell_syntax(native):
    for stage in experiment.STAGES:
        config = args(native, stage)
        job = batch.build_job(config, "a" * 64)
        group = job["taskGroups"][0]
        assert group["taskCount"] == group["parallelism"] == 1
        task = group["taskSpec"]
        assert task["maxRetryCount"] == 0
        assert task["maxRunDuration"] == f"{experiment.JOB_HOURS[stage] * 3600}s"
        assert task["runnables"][1]["alwaysRun"]
        assert job["allocationPolicy"]["instances"][0]["policy"]["bootDisk"]["sizeGb"] == 200
        for runnable in task["runnables"]:
            subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)
        execution = experiment.execution_config(stage)
        assert execution["workers"] * experiment.THREADS <= 16
        assert execution["session_seconds"] + 7200 == experiment.JOB_HOURS[stage] * 3600
        assert "training_state" not in json.dumps(job)


def test_source_only_offline_prepare_and_immutable_request(native, tmp_path):
    directory = tmp_path / "prepared"
    command = [sys.executable, str(ROOT / batch.EXTRA_FILES[0]), "prepare", "--run-id", "fhp-sd6-audit-test",
               "--project", "clever-overview-399515", "--region", "europe-west1", "--bucket", "gs://example-bucket",
               "--service-account", "runner@example-project.iam.gserviceaccount.com",
               "--native-repo", str(native), "--run-dir", str(directory)]
    # No gcloud executable is available to this subprocess; prepare must be offline.
    subprocess.run(command, check=True, capture_output=True, text=True, env=dict(os.environ, PATH="/usr/bin:/bin"))
    request = batch.verify_request(args(native), directory)
    with tarfile.open(directory / "source.tar.gz") as archive:
        names = archive.getnames()
        assert "native/deep_cfr_poker/__init__.py" in names
        assert "evaluator/fhp_evaluation/strategic_audit/sd_exp6.py" in names
        assert not any("secret" in p or "replay.pkl" in p or "/.git/" in p for p in names)
        for name, digest in request["source"]["files"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    (directory / "job.json").write_text("{}")
    with pytest.raises(ValueError, match="Prepared file"):
        batch.verify_request(args(native), directory)


def test_skip_pilot_only_allows_reviewed_development(native):
    config = args(native)
    config.skip_pilot = True
    for stage in ("profile", "pilot"):
        config.stage = stage
        with pytest.raises(ValueError, match="only valid for development"):
            batch.validate(config, preparing=True)
    config.stage = "development"
    with pytest.raises(ValueError, match="previous-reviewed"):
        batch.validate(config, preparing=True)
    config.previous_run_id, config.previous_reviewed = "fhp-sd6-audit-profile", True
    batch.validate(config, preparing=True)


def test_direct_prepare_reuses_exact_profile_offline_and_records_skip(native, tmp_path):
    profile, development = tmp_path / "profile", tmp_path / "development"
    common = [sys.executable, str(ROOT / batch.EXTRA_FILES[0]), "prepare",
              "--project", "clever-overview-399515", "--region", "europe-west1",
              "--bucket", "gs://example-bucket", "--service-account",
              "runner@example-project.iam.gserviceaccount.com", "--native-repo", str(native)]
    env = dict(os.environ, PATH="/usr/bin:/bin")  # gcloud unavailable; both prepare steps are offline.
    subprocess.run(common + ["--run-id", "fhp-sd6-audit-profile", "--run-dir", str(profile)],
                   check=True, capture_output=True, text=True, env=env)
    # Current native edits must not be bundled into a direct launch.
    (native / "deep_cfr_poker/__init__.py").write_text("# unrelated later edit\n")
    command = common + ["--stage", "development", "--run-id", "fhp-sd6-audit-test",
                        "--run-dir", str(development), "--previous-run-id", "fhp-sd6-audit-profile",
                        "--previous-reviewed", "--skip-pilot", "--profile-run-dir", str(profile)]
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    assert (profile / "source.tar.gz").read_bytes() == (development / "source.tar.gz").read_bytes()
    config = args(native, "development")
    config.previous_run_id, config.previous_reviewed = "fhp-sd6-audit-profile", True
    config.skip_pilot, config.profile_run_dir = True, profile
    request = batch.verify_request(config, development)
    assert request["skip_pilot"] is True
    assert request["profile_request_sha256"] == experiment.sha256(profile / "request.json")
    original = json.loads((profile / "request.json").read_text())
    assert request["source"] == original["source"]
    job = json.loads((development / "job.json").read_text())
    task = job["taskGroups"][0]["taskSpec"]
    assert task["environment"]["variables"]["FHP_STAGE"] == "development"
    assert task["maxRunDuration"] == "259200s"
    original_job = json.loads((profile / "job.json").read_text())
    for new, old in zip(task["runnables"], original_job["taskGroups"][0]["taskSpec"]["runnables"]):
        assert new["script"] == old["script"]
    config.skip_pilot = False
    with pytest.raises(ValueError, match="skip_pilot"):
        batch.verify_request(config, development)
    config.skip_pilot = True
    # Recover accepts the same immutable request without requiring the local profile folder.
    config.profile_run_dir = None
    batch.verify_request(config, development)
    config.profile_run_dir = profile
    (profile / "source.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        batch.frozen_profile(config)


def test_sd_repository_wrapper_prepares_real_source_offline(tmp_path):
    native = os.environ.get("FHP_SD_REPO")
    if not native:
        pytest.skip("Set FHP_SD_REPO to check the user-facing wrapper")
    directory = tmp_path / "prepared"
    wrapper = Path(native) / "gcp/run_sd_exp6_strategic_audit.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    command = ["bash", str(wrapper), "prepare", "--run-id", "fhp-sd6-audit-wrapper",
               "--project", "clever-overview-399515", "--region", "europe-west1", "--bucket", "gs://example-bucket",
               "--service-account", "runner@example-project.iam.gserviceaccount.com", "--run-dir", str(directory)]
    subprocess.run(command, check=True, capture_output=True, text=True, cwd=native, timeout=30,
                   env=dict(os.environ, FHP_EVALUATION_REPO=str(ROOT), FHP_AUDIT_PYTHON=sys.executable,
                            PATH="/usr/bin:/bin"))
    with tarfile.open(directory / "source.tar.gz") as archive:
        names = archive.getnames()
        assert "native/deep_cfr_poker/sd_cfr_lbr.py" in names
        assert not any(p.startswith("native/fhp_evaluation/") for p in names)
        assert not any("cloud_outputs/" in p or "thesis/" in p or "training_state/" in p for p in names)
        assert all(p.endswith(".py") for p in names if p.startswith("native/"))


@pytest.mark.parametrize("stage,skip_pilot", [("pilot", False), ("development", False), ("development", True)])
def test_predecessor_gate_checks_stage_archives_runtime_and_source(monkeypatch, stage, skip_pilot):
    config = args(stage=stage)
    config.skip_pilot = skip_pilot
    config.previous_run_id, config.previous_reviewed = "fhp-sd6-audit-prior", True
    previous_stage = "profile" if stage == "pilot" or skip_pilot else "pilot"
    source = {"files": {"source.py": "digest"}}
    previous_text = json.dumps(dict(stage=previous_stage, source=source, bundle_sha256="a" * 64))
    request = dict(source=source, bundle_sha256="a" * 64,
                   profile_request_sha256=hashlib.sha256(previous_text.encode()).hexdigest())
    checkpoints = experiment.checkpoint_registry(previous_stage)
    runtime = dict(experiment=experiment.EXPERIMENT, stage=previous_stage,
                   execution=experiment.execution_config(previous_stage), full_archive_exports=[
                       dict(policy=k, models_per_player=v["completed_iterations"], board=[b, b + 1, b + 2])
                       for k, v in checkpoints.items() for b in range(experiment.BOARD_COUNTS[previous_stage])])
    runtime_text = json.dumps(runtime)
    success = dict(experiment=experiment.EXPERIMENT, status="succeeded", stage=previous_stage,
                   full_archive=True, assessment_evaluated=False,
                   checkpoint_sha256={k: v["sha256"] for k, v in checkpoints.items()},
                   report_sha256={"runtime_summary.json": hashlib.sha256(runtime_text.encode()).hexdigest()})
    def read(*command):
        name = command[-1].rsplit("/", 1)[-1]
        return {"SUCCESS.json": json.dumps(success), "runtime_summary.json": runtime_text,
                "request.json": previous_text}[name]
    monkeypatch.setattr(batch, "read", read)
    monkeypatch.setattr(batch, "list_jobs", lambda *a: [dict(status=dict(state="SUCCEEDED"))])
    batch.verify_previous(config, request)
    with pytest.raises(ValueError, match="Source changed"):
        batch.verify_previous(config, dict(request, source={"files": {}}))
    if skip_pilot:
        for key in ("bundle_sha256", "profile_request_sha256"):
            with pytest.raises(ValueError, match="exact successful profile"):
                batch.verify_previous(config, dict(request, **{key: "b" * 64}))
        # Omitting the opt-in still requires a pilot, not a profile.
        config.skip_pilot = False
        with pytest.raises(ValueError, match="full-archive pilot"):
            batch.verify_previous(config, request)
        config.skip_pilot = True
    success["full_archive"] = False
    with pytest.raises(ValueError, match="full-archive"):
        batch.verify_previous(config, request)
    success["full_archive"] = True
    success["report_sha256"]["runtime_summary.json"] = "x" * 64
    with pytest.raises(ValueError, match="checksum"):
        batch.verify_previous(config, request)


def test_recovery_rejects_active_attempt_or_changed_request(monkeypatch):
    config = args()
    monkeypatch.setattr(batch, "read", lambda *a: "[]")
    monkeypatch.setattr(batch, "list_jobs", lambda *a: [dict(name=config.run_id + "-audit", status=dict(state="RUNNING"))])
    with pytest.raises(ValueError, match="active"):
        batch.cloud_preflight(config, {}, config.run_id + "-r-1", recovery=True)
    monkeypatch.setattr(batch, "list_jobs", lambda *a: [])
    monkeypatch.setattr(batch, "read", lambda *a: '[{"name":"present"}]' if "list" in a else "{}")
    with pytest.raises(ValueError, match="immutable"):
        batch.cloud_preflight(config, {"changed": True}, config.run_id + "-r-1", recovery=True)


def test_source_preflight_is_read_only_and_rejects_wrong_manifest_hash(monkeypatch):
    config = args()
    raw = json.dumps(tiny_manifest())
    record = dict(run="pinned-source", relative="archive/time_48h.json", completed_iterations=2,
                  sha256=hashlib.sha256(raw.encode()).hexdigest())
    monkeypatch.setattr(batch.experiment, "checkpoint_registry", lambda _: {"source": record})
    monkeypatch.setattr(batch, "list_jobs", lambda *a: [])
    calls = []
    def read(*command):
        calls.append(command)
        if command[:4] == ("gcloud", "storage", "objects", "list"):
            return "[]"
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            assert command[4].startswith(experiment.SOURCE_BUCKET + "/pinned-source/")
            return json.dumps(dict(size=len(raw)))
        if command[:3] == ("gcloud", "storage", "cat"):
            return raw
        return "runner@example-project.iam.gserviceaccount.com"
    monkeypatch.setattr(batch, "read", read)
    batch.cloud_preflight(config, {}, config.run_id + "-audit")
    assert all("submit" not in c and "cp" not in c and "delete" not in c for c in calls)
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Source manifest checksum"):
        batch.cloud_preflight(config, {}, config.run_id + "-audit")


def test_native_full_archive_profile_integration(tmp_path, monkeypatch):
    """Tiny full archives exercise the real evaluator, not production strength/timing."""
    native = os.environ.get("FHP_SD_REPO")
    if not native:
        pytest.skip("Set FHP_SD_REPO to enable the native end-to-end profile")
    fixture = tmp_path / "fixture"
    subprocess.run([sys.executable, str(ROOT / "tests/native_adapter_probe.py"), "sd_cfr", native, str(fixture)],
                   check=True, capture_output=True, text=True, timeout=120,
                   env=dict(os.environ, PYTHONPATH=str(ROOT)))
    # Use all three iterations of the encoded fixture; production hashes remain immutable.
    digest = experiment.sha256(fixture / "183/policy.json")
    monkeypatch.setattr(experiment, "HASHES", {0: ((3, digest), (3, digest))})
    inputs = tmp_path / "inputs/seed_0"
    inputs.mkdir(parents=True)
    import shutil
    for path in (fixture / "183").iterdir():
        shutil.copyfile(path, inputs / ("time_48h.json" if path.name == "policy.json" else path.name))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["sd-audit", "--stage", "profile", "--input-root", str(inputs.parent),
                                     "--native-root", native, "--output", str(output)])
    experiment.main()
    marker = json.loads((output / "analysis/SUCCESS.json").read_text())
    runtime = json.loads((output / "analysis/runtime_summary.json").read_text())
    assert marker["full_archive"] and marker["stage"] == "profile"
    assert not marker["assessment_evaluated"]
    assert runtime["full_archive_exports"][0]["models_per_player"] == 3
    assert len(runtime["full_archive_exports"]) == 1
    assert not (output / "analysis/temporal_summary.json").exists()
    # A new VM restores identical chunks with different download timestamps.
    for path in inputs.glob("*.npy"):
        os.utime(path, (2000000000, 2000000000))
    # A resumed invocation must still reuse all verified board tables.
    attempts = sorted(output.glob("workers/*/*/attempt_*"))
    experiment.main()
    assert sorted(output.glob("workers/*/*/attempt_*")) == attempts
    # The completeness report refuses even a checksum-valid truncated archive diagnostic.
    from fhp_evaluation.strategic_audit.workflow import digest as fingerprint, mark
    table = next((output / "tables/sd_exp6_seed0_48h").iterdir())
    diagnostic = json.loads((table / "diagnostics.json").read_text())
    diagnostic["adapter"]["models_per_player"] = 2
    (table / "diagnostics.json").write_text(json.dumps(diagnostic))
    mark(table, fingerprint(json.loads((output / "manifest.json").read_text())), "tables.npz")
    with pytest.raises(ValueError, match="complete uniform archive"):
        experiment.write_runtime_report(output, "profile", experiment.make_library("profile"))
