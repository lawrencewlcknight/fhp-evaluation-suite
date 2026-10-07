import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from fhp_evaluation.strategic_audit import exp16
from fhp_evaluation.strategic_audit.library import build_library, board_schedule, canonical_board
from fhp_evaluation.strategic_audit.library import validate_library
from fhp_evaluation.strategic_audit.workflow import validate_spec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strategic_batch", ROOT / "gcp/exp6_ucv_exp16_strategic_audit_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def args(native=None, stage="pilot"):
    return argparse.Namespace(run_id="fhp-strat16-test", project="clever-overview-399515", region="europe-west1",
                              bucket="gs://example-bucket", service_account="runner@example-project.iam.gserviceaccount.com",
                              native_repo=native, stage=stage, pilot_run_id=None, pilot_reviewed=False)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "native"
    for package in ("fhp_escher", "vr_deep_cfr"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("# native source\n")
    (root / "secret.json").write_text("not for upload")
    return root


def synthetic_result(stage="development"):
    spec = exp16.make_spec(stage, "/input", "/native")
    comparisons = {}
    for name in spec["candidates"]:
        meta = spec["policies"][name]
        for opponent_index, opponent in enumerate(spec["opponents"]):
            # Strongly varying opponent levels/support catch erroneous pooling.
            value = 10 * opponent_index + (48 - meta["hours"]) / 100 * (meta["training_seed"] + 1)
            for tilt in spec["range_tilts"]:
                comparisons[f"{name}__vs__{opponent}__tilt_{tilt:g}"] = dict(
                    mean_root_response_gap_bb=value,
                    boards=[dict(board=[0, 5, 10], core=True, root_count=opponent_index + 1)])
    return dict(provenance=dict(spec=spec), comparisons=comparisons)


def test_prespecified_checkpoint_and_panel_contract():
    pilot = exp16.checkpoint_registry("pilot")
    main = exp16.checkpoint_registry("development")
    assert len(pilot) == 5 and len(main) == 15
    assert set(pilot) <= set(main)
    assert all(main[k] == row for k, row in pilot.items())
    assert {main[k]["training_seed"] for k in main if k.startswith("exp16")} == {0, 1, 2}
    assert {main[k]["hours"] for k in main if k.startswith("exp16")} == {24, 36, 42, 48}
    for label, row in main.items():
        assert row["relative"].endswith(".pkl") and "training_state" not in row["relative"]
        assert len(row["sha256"]) == 64
        assert row["run"] == (exp16.EXP16_RUN if label.startswith("exp16") else exp16.EXP9_RUN)
    config = exp16.make_spec("pilot", "/input", "/native")
    assert config["candidates"] == ["exp16_seed0_24h", "exp16_seed0_48h"]
    assert len(config["opponents"]) == 8
    assert config["reference_policy"] == "exp9_seed0_24h"
    with pytest.raises(ValueError, match="locked"):
        exp16.checkpoint_registry("assessment")


def test_real_checkpoint_checksum_is_required(tmp_path, native):
    config = exp16.make_spec("pilot", tmp_path, native)
    for record in config["policies"].values():
        if record["family"] == "ucv":
            Path(record["checkpoint"]).write_bytes(b"incorrect checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        validate_spec(config)


def test_pilot_and_development_never_touch_assessment():
    full = build_library(exp16.LIBRARY_SEED, reference_deals=64)
    pilot = build_library(exp16.LIBRARY_SEED, pilot=True, reference_deals=8)
    locked = {canonical_board(r["board"]) for r in full["boards"] if r["partition"] == "assessment"}
    assert len(full["boards"]) == 96 and len(pilot["boards"]) == 8
    for library in (full, pilot):
        assert not locked.intersection(canonical_board(b) for b in board_schedule(library, "development", True))
    assert {tuple(b) for b in pilot["assessment_board_classes"]} == locked
    bad = copy.deepcopy(pilot)
    board = list(next(iter(locked)))
    bad["reference_deals"][0]["cards"] = [c for c in range(52) if c not in board][:4] + board
    with pytest.raises(ValueError, match="leakage"):
        validate_library(bad)


def test_paired_seed_inference_and_equal_opponent_weights(tmp_path):
    result = synthetic_result()
    summary = exp16.write_temporal_report(tmp_path, result, "development")
    primary = summary["paired_changes"]["48_minus_24h_tilt_0"]["equal_weight_panel"]
    assert primary["n"] == 3  # not 24 opponents x seeds, nor thousands of positions
    assert primary["mean"] == pytest.approx(-.48)
    assert list(primary["seed_values"].values()) == pytest.approx([-.24, -.48, -.72])
    assert primary["standard_error"] == pytest.approx(.24 / 3**.5)
    endpoint = summary["endpoints"]["48h_tilt_0"]["equal_weight_panel"]
    assert endpoint["mean"] == 35  # not root-count weighted across opponents
    assert len(summary["paired_changes"]) == 9
    assert (tmp_path / "exp16_report.html").exists()
    assert (tmp_path / "panel_root_gap.svg").exists()


def test_pilot_is_descriptive_only():
    result = exp16.temporal_summary(synthetic_result("pilot"), "pilot")
    row = result["paired_changes"]["48_minus_24h_tilt_0"]["equal_weight_panel"]
    assert row["n"] == 1
    assert row["standard_error"] is row["descriptive_95pct_t_interval"] is None


@pytest.mark.parametrize("corruption", ["count", "missing", "nonfinite", "panel"])
def test_aggregation_refuses_incomplete_or_unmatched_data(corruption):
    result = synthetic_result()
    key = "exp16_seed0_48h__vs__candid_statistician__tilt_0"
    if corruption == "count":
        result["comparisons"][key]["boards"][0]["root_count"] = 7
    elif corruption == "missing":
        del result["comparisons"][key]
    elif corruption == "panel":
        result["provenance"]["spec"]["opponents"].pop()
    else:
        result["comparisons"][key]["mean_root_response_gap_bb"] = float("nan")
    with pytest.raises((KeyError, ValueError)):
        exp16.temporal_summary(result, "development")


def test_stage_gate_requires_explicit_pilot_review(native):
    config = args(native, "development")
    with pytest.raises(ValueError, match="pilot-reviewed"):
        batch.validate(config, preparing=True)
    config.pilot_run_id, config.pilot_reviewed = "fhp-strat16-pilot", True
    batch.validate(config, preparing=True)


def test_job_bounded_inference_only_and_shell_syntax(native):
    config = args(native)
    batch.validate(config, preparing=True)
    for stage, duration in (("pilot", "21600s"), ("development", "86400s")):
        config.stage = stage
        job = batch.build_job(config, "a" * 64)
        group = job["taskGroups"][0]
        assert group["taskCount"] == group["parallelism"] == 1
        task = group["taskSpec"]
        assert task["maxRetryCount"] == 0 and task["maxRunDuration"] == duration
        assert task["runnables"][1]["alwaysRun"]
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        for runnable in task["runnables"]:
            subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)
        assert "training_state" not in json.dumps(job)


def test_bundle_and_offline_prepare(native, tmp_path):
    directory = tmp_path / "prepared"
    command = ["python3", str(ROOT / batch.EXTRA_FILES[0]), "prepare", "--run-id", "fhp-strat16-offline",
               "--project", "clever-overview-399515", "--region", "europe-west1", "--bucket", "gs://example-bucket",
               "--service-account", "runner@example-project.iam.gserviceaccount.com",
               "--native-repo", str(native), "--run-dir", str(directory)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    request = json.loads((directory / "request.json").read_text())
    with tarfile.open(directory / "source.tar.gz") as archive:
        names = archive.getnames()
        assert "evaluator/fhp_evaluation/strategic_audit/exp16.py" in names
        assert "evaluator/" + batch.EXTRA_FILES[1] in names
        assert not any("secret" in n or "/.git/" in n for n in names)
        for name, digest in request["source"]["files"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    config = args(native)
    config.run_id = "fhp-strat16-offline"
    assert batch.verify_request(config, directory) == request
    (directory / "job.json").write_text("{}")
    with pytest.raises(ValueError, match="Prepared file"):
        batch.verify_request(config, directory)


def test_recovery_rejects_active_attempt_and_changed_request(monkeypatch):
    config = args()
    monkeypatch.setattr(batch, "list_jobs", lambda *_: [dict(name=config.run_id + "-audit", status=dict(state="RUNNING"))])
    monkeypatch.setattr(batch, "read", lambda *_: "[]")
    with pytest.raises(ValueError, match="active"):
        batch.cloud_preflight(config, {}, config.run_id + "-r-1", recovery=True)
    monkeypatch.setattr(batch, "list_jobs", lambda *_: [])
    def read(*command):
        return '[{"name":"present"}]' if "list" in command else '{}'
    monkeypatch.setattr(batch, "read", read)
    with pytest.raises(ValueError, match="immutable"):
        batch.cloud_preflight(config, {"changed": True}, config.run_id + "-r-1", recovery=True)


def test_main_submission_requires_successful_same_source_pilot(monkeypatch):
    config = args(stage="development")
    config.pilot_run_id, config.pilot_reviewed = "fhp-strat16-pilot", True
    source = {"files": {"source.py": "hash"}}
    success = dict(status="succeeded", stage="pilot", checkpoint_sha256={
        k: v["sha256"] for k, v in exp16.checkpoint_registry("pilot").items()})
    def read(*command):
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            return '{"size":1000}'
        if command[:4] == ("gcloud", "storage", "objects", "list"):
            return '[]'
        if command[:3] == ("gcloud", "storage", "cat"):
            return json.dumps(success if command[3].endswith("SUCCESS.json") else {"source": source})
        return ""
    monkeypatch.setattr(batch, "read", read)
    monkeypatch.setattr(batch, "list_jobs", lambda _args, prefix: [dict(status=dict(state="SUCCEEDED"))]
                        if prefix.startswith(config.pilot_run_id) else [])
    batch.cloud_preflight(config, {"source": source}, config.run_id + "-audit")
    with pytest.raises(ValueError, match="Source changed"):
        batch.cloud_preflight(config, {"source": {"files": {}}}, config.run_id + "-audit")
    success["status"] = "failed"
    with pytest.raises(ValueError, match="completed real-model pilot"):
        batch.cloud_preflight(config, {"source": source}, config.run_id + "-audit")
