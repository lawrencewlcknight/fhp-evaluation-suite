import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.loaders import sha256_file
from fhp_evaluation.safeguard.experiment import config, tasks
from fhp_evaluation.safeguard.runner import completed, aggregate_tasks
from fhp_evaluation.safeguard.policy import TerminalCallVerifier
from fhp_evaluation.game import load_fhp_game
from fhp_evaluation.strategic_audit import PROTOCOL
from fhp_evaluation.strategic_audit.library import texture
from fhp_evaluation.strategic_audit.workflow import run

ROOT = Path(__file__).resolve().parents[1]


def test_guarded_strategic_workflow_and_recovery(tmp_path):
    spec = dict(protocol=PROTOCOL, policies={
        "base": dict(family="control",name="uniform",cohort="base",training_seed=0,hours=48),
        "guard": dict(family="control",name="uniform",cohort="guard",training_seed=0,hours=48,
                      deployment="exact_terminal_call_v1")},
        candidates=["base","guard"],opponents=["base"],reference_policy="base",range_tilts=[0])
    board = [6,10,50]
    library = dict(protocol=PROTOCOL,boards=[dict(board=board,texture=texture(board),partition="development")],
                   assessment_board_classes=[],reference_deals=[])
    path = run(spec,library,tmp_path,threads=1,max_seconds=90,max_rss_mb=4096)
    result = json.loads(path.read_text())
    base = result["comparisons"]["base__vs__base__tilt_0"]["mean_root_response_gap_bb"]
    guarded = result["comparisons"]["guard__vs__base__tilt_0"]["mean_root_response_gap_bb"]
    assert guarded < base
    assert run(spec,library,tmp_path,threads=1,max_seconds=90,max_rss_mb=4096,resume=True) == path


def test_offline_cloud_preparation_bundles_both_repos(tmp_path):
    native = tmp_path / "native"
    for name in ("fhp_escher", "vr_deep_cfr"):
        (native/name).mkdir(parents=True)
        (native/name/"__init__.py").write_text("")
    (native/"fhp_escher/checkpointing.py").write_text("# test fixture\n")
    for mode in ("smoke","full"):
        target = tmp_path/mode
        result = subprocess.run([sys.executable,str(ROOT/"gcp/exp34_ucv_terminal_safeguard_batch.py"),
            "prepare","--project","clever-overview-399515","--region","europe-west1",
            "--bucket","gs://clever-overview-399515-fhp-escher-results",
            "--service-account","fhp-runner@clever-overview-399515.iam.gserviceaccount.com",
            "--run-id",f"exp34-safeguard-test-{mode}","--mode",mode,
            "--native-repo",str(native),"--run-dir",str(target)],capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        request = json.loads((target/"request.json").read_text())
        assert request["experiment"] == config(mode)
        assert "evaluator/fhp_evaluation/safeguard/policy.py" in request["source"]["files"]
        assert "native/fhp_escher/checkpointing.py" in request["source"]["files"]
        job = json.loads((target/"job.json").read_text())
        task = job["taskGroups"][0]["taskSpec"]
        assert task["environment"]["variables"]["FHP_SAFEGUARD_MODE"] == mode
        assert task["maxRetryCount"] == 0
        assert task["runnables"][-1]["alwaysRun"]
        assert "fhp_evaluation.safeguard.runner" in task["runnables"][0]["script"]["text"]


def test_partial_shard_not_treated_as_complete(tmp_path):
    task = dict(id="test",seed=0,opponent="base",start=0,count=5)
    (tmp_path/"test").mkdir()
    (tmp_path/"test/result.json").write_text("{}")
    assert not completed(tmp_path,task,"identity")


def test_complete_aggregation_and_checksum_admission(tmp_path):
    spec = config("smoke")
    fingerprint = "test-identity"
    timing = TerminalCallVerifier(load_fhp_game()).diagnostics()
    for task in tasks(spec):
        path = tmp_path/"task_results"/task["id"]
        path.mkdir(parents=True)
        np.savez_compressed(path/"pairs.npz",rows=np.zeros((task["count"],8)))
        write_json(path/"result.json",dict(task=task,identity=fingerprint,deployment_verifier=timing))
        write_json(path/"SUCCESS.json",dict(identity=fingerprint,
            result_sha256=sha256_file(path/"result.json"),pairs_sha256=sha256_file(path/"pairs.npz")))
    aggregate_tasks(SimpleNamespace(output=tmp_path),spec,fingerprint)
    summary = json.loads((tmp_path/"analysis/summary.json").read_text())
    assert summary["primary"]["integrated_delta_mbb"]["n"] == 1
    assert summary["primary_by_seed"]["0"]["zero_event_warning"]
    assert len(summary["panels"]) == 7
    latency = json.loads((tmp_path/"analysis/latency.json").read_text())
    assert "ineligible" in latency["verifier_overhead"]["deployment_verifier"]["latency_counts"]
    task = next(iter(tasks(spec)))
    (tmp_path/"task_results"/task["id"]/"result.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        completed(tmp_path/"task_results",task,fingerprint)
