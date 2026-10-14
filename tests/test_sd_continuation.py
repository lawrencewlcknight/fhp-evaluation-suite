import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fhp_evaluation.strategic_audit import sd_continuation as continuation
from fhp_evaluation.strategic_audit.workflow import digest, mark, completed
from fhp_evaluation.best_response.runner import write_json
from fhp_evaluation.loaders import sha256_file


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from fhp_evaluation.strategic_audit import library, tables, sd_exp6
    source, target = tmp_path / "old", tmp_path / "new"
    item = dict(family="sd_cfr", sha256="archive", checkpoint="/input/a", repo_root="/native")
    old = dict(protocol="test", spec=dict(policies={"sd": item}), library={}, partition="development",
               include_reference=False, packages={}, threads=4,
               policies={"sd": dict(native=dict(checkpoint_sha256="archive", chunk_hashes={"a":"b"}, spec=dict(item)))})
    write_json(source / "manifest.json", old)
    monkeypatch.setattr(continuation, "MANIFEST_SHA256", sha256_file(source / "manifest.json"))
    monkeypatch.setattr(continuation, "EXPECTED_BOARDS", 1)
    monkeypatch.setattr(library, "board_schedule", lambda *args: [[23,24,42]])
    monkeypatch.setattr(sd_exp6, "checkpoint_registry", lambda *args: {"sd":dict(sha256="archive", completed_iterations=2)})
    monkeypatch.setattr(tables, "load_tables", lambda path, context: np.load(path, allow_pickle=False))
    path = source / "tables/sd" / library.board_key([23,24,42])
    path.mkdir(parents=True)
    np.savez_compressed(path / "tables.npz", x=np.ones((1,3))/3)
    write_json(path / "diagnostics.json", dict(adapter=dict(checkpoint_sha256="archive", models_per_player=2,
        weighting="uniform", archive_integrity_verified=True), checks=dict(native_scalar_parity=dict(passed=True))))
    mark(path, digest(old), "tables.npz")
    new = copy.deepcopy(old)
    new["spec"]["policies"]["sd"].update(sd_float64_fallback=True, sd_query_order="hand_major")
    return source, target, old, new, path


def test_verified_import_is_idempotent_and_preserves_original(cache):
    source,target,old,new,path = cache
    original = (path / "SUCCESS.json").read_bytes()
    fingerprint = digest(new)
    for _ in range(2):
        result = continuation.import_verified(source,target,new,fingerprint)
        assert result["imported_boards"] == 1
    dest = target / path.relative_to(source)
    assert completed(dest, fingerprint, "tables.npz")
    assert (dest / "tables.npz").read_bytes() == (path / "tables.npz").read_bytes()
    assert (path / "SUCCESS.json").read_bytes() == original
    assert json.loads((dest / "imported_SUCCESS.json").read_text()) == json.loads(original)


@pytest.mark.parametrize("change", ["archive", "library", "table", "manifest", "missing"])
def test_incompatible_or_corrupt_cache_is_not_published(cache, change):
    source,target,old,new,path = cache
    if change == "archive":
        new["policies"]["sd"]["native"]["chunk_hashes"]["a"] = "changed"
    elif change == "library":
        new["library"] = {"changed":True}
    elif change == "table":
        (path / "tables.npz").write_bytes(b"corrupt")
    elif change == "manifest":
        (source / "manifest.json").write_text("{}")
    else:
        (path / "SUCCESS.json").unlink()
    with pytest.raises(ValueError):
        continuation.import_verified(source,target,new,digest(new))
    assert not list(target.rglob("SUCCESS.json"))


def test_continuation_prepare_is_offline_and_pins_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    native = tmp_path / "native"
    (native / "deep_cfr_poker").mkdir(parents=True)
    (native / "deep_cfr_poker/__init__.py").write_text("")
    target = tmp_path / "prepared"
    subprocess.run([sys.executable,str(root / "gcp/sd_exp6_continue_batch.py"),"prepare",
        "--run-id","fhp-sd6-audit-fix-test","--project","clever-overview-399515","--region","europe-west1",
        "--bucket","gs://example-results","--service-account","runner@example-project.iam.gserviceaccount.com",
        "--native-repo",str(native),"--run-dir",str(target)],check=True)
    job = json.loads((target / "job.json").read_text())
    task = job["taskGroups"][0]["taskSpec"]
    assert task["environment"]["variables"]["FHP_IMPORT_SOURCE"] == continuation.SOURCE_URI
    assert task["maxRetryCount"] == 0
    assert task["maxRunDuration"] == "259200s"
    assert "--import-sd-cache" in task["runnables"][0]["script"]["text"]


@pytest.mark.skipif(not os.environ.get("FHP_SD_OLD_CACHE"), reason="optional downloaded production cache")
def test_production_cache_import(tmp_path):
    from fhp_evaluation.strategic_audit.sd_exp6 import make_spec, make_library
    source = Path(os.environ["FHP_SD_OLD_CACHE"])
    new = json.loads((source / "manifest.json").read_text())
    new["spec"] = make_spec("development", "/workspace/fhp-sd6-audit/input", "/workspace/fhp-sd6-audit/source/native")
    new["library"] = make_library("development")
    report = continuation.import_verified(source,tmp_path,new,digest(new))
    assert report["imported_boards"] == 115
    for row in report["boards"]:
        assert completed(tmp_path / row["relative"],digest(new),"tables.npz")
