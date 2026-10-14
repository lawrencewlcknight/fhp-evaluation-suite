"""Explicit, checksummed import of the stopped SD6 development audit only."""
import copy
import json
from pathlib import Path
import shutil

SOURCE_RUN = "fhp-sd6-audit-dev-20261013-104108"
SOURCE_URI = "gs://clever-overview-399515-fhp-deep-cfr-results/" + SOURCE_RUN
MANIFEST_SHA256 = "831f1e330135611370173407f914c04a4b0db9f9c91ad131fedc8e2ee35b0c29"
EXPECTED_BOARDS = 115


def import_verified(source, destination, provenance, fingerprint):
    from ..best_response.runner import write_json
    from ..loaders import sha256_file
    from ..game import load_fhp_game
    from .library import board_schedule, board_key
    from .tables import BoardContext, load_tables
    from .workflow import digest, completed, mark
    from .sd_exp6 import checkpoint_registry
    source, destination = Path(source), Path(destination)
    if sha256_file(source / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Continuation source is not the pinned stopped audit")
    old = json.loads((source / "manifest.json").read_text())
    expected_spec = copy.deepcopy(provenance["spec"])
    for item in expected_spec["policies"].values():
        item.pop("sd_float64_fallback", None)
        item.pop("sd_query_order", None)
    if expected_spec != old["spec"]:
        raise ValueError("Continuation changed the scientific policy specification")
    for key in ("protocol", "library", "partition", "include_reference", "packages", "threads"):
        if provenance[key] != old[key]:
            raise ValueError(f"Continuation changed {key}")
    registry = checkpoint_registry("development")
    for label in registry:
        before, after = old["policies"][label]["native"], provenance["policies"][label]["native"]
        for key in ("checkpoint_sha256", "chunk_hashes", "spec"):
            if before[key] != after[key]:
                raise ValueError(f"Continuation changed archive: {label}/{key}")
    old_identity = digest(old)
    boards = board_schedule(old["library"], "development", False)
    rows = []
    # Validate all imported artifacts before publishing any new success markers.
    game = load_fhp_game()
    for label, record in registry.items():
        for board in boards:
            relative = Path("tables") / label / board_key(board)
            original = source / relative
            if not completed(original, old_identity, "tables.npz"):
                continue
            diagnostic = json.loads((original / "diagnostics.json").read_text())
            adapter, checks = diagnostic["adapter"], diagnostic["checks"]
            if (adapter.get("checkpoint_sha256") != record["sha256"]
                    or adapter.get("models_per_player") != record["completed_iterations"]
                    or adapter.get("weighting") != "uniform"
                    or adapter.get("archive_integrity_verified") is not True
                    or checks.get("native_scalar_parity", {}).get("passed") is not True):
                raise ValueError("Imported board lacks verified full-archive diagnostics")
            load_tables(original / "tables.npz", BoardContext(game, board))
            rows.append(dict(policy=label, board=board, relative=str(relative),
                             original_marker=json.loads((original / "SUCCESS.json").read_text())))
    if len(rows) != EXPECTED_BOARDS:
        raise ValueError(f"Expected {EXPECTED_BOARDS} verified source boards, found {len(rows)}")
    lineage = dict(source_uri=SOURCE_URI, source_manifest_sha256=MANIFEST_SHA256,
                   original_identity=old_identity, destination_identity=fingerprint,
                   imported_boards=len(rows), boards=rows,
                   note="Unchanged old table bytes; original diagnostics and marker retained. "
                        "New boards use numerical retry and hand-major export. No old analysis reused.")
    write_json(destination / "cache_import.json", lineage)
    # Preserve the complete old provenance, not just a rewritten completion marker.
    write_json(destination / "imported_source_manifest.json", old)
    for row in rows:
        original, target = source / row["relative"], destination / row["relative"]
        if completed(target, fingerprint, "tables.npz"):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in ("tables.npz", "diagnostics.json"):
            temporary = target / (name + ".importing")
            shutil.copyfile(original / name, temporary)
            temporary.replace(target / name)
        write_json(target / "imported_SUCCESS.json", row["original_marker"])
        mark(target, fingerprint, "tables.npz")
    return lineage
