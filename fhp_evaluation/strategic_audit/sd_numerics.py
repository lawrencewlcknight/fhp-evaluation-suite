"""Local, full-archive SD6 numerical reproduction; never submits cloud compute."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from .sd_exp6 import SOURCE_BUCKET, checkpoint_registry, archive_chunks, sha256


def download(root):
    record = checkpoint_registry("profile")["sd_exp6_seed0_48h"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "time_48h.json"
    uri = f"{SOURCE_BUCKET}/{record['run']}/{record['relative']}"
    if not manifest.exists():
        subprocess.run(["gcloud", "storage", "cp", uri, str(manifest)], check=True)
    if sha256(manifest) != record["sha256"]:
        raise ValueError("Pinned manifest checksum mismatch")
    chunks = archive_chunks(json.loads(manifest.read_text()), record)
    missing = []
    for chunk in chunks:
        target = root / chunk["path"]
        if not target.exists():
            missing.append(chunk)
        elif target.stat().st_size != chunk["size_bytes"] or sha256(target) != chunk["sha256"]:
            raise ValueError(f"Existing local chunk differs: {target}")
    if missing:
        def fetch(chunk):
            # gcloud cp on this host reported empty CRCs. Stream bytes instead,
            # then independently verify the pinned size AND SHA before publishing.
            source = uri.rsplit("/", 1)[0] + "/" + chunk["path"]
            raw = subprocess.run(["gcloud", "storage", "cat", source],
                                 check=True, capture_output=True, timeout=180).stdout
            if len(raw) != chunk["size_bytes"] or hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                raise ValueError(f"Streamed chunk failed pinned integrity: {source}")
            temporary = root / (chunk["path"] + ".partial")
            temporary.write_bytes(raw)
            temporary.replace(root / chunk["path"])
            print(f"Verified {chunk['path']}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, missing))
    # Full reader verification also runs before any numerical inference.
    return manifest


def probe(path, native, output, threads=1, precision="float32", all_checks=False):
    import numpy as np
    import torch
    from ..best_response.adapters import load_target
    from ..best_response.flop import make_state
    from ..best_response.runner import write_json
    from ..game import load_fhp_game
    from .tables import BoardContext
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    game = load_fhp_game()
    adapter = load_target(game, path, family="sd_cfr", repo_root=native,
                          batch_size=1024, model_batch_size=32)
    if precision == "float64":
        adapter = adapter.sd_float64_copy()
    context = BoardContext(game, [23, 24, 42])
    hand, pre = tuple(context.space.hands[0]), context.entries[0]
    states = [make_state(game, context.deck, hand, n.player, board=context.space.board,
                        preflop=pre, flop=n.path) for n in context.trees[pre] if n.player >= 0]
    records = []
    for player in (0, 1):
        selected = [s for s in states if s.current_player() == player]
        batch = adapter.batch_probabilities(selected, player)
        for state, row in zip(selected, batch):
            raw = adapter.reference.action_probabilities(state, player)
            expected = np.array([raw.get(a, 0.) for a in range(3)])
            expected /= expected.sum()
            records.append(dict(player=player, history=state.history(),
                                information_state=state.information_state_string(player),
                                batched=row.tolist(), scalar=expected.tolist(),
                                max_error=float(np.max(np.abs(row - expected)))))
            write_json(output, dict(metadata=adapter.metadata, rows=records,
                       elapsed_seconds=time.perf_counter()-started, threads=threads))
    if all_checks:
        from .checks import check_adapter
        checks = check_adapter(context, adapter, require_suit_invariance=True)
        write_json(output, dict(metadata=adapter.metadata, rows=records, checks=checks,
                   elapsed_seconds=time.perf_counter()-started, threads=threads))
    print(json.dumps(dict(max_error=max(r["max_error"] for r in records), rows=len(records),
                         seconds=time.perf_counter()-started)))


def layout_probe(root, native, output, threads=1, precision="float32"):
    """Cheap CPU code-path test: full board but explicitly only first 128 models."""
    import numpy as np
    import torch
    from ..best_response.adapters import load_target
    from ..best_response.runner import write_json
    from ..game import load_fhp_game
    from .tables import BoardContext, build_tables
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((root / "time_48h.json").read_text())
    prefix = dict(manifest, completed_iterations=manifest["chunks"][0]["last_iteration"],
                  chunks=manifest["chunks"][:1])
    path = root / "diagnostic_prefix.json"
    write_json(path, prefix)
    game = load_fhp_game()
    context = BoardContext(game, [23, 24, 42])
    measurements, reference = [], None
    for order in ("schema", "hand_major", "hand_major", "schema"):
        adapter = load_target(game, path, family="sd_cfr", repo_root=native,
                              batch_size=1024, model_batch_size=32)
        if precision == "float64":
            adapter = adapter.sd_float64_copy()
        started = time.perf_counter()
        tables = build_tables(context, adapter, query_order=order)
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = tables
        error = max(float(np.max(np.abs(tables[k] - reference[k]))) for k in reference)
        measurements.append(dict(query_order=order, seconds=elapsed, max_error_vs_first_schema=error,
                                 adapter=adapter.stats, inference=adapter.native.stats))
        report = dict(scope="local Apple CPU; full 1176 hands and 64 histories; first 128 archived models ONLY",
                      models_per_player=prefix["completed_iterations"], threads=threads,
                      precision=precision, measurements=measurements,
                      full_archive_speedup_claim=False, cloud_runtime_forecast=False)
        write_json(output, report)
        print(json.dumps(measurements[-1]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("download", "probe", "layout"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--native-repo", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--all-checks", action="store_true", help="Also run privacy, card-order and suit diagnostics")
    a = p.parse_args()
    if a.action == "download":
        print(download(a.root))
    else:
        if not a.native_repo or not a.output:
            p.error("probe requires native-repo and output")
        if a.action == "layout":
            layout_probe(a.root, a.native_repo, a.output, a.threads, a.precision)
        else:
            probe(a.root / "time_48h.json", a.native_repo, a.output, a.threads, a.precision, a.all_checks)


if __name__ == "__main__":
    main()
