"""Pinned Exp10 inputs; extract policy replay once from historical full states."""

from pathlib import Path
import json
import subprocess

import numpy as np
import torch

from ..loaders import sha256_file
from ..best_response.runner import write_json
from .config import MANIFEST_HASHES, SOURCE_RUN

MARKER = "__exp1_numpy_shard__"


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError("Source path escapes its root")
    return path


def checked(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return Path(path)


def fetch_panel(bucket, directory):
    """All three independently trained Exp10 policies; no replay downloads."""
    paths = {}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    for seed in (0, 1, 2):
        worker = f"task_{seed:03d}_hand_board_cached_parallel_ucv_escher_seed_{seed}"
        uri = f"{bucket.rstrip('/')}/{SOURCE_RUN}/workers/{worker}"
        manifest = root / f"manifest_{seed}.json"
        subprocess.run(["gcloud", "storage", "cp", f"{uri}/checkpoint_manifest.json", str(manifest)],
                       check=True, timeout=180)
        checked(manifest, MANIFEST_HASHES[seed])
        rows = [r for r in json.loads(manifest.read_text()) if r["checkpoint_id"] == "time_24h"]
        if len(rows) != 1:
            raise ValueError("Ambiguous panel source")
        row = rows[0]
        safe_path(root, row["path"])
        path = root / f"exp10_seed_{seed}.pkl"
        subprocess.run(["gcloud", "storage", "cp", f"{uri}/{row['path']}", str(path)], check=True, timeout=180)
        paths[f"exp10_seed_{seed}"] = checked(path, row["sha256"])
    return paths


def fetch_source(bucket, seed, directory, *, include_replay=True):
    """Read-only GCS access. Trust only the declared historical manifest hash."""
    if seed not in (0, 1, 2):
        raise ValueError("Expected source seed 0, 1 or 2")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    worker = f"task_{seed:03d}_hand_board_cached_parallel_ucv_escher_seed_{seed}"
    prefix = f"{bucket.rstrip('/')}/{SOURCE_RUN}/workers/{worker}"

    def get(relative, destination, digest=None):
        safe_path(root, str(Path(destination).relative_to(root)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", f"{prefix}/{relative}", str(destination)],
                       check=True, timeout=900)
        if digest is not None:
            checked(destination, digest)

    manifest_path = root / "checkpoint_manifest.json"
    get("checkpoint_manifest.json", manifest_path, MANIFEST_HASHES[seed])
    records = json.loads(manifest_path.read_text())
    rows = [row for row in records if row["checkpoint_id"] == "time_24h"]
    if len(rows) != 1:
        raise ValueError("Missing/ambiguous final source policy")
    row = rows[0]
    policy = safe_path(root, row["path"])
    get(row["path"], policy, row["sha256"])
    metadata = dict(seed=seed, source_run=SOURCE_RUN,
                    checkpoint_manifest_sha256=MANIFEST_HASHES[seed], checkpoint=row)
    if not include_replay:
        write_json(root / "source.json", metadata)
        return root
    state = safe_path(root, row["training_state_path"])
    # Historical Exp10 really is a ~5.4 GB monolithic Torch checkpoint. Read it
    # ONCE at qualification. Subsequent phases use a hashed policy-only cache.
    if state.suffix == ".pt":
        if not 0 < row["training_state_size_bytes"] < 12 * 1024**3:
            raise ValueError("Unexpected historical training-state size")
        get(row["training_state_path"], state, row["training_state_sha256"])
        write_json(root / "source.json", metadata)
        return root
    get(f"{row['training_state_path']}/manifest.json", state / "manifest.json",
        row["training_state_sha256"])
    manifest = json.loads((state / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("schema_version") != 1:
        raise ValueError("Incomplete/unsupported source state")
    metadata_path = safe_path(state, manifest["metadata_path"])
    get(f"{row['training_state_path']}/{manifest['metadata_path']}", metadata_path,
        manifest["metadata_sha256"])
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    buffer = metadata["average_policy_trainer"]["buffer"]
    records_by_path = {record["path"]: record for record in manifest["array_shards"]}
    for key in ("infostate", "q_value", "q_value_mask", "iteration"):
        if set(buffer[key]) != {MARKER}:
            raise ValueError("Expected historical dense replay in separate numpy shards")
        relative = buffer[key][MARKER]
        record = records_by_path[relative]
        if int(record["size_bytes"]) > 2 * 1024**3:
            raise ValueError("Unexpectedly large policy replay shard")
        get(f"{row['training_state_path']}/{relative}", safe_path(state, relative), record["sha256"])
    write_json(root / "source.json", metadata)
    return root


class Replay:
    """Grouped CE data with the original iteration mass and objective scale."""
    def __init__(self, features, targets, masks, times, iteration, gamma=2):
        size = len(features)
        if size == 0 or features.shape != (size, 213) or iteration <= 0:
            raise ValueError("Invalid Exp10 replay shape/iteration")
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if (targets.shape != (size, 3) or masks.shape != (size, 3) or times.shape != (size,)
                or np.any(times <= 0) or np.any(times > iteration)):
            raise ValueError("Invalid replay fields")
        for start in range(0, size, 65536):
            sl = slice(start, start + 65536)
            if (not all(np.isfinite(v[sl]).all() for v in (features, targets, masks, times))
                    or not np.isin(masks[sl], [0, 1]).all() or np.any(masks[sl].sum(1) == 0)
                    or np.any(targets[sl] < 0) or np.any(targets[sl][masks[sl] == 0] != 0)
                    or not np.allclose(targets[sl].sum(1), 1, atol=1e-5)):
                raise ValueError("Invalid replay values/legal probabilities")
        unique, first, inverse = np.unique(features, axis=0, return_index=True, return_inverse=True)
        mass = np.zeros(len(unique), dtype=np.float64)
        sums = np.zeros((len(unique), 3), dtype=np.float64)
        weights = np.power(times / iteration * 2, gamma)
        np.add.at(mass, inverse, weights)
        np.add.at(sums, inverse, np.asarray(targets, dtype=np.float64) * weights[:, None])
        self.x, self.y, self.mask = unique, (sums / mass[:, None]).astype(np.float32), masks[first].astype(bool)
        for start in range(0, size, 65536):
            if not np.array_equal(masks[start:start + 65536], self.mask[inverse[start:start + 65536]]):
                raise ValueError("Masks disagree inside an information set")
        # IID mass-proportional groups with constant loss scale: unbiased for
        # exactly the original grouped weighted CE, not uniform unweighted CE.
        self.cdf = np.cumsum(mass / mass.sum())
        self.cdf[-1] = 1
        self.scale = float(mass.sum() / size)
        self.rows = size

    def sample(self, rng, batch_size):
        indices = np.searchsorted(self.cdf, rng.random(batch_size), side="right")
        return tuple(torch.from_numpy(v[indices].copy()) for v in (self.x, self.y, self.mask))

    def save(self, directory, source_hash):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        files = {}
        for name in ("x", "y", "mask", "cdf"):
            path = directory / f"{name}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, getattr(self, name), allow_pickle=False)
            temporary.replace(path)
            files[name] = sha256_file(path)
        write_json(directory / "manifest.json", dict(files=files, scale=self.scale, rows=self.rows,
                   source_policy_sha256=source_hash, schema_version=1))
        return sha256_file(directory / "manifest.json")

    @classmethod
    def load(cls, directory, manifest_hash, source_hash):
        directory = Path(directory)
        checked(directory / "manifest.json", manifest_hash)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest["source_policy_sha256"] != source_hash:
            raise ValueError("Replay cache provenance mismatch")
        instance = cls.__new__(cls)
        for name in ("x", "y", "mask", "cdf"):
            path = checked(directory / f"{name}.npy", manifest["files"][name])
            setattr(instance, name, np.load(path, mmap_mode="r", allow_pickle=False))
        instance.scale, instance.rows = manifest["scale"], manifest["rows"]
        return instance


def load_source(root, seed, target, *, need_replay=True):
    root = Path(root)
    metadata = json.loads((root / "source.json").read_text())
    if (metadata["seed"] != seed or metadata["source_run"] != SOURCE_RUN
            or metadata["checkpoint_manifest_sha256"] != MANIFEST_HASHES[seed]):
        raise ValueError("Source identity mismatch")
    checked(root / "checkpoint_manifest.json", MANIFEST_HASHES[seed])
    row = metadata["checkpoint"]
    if row not in json.loads((root / "checkpoint_manifest.json").read_text()):
        raise ValueError("Unpinned checkpoint row")
    checked(target.native.checkpoint_path, row["sha256"])
    policy = target.native.checkpoint
    if (policy["seed"] != seed or policy["experiment_id"] != 10
            or policy["checkpoint_target_seconds"] != 86400
            or policy["input_size"] != 213
            or policy["policy_model"]["type"] != "structured_fhp_mlp_v1"
            or policy["policy_model"]["hidden_layers"] != [192, 192]
            or policy["policy_model"]["branch_width"] != 64):
        raise ValueError("Expected original Experiment 10 policy and architecture")
    if not need_replay:
        return None, metadata
    state = safe_path(root, row["training_state_path"])
    manifest = None
    if state.is_file():
        checked(state, row["training_state_sha256"])
        payload = torch.load(state, map_location="cpu", weights_only=False)
    else:
        checked(state / "manifest.json", row["training_state_sha256"])
        manifest = json.loads((state / "manifest.json").read_text())
        state_metadata = checked(safe_path(state, manifest["metadata_path"]), manifest["metadata_sha256"])
        payload = torch.load(state_metadata, map_location="cpu", weights_only=False)
    config = payload["config"]
    policy = target.native.checkpoint
    if (payload.get("schema_version") != 1 or payload.get("seed") != seed
            or payload.get("checkpoint_id") != "time_24h"
            or policy["seed"] != seed or policy["experiment_id"] != 10
            or policy["checkpoint_target_seconds"] != 86400
            or config["feature_encoder_id"] != "fhp_lossless_hand_board_v2"
            or config["gamma"] != 2 or policy["input_size"] != 213
            or payload["solver"]["training_elapsed_seconds"] < 86400):
        raise ValueError("Expected original 24h Experiment 10 hand/board policy")
    if (policy["policy_model"]["type"] != "structured_fhp_mlp_v1"
            or policy["policy_model"]["hidden_layers"] != [192, 192]
            or policy["policy_model"]["branch_width"] != 64
            or payload["solver"]["num_iteration"] != policy["outer_iteration"]):
        raise ValueError("Source policy architecture/iteration mismatch")
    for key, value in target.native.model.state_dict().items():
        if not torch.equal(value, payload["average_policy_trainer"]["model"][key]):
            raise ValueError("Policy and saved replay state disagree")
    buffer = payload["average_policy_trainer"]["buffer"]
    records = {r["path"]: r for r in manifest["array_shards"]} if manifest else {}
    arrays = []
    for key in ("infostate", "q_value", "q_value_mask", "iteration"):
        if manifest is None:
            arrays.append(np.asarray(buffer[key])[:buffer["size"]])
            continue
        relative = buffer[key][MARKER]
        record = records[relative]
        path = checked(safe_path(state, relative), record["sha256"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError("Replay shard layout mismatch")
        arrays.append(array[:buffer["size"]])
    return Replay(*arrays, iteration=payload["solver"]["num_iteration"]), metadata
