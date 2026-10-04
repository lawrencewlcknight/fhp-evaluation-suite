"""Process-isolated evaluation with wall-clock/RSS watchdog and durable diagnostics."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(spec):
    from ..loaders import sha256_file
    package = Path(__file__).resolve().parents[1]
    value = dict(spec=spec, evaluator_source_sha256=source_hash(package))
    if spec.get("checkpoint"):
        path = Path(spec["checkpoint"])
        value["checkpoint_sha256"] = sha256_file(path)
        if spec.get("family") == "sd_cfr":
            manifest = json.loads(path.read_text())
            # Fresh workers cryptographically check every chunk. Resuming a
            # completed measurement additionally requires unchanged file stamps.
            value["archive_file_stamps"] = [
                [c["path"], (path.parent / c["path"]).stat().st_size,
                 (path.parent / c["path"]).stat().st_mtime_ns] for c in manifest["chunks"]]
    if spec.get("repo_root"):
        root = Path(spec["repo_root"])
        packages = {"ucv": ("fhp_escher", "vr_deep_cfr"),
                    "vr_deep": ("fhp_vr_deep", "vr_deep_cfr"),
                    "sd_cfr": ("deep_cfr_poker",)}[spec["family"]]
        value["native_source_sha256"] = {p: source_hash(root / p) for p in packages}
    return value


def run_bounded(spec, directory, *, max_seconds=300., max_rss_mb=4096., resume=False,
                command=None):
    if max_seconds <= 0 or max_rss_mb <= 0:
        raise ValueError("Positive runtime and RSS limits required")
    directory = Path(directory).resolve()
    request = dict(identity=identity(spec), max_seconds=max_seconds, max_rss_mb=max_rss_mb)
    if directory.exists():
        record = directory / "run.json"
        if resume and record.exists():
            previous = json.loads(record.read_text())
            if previous["request"] == request and previous["status"] == "succeeded":
                return previous
        raise FileExistsError(f"Refusing to overwrite {directory}; choose a new output directory")
    directory.mkdir(parents=True)
    spec_path = directory / "spec.json"
    write_json(spec_path, spec)
    env = dict(os.environ)
    threads = str(spec.get("threads", 1))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = threads
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = command or [sys.executable, "-m", "fhp_evaluation.best_response.worker", str(spec_path)]
    started, peak, status = time.monotonic(), 0, "running"
    record = dict(request=request, status=status, pid=None, peak_rss_bytes=0,
                  resource_limit_kind="process_tree_RSS_watchdog", poll_seconds=.1)
    write_json(directory / "run.json", record)
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        except OSError as error:
            record.update(status="failed", returncode=None, error=f"Worker launch failed: {error}",
                          elapsed_seconds=time.monotonic() - started)
            write_json(directory / "run.json", record)
            return record
        try:
            record["pid"] = process.pid
            write_json(directory / "run.json", record)
            with (directory / "resources.jsonl").open("w") as resources:
                while process.poll() is None:
                    rss = 0
                    try:
                        monitor = psutil.Process(process.pid)
                        members = [monitor, *monitor.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        members = []
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    resources.write(json.dumps(dict(elapsed_seconds=elapsed, rss_bytes=rss)) + "\n")
                    resources.flush()
                    if rss > max_rss_mb * 1024**2:
                        status = "memory_limit"
                        break
                    if elapsed > max_seconds:
                        status = "time_limit"
                        break
                    time.sleep(.1)
        except (psutil.Error, OSError) as error:
            status = "monitor_error"
            record["error"] = f"Resource monitoring unavailable; refusing an unbounded run: {error}"
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            if process.poll() is None:
                # Only the worker's freshly created process group is signalled.
                import signal
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if status == "running":
            status = "succeeded" if process.returncode == 0 else "failed"
    record.update(status=status, returncode=process.returncode, peak_rss_bytes=peak,
                  elapsed_seconds=time.monotonic() - started)
    if status == "succeeded" and (directory / "result.json").exists():
        try:
            record["result"] = json.loads((directory / "result.json").read_text())
        except (OSError, ValueError) as error:
            record.update(status="failed", error=f"Invalid worker result: {error}")
    elif status == "succeeded":
        record["status"] = "failed"
        record["error"] = "Worker exited without a result"
    write_json(directory / "run.json", record)
    return record
