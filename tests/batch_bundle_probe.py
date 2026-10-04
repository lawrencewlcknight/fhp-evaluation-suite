"""Fresh-process acceptance of the source-only deployment bundle, no cloud calls."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("builder", root / "gcp/exp4_ucv_exp9_br_pilot_batch.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = args.output_dir / "source.tar.gz"
    builder.make_bundle(root, args.native_repo, archive_path)
    extracted = args.output_dir / "source"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted, filter="data")
    import os
    env = dict(os.environ, PYTHONPATH=str(extracted / "evaluator"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Use a different saved UCV checkpoint if Exp9's checkpoint is not downloaded.
    # Production CLI does not offer this override and always enforces the Exp9 hash.
    command = [sys.executable, "-c",
               "from fhp_evaluation.best_response.pilot import run_pilot; "
               "from fhp_evaluation.loaders import sha256_file; import sys; "
               "run_pilot(sys.argv[1],sys.argv[2],sys.argv[3],smoke=True,"
               "expected_sha256=sha256_file(sys.argv[1]))",
               str(args.checkpoint.resolve()), str(extracted / "native"), str(args.output_dir / "result")]
    subprocess.run(command, cwd=extracted, env=env, check=True, timeout=900)


if __name__ == "__main__":
    main()
