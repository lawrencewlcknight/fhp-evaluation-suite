#!/usr/bin/env bash
# VM runnable, invoked by the Python Batch launcher; not a local launch script.
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
export WORK=/workspace/fhp-strategic-exp16
OUTPUT="$WORK/output"
mkdir -p "$OUTPUT" "$WORK/source" "$WORK/input"
exec > >(tee -a "$WORK/bootstrap.log") 2>&1
SYNC_PID=""
sync_outputs() {
  flock -w 10 "$WORK/upload.lock" timeout 300 gcloud storage rsync --recursive \
    --exclude='(^inputs/|.*[.]tmp([.]npz)?$|.*[.]lock$)' "$OUTPUT" "$FHP_DESTINATION"
}
cleanup() {
  code="$?"; set +e
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  cp "$WORK/bootstrap.log" "$OUTPUT/bootstrap.log"
  printf '{"main_exit_code":%s}\n' "$code" > "$OUTPUT/main_exit.json"
  sync_outputs || echo "Exit upload failed; final runnable will retry" >&2
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
REMOTE_OBJECT="$(gcloud storage objects list "$FHP_DESTINATION/status.json" --limit=1 --format='value(name)')"
if [[ -n "$REMOTE_OBJECT" ]]; then
  gcloud storage rsync --recursive --exclude='(^inputs/|.*[.]lock$)' "$FHP_DESTINATION" "$OUTPUT"
fi
while sleep 60; do
  cp "$WORK/bootstrap.log" "$OUTPUT/bootstrap.log"
  { date -u; free -m; df -h /workspace; } >> "$OUTPUT/resources.log"
  sync_outputs || echo "Periodic upload failed" >&2
done & SYNC_PID="$!"

timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
cp "$WORK/source/source_manifest.json" "$OUTPUT/source_manifest.json"

timeout --signal=TERM --kill-after=15 1800 /bin/bash -Eeuo pipefail <<'BOOTSTRAP'
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv "$WORK/bootstrap-venv"
"$WORK/bootstrap-venv/bin/pip" install --disable-pip-version-check uv==0.8.22
export UV_PYTHON_INSTALL_DIR="$WORK/python"
UV="$WORK/bootstrap-venv/bin/uv"
"$UV" python install 3.11.16
"$UV" venv --python 3.11.16 --seed "$WORK/venv"
"$WORK/venv/bin/pip" install --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$WORK/venv/bin/pip" install --no-cache-dir -r "$WORK/source/evaluator/gcp/requirements-br-pilot.txt"
"$WORK/venv/bin/pip" check
BOOTSTRAP
export PYTHONPATH="$WORK/source/evaluator"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
"$WORK/venv/bin/python" --version > "$OUTPUT/python_version.txt"
"$WORK/venv/bin/pip" freeze > "$OUTPUT/pip_freeze.txt"
"$WORK/venv/bin/python" - <<'DOWNLOAD'
import json, os, subprocess
from pathlib import Path
from fhp_evaluation.strategic_audit.exp16 import checkpoint_registry
from fhp_evaluation.loaders import sha256_file
work = Path(os.environ["WORK"])
sources = checkpoint_registry(os.environ["FHP_STAGE"])
for name, item in sources.items():
    uri = f"{os.environ['FHP_SOURCE_BUCKET']}/{item['run']}/{item['relative']}"
    target = work / "input" / f"{name}.pkl"
    subprocess.run(["gcloud", "storage", "cp", uri, str(target)], check=True, timeout=180)
    if sha256_file(target) != item["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {name}")
(work / "output/checkpoint_sources.json").write_text(json.dumps(sources, indent=2))
DOWNLOAD
"$WORK/venv/bin/python" -m fhp_evaluation.strategic_audit.exp16 \
  --stage "$FHP_STAGE" --input-root "$WORK/input" --native-root "$WORK/source/native" \
  --output "$OUTPUT" --threads 4
