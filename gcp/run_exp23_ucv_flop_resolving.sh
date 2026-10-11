#!/usr/bin/env bash
# VM runnable; use the training-repo launch wrapper from your laptop.
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
: "${FHP_SEARCH_MODE:?Prepared bundle must specify gameplay or feasibility mode}"
case "$FHP_SEARCH_MODE" in gameplay|feasibility) ;; *) echo 'Invalid search mode' >&2; exit 2 ;; esac
export FHP_SEARCH_WORK=/workspace/fhp-resolving-exp23
OUTPUT="$FHP_SEARCH_WORK/output"
mkdir -p "$OUTPUT" "$FHP_SEARCH_WORK/source" "$FHP_SEARCH_WORK/input"
exec > >(tee -a "$FHP_SEARCH_WORK/bootstrap.log") 2>&1
SYNC_PID=""
sync_outputs() {
  flock -w 10 "$FHP_SEARCH_WORK/upload.lock" timeout 300 gcloud storage rsync --recursive \
    --exclude='(^inputs/|.*[.]tmp$|.*[.]lock$)' "$OUTPUT" "$FHP_DESTINATION"
}
cleanup() {
  code="$?"; set +e
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  cp "$FHP_SEARCH_WORK/bootstrap.log" "$OUTPUT/bootstrap.log"
  printf '{"main_exit_code":%s}\n' "$code" > "$OUTPUT/main_exit.json"
  sync_outputs || echo 'Exit upload failed; final runnable will retry' >&2
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
REMOTE_OBJECT="$(gcloud storage objects list "$FHP_DESTINATION/status.json" --limit=1 --format='value(name)')"
if [[ -n "$REMOTE_OBJECT" ]]; then
  gcloud storage rsync --recursive --exclude='(^inputs/|.*[.]tmp$|.*[.]lock$)' "$FHP_DESTINATION" "$OUTPUT"
fi
while sleep 60; do
  cp "$FHP_SEARCH_WORK/bootstrap.log" "$OUTPUT/bootstrap.log"
  { date -u; free -m; } >> "$OUTPUT/resources.log"
  sync_outputs || echo 'Periodic upload failed' >&2
done & SYNC_PID="$!"
timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$FHP_SEARCH_WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$FHP_SEARCH_WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$FHP_SEARCH_WORK/source.tar.gz" -C "$FHP_SEARCH_WORK/source"
cp "$FHP_SEARCH_WORK/source/source_manifest.json" "$OUTPUT/source_manifest.json"
timeout --signal=TERM --kill-after=15 1800 /bin/bash -Eeuo pipefail <<'BOOTSTRAP'
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv "$FHP_SEARCH_WORK/bootstrap-venv"
"$FHP_SEARCH_WORK/bootstrap-venv/bin/pip" install --disable-pip-version-check uv==0.8.22
export UV_PYTHON_INSTALL_DIR="$FHP_SEARCH_WORK/python"
UV="$FHP_SEARCH_WORK/bootstrap-venv/bin/uv"
"$UV" python install 3.11.13
"$UV" venv --python 3.11.13 --managed-python --no-python-downloads --seed "$FHP_SEARCH_WORK/venv"
"$FHP_SEARCH_WORK/venv/bin/python" -I -c 'import sys; assert sys.version_info[:3] == (3,11,13), sys.version'
"$FHP_SEARCH_WORK/venv/bin/pip" install --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$FHP_SEARCH_WORK/venv/bin/pip" install --no-cache-dir -r "$FHP_SEARCH_WORK/source/evaluator/gcp/requirements-br-pilot.txt"
"$FHP_SEARCH_WORK/venv/bin/pip" check
BOOTSTRAP
export PYTHONPATH="$FHP_SEARCH_WORK/source/evaluator"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
"$FHP_SEARCH_WORK/venv/bin/pip" freeze > "$OUTPUT/pip_freeze.txt"
"$FHP_SEARCH_WORK/venv/bin/python" - <<'DOWNLOAD'
import json, os, subprocess
from pathlib import Path
from fhp_evaluation.resolving.experiment import required_sources
from fhp_evaluation.loaders import sha256_file
work = Path(os.environ["FHP_SEARCH_WORK"])
sources = required_sources(os.environ['FHP_SEARCH_MODE'])
for name, record in sources.items():
    uri = f"{os.environ['FHP_SOURCE_BUCKET']}/{record['run']}/{record['relative']}"
    target = work / "input" / f"{name}.pkl"
    subprocess.run(["gcloud", "storage", "cp", uri, str(target)], check=True, timeout=180)
    if sha256_file(target) != record["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {name}")
(work / "output/checkpoint_sources.json").write_text(json.dumps(sources, indent=2))
DOWNLOAD
EXTRA_ARGS=()
SEARCH_TIMEOUT=166000
if [[ "$FHP_SEARCH_MODE" == feasibility ]]; then
  EXTRA_ARGS+=(--preflight-only)
  SEARCH_TIMEOUT=600
fi
timeout --signal=TERM --kill-after=60 "$SEARCH_TIMEOUT" "$FHP_SEARCH_WORK/venv/bin/python" \
  -m fhp_evaluation.resolving.runner --input-root "$FHP_SEARCH_WORK/input" \
  --native-root "$FHP_SEARCH_WORK/source/native" --output "$OUTPUT" "${EXTRA_ARGS[@]}"
