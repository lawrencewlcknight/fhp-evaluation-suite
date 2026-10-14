#!/usr/bin/env bash
# VM runnable: invoke the Python Batch launcher, not this file, locally.
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
export WORK=/workspace/fhp-sd6-audit
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
REMOTE_OBJECT="$(timeout 180 gcloud storage objects list "$FHP_DESTINATION/status.json" --limit=1 --format='value(name)')"
if [[ -n "$REMOTE_OBJECT" ]]; then
  timeout 1800 gcloud storage rsync --recursive --exclude='(^inputs/|.*[.]lock$)' "$FHP_DESTINATION" "$OUTPUT"
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
FHP_BOOTSTRAP_STAGE=initialization
bootstrap_stage() {
  FHP_BOOTSTRAP_STAGE="$1"
  printf '[%s] BOOTSTRAP stage=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FHP_BOOTSTRAP_STAGE"
}
trap 'printf "ERROR: BOOTSTRAP stage=%s line=%s exit_code=%s\n" "$FHP_BOOTSTRAP_STAGE" "$LINENO" "$?" >&2' ERR
FHP_AUDIT_PYTHON_VERSION=3.11.13
FHP_AUDIT_UV_VERSION=0.8.22
FHP_AUDIT_PYTHON_REQUEST="cpython-${FHP_AUDIT_PYTHON_VERSION}-linux-x86_64-gnu"
bootstrap_stage system_packages
apt-get -o DPkg::Lock::Timeout=120 update
apt-get -o DPkg::Lock::Timeout=120 install -y python3 python3-venv ca-certificates git
bootstrap_stage bootstrap_venv
/usr/bin/python3 -I -m venv "$WORK/bootstrap-venv"
bootstrap_stage uv_install
"$WORK/bootstrap-venv/bin/pip" install --disable-pip-version-check "uv==$FHP_AUDIT_UV_VERSION"
export UV_PYTHON_INSTALL_DIR="$WORK/python"
UV="$WORK/bootstrap-venv/bin/uv"
bootstrap_stage python_availability
"$UV" python list "$FHP_AUDIT_PYTHON_REQUEST" --all-versions --all-platforms --all-arches \
  --only-downloads --output-format json --offline --no-cache --no-config > "$WORK/output/python_downloads.json"
/usr/bin/python3 -I - "$WORK/output/python_downloads.json" "$FHP_AUDIT_PYTHON_REQUEST" <<'PYTHON_CATALOG_CHECK'
import json, sys
with open(sys.argv[1]) as stream:
    rows = json.load(stream)
matches = [r for r in rows if isinstance(r, dict) and r.get("key") == sys.argv[2]
           and isinstance(r.get("url"), str) and r["url"].startswith("https://")] if isinstance(rows, list) else []
if len(matches) != 1:
    raise SystemExit(f"Pinned uv cannot supply {sys.argv[2]}; align the Python and uv pins before retrying.")
print(f"Verified managed Python download: {matches[0]['key']}")
PYTHON_CATALOG_CHECK
bootstrap_stage managed_python
"$UV" python install "$FHP_AUDIT_PYTHON_VERSION"
bootstrap_stage evaluation_venv
"$UV" venv --python "$FHP_AUDIT_PYTHON_VERSION" --managed-python --no-python-downloads --seed "$WORK/venv"
bootstrap_stage runtime_version
"$WORK/venv/bin/python" -I -c 'import sys; assert sys.version_info[:3] == tuple(map(int, sys.argv[1].split("."))), sys.version' "$FHP_AUDIT_PYTHON_VERSION"
bootstrap_stage torch_install
"$WORK/venv/bin/pip" install --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
bootstrap_stage evaluation_dependencies
"$WORK/venv/bin/pip" install --no-cache-dir -r "$WORK/source/evaluator/gcp/requirements-br-pilot.txt"
bootstrap_stage dependency_check
"$WORK/venv/bin/pip" check
bootstrap_stage complete
BOOTSTRAP
export PYTHONPATH="$WORK/source/evaluator"
IMPORT_ARGS=()
if [[ -n "${FHP_IMPORT_SOURCE:-}" ]]; then
  mkdir -p "$WORK/import/tables"
  timeout 180 gcloud storage cp "$FHP_IMPORT_SOURCE/manifest.json" "$WORK/import/manifest.json"
  timeout 1800 gcloud storage rsync --recursive "$FHP_IMPORT_SOURCE/tables" "$WORK/import/tables"
  IMPORT_ARGS=(--import-sd-cache "$WORK/import")
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
"$WORK/venv/bin/python" --version > "$OUTPUT/python_version.txt"
"$WORK/venv/bin/pip" freeze > "$OUTPUT/pip_freeze.txt"
"$WORK/venv/bin/python" -m fhp_evaluation.strategic_audit.sd_exp6 \
  --stage "$FHP_STAGE" --input-root "$WORK/input" --native-root "$WORK/source/native" \
  --output "$OUTPUT" --download "${IMPORT_ARGS[@]}"
