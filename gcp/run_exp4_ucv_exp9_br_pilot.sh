#!/usr/bin/env bash
set -Eeuo pipefail
# Batch can supply a minimal environment. Bash's fallback PATH need not be
# exported to Python, which must resolve its base executable when creating venvs.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
WORK=/workspace/fhp-br-exp9
OUTPUT="$WORK/output"
mkdir -p "$OUTPUT" "$WORK/source" "$WORK/input"
exec > >(tee -a "$OUTPUT/bootstrap.log") 2>&1
FHP_MAIN_STAGE=initialization
stage() {
  FHP_MAIN_STAGE="$1"
  printf '[%s] MAIN stage=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FHP_MAIN_STAGE"
}
trap 'printf "ERROR: MAIN stage=%s line=%s exit_code=%s\n" "$FHP_MAIN_STAGE" "$LINENO" "$?" >&2' ERR
SYNC_PID=""
sync_outputs() {
  flock -w 5 "$WORK/upload.lock" timeout 45 gcloud storage rsync --recursive --exclude='.*[.]tmp$' \
    "$OUTPUT" "$FHP_DESTINATION/analysis"
}
cleanup() {
  exit_code="$?"
  set +e
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  printf '{"main_exit_code":%s,"stage":"%s"}\n' "$exit_code" "$FHP_MAIN_STAGE" > "$OUTPUT/main_exit.json"
  sync_outputs || echo "WARNING: exit upload failed; the final Batch runnable will retry."
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Preserve bootstrap diagnostics too, not just evaluator outputs.
while sleep 60; do sync_outputs || echo "WARNING: periodic output upload failed"; done &
SYNC_PID="$!"

stage source_download
timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
cp "$WORK/source/source_manifest.json" "$OUTPUT/source_manifest.json"
stage checkpoint_download
timeout 180 gcloud storage cp "$FHP_CHECKPOINT_URI" "$WORK/input/policy.pkl"
printf '%s  %s\n' "$FHP_CHECKPOINT_SHA256" "$WORK/input/policy.pkl" | sha256sum -c -
printf '{"uri":"%s","sha256":"%s"}\n' "$FHP_CHECKPOINT_URI" "$FHP_CHECKPOINT_SHA256" > "$OUTPUT/checkpoint_identity.json"

# Batch script runnables run as root by default. Keep setup inside a 20-minute cap.
stage python_setup
timeout --signal=TERM --kill-after=15 1200 /bin/bash -Eeuo pipefail <<'FHP_BOOTSTRAP'
FHP_BOOTSTRAP_STAGE=initialization
bootstrap_stage() {
  FHP_BOOTSTRAP_STAGE="$1"
  printf '[%s] BOOTSTRAP stage=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FHP_BOOTSTRAP_STAGE"
}
trap 'printf "ERROR: BOOTSTRAP stage=%s line=%s exit_code=%s\n" "$FHP_BOOTSTRAP_STAGE" "$LINENO" "$?" >&2' ERR
bootstrap_stage system_packages
apt-get update
apt-get install -y python3 python3-venv ca-certificates
bootstrap_stage python_diagnostics
printf 'PATH=%s\n' "$PATH"
command -v python3
# Use Debian's interpreter explicitly; -I also excludes inherited Python
# environment overrides and user-site packages from this bootstrap environment.
/usr/bin/python3 -I -c 'import json, os, sys; print(json.dumps(dict(version=sys.version, executable=sys.executable, base_executable=sys._base_executable, prefix=sys.prefix))); assert sys._base_executable and os.access(sys._base_executable, os.X_OK), "Base Python executable is missing or not executable"'
bootstrap_stage bootstrap_venv
/usr/bin/python3 -I -m venv /workspace/fhp-br-exp9/bootstrap-venv
bootstrap_stage uv_install
/workspace/fhp-br-exp9/bootstrap-venv/bin/pip install --disable-pip-version-check uv==0.8.22
export UV_PYTHON_INSTALL_DIR=/workspace/fhp-br-exp9/python
UV=/workspace/fhp-br-exp9/bootstrap-venv/bin/uv
bootstrap_stage managed_python
"$UV" python install 3.11.13
bootstrap_stage evaluation_venv
"$UV" venv --python 3.11.13 --seed /workspace/fhp-br-exp9/venv
PIP=/workspace/fhp-br-exp9/venv/bin/pip
bootstrap_stage torch_install
"$PIP" install --disable-pip-version-check --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
bootstrap_stage evaluation_dependencies
"$PIP" install --disable-pip-version-check --no-cache-dir -r /workspace/fhp-br-exp9/source/evaluator/gcp/requirements-br-pilot.txt
bootstrap_stage dependency_check
"$PIP" check
bootstrap_stage complete
FHP_BOOTSTRAP
stage runtime_diagnostics
PYTHON="$WORK/venv/bin/python"
"$PYTHON" -m pip freeze > "$OUTPUT/pip_freeze.txt"
"$PYTHON" --version > "$OUTPUT/python_version.txt"
export PYTHONPATH="$WORK/source/evaluator"
cd "$WORK/source/evaluator"
ARGS=()
if [[ "$FHP_SMOKE" == 1 ]]; then ARGS=(--smoke); fi
stage evaluation
"$PYTHON" -m fhp_evaluation.best_response.pilot \
  --checkpoint "$WORK/input/policy.pkl" --repo-root "$WORK/source/native" \
  --output-dir "$OUTPUT" "${ARGS[@]}"
stage complete
