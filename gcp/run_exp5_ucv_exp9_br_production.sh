#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
TASK_INDEX="${BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}"
if (( TASK_INDEX < 0 || TASK_INDEX >= 12 )); then echo "Invalid task index: $TASK_INDEX" >&2; exit 2; fi
TRAINING_SEED=$(( TASK_INDEX / 4 ))
SHARD_INDEX=$(( TASK_INDEX % 4 ))
TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_seed_${TRAINING_SEED}_shard_${SHARD_INDEX}"
export WORK="/workspace/fhp-br-exp9-production-${TASK_INDEX}"
OUTPUT="$WORK/output/workers/$TASK_NAME"
REMOTE="$FHP_DESTINATION/workers/$TASK_NAME"
mkdir -p "$OUTPUT" "$WORK/source" "$WORK/input"
exec > >(tee -a "$OUTPUT/bootstrap.log") 2>&1
MAIN_STAGE=initialization
stage() { MAIN_STAGE="$1"; printf '[%s] stage=%s task=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MAIN_STAGE" "$TASK_NAME"; }
SYNC_PID=""
sync_outputs() {
  flock -w 5 "$WORK/upload.lock" timeout 90 gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUTPUT" "$REMOTE"
}
cleanup() {
  exit_code="$?"; set +e
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  printf '{"main_exit_code":%s,"stage":"%s"}\n' "$exit_code" "$MAIN_STAGE" > "$OUTPUT/main_exit.json"
  sync_outputs || echo "WARNING: exit upload failed; final runnable will retry."
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

stage restore
REMOTE_OBJECT="$(gcloud storage objects list "$REMOTE/**" --exhaustive --limit=1 --format='value(name)')"
if [[ -n "$REMOTE_OBJECT" ]]; then
  gcloud storage rsync --recursive "$REMOTE" "$OUTPUT"
fi
while sleep 60; do sync_outputs || echo "WARNING: periodic upload failed"; done & SYNC_PID="$!"

stage source_download
timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
cp "$WORK/source/source_manifest.json" "$OUTPUT/source_manifest.json"

case "$TRAINING_SEED" in
  0) CHECKPOINT_URI="$FHP_CHECKPOINT_URI_0"; CHECKPOINT_SHA256="$FHP_CHECKPOINT_SHA256_0" ;;
  1) CHECKPOINT_URI="$FHP_CHECKPOINT_URI_1"; CHECKPOINT_SHA256="$FHP_CHECKPOINT_SHA256_1" ;;
  2) CHECKPOINT_URI="$FHP_CHECKPOINT_URI_2"; CHECKPOINT_SHA256="$FHP_CHECKPOINT_SHA256_2" ;;
  *) echo "Invalid training seed" >&2; exit 2 ;;
esac
stage checkpoint_download
timeout 180 gcloud storage cp "$CHECKPOINT_URI" "$WORK/input/policy.pkl"
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$WORK/input/policy.pkl" | sha256sum -c -
printf '{"uri":"%s","sha256":"%s","training_seed":%s}\n' \
  "$CHECKPOINT_URI" "$CHECKPOINT_SHA256" "$TRAINING_SEED" > "$OUTPUT/checkpoint_identity.json"

stage python_setup
timeout --signal=TERM --kill-after=15 1200 /bin/bash -Eeuo pipefail <<'FHP_BOOTSTRAP'
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv "$WORK/bootstrap-venv"
"$WORK/bootstrap-venv/bin/pip" install --disable-pip-version-check uv==0.8.22
export UV_PYTHON_INSTALL_DIR="$WORK/python"
UV="$WORK/bootstrap-venv/bin/uv"
"$UV" python install 3.11.13
"$UV" venv --python 3.11.13 --seed "$WORK/venv"
"$WORK/venv/bin/pip" install --disable-pip-version-check --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$WORK/venv/bin/pip" install --disable-pip-version-check --no-cache-dir -r "$WORK/source/evaluator/gcp/requirements-br-pilot.txt"
"$WORK/venv/bin/pip" check
FHP_BOOTSTRAP

stage evaluation
export PYTHONPATH="$WORK/source/evaluator"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
ARGS=()
if [[ "$FHP_SMOKE" == 1 ]]; then ARGS=(--smoke); fi
"$WORK/venv/bin/python" -m fhp_evaluation.best_response.production worker \
  --task-index "$TASK_INDEX" --checkpoint "$WORK/input/policy.pkl" \
  --repo-root "$WORK/source/native" --output-dir "$OUTPUT" \
  --source-bundle-sha256 "$FHP_BUNDLE_SHA256" "${ARGS[@]}"
stage complete
