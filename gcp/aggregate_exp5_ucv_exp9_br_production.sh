#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
export WORK=/workspace/fhp-br-exp9-production-aggregate
mkdir -p "$WORK/source" "$WORK/output/workers" "$WORK/output/analysis"
timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
# Importing the shared response package loads its Torch adapters even though
# aggregation itself performs no inference. Use the workers' pinned runtime.
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
gcloud storage rsync --recursive "$FHP_DESTINATION/workers" "$WORK/output/workers"
export PYTHONPATH="$WORK/source/evaluator"
ARGS=()
if [[ "$FHP_SMOKE" == 1 ]]; then ARGS=(--smoke); fi
"$WORK/venv/bin/python" -m fhp_evaluation.best_response.production aggregate \
  --workers-root "$WORK/output/workers" --output-dir "$WORK/output/analysis" \
  --source-bundle-sha256 "$FHP_BUNDLE_SHA256" "${ARGS[@]}"
"$WORK/venv/bin/python" --version > "$WORK/output/analysis/python_version.txt"
"$WORK/venv/bin/pip" freeze > "$WORK/output/analysis/pip_freeze.txt"
cp "$WORK/source/source_manifest.json" "$WORK/output/analysis/source_manifest.json"
gcloud storage rsync --recursive "$WORK/output/analysis" "$FHP_DESTINATION/analysis"
