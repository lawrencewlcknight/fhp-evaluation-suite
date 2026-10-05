#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
WORK=/workspace/fhp-br-exp9-production-aggregate
mkdir -p "$WORK/source" "$WORK/output/workers" "$WORK/output/analysis"
timeout 180 gcloud storage cp "$FHP_BUNDLE_URI" "$WORK/source.tar.gz"
printf '%s  %s\n' "$FHP_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --disable-pip-version-check --no-cache-dir -r "$WORK/source/evaluator/gcp/requirements-br-pilot.txt"
gcloud storage rsync --recursive "$FHP_DESTINATION/workers" "$WORK/output/workers"
export PYTHONPATH="$WORK/source/evaluator"
ARGS=()
if [[ "$FHP_SMOKE" == 1 ]]; then ARGS=(--smoke); fi
"$WORK/venv/bin/python" -m fhp_evaluation.best_response.production aggregate \
  --workers-root "$WORK/output/workers" --output-dir "$WORK/output/analysis" \
  --source-bundle-sha256 "$FHP_BUNDLE_SHA256" "${ARGS[@]}"
gcloud storage rsync --recursive "$WORK/output/analysis" "$FHP_DESTINATION/analysis"
