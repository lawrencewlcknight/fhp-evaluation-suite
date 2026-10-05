#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
TASK_INDEX="${BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}"
TRAINING_SEED=$(( TASK_INDEX / 4 )); SHARD_INDEX=$(( TASK_INDEX % 4 ))
TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_seed_${TRAINING_SEED}_shard_${SHARD_INDEX}"
WORK="/workspace/fhp-br-exp9-production-${TASK_INDEX}"
OUTPUT="$WORK/output/workers/$TASK_NAME"
REMOTE="$FHP_DESTINATION/workers/$TASK_NAME"
mkdir -p "$OUTPUT"
exec 9>"$WORK/upload.lock"
flock -w 60 9
{ date -u; free -m || true; df -h /workspace || true; } > "$OUTPUT/final_system_diagnostics.txt"
for attempt in 1 2; do
  if timeout 300 gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUTPUT" "$REMOTE"; then exit 0; fi
  sleep 10
done
echo "Final shard upload failed" >&2
exit 1
