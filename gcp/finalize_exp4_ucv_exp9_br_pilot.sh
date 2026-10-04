#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
OUTPUT=/workspace/fhp-br-exp9/output
mkdir -p "$OUTPUT"
exec 9>/workspace/fhp-br-exp9/upload.lock
flock -w 60 9
# This separate alwaysRun runnable is still eligible after main failure/timeout.
# It cannot survive VM loss or the task's overall deadline; periodic upload helps.
{
  date -u
  free -m || true
  df -h /workspace || true
  if [[ -r /sys/fs/cgroup/memory.events ]]; then cat /sys/fs/cgroup/memory.events; fi
  dmesg 2>&1 | tail -80 || true
} > "$OUTPUT/final_system_diagnostics.txt"
for attempt in 1 2; do
  if timeout 240 gcloud storage rsync --recursive --exclude='.*[.]tmp$' \
       "$OUTPUT" "$FHP_DESTINATION/analysis"; then
    echo "Final upload completed: $FHP_DESTINATION/analysis/"
    exit 0
  fi
  echo "WARNING: final upload attempt $attempt failed" >&2
  sleep 10
done
echo "ERROR: results upload failed; inspect Cloud Logging." >&2
exit 1
