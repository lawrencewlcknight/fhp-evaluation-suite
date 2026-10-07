#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
WORK=/workspace/fhp-strategic-exp16
mkdir -p "$WORK/output"
exec 9>"$WORK/upload.lock"
flock -w 360 9
if [[ -f "$WORK/bootstrap.log" ]]; then cp "$WORK/bootstrap.log" "$WORK/output/bootstrap.log"; fi
for attempt in 1 2; do
  if timeout 360 gcloud storage rsync --recursive --exclude='(^inputs/|.*[.]tmp([.]npz)?$|.*[.]lock$)' \
    "$WORK/output" "$FHP_DESTINATION"; then exit 0; fi
  sleep 10
done
echo 'Final upload failed' >&2
exit 1
