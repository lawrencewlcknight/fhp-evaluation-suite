#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
FHP_SAFEGUARD_WORK=/workspace/fhp-safeguard-exp34
mkdir -p "$FHP_SAFEGUARD_WORK/output"
exec 9>"$FHP_SAFEGUARD_WORK/upload.lock"
flock -w 360 9
if [[ -f "$FHP_SAFEGUARD_WORK/bootstrap.log" ]]; then
  cp "$FHP_SAFEGUARD_WORK/bootstrap.log" "$FHP_SAFEGUARD_WORK/output/bootstrap.log"
fi
for attempt in 1 2; do
  if timeout 240 gcloud storage rsync --recursive --exclude='(^inputs/|.*[.]tmp$|.*[.]lock$)' \
    "$FHP_SAFEGUARD_WORK/output" "$FHP_DESTINATION"; then exit 0; fi
  sleep 10
done
echo 'Final upload failed' >&2
exit 1
