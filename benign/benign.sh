#!/bin/bash
# SPEEDED UP (~15x) for a 24-48h lab run instead of 14 days, to get a usable benign:attack ratio quickly.
set -u
until nc -z "$TARGET_HOST" 22; do sleep 1; done
declare -A CREDS=( [alice]="Summer2026!" [bob]='Blue$kyToday' [carol]="Horse!Battery9" [dave]="Welcome#123" [erin]="Riverside!7" )
end=$((SECONDS + ${DURATION_S:-1209600}))
while [ $SECONDS -lt $end ]; do
  hour=$(( ($(date +%s) / 3600) % 24 ))
  if [ "$hour" -ge 0 ] && [ "$hour" -lt 6 ] && [ $((RANDOM % 5)) -ne 0 ]; then sleep 20; continue; fi
  u=$(shuf -e "${!CREDS[@]}" -n1); pass="${CREDS[$u]}"
  if [ $((RANDOM % 25)) -eq 0 ]; then
    sshpass -p "${pass}x" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$u@$TARGET_HOST" true 2>/dev/null
    sleep $(( (RANDOM % 3) + 1 ))
  fi
  sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$u@$TARGET_HOST" true 2>/dev/null
  sleep $(( (RANDOM % 40) + 5 ))
done
