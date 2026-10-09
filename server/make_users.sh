#!/bin/bash
set -e
# Real benign accounts. Edit/add lines "name:pass" to mirror your real user base.
USERS="alice:Summer2026! bob:Blue\$kyToday carol:Horse!Battery9 dave:Welcome#123 erin:Riverside!7"
for up in $USERS; do
  name="${up%%:*}"; pass="${up##*:}"
  useradd -m -s /bin/bash "$name" 2>/dev/null || true
  echo "$name:$pass" | chpasswd
done
ssh-keygen -A
# sshd's own debug/file log (-E) has NO timestamp per line (no syslog daemon in this container).
# Prefix every line with a real UTC timestamp as it is written, so downstream analysis gets
# genuine event times instead of "now at export time" for everything.
/usr/sbin/sshd -D -e 2>&1 | while IFS= read -r line; do
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)" "$line"
done >> /var/log/auth.log
