#!/usr/bin/env python3
"""
prepare_logs.py - convert raw logs to the event CSV used by bf_eval.py
  python prepare_logs.py --out d1.csv --year 2026 --salt SECRET \
      /var/log/auth.log:sshd:0   cowrie.json:cowrie:1
Each input is  path:format:label  (format = sshd | cowrie ; label 0 = benign source, 1 = attack source)
 - sshd  : classic syslog ("Oct  5 07:53:36 host sshd[12]: Failed password for invalid user admin from 1.2.3.4 port 22 ssh2")
           or RFC3339 timestamps. "Failed ..." -> fail, "Accepted ..." -> success.
 - cowrie: JSON lines with eventid cowrie.login.failed / cowrie.login.success, username, src_ip, timestamp.
Labels are per FILE: use label 1 only for files whose every source is known malicious (honeypot) and
label 0 for institutional logs whose attack periods you do NOT know; if a benign file contains known
incidents, add a 'label' column manually from incident tickets before running bf_eval.py.
--salt hashes source IPs and usernames (SHA-256, 16 hex chars) so the CSV can be stored/shared (ethics statement).
"""
import argparse, hashlib, json, re, sys
import pandas as pd
SYS = re.compile(r'^(?P<mon>[A-Z][a-z]{2})\s+(?P<d>\d+)\s+(?P<hms>\d\d:\d\d:\d\d)\s+\S+\s+sshd(?:\[\d+\])?:\s+'
                 r'(?P<res>Failed|Accepted)\s+\S+\s+for\s+(?:invalid user\s+)?(?P<user>\S+)\s+from\s+(?P<ip>\S+)')
ISO = re.compile(r'^(?P<ts>\d{4}-\d\d-\d\dT[\d:.]+(?:Z|[+-]\d\d:?\d\d))\s+\S+\s+sshd(?:\[\d+\])?:\s+'
                 r'(?P<res>Failed|Accepted)\s+\S+\s+for\s+(?:invalid user\s+)?(?P<user>\S+)\s+from\s+(?P<ip>\S+)')
def parse_sshd(path, year, tz):
    out = []
    for line in open(path, errors='replace'):
        m = ISO.match(line)
        if m: ts = pd.Timestamp(m['ts'])
        else:
            m = SYS.match(line)
            if not m: continue
            ts = pd.Timestamp(f"{year} {m['mon']} {m['d']} {m['hms']}", tz=tz).tz_convert('UTC')
        out.append((ts.isoformat(), m['ip'], m['user'], 'fail' if m['res'] == 'Failed' else 'success'))
    return out
def parse_cowrie(path):
    out = []
    for line in open(path, errors='replace'):
        try: j = json.loads(line)
        except Exception: continue
        e = j.get('eventid', '')
        if e not in ('cowrie.login.failed', 'cowrie.login.success'): continue
        out.append((j['timestamp'], j.get('src_ip', ''), j.get('username', ''), 'fail' if e.endswith('failed') else 'success'))
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inputs', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--year', type=int, default=pd.Timestamp.now().year); ap.add_argument('--tz', default='UTC')
    ap.add_argument('--salt', default=None); a = ap.parse_args()
    frames = []
    for spec in a.inputs:
        path, fmt, lab = spec.rsplit(':', 2)
        rows = parse_sshd(path, a.year, a.tz) if fmt == 'sshd' else parse_cowrie(path)
        df = pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome']); df['label'] = int(lab)
        print(f"{path}: {len(df)} login events, {int((df.outcome=='fail').sum())} failures"); frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')
    df = df.sort_values('timestamp')
    if a.salt:
        h = lambda x: hashlib.sha256((a.salt + str(x)).encode()).hexdigest()[:16]
        df['source'] = df['source'].map(h); df['username'] = df['username'].map(h)
    df.to_csv(a.out, index=False); print('wrote', a.out, len(df))
if __name__ == '__main__': main()
