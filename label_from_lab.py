#!/usr/bin/env python3
"""
label_from_lab.py - turn the real lab's auth.log + network_map.json into D1.csv for bf_eval.py.
Ground truth comes from container role (real sshd, real attack tools), not from a heuristic.
Usage: python label_from_lab.py --log real_lab_export/auth.log --map real_lab_export/network_map.json --out D1.csv --salt SECRET
"""
import argparse, hashlib, json, re
import pandas as pd
RE = re.compile(r'^(?P<ts>\d{4}-\d\d-\d\dT[\d:.]+(?:Z|[+-]\d\d:?\d\d))?.*?'
                r'(?P<res>Failed|Accepted)\s+(?:password|publickey|keyboard-interactive(?:/pam)?)\s+'
                r'for\s+(?:invalid user\s+)?(?P<user>\S+)\s+from\s+(?P<ip>\S+)')
ROLE_SCEN = {'attacker_single': 'single_source', 'attacker_spray': 'spraying',
             'attacker_dist': 'distributed', 'attacker_slow': 'low_and_slow'}
def ip_role_map(net):
    j = json.load(open(net)); out = {}
    for net_ in j:
        for cid, info in net_.get('Containers', {}).items():
            name = info.get('Name', '')
            ip = info.get('IPv4Address', '').split('/')[0]
            role = next((r for r in ('attacker_single', 'attacker_spray', 'attacker_dist', 'attacker_slow',
                                     'benign_user', 'sshd_server') if r in name), 'unknown')
            if ip: out[ip] = role
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--log', required=True); ap.add_argument('--map', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--salt', default=None)
    ap.add_argument('--year', type=int, default=pd.Timestamp.now().year)
    a = ap.parse_args()
    roles = ip_role_map(a.map)
    rows = []; unknown_ips = set()
    for line in open(a.log, errors='replace'):
        m = RE.search(line)
        if not m: continue
        ip = m['ip']; role = roles.get(ip)
        if role is None: unknown_ips.add(ip); continue
        if role == 'sshd_server': continue
        label = 0 if role == 'benign_user' else 1
        scen = ROLE_SCEN.get(role, 'benign') if label else 'benign'
        ts = m['ts'] if m['ts'] else pd.Timestamp.now(tz='UTC').isoformat()
        rows.append((ts, ip, m['user'], 'fail' if m['res'] == 'Failed' else 'success', label, scen))
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome', 'label', 'scenario'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')
    df = df.sort_values('timestamp')
    if a.salt:
        h = lambda x: hashlib.sha256((a.salt + str(x)).encode()).hexdigest()[:16]
        df['source'] = df['source'].map(h); df['username'] = df['username'].map(h)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} events, {int((df.label==1).sum())} attack events")
    print(df.groupby('scenario').size().to_string())
    if unknown_ips: print(f"NOTE: {len(unknown_ips)} source IPs not found in network_map.json (ignored): {sorted(unknown_ips)[:5]}...")
if __name__ == '__main__': main()
