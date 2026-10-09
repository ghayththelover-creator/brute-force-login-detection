#!/usr/bin/env python3
"""
inject_attacks.py - inject labelled brute-force campaigns into REAL benign authentication logs.
  py inject_attacks.py --in benign.csv --out D1.csv
Input : CSV from prepare_logs.py (or any CSV with timestamp,source,username,outcome[,label]); all rows are treated as benign (label 0).
Output: CSV with label (0/1) and scenario columns for bf_eval.py, plus <out>_manifest.csv listing every injected campaign.
Campaign start times are spread evenly over the whole time span so train/val/test each contain campaigns.
REPORT IN THE PAPER: "real benign background + injected attack scenarios with known ground truth", and the parameters below.
"""
import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--n-single', type=int, default=12); ap.add_argument('--n-spray', type=int, default=12)
ap.add_argument('--n-dist', type=int, default=8); ap.add_argument('--n-slow', type=int, default=8)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
b = pd.read_csv(a.inp)
ts = b['timestamp']
t = ts.astype(float).values if pd.api.types.is_numeric_dtype(ts) else \
    (pd.to_datetime(ts, utc=True, format='mixed') - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
b = pd.DataFrame({'timestamp': t, 'source': b['source'].astype(str), 'username': b['username'].astype(str),
                  'outcome': b['outcome'].astype(str).str.lower(), 'label': 0, 'scenario': 'benign'})
t0, t1 = t.min(), t.max(); span = t1 - t0
if span < 7 * 86400: print(f"WARNING: only {span/86400:.1f} days of logs; >=14 days recommended (each partition needs >=10 attack windows).")
real_users = b['username'].unique().tolist()
common = ['root', 'admin', 'test', 'ubuntu', 'user', 'oracle', 'postgres', 'ftpuser', 'guest', 'pi', 'git', 'support']
pool = real_users + common
rows, man = [], []
def starts(n, margin):   # one start per equal slot -> spread across time
    edges = np.linspace(t0, t1 - margin, n + 1)
    return [rng.uniform(edges[i], edges[i + 1]) for i in range(n)]
def add(ts_, src, u, sc, camp):
    rows.append((ts_, src, u, 'fail', 1, sc)); man.append(camp)
cid = 0
for s0 in starts(a.n_single, 600):                       # single-source high-rate
    cid += 1; src = f"atk-single-{cid}"; u = rng.choice(pool); n = int(rng.integers(30, 101))
    for _ in range(n): add(s0 + rng.uniform(0, 90), src, u, 'single_source', cid)
for s0 in starts(a.n_spray, 600):                        # credential spraying
    cid += 1; src = f"atk-spray-{cid}"
    for u in rng.choice(pool, size=min(60, len(pool)), replace=False): add(s0 + rng.uniform(0, 110), src, u, 'spraying', cid)
for s0 in starts(a.n_dist, 900):                         # distributed
    cid += 1; targets = rng.choice(pool, 2, replace=False)
    for i in range(30):
        for _ in range(int(rng.integers(2, 4))): add(s0 + rng.uniform(0, 300), f"atk-dist-{cid}-{i}", rng.choice(targets), 'distributed', cid)
for s0 in starts(a.n_slow, 86400 + 600):                 # low-and-slow (1 failure every 15-75 min for 24 h)
    cid += 1; src = f"atk-slow-{cid}"; targets = rng.choice(pool, int(rng.integers(1, 3)), replace=False); tt = s0
    while tt < s0 + 86400: add(tt, src, rng.choice(targets), 'low_and_slow', cid); tt += rng.uniform(900, 4500)
atk = pd.DataFrame(rows, columns=b.columns)
out = pd.concat([b, atk], ignore_index=True).sort_values('timestamp')
out.to_csv(a.out, index=False)
m = atk.assign(campaign=man).groupby(['campaign', 'scenario']).agg(start=('timestamp', 'min'), events=('timestamp', 'size')).reset_index()
m.to_csv(a.out.replace('.csv', '') + '_manifest.csv', index=False)
print(f"benign events: {len(b)}  injected attack events: {len(atk)}  campaigns: {len(m)}")
print(atk.groupby('scenario').size().to_string())
print('wrote', a.out)
