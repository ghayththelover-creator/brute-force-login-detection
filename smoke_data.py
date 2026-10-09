"""Generates a SYNTHETIC smoke-test CSV (NOT valid for paper results). Usage: python smoke_data.py out.csv [seed]"""
import sys, numpy as np, pandas as pd
out = sys.argv[1]; rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
D = 14 * 86400; rows = []
users = [f"u{i}" for i in range(200)]; ips = [f"10.0.{i//250}.{i%250}" for i in range(120)]
def ev(t, s, u, ok, lab, sc): rows.append((t, s, u, 'success' if ok else 'fail', lab, sc))
for _ in range(60000):                       # benign with typos, diurnal
    t = rng.uniform(0, D); h = (t % 86400) / 3600
    if rng.random() > 0.3 + 0.7 * np.exp(-((h - 11) ** 2) / 18): continue
    u = users[rng.integers(200)]; s = ips[hash(u) % 120]
    if rng.random() < 0.04:
        ev(t, s, u, 0, 0, 'benign'); ev(t + rng.uniform(5, 40), s, u, 1, 0, 'benign')
    else: ev(t, s, u, 1, 0, 'benign')
for _ in range(14):                          # single-source bursts
    t0 = rng.uniform(0, D - 600); s = f"203.0.113.{rng.integers(250)}"; u = users[rng.integers(200)]
    for k in range(rng.integers(30, 100)): ev(t0 + rng.uniform(0, 90), s, u, 0, 1, 'single_source')
for _ in range(14):                          # spraying
    t0 = rng.uniform(0, D - 600); s = f"198.51.100.{rng.integers(250)}"
    for u in rng.choice(users, 60): ev(t0 + rng.uniform(0, 110), s, u, 0, 1, 'spraying')
for _ in range(10):                          # distributed
    t0 = rng.uniform(0, D - 900); tu = rng.choice(users, 2)
    for i in range(30):
        s = f"192.0.2.{rng.integers(250)}.{i}"
        for _ in range(rng.integers(2, 4)): ev(t0 + rng.uniform(0, 300), s, tu[rng.integers(2)], 0, 1, 'distributed')
for _ in range(8):                           # low-and-slow
    t0 = rng.uniform(0, D - 86400); s = f"100.64.{rng.integers(250)}.{rng.integers(250)}"; u = users[rng.integers(200)]
    t = t0
    while t < t0 + 86400: ev(t, s, u, 0, 1, 'low_and_slow'); t += rng.uniform(900, 4500)
pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome', 'label', 'scenario']).sort_values('timestamp').to_csv(out, index=False)
print("wrote", out, len(rows), "events")
