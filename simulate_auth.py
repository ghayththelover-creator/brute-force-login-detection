#!/usr/bin/env python3
"""
simulate_auth.py - synthetic authentication-log generator (Section 4 of the paper).

14 days, 400 benign accounts, 300 benign source IPs (some shared, NAT-like), diurnal
benign activity with occasional mistyped passwords, and four attack scenarios:
single-source high-rate, credential spraying, distributed, low-and-slow.
Campaigns of every scenario are spread across the whole time span (stratified start
times) so that each chronological partition receives attack campaigns.

Output CSV columns: timestamp (s), source, username, outcome, label, scenario, campaign
Usage: python simulate_auth.py --seed 0 --out sim_seed0.csv
"""
import argparse
import numpy as np
import pandas as pd

DAY = 86400
P_TYPO, P_GIVEUP, RETRY_MED = 0.055, 0.010, 80.0   # benign typo model
SPAN = 14 * DAY


def diurnal_times(rng, n, span):
    """Draw n timestamps with a working-hours profile and lower weekend activity."""
    hours = np.arange(24)
    w = 0.15 + np.exp(-0.5 * ((hours - 13) / 3.5) ** 2)          # peak early afternoon
    w = w / w.sum()
    day = rng.integers(0, span // DAY, n)
    wk = (day % 7) >= 5                                           # days 5,6 = weekend
    keep = rng.random(n) < np.where(wk, 0.45, 1.0)
    while not keep.all():                                         # re-draw dropped weekend sessions
        m = ~keep
        day[m] = rng.integers(0, span // DAY, m.sum())
        keep[m] = rng.random(m.sum()) < np.where((day[m] % 7) >= 5, 0.45, 1.0)
    hr = rng.choice(24, n, p=w)
    return day * DAY + hr * 3600 + rng.random(n) * 3600


def benign(rng, n_users=400, n_ips=300, target_events=156000):
    users = np.array([f"user{i:03d}" for i in range(n_users)])
    home = rng.integers(0, n_ips, n_users)                        # 400 users on 300 IPs -> some sharing
    second = np.where(rng.random(n_users) < 0.2, rng.integers(0, n_ips, n_users), -1)
    rate = rng.gamma(2.0, 1.0, n_users); rate /= rate.sum()
    n_sess = int(target_events / (1 + P_TYPO + 0.012 + P_GIVEUP))
    who = rng.choice(n_users, n_sess, p=rate)
    t = diurnal_times(rng, n_sess, SPAN)
    use2 = (second[who] >= 0) & (rng.random(n_sess) < 0.3)
    ip = np.where(use2, second[who], home[who])
    rows = []
    u_typo = rng.random(n_sess)
    for i in range(n_sess):
        src = f"10.0.{ip[i] // 256}.{ip[i] % 256}"; u = users[who[i]]; ti = t[i]
        r = u_typo[i]
        if r < P_TYPO:                                            # one typo, then success after a pause
            rows.append((ti, src, u, 'fail'))
            ti += rng.lognormal(np.log(RETRY_MED), 1.3); rows.append((ti, src, u, 'success'))
        elif r < P_TYPO + 0.006:                                  # two typos, then success
            rows.append((ti, src, u, 'fail')); ti += rng.lognormal(np.log(RETRY_MED), 1.3)
            rows.append((ti, src, u, 'fail')); ti += rng.lognormal(np.log(RETRY_MED), 1.3)
            rows.append((ti, src, u, 'success'))
        elif r < P_TYPO + 0.006 + P_GIVEUP:                       # gives up after one failure
            rows.append((ti, src, u, 'fail'))
        else:
            rows.append((ti, src, u, 'success'))
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome'])
    df['label'] = 0; df['scenario'] = 'benign'; df['campaign'] = -1
    return df, users


def starts(rng, n, dur):
    """Stratified start times: campaign i begins inside the i-th slice of the span."""
    slot = (SPAN - dur) / n
    return np.array([(i + rng.random()) * slot for i in range(n)])


def attacks(rng, users):
    rows, cid = [], 0
    common = np.array(['root', 'admin', 'test', 'oracle', 'ubuntu', 'guest', 'postgres', 'ftp'])
    pool = np.r_[users, common]
    nxt = [0]

    def new_ip():
        nxt[0] += 1
        return f"203.0.{nxt[0] // 256}.{nxt[0] % 256}"

    # (1) single-source high-rate: 12 campaigns x ~522 attempts at 0.25-0.6 attempts/s
    for s in starts(rng, 12, 3000):
        src, tg = new_ip(), rng.choice(pool, rng.integers(1, 4), replace=False)
        rate = rng.uniform(0.25, 0.6); t = s
        for _ in range(522):
            t += rng.exponential(1 / rate); rows.append((t, src, rng.choice(tg), 'fail', 'single_source', cid))
        cid += 1
    # (2) credential spraying: 12 campaigns, ~3 passwords over ~175 accounts, 0.15-0.3 attempts/s
    for s in starts(rng, 12, 4000):
        src = new_ip(); acc = rng.choice(pool, 174, replace=False); t = s
        rate = rng.uniform(0.15, 0.3)
        for _ in range(3):
            for u in acc:
                t += rng.exponential(1 / rate); rows.append((t, src, u, 'fail', 'spraying', cid))
        cid += 1
    # (3) distributed: 8 campaigns, 25-40 sources sharing 3-5 target accounts, ~3 attempts / source / 120 s
    for s in starts(rng, 8, 6 * 3600):
        n_src = rng.integers(25, 41); tg = rng.choice(pool, rng.integers(3, 6), replace=False)
        per = 824 // n_src
        for _ in range(n_src):
            src = new_ip(); t = s + rng.random() * 600
            for _ in range(per):
                t += rng.exponential(40); rows.append((t, src, rng.choice(tg), 'fail', 'distributed', cid))
        cid += 1
    # (4) low-and-slow: 20 sources x 66 attempts, gaps U(15, 75) min, 1-3 target accounts
    for s in starts(rng, 20, 66 * 45 * 60):
        src, tg = new_ip(), rng.choice(pool, rng.integers(1, 4), replace=False); t = s
        for _ in range(66):
            t += rng.uniform(15, 75) * 60; rows.append((t, src, rng.choice(tg), 'fail', 'low_and_slow', cid))
        cid += 1
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome', 'scenario', 'campaign'])
    df['label'] = 1
    return df


def stale_credentials(rng, users, n):
    """Stress test only (not in the main corpus): benign hosts whose scripts keep retrying a stale
    password every 15-75 min for 1-3 days, i.e. benign sources that look like low-and-slow attackers."""
    rows = []
    for k, s in enumerate(starts(rng, n, 3 * DAY)):
        src, u, t, end = f"10.9.{k // 256}.{k % 256}", rng.choice(users), s, s + rng.uniform(1, 3) * DAY
        while t < end:
            t += rng.uniform(15, 75) * 60; rows.append((t, src, u, 'fail'))
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'username', 'outcome'])
    df['label'] = 0; df['scenario'] = 'benign'; df['campaign'] = -1
    return df


def simulate(seed, stale=0):
    rng = np.random.default_rng(seed)
    b, users = benign(rng)
    a = attacks(rng, users)
    parts = [b, a]
    if stale:
        parts.append(stale_credentials(np.random.default_rng(10_000 + seed), users, stale))
    df = pd.concat(parts, ignore_index=True)
    df = df[df['timestamp'] < SPAN].sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', 'source', 'username', 'outcome', 'label', 'scenario', 'campaign']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='sim.csv')
    ap.add_argument('--stale', type=int, default=0, help='stress test: number of benign stale-credential sources')
    a = ap.parse_args()
    d = simulate(a.seed, a.stale)
    d.to_csv(a.out, index=False)
    print(d.groupby('scenario').size().to_string(), '\ntotal', len(d))
