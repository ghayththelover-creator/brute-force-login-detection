#!/usr/bin/env python3
"""
run_real.py - revised real-traffic (Docker lab, D1) experiments, same protocol as run_synthetic.py.

  python run_real.py --data D1.csv --out results_real --seeds 10

D1.csv columns: timestamp, source, username, outcome, label, scenario (from label_from_lab.py).
Outputs: real_main_raw.csv (all detectors x seeds, with TP/FP/TN/FN and per-scenario counts),
real_windows_raw.csv (60/120/300/900 s, with and without 50% overlap), real_ablation_raw.csv,
real_mcnemar.csv, real_split_info.csv. Strict chronological 60/20/20 split, no campaign re-grouping.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf_eval as B
from run_synthetic import (SINGLE, STATEFUL, TIMING, chrono_split, overlap_stats, train_score, evaluate,
                           rule_baselines)
from scipy.stats import binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='results_real')
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ev = B.load_events(a.data)
    rows, info, mc = [], [], []
    W = B.build_windows(ev, 120, False); part = chrono_split(W)
    y, scen, s = W['label'].values, W['scenario'].values, W['s'].values
    tr, va, te = part == 0, part == 1, part == 2
    o = overlap_stats(W, part); o.update(events=len(ev), windows=len(W), attack_windows=int(y.sum()),
                                         train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))
    for k, nm in enumerate(['train', 'val', 'test']):
        for sc in sorted(set(scen)):
            o[f'{nm}_{sc}'] = int(((part == k) & (scen == sc)).sum())
    info.append(o); pd.DataFrame(info).to_csv(f"{a.out}/real_split_info.csv", index=False)
    print(o)
    unseen = ~np.isin(s[te], list(set(s[tr & (y == 1)])))
    preds = {}
    for det, (sc_, pr, p) in rule_baselines(W, part).items():
        m = evaluate(y[te], sc_, pr, scen[te], unseen); m.update(seed=0, detector=det, params=str(p)); rows.append(m)
        preds[(det, 0)] = pr
    for fs, cols in (('single-window', SINGLE), ('stateful', STATEFUL)):
        X = W[cols].values.astype(float)
        for name in ('LR', 'RF', 'XGB', 'MLP', 'Tree-d3'):
            for seed in range(a.seeds):
                _, sv, st = train_score(name, X[tr], y[tr], X[va], X[te], seed)
                thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
                m = evaluate(y[te], st, pr, scen[te], unseen); m.update(seed=seed, detector=f'{name}, {fs}')
                rows.append(m); preds[(f'{name}, {fs}', seed)] = pr
    pd.DataFrame(rows).to_csv(f"{a.out}/real_main_raw.csv", index=False)
    yt = y[te]
    for A, Bn in [('XGB, stateful', 'XGB, single-window'), ('XGB, single-window', 'Failure-count rule'),
                  ('XGB, single-window', 'fail2ban-style rule'), ('XGB, single-window', 'Count + source-diversity rule'),
                  ('XGB, stateful', 'Count + diversity + history rule'), ('LR, single-window', 'Failure-count rule')]:
        for seed in range(a.seeds):
            pa = preds[(A, seed)]; pb = preds.get((Bn, seed), preds.get((Bn, 0)))
            ca, cb = pa == yt, pb == yt; b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
            mc.append(dict(seed=seed, A=A, B=Bn, A_right_B_wrong=b_, B_right_A_wrong=c_, p=p))
    pd.DataFrame(mc).to_csv(f"{a.out}/real_mcnemar.csv", index=False)
    # ablations
    abl = []
    conf = {'stateful': STATEFUL, 'single-window': SINGLE,
            'single-window w/o success_count': [c for c in SINGLE if c != 'success_count'],
            'stateful w/o success_count': [c for c in STATEFUL if c != 'success_count'],
            'single-window w/o max_distinct_src_per_user': [c for c in SINGLE if c != 'max_distinct_src_per_user'],
            'single-window w/o timing features': [c for c in SINGLE if c not in TIMING]}
    for nm, cols in conf.items():
        X = W[cols].values.astype(float)
        for sm in (True, False):
            for seed in range(a.seeds):
                _, sv, st = train_score('XGB', X[tr], y[tr], X[va], X[te], seed, smote=sm)
                thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
                m = evaluate(y[te], st, pr, scen[te]); m.update(seed=seed, config=nm, smote=sm); abl.append(m)
    pd.DataFrame(abl).to_csv(f"{a.out}/real_ablation_raw.csv", index=False)
    # window sensitivity
    win = []
    for d in (60, 120, 300, 900):
        for ov in (False, True):
            Wd = B.build_windows(ev, d, ov); pt = chrono_split(Wd); yy = Wd['label'].values; sc2 = Wd['scenario'].values
            a_, b_, c_ = pt == 0, pt == 1, pt == 2
            if yy[b_].sum() == 0 or yy[a_].sum() == 0:
                print(f"window {d}s overlap={ov}: no attack windows in a partition, skipped"); continue
            for fs, cols in (('single-window', SINGLE), ('stateful', STATEFUL)):
                X = Wd[cols].values.astype(float)
                for seed in range(a.seeds):
                    _, sv, st = train_score('XGB', X[a_], yy[a_], X[b_], X[c_], seed)
                    thr = B.best_threshold(yy[b_], sv); pr = (st >= thr).astype(int)
                    m = evaluate(yy[c_], st, pr, sc2[c_]); m.update(seed=seed, window_s=d, overlap50=ov, features=fs,
                                                                   windows=len(Wd)); win.append(m)
            f = Wd['fail_count'].values
            k = max(range(1, 11), key=lambda k: B.metrics(yy[b_], f[b_].astype(float), (f[b_] >= k).astype(int))['f1'])
            pr = (f[c_] >= k).astype(int); m = evaluate(yy[c_], pr.astype(float), pr, sc2[c_])
            m.update(seed=0, window_s=d, overlap50=ov, features=f'failure-count (k={k})', windows=len(Wd)); win.append(m)
    pd.DataFrame(win).to_csv(f"{a.out}/real_windows_raw.csv", index=False)
    print("done ->", a.out)


if __name__ == '__main__':
    main()
