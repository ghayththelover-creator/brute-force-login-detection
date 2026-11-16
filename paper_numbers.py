#!/usr/bin/env python3
"""numbers.py - pool raw run files into the exact values quoted in the revised paper (numbers.json)."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

R = 'results_synth'


def agg(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return dict(mean=m, sd=sd, lo=m - h, hi=m + h, n=len(v), min=(v.min() if len(v) else np.nan), max=(v.max() if len(v) else np.nan))


def pool(df, key, cols):
    out = {}
    for k, g in df.groupby(key, sort=False):
        k = k if isinstance(k, str) else '|'.join(map(str, k))
        out[k] = {c: agg(g[c]) for c in cols if c in g}
    return out


N = {}
MET = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mcc', 'tpr_at_fpr1', 'tpr_at_fpr01', 'brier',
       'rec_single_source', 'rec_spraying', 'rec_distributed', 'rec_low_and_slow', 'TP', 'FP', 'FN', 'TN',
       'n_single_source', 'det_single_source', 'n_spraying', 'det_spraying', 'n_distributed', 'det_distributed',
       'n_low_and_slow', 'det_low_and_slow', 'alerts_per_10k_benign']
d = pd.read_csv(f'{R}/main_raw.csv'); N['main'] = pool(d, 'detector', MET)
N['params'] = d.groupby('detector')['params'].agg(lambda s: s.value_counts().to_dict()).dropna().to_dict()
c = pd.read_csv(f'{R}/corpus_and_overlap.csv'); N['corpus'] = {k: agg(c[k]) for k in c.columns if k != 'run'}
N['corpus_run0'] = c.iloc[0].to_dict()
m = pd.read_csv(f'{R}/mcnemar_raw.csv')
N['mcnemar'] = {f'{a}|{b}': dict(b=g['A_right_B_wrong'].mean(), c=g['B_right_A_wrong'].mean(),
                                  sig=int((g['p'] < 0.05).sum()), med_p=float(g['p'].median()), max_p=float(g['p'].max()))
                for (a, b), g in m.groupby(['A', 'B'])}
imp = pd.read_csv(f'{R}/importance_raw.csv')
N['importance'] = {f: g.groupby('feature')['mean_abs_shap'].mean().sort_values(ascending=False).to_dict()
                   for f, g in imp.groupby('features')}
for f, key in (('ablation_raw.csv', 'config'), ('ablation2_raw.csv', 'config'), ('stress_raw.csv', ['stale_sources', 'features'])):
    if os.path.exists(f'{R}/{f}'):
        N[f.replace('_raw.csv', '')] = pool(pd.read_csv(f'{R}/{f}'), key, MET + ['f1_at_0_5', 'precision_at_0_5', 'recall_at_0_5'])
w = pd.read_csv(f'{R}/windows_raw.csv')
w['feat'] = np.where(w['features'].str.startswith('failure-count'), 'failure-count', w['features'])
N['windows'] = pool(w, ['window_s', 'overlap50', 'feat'], MET)
N['windows_k'] = {f'{a}|{b}': v for (a, b), v in w[w.feat == 'failure-count'].groupby(['window_s', 'overlap50'])['features'].agg(lambda s: s.value_counts().to_dict()).items()}
N['windows_n'] = {f'{a}|{b}': v for (a, b), v in w.groupby(['window_s', 'overlap50'])['windows'].mean().items()}
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=str)
print('ok', list(N))
