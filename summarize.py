#!/usr/bin/env python3
"""summarize.py - aggregate raw run files into the tables used in the revised paper (mean, sd, 95% CI)."""
import sys, os
import numpy as np, pandas as pd
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else 'results_synth'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def table(df, by, cols):
    out = []
    for k, g in df.groupby(by, sort=False):
        r = dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,)))
        for c in cols:
            v = g[c].astype(float)
            lo, hi = ci(v)
            r[c] = v.mean(); r[c + '_sd'] = v.std(ddof=1); r[c + '_lo'] = lo; r[c + '_hi'] = hi
        r['n'] = len(g); out.append(r)
    return pd.DataFrame(out)


MET = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mcc', 'tpr_at_fpr1', 'tpr_at_fpr01']
if os.path.exists(f'{R}/main_raw.csv'):
    d = pd.read_csv(f'{R}/main_raw.csv')
    T = table(d, 'detector', MET + ['rec_single_source', 'rec_spraying', 'rec_distributed', 'rec_low_and_slow',
                                    'rec_unseen_attack', 'TP', 'FP', 'FN', 'TN', 'brier'])
    T.to_csv(f'{R}/T_main.csv', index=False)
    show = ['detector', 'n', 'precision', 'precision_sd', 'recall', 'recall_sd', 'f1', 'f1_sd', 'f1_lo', 'f1_hi', 'pr_auc', 'pr_auc_sd',
            'fpr', 'mcc', 'mcc_sd', 'rec_single_source', 'rec_spraying', 'rec_distributed', 'rec_low_and_slow', 'rec_low_and_slow_sd',
            'rec_unseen_attack', 'TP', 'FP', 'FN', 'TN', 'brier']
    print(T[show].round(4).to_string())
    cnt = d.groupby('detector')[[c for c in d.columns if c.startswith('n_') or c.startswith('det_')]].mean()
    print(cnt.round(1).to_string())
    print(d.groupby('detector')['params'].agg(lambda s: s.value_counts().to_dict()).dropna().to_string())
    c = pd.read_csv(f'{R}/corpus_and_overlap.csv'); print(c.describe().T[['mean', 'std', 'min', 'max']].round(1).to_string())
    m = pd.read_csv(f'{R}/mcnemar_raw.csv')
    print(m.groupby(['A', 'B']).agg(b=('A_right_B_wrong', 'mean'), c=('B_right_A_wrong', 'mean'),
                                     sig=('p', lambda p: int((p < 0.05).sum())), med_p=('p', 'median')).round(4).to_string())
    imp = pd.read_csv(f'{R}/importance_raw.csv')
    print(imp.groupby(['features', 'feature'])['mean_abs_shap'].agg(['mean', 'std']).round(4).sort_values(['features', 'mean'], ascending=[True, False]).to_string())
for f, by in (('ablation_raw.csv', 'config'), ('ablation2_raw.csv', 'config'), ('windows_raw.csv', ['window_s', 'overlap50', 'features']),
              ('stress_raw.csv', ['stale_sources', 'features'])):
    if os.path.exists(f'{R}/{f}'):
        d = pd.read_csv(f'{R}/{f}')
        extra = [c for c in ('f1_at_0_5', 'precision_at_0_5', 'recall_at_0_5', 'brier') if c in d.columns]
        T = table(d, by, ['precision', 'recall', 'f1', 'pr_auc', 'fpr', 'mcc', 'rec_low_and_slow', 'rec_distributed'] + extra)
        T.to_csv(f'{R}/T_{f.replace("_raw", "")}', index=False)
        cols = (by if isinstance(by, list) else [by]) + ['n', 'precision', 'recall', 'f1', 'f1_sd', 'pr_auc', 'pr_auc_sd', 'fpr', 'mcc',
                                                       'rec_low_and_slow', 'rec_distributed'] + extra
        print('\n==', f); print(T[cols].round(4).to_string())
