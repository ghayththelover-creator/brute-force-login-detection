#!/usr/bin/env python3
"""
bf_eval.py - Real-data evaluation pipeline for the paper
"Brute-Force Login Detection from Authentication Logs Using Machine Learning"
Implements Sections 5.2, 5.2.1, 6.2 and 7.7 of the revised paper.

Input  : one CSV per dataset with columns
         timestamp, source, username, outcome, label   (+ optional: scenario, service)
         outcome: fail/failed/failure/0  or  success/accepted/ok/1
         label  : 0 = benign event, 1 = attack event
Output : <out>/raw_runs.csv, table11_overall.csv, table12_per_scenario.csv, ablation.csv,
         window_sensitivity.csv, cross_dataset.csv, mcnemar.csv, latency.csv,
         tables_for_paper.md, run_config.json
"""
import argparse, json, os, sys, time, warnings, platform
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             precision_recall_curve, matthews_corrcoef)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
warnings.filterwarnings("ignore")

SINGLE = ['fail_count', 'success_count', 'fail_ratio', 'attempt_rate', 'distinct_users',
          'username_entropy', 'gap_mean', 'gap_std', 'max_distinct_src_per_user']
HN = {600: '10m', 3600: '1h', 86400: '24h'}
STATEFUL = [f'{k}_{nm}' for nm in ('1h', '24h') for k in ('F', 'rho', 'D', 'A', 'mu', 'cv')]
FAIL_SET = {'fail', 'failed', 'failure', '0', 'false', 'f'}

# ------------------------------------------------------------------ loading
def load_events(path):
    df = pd.read_csv(path)
    need = {'timestamp', 'source', 'username', 'outcome', 'label'}
    if need - set(df.columns):
        sys.exit(f"{path}: missing columns {sorted(need - set(df.columns))}")
    ts = df['timestamp']
    if pd.api.types.is_numeric_dtype(ts):
        t = ts.astype(float).values
    else:
        t = (pd.to_datetime(ts, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
    ev = pd.DataFrame({
        't': t,
        's': pd.factorize(df['source'].astype(str))[0],
        'u': pd.factorize(df['username'].astype(str))[0],
        'fail': df['outcome'].astype(str).str.lower().isin(FAIL_SET).astype(np.int8).values,
        'label': df['label'].astype(int).values})
    ev['scen'] = df['scenario'].astype(str).values if 'scenario' in df.columns else \
        np.where(ev['label'] == 1, 'attack', 'benign')
    return ev.sort_values(['s', 't']).reset_index(drop=True)

# ------------------------------------------------------------------ features
def stateful_feats(ev, Widx, delta, horizons=(3600, 86400), extra=(600,)):
    allH = sorted(set(horizons) | set(extra))
    n = len(Widx); out = {}
    for H in allH:
        nm = HN[H]; out[f'F_{nm}'] = np.zeros(n)
        if H in horizons:
            for k in ('rho', 'D', 'A', 'mu', 'cv'): out[f'{k}_{nm}'] = np.zeros(n)
    es, et, eu, ef = ev['s'].values, ev['t'].values, ev['u'].values, ev['fail'].values.astype(np.int8)
    ws, wend = Widx['s'].values, Widx['end'].values
    us, e0 = np.unique(es, return_index=True); e1 = np.r_[e0[1:], len(es)]
    emap = {s: (a, b) for s, a, b in zip(us, e0, e1)}
    wu, w0 = np.unique(ws, return_index=True); w1 = np.r_[w0[1:], len(ws)]
    for s, a, b in zip(wu, w0, w1):
        ea, eb = emap[s]
        t, u, f = et[ea:eb], eu[ea:eb], ef[ea:eb]
        ends = wend[a:b]
        cf = np.r_[0, np.cumsum(f)]
        tf = t[f == 1]
        slot = np.floor(tf / delta).astype(np.int64)
        gf = np.diff(tf) if len(tf) > 1 else np.array([])
        cg, cg2 = np.r_[0, np.cumsum(gf)], np.r_[0, np.cumsum(gf ** 2)]
        nwf = np.ones(len(tf), dtype=np.int64)
        if len(tf) > 1: nwf[1:] = (slot[1:] != slot[:-1])
        cnw = np.cumsum(nwf)
        for H in allH:
            nm = HN[H]
            r = np.searchsorted(t, ends, 'left'); l = np.searchsorted(t, ends - H, 'left')
            F = cf[r] - cf[l]
            out[f'F_{nm}'][a:b] = F
            if H not in horizons: continue
            tot = r - l
            out[f'rho_{nm}'][a:b] = np.where(tot > 0, F / np.maximum(tot, 1), 0.0)
            if len(tf):
                rf = np.searchsorted(tf, ends, 'left'); lf = np.searchsorted(tf, ends - H, 'left')
                nf = rf - lf
                A = np.zeros(b - a); m = nf > 0
                A[m] = cnw[rf[m] - 1] - cnw[lf[m]] + 1
                out[f'A_{nm}'][a:b] = A
                mu = np.zeros(b - a); cv = np.zeros(b - a); m2 = nf >= 2
                if m2.any():
                    k = (nf[m2] - 1).astype(float)
                    S = cg[rf[m2] - 1] - cg[lf[m2]]; S2 = cg2[rf[m2] - 1] - cg2[lf[m2]]
                    mean = S / k; var = np.maximum(S2 / k - mean ** 2, 0)
                    mu[m2] = mean; cv[m2] = np.where(mean > 0, np.sqrt(var) / np.where(mean > 0, mean, 1), 0)
                out[f'mu_{nm}'][a:b] = mu; out[f'cv_{nm}'][a:b] = cv
            D = np.zeros(b - a); cnt = {}; lp = rp = dist = 0
            for i in range(b - a):
                while rp < r[i]:
                    x = u[rp]; c = cnt.get(x, 0); cnt[x] = c + 1; dist += (c == 0); rp += 1
                while lp < l[i]:
                    x = u[lp]; c = cnt[x] - 1; cnt[x] = c; dist -= (c == 0); lp += 1
                D[i] = dist
            out[f'D_{nm}'][a:b] = D
    return pd.DataFrame(out, index=Widx.index)

def build_windows(ev, delta=120, overlap=False, auto_scenario=False):
    m = 2 if overlap else 1
    stride = delta / m
    base = np.floor(ev['t'].values / stride).astype(np.int64)
    parts = []
    for i in range(m):
        w = base - i; ok = w >= 0
        parts.append(ev.loc[ok].assign(w=w[ok]))
    E = pd.concat(parts, ignore_index=True).sort_values(['s', 'w', 't']).reset_index(drop=True)
    key = ['s', 'w']
    g = E.groupby(key, sort=True)
    W = g.agg(n=('t', 'size'), fail_count=('fail', 'sum'), label=('label', 'max'), nu=('u', 'nunique'))
    W['success_count'] = W['n'] - W['fail_count']
    W['fail_ratio'] = W['fail_count'] / W['n']
    W['attempt_rate'] = W['n'] / delta
    W['distinct_users'] = W['nu']
    c = E.groupby(['s', 'w', 'u']).size().rename('c').reset_index().merge(
        W[['n']].reset_index(), on=key)
    p = c['c'] / c['n']; c['h'] = -p * np.log2(p)
    W['username_entropy'] = c.groupby(key)['h'].sum()
    E['gap'] = E.groupby(key)['t'].diff()
    gg = E.groupby(key)['gap']
    W['gap_mean'] = gg.mean().fillna(0); W['gap_std'] = gg.std(ddof=0).fillna(0)
    ds = E.groupby(['w', 'u'])['s'].nunique().rename('ds').reset_index()
    E2 = E[['s', 'w', 'u']].drop_duplicates().merge(ds, on=['w', 'u'])
    W['max_distinct_src_per_user'] = E2.groupby(key)['ds'].max()
    atk = E[E['label'] == 1].groupby(key)['scen'].first()
    W['scenario'] = atk.reindex(W.index).fillna('benign')
    W = W.reset_index().sort_values(['s', 'w']).reset_index(drop=True)
    W['end'] = W['w'] * stride + delta
    W['label'] = W['label'].astype(int)
    SF = stateful_feats(ev, W[['s', 'end']], delta)
    W = pd.concat([W, SF.reset_index(drop=True)], axis=1)
    if auto_scenario:   # heuristic labels for REPORTING only (documented in the paper)
        sc = np.where(W['max_distinct_src_per_user'] >= 3, 'distributed',
             np.where(W['distinct_users'] >= 5, 'spraying',
             np.where(W['fail_count'] >= 10, 'single_source', 'low_and_slow')))
        W['scenario'] = np.where(W['label'] == 1, sc, 'benign')
    return W

# ------------------------------------------------------------------ split
def make_split(W, gap=7200, min_pos=10):
    end = W['end'].values; t0, t1 = end.min(), end.max()
    c1, c2 = t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)
    part = np.where(end <= c1, 0, np.where(end <= c2, 1, 2))
    ai = np.flatnonzero(W['label'].values == 1)
    if len(ai):   # whole campaigns (per scenario, time-contiguous) stay in one partition
        d = pd.DataFrame({'i': ai, 'end': end[ai], 'sc': W['scenario'].values[ai]}).sort_values(['sc', 'end'])
        new = (d['sc'] != d['sc'].shift()) | (d['end'].diff() > gap)
        d['camp'] = new.cumsum()
        first = d.groupby('camp')['end'].transform('min').values
        part[d['i'].values] = np.where(first <= c1, 0, np.where(first <= c2, 1, 2))
    for k, nm in enumerate(['train', 'val', 'test']):
        npos = int(((part == k) & (W['label'].values == 1)).sum())
        if npos < min_pos:
            sys.exit(f"Partition '{nm}' has only {npos} attack windows (<{min_pos}). "
                     f"Collect more/longer attack campaigns or lower --min-pos.")
    return part

# ------------------------------------------------------------------ models / metrics
def make_model(name, seed, spw, fast):
    ne = 100 if fast else 300
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed))
    if name == 'RF':
        return RandomForestClassifier(n_estimators=ne, class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    if name == 'XGB':
        if XGBClassifier is not None:
            return XGBClassifier(n_estimators=ne, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric='logloss', tree_method='hist', n_jobs=-1, random_state=seed)
        return HistGradientBoostingClassifier(max_iter=ne, class_weight='balanced', random_state=seed)
    if name == 'MLP':
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=seed))
    raise ValueError(name)

def best_threshold(y, s):
    P, R, T = precision_recall_curve(y, s)
    f1 = 2 * P[:-1] * R[:-1] / np.maximum(P[:-1] + R[:-1], 1e-12)
    return T[int(np.argmax(f1))]

def metrics(y, score, pred, scen=None):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F1 = 2 * P * R / max(P + R, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fprs, tprs, _ = roc_curve(y, score)
    out = dict(precision=P, recall=R, f1=F1, fpr=fpr, mcc=matthews_corrcoef(y, pred),
               roc_auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score),
               tpr_at_fpr1=float(np.interp(0.01, fprs, tprs)), tpr_at_fpr01=float(np.interp(0.001, fprs, tprs)),
               alerts_per_10k_benign=fpr * 1e4)
    if scen is not None:
        for sc in sorted(set(scen[y == 1])):
            m = (scen == sc) & (y == 1)
            out[f'rec_{sc}'] = float((pred[m] == 1).mean())
    return out

def boot_ci(y, score, pred, B, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    f, p, r = [], [], []
    for _ in range(B):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yy, pp, ss = y[idx], pred[idx], score[idx]
        tp = ((pp == 1) & (yy == 1)).sum(); fp = ((pp == 1) & (yy == 0)).sum(); fn = ((pp == 0) & (yy == 1)).sum()
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        f.append(2 * P * R / max(P + R, 1e-12)); r.append(R); p.append(average_precision_score(yy, ss))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(f1_ci=q(f), prauc_ci=q(p), recall_ci=q(r))

def fit_eval(Xtr, ytr, Xva, yva, Xte, name, seed, fast, smote=False):
    spw = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1)
    if smote:
        if SMOTE is None: return None
        Xtr, ytr = SMOTE(sampling_strategy=0.3, random_state=seed).fit_resample(Xtr, ytr)
        spw = 1.0
    mdl = make_model(name, seed, spw, fast); mdl.fit(Xtr, ytr)
    sv = mdl.predict_proba(Xva)[:, 1]; thr = best_threshold(yva, sv)
    t0 = time.time(); st = mdl.predict_proba(Xte)[:, 1]; lat = (time.time() - t0) / max(len(Xte), 1) * 1000
    return mdl, thr, st, lat

# ------------------------------------------------------------------ per dataset
def run_dataset(dname, ev, a, rows, preds, lat_rows):
    t0 = time.time(); W = build_windows(ev, a.delta, False, a.auto_scenario); t_feat = time.time() - t0
    lat_rows.append(dict(dataset=dname, what='feature extraction (ms per 1000 events)', value=t_feat / len(ev) * 1e6))
    W['part'] = make_split(W, a.campaign_gap, a.min_pos)
    drop = set(a.drop_features or [])
    sing = [c for c in SINGLE if c not in drop]; stat = sing + [c for c in STATEFUL if c not in drop]
    tr, va, te = (W['part'] == 0).values, (W['part'] == 1).values, (W['part'] == 2).values
    y = W['label'].values; scen = W['scenario'].values
    print(f"[{dname}] events={len(ev)} windows={len(W)} attack_windows={int(y.sum())} "
          f"({100*y.mean():.2f}%) train/val/test={tr.sum()}/{va.sum()}/{te.sum()}")
    info = dict(dataset=dname, events=len(ev), windows=len(W), attack_windows=int(y.sum()),
                train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))
    # ---- baselines (validation-calibrated)
    def add(det, seed, sc, pr, extra=None):
        m = metrics(y[te], sc, pr, scen[te]); m.update(dataset=dname, detector=det, seed=seed)
        if extra: m.update(extra)
        rows.append(m); preds[(dname, det, seed)] = (sc, pr)
    f = W['fail_count'].values.astype(float)
    k = max(range(1, 11), key=lambda k: metrics(y[va], f[va], (f[va] >= k).astype(int))['f1'])
    add('Failure-count threshold', 0, f[te], (f[te] >= k).astype(int), dict(k_star=k))
    best = None
    for T in ('10m', '1h'):
        col = W[f'F_{T}'].values.astype(float)
        for N in range(2, 31):
            f1 = metrics(y[va], col[va], (col[va] >= N).astype(int))['f1']
            if best is None or f1 > best[0]: best = (f1, T, N)
    col = W[f'F_{best[1]}'].values.astype(float)
    add('fail2ban-style rule', 0, col[te], (col[te] >= best[2]).astype(int), dict(k_star=best[2], findtime=best[1]))
    # ---- learned models
    for fs, cols in (('single', sing), ('stateful', stat)):
        X = W[cols].values.astype(float)
        for mname in ('LR', 'RF', 'XGB', 'MLP'):
            for seed in range(a.seeds):
                res = fit_eval(X[tr], y[tr], X[va], y[va], X[te], mname, seed, a.fast)
                mdl, thr, st, lat = res
                add(f'{mname} {fs}', seed, st, (st >= thr).astype(int))
                if seed == 0 and mname == 'XGB':
                    lat_rows.append(dict(dataset=dname, what=f'XGB {fs} inference (ms per window)', value=lat))
    # ---- ablations (XGB)
    abl = {'single': (sing, False), 'stateful': (stat, False),
           'stateful w/o success_count': ([c for c in stat if c != 'success_count'], False),
           'stateful w/o max_distinct_src_per_user': ([c for c in stat if c != 'max_distinct_src_per_user'], False),
           'single + SMOTE': (sing, True), 'stateful + SMOTE': (stat, True)}
    for an, (cols, sm) in abl.items():
        X = W[cols].values.astype(float)
        for seed in range(a.seeds):
            res = fit_eval(X[tr], y[tr], X[va], y[va], X[te], 'XGB', seed, a.fast, sm)
            if res is None: break
            _, thr, st, _ = res
            m = metrics(y[te], st, (st >= thr).astype(int), scen[te])
            m.update(dataset=dname, detector=f'ABL XGB {an}', seed=seed); rows.append(m)
    # ---- window sensitivity (XGB, seed 0)
    sens = []
    for d in a.windows:
        for ov in (False, True):
            try:
                Wd = build_windows(ev, d, ov, a.auto_scenario); pt = make_split(Wd, a.campaign_gap, a.min_pos)
            except SystemExit as e:
                print(f"  window {d}s overlap={ov}: skipped ({e})"); continue
            a_, b_, c_ = pt == 0, pt == 1, pt == 2; yy = Wd['label'].values
            for fs, base in (('single', sing), ('stateful', stat)):
                cols = [c for c in base if c in Wd.columns]; X = Wd[cols].values.astype(float)
                _, thr, st, _ = fit_eval(X[a_], yy[a_], X[b_], yy[b_], X[c_], 'XGB', 0, a.fast)
                m = metrics(yy[c_], st, (st >= thr).astype(int)); m.update(dataset=dname, window_s=d, overlap50=ov, features=fs)
                sens.append(m)
    return W, tr, va, te, sing, stat, info, pd.DataFrame(sens)

def cross_dataset(sets, a):
    rows = []
    for src, tgt in a.cross_pairs:
        if src not in sets or tgt not in sets: continue
        Ws, trs, vas, _, sing, stat, _, _ = sets[src]; Wt, _, _, tet, *_ = sets[tgt]
        ys, yt = Ws['label'].values, Wt['label'].values
        for fs, cols in (('single', sing), ('stateful', stat)):
            cols = [c for c in cols if c in Wt.columns]
            Xs, Xt = Ws[cols].values.astype(float), Wt[cols].values.astype(float)
            for seed in range(a.seeds):
                _, thr, st, _ = fit_eval(Xs[trs], ys[trs], Xs[vas], ys[vas], Xt[tet], 'XGB', seed, a.fast)
                m = metrics(yt[tet], st, (st >= thr).astype(int), Wt['scenario'].values[tet])
                m.update(train=src, test=tgt, detector=f'XGB {fs}', seed=seed); rows.append(m)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ reporting
MET = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mcc', 'tpr_at_fpr1', 'tpr_at_fpr01', 'alerts_per_10k_benign']
def agg(df, by):
    g = df.groupby(by)
    mean, sd = g[MET].mean(), g[MET].std(ddof=0).fillna(0)
    out = mean.copy()
    for c in MET: out[c + '_sd'] = sd[c]
    out['n_runs'] = g.size()
    return out.reset_index()

def fmt(r, c, d=3):
    return f"{r[c]:.{d}f}" if r[c + '_sd'] < 5e-4 else f"{r[c]:.{d}f} ± {r[c + '_sd']:.{d}f}"

def mcnemar_rows(sets, preds, ds):
    pairs = [('XGB stateful', 'XGB single'), ('XGB single', 'Failure-count threshold'),
             ('XGB stateful', 'Failure-count threshold'), ('XGB stateful', 'fail2ban-style rule')]
    out = []
    for dname in ds:
        W, _, _, te, *_ = sets[dname]; y = W['label'].values[te]
        for A, B in pairs:
            ka, kb = (dname, A, 0), (dname, B, 0)
            if ka not in preds or kb not in preds: continue
            pa, pb = preds[ka][1], preds[kb][1]
            ca, cb = (pa == y), (pb == y)
            b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
            out.append(dict(dataset=dname, A=A, B=B, A_right_B_wrong=b, B_right_A_wrong=c, p_value=p, significant_0_05=p < 0.05))
    return pd.DataFrame(out)

def write_md(path, T11, T12, ABL, SENS, CROSS, MC, LAT, infos, boot):
    L = ["# Tables for Section 7.7 (generated by bf_eval.py; real-data results only if the input was real)\n"]
    L.append("## Dataset sizes\n\n" + pd.DataFrame(infos).to_markdown(index=False) + "\n")
    for d, g in T11.groupby('dataset'):
        L.append(f"## Table 11 - {d} (mean ± sd over seeds; chronological, campaign-grouped split)\n")
        L.append("| Detector | Precision | Recall | F1 | FPR | MCC | PR-AUC | TPR@FPR1% | TPR@FPR0.1% | Alerts/10k benign |\n|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            ci = boot.get((d, r['detector']))
            f1 = fmt(r, 'f1') + (f" [{ci['f1_ci'][0]:.3f}, {ci['f1_ci'][1]:.3f}]" if ci else "")
            L.append(f"| {r['detector']} | {fmt(r,'precision')} | {fmt(r,'recall')} | {f1} | {fmt(r,'fpr',4)} | {fmt(r,'mcc')} | "
                     f"{fmt(r,'pr_auc')} | {fmt(r,'tpr_at_fpr1')} | {fmt(r,'tpr_at_fpr01')} | {fmt(r,'alerts_per_10k_benign',1)} |")
        L.append("\n(95% bootstrap CI in brackets, computed on the seed-0 predictions.)\n")
    if len(T12):
        L.append("## Table 12 - per-scenario recall (mean over seeds)\n\n" + T12.round(3).to_markdown(index=False) + "\n")
    if len(ABL): L.append("## Ablations (XGB)\n\n" + ABL.round(3).to_markdown(index=False) + "\n")
    if len(SENS): L.append("## Window sensitivity (XGB, seed 0)\n\n" + SENS.round(3).to_markdown(index=False) + "\n")
    if len(CROSS): L.append("## Cross-dataset (train -> test)\n\n" + CROSS.round(3).to_markdown(index=False) + "\n")
    if len(MC): L.append("## McNemar tests (seed 0, test windows)\n\n" + MC.round(4).to_markdown(index=False) + "\n")
    L.append("## Latency\n\n" + LAT.round(4).to_markdown(index=False) + "\n")
    open(path, 'w').write("\n".join(L))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help="NAME=path.csv (e.g. D1=d1.csv D2=d2.csv)")
    ap.add_argument('--out', default='results'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--delta', type=int, default=120)
    ap.add_argument('--windows', type=int, nargs='*', default=[60, 120, 300, 900])
    ap.add_argument('--campaign-gap', dest='campaign_gap', type=float, default=7200)
    ap.add_argument('--min-pos', dest='min_pos', type=int, default=10)
    ap.add_argument('--auto-scenario', action='store_true', help="heuristic scenario labels when the CSV has no 'scenario' column")
    ap.add_argument('--drop-features', nargs='*', default=[], help="e.g. success_count for flow-derived data")
    ap.add_argument('--cross', nargs='*', default=None, help="pairs like D1>D2")
    ap.add_argument('--fast', action='store_true', help="smaller forests (smoke tests only)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    names, files = zip(*[x.split('=', 1) for x in a.data])
    a.cross_pairs = [tuple(p.split('>')) for p in a.cross] if a.cross else \
        [(x, y) for x in names for y in names if x != y]
    rows, preds, lat_rows, sets, infos, sens_all = [], {}, [], {}, [], []
    for n, f in zip(names, files):
        res = run_dataset(n, load_events(f), a, rows, preds, lat_rows)
        sets[n] = res; infos.append(res[6]); sens_all.append(res[7])
    raw = pd.DataFrame(rows); raw.to_csv(f"{a.out}/raw_runs.csv", index=False)
    main_det = raw[~raw['detector'].str.startswith('ABL')]
    T11 = agg(main_det, ['dataset', 'detector']); T11.to_csv(f"{a.out}/table11_overall.csv", index=False)
    scol = [c for c in raw.columns if c.startswith('rec_')]
    T12 = main_det[main_det['detector'].isin(['Failure-count threshold', 'fail2ban-style rule', 'XGB single', 'XGB stateful'])] \
        .groupby(['dataset', 'detector'])[scol].mean().reset_index(); T12.to_csv(f"{a.out}/table12_per_scenario.csv", index=False)
    ABL = agg(raw[raw['detector'].str.startswith('ABL')], ['dataset', 'detector'])[['dataset', 'detector', 'precision', 'recall', 'f1', 'pr_auc', 'fpr', 'mcc']]
    ABL.to_csv(f"{a.out}/ablation.csv", index=False)
    SENS = pd.concat(sens_all, ignore_index=True)
    if len(SENS): SENS = SENS[['dataset', 'window_s', 'overlap50', 'features', 'precision', 'recall', 'f1', 'pr_auc', 'fpr']]
    SENS.to_csv(f"{a.out}/window_sensitivity.csv", index=False)
    CROSS = cross_dataset(sets, a) if len(names) > 1 else pd.DataFrame()
    if len(CROSS):
        CROSS = agg(CROSS, ['train', 'test', 'detector'])[['train', 'test', 'detector', 'precision', 'recall', 'f1', 'pr_auc', 'fpr']]
        CROSS.to_csv(f"{a.out}/cross_dataset.csv", index=False)
    MC = mcnemar_rows(sets, preds, names); MC.to_csv(f"{a.out}/mcnemar.csv", index=False)
    LAT = pd.DataFrame(lat_rows); LAT.to_csv(f"{a.out}/latency.csv", index=False)
    rng = np.random.default_rng(0); boot = {}
    for n in names:
        W, _, _, te, *_ = sets[n]; y = W['label'].values[te]
        for det in T11[T11['dataset'] == n]['detector']:
            sc, pr = preds[(n, det, 0)]; boot[(n, det)] = boot_ci(y, sc, pr, a.boot, rng)
    write_md(f"{a.out}/tables_for_paper.md", T11, T12, ABL, SENS, CROSS, MC, LAT, infos, boot)
    json.dump(dict(args={k: v for k, v in vars(a).items() if k != 'cross_pairs'}, python=platform.python_version(),
                   xgboost=XGBClassifier is not None, smote=SMOTE is not None), open(f"{a.out}/run_config.json", 'w'), indent=2)
    print(f"Done -> {a.out}/tables_for_paper.md")

if __name__ == '__main__':
    main()
