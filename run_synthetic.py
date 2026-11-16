#!/usr/bin/env python3
"""
run_synthetic.py - revised synthetic-corpus experiments (Sections 6.2 and 7 of the revised paper).

Protocol (addresses reviewer comments on splitting, repeated runs and fair baselines):
  * R independent runs; run r regenerates the corpus with simulator seed r and trains
    every model with random seed r (data + model variability).
  * Strict chronological split on window end time: first 60% train, next 20% validation,
    last 20% test. No campaign re-grouping, so no window is ever moved to an earlier
    partition (the released v1 code moved whole campaigns to the partition of their first
    window, which could put later observations into training).
  * Overlap of sources/campaigns across partitions is measured and reported, and the test
    set is also scored on attack windows from sources never seen in training.
  * SMOTE (minority:majority = 0.3) on the training partition only; thresholds selected on
    validation (F1); test set touched once.
Stages: main | ablation | windows   (python run_synthetic.py --stage main --runs 10)
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf_eval as B
from simulate_auth import simulate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SINGLE = B.SINGLE
STATEFUL = SINGLE + B.STATEFUL
TIMING = ['gap_mean', 'gap_std', 'attempt_rate']


def windows_for(run, delta=120, overlap=False, cache='cache', stale=0):
    os.makedirs(cache, exist_ok=True)
    p = f"{cache}/W_r{run}_d{delta}_o{int(overlap)}_s{stale}.pkl"
    if os.path.exists(p):
        return pd.read_pickle(p)
    csv = f"{cache}/sim_seed{run}_stale{stale}.csv"
    if not os.path.exists(csv):
        simulate(run, stale).to_csv(csv, index=False)
    ev = B.load_events(csv)
    W = B.build_windows(ev, delta, overlap)
    W.to_pickle(p)
    return W


def chrono_split(W):
    end = W['end'].values; t0, t1 = end.min(), end.max()
    c1, c2 = t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)
    return np.where(end <= c1, 0, np.where(end <= c2, 1, 2))


def overlap_stats(W, part):
    atk = W['label'].values == 1
    s = W['s'].values
    tr_src = set(s[(part == 0) & atk]); te_src = set(s[(part == 2) & atk])
    va_src = set(s[(part == 1) & atk])
    return dict(attack_src_train=len(tr_src), attack_src_val=len(va_src), attack_src_test=len(te_src),
                attack_src_train_and_test=len(tr_src & te_src), attack_src_train_and_val=len(tr_src & va_src),
                test_attack_windows_seen_src=int(((part == 2) & atk & np.isin(s, list(tr_src))).sum()),
                test_attack_windows=int(((part == 2) & atk).sum()))


def fit_xgb(X, y, seed, spw=1.0):
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                      scale_pos_weight=spw, eval_metric='logloss', tree_method='hist', n_jobs=2, random_state=seed)
    return m.fit(X, y)


def make(name, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, max_features='sqrt', n_jobs=2, random_state=seed)
    if name == 'XGB':
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             eval_metric='logloss', tree_method='hist', n_jobs=2, random_state=seed)
    if name == 'MLP':
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, random_state=seed))
    if name == 'Tree-d3':
        return DecisionTreeClassifier(max_depth=3, random_state=seed)
    raise ValueError(name)


def train_score(name, Xtr, ytr, Xva, Xte, seed, smote=True, weight=False):
    if smote:
        Xtr, ytr = SMOTE(sampling_strategy=0.3, random_state=seed).fit_resample(Xtr, ytr)
    if weight and name == 'XGB':
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m = fit_xgb(Xtr, ytr, seed, spw)
    else:
        m = make(name, seed).fit(Xtr, ytr)
    return m, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def evaluate(y, score, pred, scen, unseen_mask=None):
    m = B.metrics(y, score, pred, scen)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    m.update(TP=tp, FP=fp, FN=fn, TN=tn)
    for sc in ('single_source', 'spraying', 'distributed', 'low_and_slow'):
        mm = (scen == sc) & (y == 1)
        m[f'n_{sc}'] = int(mm.sum()); m[f'det_{sc}'] = int((pred[mm] == 1).sum())
    if unseen_mask is not None:
        um = unseen_mask & (y == 1)
        m['n_unseen_attack'] = int(um.sum())
        m['rec_unseen_attack'] = float((pred[um] == 1).mean()) if um.any() else np.nan
    return m


def grid_rule(Wv, yv, cand):
    """cand: list of (params, fn(W)->bool). Return best params by validation F1."""
    best = None
    for p, fn in cand:
        pr = fn(Wv).astype(int)
        f1 = B.metrics(yv, pr.astype(float), pr)['f1']
        if best is None or f1 > best[0]:
            best = (f1, p, fn)
    return best


def rule_baselines(W, part):
    y = W['label'].values
    Wv, Wt = W[part == 1], W[part == 2]
    f = lambda d: d['fail_count'].values
    m = lambda d: d['max_distinct_src_per_user'].values
    out = {}
    # (a) per-window failure count
    c = [((k,), (lambda d, k=k: f(d) >= k)) for k in range(1, 11)]
    out['Failure-count rule'] = grid_rule(Wv, y[part == 1], c)
    # (b) fail2ban-style: >= N failures from a source within findtime
    c = [((T, N), (lambda d, T=T, N=N: d[f'F_{T}'].values >= N)) for T in ('10m', '1h', '24h') for N in range(2, 41)]
    out['fail2ban-style rule'] = grid_rule(Wv, y[part == 1], c)
    # (c) information-matched single-window rule: count OR cross-source diversity
    c = [((k, j), (lambda d, k=k, j=j: (f(d) >= k) | ((m(d) >= j) & (f(d) >= 1))))
         for k in range(1, 11) for j in range(2, 11)]
    out['Count + source-diversity rule'] = grid_rule(Wv, y[part == 1], c)
    # (d) information-matched stateful rule: (c) OR >= N failures in last H (and no success in the window)
    kj = out['Count + source-diversity rule'][1]
    c = [((kj[0], kj[1], T, N), (lambda d, k=kj[0], j=kj[1], T=T, N=N:
          (f(d) >= k) | ((m(d) >= j) & (f(d) >= 1)) | ((d[f'F_{T}'].values >= N) & (d['success_count'].values == 0))))
         for T in ('1h', '24h') for N in range(2, 41)]
    out['Count + diversity + history rule'] = grid_rule(Wv, y[part == 1], c)
    res = {}
    for k, (f1, p, fn) in out.items():
        pr = fn(Wt).astype(int)
        res[k] = (pr.astype(float), pr, p)
    return res


def stage_main(a):
    rows, ovl, shap_rows, mc = [], [], [], []
    for r in range(a.runs):
        t0 = time.time()
        W = windows_for(r); part = chrono_split(W)
        y = W['label'].values; scen = W['scenario'].values; s = W['s'].values
        tr, va, te = part == 0, part == 1, part == 2
        o = overlap_stats(W, part); o.update(run=r, windows=len(W), attack_windows=int(y.sum()),
                                          train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))
        for sc in ('single_source', 'spraying', 'distributed', 'low_and_slow'):
            o[f'win_{sc}'] = int((scen == sc).sum())
        ovl.append(o)
        unseen = ~np.isin(s[te], list(set(s[tr & (y == 1)])))
        preds = {}
        for det, (sc_, pr, p) in rule_baselines(W, part).items():
            m = evaluate(y[te], sc_, pr, scen[te], unseen); m.update(run=r, detector=det, params=str(p)); rows.append(m)
            preds[det] = pr
        for fs, cols in (('single-window', SINGLE), ('stateful', STATEFUL)):
            X = W[cols].values.astype(float)
            for name in ('LR', 'RF', 'XGB', 'MLP', 'Tree-d3'):
                mdl, sv, st = train_score(name, X[tr], y[tr], X[va], X[te], r)
                thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
                m = evaluate(y[te], st, pr, scen[te], unseen)
                m.update(run=r, detector=f'{name}, {fs}', threshold=float(thr), brier=brier_score_loss(y[te], st))
                rows.append(m); preds[f'{name}, {fs}'] = pr
                if name == 'XGB':
                    import xgboost as xgb
                    contrib = mdl.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
                    for c_, v in zip(cols, np.abs(contrib).mean(0)):
                        shap_rows.append(dict(run=r, features=fs, feature=c_, mean_abs_shap=float(v)))
                    if fs == 'single-window':
                        gain = mdl.get_booster().get_score(importance_type='gain')
                if name == 'RF' and fs == 'single-window':
                    for c_, v in zip(cols, mdl.feature_importances_):
                        shap_rows.append(dict(run=r, features='RF-impurity', feature=c_, mean_abs_shap=float(v)))
        yt = y[te]
        for A, Bn in [('XGB, stateful', 'XGB, single-window'), ('XGB, single-window', 'Failure-count rule'),
                      ('XGB, single-window', 'Count + source-diversity rule'),
                      ('XGB, stateful', 'Count + diversity + history rule'), ('XGB, stateful', 'fail2ban-style rule'),
                      ('LR, single-window', 'XGB, single-window'), ('RF, stateful', 'RF, single-window'),
                      ('MLP, stateful', 'MLP, single-window')]:
            ca, cb = preds[A] == yt, preds[Bn] == yt
            b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
            from scipy.stats import binomtest
            p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
            mc.append(dict(run=r, A=A, B=Bn, A_right_B_wrong=b_, B_right_A_wrong=c_, p=p))
        print(f"run {r} done in {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/main_raw.csv", index=False)
        pd.DataFrame(ovl).to_csv(f"{a.out}/corpus_and_overlap.csv", index=False)
        pd.DataFrame(shap_rows).to_csv(f"{a.out}/importance_raw.csv", index=False)
        pd.DataFrame(mc).to_csv(f"{a.out}/mcnemar_raw.csv", index=False)
    rng = np.random.default_rng(0)


def stage_ablation(a):
    rows = []
    conf = {
        'stateful (full, SMOTE)': (STATEFUL, True, False),
        'single-window (SMOTE)': (SINGLE, True, False),
        'single-window, no resampling': (SINGLE, False, False),
        'single-window, class weighting': (SINGLE, False, True),
        'stateful, no resampling': (STATEFUL, False, False),
        'stateful w/o success_count': ([c for c in STATEFUL if c != 'success_count'], True, False),
        'single-window w/o success_count': ([c for c in SINGLE if c != 'success_count'], True, False),
        'single-window w/o max_distinct_src_per_user': ([c for c in SINGLE if c != 'max_distinct_src_per_user'], True, False),
        'stateful w/o max_distinct_src_per_user': ([c for c in STATEFUL if c != 'max_distinct_src_per_user'], True, False),
        'single-window w/o timing features': ([c for c in SINGLE if c not in TIMING], True, False),
        'stateful, 1 h horizon only': (SINGLE + [c for c in B.STATEFUL if c.endswith('_1h')], True, False),
        'stateful, 24 h horizon only': (SINGLE + [c for c in B.STATEFUL if c.endswith('_24h')], True, False),
    }
    for r in range(a.runs):
        W = windows_for(r); part = chrono_split(W)
        y = W['label'].values; scen = W['scenario'].values
        tr, va, te = part == 0, part == 1, part == 2
        for nm, (cols, sm, wt) in conf.items():
            X = W[cols].values.astype(float)
            _, sv, st = train_score('XGB', X[tr], y[tr], X[va], X[te], r, smote=sm, weight=wt)
            thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
            m = evaluate(y[te], st, pr, scen[te]); m.update(run=r, config=nm, brier=brier_score_loss(y[te], st))
            # fixed default threshold 0.5 (no threshold selection) for the resampling question
            pr5 = (st >= 0.5).astype(int); m5 = B.metrics(y[te], st, pr5)
            m.update(f1_at_0_5=m5['f1'], precision_at_0_5=m5['precision'], recall_at_0_5=m5['recall'])
            rows.append(m)
        print(f"ablation run {r}", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/ablation_raw.csv", index=False)


def stage_ablation2(a):
    """Does the detector depend on success-derived information? Remove every success-derived feature."""
    SUCC = {'success_count', 'fail_ratio', 'rho_1h', 'rho_24h'}
    conf = {'single-window w/o success-derived features': [c for c in SINGLE if c not in SUCC],
            'stateful w/o success-derived features': [c for c in STATEFUL if c not in SUCC]}
    rows = []
    for r in range(a.runs):
        W = windows_for(r); part = chrono_split(W)
        y = W['label'].values; scen = W['scenario'].values
        tr, va, te = part == 0, part == 1, part == 2
        for nm, cols in conf.items():
            X = W[cols].values.astype(float)
            _, sv, st = train_score('XGB', X[tr], y[tr], X[va], X[te], r)
            thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
            m = evaluate(y[te], st, pr, scen[te]); m.update(run=r, config=nm); rows.append(m)
    pd.DataFrame(rows).to_csv(f"{a.out}/ablation2_raw.csv", index=False)


def stage_windows(a):
    rows = []
    for r in range(a.runs):
        for d in (60, 120, 300, 900):
            for ov in (False, True):
                W = windows_for(r, d, ov); part = chrono_split(W)
                y = W['label'].values; scen = W['scenario'].values
                tr, va, te = part == 0, part == 1, part == 2
                for fs, cols in (('single-window', SINGLE), ('stateful', STATEFUL)):
                    X = W[cols].values.astype(float)
                    _, sv, st = train_score('XGB', X[tr], y[tr], X[va], X[te], r)
                    thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
                    m = evaluate(y[te], st, pr, scen[te])
                    m.update(run=r, window_s=d, overlap50=ov, features=fs, windows=len(W)); rows.append(m)
                # count baseline per window length
                f = W['fail_count'].values
                k = max(range(1, 11), key=lambda k: B.metrics(y[va], f[va].astype(float), (f[va] >= k).astype(int))['f1'])
                pr = (f[te] >= k).astype(int); m = evaluate(y[te], pr.astype(float), pr, scen[te])
                m.update(run=r, window_s=d, overlap50=ov, features=f'failure-count (k={k})', windows=len(W)); rows.append(m)
        print(f"windows run {r}", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/windows_raw.csv", index=False)


def stage_stress(a):
    """Stateful features under benign stale-credential sources (benign look-alikes of low-and-slow)."""
    rows = []
    for stale in (0, 10, 20, 40):
        for r in range(a.runs):
            W = windows_for(r, stale=stale); part = chrono_split(W)
            y = W['label'].values; scen = W['scenario'].values
            tr, va, te = part == 0, part == 1, part == 2
            for fs, cols in (('single-window', SINGLE), ('stateful', STATEFUL)):
                X = W[cols].values.astype(float)
                _, sv, st = train_score('XGB', X[tr], y[tr], X[va], X[te], r)
                thr = B.best_threshold(y[va], sv); pr = (st >= thr).astype(int)
                m = evaluate(y[te], st, pr, scen[te]); m.update(run=r, stale_sources=stale, features=fs); rows.append(m)
            res = rule_baselines(W, part)
            for det in ('fail2ban-style rule', 'Count + diversity + history rule'):
                sc_, pr, p = res[det]; m = evaluate(y[te], sc_, pr, scen[te]); m.update(run=r, stale_sources=stale, features=det); rows.append(m)
        print(f"stress stale={stale}", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/stress_raw.csv", index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['main', 'ablation', 'ablation2', 'windows', 'stress'])
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--out', default='results_synth')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {'main': stage_main, 'ablation': stage_ablation, 'windows': stage_windows, 'stress': stage_stress, 'ablation2': stage_ablation2}[a.stage](a)
