#!/usr/bin/env python3
"""
run_success_baselines.py - Reviewer 2 (round 2), comment 1: laboratory baseline comparison.

Adds two one-feature baselines to the laboratory comparison (Table 14):
  * Zero-success rule : flag a window when success_count == 0 (no parameter, nothing is fitted);
  * Decision stump    : depth-1 decision tree fitted on the TRAINING windows using success_count only.
It also re-computes the McNemar tests of XGBoost (single-window) against every rule, and splits the
discordant windows into "rule false positives" and "attacks missed by the rule" (Reviewer 2, comment 3).

Same data, windows (120 s, non-overlapping), chronological 60/20/20 split and evaluation code as run_real.py.

  python run_success_baselines.py --data data/D1_hashed.csv --out results_real --seeds 10

New output files (existing results are not modified):
  real_success_baselines.csv   one row per baseline (metrics, TP/FP/FN/TN, exact 95% CI of the FPR,
                               stump threshold and validation accuracy)
  real_mcnemar_discordant.csv  one row per (seed, rule): discordant windows split into FP / missed attacks
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf_eval as B
from run_synthetic import SINGLE, chrono_split, train_score, evaluate, rule_baselines


def fpr_ci(fp, tn):
    """Exact (Clopper-Pearson) 95% interval of FPR = FP / (FP + TN)."""
    n = fp + tn
    ci = binomtest(fp, n).proportion_ci(confidence_level=0.95, method='exact')
    return ci.low, ci.high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results_real')
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    ev = B.load_events(a.data)
    W = B.build_windows(ev, 120, False)
    part = chrono_split(W)
    y, scen = W['label'].values.astype(int), W['scenario'].values
    tr, va, te = part == 0, part == 1, part == 2
    yt = y[te]
    sc = W['success_count'].values.astype(float)
    print(f"windows: train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"(test: {yt.sum()} attack, {(yt == 0).sum()} benign)")

    rows, preds = [], {}

    # (1) Zero-success rule: no fitted parameter
    pr = (sc[te] == 0).astype(int)
    m = evaluate(yt, pr.astype(float), pr, scen[te])
    m.update(detector='Zero-success rule (success_count = 0)', params='success_count == 0')
    rows.append(m); preds['Zero-success rule'] = pr

    # (2) Decision stump fitted on the training windows, success_count only
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(sc[tr].reshape(-1, 1), y[tr])
    thr = float(stump.tree_.threshold[0])
    left_class = int(np.argmax(stump.tree_.value[stump.tree_.children_left[0]]))
    rule_txt = (f"attack if success_count <= {thr:.2f}" if left_class == 1
                else f"attack if success_count > {thr:.2f}")
    val_acc = float((stump.predict(sc[va].reshape(-1, 1)) == y[va]).mean())
    st = stump.predict_proba(sc[te].reshape(-1, 1))[:, 1]
    pr = (st >= 0.5).astype(int)
    m = evaluate(yt, st, pr, scen[te])
    m.update(detector='Decision stump (success_count only)', params=rule_txt,
             stump_threshold=thr, val_accuracy=val_acc, n_val=int(va.sum()))
    rows.append(m); preds['Decision stump'] = pr
    print(f"stump learned: {rule_txt}; validation accuracy = {val_acc:.3f} on {va.sum()} windows")

    for r in rows:
        lo, hi = fpr_ci(r['FP'], r['TN'])
        r.update(fpr_ci_low=lo, fpr_ci_high=hi)
        print(f"{r['detector']}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"FPR={r['fpr']:.4f} [{lo:.3f}, {hi:.3f}] MCC={r['mcc']:.3f} "
              f"TP/FP/FN/TN={r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}")
    pd.DataFrame(rows).to_csv(f"{a.out}/real_success_baselines.csv", index=False)

    # (3) McNemar: XGBoost (single-window) vs each rule, discordant windows split by error type
    for det, (_, p_rule, _) in rule_baselines(W, part).items():
        preds[det] = p_rule
    X = W[SINGLE].values.astype(float)
    mc = []
    for seed in range(a.seeds):
        _, sv, stx = train_score('XGB', X[tr], y[tr], X[va], X[te], seed, smote=False)
        px = (stx >= B.best_threshold(y[va], sv)).astype(int)
        for det in ('Failure-count rule', 'fail2ban-style rule', 'Count + source-diversity rule',
                    'Count + diversity + history rule', 'Zero-success rule', 'Decision stump'):
            pb = preds[det]
            x_ok, r_ok = px == yt, pb == yt
            b_, c_ = int((x_ok & ~r_ok).sum()), int((~x_ok & r_ok).sum())
            p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0   # exact McNemar
            mc.append(dict(seed=seed, A='XGB, single-window', B=det,
                           discordant=b_ + c_, A_right_B_wrong=b_, B_right_A_wrong=c_, p=p,
                           rule_false_positives=int((x_ok & ~r_ok & (yt == 0)).sum()),
                           rule_missed_attacks=int((x_ok & ~r_ok & (yt == 1)).sum())))
    mc = pd.DataFrame(mc)
    mc.to_csv(f"{a.out}/real_mcnemar_discordant.csv", index=False)
    print(mc.groupby('B')[['discordant', 'A_right_B_wrong', 'B_right_A_wrong',
                           'rule_false_positives', 'rule_missed_attacks', 'p']].agg(['min', 'max']))
    print("done ->", a.out)


if __name__ == '__main__':
    main()
