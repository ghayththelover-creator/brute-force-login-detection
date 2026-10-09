"""Brute-force check that the fast stateful features equal a naive recomputation (and use no future events)."""
import numpy as np, pandas as pd, bf_eval as B
ev = B.load_events('smoke.csv'); W = B.build_windows(ev, 120, False)
rng = np.random.default_rng(1); bad = 0
for i in rng.choice(len(W), 300, replace=False):
    r = W.iloc[i]; e = ev[(ev.s == r.s)]
    for H, nm in ((3600, '1h'), (86400, '24h')):
        x = e[(e.t >= r.end - H) & (e.t < r.end)]; f = x[x.fail == 1]
        gaps = np.diff(f.t.values)
        exp = dict(F=len(f), rho=len(f) / len(x) if len(x) else 0, D=x.u.nunique(),
                   A=f.t.floordiv(120).nunique(), mu=gaps.mean() if len(gaps) else 0,
                   cv=(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else 0)
        for k, v in exp.items():
            if abs(r[f'{k}_{nm}'] - v) > 1e-4 * max(1, abs(v)): bad += 1; print('MISMATCH', i, k, nm, r[f'{k}_{nm}'], v)
    # window-level checks
    w = e[(e.t >= r.end - 120) & (e.t < r.end)]
    assert r.fail_count == w.fail.sum() and r.n == len(w), (i, 'window counts')
print('stateful/window mismatches:', bad)
