#!/usr/bin/env python3
"""make_figures.py - figures for the revised paper, drawn from the run outputs (results_synth/*)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = 'results_synth'; OUT = 'figs'; os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e87ba4']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})


def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
    boxes = [
        ('Raw events', 'authentication logs', 0), ('Parsing', 'normalised schema, Eq. (1)', 0),
        ('Features', '9 single-window (120 s)\n+ 12 stateful (1 h, 24 h)', 0), ('Chronological split', '60 / 20 / 20 by time', 0),
        ('Scaling + SMOTE', 'fitted on training only', 1), ('Model training', 'LR, RF, XGBoost, MLP,\ndepth-3 tree', 1),
        ('Threshold selection', 'max F1 on validation', 1), ('Evaluation', 'test set, used once', 2),
        ('Rule baselines', 'same features, same split', 2), ('TreeSHAP', 'feature attribution', 2)]
    pos = {0: [(0.1 + i * 2.5, 3.0) for i in range(4)], 1: [(0.1 + i * 2.5, 1.65) for i in range(3)],
           2: [(0.1 + i * 2.5, 0.3) for i in range(3)]}
    fills = {0: '#f1eee8', 1: '#ecebfb', 2: '#e1f3ec'}; edges = {0: '#8a877f', 1: '#6250d6', 2: '#1b8f64'}
    cnt = {0: 0, 1: 0, 2: 0}; centers = []
    for title, sub, row in boxes:
        x, y = pos[row][cnt[row]]; cnt[row] += 1
        ax.add_patch(FancyBboxPatch((x, y), 2.1, 0.95, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    fc=fills[row], ec=edges[row], lw=1))
        ax.text(x + 1.05, y + 0.66, title, ha='center', va='center', fontsize=10, weight='bold', color=INK)
        ax.text(x + 1.05, y + 0.3, sub, ha='center', va='center', fontsize=7.6, color=MUTED, linespacing=1.15)
        centers.append((x, y, row))
    arrow = dict(arrowstyle='->', color='#77746c', lw=1.2)
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9)]:
        xa, ya, _ = centers[a]; xb, yb, _ = centers[b]
        ax.annotate('', xy=(xb, yb + 0.47), xytext=(xa + 2.1, ya + 0.47), arrowprops=arrow)
    for (x0, y0, ymid, x1, y1) in [(8.65, 3.0, 2.8, 1.15, 2.6), (6.15, 1.65, 1.45, 1.15, 1.25)]:
        ax.plot([x0, x0, x1], [y0, ymid, ymid], color='#77746c', lw=1.2)
        ax.annotate('', xy=(x1, y1), xytext=(x1, ymid), arrowprops=arrow)
    fig.savefig(f'{OUT}/fig1_pipeline.png'); plt.close(fig)


def fig2():
    W = pd.read_pickle('cache/W_r0_d120_o0_s0.pkl')
    vc = W['scenario'].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ben, att = int((W.label == 0).sum()), int((W.label == 1).sum())
    ax[0].bar(['Benign', 'Attack'], [ben, att], color=[C[0], C[1]], width=0.55)
    ax[0].set_yscale('log'); ax[0].set_ylabel('Windows (log scale)'); ax[0].set_title('(a) Benign vs attack windows', fontsize=9, color=INK)
    for i, v in enumerate([ben, att]):
        ax[0].text(i, v * 1.15, f'{v:,}\n({100 * v / len(W):.2f}%)', ha='center', fontsize=8, color=INK)
    ax[0].set_ylim(50, 1e6)
    order = [('distributed', 'Distributed'), ('low_and_slow', 'Low-and-slow'), ('spraying', 'Credential spraying'),
             ('single_source', 'Single-source high-rate')]
    vals = [int(vc.get(k, 0)) for k, _ in order]
    ax[1].barh([n for _, n in order][::-1], vals[::-1], color=C[1], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax[1].text(v + 25, i, f'{v:,}', va='center', fontsize=8, color=INK)
    ax[1].set_xlabel('Attack windows'); ax[1].set_title('(b) Attack windows by scenario', fontsize=9, color=INK)
    ax[1].set_xlim(0, max(vals) * 1.2); ax[1].grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig2_classes.png'); plt.close(fig)


def fig3():
    T = pd.read_csv(f'{R}/T_main.csv').set_index('detector')
    models = ['LR', 'RF', 'XGB', 'MLP', 'Tree-d3']; names = ['LR', 'RF', 'XGBoost', 'MLP', 'Depth-3 tree']
    mets = [('f1', 'F1'), ('pr_auc', 'PR-AUC'), ('mcc', 'MCC')]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, fs, ttl in zip(axs, ['single-window', 'stateful'], ['(a) Single-window features', '(b) Single-window + stateful features']):
        x = np.arange(len(models)); w = 0.26
        for k, (m, lab) in enumerate(mets):
            v = [T.loc[f'{mm}, {fs}', m] for mm in models]; e = [T.loc[f'{mm}, {fs}', m + '_sd'] for mm in models]
            ax.bar(x + (k - 1) * w, v, w * 0.92, yerr=e, color=C[k], label=lab, error_kw=dict(lw=0.8, capsize=2, ecolor=MUTED))
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.1); ax.set_title(ttl, fontsize=9, color=INK)
        ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('Score (mean ± sd, 10 runs)'); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); fig.savefig(f'{OUT}/fig3_models.png'); plt.close(fig)


def fig4():
    imp = pd.read_csv(f'{R}/importance_raw.csv')
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, fs, ttl, xl in zip(axs, ['RF-impurity', 'stateful'], ['(a) Random Forest, single-window: impurity importance',
                                                               '(b) XGBoost, stateful: mean |SHAP| (top 10)'],
                               ['Importance (mean over 10 runs)', 'Mean |SHAP| (log-odds)']):
        g = imp[imp.features == fs].groupby('feature')['mean_abs_shap'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
        ax.barh(g.index[::-1], g['mean'][::-1], xerr=g['std'][::-1], color=C[0] if fs == 'RF-impurity' else C[4], height=0.6,
                error_kw=dict(lw=0.8, capsize=2, ecolor=MUTED))
        ax.set_title(ttl, fontsize=9, color=INK); ax.set_xlabel(xl); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig4_importance.png'); plt.close(fig)


DET = [('Failure-count rule', 'Failure-count rule'), ('fail2ban-style rule', 'fail2ban-style rule'),
       ('Count + source-diversity rule', 'Count + diversity rule'), ('XGB, single-window', 'XGBoost, single-window'),
       ('Count + diversity + history rule', 'Count + diversity + history rule'), ('XGB, stateful', 'XGBoost, stateful')]


def fig5():
    T = pd.read_csv(f'{R}/T_main.csv').set_index('detector')
    fig, ax = plt.subplots(figsize=(10, 2.9))
    x = np.arange(len(DET)); w = 0.38
    for k, (m, lab) in enumerate([('f1', 'F1'), ('recall', 'Recall')]):
        ax.bar(x + (k - 0.5) * w, [T.loc[d, m] for d, _ in DET], w * 0.92, yerr=[T.loc[d, m + '_sd'] for d, _ in DET],
               color=C[k], label=lab, error_kw=dict(lw=0.8, capsize=2, ecolor=MUTED))
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in DET], fontsize=8); ax.set_ylim(0, 1.08); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Mean ± sd, 10 runs'); ax.legend(loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_baselines.png'); plt.close(fig)


def fig6():
    T = pd.read_csv(f'{R}/T_main.csv').set_index('detector')
    sc = [('rec_single_source', 'Single-source'), ('rec_spraying', 'Spraying'), ('rec_distributed', 'Distributed'),
          ('rec_low_and_slow', 'Low-and-slow')]
    fig, ax = plt.subplots(figsize=(10, 3.0)); x = np.arange(len(sc)); w = 0.135
    for k, (d, lab) in enumerate(DET):
        ax.bar(x + (k - 2.5) * w, [T.loc[d, s] for s, _ in sc], w * 0.9, color=C[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in sc]); ax.set_ylim(0, 1.05); ax.set_ylabel('Recall (mean, 10 runs)')
    ax.grid(axis='x', visible=False); ax.legend(ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(f'{OUT}/fig6_scenarios.png'); plt.close(fig)


def fig7():
    T = pd.read_csv(f'{R}/windows_raw.csv')
    T['feat'] = np.where(T['features'].str.startswith('failure-count'), 'failure-count', T['features'])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
    for ax, ov, ttl in zip(axs, [False, True], ['(a) Non-overlapping windows', '(b) 50% overlapping windows']):
        g = T[T.overlap50 == ov]
        for k, (lab, f) in enumerate([('XGBoost, single-window', 'single-window'), ('XGBoost, stateful', 'stateful'),
                                      ('Failure-count rule', 'failure-count')]):
            s = g[g.feat == f].groupby('window_s')['f1'].agg(['mean', 'std'])
            ax.errorbar(s.index, s['mean'], yerr=s['std'], color=C[k], lw=2, marker='o', ms=5, capsize=2, label=lab)
        ax.set_xscale('log'); ax.set_xticks([60, 120, 300, 900]); ax.set_xticklabels(['60', '120', '300', '900'])
        ax.minorticks_off()
        ax.set_xlabel('Window length Δ (s)'); ax.set_title(ttl, fontsize=9, color=INK); ax.set_ylim(0, 1.05)
    axs[0].set_ylabel('F1 (mean ± sd, 10 runs)'); axs[1].legend(loc='lower left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig7_windows.png'); plt.close(fig)


if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7']:
        globals()[f]()
