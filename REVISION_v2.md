# Release v2.0 — code and outputs behind the revised paper

Reproduce every simulated-data table (Python 3.13, scikit-learn 1.9.1, xgboost 3.4.1, imbalanced-learn 0.14.2):

```
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn matplotlib
python run_synthetic.py --stage main      --runs 10   # Tables 5,6,8,9,10 (+ McNemar, TreeSHAP)
python run_synthetic.py --stage ablation  --runs 10   # Tables 7,11
python run_synthetic.py --stage ablation2 --runs 10   # Table 11 (success-derived features)
python run_synthetic.py --stage windows   --runs 10   # Table 13, Figure 7
python run_synthetic.py --stage stress    --runs 10   # Table 12
python paper_numbers.py && python make_figures.py      # pooled numbers (numbers.json) and figures
```

Run r generates corpus r with `simulate_auth.py --seed r` and trains all models with seed r.
Split: strict chronological 60/20/20 on window end time; no campaign re-grouping
(the v1 `make_split` campaign grouping in bf_eval.py is NOT used by these scripts).

Real-traffic laboratory (D1.csv from label_from_lab.py):

```
python run_real.py --data D1.csv --out results_real --seeds 10   # no SMOTE: attacks are the majority class
```

`results_synth/` holds the per-run raw outputs (`*_raw.csv`) and pooled values (`numbers.json`) used in the paper.
The hashed laboratory event table is `data/D1_hashed.csv` (10,409 events; source IPs and usernames hashed).
Its outputs (Tables 14-15 and the laboratory window/ablation analyses) are in `results_real/`:

```
python run_real.py --data data/D1_hashed.csv --out results_real --seeds 10
```
