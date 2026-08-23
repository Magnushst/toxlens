# Result tables

The root CSV files report the selected single-model analysis unless their names
state otherwise. `ensemble/ensemble_metrics.csv` contains the per-endpoint
five-seed soft-voting ensemble results; its primary-task macro values are MCC
0.437338, AUROC 0.834059, AUPRC 0.579910, and F1 0.537157.

`baselines/` contains Random Forest, XGBoost, MLP, and SVM results for all 11
primary endpoints. `interpretability/` contains the reported GradientSHAP,
occlusion-faithfulness, and toxicophore tables. `tox21_challenge/` is a separate
published-fold benchmark and must not be pooled with the in-house test fold.

Threshold-dependent test metrics use thresholds selected on validation data.
The held-out test fold is not used for threshold selection.
