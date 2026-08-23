# Model and graph provenance

## Authoritative architecture

`src/deep_tox.py` is the complete 11-endpoint training and evaluation program.
It is byte-derived from the final 8,174-line `deep_tox 2.py` research source.
The separate `github_deep_tox.py` file found in the working directory was not
included because it is a 418-line demonstration program that trains on three
dummy molecules and inserts a zero-filled PubChem block; it is not the model
used for the reported experiments.

## Selected single model

`checkpoints/main/toxlens_single_best.ckpt` was copied from:

```text
deep_tox_classification_20260526_143318/
deep_tox_classification_20260526_143318-mcc-016-0.5133.ckpt
```

This checkpoint is used for the reported single-model evaluation and the
interpretability analyses.

## Selected five-seed ensemble

The ensemble assets come from `deep_tox_ensemble_20260526_190534`, the run whose
saved `ensemble_metrics.csv` reproduces the reported macro test values:

| Metric | Value |
|---|---:|
| MCC | 0.437338 |
| AUROC | 0.834059 |
| AUPRC | 0.579910 |
| F1 | 0.537157 |

The source checkpoints selected by validation MCC were:

| Seed | Source checkpoint | Validation MCC |
|---:|---|---:|
| 42 | `ensemble_model_1_seed_42-040-0.5178.ckpt` | 0.5178 |
| 43 | `ensemble_model_2_seed_43-021-0.5211.ckpt` | 0.5211 |
| 44 | `ensemble_model_3_seed_44-018-0.5137.ckpt` | 0.5137 |
| 45 | `ensemble_model_4_seed_45-014-0.5146.ckpt` | 0.5146 |
| 46 | `ensemble_model_5_seed_46-016-0.5085.ckpt` | 0.5085 |

They are renamed `toxlens_seed_<seed>.ckpt` in this package. The saved
validation probabilities, test probabilities, validation-derived thresholds,
and per-task ensemble metrics accompany the checkpoints.

## Tox21 Challenge model

`checkpoints/tox21/toxlens_tox21_challenge.ckpt` was copied from:

```text
toxlens_tox21_challenge_20260527_092312/
toxlens_tox21_challenge_20260527_092312-mcc-013-0.4041.ckpt
```

## Graph bundle

`graph_bundle/pyg_graphs_class.pkl` was copied from the later
`pyg_graphs_class_3.pkl` serialisation. Both archived serialisations produced
effectively identical predictions from the selected single checkpoint against
the saved Ames probabilities. The later serialisation was retained as the
checkpoint-era feature bundle.

The graph bundle contains 26,958 training graphs, 2,796 validation graphs, and
2,970 test graphs. Its associated training dataframe includes 5,394 legacy rows
with all 11 primary labels missing; those rows receive no supervised loss under
the masked objective. The clean publication split therefore contains the
21,578 training molecules with at least one primary label.

## Inactive compatibility block

The final 200 positions of every 3,190-dimensional global-feature tensor were
audited across all 32,724 bundled graphs. Every value is zero. PubChem
bioactivity was therefore not an active computed modality in the reported
models and must not be claimed as such. The positions remain in the tensor only
to preserve checkpoint dimensions; fresh featurisation deterministically fills
them with zeros.
