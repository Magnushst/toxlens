# Data layout

`curated/tox_data_classification.csv` is the consolidated 11-endpoint source
table used to construct the study dataset. Original dataset attribution and
licensing remain with the providers cited in the manuscript.

`splits/train.csv`, `splits/validation.csv`, and `splits/test.csv` are the
frozen publication folds. They contain 21,578, 2,798, and 2,971 unique SMILES,
respectively, with no exact-SMILES overlap between folds. Each file uses the
same ordered columns:

```text
smiles, Ames, LD50_Zhu, hERG_Karim, NR-AhR, NR-Aromatase, NR-ER,
NR-ER-LBD, SR-ARE, SR-HSE, SR-MMP, SR-p53
```

Missing endpoint labels are represented as empty CSV fields and are masked in
the multi-task objective. Do not resplit these files when reproducing reported
results. The checkpoint-era graph serialisation is separately documented under
`release_assets/` because it retains legacy unlabelled training records.
