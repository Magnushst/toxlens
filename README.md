# ToxLens

ToxLens is a leakage-aware, uncertainty-calibrated, multi-task molecular
toxicity framework covering 11 binary endpoints: Ames, LD50_Zhu, hERG_Karim,
NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, SR-ARE, SR-HSE, SR-MMP, and SR-p53.

The repository contains the fixed train, validation, and held-out test folds;
the model, benchmarking, uncertainty, and interpretability code; the exact
result tables used by the manuscript; and environment specifications. Large
checkpoint files are kept under `release_assets/` for upload as GitHub Release
or archival-repository assets and are excluded from ordinary Git history.

## Installation

The readable environment specification is `environment.yml`. The accompanying
Windows Conda and pip lock files record the package state used for the reported
analyses.

```bash
conda env create -f environment.yml
conda activate toxlens
```

For an exact Windows reconstruction, create the Conda environment from
`environment-win-64.lock.txt`, then install the pip-resolved packages recorded
in `requirements-pip-lock.txt`. Use `environment.yml` for a portable
cross-platform installation.

## Main pipeline

Run commands from the repository root.

```bash
# Build all molecular features and the fixed graph bundle.
python -u run_pipeline.py featurise

# Train the 11-endpoint model.
python -u run_pipeline.py train

# Evaluate a selected checkpoint on the frozen test fold.
python -u run_pipeline.py evaluate \
  --graph-cache release_assets/graph_bundle/pyg_graphs_class.pkl \
  --checkpoint release_assets/checkpoints/main/toxlens_single_best.ckpt

# Reproduce the four ECFP4 shallow baselines on all 11 endpoints.
python -u run_pipeline.py baselines
```

The first featurisation is computationally intensive. Generated feature and
graph caches are written to `artifacts/` and are intentionally excluded from
Git.

For exact checkpoint-era inference, the release-assets bundle also contains
`graph_bundle/pyg_graphs_class.pkl`. New training should use the clean split
CSVs; see `release_assets/README.md` for the provenance distinction.

The archived checkpoints retain a 200-dimensional zero-filled compatibility
block in their global input tensors. It is not a computed PubChem modality; the
complete tensor audit is recorded in `release_assets/MODEL_PROVENANCE.md`.

## Interpretability

Full GradientSHAP and graph-feature occlusion:

```bash
python -u src/shap_and_saliency_vis.py \
  --graph_path release_assets/graph_bundle/pyg_graphs_class.pkl \
  --class_ckpt release_assets/checkpoints/main/toxlens_single_best.ckpt \
  --tasks Ames,LD50_Zhu,hERG_Karim,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,SR-ARE,SR-HSE,SR-MMP,SR-p53 \
  --device cuda \
  --out_dir results/interpretability/shap_occlusion
```

Consensus toxicophore discovery and counterfactual validation:

```bash
python -u src/auto_toxico_disc_algo.py \
  --graph_path release_assets/graph_bundle/pyg_graphs_class.pkl \
  --class_ckpt release_assets/checkpoints/main/toxlens_single_best.ckpt \
  --tasks Ames,LD50_Zhu,hERG_Karim,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,SR-ARE,SR-HSE,SR-MMP,SR-p53 \
  --num_molecules 2970 \
  --device cuda \
  --out_dir results/interpretability/toxicophore_discovery
```

## Published-fold benchmarks

The benchmark pipeline downloads the published folds, creates validation data
only from each development fold, retrains ToxLens from scratch, and evaluates
the untouched published test fold.

```bash
python -u src/fetch_external_benchmarks.py \
  --deep-tox src/deep_tox.py \
  --benchmarks tdc_ames,tdc_herg,tdc_dili \
  --seeds 1,2,3,4,5 \
  --output-dir published_benchmark_runs
```

The separate Tox21 Challenge analysis and its machine-readable results are
included under `results/tox21_challenge/`.

## Repository contents

- `data/`: curated 11-endpoint source table.
- `data/`: fixed 80:10:10 train, validation, and test assignments.
- `src/`: model, benchmark, figure, uncertainty, and interpretation programs.
- `benchmarks/`: benchmark provenance and published-method comparisons.
- `release_assets/`: selected checkpoint binaries and SHA-256 checksums.

## Citation

Citation metadata are provided in `CITATION.cff`. Until a journal DOI is
assigned, cite the repository release and the accompanying manuscript.

## Licence

The source code is distributed under the MIT Licence. Dataset components remain
subject to the terms of their original providers; see the manuscript and
benchmark provenance records for source citations.
