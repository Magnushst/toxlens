<img width="300" height="100" alt="toxlens-logo" src="https://github.com/user-attachments/assets/d1b8ae96-f1d5-4225-818a-cb239991be32" />![U<svg viewBox="0 0 480 160" xmlns="http://www.w3.org/2000/svg" font-family="'Helvetica Neue', Arial, sans-serif">
  <defs>
    <radialGradient id="glass" cx="38%" cy="34%" r="75%">
      <stop offset="0%" stop-color="#8FD0E8" stop-opacity="0.35"/>
      <stop offset="60%" stop-color="#2E7DA1" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#17324D" stop-opacity="0.04"/>
    </radialGradient>
  </defs>

  <!-- ICON -->
  <g stroke-linecap="round" stroke-linejoin="round">
    <!-- lens glass -->
    <circle cx="80" cy="78" r="42" fill="url(#glass)"/>

    <!-- molecule: aromatic ring as node-graph -->
    <g stroke="#17324D" fill="none" stroke-width="3.4">
      <polygon points="80,52 100.8,64 100.8,88 80,100 59.2,88 59.2,64"/>
      <circle cx="80" cy="76" r="12.5" stroke-width="2.6"/>
    </g>
    <g fill="#17324D">
      <circle cx="100.8" cy="64" r="5"/>
      <circle cx="100.8" cy="88" r="5"/>
      <circle cx="80" cy="100" r="5"/>
      <circle cx="59.2" cy="88" r="5"/>
      <circle cx="59.2" cy="64" r="5"/>
    </g>
    <!-- SHAP-flagged node -->
    <circle cx="80" cy="52" r="6" fill="#E2564A"/>

    <!-- lens ring + handle -->
    <circle cx="80" cy="78" r="46" fill="none" stroke="#2E7DA1" stroke-width="7"/>
    <line x1="111" y1="109" x2="130" y2="128" stroke="#2E7DA1" stroke-width="12"/>
    <!-- glass highlight -->
    <path d="M 58 58 A 30 30 0 0 1 74 46" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-opacity="0.55"/>
  </g>

  <!-- WORDMARK -->
  <text x="290" y="90" font-size="58" font-weight="700" letter-spacing="-1" text-anchor="middle">
    <tspan fill="#17324D">Tox</tspan><tspan fill="#2E7DA1">Lens</tspan>
  </text>
  <text x="290" y="110" font-size="9" font-weight="600" letter-spacing="0.2" fill="#5B6B7A" text-anchor="middle">
    UNCERTAINTY-CALIBRATED · INTERPRETABLE · LEAKAGE-AWARE
  </text>
  <text x="290" y="126" font-size="11.5" font-weight="500" letter-spacing="1.4" fill="#5B6B7A" text-anchor="middle">
    MOLECULAR TOXICITY PREDICTION
  </text>
</svg>
ploading toxlens-logo.svg…]()


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
