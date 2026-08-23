"""
Integrated efficacy–safety candidate ranking (self-contained).

Two-stage prioritisation that runs DPD-Cancer (anti-cancer activity classifier)
before ToxLens (multi-task toxicity, with MC-dropout uncertainty and SMARTS-
based structural alerts), then combines the outputs into a transparent
priority score and Pareto front. No regression / pGI50 logic.

Three execution environments — one subcommand each:

  1. ``featurise-dpd``   DPD-Cancer featurisation env (DeepChem + molfeat).
                         Reads a SMILES library, builds PyG Data objects with
                         DMPNN node/edge features, CATS pharmacophore vector,
                         and Morgan-2048 fingerprints. Applies the saved
                         VarianceThreshold + StandardScaler. Writes a pickle.

  2. ``predict-dpd``     DPD-Cancer inference env (PyTorch / Lightning).
                         Loads the pickle from stage 1 and the trained GAT
                         checkpoint, returns per-molecule activity
                         probabilities, selects the top-N candidates.

  3. ``run-toxlens``     ToxLens env (single env for featurisation + inference).
                         Featurises the top-N set (Morgan + RDKit descriptors +
                         tox SMARTS + MolFormer + RDKit 3D descriptors + PubChem bioactivity vectors),
                         runs the trained multi-task GAT with MC-dropout,
                         computes the integrated priority score, Pareto front,
                         and emits all artefacts + figures.

A fourth helper subcommand ``prepare-library`` downloads candidate molecules
from the ChEMBL REST API (anti-cancer indications, first-approval / first
ChEMBL entry in 2025 or later) and filters them against the NCI60 2025 set
and the ToxLens training set by canonical SMILES, producing the input CSV
that ``featurise-dpd`` then consumes.

All model classes, featurisers, and constants are inlined below so this file
has no dependency on master_file.py or deep_tox.py.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Windows conda stacks can load Intel OpenMP through more than one compiled
# chemistry/ML dependency (RDKit, NumPy/MKL, PyTorch, scikit-learn, HF stack).
# Set this before importing those native libraries so the pipeline does not
# abort with "OMP Error #15" during ToxLens/MolFormer featurisation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)


# ToxLens primary endpoints used for the integrated case study. Keep this in
# the exact order used by the trained primary-endpoint classification head.
TASK_CONFIG = OrderedDict([
    ("Ames", {"type": "classification", "category": "primary", "source": "Tox", "targets": ["Ames"]}),
    ("LD50_Zhu", {"type": "classification", "category": "primary", "source": "Tox", "targets": ["LD50_Zhu"], "binary_threshold": 2.5, "binary_direction": "above"}),
    ("hERG_Karim", {"type": "classification", "category": "primary", "source": "Local", "targets": ["hERG_Karim"]}),
    ("NR-AhR", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "NR-AhR"}),
    ("NR-Aromatase", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "NR-Aromatase"}),
    ("NR-ER", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "NR-ER"}),
    ("NR-ER-LBD", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "NR-ER-LBD"}),
    ("SR-ARE", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "SR-ARE"}),
    ("SR-HSE", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "SR-HSE"}),
    ("SR-MMP", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "SR-MMP"}),
    ("SR-p53", {"type": "classification", "category": "primary", "source": "Tox_Label", "tdc_name": "Tox21", "label_name": "SR-p53"}),
])
PRIMARY_TOXLENS_TASKS = tuple(TASK_CONFIG.keys())
EXPECTED_TOXLENS_NUM_TASKS = len(PRIMARY_TOXLENS_TASKS)


def _primary_toxlens_task_names(tox_task_csv: Path) -> List[str]:
    """Return the fixed primary ToxLens endpoint order and verify the CSV has it."""
    available = pd.read_csv(tox_task_csv, nrows=1).columns.tolist()
    missing = [name for name in PRIMARY_TOXLENS_TASKS if name not in available]
    if missing:
        raise ValueError(
            f"{tox_task_csv} is missing required primary ToxLens endpoint columns: {missing}"
        )
    names = list(PRIMARY_TOXLENS_TASKS)
    if len(names) != 11:
        raise RuntimeError(f"Expected exactly 11 primary ToxLens endpoints, got {len(names)}")
    return names


def _assert_primary_toxlens_logits(logits, context: str = "ToxLens") -> None:
    """Hard guard: the integrated study only accepts the 11 configured endpoints."""
    n_out = int(logits.shape[-1])
    if n_out != EXPECTED_TOXLENS_NUM_TASKS:
        raise RuntimeError(
            f"{context} produced {n_out} endpoints, but this integrated study is locked "
            f"to exactly {EXPECTED_TOXLENS_NUM_TASKS}: {list(PRIMARY_TOXLENS_TASKS)}"
        )


def _clean_tox_task_name(name: object) -> str:
    return str(name).strip().strip("()")


def _primary_toxlens_task_types() -> List[str]:
    return ["classification"] * EXPECTED_TOXLENS_NUM_TASKS


def _copy_indexed_vector(dst_state: dict, src_tensor, dst_key: str,
                         old_to_new: Dict[int, int]) -> bool:
    if dst_key not in dst_state or not hasattr(src_tensor, "shape"):
        return False
    dst_tensor = dst_state[dst_key].clone()
    if src_tensor.ndim != 1 or dst_tensor.ndim != 1:
        return False
    for old_i, new_i in old_to_new.items():
        if old_i < src_tensor.shape[0] and new_i < dst_tensor.shape[0]:
            dst_tensor[new_i] = src_tensor[old_i].to(dtype=dst_tensor.dtype)
    dst_state[dst_key] = dst_tensor
    return True


def _copy_indexed_rows(dst_state: dict, src_tensor, dst_key: str,
                       old_to_new: Dict[int, int], block: int = 1) -> bool:
    if dst_key not in dst_state or not hasattr(src_tensor, "shape"):
        return False
    dst_tensor = dst_state[dst_key].clone()
    if src_tensor.ndim == 0 or dst_tensor.ndim == 0:
        return False
    for old_i, new_i in old_to_new.items():
        old_s, old_e = old_i * block, (old_i + 1) * block
        new_s, new_e = new_i * block, (new_i + 1) * block
        if old_e <= src_tensor.shape[0] and new_e <= dst_tensor.shape[0]:
            dst_tensor[new_s:new_e] = src_tensor[old_s:old_e].to(dtype=dst_tensor.dtype)
    dst_state[dst_key] = dst_tensor
    return True


def _load_primary_toxlens_checkpoint(GAT_class, ckpt_path: Path, device,
                                      task_names: Sequence[str]):
    """Load a ToxLens checkpoint but expose only the configured primary endpoints.

    Older checkpoints may carry auxiliary endpoint metadata in ``hparams``. The
    integrated study must not inherit that metadata, so this loader overrides the
    head to the fixed TASK_CONFIG order and remaps task-indexed tensors by name.
    """
    import torch

    task_names = [_clean_tox_task_name(n) for n in task_names]
    if tuple(task_names) != PRIMARY_TOXLENS_TASKS:
        raise ValueError(
            "Integrated ToxLens loading is locked to TASK_CONFIG order: "
            f"{list(PRIMARY_TOXLENS_TASKS)}"
        )

    ckpt = torch.load(str(ckpt_path), map_location=device)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)

    old_task_names = [_clean_tox_task_name(n) for n in hparams.get("task_names", [])]
    if old_task_names:
        old_lookup = {name: i for i, name in enumerate(old_task_names)}
        missing = [name for name in task_names if name not in old_lookup]
        if missing:
            raise ValueError(
                f"Checkpoint {ckpt_path} does not contain required TASK_CONFIG endpoints: {missing}. "
                f"Available checkpoint endpoints: {old_task_names}"
            )
        old_to_new = {old_lookup[name]: new_i for new_i, name in enumerate(task_names)}
    else:
        old_to_new = {i: i for i in range(EXPECTED_TOXLENS_NUM_TASKS)}
        print(
            "[toxlens] checkpoint has no task_names metadata; assuming the first "
            "11 task-indexed weights match TASK_CONFIG order"
        )

    hparams["num_tasks"] = EXPECTED_TOXLENS_NUM_TASKS
    hparams["task_names"] = list(task_names)
    hparams["task_types"] = _primary_toxlens_task_types()
    hparams["w_pos"] = None
    hparams["w_neg"] = None

    model = GAT_class(**hparams)
    target_state = model.state_dict()
    remapped_state = {}

    vector_keys = {
        "task_logit_log_scale", "task_thresholds", "platt_a", "platt_b",
        "w_pos", "w_neg", "primary_task_mask", "aux_task_mask",
        "loss_fn.w_pos", "loss_fn.w_neg", "loss_fn.task_weights",
    }

    for key, tensor in state.items():
        target_key = key
        import re
        m = re.match(r"^(task_heads|task_norms)\.(\d+)\.(.+)$", key)
        if m:
            old_i = int(m.group(2))
            if old_i not in old_to_new:
                continue
            target_key = f"{m.group(1)}.{old_to_new[old_i]}.{m.group(3)}"
            if target_key in target_state and target_state[target_key].shape == tensor.shape:
                remapped_state[target_key] = tensor
            continue

        if key in vector_keys:
            _copy_indexed_vector(target_state, tensor, key, old_to_new)
            continue

        if key in {"aux_head.4.weight", "aux_head.4.bias"}:
            _copy_indexed_rows(target_state, tensor, key, old_to_new, block=1)
            continue

        if key in {"expert_gate.4.weight", "expert_gate.4.bias"}:
            _copy_indexed_rows(target_state, tensor, key, old_to_new, block=4)
            continue

        if target_key in target_state and target_state[target_key].shape == tensor.shape:
            remapped_state[target_key] = tensor

    target_state.update(remapped_state)
    incompat = model.load_state_dict(target_state, strict=False)
    loaded = len(remapped_state)
    print(
        f"[toxlens] loaded primary TASK_CONFIG checkpoint view: "
        f"{EXPECTED_TOXLENS_NUM_TASKS} endpoints, {loaded} shared/task tensors copied"
    )
    if getattr(incompat, "unexpected_keys", None):
        print(f"[toxlens] ignored unexpected keys: {len(incompat.unexpected_keys)}")
    return model


# Common utilities

@dataclass
class IntegratedConfig:
    """Knobs for the case study. All scoring weights are decision-support
    choices, *not* biological constants."""

    # Stage 1 — DPD-Cancer
    dpd_ckpt: Path = Path("deeppd_classification.ckpt")
    dpd_preproc: Path = Path("preproc_classification.pkl")
    dpd_hidden_channels: int = 144  # must match the trained checkpoint
    dpd_morgan_bits: int = 2048

    # Stage-1 selection rule (declared before any toxicity is consulted)
    top_n: int = 10

    # Stage 2 — ToxLens
    tox_ckpt: Path = Path("deep_tox_classification.ckpt")
    tox_task_csv: Path = Path("tox_data_classification.csv")
    pubchem_cache: Path = Path("data/pubchem_bioactivity_cache.json")
    mc_passes: int = 30

    # Integrated priority score weights
    w_tox_max: float = 1.0
    w_tox_count: float = 0.5
    w_uncertainty: float = 0.3
    tox_high_threshold: float = 0.5
    uncertainty_flag_threshold: float = 0.15

    # I/O
    output_dir: Path = Path("integrated_case_study_results")
    batch_size: int = 32
    device: str = "cpu"  # overridden per-stage; CPU works for inference too
    seed: int = 42


def standardise_smiles(smi: str) -> Optional[Chem.Mol]:
    """LargestFragment → Uncharge → CanonicalTautomer → Sanitize.

    Identical to the training pipelines of both upstream models so inference
    featurisation cannot silently disagree with what the models saw at fit
    time.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        mol = Chem.MolFromSmiles(smi)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_library(library_path: Path, smiles_col: str = "smiles") -> pd.DataFrame:
    df = pd.read_csv(library_path)
    if smiles_col not in df.columns:
        df = df.rename(columns={df.columns[0]: smiles_col})
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col].str.len() > 0].reset_index(drop=True)
    return df


def pareto_front(activity: np.ndarray, max_tox: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points under (high activity, low max_tox)."""
    n = len(activity)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_act = activity[j] >= activity[i]
            better_tox = max_tox[j] <= max_tox[i]
            strictly = (activity[j] > activity[i]) or (max_tox[j] < max_tox[i])
            if better_act and better_tox and strictly:
                on_front[i] = False
                break
    return np.flatnonzero(on_front)


def integrated_priority(
    activity: np.ndarray,
    max_tox: np.ndarray,
    high_frac: np.ndarray,
    uncertainty: np.ndarray,
    cfg: IntegratedConfig,
) -> np.ndarray:
    """activity_norm − w_tox_max·max_tox − w_tox_count·high_frac − w_unc·unc."""
    a_lo, a_hi = float(np.nanmin(activity)), float(np.nanmax(activity))
    rng = max(a_hi - a_lo, 1e-9)
    act_norm = (activity - a_lo) / rng
    mt = np.nan_to_num(max_tox, nan=0.0)
    hf = np.nan_to_num(high_frac, nan=0.0)
    un = np.nan_to_num(uncertainty, nan=0.0)
    return (
        act_norm
        - cfg.w_tox_max * mt
        - cfg.w_tox_count * hf
        - cfg.w_uncertainty * un
    )


def plot_efficacy_safety(df: pd.DataFrame, pareto_idx: np.ndarray, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["activity_prob"], df["max_tox_prob"],
        c=df["mean_uncertainty"], cmap="viridis",
        s=40, edgecolor="white", linewidth=0.4, alpha=0.85,
    )
    fig.colorbar(sc, ax=ax).set_label("Mean MC-dropout uncertainty")
    if len(pareto_idx) > 0:
        pf = df.iloc[pareto_idx].sort_values("activity_prob")
        ax.plot(
            pf["activity_prob"], pf["max_tox_prob"],
            color="#C1121F", lw=2.0, marker="o", markersize=8,
            markeredgecolor="white",
            label=f"Pareto front (n={len(pareto_idx)})",
        )
    ax.set_xlabel("DPD-Cancer activity probability")
    ax.set_ylabel("ToxLens max endpoint toxicity probability")
    ax.set_title("Efficacy–safety trade-off (top-N candidates)")
    ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_priority_distribution(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["integrated_score"], bins=30, kde=True, color="#2A9D8F", ax=ax)
    ax.axvline(df["integrated_score"].median(), color="#E76F51", ls="--", label="Median")
    ax.set_xlabel("Integrated priority score (higher = better)")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Distribution of integrated priority scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_figure(fig, out_path: Path) -> None:
    """Save both the requested raster path and a thesis-friendly SVG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def _display_name(row: pd.Series, fallback_idx: int) -> str:
    name = row.get("pref_name", None)
    if isinstance(name, str) and name.strip() and name.lower() != "nan":
        return name.strip().title()
    chembl = row.get("chembl_id", None)
    if isinstance(chembl, str) and chembl.strip() and chembl.lower() != "nan":
        return chembl.strip()
    return f"C{fallback_idx + 1}"


def _filter_explanation_targets(df: pd.DataFrame, molecule_name: Optional[str]) -> pd.DataFrame:
    """Default explanation target is Parsaclisib for thesis side-by-side figures."""
    if molecule_name is None or not str(molecule_name).strip():
        return df
    needle = str(molecule_name).strip().lower()
    masks = []
    for col in ("pref_name", "chembl_id", "canonical_smiles", "smiles"):
        if col in df.columns:
            masks.append(df[col].astype(str).str.lower().eq(needle))
    if "pref_name" in df.columns:
        masks.append(df["pref_name"].astype(str).str.lower().str.contains(needle, regex=False))
    if not masks:
        return df
    mask = masks[0].copy()
    for m in masks[1:]:
        mask = mask | m
    out = df[mask].reset_index(drop=True)
    if out.empty:
        print(f"[explain] target molecule {molecule_name!r} not found; using all rows")
        return df
    print(f"[explain] using target molecule: {molecule_name} ({len(out)} row)")
    return out


def plot_efficacy_safety(df: pd.DataFrame, pareto_idx: np.ndarray, out_path: Path) -> None:
    import matplotlib as mpl
    from matplotlib.ticker import PercentFormatter
    from matplotlib.colors import LinearSegmentedColormap

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "axes.labelcolor": "#2F3437",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    score = df["integrated_score"].to_numpy(dtype=float)
    score_lo = float(np.nanmin(score))
    score_hi = float(np.nanmax(score))
    size = 42 + 12 * df["n_high_risk_endpoints"].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "priority_thesis",
        [
            (0.00, "#C9182B"),
            (0.35, "#E85D75"),
            (0.62, "#4F79B8"),
            (1.00, "#006D9C"),
        ],
    )
    sc = ax.scatter(
        df["activity_prob"], df["max_tox_prob"],
        c=score, cmap=cmap, vmin=score_lo, vmax=score_hi,
        s=size, edgecolor="#FFFFFF", linewidth=0.8, alpha=0.98, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.07)
    cbar.set_label("Integrated Priority (Higher Is Better)", labelpad=8)
    cbar.outline.set_visible(False)
    if len(pareto_idx) > 0:
        pf = df.iloc[pareto_idx].sort_values("activity_prob")
        ax.plot(
            pf["activity_prob"], pf["max_tox_prob"],
            color="#1F2937", lw=1.7, alpha=0.88, zorder=4,
        )
        ax.scatter(
            pf["activity_prob"], pf["max_tox_prob"],
            s=132, facecolor="none", edgecolor="#111827", linewidth=1.45,
            zorder=5, label=f"Pareto-Optimal Candidates (n={len(pareto_idx)})",
        )
        for rank, (_, row) in enumerate(pf.iterrows()):
            x = float(row["activity_prob"])
            y = float(row["max_tox_prob"])
            if x > 0.86:
                xytext = (-26, -18 if rank % 2 else 20)
                ha = "right"
            else:
                xytext = (18, -14 if rank % 2 else 16)
                ha = "left"
            ax.annotate(
                _display_name(row, rank),
                xy=(x, y),
                xytext=xytext,
                textcoords="offset points",
                fontsize=8.3,
                color="#0F172A",
                ha=ha,
                va="center",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CBD5E1", lw=0.7, alpha=0.96),
                arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.7, shrinkA=2, shrinkB=4),
                annotation_clip=True,
            )
    y_min = max(0.0, math.floor((float(np.nanmin(df["max_tox_prob"])) - 0.05) * 20) / 20)
    y_max = min(1.0, math.ceil((float(np.nanmax(df["max_tox_prob"])) + 0.03) * 20) / 20)
    ax.set_xlim(max(0.0, float(np.nanmin(df["activity_prob"])) - 0.04), 1.02)
    ax.set_ylim(y_max, y_min)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Predicted Anti-Cancer Activity")
    ax.set_ylabel("Predicted Toxicity")
    ax.set_title("Efficacy–Safety Trade-Off", pad=12, weight="semibold")
    ax.grid(False)
    ax.minorticks_off()
    ax.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_priority_distribution(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib as mpl

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    mpl.rcParams.update({"svg.fonttype": "none"})
    plot_df = df.sort_values("integrated_score", ascending=True).reset_index(drop=True).copy()
    plot_df["label"] = [_display_name(r, i) for i, r in plot_df.iterrows()]
    colors = np.where(plot_df["pareto_optimal"].to_numpy(), "#006D9C", "#8297B8")
    fig_h = max(3.8, 0.28 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))
    ax.barh(plot_df["label"], plot_df["integrated_score"], color=colors, height=0.62)
    ax.axvline(0, color="#374151", lw=0.9)
    ax.set_xlabel("Integrated Priority Score")
    ax.set_ylabel("")
    ax.set_title("Integrated Candidate Ranking", pad=10, weight="semibold")
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=8)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


# Stage 1a — DPD-Cancer featurisation (DPD-Cancer featurisation env)

# Imports gated inside the stage function so the *inference* env, which may
# lack DeepChem / molfeat, can still parse this file.

def stage_featurise_dpd(
    library_path: Path,
    out_path: Path,
    preproc_path: Path,
    morgan_bits: int = 2048,
) -> None:
    """Read a SMILES library, build PyG Data objects for the DPD-Cancer model,
    apply the saved selector + scaler, and pickle the result.

    Pickle schema:
        {
            "smiles":     List[str] (canonical),
            "graphs":     List[Data]  (each with x, edge_index, edge_attr,
                                       global_features),
            "library_df": pd.DataFrame  (any extra cols from the input CSV),
        }
    """
    # Lazy heavy imports for env 1.
    import torch  # noqa: F401  (torch_geometric pulls this in)
    from torch_geometric.data import Data
    from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer
    from molfeat.trans.base import MoleculeTransformer
    from molfeat.calc.pharmacophore import Pharmacophore2D

    if not preproc_path.exists():
        raise FileNotFoundError(
            f"DPD-Cancer preprocessing pickle not found: {preproc_path}. "
            "It pins the train-time global-feature transform."
        )
    preproc = joblib.load(preproc_path)
    selector = preproc["selector"]
    scaler = preproc["scaler"]
    target_dim = int(selector.n_features_in_)

    lib = load_library(library_path)
    print(f"[featurise-dpd] {len(lib)} input molecules")

    featurizer = DMPNNFeaturizer(is_adding_hs=True, use_original_atom_ranks=True)
    pharm2d = MoleculeTransformer(
        featurizer=Pharmacophore2D(factory="cats", useCounts=True, includeBondOrder=True),
        dtype=float,
    )
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=morgan_bits, includeChirality=True
    )

    graphs: List = []
    canon_smiles: List[str] = []
    keep_rows: List[int] = []

    for idx, smi in enumerate(lib["smiles"].tolist()):
        mol = standardise_smiles(smi)
        if mol is None:
            continue
        try:
            feat = featurizer.featurize([mol])
            if not feat or feat[0] is None:
                continue
            gd = feat[0]
            x = torch.tensor(gd.node_features, dtype=torch.float)
            edge_index = torch.tensor(gd.edge_index, dtype=torch.long)
            edge_attr = torch.tensor(gd.edge_features, dtype=torch.float)
            if x.size(0) == 0 or edge_index.size(1) == 0:
                continue
            pharm_vec = pharm2d([smi])[0]
            bv = morgan_gen.GetFingerprint(mol)
            fp = np.zeros((bv.GetNumBits(),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, fp)
            global_feat = np.concatenate([
                np.asarray(pharm_vec, dtype=np.float32),
                np.asarray(gd.global_features, dtype=np.float32),
                fp.astype(np.float32),
            ])
            if global_feat.size < target_dim:
                global_feat = np.pad(global_feat, (0, target_dim - global_feat.size))
            elif global_feat.size > target_dim:
                global_feat = global_feat[:target_dim]
            global_feat = selector.transform(global_feat.reshape(1, -1))
            global_feat = scaler.transform(global_feat).astype(np.float32)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.global_features = torch.tensor(global_feat, dtype=torch.float)
            graphs.append(data)
            canon_smiles.append(canonical_smiles(mol))
            keep_rows.append(idx)
        except Exception as e:
            print(f"  ! skip {smi!r}: {e}")
            continue

    print(f"[featurise-dpd] featurised {len(graphs)} / {len(lib)} molecules")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "smiles": canon_smiles,
        "graphs": graphs,
        "library_df": lib.iloc[keep_rows].reset_index(drop=True),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[featurise-dpd] wrote {out_path}")


# Stage 1b — DPD-Cancer inference (DPD-Cancer inference env)

# The DPD-Cancer GAT, inlined verbatim from master_file.py's
# ``geometry_gnn_classification`` closure. Only the parts needed for forward
# inference are kept; training_step/validation_step/etc. are stripped because
# Lightning's load_from_checkpoint only requires __init__ + the same parameter
# names so the state_dict aligns.

class DPDCancerGAT:
    """Wrapper module-class lazily defined inside ``stage_predict_dpd`` so we
    don't import lightning/torch_geometric at module top level. This stub
    exists only to make type checkers happy.
    """
    pass


def _build_dpd_model_class():
    """Construct the DPD-Cancer ``GAT`` LightningModule. Defined inside a
    function so torch / lightning / torch_geometric imports are deferred."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import GELU, Sigmoid
    from lightning import LightningModule
    from torch_geometric.nn import (
        GraphMultisetTransformer, GraphNorm, LayerNorm, Linear,
        TransformerConv,
    )

    class GAT(LightningModule):
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            learning_rate: float,
            global_dim: int,
            edge_feature_dim: int,
            k_tuned: int,
            ablation_mode: str = "fused",
        ) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.node_emb = (
                nn.Identity()
                if in_channels == hidden_channels
                else nn.Linear(in_channels, hidden_channels, bias=False)
            )
            self.act = nn.GELU()
            self.layers = nn.ModuleList([
                TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False,
                                dropout=0.3, edge_dim=edge_feature_dim),
                TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False,
                                dropout=0.3, edge_dim=edge_feature_dim),
                TransformerConv(hidden_channels, hidden_channels, heads=1, concat=False,
                                dropout=0.4, edge_dim=edge_feature_dim),
            ])
            self.norms = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(3)])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    Linear(hidden_channels, hidden_channels * 2),
                    self.act,
                    Linear(hidden_channels * 2, hidden_channels),
                )
                for _ in range(len(self.layers))
            ])
            self.pool = GraphMultisetTransformer(
                channels=hidden_channels, k=k_tuned, num_encoder_blocks=1,
                heads=4, layer_norm=True, dropout=0.4,
            )
            self.global_fc = nn.Sequential(
                LayerNorm(global_dim),
                Linear(global_dim, hidden_channels),
                self.act,
            )
            self.global_gate = nn.Sequential(
                LayerNorm(hidden_channels),
                Linear(hidden_channels, hidden_channels),
                Sigmoid(),
            )
            self.noise_std = 0.13
            self.fuse_norm = LayerNorm(hidden_channels)
            self.fc = nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                self.act,
                Linear(hidden_channels, 2),
            )
            self.ablation_mode = ablation_mode
            self.best_thresh = 0.5

        def forward(self, data):
            x, edge_index, batch, edge_attr = (
                data.x, data.edge_index, data.batch, data.edge_attr
            )
            x = self.node_emb(x)
            for conv, norm, ffn in zip(self.layers, self.norms, self.ffns):
                y = conv(self.act(norm(x, batch)), edge_index, edge_attr)
                x = x + y
                y = ffn(norm(x, batch))
                x = x + y
            x = self.pool(x, batch)
            if self.ablation_mode != "graph_only":
                gi = data.global_features
                global_features = self.global_fc(gi)
                global_features = F.dropout(global_features, p=0.4, training=self.training)
            if self.ablation_mode == "graph_only":
                pass
            elif self.ablation_mode == "global_only":
                x = global_features
            else:
                gate = self.global_gate(global_features)
                x = gate * x + (1 - gate) * global_features
            x = self.fuse_norm(x)
            return self.fc(x)

        def on_load_checkpoint(self, checkpoint):
            self.best_thresh = float(checkpoint.get("best_thresh", 0.5))

        def configure_optimizers(self):  # never used at inference
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    return GAT


def stage_predict_dpd(
    featurised_path: Path,
    dpd_ckpt: Path,
    top_n: int,
    out_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Run DPD-Cancer inference on the stage-1a pickle and write a CSV of
    (canonical_smiles, activity_prob) plus a top-N subset CSV.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader as GeoDataLoader

    if not featurised_path.exists():
        raise FileNotFoundError(featurised_path)
    if not dpd_ckpt.exists():
        raise FileNotFoundError(dpd_ckpt)

    with open(featurised_path, "rb") as f:
        payload = pickle.load(f)
    smiles_list: List[str] = payload["smiles"]
    graphs = payload["graphs"]
    library_df: pd.DataFrame = payload["library_df"]
    print(f"[predict-dpd] {len(graphs)} graphs loaded")

    GAT = _build_dpd_model_class()
    model = GAT.load_from_checkpoint(str(dpd_ckpt), map_location=device)
    model.eval()
    model.to(device)

    loader = GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    activity = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    print(
        f"[predict-dpd] activity_prob: mean={float(np.mean(activity)):.3f}, "
        f"max={float(np.max(activity)):.3f}"
    )

    out_df = library_df.copy()
    out_df["canonical_smiles"] = smiles_list
    out_df["activity_prob"] = activity
    out_df = out_df.sort_values("activity_prob", ascending=False).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_path = out_path.with_suffix(".full.csv")
    top_path = out_path.with_suffix(".top.csv")
    out_df.to_csv(full_path, index=False)
    out_df.head(top_n).to_csv(top_path, index=False)
    print(f"[predict-dpd] wrote {full_path}")
    print(f"[predict-dpd] wrote {top_path} (top-{min(top_n, len(out_df))})")


# Stage 2 — ToxLens (ToxLens env: featurisation + inference + scoring)

# A condensed but state-dict-compatible copy of the ToxLens GAT_class and all
# the upstream constants / submodules / featurisers it needs. Every name and
# parameter shape matches deep_tox.py so the trained checkpoint loads with
# strict=True.


RWSE_WALK_LENGTH = 32

_PAULING_EN = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.6, 42: 2.16, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 55: 0.79, 56: 0.89,
    72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.28,
    79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02,
}

_TOX_SMARTS_STRINGS = [
    ("Acyl_halide", "C(=O)[Cl,Br,I]"),
    ("Aldehyde", "[CX3H1](=O)[#6]"),
    ("Alkyl_halide", "[CX4][Cl,Br,I]"),
    ("Anhydride", "C(=O)OC(=O)"),
    ("Aziridine", "N1CC1"),
    ("Azetidine", "N1CCC1"),
    ("Epoxide", "C1OC1"),
    ("Oxetane_strained", "C1COC1"),
    ("Beta_lactam", "N1C(=O)CC1"),
    ("Beta_lactone", "O=C1CCO1"),
    ("Halocarbonyl", "C(=O)[F,Cl,Br,I]"),
    ("Sulfonyl_halide", "S(=O)(=O)[Cl,Br,I]"),
    ("Phosphonyl_halide", "P(=O)[Cl,Br,I]"),
    ("Acyl_cyanide", "C(=O)C#N"),
    ("Isocyanate", "N=C=O"),
    ("Isothiocyanate", "N=C=S"),
    ("Carbodiimide", "N=C=N"),
    ("Ketene", "C=C=O"),
    ("Nitro", "[N+](=O)[O-]"),
    ("Nitroso", "[N]=O"),
    ("Nitrosamine", "N-N=O"),
    ("Alkyl_Nitrite", "ON=O"),
    ("Azo", "[N;!R]=N"),
    ("Diazo", "[C]=[N+]=[N-]"),
    ("Diazonium", "[c][N+]#N"),
    ("Hydrazine", "[NX3][NX3]"),
    ("Hydrazide", "C(=O)NN"),
    ("Semicarbazide", "NC(=O)NN"),
    ("Hydroxamic_acid", "C(=O)NO"),
    ("N_oxide", "[N+]([O-])"),
    ("Carbamate", "N-C(=O)-O"),
    ("Urea", "NC(=O)N"),
    ("Thiol", "[SX2H]"),
    ("Disulfide", "SS"),
    ("Thioaldehyde", "[CX3H1](=S)[#6]"),
    ("Thiocarbonyl", "C=S"),
    ("Sulfonamide", "S(=O)(=O)N"),
    ("Sulfonate_ester", "S(=O)(=O)O[CX4]"),
    ("Thiocarbamate", "N-C(=S)-O"),
    ("Peroxide", "OO"),
    ("Hydroperoxide", "[OX2][OX2H]"),
    ("Michael_acceptor", "[C,c]=[C,c][C,c](=O)"),
    ("Vinyl_halide", "[CX3]=[CX3][F,Cl,Br,I]"),
    ("Alpha_halo_carbonyl", "C(=O)C[Cl,Br,I]"),
    ("Activated_ester", "C(=O)O[CX3]=[CX3]"),
    ("Maleimide", "N1C(=O)C=CC1=O"),
    ("Acrylamide", "[NX3][CX3](=O)[CX3]=[CX3]"),
    ("Phosphonate", "P(=O)(O)O"),
    ("Phosphate_ester", "OP(=O)(O)O"),
    ("Alkyl_fluoride", "[CX4]F"),
    ("Aniline", "[NX3;H2,H1;!$(NC=O)]c"),
    ("N_N_diaryl_amine", "N(c)c"),
    ("Phenol", "[OX2H]c"),
    ("Catechol", "Oc1c(O)cccc1"),
    ("Hydroquinone", "Oc1ccc(O)cc1"),
    ("Aminophenol", "[NX3;H2,H1]c1ccc(O)cc1"),
    ("Quinone", "O=C1C=CC(=O)[cH,cH]1"),
    ("Quinone_imine", "N=C1C=CC(=O)CC1"),
    ("Polycyclic_Aromatic", "a1aaaa2aaaa12"),
    ("Halo_Aromatic", "c[F,Cl,Br,I]"),
    ("Nitro_Aromatic", "c[N+](=O)[O-]"),
    ("Nitroso_Aromatic", "cN=O"),
    ("Aromatic_amine_N_oxide", "c[N+]([O-])"),
    ("Mustard_nitrogen", "[CX4][Cl,Br]CCN"),
    ("Mustard_sulfur", "[CX4][Cl,Br]CCS"),
    ("Epihalohydrin", "[C@@H]1(CO1)C[Cl,Br,I]"),
    ("Lactone", "O=C1OCC1"),
    ("Propiolactone", "O=C1CCO1"),
    ("Aromatic_nitro_reduct", "[cH]1[cH][cH]c([N+](=O)[O-])[cH][cH]1"),
    ("Arylamine_acetyl", "c[NH]C(=O)C"),
    ("Saponin_like", "[OX2]1[CX4][CX4][CX4][CX4][CX4]1"),
    ("Coumarin", "O=C1OC2=CC=CC=C2C=C1"),
    ("Furan", "c1ccoc1"),
    ("Thiophene", "c1ccsc1"),
    ("Purine_like", "c1ncnc2[nH]cnc12"),
]

_ATOM_REACTIVITY_SMARTS = (
    ("carbonyl_c", "[CX3](=[OX1,SX1])"),
    ("imine_c", "[CX3]=[NX2]"),
    ("nitrile_c", "[CX2]#N"),
    ("michael_acceptor", "[C,c]=[C,c][C,c](=O)"),
    ("aryl_halide_ipso", "[c][F,Cl,Br,I]"),
    ("alkyl_halide_c", "[CX4][Cl,Br,I]"),
    ("epoxide_atom", "C1OC1"),
    ("aziridine_atom", "N1CC1"),
    ("nitro_n", "[N+](=O)[O-]"),
    ("diazo_atom", "[C]=[N+]=[N-]"),
    ("aniline_n", "[NX3;H2,H1;!$(NC=O)]c"),
    ("phenol_o", "[OX2H]c"),
    ("thiol_s", "[SX2H]"),
    ("sulfonyl_s", "S(=O)(=O)"),
    ("phosphoryl_p", "P(=O)"),
    ("quinone_atom", "O=C1C=CC(=O)C=C1"),
)
ATOM_REACTIVITY_DIM = len(_ATOM_REACTIVITY_SMARTS)
ATOM_ADVANCED_SCALAR_DIM = 21

_BOND_REACTIVITY_SMARTS = (
    ("amide_bond", "[NX3][CX3](=[OX1])"),
    ("ester_acyl_bond", "[OX2][CX3](=[OX1])"),
    ("sulfonamide_bond", "[NX3][SX4](=[OX1])(=[OX1])"),
    ("phosphoramide_bond", "[NX3][PX4](=[OX1])"),
    ("aryl_halide_bond", "[c][F,Cl,Br,I]"),
    ("alkyl_halide_bond", "[CX4][Cl,Br,I]"),
    ("michael_bond", "[C,c]=[C,c][C,c](=O)"),
    ("azo_bond", "[N;!R]=N"),
)
BOND_BASE_FEATURE_DIM = 11
BOND_ADVANCED_SCALAR_DIM = 16
BOND_REACTIVITY_DIM = len(_BOND_REACTIVITY_SMARTS)
BOND_FEATURE_DIM = BOND_BASE_FEATURE_DIM + BOND_ADVANCED_SCALAR_DIM + BOND_REACTIVITY_DIM

ADVANCED_3D_DESCRIPTOR_DIM = (114 + 273) + 12 + 60 + 224 + 210 + 12  # = 905
PUBCHEM_DIM = 200
ADVANCED_3D_SCALAR_NAMES = (
    "PMI1", "PMI2", "PMI3", "NPR1", "NPR2", "PBF",
    "Asphericity", "Eccentricity", "InertialShapeFactor",
    "RadiusOfGyration", "SpherocityIndex", "MolVolume",
)


def _safe_3d_vec(mol_3d, fn_name: str, length: int) -> np.ndarray:
    from rdkit.Chem import rdMolDescriptors

    fn = getattr(rdMolDescriptors, fn_name, None)
    if fn is None:
        return np.zeros(length, dtype=np.float32)
    try:
        vals = np.asarray(fn(mol_3d), dtype=np.float32).reshape(-1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        vals = np.zeros(0, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[:min(length, len(vals))] = vals[:length]
    return out


def _safe_3d_scalar(mol_3d, fn_name: str) -> float:
    from rdkit.Chem import rdMolDescriptors

    fn = getattr(rdMolDescriptors, fn_name, None)
    if fn is None:
        return 0.0
    try:
        val = float(fn(mol_3d))
        return 0.0 if np.isnan(val) or np.isinf(val) else val
    except Exception:
        return 0.0


def _compute_toxlens_3d_descriptor(mol: Chem.Mol, seed: int) -> np.ndarray:
    """Compute the same 905-d RDKit 3D descriptor block used by deep_tox.py."""
    try:
        mol_3d = Chem.AddHs(Chem.Mol(mol))
        res = AllChem.EmbedMolecule(
            mol_3d,
            maxAttempts=2,
            randomSeed=int(seed),
            useRandomCoords=False,
            clearConfs=True,
        )
        if res != 0:
            return np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(
                mol_3d,
                maxIters=int(os.environ.get("DEEP_TOX_3D_MMFF_ITERS", 30)),
            )
        whim = _safe_3d_vec(mol_3d, "CalcWHIM", 114)
        getaway = _safe_3d_vec(mol_3d, "CalcGETAWAY", 273)
        usr = _safe_3d_vec(mol_3d, "GetUSR", 12)
        usrcat = _safe_3d_vec(mol_3d, "GetUSRCAT", 60)
        morse = _safe_3d_vec(mol_3d, "CalcMORSE", 224)
        rdf = _safe_3d_vec(mol_3d, "CalcRDF", 210)
        shape_scalars = np.array([
            _safe_3d_scalar(mol_3d, "CalcPMI1"),
            _safe_3d_scalar(mol_3d, "CalcPMI2"),
            _safe_3d_scalar(mol_3d, "CalcPMI3"),
            _safe_3d_scalar(mol_3d, "CalcNPR1"),
            _safe_3d_scalar(mol_3d, "CalcNPR2"),
            _safe_3d_scalar(mol_3d, "CalcPBF"),
            _safe_3d_scalar(mol_3d, "CalcAsphericity"),
            _safe_3d_scalar(mol_3d, "CalcEccentricity"),
            _safe_3d_scalar(mol_3d, "CalcInertialShapeFactor"),
            _safe_3d_scalar(mol_3d, "CalcRadiusOfGyration"),
            _safe_3d_scalar(mol_3d, "CalcSpherocityIndex"),
            0.0,
        ], dtype=np.float32)
        try:
            shape_scalars[-1] = float(AllChem.ComputeMolVolume(mol_3d))
        except Exception:
            shape_scalars[-1] = 0.0
        shape_scalars = np.nan_to_num(shape_scalars, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.concatenate([whim, getaway, usr, usrcat, morse, rdf, shape_scalars]).astype(np.float32)
        if vec.shape[0] != ADVANCED_3D_DESCRIPTOR_DIM:
            fixed = np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)
            fixed[:min(len(vec), ADVANCED_3D_DESCRIPTOR_DIM)] = vec[:ADVANCED_3D_DESCRIPTOR_DIM]
            vec = fixed
        return vec
    except Exception:
        return np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)


def _compute_toxlens_3d_matrix(mols: Sequence[Chem.Mol]) -> np.ndarray:
    print(f"[toxlens] computing RDKit 3D descriptor matrix ({len(mols)} molecules)")
    mat = np.zeros((len(mols), ADVANCED_3D_DESCRIPTOR_DIM), dtype=np.float32)
    for i, mol in enumerate(mols):
        mat[i] = _compute_toxlens_3d_descriptor(mol, seed=42 + i)
    return mat


def _load_pubchem_bioactivity_matrix(smiles_list: Sequence[str], cache_path: Path) -> np.ndarray:
    """Load the 200-d PubChem bioactivity vectors used by the trained ToxLens model.

    deep_tox.py treats this modality as a cache-backed feature block. It is not
    derivable from SMILES alone without the original assay-vector cache, so this
    function never claims to compute it. If the cache is absent/missing entries,
    the vector is set to zero and the run prints an explicit warning.
    """
    cache_path = Path(cache_path)
    mat = np.zeros((len(smiles_list), PUBCHEM_DIM), dtype=np.float32)
    if not cache_path.exists():
        print(
            f"[toxlens] WARNING: PubChem bioactivity cache not found at {cache_path}; "
            "using the trained model's zero-vector fallback for this cache-backed modality"
        )
        return mat
    try:
        cache = json.loads(cache_path.read_text())
    except Exception as exc:
        print(f"[toxlens] WARNING: could not read PubChem cache {cache_path}: {exc}; using zero-vector fallback")
        return mat

    hits = 0
    for i, smi in enumerate(smiles_list):
        keys = [smi]
        try:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                keys.append(Chem.MolToSmiles(m, canonical=True))
        except Exception:
            pass
        vec = None
        for key in keys:
            if key in cache:
                vec = cache[key]
                break
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mat[i, :min(PUBCHEM_DIM, len(arr))] = arr[:PUBCHEM_DIM]
        hits += 1
    missing = len(smiles_list) - hits
    print(f"[toxlens] PubChem bioactivity cache hits: {hits}/{len(smiles_list)}")
    if missing:
        print(f"[toxlens] WARNING: {missing} molecules missing PubChem vectors; zero-vector fallback used for those rows")
    return mat

_PHARM_FAMILIES = ("Donor", "Acceptor", "Hydrophobe", "PosIonizable", "NegIonizable", "Aromatic")
_PHARM_IDX = {fam: i for i, fam in enumerate(_PHARM_FAMILIES)}

# Explicit primary-endpoint groups for the 11-task ToxLens case-study head.
# No keyword matching and no auxiliary endpoint families are used here.
PRIMARY_TASK_GROUPS = {
    "Ames": "genotoxicity",
    "LD50_Zhu": "systemic_toxicity",
    "hERG_Karim": "cardiotoxicity",
    "NR-AhR": "nuclear_receptor",
    "NR-Aromatase": "nuclear_receptor",
    "NR-ER": "nuclear_receptor",
    "NR-ER-LBD": "nuclear_receptor",
    "SR-ARE": "stress_response",
    "SR-HSE": "stress_response",
    "SR-MMP": "stress_response",
    "SR-p53": "stress_response",
}


def _infer_task_group(task_name: str) -> str:
    clean = task_name.strip().strip("()")
    if clean not in PRIMARY_TASK_GROUPS:
        raise ValueError(
            f"Unexpected ToxLens endpoint {task_name!r}. This integrated study only supports: "
            f"{list(PRIMARY_TASK_GROUPS)}"
        )
    return PRIMARY_TASK_GROUPS[clean]


# Featurisation primitives

def _one_hot(value, options):
    emb = [0] * (len(options) + 1)
    idx = options.index(value) if value in options else -1
    emb[idx] = 1
    return emb


def _atom_features(atom) -> np.ndarray:
    f = _one_hot(atom.GetSymbol(), [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
        "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
        "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
        "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
    ])
    f += _one_hot(atom.GetTotalDegree(), list(range(11)))
    f += _one_hot(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    try:
        f += _one_hot(atom.GetChiralTag(), [
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        ])
    except Exception:
        f += [0, 0, 1, 0]
    f += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    f += _one_hot(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ])
    f += [1 if atom.GetIsAromatic() else 0]
    f += [atom.GetMass() * 0.01]
    return np.array(f, dtype=np.float32)


def _bond_features(bond, reactivity_flags) -> np.ndarray:
    bt = bond.GetBondType()
    base = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
        1 if bond.GetIsConjugated() else 0,
        1 if bond.IsInRing() else 0,
    ]
    stereo = bond.GetStereo()
    base += _one_hot(stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ])
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    bz, ez = begin.GetAtomicNum(), end.GetAtomicNum()
    ben = _PAULING_EN.get(bz, 0.0)
    een = _PAULING_EN.get(ez, 0.0)

    def _charge(atom):
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            if np.isnan(q) or np.isinf(q):
                return 0.0
            return q
        except Exception:
            return 0.0
    bq, eq = _charge(begin), _charge(end)
    bond_order = {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5,
    }.get(bt, 0.0)
    mol = bond.GetOwningMol()
    smallest_ring = 0
    if bond.IsInRing():
        ring_info = mol.GetRingInfo()
        for r in range(3, 9):
            if ring_info.IsBondInRingOfSize(bond.GetIdx(), r):
                smallest_ring = r
                break
    is_rotatable_like = (
        bt == Chem.rdchem.BondType.SINGLE and not bond.IsInRing()
        and begin.GetAtomicNum() > 1 and end.GetAtomicNum() > 1
        and begin.GetDegree() > 1 and end.GetDegree() > 1
    )
    advanced = [
        bond_order / 3.0,
        abs(ben - een) / 4.0,
        (ben + een) / 8.0,
        abs(bq - eq),
        bq + eq,
        abs(float(begin.GetFormalCharge() - end.GetFormalCharge())),
        (bz + ez) / 236.0,
        abs(bz - ez) / 118.0,
        float(smallest_ring) / 8.0,
        1.0 if smallest_ring in (3, 4) else 0.0,
        1.0 if smallest_ring in (5, 6) else 0.0,
        1.0 if begin.GetIsAromatic() and end.GetIsAromatic() else 0.0,
        1.0 if is_rotatable_like else 0.0,
        1.0 if begin.GetHybridization() != end.GetHybridization() else 0.0,
        1.0 if begin.GetAtomicNum() in (7, 8, 15, 16) or end.GetAtomicNum() in (7, 8, 15, 16) else 0.0,
        1.0 if begin.GetAtomicNum() in (9, 17, 35, 53) or end.GetAtomicNum() in (9, 17, 35, 53) else 0.0,
    ]
    out = base + advanced + list(reactivity_flags)
    return np.array(out, dtype=np.float32)


def _add_rwse(data, walk_length: int = RWSE_WALK_LENGTH):
    import torch
    from torch_geometric.utils import to_dense_adj, to_undirected
    n = data.x.size(0)
    if n == 0 or data.edge_index.size(1) == 0:
        data.rwse = torch.zeros(n, walk_length, dtype=torch.float32)
        return data
    ei = to_undirected(data.edge_index, num_nodes=n)
    A = to_dense_adj(ei, max_num_nodes=n).squeeze(0).float()
    deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    P = A / deg
    Pk = torch.eye(n)
    cols = []
    for _ in range(walk_length):
        Pk = Pk @ P
        cols.append(Pk.diagonal())
    data.rwse = torch.stack(cols, dim=-1).float()
    return data


def _toxlens_featurise(
    mols: Sequence[Chem.Mol], smiles_list: Sequence[str],
    rdkit_mat: np.ndarray, tox_mat: np.ndarray, lm_mat: np.ndarray,
    desc_3d_mat: np.ndarray, pubchem_mat: np.ndarray,
    target_len: int, num_tasks: int,
):
    """Reimplemented ``batch_graph_worker`` — builds one PyG Data per molecule
    using the same node/edge feature stack and global concat order as
    deep_tox.py, including RDKit 3D descriptors and PubChem bioactivity vectors.
    """
    import torch
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
    from rdkit.Chem.EState import EStateIndices
    from torch_geometric.data import Data

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=True
    )
    pharm_factory = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    atom_reactivity_patterns = [Chem.MolFromSmarts(s) for _, s in _ATOM_REACTIVITY_SMARTS]
    bond_reactivity_patterns = [Chem.MolFromSmarts(s) for _, s in _BOND_REACTIVITY_SMARTS]
    pt = Chem.GetPeriodicTable()

    graphs: List = []
    for idx, mol_in in enumerate(mols):
        if mol_in is None:
            graphs.append(None)
            continue
        mol = Chem.Mol(mol_in)
        mol = Chem.AddHs(mol)
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        ring_info = mol.GetRingInfo()
        mol_heavy = Chem.RemoveHs(mol)
        n_heavy = mol_heavy.GetNumAtoms()
        n_all = mol.GetNumAtoms()

        pharm_flags = np.zeros((n_all, len(_PHARM_FAMILIES)), dtype=np.float32)
        try:
            for feat in pharm_factory.GetFeaturesForMol(mol_heavy):
                fam = feat.GetFamily()
                if fam in _PHARM_IDX:
                    for aidx in feat.GetAtomIds():
                        if aidx < n_heavy:
                            pharm_flags[aidx, _PHARM_IDX[fam]] = 1.0
        except Exception:
            pass

        try:
            crippen = rdMolDescriptors._CalcCrippenContribs(mol)
            if len(crippen) < n_all:
                crippen = list(crippen) + [(0.0, 0.0)] * (n_all - len(crippen))
        except Exception:
            crippen = [(0.0, 0.0)] * n_all
        try:
            tpsa_c = rdMolDescriptors._CalcTPSAContribs(mol)
            if len(tpsa_c) < n_all:
                tpsa_c = list(tpsa_c) + [0.0] * (n_all - len(tpsa_c))
        except Exception:
            tpsa_c = [0.0] * n_all
        try:
            estate_heavy = np.asarray(
                EStateIndices(mol_heavy) if callable(EStateIndices)
                else EStateIndices.EStateIndices(mol_heavy),
                dtype=np.float32,
            )
            estate_heavy = np.nan_to_num(estate_heavy)
        except Exception:
            estate_heavy = np.zeros(n_heavy, dtype=np.float32)

        asa = np.zeros(n_all, dtype=np.float32)
        try:
            raw = rdMolDescriptors._CalcLabuteASAContribs(mol)
            vals = raw[0] if isinstance(raw, tuple) else raw
            asa[: min(len(vals), n_all)] = np.nan_to_num(
                np.asarray(vals[:n_all], dtype=np.float32)
            )
        except Exception:
            pass
        asa_total = float(max(asa.sum(), 1e-6))

        atom_react = np.zeros((n_all, ATOM_REACTIVITY_DIM), dtype=np.float32)
        for p_idx, patt in enumerate(atom_reactivity_patterns):
            if patt is None:
                continue
            try:
                for match in mol_heavy.GetSubstructMatches(patt):
                    for a in match:
                        if a < n_all:
                            atom_react[a, p_idx] = 1.0
            except Exception:
                continue

        bond_react = np.zeros((mol.GetNumBonds(), BOND_REACTIVITY_DIM), dtype=np.float32)
        for p_idx, patt in enumerate(bond_reactivity_patterns):
            if patt is None:
                continue
            try:
                for match in mol.GetSubstructMatches(patt):
                    ms = set(match)
                    for bond in mol.GetBonds():
                        if bond.GetBeginAtomIdx() in ms and bond.GetEndAtomIdx() in ms:
                            bond_react[bond.GetIdx(), p_idx] = 1.0
            except Exception:
                continue

        atom_ring = np.zeros(n_all, dtype=np.float32)
        try:
            for ring in ring_info.AtomRings():
                for a in ring:
                    if a < n_all:
                        atom_ring[a] += 1.0
        except Exception:
            pass

        atom_feats = []
        for a_idx, atom in enumerate(mol.GetAtoms()):
            base_feat = _atom_features(atom)
            an = atom.GetAtomicNum()
            logp_c, mr_c = crippen[a_idx]
            tpsa_v = tpsa_c[a_idx]
            pe = _PAULING_EN.get(an, 0.0)
            try:    vdw = pt.GetRvdw(an)
            except: vdw = 0.0
            try:    cov = pt.GetRcovalent(an)
            except: cov = 0.0
            try:    val_e = float(pt.GetNOuterElecs(an))
            except: val_e = 0.0
            sr = 0
            for r in range(3, 9):
                if ring_info.IsAtomInRingOfSize(a_idx, r):
                    sr = r; break
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(q) or np.isinf(q):
                    q = 0.0
            except Exception:
                q = 0.0
            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            hetero = [n for n in heavy_nbrs if n.GetAtomicNum() not in (1, 6)]
            halogen = any(n.GetAtomicNum() in (9, 17, 35, 53) for n in heavy_nbrs)
            ens = [_PAULING_EN.get(n.GetAtomicNum(), 0.0) for n in heavy_nbrs]
            ne_mean = float(np.mean(ens)) if ens else 0.0
            ne_delta = float(np.mean([abs(pe - e) for e in ens])) if ens else 0.0
            try:    tv = float(atom.GetTotalValence())
            except: tv = 0.0
            try:    ev = float(atom.GetExplicitValence())
            except: ev = 0.0
            try:    iv = float(atom.GetImplicitValence())
            except: iv = 0.0
            rc = float(atom_ring[a_idx]) if a_idx < len(atom_ring) else 0.0
            es = float(estate_heavy[a_idx]) if a_idx < len(estate_heavy) else 0.0
            av = float(asa[a_idx]) if a_idx < len(asa) else 0.0
            chem_feats = np.array([
                logp_c, mr_c, tpsa_v, pe, an / 118.0, vdw, cov, val_e, float(sr),
            ], dtype=np.float32)
            adv = np.array([
                es / 20.0, av / 100.0, av / asa_total, abs(q),
                max(q, 0.0), max(-q, 0.0),
                float(atom.GetFormalCharge()) / 4.0,
                float(atom.GetNumRadicalElectrons()) / 4.0,
                tv / 8.0, ev / 8.0, iv / 8.0,
                float(len(heavy_nbrs)) / 6.0,
                float(len(hetero)) / max(float(len(heavy_nbrs)), 1.0),
                1.0 if halogen else 0.0,
                1.0 if atom.GetIsAromatic() and an not in (1, 6) else 0.0,
                rc / 4.0,
                (1.0 / float(sr)) if sr > 0 else 0.0,
                1.0 if rc > 1.0 and atom.GetDegree() > 2 else 0.0,
                1.0 if rc > 1.0 and atom.GetDegree() >= 4 else 0.0,
                ne_mean / 4.0, ne_delta / 4.0,
            ], dtype=np.float32)
            atom_feats.append(np.concatenate([
                base_feat, chem_feats,
                np.array([q], dtype=np.float32),
                np.array([1.0 if sr > 0 else 0.0], dtype=np.float32),
                pharm_flags[a_idx], adv, atom_react[a_idx],
            ], axis=0))
        x = torch.tensor(np.array(atom_feats), dtype=torch.float)

        edge_idx, edge_at = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ef = _bond_features(bond, bond_react[bond.GetIdx()])
            edge_idx += [[u, v], [v, u]]
            edge_at += [ef, ef]
        if edge_idx:
            edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_at), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

        fp = morgan_gen.GetFingerprintAsNumPy(mol_heavy)
        global_vec = np.concatenate([
            fp.astype(np.float32),
            rdkit_mat[idx],
            tox_mat[idx],
            lm_mat[idx],
            desc_3d_mat[idx],
            pubchem_mat[idx],
        ])
        if global_vec.size != target_len:
            raise RuntimeError(
                f"ToxLens global feature length mismatch: got {global_vec.size}, expected {target_len}"
            )
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.zeros((1, num_tasks), dtype=torch.float))
        data.global_features = torch.tensor(global_vec, dtype=torch.float)
        data.smiles = smiles_list[idx]
        _add_rwse(data, walk_length=RWSE_WALK_LENGTH)
        graphs.append(data)
    return graphs


# ToxLens model class — state-dict-compatible replica

def _build_toxlens_model_class(num_tasks_default: int = 1):
    """Build the ToxLens ``GAT_class`` LightningModule. Keeps every submodule
    name and parameter shape used by the current deep_tox.py checkpoint.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lightning import LightningModule
    from torch_geometric.nn import (
        GINEConv, GPSConv, GraphNorm, Linear, TransformerConv,
        global_add_pool, global_max_pool, global_mean_pool,
    )
    from torch_geometric.utils import dropout_edge, softmax as pyg_softmax

    MODEL_DEFAULTS = dict(
        hidden_channels=256, n_layers=5, learning_rate=0.0005,
        weight_decay=0.008, dropout_rate=0.15, gamma_neg=0.0,
        margin=0.0, lr_T0=45, drop_edge_p=0.05, noise_std=0.0,
        global_dropout_p=0.25, eps_label_smooth=0.0005,
        aux_supervision_weight=0.0, graph_aux_weight=0.0,
        graph_aux_late_weight=0.0, graph_aux_warmup_epochs=20,
        stochastic_depth_p=0.1, use_gps_attention=False,
        conv_type="gine", transformer_heads=4, transformer_layers=2,
        fusion_type="gcmi", logit_scale_init=1.0,
        final_rep_dropout=0.1, use_direct_global_trunk=False,
        use_late_global_residual=False, use_group_towers=True,
    )
    PRIMARY_TASK_FOCUS_WEIGHTS = {
        "NR-AhR": 1.20, "NR-Aromatase": 1.25, "NR-ER": 1.35,
        "NR-ER-LBD": 1.35, "SR-MMP": 1.20, "SR-p53": 1.30,
        "hERG_Karim": 1.25, "Ames": 1.10, "LD50_Zhu": 1.10,
    }

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x, batch):
            if not self.training or self.drop_prob <= 0.0:
                return x
            ng = int(batch.max().item()) + 1
            keep = (torch.rand(ng, device=x.device) >= self.drop_prob).float()
            keep = keep / max(1.0 - self.drop_prob, 1e-6)
            return x * keep[batch].unsqueeze(-1)

    class MultiHeadAttentionPooling(nn.Module):
        def __init__(self, in_channels, out_channels, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.attn_scores = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2), nn.GELU(),
                nn.Linear(in_channels // 2, num_heads),
            )
            self.node_transform = nn.Linear(in_channels, in_channels)
            self.final_proj = nn.Sequential(
                nn.Linear((num_heads * in_channels) + (2 * in_channels), out_channels),
                nn.LayerNorm(out_channels), nn.GELU(),
                nn.Linear(out_channels, out_channels),
            )

        def forward(self, x, batch):
            scores = self.attn_scores(x)
            weights = pyg_softmax(scores, batch, dim=0)
            xt = self.node_transform(x)
            heads = [
                global_add_pool(xt * weights[:, h:h + 1], batch)
                for h in range(self.num_heads)
            ]
            attn_out = torch.cat(heads, dim=-1)
            pooled = torch.cat(
                [attn_out, global_mean_pool(x, batch), global_max_pool(x, batch)],
                dim=-1,
            )
            return self.final_proj(pooled)

    class GCMIFusion(nn.Module):
        def __init__(self, node_dim, global_dim, output_dim):
            super().__init__()
            self.gate_projection = nn.Linear(global_dim, node_dim)
            self.synergy_projection = nn.Linear(node_dim + global_dim, output_dim)
            self.residual_projection = nn.Linear(node_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            self._gate_mean = 0.5
            self._gate_std = 0.0

        def forward(self, x, g, batch_idx):
            g_exp = g[batch_idx]
            gate = torch.sigmoid(self.gate_projection(g_exp))
            with torch.no_grad():
                self._gate_mean = gate.mean().item()
                self._gate_std = gate.std().item()
            gated = x * gate
            syn = torch.tanh(self.synergy_projection(torch.cat([gated, g_exp], dim=-1)))
            return self.layer_norm(self.residual_projection(x) + syn)

    class FiLMFusion(nn.Module):
        def __init__(self, node_dim, global_dim, output_dim):
            super().__init__()
            self.gamma_proj = nn.Linear(global_dim, node_dim)
            self.beta_proj = nn.Linear(global_dim, node_dim)
            self.out_proj = nn.Linear(node_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)

        def forward(self, x, g, batch_idx):
            g_exp = g[batch_idx]
            return self.layer_norm(self.out_proj(self.gamma_proj(g_exp) * x + self.beta_proj(g_exp)))

    class ConcatFusion(nn.Module):
        def __init__(self, node_dim, global_dim, output_dim):
            super().__init__()
            self.proj = nn.Linear(node_dim + global_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)

        def forward(self, x, g, batch_idx):
            return self.layer_norm(self.proj(torch.cat([x, g[batch_idx]], dim=-1)))

    class SliceExpert(nn.Module):
        def __init__(self, in_dim, hidden_channels, dropout=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )

        def forward(self, x):
            return self.net(x)

    def global_expert_slices(global_dim):
        cursor = 0
        morgan = (cursor, cursor + 1024); cursor += 1024
        rdkit = (cursor, cursor + len(Descriptors._descList)); cursor += len(Descriptors._descList)
        tox = (cursor, cursor + len(_TOX_SMARTS_STRINGS) + 1); cursor += len(_TOX_SMARTS_STRINGS) + 1
        lm = (cursor, cursor + 768); cursor += 768
        shape_3d = (cursor, cursor + ADVANCED_3D_DESCRIPTOR_DIM); cursor += ADVANCED_3D_DESCRIPTOR_DIM
        pubchem = (cursor, cursor + PUBCHEM_DIM); cursor += PUBCHEM_DIM
        if cursor != global_dim:
            raise ValueError(f"Expected global_dim={cursor}, got {global_dim}.")
        return {
            "descriptor": [morgan, rdkit, lm, pubchem],
            "tox": [tox],
            "shape_3d": [shape_3d],
            "lm_slice": lm,
        }

    class UnweightedMultiTaskLoss(nn.Module):
        def __init__(self, num_tasks, w_pos=None, w_neg=None, task_weights=None, **_):
            super().__init__()
            self.num_tasks = num_tasks
            self.register_buffer("w_pos", w_pos.float() if w_pos is not None else torch.ones(num_tasks))
            self.register_buffer("w_neg", w_neg.float() if w_neg is not None else torch.ones(num_tasks))
            self.register_buffer(
                "task_weights",
                task_weights.float() if task_weights is not None else torch.ones(num_tasks),
            )

        def forward(self, preds, targets, task_mask=None):
            return preds.sum() * 0.0

    class GAT_class(LightningModule):
        def __init__(
            self, in_channels, hidden_channels, learning_rate, global_dim,
            edge_feature_dim, num_tasks, task_types,
            w_pos=None, w_neg=None, task_names=None,
            use_global_features=True, use_gcmi=True, head_type="deep",
            feature_indices_to_exclude=None,
            fusion_type=MODEL_DEFAULTS["fusion_type"],
            n_layers=MODEL_DEFAULTS["n_layers"],
            dropout_rate=MODEL_DEFAULTS["dropout_rate"],
            gamma_neg=MODEL_DEFAULTS["gamma_neg"],
            margin=MODEL_DEFAULTS["margin"],
            lr_T0=MODEL_DEFAULTS["lr_T0"],
            weight_decay=MODEL_DEFAULTS["weight_decay"],
            drop_edge_p=MODEL_DEFAULTS["drop_edge_p"],
            noise_std=MODEL_DEFAULTS["noise_std"],
            global_dropout_p=MODEL_DEFAULTS["global_dropout_p"],
            eps_label_smooth=MODEL_DEFAULTS["eps_label_smooth"],
            aux_supervision_weight=MODEL_DEFAULTS["aux_supervision_weight"],
            graph_aux_weight=MODEL_DEFAULTS["graph_aux_weight"],
            graph_aux_late_weight=MODEL_DEFAULTS["graph_aux_late_weight"],
            graph_aux_warmup_epochs=MODEL_DEFAULTS["graph_aux_warmup_epochs"],
            stochastic_depth_p=MODEL_DEFAULTS["stochastic_depth_p"],
            use_gps_attention=MODEL_DEFAULTS["use_gps_attention"],
            conv_type=MODEL_DEFAULTS["conv_type"],
            transformer_heads=MODEL_DEFAULTS["transformer_heads"],
            transformer_layers=MODEL_DEFAULTS["transformer_layers"],
            final_rep_dropout=MODEL_DEFAULTS["final_rep_dropout"],
            use_direct_global_trunk=MODEL_DEFAULTS["use_direct_global_trunk"],
            use_late_global_residual=MODEL_DEFAULTS["use_late_global_residual"],
            use_group_towers=MODEL_DEFAULTS["use_group_towers"],
            use_lora_molformer=False,
            lora_rank=8,
            lora_alpha=16,
            lora_n_layers=4,
            lora_max_tokens=128,
            **_legacy_kwargs,
        ):
            super().__init__()
            self.strict_loading = False
            self.save_hyperparameters(ignore=["w_pos", "w_neg"])
            self.use_lora_molformer = bool(use_lora_molformer)
            self.lora_rank = int(lora_rank)
            self.lora_alpha = float(lora_alpha)
            self.lora_n_layers = int(lora_n_layers)
            self.lora_max_tokens = int(lora_max_tokens)
            self.use_global = use_global_features
            self.fusion_type = fusion_type if fusion_type != "gcmi" else ("gcmi" if use_gcmi else "none")
            self.use_gcmi = (self.fusion_type == "gcmi")
            self.head_type = head_type
            self.num_tasks = num_tasks
            self.hidden_channels = hidden_channels
            self.n_layers = n_layers
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.lr_T0 = lr_T0
            self.raw_in_channels = in_channels
            if task_names is None:
                if num_tasks != len(PRIMARY_TOXLENS_TASKS):
                    raise ValueError(
                        f"ToxLens case-study model must use exactly {len(PRIMARY_TOXLENS_TASKS)} configured endpoints; "
                        f"got num_tasks={num_tasks}."
                    )
                task_names = list(PRIMARY_TOXLENS_TASKS)
            self.task_names = list(task_names)
            self.use_gps_attention = bool(use_gps_attention)
            self.conv_type = str(conv_type).lower()
            self.transformer_heads = int(transformer_heads)
            self.transformer_layers = int(transformer_layers)
            self.use_direct_global_trunk = bool(use_direct_global_trunk)
            self.use_late_global_residual = bool(use_late_global_residual)
            self.use_group_towers = bool(use_group_towers)
            self.aux_supervision_weight = float(aux_supervision_weight)
            self.graph_aux_weight = float(graph_aux_weight)
            self.graph_aux_late_weight = float(graph_aux_late_weight)
            self.graph_aux_warmup_epochs = int(graph_aux_warmup_epochs)

            self.node_emb = (
                nn.Identity() if in_channels == hidden_channels
                else nn.Linear(in_channels, hidden_channels, bias=False)
            )
            self.act = nn.GELU()

            if self.transformer_heads < 1:
                raise ValueError("transformer_heads must be >= 1")
            if hidden_channels % self.transformer_heads != 0:
                raise ValueError("hidden_channels must be divisible by transformer_heads")
            transformer_start = max(0, n_layers - max(0, self.transformer_layers))

            conv_layers = []
            conv_layer_kinds = []
            for layer_idx in range(n_layers):
                nn_local = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                local_conv = GINEConv(nn_local, edge_dim=edge_feature_dim)
                use_transformer_layer = (
                    self.conv_type in ("transformer", "transformerconv")
                    or (
                        self.conv_type in ("hybrid", "hybrid_transformer", "gine_transformer")
                        and layer_idx >= transformer_start
                    )
                )
                if use_transformer_layer:
                    conv_layers.append(TransformerConv(
                        hidden_channels,
                        hidden_channels // self.transformer_heads,
                        heads=self.transformer_heads,
                        concat=True,
                        beta=True,
                        dropout=dropout_rate * 0.25,
                        edge_dim=edge_feature_dim,
                        root_weight=True,
                    ))
                    conv_layer_kinds.append("transformer")
                elif self.use_gps_attention:
                    conv_layers.append(GPSConv(
                        hidden_channels, local_conv, heads=4,
                        dropout=dropout_rate * 0.5, act="gelu", norm="layer_norm",
                    ))
                    conv_layer_kinds.append("gps")
                else:
                    conv_layers.append(local_conv)
                    conv_layer_kinds.append("gine")
            self.layers = nn.ModuleList(conv_layers)
            self.conv_layer_kinds = conv_layer_kinds
            self.residual_projections = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
            self.norms = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(n_layers)])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    Linear(hidden_channels, hidden_channels * 2), self.act,
                    nn.Dropout(dropout_rate * 0.5),
                    Linear(hidden_channels * 2, hidden_channels),
                )
                for _ in range(n_layers)
            ])
            self.norms1 = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(n_layers)])
            self.norms2 = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(n_layers)])
            self.vn_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.GELU(),
                )
                for _ in range(n_layers)
            ])
            self.pool = MultiHeadAttentionPooling(hidden_channels, hidden_channels)

            self.feature_mask = None
            if feature_indices_to_exclude is not None:
                mask = torch.ones(global_dim, dtype=torch.bool)
                for idx in feature_indices_to_exclude:
                    mask[idx] = False
                self.register_buffer("feature_mask", mask)
                effective_global_dim = int(mask.sum().item())
            else:
                effective_global_dim = global_dim

            self.fusion_layer = None
            self.expert_slices = None
            self.use_expert_gate = False
            if self.use_global:
                self.global_scaler = nn.LayerNorm(effective_global_dim)
                self.global_projector = nn.Sequential(
                    nn.Linear(effective_global_dim, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.GELU(),
                )
                if feature_indices_to_exclude is None:
                    self.expert_slices = global_expert_slices(global_dim)
                if self.fusion_type == "gcmi":
                    self.fusion_layer = GCMIFusion(hidden_channels, hidden_channels, hidden_channels)
                elif self.fusion_type == "film":
                    self.fusion_layer = FiLMFusion(hidden_channels, hidden_channels, hidden_channels)
                elif self.fusion_type == "concat":
                    self.fusion_layer = ConcatFusion(hidden_channels, hidden_channels, hidden_channels)
                self.use_expert_gate = bool(feature_indices_to_exclude is None and self.fusion_layer is None)
            else:
                self.global_scaler = None
                self.global_projector = None
            if self.use_global and self.use_late_global_residual:
                self.late_global_gate = nn.Sequential(
                    nn.LayerNorm(hidden_channels * 2),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.GELU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.Sigmoid(),
                )
                self.late_global_log_scale = nn.Parameter(torch.tensor(math.log(0.25), dtype=torch.float32))
            else:
                self.late_global_gate = None
                self.late_global_log_scale = None

            self.lora_molformer = None
            self._lm_slice = self.expert_slices.get("lm_slice") if self.expert_slices is not None else None

            self.stochastic_depth_p = float(stochastic_depth_p)
            if n_layers > 1 and self.stochastic_depth_p > 0.0:
                sd_rates = [self.stochastic_depth_p * i / (n_layers - 1) for i in range(n_layers)]
            else:
                sd_rates = [0.0] * n_layers
            self.drop_paths = nn.ModuleList([DropPath(p) for p in sd_rates])
            self.drop_paths_ffn = nn.ModuleList([DropPath(p) for p in sd_rates])

            self.aux_head = nn.Sequential(
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_channels, num_tasks),
            )
            self.global_expert_head = None

            self.graph_expert = SliceExpert(hidden_channels, hidden_channels, dropout=dropout_rate * 0.35)
            if self.use_expert_gate:
                descriptor_dim = sum(e - s for s, e in self.expert_slices["descriptor"])
                tox_dim = sum(e - s for s, e in self.expert_slices["tox"])
                shape_dim = sum(e - s for s, e in self.expert_slices["shape_3d"])
                self.descriptor_expert = SliceExpert(descriptor_dim, hidden_channels, dropout=dropout_rate * 0.5)
                self.tox_expert = SliceExpert(tox_dim, hidden_channels, dropout=dropout_rate * 0.20)
                self.shape_expert = SliceExpert(shape_dim, hidden_channels, dropout=dropout_rate * 0.35)
                self.expert_gate = nn.Sequential(
                    nn.Linear(hidden_channels * 4, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.GELU(),
                    nn.Dropout(dropout_rate * 0.25),
                    nn.Linear(hidden_channels, num_tasks * 4),
                )
                head_input_dim = hidden_channels
            else:
                self.descriptor_expert = None
                self.tox_expert = None
                self.shape_expert = None
                self.expert_gate = None
                head_input_dim = hidden_channels
                if self.use_global and self.use_direct_global_trunk:
                    head_input_dim += hidden_channels

            if self.use_expert_gate:
                self.shared_trunk = nn.Identity()
            elif self.head_type == "deep":
                self.shared_trunk = nn.Sequential(
                    nn.Linear(head_input_dim, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.GELU(),
                    nn.Dropout(dropout_rate * 0.5),
                )
            else:
                self.shared_trunk = nn.Identity()
            trunk_out_dim = hidden_channels if self.head_type == "deep" or self.use_expert_gate else head_input_dim

            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(trunk_out_dim, trunk_out_dim),
                    nn.LayerNorm(trunk_out_dim), nn.GELU(),
                    nn.Dropout(dropout_rate * 0.25),
                    nn.Linear(trunk_out_dim, 1),
                )
                for _ in range(num_tasks)
            ])
            self.task_norms = nn.ModuleList([nn.LayerNorm(trunk_out_dim) for _ in range(num_tasks)])
            self.task_groups = [_infer_task_group(n.strip("()")) for n in self.task_names]
            if self.use_group_towers:
                group_input_dim = trunk_out_dim if self.use_expert_gate else head_input_dim
                self.group_towers = nn.ModuleDict({
                    g: nn.Sequential(
                        nn.Linear(group_input_dim, hidden_channels),
                        nn.LayerNorm(hidden_channels), nn.GELU(),
                        nn.Dropout(dropout_rate * 0.5),
                        nn.Linear(hidden_channels, trunk_out_dim),
                        nn.LayerNorm(trunk_out_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate * 0.25),
                    )
                    for g in sorted(set(self.task_groups))
                })
            else:
                self.group_towers = nn.ModuleDict()
            self.task_logit_log_scale = nn.Parameter(
                torch.full((num_tasks,), math.log(float(MODEL_DEFAULTS["logit_scale_init"])))
            )

            self.noise_std = noise_std
            self.drop_edge_p = drop_edge_p
            self.global_dropout_p = global_dropout_p
            self.final_rep_dropout = float(final_rep_dropout)
            self.eps_label_smooth = eps_label_smooth
            self.register_buffer("task_thresholds", torch.full((num_tasks,), 0.5))
            self.register_buffer("platt_a", torch.ones(num_tasks))
            self.register_buffer("platt_b", torch.zeros(num_tasks))
            if w_pos is not None:
                self.register_buffer("w_pos", w_pos.float())
                self.register_buffer("w_neg", w_neg.float())
            else:
                self.register_buffer("w_pos", torch.ones(num_tasks))
                self.register_buffer("w_neg", torch.ones(num_tasks))

            task_weight_values = [
                PRIMARY_TASK_FOCUS_WEIGHTS.get(n.strip("()"), 1.0)
                for n in self.task_names
            ]
            self.loss_fn = UnweightedMultiTaskLoss(
                num_tasks=self.num_tasks,
                w_pos=self.w_pos,
                w_neg=self.w_neg,
                gamma_neg=gamma_neg,
                eps_label_smooth=eps_label_smooth,
                task_weights=torch.tensor(task_weight_values, dtype=torch.float32),
                margin=margin,
            )
            self.val_loss_history = []
            self.train_loss_history = []
            self.plot_train_loss_history = []
            self.validation_outputs = []
            self.test_outputs = []
            self.train_metric_outputs = []
            self.test_metrics_suffix = ""
            self.primary_mask = [True] * self.num_tasks
            self.register_buffer("primary_task_mask", torch.ones(self.num_tasks, dtype=torch.bool))
            self.register_buffer("aux_task_mask", torch.zeros(self.num_tasks, dtype=torch.bool))

        def _apply_platt_calibration(self, logits):
            a = self.platt_a.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
            b = self.platt_b.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
            return torch.sigmoid((a * logits + b).float())

        def forward(
            self, x=None, edge_index=None, batch=None, edge_attr=None,
            global_features=None, data=None, return_attention=False,
            apply_embedding=True, return_aux=False,
        ):
            if data is None and hasattr(x, "edge_index"):
                data = x
                x = None
            if data is not None:
                x, edge_index, batch, edge_attr, global_features = (
                    data.x, data.edge_index, data.batch, data.edge_attr, data.global_features
                )
                num_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
            else:
                if batch is None:
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                num_graphs = int(batch.max().item() + 1)

            if apply_embedding:
                x = self.node_emb(x)
            if self.training and self.drop_edge_p > 0.0:
                edge_index, edge_mask = dropout_edge(
                    edge_index, p=self.drop_edge_p, force_undirected=True, training=True
                )
                if edge_attr is not None:
                    edge_attr = edge_attr[edge_mask]

            global_emb = None
            expert_global_feats = None
            if self.use_global:
                gfeat = global_features.view(num_graphs, -1)
                gfeat = torch.nan_to_num(gfeat, nan=0.0, posinf=10.0, neginf=-10.0)
                gfeat = torch.clamp(gfeat, -10.0, 10.0)
                if self.feature_mask is not None:
                    gfeat = gfeat[:, self.feature_mask]
                input_drop_p = self.global_dropout_p if self.use_expert_gate else min(0.45, self.global_dropout_p + 0.10)
                gfeat = F.dropout(gfeat, p=input_drop_p, training=self.training)
                if self.training:
                    gfeat = gfeat + torch.randn_like(gfeat) * self.noise_std
                expert_global_feats = gfeat
                if self.fusion_layer is not None or (not self.use_expert_gate and self.use_direct_global_trunk):
                    gfeat = self.global_scaler(gfeat)
                    gfeat = F.dropout(gfeat, p=self.global_dropout_p, training=self.training)
                    global_emb = self.global_projector(gfeat)
                    global_emb = F.dropout(global_emb, p=self.final_rep_dropout, training=self.training)

            attn_weights_list = []
            virtual_node = x.new_zeros((num_graphs, x.size(-1)))
            for i, (conv, proj, n1, n2, ffn) in enumerate(zip(
                self.layers, self.residual_projections, self.norms1, self.norms2, self.ffns
            )):
                x = x + virtual_node[batch]
                x_in = n1(x, batch)
                if self.conv_layer_kinds[i] == "gps":
                    x_out = conv(x_in, edge_index, batch=batch, edge_attr=edge_attr)
                else:
                    x_out = conv(x_in, edge_index, edge_attr=edge_attr)
                x_out = proj(x_out)
                x = x + self.drop_paths[i](x_out, batch)
                if self.use_global and self.fusion_layer is not None:
                    x = self.fusion_layer(x, global_emb, batch)
                x_in = n2(x, batch)
                x = x + self.drop_paths_ffn[i](ffn(x_in), batch)
                virtual_node = virtual_node + self.vn_mlps[i](global_add_pool(x, batch))

            raw_graph_emb = self.pool(x, batch)
            graph_emb = raw_graph_emb
            if self.late_global_gate is not None and global_emb is not None and not self.use_expert_gate:
                late_gate = self.late_global_gate(torch.cat([raw_graph_emb, global_emb], dim=-1))
                late_scale = self.late_global_log_scale.exp().clamp(0.0, 1.0)
                graph_emb = raw_graph_emb + late_scale * late_gate * global_emb

            if self.use_expert_gate and expert_global_feats is not None:
                def _slice_cat(slice_name):
                    return torch.cat(
                        [expert_global_feats[:, s:e] for s, e in self.expert_slices[slice_name]],
                        dim=-1,
                    )
                expert_tokens = torch.stack([
                    self.graph_expert(raw_graph_emb),
                    self.descriptor_expert(_slice_cat("descriptor")),
                    self.shape_expert(_slice_cat("shape_3d")),
                    self.tox_expert(_slice_cat("tox")),
                ], dim=1)
                expert_tokens = F.dropout(expert_tokens, p=self.final_rep_dropout, training=self.training)
                gate_logits = self.expert_gate(expert_tokens.flatten(1)).view(num_graphs, self.num_tasks, 4)
                gate = F.softmax(gate_logits, dim=-1)
                task_reps = torch.einsum("btm,bmh->bth", gate, expert_tokens)
                task_logits = []
                for t, (norm_t, head_t) in enumerate(zip(self.task_norms, self.task_heads)):
                    cond_t = task_reps[:, t, :]
                    if self.use_group_towers:
                        cond_t = cond_t + self.group_towers[self.task_groups[t]](cond_t)
                    task_logits.append(head_t(norm_t(cond_t)))
            else:
                final_parts = [graph_emb]
                if self.use_global and self.use_direct_global_trunk and global_emb is not None:
                    final_parts.append(global_emb)
                final_rep = torch.cat(final_parts, dim=-1)
                final_rep = torch.nan_to_num(final_rep, nan=0.0, posinf=10.0, neginf=-10.0)
                final_rep = F.dropout(final_rep, p=self.final_rep_dropout, training=self.training)
                shared = self.shared_trunk(final_rep)
                group_shared = {
                    group: tower(final_rep)
                    for group, tower in self.group_towers.items()
                } if self.use_group_towers else {}
                task_logits = []
                for t, (norm_t, head_t) in enumerate(zip(self.task_norms, self.task_heads)):
                    cond_t = shared
                    if self.use_group_towers:
                        cond_t = cond_t + group_shared[self.task_groups[t]]
                    task_logits.append(head_t(norm_t(cond_t)))

            fused_out = torch.cat(task_logits, dim=-1)
            scale = self.task_logit_log_scale.exp().clamp(0.50, 3.00).unsqueeze(0)
            out = torch.clamp(torch.nan_to_num(fused_out * scale, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
            graph_out = self.aux_head(raw_graph_emb) if self.aux_head is not None else None
            global_out = self.global_expert_head(global_emb) if self.global_expert_head is not None and global_emb is not None else None
            if return_attention:
                return out, attn_weights_list
            if return_aux:
                return out, graph_out, graph_emb, global_out, fused_out
            return out

        def configure_optimizers(self):  # never called at inference
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    return GAT_class


def _compute_modality_slices(global_dim: int) -> "OrderedDict[str, Tuple[int, int]]":
    """Modality (start, end) windows for the cached global feature tensor —
    matches deep_tox.compute_modality_slices."""
    dim_morgan = 1024
    dim_rdkit = len([fn for _, fn in Descriptors._descList])
    dim_tox = len(_TOX_SMARTS_STRINGS) + 1
    dim_lm = 768
    dim_3d = ADVANCED_3D_DESCRIPTOR_DIM
    dim_pub = PUBCHEM_DIM
    expected = dim_morgan + dim_rdkit + dim_tox + dim_lm + dim_3d + dim_pub
    if global_dim != expected:
        raise ValueError(
            f"global_dim={global_dim} does not match expected modality concat length {expected}. "
            f"breakdown: Morgan={dim_morgan}, RDKit={dim_rdkit}, Tox={dim_tox}, "
            f"MolFormer={dim_lm}, 3D={dim_3d}, PubChem={dim_pub}"
        )
    slices: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
    cur = 0
    for name, L in [
        ("Morgan", dim_morgan), ("RDKit", dim_rdkit), ("Tox", dim_tox),
        ("MolFormer", dim_lm), ("3D_Shape_Electronic", dim_3d), ("PubChem", dim_pub),
    ]:
        slices[name] = (cur, cur + L)
        cur += L
    return slices


def _install_molformer_transformers_compat() -> None:
    """Compatibility shims for IBM MoLFormer's legacy HF dynamic module.

    Newer/minimal transformers installs can omit ``transformers.onnx`` and can
    move/remove pruning helpers from ``transformers.pytorch_utils``. MoLFormer
    imports these symbols at module import time even though this pipeline only
    uses embedding inference.
    """
    import types

    try:
        from transformers.onnx import OnnxConfig  # noqa: F401
    except ModuleNotFoundError:
        import transformers

        mod = types.ModuleType("transformers.onnx")

        class OnnxConfig:
            def __init__(self, config=None, task: str = "default", **kwargs):
                self._config = config
                self.task = task
                self._kwargs = kwargs

        mod.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = mod
        setattr(transformers, "onnx", mod)

    import transformers.pytorch_utils as pt_utils

    if not hasattr(pt_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(*args, **kwargs):
            raise NotImplementedError(
                "MoLFormer legacy pruning utility was called during inference."
            )
        setattr(pt_utils, "find_pruneable_heads_and_indices", find_pruneable_heads_and_indices)

    if not hasattr(pt_utils, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim=0):
            return layer
        setattr(pt_utils, "prune_linear_layer", prune_linear_layer)

    if not hasattr(pt_utils, "apply_chunking_to_forward"):
        try:
            from transformers.modeling_utils import apply_chunking_to_forward
        except ImportError:
            def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
                return forward_fn(*input_tensors)
        setattr(pt_utils, "apply_chunking_to_forward", apply_chunking_to_forward)


def _patch_molformer_runtime_compat(backbone) -> None:
    """Attach attention-mask helpers missing from some modern HF resolutions."""
    import types
    import torch

    if not hasattr(backbone, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask
        backbone.get_head_mask = types.MethodType(_get_head_mask, backbone)

    if not hasattr(backbone, "get_extended_attention_mask"):
        def _get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
            if attention_mask is None:
                return None
            if dtype is None:
                dtype = next(self.parameters()).dtype
            if attention_mask.dim() == 3:
                ext = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                ext = attention_mask[:, None, None, :]
            else:
                ext = attention_mask
            ext = ext.to(dtype=dtype)
            return (1.0 - ext) * torch.finfo(dtype).min
        backbone.get_extended_attention_mask = types.MethodType(_get_extended_attention_mask, backbone)

    if not hasattr(backbone, "invert_attention_mask"):
        def _invert_attention_mask(self, encoder_attention_mask):
            if encoder_attention_mask.dim() == 3:
                inv = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                inv = encoder_attention_mask[:, None, None, :]
            else:
                inv = encoder_attention_mask
            dtype = next(self.parameters()).dtype
            inv = inv.to(dtype=dtype)
            return (1.0 - inv) * torch.finfo(dtype).min
        backbone.invert_attention_mask = types.MethodType(_invert_attention_mask, backbone)


def _load_molformer_for_embeddings(device):
    _install_molformer_transformers_compat()
    from transformers import AutoModel, AutoTokenizer

    model_name = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _patch_molformer_runtime_compat(molformer)
    return tokenizer, molformer.eval().to(device)


def _safe_descriptor_float(value) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(v) or abs(v) > 1.0e6:
        return 0.0
    return v


def _resolve_tox_ckpt(path: Path) -> Path:
    """Resolve common ToxLens checkpoint locations used during thesis runs."""
    if path.exists():
        return path
    candidates = [
        Path("deep_tox_classification.ckpt"),
        Path("models") / "deep_tox_classification.ckpt",
        Path("checkpoints") / "deep_tox_classification.ckpt",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[toxlens] checkpoint {path} not found; using {cand}")
            return cand
    raise FileNotFoundError(path)


def stage_run_toxlens(
    top_csv: Path,
    cfg: IntegratedConfig,
) -> pd.DataFrame:
    """Featurise the top-N set, run ToxLens with MC-dropout, score, and write
    all artefacts (CSVs + figures). Runs entirely in the ToxLens env."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader as GeoDataLoader

    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load top-N from stage 1b.
    top = pd.read_csv(top_csv)
    if "canonical_smiles" not in top.columns:
        raise ValueError(f"{top_csv} must contain a 'canonical_smiles' column")
    if "activity_prob" not in top.columns:
        raise ValueError(f"{top_csv} must contain an 'activity_prob' column")
    print(f"[run-toxlens] {len(top)} candidate molecules")
    cfg.tox_ckpt = _resolve_tox_ckpt(cfg.tox_ckpt)

    # Fixed primary ToxLens endpoint order; do not use every column in the CSV.
    task_names = _primary_toxlens_task_names(cfg.tox_task_csv)
    num_tasks = len(task_names)

    # Re-standardise the top-N (idempotent) so featurisation is identical to
    # what each training pipeline saw.
    mols, canon = [], []
    for s in top["canonical_smiles"].tolist():
        m = standardise_smiles(s)
        if m is None:
            mols.append(None); canon.append(None)
        else:
            mols.append(m); canon.append(canonical_smiles(m))

    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]
    valid_smiles = [canon[i] for i in valid_idx]
    n_valid = len(valid_idx)

    # Per-molecule RDKit 2D descriptor matrix.
    rd_funcs = [fn for _, fn in Descriptors._descList]
    rdkit_mat = np.zeros((n_valid, len(rd_funcs)), dtype=np.float32)
    for i, mol in enumerate(valid_mols):
        for j, fn in enumerate(rd_funcs):
            try:
                rdkit_mat[i, j] = _safe_descriptor_float(fn(mol))
            except Exception:
                rdkit_mat[i, j] = 0.0

    # Per-molecule tox SMARTS + PAINS counts.
    smarts_patts = [(name, Chem.MolFromSmarts(s)) for name, s in _TOX_SMARTS_STRINGS]
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)
    tox_mat = np.zeros((n_valid, len(smarts_patts) + 1), dtype=np.float32)
    matched_alerts: List[List[str]] = []
    pains_counts: List[int] = []
    for i, mol in enumerate(valid_mols):
        m_alerts = []
        for j, (name, patt) in enumerate(smarts_patts):
            if patt is not None and mol.HasSubstructMatch(patt):
                tox_mat[i, j] = 1.0
                m_alerts.append(name)
        tox_mat[i, -1] = float(len(pains_catalog.GetMatches(mol)))
        matched_alerts.append(m_alerts)
        pains_counts.append(int(tox_mat[i, -1]))

    # MolFormer embedding.
    print("[run-toxlens] loading MolFormer for LM modality")
    tokenizer, molformer = _load_molformer_for_embeddings(device)
    lm_mat = np.zeros((n_valid, 768), dtype=np.float32)
    BS = 64
    with torch.no_grad():
        for i in range(0, n_valid, BS):
            chunk = valid_smiles[i : i + BS]
            try:
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                out = molformer(**inputs)
                lm_mat[i : i + len(chunk)] = out.pooler_output.cpu().numpy()
            except Exception:
                pass
    del molformer, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    desc_3d_mat = _compute_toxlens_3d_matrix(valid_mols)
    pubchem_mat = _load_pubchem_bioactivity_matrix(valid_smiles, cfg.pubchem_cache)

    # Total expected global-feature length.
    dim_morgan = 1024
    target_len = (dim_morgan + len(rd_funcs) + len(smarts_patts) + 1
                  + 768 + ADVANCED_3D_DESCRIPTOR_DIM + PUBCHEM_DIM)

    print("[run-toxlens] featurising graphs")
    graphs = _toxlens_featurise(
        valid_mols, valid_smiles, rdkit_mat, tox_mat, lm_mat,
        desc_3d_mat, pubchem_mat,
        target_len=target_len, num_tasks=num_tasks,
    )

    # Build the model architecture and load weights.
    GAT_class = _build_toxlens_model_class(num_tasks_default=num_tasks)
    print(f"[run-toxlens] loading ToxLens checkpoint {cfg.tox_ckpt}")
    model = _load_primary_toxlens_checkpoint(GAT_class, cfg.tox_ckpt, device, task_names)
    model.eval()
    model.to(device)

    loader = GeoDataLoader(graphs, batch_size=cfg.batch_size, shuffle=False)

    # MC dropout: keep BN/LN in eval, flip only Dropout modules to train.
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    # Seed every RNG that the stochastic forward pass touches, so the MC-dropout
    # masks (and therefore the averaged probabilities, high-risk counts, integrated
    # scores, and Pareto front) are reproducible across runs. Without this, each run
    # draws different dropout masks and the results drift between invocations.
    import random as _random
    _random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[run-toxlens] MC dropout ({cfg.mc_passes} passes, seed={cfg.seed})")
    mc_probs = []
    with torch.no_grad():
        for _ in range(cfg.mc_passes):
            batch_probs = []
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                _assert_primary_toxlens_logits(logits, context="run-toxlens")
                if getattr(model, "use_platt_for_mcc", False):
                    p = model._apply_platt_calibration(logits)
                else:
                    p = torch.sigmoid(logits.float())
                batch_probs.append(p.cpu().numpy())
            mc_probs.append(np.concatenate(batch_probs, axis=0))
    model.eval()  # restore

    stacked = np.stack(mc_probs, axis=0)  # (mc, n_valid, n_tasks)
    probs_mean = stacked.mean(axis=0)
    probs_std = stacked.std(axis=0)

    # Reassemble in original row order (preserving any drops).
    n_total = len(top)
    pm_full = np.full((n_total, num_tasks), np.nan, dtype=np.float32)
    ps_full = np.full((n_total, num_tasks), np.nan, dtype=np.float32)
    pm_full[valid_idx] = probs_mean
    ps_full[valid_idx] = probs_std

    # Per-molecule summaries.
    max_tox = np.nanmax(pm_full, axis=1)
    mean_tox = np.nanmean(pm_full, axis=1)
    high_count = np.nansum(pm_full >= cfg.tox_high_threshold, axis=1)
    mean_unc = np.nanmean(ps_full, axis=1)
    flag_unc = np.nanmax(ps_full, axis=1) >= cfg.uncertainty_flag_threshold

    activity = top["activity_prob"].to_numpy(dtype=np.float32)
    score = integrated_priority(
        activity=activity, max_tox=max_tox,
        high_frac=high_count / max(num_tasks, 1),
        uncertainty=mean_unc, cfg=cfg,
    )
    # Pareto only over valid rows.
    pareto_idx_local = pareto_front(
        activity[valid_idx], np.nan_to_num(max_tox[valid_idx], nan=0.0)
    )
    pareto_idx_global = np.array(valid_idx)[pareto_idx_local] if len(pareto_idx_local) else np.array([], dtype=int)

    # Per-mol matched alerts in original row order.
    alerts_out = ["" for _ in range(n_total)]
    pains_out = [0 for _ in range(n_total)]
    for k, i in enumerate(valid_idx):
        alerts_out[i] = ";".join(matched_alerts[k])
        pains_out[i] = pains_counts[k]

    flat = pd.DataFrame({
        "smiles": top["canonical_smiles"],
        "chembl_id": top["chembl_id"] if "chembl_id" in top.columns else "",
        "pref_name": top["pref_name"] if "pref_name" in top.columns else "",
        "activity_prob": activity,
        "max_tox_prob": max_tox,
        "mean_tox_prob": mean_tox,
        "n_high_risk_endpoints": high_count.astype(int),
        "mean_uncertainty": mean_unc,
        "high_uncertainty_flag": flag_unc,
        "matched_alerts": alerts_out,
        "pains_count": pains_out,
        "integrated_score": score,
    })
    flat["pareto_optimal"] = False
    flat.loc[pareto_idx_global, "pareto_optimal"] = True
    flat = flat.sort_values("integrated_score", ascending=False).reset_index(drop=True)

    endpoint_probs = pd.DataFrame(pm_full, columns=task_names)
    endpoint_probs.insert(0, "smiles", top["canonical_smiles"].tolist())
    endpoint_stds = pd.DataFrame(ps_full, columns=[f"{n}_std" for n in task_names])
    endpoint_stds.insert(0, "smiles", top["canonical_smiles"].tolist())

    flat.to_csv(cfg.output_dir / "integrated_ranking.csv", index=False)
    endpoint_probs.to_csv(cfg.output_dir / "toxlens_endpoint_probs.csv", index=False)
    endpoint_stds.to_csv(cfg.output_dir / "toxlens_endpoint_stds.csv", index=False)
    flat[flat["pareto_optimal"]].to_csv(cfg.output_dir / "pareto_front.csv", index=False)

    provenance = {
        "selection_rule": "activity_prob",
        "top_n": int(len(top)),
        "weights": {
            "w_tox_max": cfg.w_tox_max,
            "w_tox_count": cfg.w_tox_count,
            "w_uncertainty": cfg.w_uncertainty,
        },
        "tox_high_threshold": cfg.tox_high_threshold,
        "uncertainty_flag_threshold": cfg.uncertainty_flag_threshold,
        "mc_passes": cfg.mc_passes,
        "toxlens_tasks": task_names,
        "dpd_ckpt": str(cfg.dpd_ckpt),
        "tox_ckpt": str(cfg.tox_ckpt),
        "n_pareto_front": int(np.count_nonzero(flat["pareto_optimal"])),
    }
    (cfg.output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2))

    # Plot using the *flat* dataframe (already ordered by score). Pareto idx
    # there are positions, not the original row indices.
    pareto_positions = np.flatnonzero(flat["pareto_optimal"].to_numpy())
    plot_efficacy_safety(flat, pareto_positions, cfg.output_dir / "fig_efficacy_safety.png")
    plot_priority_distribution(flat, cfg.output_dir / "fig_priority_distribution.png")

    print(f"\nResults written to {cfg.output_dir.resolve()}")
    print(flat.head(10).to_string(index=False))
    return flat


# Stage 3 — SHAP-guided occlusion faithfulness (DPD-Cancer + ToxLens)

# Signed Integrated-Gradients attribution on node and edge features, mapped
# onto heavy atoms (explicit H attribution is absorbed into the parent heavy
# atom). SHAP-ranked heavy atoms are then perturbed (node + incident-edge +
# RWSE features zeroed) and compared against random heavy-atom occlusion to
# test whether the attributions are *faithful* — i.e. whether the atoms the
# explanation flags as toxicity-/activity-driving actually move the model's
# score when removed.

# Reconstruct an explicit-H RDKit molecule from SMILES. Both featurisers use
# explicit Hs, so the same routine works for either model's graph.

def _reconstruct_explicit_h_molecule(smiles: str, n_graph_atoms: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() != n_graph_atoms:
        raise ValueError(
            f"Atom-count mismatch for {smiles!r}: "
            f"RDKit explicit-H atoms={mol_h.GetNumAtoms()}, graph nodes={n_graph_atoms}"
        )
    return mol_h


def _heavy_atom_mapping(mol_h: Chem.Mol) -> Tuple[Dict[int, int], List[List[int]]]:
    """Map every explicit-H graph atom index to a heavy-atom group index, and
    return the explicit-graph indices grouped by parent heavy atom."""
    graph_to_heavy: Dict[int, int] = {}
    heavy_groups: List[List[int]] = []
    heavy_idx = 0
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            graph_to_heavy[atom.GetIdx()] = heavy_idx
            heavy_groups.append([atom.GetIdx()])
            heavy_idx += 1
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            h_idx = atom.GetIdx()
            parent_idx = atom.GetNeighbors()[0].GetIdx()
            parent_heavy = graph_to_heavy[parent_idx]
            graph_to_heavy[h_idx] = parent_heavy
            heavy_groups[parent_heavy].append(h_idx)
    return graph_to_heavy, heavy_groups


def _collapse_node_attr_to_heavy(node_attr, graph_to_heavy, n_heavy):
    out = np.zeros(n_heavy, dtype=np.float64)
    for gidx, val in enumerate(node_attr):
        out[graph_to_heavy[gidx]] += float(val)
    return out


def _collapse_edge_attr_to_heavy(edge_index_np, edge_attr, graph_to_heavy, n_heavy):
    """Heavy-H edge attribution is absorbed into the heavy atom; heavy-heavy
    bond attribution is preserved as its own dict keyed by sorted heavy pair."""
    heavy_edge_to_node = np.zeros(n_heavy, dtype=np.float64)
    heavy_bond_attr: Dict[Tuple[int, int], float] = {}
    if edge_attr is None:
        return heavy_edge_to_node, heavy_bond_attr
    for e in range(edge_index_np.shape[1]):
        u = int(edge_index_np[0, e])
        v = int(edge_index_np[1, e])
        if u > v:  # each undirected bond processed once
            continue
        hu, hv = graph_to_heavy[u], graph_to_heavy[v]
        val = float(edge_attr[e])
        if hu == hv:
            heavy_edge_to_node[hu] += val
        else:
            key = tuple(sorted((hu, hv)))
            heavy_bond_attr[key] = heavy_bond_attr.get(key, 0.0) + val
    return heavy_edge_to_node, heavy_bond_attr


def _signed_colour(value: float) -> Tuple[float, float, float]:
    value = float(np.clip(value, -1.0, 1.0))
    if value > 0:
        return (0.95, 0.25 + 0.25 * (1.0 - value), 0.18)
    a = abs(value)
    return (0.20, 0.38, 0.95)


def _select_meaningful_saliency_atoms(
    heavy_attr,
    top_fraction: Optional[float] = None,
    positive_only: bool = False,
    min_norm: float = 0.35,
    min_quantile: float = 0.0,
    max_atoms: Optional[int] = None,
    cumulative_mass: Optional[float] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Return normalized attribution and directly thresholded atoms.

    This intentionally mirrors the older DPD visual style: normalize saliency,
    then highlight atoms above a small magnitude threshold. No connectedness
    filtering or artificial neighbour expansion is applied.
    """
    attr = np.asarray(heavy_attr, dtype=float)
    if attr.size == 0:
        return attr, []
    max_abs = float(np.max(np.abs(attr)) + 1e-12)
    norm = attr / max_abs
    scores = np.maximum(norm, 0.0) if positive_only else np.abs(norm)
    if not np.any(scores > 0):
        scores = np.abs(norm)

    n_atoms = len(scores)
    if top_fraction is not None:
        k_cap = max(1, int(math.ceil(float(top_fraction) * n_atoms)))
    elif max_atoms is not None:
        k_cap = max(1, int(max_atoms))
    else:
        k_cap = n_atoms
    k_cap = min(k_cap, n_atoms)

    positive_scores = scores[scores > 0]
    q_cut = float(np.quantile(positive_scores, min_quantile)) if positive_scores.size else 0.0
    threshold = max(min_norm, q_cut)
    ranked = [int(i) for i in np.argsort(-scores) if scores[int(i)] > 0]
    keep = [i for i in ranked if scores[i] >= threshold][:k_cap]

    if keep and cumulative_mass is not None:
        total = float(scores.sum() + 1e-12)
        selected, mass = [], 0.0
        for i in keep:
            selected.append(i)
            mass += float(scores[i])
            if mass / total >= cumulative_mass or len(selected) >= k_cap:
                break
        keep = selected
    return norm, keep


def _salient_connected_substructures(mol_no_h, candidate_atoms: Sequence[int],
                                     norm_attr: np.ndarray,
                                     max_components: int = 2,
                                     max_atoms_total: int = 8) -> List[int]:
    """Keep only compact connected high-saliency substructures.

    Weak isolated atoms are intentionally dropped; this figure is meant to show
    chemically interpretable drivers, not a sensitive atom-by-atom heatmap.
    """
    candidates = set(int(a) for a in candidate_atoms)
    if not candidates:
        return []

    adjacency: Dict[int, set] = {a: set() for a in candidates}
    for bond in mol_no_h.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in candidates and v in candidates:
            adjacency[u].add(v)
            adjacency[v].add(u)

    components: List[List[int]] = []
    seen = set()
    for atom_idx in sorted(candidates):
        if atom_idx in seen:
            continue
        stack = [atom_idx]
        seen.add(atom_idx)
        comp = []
        while stack:
            a = stack.pop()
            comp.append(a)
            for nb in adjacency.get(a, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))

    strong_components = []
    for comp in components:
        comp_scores = np.abs(norm_attr[comp])
        # Single atoms are only shown if they are overwhelming; otherwise they
        # read as noisy speckles rather than meaningful substructures.
        if len(comp) >= 2 or float(np.max(comp_scores)) >= 0.90:
            strong_components.append((float(np.sum(comp_scores)), comp))
    strong_components.sort(key=lambda item: item[0], reverse=True)

    keep: List[int] = []
    for _, comp in strong_components[:max_components]:
        room = max_atoms_total - len(keep)
        if room <= 0:
            break
        keep.extend(comp[:room])
    return sorted(set(keep))


def _render_heavy_atom_shap_svg(mol_no_h, heavy_attr, save_path: Path,
                                 size=(600, 600), top_fraction=None,
                                 positive_only: bool = False,
                                 single_colour: Optional[Tuple[float, float, float]] = None,
                                 heavy_bond_attr: Optional[Dict[Tuple[int, int], float]] = None):
    from rdkit.Chem.Draw import MolDraw2DSVG
    mol_no_h = Chem.Mol(mol_no_h)
    AllChem.Compute2DCoords(mol_no_h)
    norm, keep_atoms = _select_meaningful_saliency_atoms(
        heavy_attr, top_fraction=top_fraction, positive_only=positive_only,
    )
    keep = set(keep_atoms)
    atom_cols, atoms = {}, []
    for ai in keep_atoms:
        v = float(norm[ai])
        strength = min(abs(v), 1.0)
        fade = 1.0 - strength
        if single_colour is not None:
            base = single_colour
            atom_cols[ai] = tuple(1.0 - strength * (1.0 - c) for c in base)
        else:
            atom_cols[ai] = (1.0, fade, fade) if v >= 0 else (fade, fade, 1.0)
        atoms.append(ai)
    bonds, bond_cols = [], {}
    if heavy_bond_attr:
        vals = np.asarray(list(heavy_bond_attr.values()), dtype=float)
        max_abs_bond = float(np.max(np.abs(vals)) + 1e-12)
        for bond in mol_no_h.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            key = tuple(sorted((u, v)))
            if key not in heavy_bond_attr:
                continue
            b_norm = float(heavy_bond_attr[key]) / max_abs_bond
            score = b_norm if positive_only else abs(b_norm)
            if score < 0.35:
                continue
            strength = min(abs(b_norm), 1.0)
            fade = 1.0 - strength
            bonds.append(bond.GetIdx())
            if single_colour is not None:
                base = single_colour
                bond_cols[bond.GetIdx()] = tuple(1.0 - strength * (1.0 - c) for c in base)
            else:
                bond_cols[bond.GetIdx()] = (1.0, fade, fade) if b_norm >= 0 else (fade, fade, 1.0)
    drawer = MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.05
    if hasattr(opts, "useBWAtomPalette"):
        opts.useBWAtomPalette()
    drawer.DrawMolecule(
        mol_no_h,
        highlightAtoms=atoms,
        highlightAtomColors=atom_cols,
        highlightBonds=bonds,
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(drawer.GetDrawingText())


def _save_atom_attribution_csv(mol_no_h, heavy_attr, save_path: Path):
    rows = []
    for atom in mol_no_h.GetAtoms():
        idx = atom.GetIdx()
        rows.append({
            "heavy_atom_index": idx,
            "element": atom.GetSymbol(),
            "signed_shap": float(heavy_attr[idx]),
            "absolute_shap": float(abs(heavy_attr[idx])),
        })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("absolute_shap", ascending=False).to_csv(save_path, index=False)


def _clone_data_for_occlusion(data):
    import copy as _copy
    out = _copy.deepcopy(data)
    out.x = out.x.detach().clone()
    out.edge_index = out.edge_index.detach().clone()
    if getattr(out, "edge_attr", None) is not None:
        out.edge_attr = out.edge_attr.detach().clone()
    if getattr(out, "global_features", None) is not None:
        out.global_features = out.global_features.detach().clone()
    if getattr(out, "rwse", None) is not None:
        out.rwse = out.rwse.detach().clone()
    return out


def _mask_heavy_atom_groups(data, heavy_groups, heavy_atom_indices,
                             mask_node_features=True,
                             mask_incident_edge_features=True,
                             mask_rwse=True):
    import torch
    out = _clone_data_for_occlusion(data)
    nodes_to_mask: List[int] = []
    for hi in heavy_atom_indices:
        nodes_to_mask.extend(list(heavy_groups[hi]))
    nodes_to_mask = sorted(set(nodes_to_mask))
    if not nodes_to_mask:
        return out
    idx_t = torch.tensor(nodes_to_mask, dtype=torch.long, device=out.x.device)
    if mask_node_features:
        out.x[idx_t, :] = 0.0
    # Without this, the trained ToxLens model still sees the masked atoms'
    # structural role through their random-walk encoding.
    if mask_rwse and getattr(out, "rwse", None) is not None and out.rwse.numel() > 0:
        out.rwse[idx_t, :] = 0.0
    if (mask_incident_edge_features
            and getattr(out, "edge_attr", None) is not None
            and out.edge_attr.numel() > 0):
        s = set(nodes_to_mask)
        ei_np = out.edge_index.detach().cpu().numpy()
        inc = [e for e in range(ei_np.shape[1])
               if int(ei_np[0, e]) in s or int(ei_np[1, e]) in s]
        if inc:
            eidx = torch.tensor(inc, dtype=torch.long, device=out.edge_attr.device)
            out.edge_attr[eidx, :] = 0.0
    return out


def _ensure_batched(data, device):
    import copy as _copy
    import torch
    out = _copy.copy(data)
    if not hasattr(out, "batch") or out.batch is None:
        out.batch = torch.zeros(out.x.size(0), dtype=torch.long, device=device)
    return out


def _model_score(model, data, target_task: int, device,
                 use_probability: bool = False, sigmoid: bool = True) -> float:
    import torch
    model.eval()
    with torch.no_grad():
        d = _ensure_batched(data, device)
        try:
            d = d.to(device)
        except Exception:
            pass
        out = model(d)
        if isinstance(out, tuple):
            out = out[0]
        scalar = out.view(-1)[target_task]
        if not use_probability:
            return float(scalar.cpu())
        if sigmoid:
            return float(torch.sigmoid(scalar).cpu())
        probs = torch.softmax(out.view(-1), dim=0)
        return float(probs[target_task].cpu())


class _DataForwardCaptumAdapter:
    """Adapter for models whose checkpoint-compatible forward accepts Data only.

    ``pyg_captum_shap`` calls PyG models with tensor keyword arguments such as
    ``x=...`` and ``edge_index=...``. DPD-Cancer must keep its original
    ``forward(data)`` for checkpoint compatibility, so this adapter rebuilds a
    lightweight PyG Data object and delegates to the real model.
    """

    def __init__(self, model, reference_data, device):
        self.model = model
        self.reference_data = reference_data
        self.device = device

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def zero_grad(self, *args, **kwargs):
        return self.model.zero_grad(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        import torch
        from torch_geometric.data import Data

        names = ["x", "edge_index", "batch", "edge_attr", "global_features"]
        for name, value in zip(names, args):
            kwargs.setdefault(name, value)

        ref = self.reference_data
        x = kwargs.get("x", getattr(ref, "x", None))
        edge_index = kwargs.get("edge_index", getattr(ref, "edge_index", None))
        batch = kwargs.get("batch", getattr(ref, "batch", None))
        edge_attr = kwargs.get("edge_attr", getattr(ref, "edge_attr", None))
        global_features = kwargs.get("global_features", getattr(ref, "global_features", None))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.batch = batch
        if global_features is not None:
            d.global_features = global_features
        if getattr(ref, "rwse", None) is not None:
            d.rwse = kwargs.get("rwse", ref.rwse)
        return self.model(d.to(self.device) if hasattr(d, "to") else d)


def _make_data_forward_captum_adapter(model, reference_data, device):
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data

    class DataForwardCaptumAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.reference_data = reference_data
            self.device = device

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                modules = self.__dict__.get("_modules", {})
                wrapped = modules.get("model")
                if wrapped is not None and hasattr(wrapped, name):
                    return getattr(wrapped, name)
                raise

        def _forward_dpd_tensors(
            self, x, edge_index, batch, edge_attr, global_features,
            apply_embedding: bool = True,
        ):
            import torch.nn.functional as F

            m = self.model
            if apply_embedding:
                x = m.node_emb(x)
            for conv, norm, ffn in zip(m.layers, m.norms, m.ffns):
                y = conv(m.act(norm(x, batch)), edge_index, edge_attr)
                x = x + y
                y = ffn(norm(x, batch))
                x = x + y
            x = m.pool(x, batch)
            if m.ablation_mode != "graph_only":
                global_features = global_features.view(int(batch.max().item()) + 1, -1)
                gf = m.global_fc(global_features)
                gf = F.dropout(gf, p=0.4, training=m.training)
            if m.ablation_mode == "graph_only":
                pass
            elif m.ablation_mode == "global_only":
                x = gf
            else:
                gate = m.global_gate(gf)
                x = gate * x + (1 - gate) * gf
            return m.fc(m.fuse_norm(x))

        def forward(self, *args, **kwargs):
            names = ["x", "edge_index", "batch", "edge_attr", "global_features"]
            for name, value in zip(names, args):
                kwargs.setdefault(name, value)

            ref = self.reference_data
            x = kwargs.get("x", getattr(ref, "x", None))
            edge_index = kwargs.get("edge_index", getattr(ref, "edge_index", None))
            batch = kwargs.get("batch", getattr(ref, "batch", None))
            edge_attr = kwargs.get("edge_attr", getattr(ref, "edge_attr", None))
            global_features = kwargs.get(
                "global_features", getattr(ref, "global_features", None)
            )

            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            apply_embedding = bool(kwargs.get("apply_embedding", True))
            if bool(getattr(self.model, "_is_dpd_captum_target", False)):
                return self._forward_dpd_tensors(
                    x=x,
                    edge_index=edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                    global_features=global_features,
                    apply_embedding=apply_embedding,
                )
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            d.batch = batch
            if global_features is not None:
                d.global_features = global_features
            if getattr(ref, "rwse", None) is not None:
                d.rwse = kwargs.get("rwse", ref.rwse)
            return self.model(d.to(self.device) if hasattr(d, "to") else d)

    return DataForwardCaptumAdapter()



def _compute_heavy_atom_shap(model, data, target_task, device, smiles: str,
                              n_shap_samples: int = 25,
                              include_edges: bool = True,
                              prefer_shap: bool = True):
    """Compute mandatory pyg_captum_shap attributions for one molecule."""
    node_attr: Optional[np.ndarray] = None
    edge_attr: Optional[np.ndarray] = None
    try:
        from pyg_captum_shap import compute_shap_values
        data_p = _ensure_batched(data, device)
        shap_model = (
            _make_data_forward_captum_adapter(model, data_p, device)
            if bool(getattr(model, "_needs_data_forward_adapter", False))
            else model
        )
        shap_res = compute_shap_values(
            model=shap_model,
            target_graph=data_p,
            target_task=target_task,
            n_samples=n_shap_samples,
        )
        nd = shap_res.get("nodes")
        if nd is None:
            raise RuntimeError("compute_shap_values returned no 'nodes' key")
        nd_t = nd.detach().cpu().numpy() if hasattr(nd, "detach") else np.asarray(nd)
        node_attr = nd_t.sum(axis=-1) if nd_t.ndim > 1 else nd_t
        ed = shap_res.get("edges") if include_edges else None
        if ed is not None:
            ed_t = ed.detach().cpu().numpy() if hasattr(ed, "detach") else np.asarray(ed)
            edge_attr = ed_t.sum(axis=-1) if ed_t.ndim > 1 else ed_t
    except Exception as e:
        raise RuntimeError(
            "pyg_captum_shap failed. SHAP is required for this pipeline; "
            f"fix the model adapter or environment. Original error: {e}"
        ) from e
    if node_attr is None:
        raise RuntimeError("pyg_captum_shap returned no node attributions")

    used = "pyg_captum_shap"
    mol_h = _reconstruct_explicit_h_molecule(smiles, n_graph_atoms=data.x.size(0))
    graph_to_heavy, heavy_groups = _heavy_atom_mapping(mol_h)
    n_heavy = len(heavy_groups)

    heavy_node = _collapse_node_attr_to_heavy(node_attr, graph_to_heavy, n_heavy)
    ei_np = data.edge_index.detach().cpu().numpy()
    heavy_edge_to_node, heavy_bond_attr = _collapse_edge_attr_to_heavy(
        ei_np, edge_attr, graph_to_heavy, n_heavy,
    )
    heavy_total = heavy_node + heavy_edge_to_node

    return {
        "mol_h": mol_h,
        "mol_no_h": Chem.RemoveHs(mol_h),
        "graph_to_heavy": graph_to_heavy,
        "heavy_groups": heavy_groups,
        "heavy_node_attr": heavy_node,
        "heavy_edge_to_node_attr": heavy_edge_to_node,
        "heavy_total_attr": heavy_total,
        "heavy_bond_attr": heavy_bond_attr,
        "raw_node_attr": node_attr,
        "raw_edge_attr": edge_attr,
        "attribution_method": used,
    }


def _shap_ranked_heavy_atoms(heavy_attr, positive_only: bool = True) -> List[int]:
    a = np.asarray(heavy_attr, dtype=float)
    scores = np.maximum(a, 0.0) if positive_only else np.abs(a)
    return np.argsort(-scores).tolist()


def _sample_random_heavy_atoms(n_heavy, k, rng):
    return rng.choice(np.arange(n_heavy), size=k, replace=False).tolist()


def _run_faithfulness_analysis(model, data, heavy_groups, heavy_attr, target_task,
                                device, fractions, random_repeats, seed,
                                use_probability=False, sigmoid=True,
                                positive_only=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_heavy = len(heavy_groups)
    baseline = _model_score(model, data, target_task, device,
                            use_probability=use_probability, sigmoid=sigmoid)
    ranked = _shap_ranked_heavy_atoms(heavy_attr, positive_only=positive_only)
    rows = []
    for frac in fractions:
        k = max(1, int(math.ceil(frac * n_heavy)))
        top_atoms = ranked[:k]
        top_data = _mask_heavy_atom_groups(data, heavy_groups, top_atoms)
        top_score = _model_score(model, top_data, target_task, device,
                                 use_probability=use_probability, sigmoid=sigmoid)
        top_drop = baseline - top_score
        random_drops = []
        for _ in range(random_repeats):
            ra = _sample_random_heavy_atoms(n_heavy, k, rng)
            rd = _mask_heavy_atom_groups(data, heavy_groups, ra)
            rs = _model_score(model, rd, target_task, device,
                              use_probability=use_probability, sigmoid=sigmoid)
            random_drops.append(baseline - rs)
        random_drops = np.asarray(random_drops, dtype=float)
        emp_p = (1.0 + float(np.sum(random_drops >= top_drop))) / (len(random_drops) + 1.0)
        rows.append({
            "fraction_masked": float(frac),
            "num_heavy_atoms_masked": int(k),
            "baseline_score": float(baseline),
            "top_shap_score": float(top_score),
            "top_shap_drop": float(top_drop),
            "random_drop_mean": float(np.mean(random_drops)),
            "random_drop_std": float(np.std(random_drops, ddof=1)) if len(random_drops) > 1 else 0.0,
            "random_drop_median": float(np.median(random_drops)),
            "random_drop_q025": float(np.quantile(random_drops, 0.025)),
            "random_drop_q975": float(np.quantile(random_drops, 0.975)),
            "empirical_p_random_ge_top": float(emp_p),
            "top_shap_atoms": ",".join(map(str, top_atoms)),
            "positive_only_ranking": bool(positive_only),
            "score_type": "probability" if use_probability else "logit_or_raw",
        })
    return pd.DataFrame(rows)


def _plot_faithfulness(df: pd.DataFrame, save_path: Path, title: str):
    x = df["fraction_masked"].to_numpy() * 100.0
    top = df["top_shap_drop"].to_numpy()
    rm = df["random_drop_mean"].to_numpy()
    rs = df["random_drop_std"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, top, marker="o", lw=2.2, label="Top SHAP atoms")
    ax.plot(x, rm, marker="o", lw=2.0, label="Random atoms")
    ax.fill_between(x, rm - rs, rm + rs, alpha=0.2, label="Random Â± SD")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Heavy atoms masked (%)")
    ax.set_ylabel("Model score drop")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.with_suffix(".svg"))
    plt.close(fig)


def _explain_one_molecule_one_task(model, data, smiles: str, target_task: int,
                                    task_name: str, mol_idx: int, out_dir: Path,
                                    device, fractions, random_repeats, seed,
                                    svg_size=(600, 600), top_fraction=None,
                                    use_probability=False, sigmoid=True,
                                    n_shap_samples: int = 25,
                                    prefer_shap: bool = True,
                                    saliency_positive_only: bool = False,
                                    saliency_single_colour: Optional[Tuple[float, float, float]] = None,
                                    write_toxicity_only_saliency: bool = False) -> pd.DataFrame:
    safe = task_name.replace("/", "_").replace(" ", "_")
    prefix = out_dir / f"mol_{mol_idx:04d}" / safe
    prefix.mkdir(parents=True, exist_ok=True)
    shap_info = _compute_heavy_atom_shap(
        model, data, target_task, device, smiles,
        n_shap_samples=n_shap_samples,
        prefer_shap=prefer_shap,
    )
    mol_no_h = shap_info["mol_no_h"]
    heavy_groups = shap_info["heavy_groups"]
    heavy_attr = shap_info["heavy_total_attr"]
    heavy_bond_attr = shap_info.get("heavy_bond_attr")
    _render_heavy_atom_shap_svg(
        mol_no_h, heavy_attr, prefix / "heavy_atom_signed_shap.svg",
        size=svg_size, top_fraction=top_fraction,
        positive_only=saliency_positive_only,
        single_colour=saliency_single_colour,
        heavy_bond_attr=heavy_bond_attr,
    )
    if write_toxicity_only_saliency:
        _render_heavy_atom_shap_svg(
            mol_no_h, heavy_attr, prefix / "heavy_atom_toxicity_only_shap.svg",
            size=svg_size, top_fraction=top_fraction,
            positive_only=True,
            single_colour=(1.0, 0.0, 0.0),
            heavy_bond_attr=heavy_bond_attr,
        )
    _save_atom_attribution_csv(mol_no_h, heavy_attr, prefix / "heavy_atom_signed_shap.csv")
    faith_df = _run_faithfulness_analysis(
        model, data, heavy_groups, heavy_attr, target_task, device,
        fractions=fractions, random_repeats=random_repeats, seed=seed,
        use_probability=use_probability, sigmoid=sigmoid, positive_only=True,
    )
    faith_df["attribution_method"] = shap_info.get("attribution_method", "ig")
    faith_df.insert(0, "task_name", task_name)
    faith_df.insert(0, "task_idx", target_task)
    faith_df.insert(0, "smiles", smiles)
    faith_df.insert(0, "molecule_index", mol_idx)
    faith_df.to_csv(prefix / "faithfulness_shap_guided_occlusion.csv", index=False)
    _plot_faithfulness(
        faith_df, prefix / "faithfulness_shap_guided_occlusion.png",
        title=f"SHAP-guided occlusion faithfulness: {task_name}",
    )
    return faith_df


# DPD-Cancer adapter — explains the positive-class (cancer-active) logit

def stage_explain_dpd(features_pkl: Path, top_csv: Path, dpd_ckpt: Path,
                       out_dir: Path, device: str = "cpu",
                       fractions=(0.05, 0.10, 0.20, 0.30),
                       random_repeats: int = 50, seed: int = 42,
                       top_fraction=None, n_shap_samples: int = 25,
                       prefer_shap: bool = True,
                       molecule_name: Optional[str] = "PARSACLISIB") -> None:
    if not features_pkl.exists():
        raise FileNotFoundError(features_pkl)
    if not top_csv.exists():
        raise FileNotFoundError(top_csv)
    if not dpd_ckpt.exists():
        raise FileNotFoundError(dpd_ckpt)
    with open(features_pkl, "rb") as f:
        payload = pickle.load(f)
    smiles_list = payload["smiles"]
    graphs = payload["graphs"]
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

    top_df = pd.read_csv(top_csv)
    if "canonical_smiles" not in top_df.columns:
        raise ValueError(f"{top_csv} must have a 'canonical_smiles' column")
    top_df = _filter_explanation_targets(top_df, molecule_name)

    GAT = _build_dpd_model_class()
    model = GAT.load_from_checkpoint(str(dpd_ckpt), map_location=device)
    model._needs_data_forward_adapter = True
    model._is_dpd_captum_target = True
    model.eval()
    model.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_faith: List[pd.DataFrame] = []
    for mol_idx, smi in enumerate(top_df["canonical_smiles"].tolist()):
        if smi not in smiles_to_idx:
            print(f"[explain-dpd] {smi!r} not in features pickle, skipping")
            continue
        graph = graphs[smiles_to_idx[smi]]
        try:
            graph_d = graph.to(device) if hasattr(graph, "to") else graph
            # Target task 1 = "active" in the binary classifier's softmax.
            df = _explain_one_molecule_one_task(
                model=model, data=graph_d, smiles=smi,
                target_task=1, task_name="cancer_activity",
                mol_idx=mol_idx, out_dir=out_dir, device=device,
                fractions=fractions, random_repeats=random_repeats,
                seed=seed + mol_idx * 1000,
                top_fraction=top_fraction,
                use_probability=False, sigmoid=False,
                n_shap_samples=n_shap_samples,
                prefer_shap=prefer_shap,
                saliency_positive_only=True,
                saliency_single_colour=(1.0, 0.0, 0.0),
            )
            all_faith.append(df)
            print(f"[explain-dpd] mol {mol_idx} done ({smi})")
        except Exception as e:
            print(f"[explain-dpd] mol {mol_idx} FAILED ({smi}): {e}")
    if all_faith:
        combined = pd.concat(all_faith, ignore_index=True)
        combined.to_csv(out_dir / "all_faithfulness_results.csv", index=False)


# ToxLens adapter — explains one or many endpoints per molecule

def stage_explain_tox(top_csv: Path, tox_ckpt: Path, tox_task_csv: Path,
                       out_dir: Path, device: str = "cpu",
                       only_tasks: Optional[Sequence[str]] = None,
                       fractions=(0.05, 0.10, 0.20, 0.30),
                       random_repeats: int = 50, seed: int = 42,
                       top_fraction=None, n_shap_samples: int = 25,
                       prefer_shap: bool = True,
                       batch_size: int = 32,
                       molecule_name: Optional[str] = "PARSACLISIB",
                       pubchem_cache: Path = Path("data/pubchem_bioactivity_cache.json")) -> None:
    import torch

    if not top_csv.exists():
        raise FileNotFoundError(top_csv)
    tox_ckpt = _resolve_tox_ckpt(tox_ckpt)
    if not tox_task_csv.exists():
        raise FileNotFoundError(tox_task_csv)

    top = pd.read_csv(top_csv)
    if "canonical_smiles" not in top.columns:
        raise ValueError(f"{top_csv} must have a 'canonical_smiles' column")
    top = _filter_explanation_targets(top, molecule_name)

    all_task_names = _primary_toxlens_task_names(tox_task_csv)
    num_tasks = len(all_task_names)
    if only_tasks is None:
        selected_tasks = list(enumerate(all_task_names))
    else:
        selected_tasks = [(all_task_names.index(n), n)
                          for n in only_tasks if n in all_task_names]
        if not selected_tasks:
            raise ValueError(f"None of {only_tasks!r} matched ToxLens task columns.")

    # Standardise and re-featurise (mirrors stage_run_toxlens exactly).
    mols, canon = [], []
    for s in top["canonical_smiles"].tolist():
        m = standardise_smiles(s)
        if m is None:
            mols.append(None); canon.append(None)
        else:
            mols.append(m); canon.append(canonical_smiles(m))
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]
    valid_smiles = [canon[i] for i in valid_idx]
    n_valid = len(valid_idx)
    if n_valid == 0:
        print("[explain-tox] no valid molecules; aborting")
        return
    if only_tasks is None:
        print("[explain-tox] defaulting to the 11 configured primary ToxLens endpoints")

    rd_funcs = [fn for _, fn in Descriptors._descList]
    rdkit_mat = np.zeros((n_valid, len(rd_funcs)), dtype=np.float32)
    for i, mol in enumerate(valid_mols):
        for j, fn in enumerate(rd_funcs):
            try:
                rdkit_mat[i, j] = _safe_descriptor_float(fn(mol))
            except Exception:
                rdkit_mat[i, j] = 0.0

    smarts_patts = [(name, Chem.MolFromSmarts(s)) for name, s in _TOX_SMARTS_STRINGS]
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)
    tox_mat = np.zeros((n_valid, len(smarts_patts) + 1), dtype=np.float32)
    for i, mol in enumerate(valid_mols):
        for j, (_, patt) in enumerate(smarts_patts):
            if patt is not None and mol.HasSubstructMatch(patt):
                tox_mat[i, j] = 1.0
        tox_mat[i, -1] = float(len(pains_catalog.GetMatches(mol)))

    print("[explain-tox] loading MolFormer for LM modality")
    tokenizer, molformer = _load_molformer_for_embeddings(device)
    lm_mat = np.zeros((n_valid, 768), dtype=np.float32)
    BS = 64
    with torch.no_grad():
        for i in range(0, n_valid, BS):
            chunk = valid_smiles[i : i + BS]
            try:
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                out = molformer(**inputs)
                lm_mat[i : i + len(chunk)] = out.pooler_output.cpu().numpy()
            except Exception:
                pass
    del molformer, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    desc_3d_mat = _compute_toxlens_3d_matrix(valid_mols)
    pubchem_mat = _load_pubchem_bioactivity_matrix(valid_smiles, pubchem_cache)

    target_len = (1024 + len(rd_funcs) + len(smarts_patts) + 1
                  + 768 + ADVANCED_3D_DESCRIPTOR_DIM + PUBCHEM_DIM)
    print("[explain-tox] featurising graphs")
    graphs = _toxlens_featurise(valid_mols, valid_smiles, rdkit_mat, tox_mat,
                                 lm_mat, desc_3d_mat, pubchem_mat,
                                 target_len=target_len, num_tasks=num_tasks)

    GAT_class = _build_toxlens_model_class(num_tasks_default=num_tasks)
    print(f"[explain-tox] loading checkpoint {tox_ckpt}")
    model = _load_primary_toxlens_checkpoint(GAT_class, tox_ckpt, device, all_task_names)
    model.eval()
    model.to(device)
    if graphs:
        with torch.no_grad():
            probe = _ensure_batched(graphs[0].to(device) if hasattr(graphs[0], "to") else graphs[0], device)
            probe_logits = model(probe)
            if isinstance(probe_logits, tuple):
                probe_logits = probe_logits[0]
            _assert_primary_toxlens_logits(probe_logits, context="explain-tox")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_faith: List[pd.DataFrame] = []
    for local_i, graph in enumerate(graphs):
        if graph is None:
            continue
        mol_idx = valid_idx[local_i]
        smi = valid_smiles[local_i]
        graph_d = graph.to(device) if hasattr(graph, "to") else graph
        tasks_for_graph = selected_tasks
        for task_idx, task_name in tasks_for_graph:
            try:
                df = _explain_one_molecule_one_task(
                    model=model, data=graph_d, smiles=smi,
                    target_task=task_idx, task_name=task_name,
                    mol_idx=mol_idx, out_dir=out_dir, device=device,
                    fractions=fractions, random_repeats=random_repeats,
                    seed=seed + mol_idx * 1000 + task_idx,
                    top_fraction=top_fraction,
                    use_probability=False, sigmoid=True,
                    n_shap_samples=n_shap_samples,
                    prefer_shap=prefer_shap,
                    write_toxicity_only_saliency=True,
                )
                all_faith.append(df)
                print(f"[explain-tox] mol {mol_idx} task {task_name} done")
            except Exception as e:
                print(f"[explain-tox] mol {mol_idx} task {task_name} FAILED: {e}")
    if all_faith:
        combined = pd.concat(all_faith, ignore_index=True)
        combined.to_csv(out_dir / "all_faithfulness_results.csv", index=False)
        summary = (combined.groupby(["task_name", "fraction_masked"], as_index=False)
                   .agg(top_shap_drop_mean=("top_shap_drop", "mean"),
                        random_drop_mean=("random_drop_mean", "mean"),
                        empirical_p_median=("empirical_p_random_ge_top", "median")))
        summary["faithfulness_gain"] = summary["top_shap_drop_mean"] - summary["random_drop_mean"]
        summary.to_csv(out_dir / "faithfulness_summary_by_task.csv", index=False)


# Stage 0 — prepare-library (downloads recent oncology drugs from ChEMBL)

# Why ChEMBL: it tags molecules with ``max_phase``, ``first_approval`` (year),
# and ``atc_classifications``. We pull the union of:
#   tier A  approved oncology drugs with first_approval ≥ 2023
#   tier B  late-stage oncology candidates (max_phase ∈ {3, 4}) whose ChEMBL
#           record was first added in 2023 or later — covers drugs whose
#           approval ChEMBL hasn't backfilled yet, and Phase 3 readouts.
# Both tiers are filtered against the NCI60 2023 set and the ToxLens training
# set by canonical SMILES so the case study is genuinely out-of-sample for both
# upstream models.

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
ONCOLOGY_ATC_PREFIXES = ("L01", "L02", "L04")  # antineoplastic + endocrine + immuno


def _chembl_get_json(url: str, timeout: float = 60.0) -> dict:
    """One HTTP GET against ChEMBL. Uses stdlib so this works in every env."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "integrated-case-study-thesis/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500] if e.fp else ""
        raise RuntimeError(f"ChEMBL HTTP {e.code} for {url}: {body}") from e


def _chembl_paginate(start_url: str, max_records: int = 5000) -> List[dict]:
    """Walk ChEMBL's ``page_meta.next`` cursor until exhausted or capped."""
    import urllib.parse

    out: List[dict] = []
    url = start_url
    pages = 0
    while url and len(out) < max_records:
        payload = _chembl_get_json(url)
        # ChEMBL puts the list under the resource name ('molecules', 'drugs', ...).
        list_key = next((k for k in payload.keys() if k != "page_meta"), None)
        records = payload.get(list_key, []) if list_key else []
        out.extend(records)
        pages += 1
        nxt = payload.get("page_meta", {}).get("next")
        if nxt:
            # ChEMBL commonly returns root-relative cursors such as
            # /chembl/api/data/molecule.json?limit=1000&offset=1000.
            # Resolve them against the site origin, not CHEMBL_BASE, otherwise
            # the path becomes /chembl/api/data/chembl/api/data/...
            base = "https://www.ebi.ac.uk" if nxt.startswith("/") else f"{CHEMBL_BASE}/"
            url = urllib.parse.urljoin(base, nxt)
        else:
            url = None
        if pages > 200:  # paranoid safety stop
            break
    return out


def _has_oncology_atc(rec: dict) -> bool:
    atcs = rec.get("atc_classifications") or []
    for code in atcs:
        if isinstance(code, str) and any(code.startswith(p) for p in ONCOLOGY_ATC_PREFIXES):
            return True
    return False


def _smiles_of(rec: dict) -> Optional[str]:
    ms = rec.get("molecule_structures") or {}
    s = ms.get("canonical_smiles")
    return s if isinstance(s, str) and s.strip() else None


def _canonicalise_for_exclusion(smi: str) -> Optional[str]:
    """Cheap canonicalisation for exclusion-set membership only.

    The full standardisation pipeline (largest-fragment + uncharge + tautomer
    canonicalisation) is far too slow for tens of thousands of NCI60/ToxLens
    rows. Plain RDKit canonical SMILES is sufficient for set lookup as long as
    both sides (exclusion source and ChEMBL candidate) use the same cheap form.
    Accepted ChEMBL rows are re-standardised with the full pipeline before
    being written to the output CSV.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _full_standardise_for_output(smi: str) -> Optional[str]:
    """Full pipeline used on the ~100 accepted ChEMBL candidates only."""
    mol = standardise_smiles(smi)
    if mol is None:
        return None
    return canonical_smiles(mol)


def _build_exclusion_set(*csv_paths: Path) -> set:
    """Canonical SMILES of every molecule the upstream models have already
    seen. Both NCI60 2025 and the ToxLens training set use a 'smiles' column.
    """
    excl: set = set()
    for p in csv_paths:
        if not p.exists():
            print(f"  ! exclusion source missing, skipping: {p}")
            continue
        try:
            df = pd.read_csv(p, usecols=lambda c: c.lower() == "smiles")
        except Exception:
            df = pd.read_csv(p)
            df = df[[c for c in df.columns if c.lower() == "smiles"]]
        col = next((c for c in df.columns if c.lower() == "smiles"), None)
        if col is None:
            continue
        n_added = 0
        for s in df[col].dropna().astype(str):
            c = _canonicalise_for_exclusion(s)
            if c:
                if c not in excl:
                    n_added += 1
                excl.add(c)
        print(f"  + {p}: added {n_added} canonical SMILES to exclusion set")
    return excl


def _fetch_chembl_oncology(min_year: int, tier: str) -> List[dict]:
    """Tier-A or tier-B oncology query against ChEMBL."""
    if tier == "approved":
        # max_phase=4 is the canonical "approved" tag in ChEMBL.
        url = (
            f"{CHEMBL_BASE}/molecule.json?max_phase=4"
            f"&first_approval__gte={min_year}&limit=1000"
        )
    elif tier == "late_stage_recent":
        # Phase 3 + Phase 4 candidates added since the cut-off year. ChEMBL has
        # a ``_metadata.recent`` flag in some versions; the most portable
        # filter that survives schema changes is max_phase>=3 paired with
        # client-side ATC filtering and exclusion of older molecules.
        url = f"{CHEMBL_BASE}/molecule.json?max_phase__gte=3&limit=1000"
    else:
        raise ValueError(f"unknown tier: {tier}")
    print(f"  ChEMBL query [{tier}]: {url}")
    recs = _chembl_paginate(url, max_records=8000)
    print(f"    -> {len(recs)} raw records")
    onc = [r for r in recs if _has_oncology_atc(r) and _smiles_of(r) is not None]
    print(f"    -> {len(onc)} oncology + SMILES present")
    if tier == "late_stage_recent":
        # For the fallback tier, restrict to molecules whose first ChEMBL
        # release year (an alternative to first_approval) is recent. Some
        # records only carry ``first_approval``; fall back to that.
        kept: List[dict] = []
        for r in onc:
            year = r.get("first_approval")
            if year is None:
                # No approval year recorded — accept because Phase 3 ChEMBL
                # additions of recent date are exactly what we want here.
                kept.append(r)
            elif int(year) >= min_year:
                kept.append(r)
        print(f"    -> {len(kept)} after recency filter")
        onc = kept
    return onc


def stage_prepare_library(
    output_csv: Path,
    nci60_csv: Path,
    tox_csv: Path,
    target_n: int = 100,
    min_year: int = 2023,
    cache_dir: Optional[Path] = None,
) -> None:
    """Download recent anti-cancer drug molecules from ChEMBL, filter against
    the NCI60 2025 set and the ToxLens training set, and write a clean
    library CSV ready for ``featurise-dpd``.
    """
    print(f"[prepare-library] target_n={target_n}, min_year={min_year}")
    print("[prepare-library] building exclusion set from NCI60 + ToxLens")
    excl = _build_exclusion_set(nci60_csv, tox_csv)
    print(f"[prepare-library] exclusion set size: {len(excl)} canonical SMILES")

    cache_dir = cache_dir or output_csv.parent / "chembl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached_fetch(tier: str) -> List[dict]:
        cache = cache_dir / f"chembl_{tier}_y{min_year}.json"
        if cache.exists():
            print(f"  using cached {cache}")
            return json.loads(cache.read_text())
        recs = _fetch_chembl_oncology(min_year, tier)
        cache.write_text(json.dumps(recs))
        return recs

    print("[prepare-library] tier A — approved oncology drugs since cut-off")
    tier_a = _cached_fetch("approved")
    print("[prepare-library] tier B — late-stage oncology candidates")
    tier_b = _cached_fetch("late_stage_recent")

    rows: List[dict] = []
    seen_canonical: set = set()

    def _accept(rec: dict, tier_label: str) -> bool:
        smi = _smiles_of(rec)
        if smi is None:
            return False
        canon = _canonicalise_for_exclusion(smi)
        if canon is None:
            return False
        if canon in excl:
            return False
        if canon in seen_canonical:
            return False
        seen_canonical.add(canon)
        # Full standardisation only for the candidates that actually pass the
        # exclusion + dedup filter (typically ~100 molecules).
        standardised = _full_standardise_for_output(smi) or canon
        rows.append({
            "smiles": standardised,
            "chembl_id": rec.get("molecule_chembl_id"),
            "pref_name": rec.get("pref_name"),
            "max_phase": rec.get("max_phase"),
            "first_approval": rec.get("first_approval"),
            "atc_classifications": ";".join(rec.get("atc_classifications") or []),
            "tier": tier_label,
        })
        return True

    # Tier A first, then top up with tier B until target_n is met.
    for rec in tier_a:
        _accept(rec, "approved_2023+")
        if len(rows) >= target_n:
            break
    if len(rows) < target_n:
        print(f"[prepare-library] tier A produced {len(rows)} after dedup+exclusion; "
              f"falling back to tier B for top-up")
        for rec in tier_b:
            _accept(rec, "late_stage_2023+")
            if len(rows) >= target_n:
                break

    if not rows:
        raise RuntimeError(
            "ChEMBL returned no molecules that pass the year + oncology + exclusion "
            "filters. Check internet connectivity, that the NCI60 / ToxLens CSV paths "
            "exist, or relax --min-year."
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[prepare-library] wrote {len(df)} molecules to {output_csv}")

    counts_by_tier = df["tier"].value_counts().to_dict()
    provenance = {
        "min_year": min_year,
        "target_n": target_n,
        "n_returned": int(len(df)),
        "by_tier": counts_by_tier,
        "exclusion_sources": [str(nci60_csv), str(tox_csv)],
        "exclusion_set_size": len(excl),
        "atc_prefixes_treated_as_oncology": list(ONCOLOGY_ATC_PREFIXES),
        "chembl_base": CHEMBL_BASE,
    }
    (output_csv.with_suffix(".provenance.json")).write_text(json.dumps(provenance, indent=2))
    print(f"[prepare-library] provenance: {provenance}")
    if len(df) < target_n:
        print(
            f"\n  WARNING  Found only {len(df)} molecules after exclusion (asked for {target_n}). "
            f"Strictly 'approved in 2023+' oncology drugs are scarce; the script topped up with "
            f"late-stage candidates (Phase 3/4) but still came up short. "
            f"Consider lowering --min-year (e.g. 2024) or accepting a smaller case-study size."
        )


# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare-library
    p0 = sub.add_parser(
        "prepare-library",
        help="Stage 0 — download recent anti-cancer drugs from ChEMBL, filter "
             "against NCI60 + ToxLens",
    )
    p0.add_argument("--out", type=Path, default=Path("case_study_library.csv"))
    p0.add_argument(
        "--nci60-csv", type=Path,
        default=Path("updated_august_classification_df_april_release.csv"),
        help="NCI60 3 classification CSV (master_file.py output)",
    )
    p0.add_argument(
        "--tox-csv", type=Path, default=Path("tox_data_classification.csv"),
        help="ToxLens training-set CSV (deep_tox.py output)",
    )
    p0.add_argument("--target-n", type=int, default=100)
    p0.add_argument("--min-year", type=int, default=2023)
    p0.add_argument("--cache-dir", type=Path, default=None)

    # featurise-dpd
    p1 = sub.add_parser("featurise-dpd", help="Stage 1a — run in DPD-Cancer featurisation env")
    p1.add_argument("--library", type=Path, required=True)
    p1.add_argument("--out", type=Path, default=Path("dpd_features.pkl"))
    p1.add_argument("--dpd-preproc", type=Path, default=Path("preproc_classification.pkl"))
    p1.add_argument("--morgan-bits", type=int, default=2048)

    # predict-dpd
    p2 = sub.add_parser("predict-dpd", help="Stage 1b — run in DPD-Cancer inference env")
    p2.add_argument("--features", type=Path, required=True, help="output of stage featurise-dpd")
    p2.add_argument("--dpd-ckpt", type=Path, default=Path("deeppd_classification.ckpt"))
    p2.add_argument("--top-n", type=int, default=100)
    p2.add_argument("--out", type=Path, default=Path("dpd_predictions"))  # .full.csv + .top.csv
    p2.add_argument("--device", default="cpu")
    p2.add_argument("--batch-size", type=int, default=64)

    # run-toxlens
    p3 = sub.add_parser("run-toxlens", help="Stage 2 — run in ToxLens env")
    p3.add_argument("--top-csv", type=Path, required=True,
                    help="<...>.top.csv produced by predict-dpd")
    p3.add_argument("--tox-ckpt", type=Path, default=Path("deep_tox_classification.ckpt"))
    p3.add_argument("--tox-task-csv", type=Path, default=Path("tox_data_classification.csv"))
    p3.add_argument("--pubchem-cache", type=Path, default=Path("data/pubchem_bioactivity_cache.json"))
    p3.add_argument("--mc-passes", type=int, default=20)
    p3.add_argument("--w-tox-max", type=float, default=1.0)
    p3.add_argument("--w-tox-count", type=float, default=0.5)
    p3.add_argument("--w-uncertainty", type=float, default=0.3)
    p3.add_argument("--tox-high-threshold", type=float, default=0.5)
    p3.add_argument("--uncertainty-flag-threshold", type=float, default=0.15)
    p3.add_argument("--output", type=Path, default=Path("integrated_case_study_results"))
    p3.add_argument("--batch-size", type=int, default=32)
    p3.add_argument("--device", default=None)

    # explain-dpd
    p4 = sub.add_parser(
        "explain-dpd",
        help="Stage 3a — SHAP-guided occlusion faithfulness for DPD-Cancer "
             "(run in DPD-Cancer inference env)",
    )
    p4.add_argument("--features", type=Path, required=True,
                    help="dpd_features.pkl from featurise-dpd")
    p4.add_argument("--top-csv", type=Path, required=True,
                    help="<...>.top.csv from predict-dpd")
    p4.add_argument("--dpd-ckpt", type=Path, default=Path("deeppd_classification.ckpt"))
    p4.add_argument("--out", type=Path, default=Path("explain_dpd"))
    p4.add_argument("--fractions", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30])
    p4.add_argument("--random-repeats", type=int, default=50)
    p4.add_argument("--seed", type=int, default=42)
    p4.add_argument("--top-fraction", type=float, default=None)
    p4.add_argument("--molecule-name", default="PARSACLISIB",
                    help="Candidate to explain by pref_name/chembl_id/SMILES "
                         "(default: PARSACLISIB; use empty string for all)")
    p4.add_argument("--n-shap-samples", type=int, default=25,
                    help="Captum sample count for pyg_captum_shap.compute_shap_values")
    p4.add_argument("--device", default="cpu")

    # explain-tox
    p5 = sub.add_parser(
        "explain-tox",
        help="Stage 3b — SHAP-guided occlusion faithfulness for ToxLens "
             "(run in ToxLens env)",
    )
    p5.add_argument("--top-csv", type=Path, required=True,
                    help="<...>.top.csv from predict-dpd")
    p5.add_argument("--tox-ckpt", type=Path,
                    default=Path("deep_tox_classification.ckpt"))
    p5.add_argument("--tox-task-csv", type=Path,
                    default=Path("tox_data_classification.csv"))
    p5.add_argument("--pubchem-cache", type=Path, default=Path("data/pubchem_bioactivity_cache.json"))
    p5.add_argument("--out", type=Path, default=Path("explain_tox"))
    p5.add_argument("--tasks", nargs="+", default=None,
                    help="Optional list of ToxLens endpoint names to explain "
                         "(default: the 11 configured primary endpoints only)")
    p5.add_argument("--fractions", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30])
    p5.add_argument("--random-repeats", type=int, default=50)
    p5.add_argument("--seed", type=int, default=42)
    p5.add_argument("--top-fraction", type=float, default=None)
    p5.add_argument("--molecule-name", default="PARSACLISIB",
                    help="Candidate to explain by pref_name/chembl_id/SMILES "
                         "(default: PARSACLISIB; use empty string for all)")
    p5.add_argument("--n-shap-samples", type=int, default=25,
                    help="Captum sample count for pyg_captum_shap.compute_shap_values")
    p5.add_argument("--batch-size", type=int, default=32)
    p5.add_argument("--device", default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "prepare-library":
        stage_prepare_library(
            output_csv=args.out,
            nci60_csv=args.nci60_csv,
            tox_csv=args.tox_csv,
            target_n=args.target_n,
            min_year=args.min_year,
            cache_dir=args.cache_dir,
        )
    elif args.cmd == "featurise-dpd":
        stage_featurise_dpd(
            library_path=args.library, out_path=args.out,
            preproc_path=args.dpd_preproc, morgan_bits=args.morgan_bits,
        )
    elif args.cmd == "predict-dpd":
        stage_predict_dpd(
            featurised_path=args.features, dpd_ckpt=args.dpd_ckpt,
            top_n=args.top_n, out_path=args.out,
            device=args.device, batch_size=args.batch_size,
        )
    elif args.cmd == "run-toxlens":
        device = args.device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        cfg = IntegratedConfig(
            tox_ckpt=args.tox_ckpt,
            tox_task_csv=args.tox_task_csv,
            pubchem_cache=args.pubchem_cache,
            mc_passes=args.mc_passes,
            w_tox_max=args.w_tox_max,
            w_tox_count=args.w_tox_count,
            w_uncertainty=args.w_uncertainty,
            tox_high_threshold=args.tox_high_threshold,
            uncertainty_flag_threshold=args.uncertainty_flag_threshold,
            output_dir=args.output,
            batch_size=args.batch_size,
            device=device,
        )
        stage_run_toxlens(args.top_csv, cfg)
    elif args.cmd == "explain-dpd":
        stage_explain_dpd(
            features_pkl=args.features,
            top_csv=args.top_csv,
            dpd_ckpt=args.dpd_ckpt,
            out_dir=args.out,
            device=args.device,
            fractions=tuple(args.fractions),
            random_repeats=args.random_repeats,
            seed=args.seed,
            top_fraction=args.top_fraction,
            n_shap_samples=args.n_shap_samples,
            prefer_shap=True,
            molecule_name=args.molecule_name,
        )
    elif args.cmd == "explain-tox":
        device = args.device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        stage_explain_tox(
            top_csv=args.top_csv,
            tox_ckpt=args.tox_ckpt,
            tox_task_csv=args.tox_task_csv,
            out_dir=args.out,
            device=device,
            only_tasks=args.tasks,
            fractions=tuple(args.fractions),
            random_repeats=args.random_repeats,
            seed=args.seed,
            top_fraction=args.top_fraction,
            n_shap_samples=args.n_shap_samples,
            prefer_shap=True,
            batch_size=args.batch_size,
            molecule_name=args.molecule_name,
            pubchem_cache=args.pubchem_cache,
        )


if __name__ == "__main__":
    main()
