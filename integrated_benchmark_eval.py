#!/usr/bin/env python3
"""
integrated_benchmark_eval.py
============================
A single, sequentially-executed Python script that integrates the external
benchmark RETRIEVAL module (fetch_external_benchmarks.py) with the ToxLens
graph-neural-network module (deep_tox.py), then runs binary-classification
inference and reports the evaluation metrics: accuracy, AUPRC, AUROC, MCC.

Pipeline (one process, no bash wrappers, no parallelism)
--------------------------------------------------------
  STEP 1  Retrieval.  Call fetch_external_benchmarks.fetch_tdc_admet() and
          fetch_moleculenet_tox21() to download + partition the standard
          datasets (TDC ADMET: AMES / LD50_Zhu / hERG; MoleculeNet Tox21
          scaffold split). These write exact-split CSVs under
          ./external_benchmarks/ and a MANIFEST.json.

  STEP 2  Hand-off.  Read those CSVs into in-memory (SMILES, label) tables and
          map each external benchmark to the matching ToxLens task column.

  STEP 3  Model.  Initialise the PyTorch-Geometric model from a trained
          checkpoint via deep_tox.GAT_class.load_from_checkpoint (architecture
          restored from saved hparams).

  STEP 4  Inference.  For every benchmark dataset, featurise the SMILES with the
          training featuriser, run the binary-classification forward loop on the
          graph data, collect per-molecule probabilities for the mapped task.

  STEP 5  Metrics.  Compute accuracy, AUPRC, AUROC, and MCC per benchmark and
          write integrated_eval_metrics.csv.

Usage
-----
  pip install torch torch_geometric pytorch_lightning rdkit pandas numpy \
              scikit-learn PyTDC deepchem requests
  python integrated_benchmark_eval.py \
      --checkpoint /path/to/best.ckpt \
      --train_csv  /path/to/tox_data_classification.csv \
      [--deep_tox deep_tox.py] [--fetch fetch_external_benchmarks.py] \
      [--device cuda] [--out_dir external_benchmarks]

Skips (missing PyTDC / DeepChem, or under-powered subsets) are reported and
never fatal; the script proceeds to whatever benchmarks are available.
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Decision threshold for the accuracy/MCC point metrics (AUROC/AUPRC are
# threshold-free). Overridable per task via saved thresholds if available.
DEFAULT_THRESHOLD = 0.5

# Map each retrieved benchmark to the ToxLens task column it should be scored on.
# TDC ADMET CSVs use the column 'Y'; MoleculeNet Tox21 CSVs use the task name.
TDC_TASK_MAP = {
    "AMES": "Ames",
    "LD50_Zhu": "LD50_Zhu",      # regression source; see note in code
    "hERG": "hERG_Karim",        # NOTE: AstraZeneca hERG != hERG_Karim dataset
}
TOX21_TASKS = [
    "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "SR-ARE", "SR-HSE", "SR-MMP", "SR-p53",
]


# ---------------------------------------------------------------------------
# Module import helpers
# ---------------------------------------------------------------------------
def import_by_path(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found at {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# STEP 1: retrieval
# ---------------------------------------------------------------------------
def run_retrieval(fetch_mod, out_dir: Path) -> Path:
    """Execute the fetch module's retrieval functions sequentially."""
    # The fetch module writes under its module-level OUT dir; point it at ours.
    fetch_mod.OUT = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[step1] Retrieving TDC ADMET benchmarks ...")
    try:
        fetch_mod.fetch_tdc_admet()
    except Exception as e:
        print(f"[step1] TDC retrieval error (continuing): {e}")
    print("[step1] Retrieving MoleculeNet Tox21 scaffold split ...")
    try:
        fetch_mod.fetch_moleculenet_tox21()
    except Exception as e:
        print(f"[step1] Tox21 retrieval error (continuing): {e}")
    # Persist the manifest the fetch module accumulated.
    try:
        fetch_mod.record_explicit_skips()
        import json
        (out_dir / "MANIFEST.json").write_text(
            json.dumps(fetch_mod.MANIFEST, indent=2), encoding="utf-8")
    except Exception:
        pass
    return out_dir


# ---------------------------------------------------------------------------
# STEP 2: load retrieved CSVs into (smiles, label, task) records
# ---------------------------------------------------------------------------
def collect_datasets(out_dir: Path) -> List[Dict]:
    """Return a list of {benchmark, task, smiles[], y[]} from retrieved CSVs."""
    datasets: List[Dict] = []

    # --- TDC ADMET test splits ---
    for bench, task in TDC_TASK_MAP.items():
        test_csv = out_dir / "tdc_admet" / bench / "test.csv"
        if not test_csv.is_file():
            print(f"[step2] {bench}: no test.csv (retrieval skipped); omit.")
            continue
        df = pd.read_csv(test_csv)
        smi_col = "Drug" if "Drug" in df.columns else (
            "smiles" if "smiles" in df.columns else None)
        y_col = "Y" if "Y" in df.columns else None
        if smi_col is None or y_col is None:
            print(f"[step2] {bench}: unexpected columns {list(df.columns)}; omit.")
            continue
        y = pd.to_numeric(df[y_col], errors="coerce")
        # LD50_Zhu is regression-valued; binary metrics are undefined on it.
        if bench == "LD50_Zhu" and set(np.unique(y.dropna())) - {0, 1}:
            print(f"[step2] {bench}: regression target (non-binary); "
                  "omitting from binary-classification metrics.")
            continue
        datasets.append({
            "benchmark": f"TDC:{bench}", "task": task,
            "smiles": df[smi_col].astype(str).tolist(),
            "y": y.to_numpy(dtype=float),
            "note": "AstraZeneca hERG != hERG_Karim" if bench == "hERG" else "",
        })
        print(f"[step2] TDC:{bench} -> task {task}: N={len(df)}")

    # --- MoleculeNet Tox21 test split (+ mask) ---
    tdir = out_dir / "moleculenet_tox21"
    test_csv, mask_csv = tdir / "test.csv", tdir / "mask_test.csv"
    if test_csv.is_file():
        df = pd.read_csv(test_csv)
        mask = pd.read_csv(mask_csv) if mask_csv.is_file() else None
        smi_col = "smiles" if "smiles" in df.columns else df.columns[0]
        for task in TOX21_TASKS:
            if task not in df.columns:
                continue
            valid = (mask[task] > 0) if (mask is not None and task in mask) \
                else df[task].notna()
            sub = df.loc[valid, [smi_col, task]].dropna()
            if len(sub) == 0:
                continue
            datasets.append({
                "benchmark": f"MoleculeNet:{task}", "task": task,
                "smiles": sub[smi_col].astype(str).tolist(),
                "y": sub[task].to_numpy(dtype=float), "note": "",
            })
            print(f"[step2] MoleculeNet:{task}: N={len(sub)} (mask applied)")
    else:
        print("[step2] MoleculeNet Tox21 test.csv absent (DeepChem missing?); omit.")

    return datasets


# ---------------------------------------------------------------------------
# STEP 3: model
# ---------------------------------------------------------------------------
def init_model(dt, checkpoint: Path, device: str):
    import torch
    if not hasattr(dt, "GAT_class"):
        raise AttributeError("deep_tox module exposes no GAT_class.")
    print(f"[step3] Loading PyG model from checkpoint: {checkpoint}")
    model = dt.GAT_class.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval().to(device)
    return model


def resolve_target_columns(dt, model, train_df: pd.DataFrame) -> List[str]:
    for attr in ("target_columns", "task_names"):
        v = getattr(model, attr, None)
        if v:
            return list(v)
    hp = getattr(model, "hparams", {}) or {}
    for k in ("target_columns", "task_names"):
        if hp.get(k):
            return list(hp[k])
    for name in ("TARGET_COLUMNS", "PRIMARY_TASKS", "TASK_NAMES"):
        v = getattr(dt, name, None)
        if v:
            return list(v)
    panel = ["Ames", "LD50_Zhu", "hERG_Karim", "NR-AhR", "NR-Aromatase", "NR-ER",
             "NR-ER-LBD", "SR-ARE", "SR-HSE", "SR-MMP", "SR-p53"]
    present = [c for c in panel if c in train_df.columns]
    if not present:
        raise RuntimeError("Cannot resolve target_columns.")
    print("[step3] WARNING: target_columns inferred from CSV panel.")
    return present


# ---------------------------------------------------------------------------
# STEP 4: featurise + inference for one benchmark
# ---------------------------------------------------------------------------
def _looks_like_pyg_data(obj) -> bool:
    """True if obj resembles a torch_geometric Data object (has x + edge_index)."""
    return hasattr(obj, "x") and hasattr(obj, "edge_index")


def discover_featuriser(dt, override: Optional[str] = None):
    """Resolve deep_tox's SMILES-list -> PyG Data converter.

    The training-faithful entry point is featurise_smiles_for_inference(
    smiles_list, num_tasks) -> (data_list, kept_idx), which builds the full
    global-descriptor pathway (Morgan + RDKit + Tox-SMARTS + MolFormer + 3D +
    PubChem) internally. We prefer it by name; --featuriser overrides; a probe
    is the last resort.

    Returns (callable, name, takes_list, takes_num_tasks).
    """
    import inspect

    # 1. Explicit override or the canonical inference featuriser by name.
    preferred = [override] if override else []
    preferred += ["featurise_smiles_for_inference", "featurize_smiles_for_inference"]
    for name in preferred:
        if not name:
            continue
        fn = getattr(dt, name, None)
        if callable(fn):
            params = []
            try:
                params = list(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                pass
            takes_num_tasks = any("task" in p.lower() for p in params)
            print(f"[step4]   using featuriser deep_tox.{name}{tuple(params)}")
            return fn, name, True, takes_num_tasks

    # 2. Fallback: probe callables on a benign SMILES and a singleton list.
    PROBE = "CCO"
    hint = ("smiles", "featuri", "mol_to", "to_data", "to_graph", "to_pyg",
            "graph_representation", "build_data", "make_data")
    names = [n for n in dir(dt) if not n.startswith("_")]
    ordered = [n for n in names if any(h in n.lower() for h in hint)]
    for name in ordered:
        fn = getattr(dt, name, None)
        if not callable(fn) or inspect.isclass(fn):
            continue
        for args, takes_list in ((([PROBE], 11), True), ((PROBE,), False)):
            try:
                out = fn(*args)
            except Exception:
                continue
            cand = out[0] if isinstance(out, (tuple, list)) and out else out
            if isinstance(cand, (list, tuple)) and cand:
                cand = cand[0]
            if _looks_like_pyg_data(cand):
                print(f"[step4]   auto-discovered featuriser: deep_tox.{name}")
                return fn, name, takes_list, (len(args) > 1)

    raise RuntimeError(
        "Could not resolve a SMILES->PyG featuriser in deep_tox. Re-run with "
        "--featuriser featurise_smiles_for_inference (or the correct name).")


def featurise_smiles(featuriser, fname: str, takes_list: bool,
                     takes_num_tasks: bool, smiles: List[str], num_tasks: int):
    """Apply the resolved featuriser to a benchmark's SMILES list.

    Handles the canonical (smiles_list, num_tasks) -> (data_list, kept_idx)
    contract, plus simpler list/single-molecule fallbacks. Returns
    (data_list, kept_idx) where kept_idx indexes into `smiles`.
    """
    if takes_list:
        out = featuriser(smiles, num_tasks) if takes_num_tasks else featuriser(smiles)
        if (isinstance(out, tuple) and len(out) == 2
                and _is_index_like(out[1])):
            data_list, kept = list(out[0]), list(np.asarray(out[1]).tolist())
        else:
            data_list = list(out[0]) if isinstance(out, (tuple, list)) else list(out)
            kept = list(range(len(data_list)))
        data_list = [d[0] if isinstance(d, (tuple, list)) else d for d in data_list]
        data_list = [d for d in data_list if _looks_like_pyg_data(d)]
        if not data_list:
            raise RuntimeError(f"deep_tox.{fname} returned no usable graphs.")
        print(f"[step4]   featurised {len(data_list)}/{len(smiles)} via "
              f"deep_tox.{fname}")
        return data_list, kept

    # Per-molecule fallback.
    data_list, kept = [], []
    for i, s in enumerate(smiles):
        try:
            d = featuriser(s)
            d = d[0] if isinstance(d, (tuple, list)) and d else d
            if _looks_like_pyg_data(d):
                data_list.append(d)
                kept.append(i)
        except Exception:
            pass
    if not data_list:
        raise RuntimeError(f"deep_tox.{fname} produced no usable graphs.")
    print(f"[step4]   featurised {len(data_list)}/{len(smiles)} via deep_tox.{fname}")
    return data_list, kept


def _is_index_like(obj) -> bool:
    try:
        arr = np.asarray(obj)
        return arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)
    except Exception:
        return False


def run_inference(dt, model, data_list, task_idx: int, device: str) -> np.ndarray:
    """Binary-classification inference loop over the graph data."""
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(data_list, batch_size=128, shuffle=False)
    probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(data=batch) if _accepts_data_kw(model) else model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                edge_attr=getattr(batch, "edge_attr", None),
                global_features=getattr(batch, "global_features", None))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits[:, task_idx] if logits.dim() == 2 else logits
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def _accepts_data_kw(model) -> bool:
    import inspect
    try:
        return "data" in inspect.signature(model.forward).parameters
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# STEP 5: metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 average_precision_score, matthews_corrcoef)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) \
        if len(np.unique(y_true)) > 1 else float("nan")
    # AUROC / AUPRC require both classes present.
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Main sequential driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--train_csv", required=True, type=Path)
    ap.add_argument("--deep_tox", default=Path("deep_tox.py"), type=Path)
    ap.add_argument("--fetch", default=Path("fetch_external_benchmarks.py"), type=Path)
    ap.add_argument("--out_dir", default=Path("external_benchmarks"), type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--featuriser", default=None,
                    help="Name of the deep_tox SMILES->PyG Data function. If "
                         "omitted, it is auto-discovered by introspection.")
    args = ap.parse_args()

    # Device fallback.
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("[main] CUDA unavailable; using CPU.")
            args.device = "cpu"
    except Exception as e:
        print(f"[main] torch import failed: {e}")
        return 1

    # Import the two modules to integrate.
    fetch_mod = import_by_path("fetch_external_benchmarks", args.fetch)
    dt = import_by_path("deep_tox", args.deep_tox)

    # STEP 1: retrieve.
    out_dir = run_retrieval(fetch_mod, args.out_dir)

    # STEP 2: collect datasets + map to tasks.
    datasets = collect_datasets(out_dir)
    if not datasets:
        print("[main] No datasets retrieved (install PyTDC/DeepChem). Nothing to do.")
        return 0

    # STEP 3: model + task order.
    train_df = pd.read_csv(args.train_csv)
    model = init_model(dt, args.checkpoint, args.device)
    target_columns = resolve_target_columns(dt, model, train_df)
    print(f"[main] target_columns ({len(target_columns)}): {target_columns}")

    # Discover the featuriser ONCE (auto or via --featuriser).
    try:
        featuriser, fname, takes_list, takes_num_tasks = discover_featuriser(
            dt, args.featuriser)
    except Exception as e:
        print(f"[main] {e}")
        return 1
    num_tasks = len(target_columns)

    # STEP 4 + 5: per-benchmark inference and metrics.
    rows = []
    for ds in datasets:
        task = ds["task"]
        if task not in target_columns:
            print(f"[main] {ds['benchmark']}: task '{task}' not in model; skip.")
            continue
        task_idx = target_columns.index(task)
        print(f"\n[main] === {ds['benchmark']}  (task {task}, idx {task_idx}) ===")
        try:
            data_list, kept = featurise_smiles(
                featuriser, fname, takes_list, takes_num_tasks,
                ds["smiles"], num_tasks)
        except Exception as e:
            print(f"[main]   featurisation failed: {e}; skip.")
            continue
        y_true = ds["y"][kept]
        if len(y_true) == 0:
            print("[main]   no featurisable molecules; skip.")
            continue
        y_prob = run_inference(dt, model, data_list, task_idx, args.device)
        n = min(len(y_true), len(y_prob))
        m = compute_metrics(y_true[:n], y_prob[:n])
        m.update({"benchmark": ds["benchmark"], "task": task,
                  "n": int(n), "note": ds.get("note", "")})
        rows.append(m)
        print(f"[main]   accuracy={m['accuracy']:.4f}  AUROC={m['auroc']:.4f}  "
              f"AUPRC={m['auprc']:.4f}  MCC={m['mcc']:.4f}  (N={n})")

    if not rows:
        print("\n[main] No benchmark produced metrics.")
        return 0

    res = pd.DataFrame(rows)[
        ["benchmark", "task", "n", "accuracy", "auroc", "auprc", "mcc", "note"]]
    out_csv = out_dir / "integrated_eval_metrics.csv"
    res.to_csv(out_csv, index=False)
    print("\n[main] ===== Evaluation metrics =====")
    with pd.option_context("display.width", 160):
        print(res.to_string(index=False))
    print(f"\n[main] Saved -> {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
