#!/usr/bin/env python3
"""
overlap_audit.py
================
Train/test leakage audit for the external benchmark evaluation.

For every external benchmark test set written by integrated_benchmark_eval.py
under ./external_benchmarks/, this script:

  1. Canonicalises the test SMILES with the SAME canonicaliser as training
     (deep_tox.get_canonical_smiles if importable; RDKit fallback otherwise).
  2. Intersects them against the canonicalised TRAINING SMILES from
     tox_data_classification.csv.
  3. Reports, per benchmark, how many test molecules also appear in training
     (exact molecular-graph overlap after canonicalisation).
  4. Writes a leakage-free copy of each test set (overlapping molecules removed)
     next to the original as test_clean.csv, plus mask_test_clean.csv for Tox21.
  5. Writes overlap_audit_report.csv summarising all benchmarks.

Why canonicalisation matters
----------------------------
Two different SMILES strings can encode the same molecule. A raw string
intersection misses those and under-reports leakage. Using the training
canonicaliser makes the comparison at the level of molecular identity, which
is what "did the model see this molecule in training?" actually means.

NOTE on tautomers/stereo: get_canonical_smiles in deep_tox standardises
(largest fragment, uncharge, tautomer-canonicalise) the same way the model's
featuriser does, so overlap is judged on the standardised structure. The RDKit
fallback is plain isomeric-SMILES canonicalisation (no tautomer merge); if you
must run without deep_tox, treat the fallback overlap count as a LOWER bound.

Usage
-----
  python overlap_audit.py \
      --train_csv tox_data_classification.csv \
      [--deep_tox deep_tox.py] [--ext_dir external_benchmarks]

Then re-run the evaluation on the cleaned sets, e.g. by pointing
integrated_benchmark_eval.py at test_clean.csv, or just read the report to
confirm overlap is zero and cite the originals.
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Canonicaliser: prefer the training one, fall back to plain RDKit.
# ---------------------------------------------------------------------------
def load_canonicaliser(deep_tox_path: Optional[Path]) -> Tuple[Callable, str]:
    """Return (canonicalise_fn, label). canonicalise_fn(str) -> Optional[str]."""
    if deep_tox_path and deep_tox_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("deep_tox", str(deep_tox_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["deep_tox"] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            fn = getattr(mod, "get_canonical_smiles", None)
            if callable(fn):
                def _wrap(s: str) -> Optional[str]:
                    try:
                        c = fn(s)
                        return c if c else None
                    except Exception:
                        return None
                return _wrap, "deep_tox.get_canonical_smiles (training-faithful)"
        except Exception as e:
            print(f"[audit] could not import deep_tox ({e}); using RDKit fallback.")

    # RDKit fallback: standardise to mirror training as closely as we can
    # without deep_tox (largest fragment + uncharge + isomeric canonical SMILES).
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        lfc = rdMolStandardize.LargestFragmentChooser()
        uc = rdMolStandardize.Uncharger()

        def _rdkit(s: str) -> Optional[str]:
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    return None
                m = lfc.choose(m)
                m = uc.uncharge(m)
                Chem.SanitizeMol(m)
                return Chem.MolToSmiles(m, isomericSmiles=True)
            except Exception:
                return None
        return _rdkit, "RDKit fallback (largest-fragment + uncharge; LOWER bound)"
    except Exception as e:
        raise RuntimeError(f"No usable canonicaliser; install rdkit ({e}).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smiles_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ("Drug", "smiles", "SMILES", "canonical_smiles", "mol"):
        if cand in df.columns:
            return cand
    return None


def canon_set(smiles: List[str], canon: Callable) -> Tuple[Set[str], int]:
    """Return (set of canonical SMILES, number that failed to canonicalise)."""
    out, failed = set(), 0
    for s in smiles:
        c = canon(str(s))
        if c:
            out.add(c)
        else:
            failed += 1
    return out, failed


def build_train_set(train_csv: Path, canon: Callable) -> Set[str]:
    df = pd.read_csv(train_csv)
    col = smiles_column(df)
    if col is None:
        raise RuntimeError(
            f"No SMILES column in {train_csv}; columns: {list(df.columns)[:12]}")
    s, failed = canon_set(df[col].astype(str).tolist(), canon)
    print(f"[audit] training molecules: {len(s)} canonical "
          f"(from '{col}', {failed} unparseable) in {train_csv.name}")
    return s


def audit_test_csv(test_csv: Path, train_set: Set[str], canon: Callable,
                   benchmark: str) -> Optional[Dict]:
    """Audit one test CSV; write *_clean.csv with overlaps removed."""
    if not test_csv.is_file():
        return None
    df = pd.read_csv(test_csv)
    col = smiles_column(df)
    if col is None:
        print(f"[audit] {benchmark}: no SMILES column; skip.")
        return None

    # Canonicalise per-row (keep alignment with the dataframe).
    canon_col = [canon(str(s)) for s in df[col].astype(str)]
    df = df.assign(_canon=canon_col)
    n_total = len(df)
    n_unparseable = df["_canon"].isna().sum()

    in_train = df["_canon"].apply(lambda c: bool(c) and c in train_set)
    n_overlap = int(in_train.sum())
    # Also count UNIQUE overlapping molecules (a molecule can repeat in a test set).
    overlap_unique = len(set(df.loc[in_train, "_canon"]) & train_set)

    # Write leakage-free copy.
    clean = df.loc[~in_train].drop(columns=["_canon"])
    clean_path = test_csv.with_name("test_clean.csv")
    clean.to_csv(clean_path, index=False)

    # If a Tox21 mask sits beside it, write the matching cleaned mask.
    mask_path = test_csv.with_name("mask_test.csv")
    if mask_path.is_file():
        mask = pd.read_csv(mask_path)
        if len(mask) == n_total:
            mask.loc[~in_train.values].to_csv(
                test_csv.with_name("mask_test_clean.csv"), index=False)

    pct = (100.0 * n_overlap / n_total) if n_total else 0.0
    print(f"[audit] {benchmark:24s} N={n_total:5d}  overlap={n_overlap:4d} "
          f"({pct:5.2f}%)  unique_overlap={overlap_unique:4d}  "
          f"unparseable={n_unparseable}  -> {clean_path.name} (N={len(clean)})")

    return {
        "benchmark": benchmark,
        "n_test": n_total,
        "n_overlap_rows": n_overlap,
        "n_overlap_unique_molecules": overlap_unique,
        "overlap_pct": round(pct, 3),
        "n_unparseable": int(n_unparseable),
        "n_clean": len(clean),
        "clean_csv": str(clean_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Train/test overlap (leakage) audit "
                                             "for external benchmarks.")
    ap.add_argument("--train_csv", required=True, type=Path)
    ap.add_argument("--deep_tox", default=Path("deep_tox.py"), type=Path)
    ap.add_argument("--ext_dir", default=Path("external_benchmarks"), type=Path)
    args = ap.parse_args()

    if not args.train_csv.is_file():
        print(f"[audit] train_csv not found: {args.train_csv}")
        return 1
    if not args.ext_dir.is_dir():
        print(f"[audit] external benchmark dir not found: {args.ext_dir}. "
              "Run integrated_benchmark_eval.py first.")
        return 1

    canon, label = load_canonicaliser(args.deep_tox)
    print(f"[audit] canonicaliser: {label}")

    train_set = build_train_set(args.train_csv, canon)

    # Discover every external test set the pipeline wrote.
    targets: List[Tuple[str, Path]] = []
    # TDC ADMET: external_benchmarks/tdc_admet/<BENCH>/test.csv
    tdc = args.ext_dir / "tdc_admet"
    if tdc.is_dir():
        for bench_dir in sorted(p for p in tdc.iterdir() if p.is_dir()):
            tc = bench_dir / "test.csv"
            if tc.is_file():
                targets.append((f"TDC:{bench_dir.name}", tc))
    # MoleculeNet Tox21: external_benchmarks/moleculenet_tox21/test.csv
    mol = args.ext_dir / "moleculenet_tox21" / "test.csv"
    if mol.is_file():
        targets.append(("MoleculeNet:Tox21", mol))

    if not targets:
        print(f"[audit] no external test.csv files found under {args.ext_dir}.")
        return 1

    print(f"[audit] auditing {len(targets)} external test set(s) ...\n")
    report = []
    for benchmark, tc in targets:
        row = audit_test_csv(tc, train_set, canon, benchmark)
        if row:
            report.append(row)

    if report:
        rep = pd.DataFrame(report)
        out = args.ext_dir / "overlap_audit_report.csv"
        rep.to_csv(out, index=False)
        print("\n[audit] ===== overlap audit summary =====")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(rep[["benchmark", "n_test", "n_overlap_rows",
                       "n_overlap_unique_molecules", "overlap_pct",
                       "n_clean"]].to_string(index=False))
        print(f"\n[audit] report -> {out.resolve()}")
        total_overlap = rep["n_overlap_rows"].sum()
        if total_overlap == 0:
            print("[audit] RESULT: zero train/test overlap detected. The original "
                  "test sets are leakage-free under this canonicaliser.")
        else:
            print(f"[audit] RESULT: {total_overlap} overlapping test rows removed. "
                  "Re-run evaluation on the test_clean.csv files before citing "
                  "any external number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
