#!/usr/bin/env python3
"""
render_figure5_toxicophores.py
==============================
Render Figure 5: the top-ranked counterfactually-validated consensus toxicophore
for each of eight endpoints, each embedded in a representative test-set positive
molecule that contains it, with the motif atoms highlighted in red.

This produces the figure the manuscript caption actually describes (motif-in-
context, red highlight), which the per-task fragment grids from
auto_toxico_disc_algo.render_top_toxicophores do NOT (those draw bare fragments).

Inputs:
  --toxicophores  all_tasks_reported_toxicophores.csv  (the motif catalogue)
  --test_csv      the evaluation test set with columns [SMILES, <endpoint>...],
                  used to pick a representative positive molecule per motif.
                  REQUIRED because the catalogue stores only the fragment SMILES,
                  not the parent molecules.
  --out           Figure5_toxicophores.svg (and .png if rdkit cairo available)

Endpoint -> panel order (a-h) follows the manuscript caption:
  SR-MMP, hERG_Karim, NR-AhR, LD50_Zhu, NR-ER, SR-ARE, SR-p53, Ames

Per panel the script:
  1. takes the top validated motif for that endpoint
     (rank: validated_fraction desc, mean_true_drop desc, n_positive_molecules desc),
  2. finds positive molecules in the test set that contain the motif as a
     substructure (SMARTS match on the fragment),
  3. picks the smallest such molecule (clearest depiction) as the representative,
  4. draws it with the matched motif atoms/bonds highlighted in red.

Usage:
  python render_figure5_toxicophores.py \
      --toxicophores all_tasks_reported_toxicophores.csv \
      --test_csv tox_data_classification_test.csv \
      --out Figure5_toxicophores.svg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

PANEL_ORDER = ["SR-MMP", "hERG_Karim", "NR-AhR", "LD50_Zhu",
               "NR-ER", "SR-ARE", "SR-p53", "Ames"]
PANEL_LETTER = list("abcdefgh")
RED = (0.85, 0.10, 0.10)


def fragment_to_query(frag_smiles: str):
    """Build a substructure query from a (possibly partial) fragment SMILES."""
    q = Chem.MolFromSmarts(frag_smiles)
    if q is not None:
        return q
    m = Chem.MolFromSmiles(frag_smiles, sanitize=False)
    if m is not None:
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
    return m


def pick_representative(frag_smiles, test_df, smi_col, endpoint):
    """Return (mol, match_atom_idxs) for the smallest positive test molecule
    that contains the motif; None if no match found."""
    query = fragment_to_query(frag_smiles)
    if query is None:
        return None
    # positive molecules for this endpoint
    if endpoint not in test_df.columns:
        pos = test_df
    else:
        pos = test_df[pd.to_numeric(test_df[endpoint], errors="coerce") == 1]
    best = None
    for s in pos[smi_col].astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        match = mol.GetSubstructMatch(query)
        if match:
            size = mol.GetNumHeavyAtoms()
            if best is None or size < best[2]:
                best = (mol, list(match), size)
    if best is None:
        return None
    return best[0], best[1]


def draw_panel(drawer, mol, atom_idxs, title):
    bond_idxs = []
    aset = set(atom_idxs)
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset:
            bond_idxs.append(b.GetIdx())
    AllChem.Compute2DCoords(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atom_idxs,
        highlightBonds=bond_idxs,
        highlightAtomColors={i: RED for i in atom_idxs},
        highlightBondColors={i: RED for i in bond_idxs},
        legend=title,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toxicophores", required=True, type=Path)
    ap.add_argument("--test_csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("Figure5_toxicophores.svg"))
    a = ap.parse_args()

    tox = pd.read_csv(a.toxicophores)
    tox = tox[tox["selection_basis"] == "counterfactual_validated"].copy()
    test_df = pd.read_csv(a.test_csv)
    smi_col = next((c for c in ("SMILES", "smiles", "Drug", "canonical_smiles")
                    if c in test_df.columns), None)
    if smi_col is None:
        raise SystemExit(f"No SMILES column in {a.test_csv}; "
                         f"columns: {list(test_df.columns)[:10]}")

    # top motif per endpoint
    panels = []
    for ep in PANEL_ORDER:
        sub = tox[tox["task_name"] == ep]
        if sub.empty:
            print(f"[fig5] {ep}: no validated motif; panel will be blank.")
            panels.append((ep, None, None, None))
            continue
        sub = sub.sort_values(
            ["validated_fraction", "mean_true_drop", "n_positive_molecules"],
            ascending=[False, False, False])
        top = sub.iloc[0]
        frag = top["representative_smiles"]
        rep = pick_representative(frag, test_df, smi_col, ep)
        if rep is None:
            print(f"[fig5] {ep}: motif '{frag}' not found in any positive test "
                  "molecule; panel will show the fragment alone.")
            m = fragment_to_query(frag)
            panels.append((ep, m, list(range(m.GetNumAtoms())) if m else None, frag))
        else:
            mol, idxs = rep
            panels.append((ep, mol, idxs, frag))

    # 4x2 grid
    n = len(panels)
    cols, rows = 4, 2
    pw, ph = 330, 300
    drawer = rdMolDraw2D.MolDraw2DSVG(cols * pw, rows * ph, pw, ph)
    opts = drawer.drawOptions()
    opts.legendFontSize = 18
    opts.padding = 0.10

    mols, legends, highlights = [], [], []
    for (ep, mol, idxs, frag), letter in zip(panels, PANEL_LETTER):
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(f"({letter}) {ep}: {frag}")
        highlights.append(idxs or [])

    # DrawMolecules with per-mol highlight atoms
    bond_h = []
    for mol, idxs in zip(mols, highlights):
        aset = set(idxs)
        bset = [b.GetIdx() for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset]
        bond_h.append(bset)
    drawer.DrawMolecules(
        mols, legends=legends,
        highlightAtoms=highlights,
        highlightBonds=bond_h,
        highlightAtomColors=[{i: RED for i in idxs} for idxs in highlights],
        highlightBondColors=[{i: RED for i in bs} for bs in bond_h],
    )
    drawer.FinishDrawing()
    a.out.write_text(drawer.GetDrawingText(), encoding="utf-8")
    print(f"[fig5] wrote {a.out} with {len(mols)} panels.")

    # Optional PNG via cairo
    try:
        from rdkit.Chem.Draw import rdMolDraw2D as _d
        png = a.out.with_suffix(".png")
        cd = _d.MolDraw2DCairo(cols * pw, rows * ph, pw, ph)
        cd.drawOptions().legendFontSize = 18
        cd.DrawMolecules(mols, legends=legends,
                         highlightAtoms=highlights, highlightBonds=bond_h,
                         highlightAtomColors=[{i: RED for i in idxs} for idxs in highlights],
                         highlightBondColors=[{i: RED for i in bs} for bs in bond_h])
        cd.FinishDrawing()
        png.write_bytes(cd.GetDrawingText())
        print(f"[fig5] wrote {png}")
    except Exception as e:
        print(f"[fig5] PNG export skipped ({e}); SVG is the deliverable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
