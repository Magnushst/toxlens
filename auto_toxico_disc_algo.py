from __future__ import annotations
import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.cluster import DBSCAN
from shap_and_saliency_vis import (
    compute_heavy_atom_shap,
    mask_heavy_atom_groups,
    model_score,
    prepare_single_graph,
    resolve_latest_classification_checkpoint,
)


PRIMARY_TOXLENS_TASKS = (
    "Ames",
    "LD50_Zhu",
    "hERG_Karim",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "SR-ARE",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def load_case_study_toxlens_helpers():
    case_dir = Path(os.environ.get(
        "TOXLENS_CASE_STUDY_DIR",
        r"C:\Users\magnu\iCloudDrive\bioinformatikk_master\research_project\case_study",
    ))
    if not case_dir.exists():
        raise FileNotFoundError(f"Could not find case-study directory {case_dir}. Set TOXLENS_CASE_STUDY_DIR.")
    case_dir_str = str(case_dir)
    if case_dir_str not in sys.path:
        sys.path.insert(0, case_dir_str)
    from integrated_case_study_thesis import (
        PRIMARY_TOXLENS_TASKS as case_primary_tasks,
        _build_toxlens_model_class,
        _load_primary_toxlens_checkpoint,
    )
    return {
        "task_names": tuple(case_primary_tasks),
        "build_model_class": _build_toxlens_model_class,
        "load_checkpoint": _load_primary_toxlens_checkpoint,
    }


def _std_mol_from_smiles(smiles: str):
    if not isinstance(smiles, str):
        return None
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        mol = AllChem.MolFromSmiles(smiles)
    return mol


def get_task_label(data, target_task: int) -> int | None:
    y = data.y
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    y = np.asarray(y).reshape(-1)
    value = float(y[target_task] if len(y) > 1 else y[0])
    if not np.isfinite(value):
        return None
    return int(value)


def choose_working_mol(std_smiles: str, n_nodes: int):
    base = Chem.MolFromSmiles(std_smiles)
    with_h = Chem.AddHs(base)
    if n_nodes == with_h.GetNumAtoms():
        return base, with_h
    if n_nodes == base.GetNumAtoms():
        return base, base
    raise ValueError(
        f"Node/atom mismatch: graph has {n_nodes} nodes, "
        f"RDKit gives {base.GetNumAtoms()} heavy-only or {with_h.GetNumAtoms()} with Hs."
    )


def absorb_hydrogen_scores(base_mol: Chem.Mol, work_mol: Chem.Mol, node_scores: np.ndarray) -> np.ndarray:
    heavy_scores = np.zeros(base_mol.GetNumAtoms(), dtype=np.float32)
    for atom in work_mol.GetAtoms():
        idx = atom.GetIdx()
        score = float(node_scores[idx])
        if atom.GetAtomicNum() == 1:
            nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if nbrs:
                heavy_scores[nbrs[0]] += score
        else:
            heavy_scores[idx] += score
    return heavy_scores


def environment_atoms(mol: Chem.Mol, root_idx: int, radius: int):
    bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, rootedAtAtom=root_idx))
    atom_ids = {root_idx}
    for bidx in bond_ids:
        bond = mol.GetBondWithIdx(bidx)
        atom_ids.add(bond.GetBeginAtomIdx())
        atom_ids.add(bond.GetEndAtomIdx())
    return sorted(atom_ids)


def fragment_smiles(mol: Chem.Mol, atom_ids, root_idx: int) -> str:
    atom_ids = sorted(set(int(i) for i in atom_ids))
    if root_idx not in atom_ids:
        return ""

    frags = Chem.GetMolFrags(mol)
    rooted = len(frags) == 1
    if len(frags) > 1:
        root_frag = next((set(frag) for frag in frags if root_idx in frag), None)
        if not root_frag:
            return ""
        atom_ids = sorted(set(atom_ids) & root_frag)
        if root_idx not in atom_ids:
            return ""
    kwargs = dict(
        mol=mol,
        atomsToUse=list(atom_ids),
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
    )
    if rooted:
        kwargs["rootedAtAtom"] = int(root_idx)
    return Chem.MolFragmentToSmiles(**kwargs)


def select_seed_atoms(heavy_scores: np.ndarray, min_positive_score: float = 0.20, quantile: float = 0.90, top_k: int = 3):
    pos = heavy_scores[heavy_scores > 0]
    if pos.size == 0:
        return []
    thresh = max(float(np.quantile(pos, quantile)), min_positive_score)
    seed_ids = np.flatnonzero(heavy_scores >= thresh).tolist()
    seed_ids = sorted(seed_ids, key=lambda i: heavy_scores[i], reverse=True)
    return seed_ids[:top_k]


def choose_adaptive_fragment(mol: Chem.Mol, root_idx: int, heavy_scores: np.ndarray, radii=(1, 2), min_local_positive_mass: float = 0.35):
    total_positive = float(np.clip(heavy_scores, 0, None).sum()) + 1e-9
    best_radius, best_atoms = radii[-1], [root_idx]
    for radius in radii:
        atom_ids = environment_atoms(mol, root_idx, radius)
        frac = float(np.clip(heavy_scores[atom_ids], 0, None).sum()) / total_positive
        best_radius, best_atoms = radius, atom_ids
        if frac >= min_local_positive_mass:
            break
    return best_radius, best_atoms


def extract_toxicophore_occurrences(
    model,
    dataset,
    target_task: int = 0,
    positive_label: int = 1,
    device: str = "cuda",
    min_positive_score: float = 0.20,
    seed_quantile: float = 0.90,
    top_k_atoms_per_molecule: int = 3,
    radii=(1, 2),
    min_local_positive_mass: float = 0.35,
    positive_only: bool = True,
):
    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []
    n_missing_label = 0
    n_non_positive = 0
    n_shap_failed = 0

    for mol_idx, data in enumerate(dataset):
        try:
            data = prepare_single_graph(data, torch_device)
            label = get_task_label(data, target_task)
            if label is None:
                n_missing_label += 1
                continue
            if positive_only and label != positive_label:
                n_non_positive += 1
                continue

            parsed_mol = Chem.MolFromSmiles(data.smiles)
            if parsed_mol is None:
                continue

            shap_info = compute_heavy_atom_shap(model, data, target_task=target_task)
            base_mol = shap_info["mol_no_h"]
            std_smiles = Chem.MolToSmiles(base_mol, canonical=True)
            heavy_net = np.asarray(shap_info["heavy_total_attr"], dtype=np.float32)
            heavy_net /= np.max(np.abs(heavy_net)) + 1e-9

            seed_ids = select_seed_atoms(
                heavy_net,
                min_positive_score=min_positive_score,
                quantile=seed_quantile,
                top_k=top_k_atoms_per_molecule,
            )
            if not seed_ids:
                continue

            seen_in_molecule = set()

            for atom_idx in seed_ids:
                radius, atom_ids = choose_adaptive_fragment(
                    base_mol, atom_idx, heavy_net, radii=radii, min_local_positive_mass=min_local_positive_mass
                )
                frag = fragment_smiles(base_mol, atom_ids, atom_idx)
                if not frag or frag in seen_in_molecule:
                    continue
                seen_in_molecule.add(frag)

                rows.append({
                    "fragment_smiles": frag,
                    "source_id": mol_idx,
                    "source_smiles": data.smiles,
                    "source_std_smiles": std_smiles,
                    "label": int(label),
                    "seed_atom_idx": int(atom_idx),
                    "seed_score": float(heavy_net[atom_idx]),
                    "radius": int(radius),
                    "is_positive_class": int(label == positive_label),
                    "fragment_atom_indices": ",".join(map(str, atom_ids)),
                    "n_fragment_atoms": int(len(atom_ids)),
                })
        except Exception as e:
            n_shap_failed += 1
            print(f"[warn] molecule {mol_idx} skipped: {e}")

    print(
        f"[Toxicophore] Label/filter summary for task index {target_task}: "
        f"missing_label={n_missing_label}, non_positive_skipped={n_non_positive}, "
        f"failed_after_filter={n_shap_failed}, occurrences={len(rows)}"
    )
    return pd.DataFrame(rows)


def _summarise_cluster_occurrences(clustered_occ: pd.DataFrame) -> pd.DataFrame:
    if clustered_occ.empty:
        return pd.DataFrame()

    cluster_rows = []
    for cluster_id, sub in clustered_occ.groupby("cluster_id"):
        rep = sub["fragment_smiles"].value_counts().idxmax()
        pos_ids = set(sub.loc[sub["is_positive_class"] == 1, "source_id"])
        method = "unknown"
        if "cluster_method" in sub.columns and not sub["cluster_method"].dropna().empty:
            method = str(sub["cluster_method"].dropna().mode().iloc[0])
        cluster_rows.append({
            "cluster_id": int(cluster_id),
            "cluster_method": method,
            "representative_smiles": rep,
            "n_occurrences": int(len(sub)),
            "n_molecules": int(sub["source_id"].nunique()),
            "n_positive_molecules": int(len(pos_ids)),
            "mean_seed_score": float(sub["seed_score"].mean()),
        })

    return pd.DataFrame(cluster_rows).sort_values(
        ["n_positive_molecules", "mean_seed_score", "n_occurrences"],
        ascending=[False, False, False],
    )


def _exact_repeat_consensus(occurrence_df: pd.DataFrame, min_samples: int = 3):
    if occurrence_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    fragment_stats = (
        occurrence_df.groupby("fragment_smiles")
        .agg(
            n_occurrences=("fragment_smiles", "size"),
            n_molecules=("source_id", "nunique"),
            n_positive_molecules=("source_id", lambda s: occurrence_df.loc[s.index, "is_positive_class"].eq(1).groupby(s).any().sum()),
        )
        .reset_index()
    )
    keep = fragment_stats[
        (fragment_stats["n_occurrences"] >= min_samples)
        & (fragment_stats["n_positive_molecules"] >= min_samples)
    ].copy()
    if keep.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep = keep.sort_values(
        ["n_positive_molecules", "n_occurrences", "fragment_smiles"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    keep["cluster_id"] = np.arange(len(keep), dtype=int)
    keep["cluster_method"] = "exact_repeat"

    frag_df = keep[["fragment_smiles", "cluster_id", "cluster_method"]]
    occ = occurrence_df.merge(frag_df, on="fragment_smiles", how="inner")
    cluster_df = _summarise_cluster_occurrences(occ)
    return cluster_df, occ


def cluster_consensus_toxicophores(
    occurrence_df: pd.DataFrame,
    eps: float = 0.30,
    min_samples: int = 3,
    fp_radius: int = 2,
    fp_size: int = 1024,
):
    if occurrence_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    exact_cluster_df, exact_occ = _exact_repeat_consensus(occurrence_df, min_samples=min_samples)
    exact_fragments = set(exact_occ["fragment_smiles"]) if not exact_occ.empty else set()

    unique_smiles = occurrence_df["fragment_smiles"].drop_duplicates().tolist()
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
    valid_smiles, fps = [], []

    for smi in unique_smiles:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid_smiles.append(smi)
        fps.append(mfpgen.GetFingerprintAsNumPy(mol))

    if not fps:
        return exact_cluster_df, exact_occ

    fp_matrix = np.asarray(fps, dtype=bool)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="jaccard").fit_predict(fp_matrix)

    frag_df = pd.DataFrame({"fragment_smiles": valid_smiles, "cluster_id": labels, "cluster_method": "dbscan"})
    dbscan_occ = occurrence_df.merge(frag_df, on="fragment_smiles", how="inner")
    dbscan_occ = dbscan_occ[dbscan_occ["cluster_id"] != -1].copy()
    if exact_fragments:
        dbscan_occ = dbscan_occ[~dbscan_occ["fragment_smiles"].isin(exact_fragments)].copy()
    if not dbscan_occ.empty:
        offset = int(exact_occ["cluster_id"].max() + 1) if not exact_occ.empty else 0
        label_map = {old: offset + i for i, old in enumerate(sorted(dbscan_occ["cluster_id"].unique()))}
        dbscan_occ["cluster_id"] = dbscan_occ["cluster_id"].map(label_map).astype(int)

    parts = [df for df in [exact_occ, dbscan_occ] if not df.empty]
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    occ = pd.concat(parts, ignore_index=True)

    cluster_df = _summarise_cluster_occurrences(occ)
    return cluster_df, occ


def parse_atom_idx_string(s: str):
    return [int(x) for x in s.split(",") if x.strip()] if isinstance(s, str) and s.strip() else []


def sample_connected_subgraph_same_size(mol: Chem.Mol, size: int, rng, forbidden=None, max_tries: int = 200, max_overlap_frac: float = 0.5):
    forbidden = set() if forbidden is None else set(forbidden)
    n = mol.GetNumAtoms()
    if size <= 0 or size >= n:
        return None

    nbrs = {a.GetIdx(): [b.GetIdx() for b in a.GetNeighbors()] for a in mol.GetAtoms()}
    nodes = list(range(n))

    for _ in range(max_tries):
        start = int(rng.choice(nodes))
        chosen = {start}
        frontier = set(nbrs[start])

        while frontier and len(chosen) < size:
            nxt = int(rng.choice(list(frontier)))
            frontier.remove(nxt)
            if nxt in chosen:
                continue
            chosen.add(nxt)
            frontier.update(nbrs[nxt])
            frontier -= chosen

        if len(chosen) != size:
            continue
        if chosen == forbidden:
            continue
        if len(chosen & forbidden) / max(1, len(forbidden)) > max_overlap_frac:
            continue
        return sorted(chosen)

    return None


def validate_occurrences_counterfactually(
    occurrence_df: pd.DataFrame,
    dataset,
    model,
    target_task: int = 0,
    positive_label: int = 1,
    n_random_controls: int = 20,
    min_score_drop: float = 0.10,
    min_delta_vs_random: float = 0.05,
    max_empirical_p: float = 0.10,
    seed: int = 42,
    device: str = "cuda",
):
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []

    for row in occurrence_df.itertuples(index=False):
        if int(row.label) != positive_label:
            continue

        atom_ids = parse_atom_idx_string(row.fragment_atom_indices)
        if not atom_ids:
            continue

        parent_data = prepare_single_graph(dataset[int(row.source_id)], torch_device)
        parent_mol = Chem.MolFromSmiles(row.source_smiles)
        if parent_mol is None or len(atom_ids) >= parent_mol.GetNumAtoms():
            continue

        try:
            shap_info = compute_heavy_atom_shap(model, parent_data, target_task=target_task)
            heavy_groups = shap_info["heavy_groups"]
            if any(i >= len(heavy_groups) for i in atom_ids):
                continue

            base_score = model_score(
                model,
                parent_data,
                target_task=target_task,
                classification=True,
                use_probability=True,
            )
            masked_data = mask_heavy_atom_groups(
                parent_data,
                heavy_groups,
                atom_ids,
                mask_node_features=True,
                mask_incident_edge_features=True,
            )
            masked_score = model_score(
                model,
                masked_data,
                target_task=target_task,
                classification=True,
                use_probability=True,
            )
            true_drop = float(base_score - masked_score)

            motif_set = set(atom_ids)
            random_drops = []
            for _ in range(n_random_controls):
                ctrl_ids = sample_connected_subgraph_same_size(parent_mol, len(atom_ids), rng, forbidden=motif_set)
                if ctrl_ids is None:
                    continue
                if any(i >= len(heavy_groups) for i in ctrl_ids):
                    continue
                ctrl_data = mask_heavy_atom_groups(
                    parent_data,
                    heavy_groups,
                    ctrl_ids,
                    mask_node_features=True,
                    mask_incident_edge_features=True,
                )
                ctrl_score = model_score(
                    model,
                    ctrl_data,
                    target_task=target_task,
                    classification=True,
                    use_probability=True,
                )
                random_drops.append(float(base_score - ctrl_score))

            random_drops = np.asarray(random_drops, dtype=np.float32)
            rand_mean = float(random_drops.mean()) if len(random_drops) else np.nan
            delta_vs_random = float(true_drop - rand_mean) if len(random_drops) else np.nan
            empirical_p = float((1 + np.sum(random_drops >= true_drop)) / (len(random_drops) + 1)) if len(random_drops) else np.nan

            validated = (
                true_drop >= min_score_drop and
                (np.isnan(delta_vs_random) or delta_vs_random >= min_delta_vs_random) and
                (np.isnan(empirical_p) or empirical_p <= max_empirical_p)
            )

            out = row._asdict()
            out.update({
                "base_toxicity_score": float(base_score),
                "counterfactual_perturbation": "graph_node_edge_occlusion",
                "counterfactual_masked_score": float(masked_score),
                "counterfactual_score_drop": float(true_drop),
                "random_control_mean_drop": rand_mean,
                "delta_vs_random_mean": delta_vs_random,
                "empirical_p_vs_random": empirical_p,
                "n_random_controls_realised": int(len(random_drops)),
                "counterfactual_validated": int(validated),
            })
            rows.append(out)
        except Exception as e:
            print(f"[warn] counterfactual failed for source_id={row.source_id}: {e}")

    return pd.DataFrame(rows)


def summarise_validated_clusters(validated_occ_df: pd.DataFrame):
    if validated_occ_df.empty:
        return pd.DataFrame()

    out = (
        validated_occ_df.groupby("cluster_id", dropna=False)
        .agg(
            n_validated_occurrences=("counterfactual_validated", "sum"),
            n_tested_occurrences=("counterfactual_validated", "count"),
            validated_fraction=("counterfactual_validated", "mean"),
            mean_true_drop=("counterfactual_score_drop", "mean"),
            mean_random_drop=("random_control_mean_drop", "mean"),
            mean_delta_vs_random=("delta_vs_random_mean", "mean"),
            mean_empirical_p=("empirical_p_vs_random", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["validated_fraction", "mean_true_drop", "mean_delta_vs_random"], ascending=[False, False, False])


def _mol_from_fragment_smiles(fragment_smiles: str):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is not None:
        return mol
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(fragment_smiles, sanitize=False)
    return mol


def render_top_toxicophores(df: pd.DataFrame, top_n: int = 10, save_path: str = "top_toxicophores.svg"):
    if df.empty:
        print("No toxicophores found to render.")
        return

    sort_cols = [c for c in ["validated_fraction", "mean_true_drop", "n_positive_molecules", "mean_seed_score"] if c in df.columns]
    top_df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_n)

    mols, legends = [], []
    for _, row in top_df.iterrows():
        mol = _mol_from_fragment_smiles(row["representative_smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        if "validated_fraction" in row.index and pd.notna(row.get("validated_fraction", np.nan)):
            legends.append(
                f"Pos:{int(row.get('n_positive_molecules', 0))}  "
                f"Val:{row.get('validated_fraction', np.nan):.2f}\n"
                f"Drop:{row.get('mean_true_drop', np.nan):.2f}  "
                f"Rnd:{row.get('mean_delta_vs_random', np.nan):.2f}"
            )
        else:
            legends.append(
                f"Pos:{int(row.get('n_positive_molecules', 0))}  "
                f"Mol:{int(row.get('n_molecules', 0))}\n"
                f"Occ:{int(row.get('n_occurrences', 0))}  "
                f"SHAP:{row.get('mean_seed_score', np.nan):.2f}"
            )

    if not mols:
        return

    per_row = min(5, len(mols))
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(per_row * 320, ((len(mols) - 1) // per_row + 1) * 300, 320, 300)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.padding = 0.08
    opts.legendFontSize = 20
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(drawer.GetDrawingText(), encoding="utf-8")
    print(f"Saved {len(mols)} toxicophores to {save_path}")


def run_pipeline(
    model,
    dataset,
    target_task: int = 0,
    positive_label: int = 1,
    out_prefix: str = "consensus_toxicophores_task_0",
    device: str = "cuda",
    cluster_eps: float = 0.30,
    min_cluster_occurrences: int = 3,
    n_random_controls: int = 20,
    min_score_drop: float = 0.10,
    min_delta_vs_random: float = 0.05,
    max_empirical_p: float = 0.10,
):
    print("Extracting toxicophore occurrences...")
    occ_df = extract_toxicophore_occurrences(
        model,
        dataset,
        target_task=target_task,
        positive_label=positive_label,
        device=device,
    )
    occ_df.to_csv(f"{out_prefix}_occurrences.csv", index=False)
    if occ_df.empty:
        print("No SHAP-positive toxicophore occurrences found.")
        empty = pd.DataFrame()
        empty.to_csv(f"{out_prefix}_final_validated_clusters.csv", index=False)
        return empty

    print("Clustering consensus toxicophores...")
    cluster_df, occ_df = cluster_consensus_toxicophores(
        occ_df,
        eps=cluster_eps,
        min_samples=min_cluster_occurrences,
    )
    cluster_df.to_csv(f"{out_prefix}_clusters.csv", index=False)
    occ_df.to_csv(f"{out_prefix}_occurrences_with_clusters.csv", index=False)
    if cluster_df.empty or occ_df.empty:
        print("No consensus toxicophore clusters passed DBSCAN or exact-repeat fallback.")
        empty = pd.DataFrame()
        empty.to_csv(f"{out_prefix}_final_validated_clusters.csv", index=False)
        empty.to_csv(f"{out_prefix}_final_consensus_clusters.csv", index=False)
        return empty

    print("Running counterfactual validation...")
    validated_occ_df = validate_occurrences_counterfactually(
        occ_df,
        dataset,
        model,
        target_task=target_task,
        positive_label=positive_label,
        n_random_controls=n_random_controls,
        min_score_drop=min_score_drop,
        min_delta_vs_random=min_delta_vs_random,
        max_empirical_p=max_empirical_p,
        device=device,
    )
    validated_occ_df.to_csv(f"{out_prefix}_counterfactual_occurrences.csv", index=False)

    validated_cluster_df = summarise_validated_clusters(validated_occ_df)
    validated_cluster_df.to_csv(f"{out_prefix}_counterfactual_cluster_summary.csv", index=False)
    cluster_df.to_csv(f"{out_prefix}_final_consensus_clusters.csv", index=False)
    if validated_cluster_df.empty:
        print("No toxicophore clusters passed counterfactual validation; rendering SHAP consensus clusters.")
        empty = pd.DataFrame()
        empty.to_csv(f"{out_prefix}_final_validated_clusters.csv", index=False)
        out_df = cluster_df.copy()
        out_df["selection_basis"] = "shap_consensus"
        return out_df

    final_df = cluster_df.merge(validated_cluster_df, on="cluster_id", how="left")
    final_df = final_df[final_df["n_validated_occurrences"].fillna(0) > 0].copy()
    if final_df.empty:
        print("No toxicophore clusters passed counterfactual validation; rendering SHAP consensus clusters.")
        empty = pd.DataFrame()
        empty.to_csv(f"{out_prefix}_final_validated_clusters.csv", index=False)
        out_df = cluster_df.copy()
        out_df["selection_basis"] = "shap_consensus"
        return out_df

    final_df = final_df.sort_values(
        ["validated_fraction", "mean_true_drop", "mean_delta_vs_random", "n_positive_molecules"],
        ascending=[False, False, False, False],
    )
    final_df["selection_basis"] = "counterfactual_validated"
    final_df.to_csv(f"{out_prefix}_final_validated_clusters.csv", index=False)
    return final_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover consensus toxicophores from mandatory pyg_captum_shap attributions."
    )
    parser.add_argument("--graph_path", type=Path, default=Path("graph_objs/pyg_graphs_class.pkl"))
    parser.add_argument("--class_ckpt", type=Path, default=None)
    parser.add_argument("--tasks", type=str, default="Ames")
    parser.add_argument("--num_molecules", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out_dir", type=Path, default=Path("figures_classification/shap/toxicophore_discovery"))
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--cluster_eps", type=float, default=0.30)
    parser.add_argument("--min_cluster_occurrences", type=int, default=3)
    parser.add_argument("--n_random_controls", type=int, default=20)
    parser.add_argument("--min_score_drop", type=float, default=0.10)
    parser.add_argument("--min_delta_vs_random", type=float, default=0.05)
    parser.add_argument("--max_empirical_p", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    _, _, test_list, _, _, _ = joblib.load(args.graph_path)
    dataset = test_list[args.start_index: args.start_index + args.num_molecules]

    helpers = load_case_study_toxlens_helpers()
    task_names = list(helpers["task_names"] or PRIMARY_TOXLENS_TASKS)
    requested_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    missing = [t for t in requested_tasks if t not in task_names]
    if missing:
        raise ValueError(f"Requested task(s) not in ToxLens primary task order: {missing}. Available={task_names}")

    ckpt = args.class_ckpt or resolve_latest_classification_checkpoint()
    print(f"[Toxicophore] Loading checkpoint: {ckpt}")
    GAT_class = helpers["build_model_class"](num_tasks_default=len(task_names))
    model = helpers["load_checkpoint"](GAT_class, ckpt, torch.device(device), task_names)
    model = model.to(device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_final = []
    for task_name in requested_tasks:
        task_idx = task_names.index(task_name)
        safe_task = task_name.replace("/", "_").replace(" ", "_")
        out_prefix = str(args.out_dir / f"{safe_task}_consensus_toxicophores")
        print(f"[Toxicophore] Running SHAP toxicophore discovery for {task_name} on {len(dataset)} molecules.")
        final_df = run_pipeline(
            model=model,
            dataset=dataset,
            target_task=task_idx,
            positive_label=1,
            out_prefix=out_prefix,
            device=device,
            cluster_eps=args.cluster_eps,
            min_cluster_occurrences=args.min_cluster_occurrences,
            n_random_controls=args.n_random_controls,
            min_score_drop=args.min_score_drop,
            min_delta_vs_random=args.min_delta_vs_random,
            max_empirical_p=args.max_empirical_p,
        )
        if not final_df.empty:
            final_df.insert(0, "task_name", task_name)
            all_final.append(final_df)
        render_top_toxicophores(
            final_df,
            top_n=args.top_n,
            save_path=str(args.out_dir / f"{safe_task}_top_{args.top_n}_toxicophores.svg"),
        )

    if all_final:
        combined = pd.concat(all_final, ignore_index=True)
        combined.to_csv(args.out_dir / "all_tasks_reported_toxicophores.csv", index=False)
        print(f"[Toxicophore] Saved combined reported toxicophores to {args.out_dir / 'all_tasks_reported_toxicophores.csv'}")


if __name__ == "__main__":
    main()

