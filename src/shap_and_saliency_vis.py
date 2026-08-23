import copy, math, os, random, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from pyg_captum_shap import compute_shap_values



"""
Run it like this:

python src/shap_and_saliency_vis.py --class_ckpt release_assets/checkpoints/main/toxlens_single_best.ckpt --tasks Ames,SR-MMP --num_molecules 3 --random_repeats 30 --device cuda

For a cleaner side-by-side saliency figure where only the strongest atoms are coloured:

python src/shap_and_saliency_vis.py --class_ckpt release_assets/checkpoints/main/toxlens_single_best.ckpt --tasks Ames,SR-MMP --num_molecules 3 --top_fraction 0.20
"""

@dataclass
class ExplainConfig:
    # Paths
    graph_path: Path = Path("graph_objs/pyg_graphs_class.pkl")
    ckpt: Optional[Path] = None
    out_dir: Path = Path("figures_classification/shap_occlusion")

    # Tasks
    task_names: Tuple[str, ...] = field(default_factory=tuple)
    # Molecules
    num_molecules: int = 10
    start_index: int = 0

    # Faithfulness settings
    fractions: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    random_repeats: int = 50
    seed: int = 42

    # Occlusion settings
    mask_node_features: bool = True
    mask_incident_edge_features: bool = True

    # Rendering
    svg_size: Tuple[int, int] = (600, 600)
    highlight_top_fraction: Optional[float] = None
    # If provided, cap the display to the top fraction of heavy atoms.
    highlight_min_norm: float = 0.35
    # Normalised attribution threshold for visible saliency. This keeps the
    # figures focused on meaningful substructures instead of faint heatmaps.
    device: str = "cuda"


PRIMARY_TOXLENS_TASKS: Tuple[str, ...] = (
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


def default_class_task_names() -> Tuple[str, ...]:
    return PRIMARY_TOXLENS_TASKS


def resolve_latest_classification_checkpoint() -> Path:
    env_candidates = [
        os.environ.get("TOXLENS_INTERPRET_CKPT"),
        os.environ.get("DEEP_TOX_EVAL_CKPT"),
        os.environ.get("TOXLENS_CKPT"),
    ]
    for value in env_candidates:
        if value and Path(value).exists():
            return Path(value)

    model_root = Path(os.environ.get(
        "DEEP_TOX_CHECKPOINT_DIR",
        str(Path(__file__).resolve().parents[1] / "release_assets" / "checkpoints"),
    ))
    patterns = [
        "deep_tox_classification_*/*.ckpt",
    ]
    candidates: list[Path] = []
    if model_root.exists():
        for pattern in patterns:
            candidates.extend(model_root.glob(pattern))

    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            "Could not find a classification checkpoint. Pass --class_ckpt explicitly "
            "or set TOXLENS_INTERPRET_CKPT."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_case_study_toxlens_model_helpers():
    """
    Import the self-contained ToxLens model replica used by the case-study
    script. This avoids importing deep_tox.py, whose top-level dependencies are
    too heavy for a standalone interpretability utility.
    """
    case_dir = Path(os.environ.get(
        "TOXLENS_CASE_STUDY_DIR",
        str(Path(__file__).resolve().parent),
    ))
    if not case_dir.exists():
        raise FileNotFoundError(
            f"Could not find case-study directory {case_dir}. Set TOXLENS_CASE_STUDY_DIR."
        )
    case_dir_str = str(case_dir)
    if case_dir_str not in sys.path:
        sys.path.insert(0, case_dir_str)

    from integrated_case_study_thesis import (
        PRIMARY_TOXLENS_TASKS as case_primary_tasks,
        _build_toxlens_model_class,
        _load_primary_toxlens_checkpoint,
    )

    return _build_toxlens_model_class, _load_primary_toxlens_checkpoint, tuple(case_primary_tasks)


class RawFeatureProxy(torch.nn.Module):
    """
    Proxy model for pyg_captum_shap.

    This mirrors your current approach: the wrapper exposes raw node features
    to the attribution library, then manually applies the original model's
    node embedding before calling the underlying architecture.
    """

    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.original_model = original_model

    def node_emb(self, x):
        return x

    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_attr=None,
        global_features=None,
        apply_embedding: bool = False,
    ):
        embedded_x = self.original_model.node_emb(x)

        return self.original_model(
            x=embedded_x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            global_features=global_features,
            apply_embedding=False,
        )
    
def _move_data_to_device(data, device: torch.device):
    data = copy.deepcopy(data)
    return data.to(device)


def model_score(
    model: torch.nn.Module,
    data,
    target_task: int,
    classification: bool = True,
    use_probability: bool = False,
) -> float:
    """
    Return scalar model score for one PyG graph.
      - use_probability=False returns the raw logit for target_task.
      - use_probability=True returns sigmoid(logit).
    """
    model.eval()

    with torch.no_grad():
        out = model(data)

        if classification:
            logit = out.view(-1)[target_task]
            if use_probability:
                return float(torch.sigmoid(logit).detach().cpu())
            return float(logit.detach().cpu())

        value = out.view(-1)[target_task] if out.numel() > 1 else out.view(-1)[0]
        return float(value.detach().cpu())
    

def reconstruct_explicit_h_molecule(data) -> Chem.Mol:
    """
    Reconstruct explicit-H RDKit molecule from data.smiles.

    The ToxLens graph contains explicit hydrogens, so we use Chem.AddHs().
    """
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {data.smiles}")

    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() != data.x.size(0):
        raise ValueError(
            f"Atom-count mismatch for {data.smiles}: "
            f"RDKit explicit-H atoms={mol_h.GetNumAtoms()}, graph nodes={data.x.size(0)}"
        )

    return mol_h


def heavy_atom_mapping(mol_h: Chem.Mol) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Map explicit-H graph atom indices onto heavy-atom indices.

    Returns
    -------
    graph_to_heavy:
        Maps every explicit graph atom index, including H, to heavy atom index.

    heavy_groups:
        List of explicit atom indices for each heavy atom group.
        Each group contains the heavy atom and all attached hydrogens.
    """
    graph_to_heavy: Dict[int, int] = {}
    heavy_groups: List[List[int]] = []

    heavy_idx = 0
    heavy_graph_indices = []

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            graph_idx = atom.GetIdx()
            graph_to_heavy[graph_idx] = heavy_idx
            heavy_graph_indices.append(graph_idx)
            heavy_groups.append([graph_idx])
            heavy_idx += 1

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            h_idx = atom.GetIdx()
            parent_idx = atom.GetNeighbors()[0].GetIdx()
            parent_heavy = graph_to_heavy[parent_idx]
            graph_to_heavy[h_idx] = parent_heavy
            heavy_groups[parent_heavy].append(h_idx)

    return graph_to_heavy, heavy_groups


def _graph_node_hydrogen_mask(data) -> np.ndarray:
    """
    Infer hydrogen nodes from the saved ToxLens atom-feature tensor.

    The first atom-feature block is element one-hot encoded and H is the 30th
    listed element in the case-study/ToxLens featuriser.
    """
    x = data.x.detach().cpu().numpy()
    if x.ndim != 2 or x.shape[1] <= 29:
        return np.zeros(int(data.x.size(0)), dtype=bool)
    return x[:, 29] > 0.5


def heavy_atom_mapping_from_graph(data, mol_no_h: Chem.Mol) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Fallback graph-to-heavy mapping when RDKit AddHs() gives a different
    explicit-H count than the saved PyG graph.

    The graph was generated from RDKit atoms, so heavy atoms preserve canonical
    atom order. Hydrogens are assigned to adjacent heavy atoms through
    edge_index.
    """
    n_nodes = int(data.x.size(0))
    n_heavy = int(mol_no_h.GetNumAtoms())
    if n_nodes < n_heavy:
        raise ValueError(
            f"Graph has fewer nodes ({n_nodes}) than RDKit heavy atoms ({n_heavy}) for {getattr(data, 'smiles', '')}"
        )

    is_h = _graph_node_hydrogen_mask(data)
    heavy_nodes = [int(i) for i in np.where(~is_h)[0]]
    if len(heavy_nodes) != n_heavy:
        heavy_nodes = list(range(n_heavy))

    graph_to_heavy: Dict[int, int] = {}
    heavy_groups: List[List[int]] = [[] for _ in range(n_heavy)]
    for heavy_idx, graph_idx in enumerate(heavy_nodes[:n_heavy]):
        graph_to_heavy[int(graph_idx)] = int(heavy_idx)
        heavy_groups[int(heavy_idx)].append(int(graph_idx))

    edge_index = data.edge_index.detach().cpu().numpy()
    neighbours: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for u, v in edge_index.T:
        neighbours[int(u)].append(int(v))

    for node_idx in range(n_nodes):
        if node_idx in graph_to_heavy:
            continue
        parent = next((nbr for nbr in neighbours.get(node_idx, []) if nbr in graph_to_heavy), None)
        if parent is None:
            continue
        parent_heavy = graph_to_heavy[parent]
        graph_to_heavy[node_idx] = parent_heavy
        heavy_groups[parent_heavy].append(node_idx)

    missing = [i for i in range(n_nodes) if i not in graph_to_heavy]
    if missing:
        raise ValueError(
            f"Could not map {len(missing)} graph nodes to heavy atoms for {getattr(data, 'smiles', '')}"
        )

    return graph_to_heavy, heavy_groups


def collapse_node_shap_to_heavy_atoms(
    node_attr: np.ndarray,
    graph_to_heavy: Dict[int, int],
    n_heavy: int,
) -> np.ndarray:
    """
    Collapse explicit-H node SHAP values onto parent heavy atoms.
    """
    heavy_attr = np.zeros(n_heavy, dtype=np.float64)

    for graph_idx, val in enumerate(node_attr):
        heavy_idx = graph_to_heavy[graph_idx]
        heavy_attr[heavy_idx] += float(val)

    return heavy_attr


def collapse_edge_shap_to_heavy_atoms(
    data,
    edge_attr: Optional[np.ndarray],
    graph_to_heavy: Dict[int, int],
    n_heavy: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """
    Collapse edge attributions.

    Heavy-H edge attribution is absorbed into the parent heavy atom.
    Heavy-heavy edge attribution is preserved separately.

    Returns
    -------
    heavy_edge_to_node_attr:
        Additional atom attribution from heavy-H bonds.

    heavy_bond_attr:
        Attribution for heavy-heavy bonds, keyed by sorted heavy atom pair.
    """
    heavy_edge_to_node_attr = np.zeros(n_heavy, dtype=np.float64)
    heavy_bond_attr: Dict[Tuple[int, int], float] = {}

    if edge_attr is None:
        return heavy_edge_to_node_attr, heavy_bond_attr

    edge_index = data.edge_index.detach().cpu().numpy()

    for e_idx in range(edge_index.shape[1]):
        u = int(edge_index[0, e_idx])
        v = int(edge_index[1, e_idx])

        # Process each undirected bond once.
        if u > v:
            continue

        hu = graph_to_heavy[u]
        hv = graph_to_heavy[v]
        val = float(edge_attr[e_idx])

        if hu == hv:
            heavy_edge_to_node_attr[hu] += val
        else:
            key = tuple(sorted((hu, hv)))
            heavy_bond_attr[key] = heavy_bond_attr.get(key, 0.0) + val

    return heavy_edge_to_node_attr, heavy_bond_attr


def compute_heavy_atom_shap(
    model: torch.nn.Module,
    data,
    target_task: int,
    include_edge_attribution: bool = True,
) -> Dict[str, object]:
    """
    Compute SHAP values and collapse them to heavy-atom attribution.

    Returns signed heavy-atom attributions.
    Positive values increase the selected model output.
    Negative values decrease the selected model output.
    """
    proxy_model = RawFeatureProxy(model)
    shap_results = compute_shap_values(proxy_model, data, target_task=target_task)

    node_attr = shap_results["nodes"].detach().cpu().numpy().sum(axis=1)

    raw_edge_attr = shap_results.get("edges", None)
    if raw_edge_attr is not None and include_edge_attribution:
        raw_edge_attr = raw_edge_attr.detach().cpu().numpy().sum(axis=1)
    else:
        raw_edge_attr = None

    try:
        mol_h = reconstruct_explicit_h_molecule(data)
        graph_to_heavy, heavy_groups = heavy_atom_mapping(mol_h)
        mol_no_h = Chem.RemoveHs(mol_h)
    except ValueError as exc:
        if "Atom-count mismatch" not in str(exc):
            raise
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            raise
        mol_no_h = Chem.RemoveHs(mol)
        graph_to_heavy, heavy_groups = heavy_atom_mapping_from_graph(data, mol_no_h)
    n_heavy = len(heavy_groups)

    heavy_node_attr = collapse_node_shap_to_heavy_atoms(
        node_attr=node_attr,
        graph_to_heavy=graph_to_heavy,
        n_heavy=n_heavy,
    )

    heavy_edge_to_node_attr, heavy_bond_attr = collapse_edge_shap_to_heavy_atoms(
        data=data,
        edge_attr=raw_edge_attr,
        graph_to_heavy=graph_to_heavy,
        n_heavy=n_heavy,
    )

    heavy_total_attr = heavy_node_attr + heavy_edge_to_node_attr

    return {
        "mol_h": Chem.AddHs(mol_no_h),
        "mol_no_h": mol_no_h,
        "graph_to_heavy": graph_to_heavy,
        "heavy_groups": heavy_groups,
        "heavy_node_attr": heavy_node_attr,
        "heavy_edge_to_node_attr": heavy_edge_to_node_attr,
        "heavy_total_attr": heavy_total_attr,
        "heavy_bond_attr": heavy_bond_attr,
        "raw_node_attr": node_attr,
        "raw_edge_attr": raw_edge_attr,
        "shap_results": shap_results,
    }


def _mix_with_white(
    base: Tuple[float, float, float],
    strength: float,
) -> Tuple[float, float, float]:
    """
    Blend a highlight colour with white while keeping weak retained highlights
    visible enough for publication figures.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    strength = 0.40 + 0.60 * strength
    return tuple(1.0 - strength * (1.0 - float(c)) for c in base)


def signed_colour(value: float) -> Tuple[float, float, float]:
    """
    Signed colour map for ToxLens:
      positive -> toxicity-increasing red
      negative -> toxicity-decreasing blue
    """
    value = float(np.clip(value, -1.0, 1.0))
    if value >= 0.0:
        return _mix_with_white((1.0, 0.02, 0.04), abs(value))
    return _mix_with_white((0.05, 0.20, 1.0), abs(value))


def select_meaningful_saliency_atoms(
    heavy_attr: np.ndarray,
    top_fraction: Optional[float] = None,
    positive_only: bool = False,
    min_norm: float = 0.35,
) -> Tuple[np.ndarray, List[int]]:
    """
    Normalise heavy-atom SHAP values and keep only meaningful highlights.

    positive_only=True is used for toxicity-only figures: blue/protective
    contributions are removed, leaving only atoms that increase toxicity.
    """
    heavy_attr = np.asarray(heavy_attr, dtype=float)
    if heavy_attr.size == 0:
        return heavy_attr, []

    max_abs = float(np.max(np.abs(heavy_attr)) + 1e-12)
    norm = heavy_attr / max_abs
    scores = np.clip(norm, 0.0, None) if positive_only else np.abs(norm)
    ranked = [int(i) for i in np.argsort(-scores) if float(scores[int(i)]) > 0.0]

    if top_fraction is not None:
        k = max(1, int(math.ceil(float(top_fraction) * len(norm))))
        ranked = ranked[:k]

    keep = [i for i in ranked if float(scores[i]) >= float(min_norm)]
    if not keep and ranked and not positive_only:
        keep = [ranked[0]]
    elif not keep and ranked and positive_only and float(scores[ranked[0]]) > 0.0:
        keep = [ranked[0]]

    return norm, sorted(set(keep))


def render_heavy_atom_shap_svg(
    mol_no_h: Chem.Mol,
    heavy_attr: np.ndarray,
    save_path: Path,
    size: Tuple[int, int] = (600, 600),
    top_fraction: Optional[float] = None,
    positive_only: bool = False,
    single_colour: Optional[Tuple[float, float, float]] = None,
    heavy_bond_attr: Optional[Dict[Tuple[int, int], float]] = None,
    min_norm: float = 0.35,
) -> None:
    """
    Render heavy-atom SHAP attribution using the thesis saliency style.

    The signed view uses red/blue highlights. The toxicity-only view should set
    positive_only=True and single_colour=(1.0, 0.0, 0.0).
    """
    mol_no_h = Chem.Mol(mol_no_h)
    AllChem.Compute2DCoords(mol_no_h)

    norm, keep_atoms = select_meaningful_saliency_atoms(
        heavy_attr=heavy_attr,
        top_fraction=top_fraction,
        positive_only=positive_only,
        min_norm=min_norm,
    )

    atom_cols: Dict[int, Tuple[float, float, float]] = {}
    atoms_to_highlight: List[int] = []
    for atom_idx in keep_atoms:
        value = float(norm[atom_idx])
        strength = min(abs(value), 1.0)
        if single_colour is not None:
            atom_cols[atom_idx] = _mix_with_white(single_colour, strength)
        else:
            atom_cols[atom_idx] = signed_colour(value)
        atoms_to_highlight.append(atom_idx)

    bonds_to_highlight: List[int] = []
    bond_cols: Dict[int, Tuple[float, float, float]] = {}
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
            if float(score) < float(min_norm):
                continue
            strength = min(abs(b_norm), 1.0)
            bonds_to_highlight.append(bond.GetIdx())
            if single_colour is not None:
                bond_cols[bond.GetIdx()] = _mix_with_white(single_colour, strength)
            else:
                bond_cols[bond.GetIdx()] = signed_colour(b_norm)

    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.05
    if hasattr(opts, "useBWAtomPalette"):
        opts.useBWAtomPalette()

    drawer.DrawMolecule(
        mol_no_h,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_cols,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(drawer.GetDrawingText())


def save_atom_attribution_csv(
    mol_no_h: Chem.Mol,
    heavy_attr: np.ndarray,
    save_path: Path,
) -> None:
    rows = []

    for atom in mol_no_h.GetAtoms():
        idx = atom.GetIdx()
        rows.append(
            {
                "heavy_atom_index": idx,
                "element": atom.GetSymbol(),
                "signed_shap": float(heavy_attr[idx]),
                "absolute_shap": float(abs(heavy_attr[idx])),
            }
        )

    df = pd.DataFrame(rows).sort_values("absolute_shap", ascending=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)


def clone_data_for_occlusion(data):
    """
    Deep-copy a PyG data object and detach tensors so masking cannot affect
    the original object or autograd history.
    """
    out = copy.deepcopy(data)

    out.x = out.x.detach().clone()
    out.edge_index = out.edge_index.detach().clone()

    if getattr(out, "edge_attr", None) is not None:
        out.edge_attr = out.edge_attr.detach().clone()

    if getattr(out, "global_features", None) is not None:
        out.global_features = out.global_features.detach().clone()

    return out


def mask_heavy_atom_groups(
    data,
    heavy_groups: Sequence[Sequence[int]],
    heavy_atom_indices: Sequence[int],
    mask_node_features: bool = True,
    mask_incident_edge_features: bool = True,
):
    """
    Mask selected heavy-atom groups.

    Each heavy-atom group contains the heavy atom and all attached explicit Hs.
    The graph topology is kept fixed, but node features and configured incident
    edge features are zeroed. This is model-level perturbation, not chemical
    deletion.
    """
    out = clone_data_for_occlusion(data)

    graph_nodes_to_mask: List[int] = []
    for heavy_idx in heavy_atom_indices:
        graph_nodes_to_mask.extend(list(heavy_groups[heavy_idx]))

    graph_nodes_to_mask = sorted(set(graph_nodes_to_mask))

    if mask_node_features and graph_nodes_to_mask:
        idx = torch.tensor(graph_nodes_to_mask, dtype=torch.long, device=out.x.device)
        out.x[idx, :] = 0.0

    if (
        mask_incident_edge_features
        and getattr(out, "edge_attr", None) is not None
        and out.edge_attr.numel() > 0
        and graph_nodes_to_mask
    ):
        mask_set = set(graph_nodes_to_mask)
        edge_index_np = out.edge_index.detach().cpu().numpy()

        incident_edges = [
            e_idx
            for e_idx in range(edge_index_np.shape[1])
            if int(edge_index_np[0, e_idx]) in mask_set
            or int(edge_index_np[1, e_idx]) in mask_set
        ]

        if incident_edges:
            eidx = torch.tensor(incident_edges, dtype=torch.long, device=out.edge_attr.device)
            out.edge_attr[eidx, :] = 0.0

    return out


def shap_ranked_heavy_atoms(
    heavy_attr: np.ndarray,
    positive_only: bool = True,
) -> List[int]:
    """
    Rank heavy atoms by SHAP attribution.

    For toxicity-positive-class explanations, positive_only=True is often most
    appropriate because it asks whether toxicity-driving atoms are faithful.

    If positive_only=False, ranks by absolute attribution.
    """
    heavy_attr = np.asarray(heavy_attr, dtype=float)

    if positive_only:
        scores = np.maximum(heavy_attr, 0.0)
    else:
        scores = np.abs(heavy_attr)

    return np.argsort(-scores).tolist()


def sample_random_heavy_atoms(
    n_heavy: int,
    k: int,
    rng: np.random.Generator,
) -> List[int]:
    return rng.choice(np.arange(n_heavy), size=k, replace=False).tolist()


def run_faithfulness_analysis(
    model: torch.nn.Module,
    data,
    heavy_groups: Sequence[Sequence[int]],
    heavy_attr: np.ndarray,
    target_task: int,
    classification: bool,
    fractions: Sequence[float],
    random_repeats: int,
    seed: int,
    mask_node_features: bool = True,
    mask_incident_edge_features: bool = True,
    positive_only: bool = True,
    use_probability: bool = False,
) -> pd.DataFrame:
    """
    Compare SHAP-guided occlusion against random occlusion.

    Score drop:
      baseline_score - perturbed_score

    For classification, score is usually the positive-class logit. This avoids
    saturation effects from sigmoid probabilities.
    """
    rng = np.random.default_rng(seed)
    n_heavy = len(heavy_groups)

    baseline_score = model_score(
        model=model,
        data=data,
        target_task=target_task,
        classification=classification,
        use_probability=use_probability,
    )

    ranked_atoms = shap_ranked_heavy_atoms(
        heavy_attr=heavy_attr,
        positive_only=positive_only,
    )

    rows = []

    for frac in fractions:
        k = max(1, int(math.ceil(frac * n_heavy)))
        top_atoms = ranked_atoms[:k]

        top_data = mask_heavy_atom_groups(
            data=data,
            heavy_groups=heavy_groups,
            heavy_atom_indices=top_atoms,
            mask_node_features=mask_node_features,
            mask_incident_edge_features=mask_incident_edge_features,
        )

        top_score = model_score(
            model=model,
            data=top_data,
            target_task=target_task,
            classification=classification,
            use_probability=use_probability,
        )
        top_drop = baseline_score - top_score

        random_drops = []
        random_atom_sets = []

        for _ in range(random_repeats):
            rand_atoms = sample_random_heavy_atoms(n_heavy=n_heavy, k=k, rng=rng)

            rand_data = mask_heavy_atom_groups(
                data=data,
                heavy_groups=heavy_groups,
                heavy_atom_indices=rand_atoms,
                mask_node_features=mask_node_features,
                mask_incident_edge_features=mask_incident_edge_features,
            )

            rand_score = model_score(
                model=model,
                data=rand_data,
                target_task=target_task,
                classification=classification,
                use_probability=use_probability,
            )
            rand_drop = baseline_score - rand_score

            random_drops.append(rand_drop)
            random_atom_sets.append(rand_atoms)

        random_drops = np.asarray(random_drops, dtype=float)

        # One-sided empirical p-value:
        # probability that random masking is at least as disruptive as SHAP masking.
        empirical_p = (
            1.0 + float(np.sum(random_drops >= top_drop))
        ) / (len(random_drops) + 1.0)

        rows.append(
            {
                "fraction_masked": float(frac),
                "num_heavy_atoms_masked": int(k),
                "baseline_score": float(baseline_score),
                "top_shap_score": float(top_score),
                "top_shap_drop": float(top_drop),
                "random_drop_mean": float(np.mean(random_drops)),
                "random_drop_std": float(np.std(random_drops, ddof=1))
                if len(random_drops) > 1
                else 0.0,
                "random_drop_median": float(np.median(random_drops)),
                "random_drop_q025": float(np.quantile(random_drops, 0.025)),
                "random_drop_q975": float(np.quantile(random_drops, 0.975)),
                "empirical_p_random_ge_top": float(empirical_p),
                "top_shap_atoms": ",".join(map(str, top_atoms)),
                "positive_only_ranking": bool(positive_only),
                "score_type": "probability" if use_probability else "logit_or_raw",
            }
        )

    return pd.DataFrame(rows)


def plot_faithfulness(
    df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    """
    Plot SHAP-guided occlusion drop versus random occlusion drop.
    """
    x = df["fraction_masked"].to_numpy() * 100.0
    top = df["top_shap_drop"].to_numpy()
    rand_mean = df["random_drop_mean"].to_numpy()
    rand_std = df["random_drop_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(
        x,
        top,
        marker="o",
        linewidth=2.2,
        label="Top SHAP atoms",
    )

    ax.plot(
        x,
        rand_mean,
        marker="o",
        linewidth=2.0,
        label="Random atoms",
    )

    ax.fill_between(
        x,
        rand_mean - rand_std,
        rand_mean + rand_std,
        alpha=0.20,
        label="Random ± SD",
    )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Heavy atoms masked (%)")
    ax.set_ylabel("Model score drop")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def load_toxlens_objects(cfg: ExplainConfig):
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"[Interpretability] Loading graph pickle: {cfg.graph_path}", flush=True)
    _, _, test_list, _, _, _ = joblib.load(cfg.graph_path)
    print(f"[Interpretability] Loaded {len(test_list)} test graphs.", flush=True)

    print("[Interpretability] Importing self-contained ToxLens model from case-study script.", flush=True)
    build_model_class, load_primary_checkpoint, case_primary_tasks = load_case_study_toxlens_model_helpers()

    saved_task_names = list(getattr(test_list[0], "task_names", None) or []) if test_list else []
    all_task_names = list(case_primary_tasks or default_class_task_names())
    if saved_task_names:
        missing_saved = [t for t in all_task_names if t not in saved_task_names]
        if missing_saved:
            raise ValueError(
                f"Saved graph task order does not contain required ToxLens primary tasks: {missing_saved}. "
                f"Saved task names={saved_task_names}"
            )

    requested = list(cfg.task_names)
    if requested:
        missing = [t for t in requested if t not in all_task_names]
        if missing:
            raise ValueError(f"Requested task(s) not in primary ToxLens task order: {missing}. Available={all_task_names}")
        task_names = [(all_task_names.index(t), t) for t in requested]
    else:
        task_names = list(enumerate(all_task_names))

    class_ckpt = cfg.ckpt or resolve_latest_classification_checkpoint()
    print(f"[Interpretability] Loading ToxLens checkpoint: {class_ckpt}", flush=True)
    GAT_class = build_model_class(num_tasks_default=len(all_task_names))
    model = load_primary_checkpoint(GAT_class, class_ckpt, device, all_task_names)
    print("[Interpretability] ToxLens checkpoint loaded.", flush=True)

    model = model.to(device)
    model.eval()

    return model, test_list, task_names, device


def prepare_single_graph(data, device: torch.device):
    """
    Move a single PyG graph to device.

    The graph object in your current saved list is already a single molecule,
    so no DataLoader is required.
    """
    data = _move_data_to_device(data, device)
    if getattr(data, "batch", None) is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    if getattr(data, "global_features", None) is not None and data.global_features.dim() == 1:
        data.global_features = data.global_features.view(1, -1)
    if getattr(data, "y", None) is not None and data.y.dim() == 1:
        data.y = data.y.view(1, -1)
    return data


def explain_one_molecule_one_task(
    model: torch.nn.Module,
    data,
    task_idx: int,
    task_name: str,
    molecule_index: int,
    cfg: ExplainConfig,
) -> None:
    safe_task = task_name.replace("/", "_").replace(" ", "_")
    prefix = cfg.out_dir / f"mol_{molecule_index:04d}" / safe_task
    prefix.mkdir(parents=True, exist_ok=True)

    print(
        f"[Interpretability] molecule={molecule_index}, task={task_name}, method=pyg_captum_shap",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 1. SHAP attribution
    # -------------------------------------------------------------------------
    shap_info = compute_heavy_atom_shap(
        model=model,
        data=data,
        target_task=task_idx,
        include_edge_attribution=cfg.mask_incident_edge_features,
    )

    mol_no_h = shap_info["mol_no_h"]
    heavy_groups = shap_info["heavy_groups"]
    heavy_attr = shap_info["heavy_total_attr"]
    heavy_bond_attr = shap_info.get("heavy_bond_attr", {})

    # -------------------------------------------------------------------------
    # 2. Render SHAP saliency SVG
    # -------------------------------------------------------------------------
    render_heavy_atom_shap_svg(
        mol_no_h=mol_no_h,
        heavy_attr=heavy_attr,
        save_path=prefix / "heavy_atom_signed_shap.svg",
        size=cfg.svg_size,
        top_fraction=cfg.highlight_top_fraction,
        heavy_bond_attr=heavy_bond_attr,
        min_norm=cfg.highlight_min_norm,
    )

    render_heavy_atom_shap_svg(
        mol_no_h=mol_no_h,
        heavy_attr=heavy_attr,
        save_path=prefix / "heavy_atom_toxicity_only_shap.svg",
        size=cfg.svg_size,
        top_fraction=cfg.highlight_top_fraction,
        positive_only=True,
        single_colour=(1.0, 0.0, 0.0),
        heavy_bond_attr=heavy_bond_attr,
        min_norm=cfg.highlight_min_norm,
    )

    save_atom_attribution_csv(
        mol_no_h=mol_no_h,
        heavy_attr=heavy_attr,
        save_path=prefix / "heavy_atom_signed_shap.csv",
    )

    # -------------------------------------------------------------------------
    # 3. SHAP-guided occlusion faithfulness
    # -------------------------------------------------------------------------
    faith_df = run_faithfulness_analysis(
        model=model,
        data=data,
        heavy_groups=heavy_groups,
        heavy_attr=heavy_attr,
        target_task=task_idx,
        classification=True,
        fractions=cfg.fractions,
        random_repeats=cfg.random_repeats,
        seed=cfg.seed + molecule_index * 1000 + task_idx,
        mask_node_features=cfg.mask_node_features,
        mask_incident_edge_features=cfg.mask_incident_edge_features,
        positive_only=True,
        use_probability=False,
    )

    faith_df.insert(0, "task_name", task_name)
    faith_df.insert(0, "task_idx", task_idx)
    faith_df.insert(0, "smiles", getattr(data, "smiles", ""))
    faith_df.insert(0, "molecule_index", molecule_index)

    faith_df.to_csv(prefix / "faithfulness_shap_guided_occlusion.csv", index=False)

    plot_faithfulness(
        df=faith_df,
        save_path=prefix / "faithfulness_shap_guided_occlusion.png",
        title=f"SHAP-Guided Occlusion Faithfulness: {task_name}",
    )

    print(f"[OK] molecule={molecule_index}, task={task_name}, out={prefix}")


def run(cfg: ExplainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    model, test_list, task_items, device = load_toxlens_objects(cfg)

    selected = test_list[cfg.start_index : cfg.start_index + cfg.num_molecules]

    all_faithfulness = []

    for mol_offset, raw_data in enumerate(selected):
        molecule_index = cfg.start_index + mol_offset
        data = prepare_single_graph(raw_data, device)

        for task_idx, task_name in task_items:
            try:
                explain_one_molecule_one_task(
                    model=model,
                    data=data,
                    task_idx=task_idx,
                    task_name=task_name,
                    molecule_index=molecule_index,
                    cfg=cfg,
                )

                faith_path = (
                    cfg.out_dir
                    / f"mol_{molecule_index:04d}"
                    / task_name.replace("/", "_").replace(" ", "_")
                    / "faithfulness_shap_guided_occlusion.csv"
                )
                all_faithfulness.append(pd.read_csv(faith_path))

            except Exception as exc:
                print(f"[FAIL] molecule={molecule_index}, task={task_name}: {exc}")

    if all_faithfulness:
        all_df = pd.concat(all_faithfulness, ignore_index=True)
        all_df.to_csv(cfg.out_dir / "all_faithfulness_results.csv", index=False)

        summary = (
            all_df.groupby(["task_name", "fraction_masked"], as_index=False)
            .agg(
                top_shap_drop_mean=("top_shap_drop", "mean"),
                random_drop_mean=("random_drop_mean", "mean"),
                empirical_p_median=("empirical_p_random_ge_top", "median"),
            )
        )
        summary["faithfulness_gain"] = (
            summary["top_shap_drop_mean"] - summary["random_drop_mean"]
        )
        summary.to_csv(cfg.out_dir / "faithfulness_summary_by_task.csv", index=False)

        print(f"[DONE] Wrote combined results to {cfg.out_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> ExplainConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="ToxLens SHAP saliency + SHAP-guided occlusion faithfulness."
    )
    parser.add_argument("--num_molecules", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--random_repeats", type=int, default=50)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--graph_path",
        type=Path,
        default=None,
        help="Path to the saved classification graph pickle. Defaults to graph_objs/pyg_graphs_class.pkl.",
    )
    parser.add_argument(
        "--class_ckpt",
        type=Path,
        default=None,
        help="Classification checkpoint to explain. If omitted, TOXLENS_INTERPRET_CKPT/DEEP_TOX_EVAL_CKPT or the newest local checkpoint is used.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to explain, e.g. Ames,SR-MMP,NR-ER-LBD. Defaults to all saved primary tasks.",
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=None,
        help="Cap the fraction of heavy atoms highlighted in SVG, e.g. 0.20.",
    )
    parser.add_argument(
        "--highlight_threshold",
        type=float,
        default=0.35,
        help="Minimum normalised SHAP magnitude to show in saliency SVGs.",
    )
    parser.add_argument(
        "--no_edge_masking",
        action="store_true",
        help="Only mask node features, not incident edge features.",
    )
    parser.add_argument(
        "--probability_score",
        action="store_true",
        help=(
            "Reserved for future use. Current implementation uses logits/raw scores "
            "for faithfulness because probabilities can saturate."
        ),
    )

    args = parser.parse_args()

    cfg = ExplainConfig()
    cfg.num_molecules = args.num_molecules
    cfg.start_index = args.start_index
    cfg.random_repeats = args.random_repeats
    cfg.device = args.device
    cfg.highlight_top_fraction = args.top_fraction
    cfg.highlight_min_norm = args.highlight_threshold
    cfg.mask_incident_edge_features = not args.no_edge_masking
    if args.graph_path is not None:
        cfg.graph_path = args.graph_path
    if args.class_ckpt is not None:
        cfg.ckpt = args.class_ckpt
    if args.tasks:
        cfg.task_names = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    else:
        cfg.out_dir = (
            Path("figures_classification/shap_occlusion")
        )

    return cfg

if __name__ == "__main__":
    run(parse_args())
