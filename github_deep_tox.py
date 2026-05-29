import os
import math
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, GraphNorm, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from joblib import Parallel, delayed

from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors, ChemicalFeatures, AllChem, Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.EState import EStateIndices

# --- MoLFormer Dummy Patch ---
try:
    import transformers.onnx
except ImportError:
    from types import ModuleType
    onnx_dummy = ModuleType("transformers.onnx")
    sys.modules["transformers.onnx"] = onnx_dummy
    class OnnxConfig: pass
    onnx_dummy.OnnxConfig = OnnxConfig
import transformers.pytorch_utils as pt_utils
if not hasattr(pt_utils, 'find_pruneable_heads_and_indices'):
    setattr(pt_utils, 'find_pruneable_heads_and_indices', lambda *args, **kwargs: None)
if not hasattr(pt_utils, 'prune_linear_layer'):
    setattr(pt_utils, 'prune_linear_layer', lambda x, y, z: x)
if not hasattr(pt_utils, 'apply_chunking_to_forward'):
    try:
        from transformers.modeling_utils import apply_chunking_to_forward
        setattr(pt_utils, 'apply_chunking_to_forward', apply_chunking_to_forward)
    except ImportError:
        setattr(pt_utils, 'apply_chunking_to_forward', lambda x, y, z: x())
from transformers import AutoTokenizer, AutoModel


# 1. CONSTANTS & SMARTS PATTERNS
_PAULING_EN = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.6, 42: 2.16, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 55: 0.79, 56: 0.89,
    72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.28,
    79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02
}

_PHARM_FAMILIES = ('Donor', 'Acceptor', 'Hydrophobe', 'PosIonizable', 'NegIonizable', 'Aromatic')
_PHARM_IDX = {fam: i for i, fam in enumerate(_PHARM_FAMILIES)}

_ATOM_REACTIVITY_SMARTS = (
    ('carbonyl_c', '[CX3](=[OX1,SX1])'), ('imine_c', '[CX3]=[NX2]'), ('nitrile_c', '[CX2]#N'),
    ('michael_acceptor', '[C,c]=[C,c][C,c](=O)'), ('aryl_halide_ipso', '[c][F,Cl,Br,I]'),
    ('alkyl_halide_c', '[CX4][Cl,Br,I]'), ('epoxide_atom', 'C1OC1'), ('aziridine_atom', 'N1CC1'),
    ('nitro_n', '[N+](=O)[O-]'), ('diazo_atom', '[C]=[N+]=[N-]'), ('aniline_n', '[NX3;H2,H1;!$(NC=O)]c'),
    ('phenol_o', '[OX2H]c'), ('thiol_s', '[SX2H]'), ('sulfonyl_s', 'S(=O)(=O)'),
    ('phosphoryl_p', 'P(=O)'), ('quinone_atom', 'O=C1C=CC(=O)C=C1')
)

_BOND_REACTIVITY_SMARTS = (
    ('amide_bond', '[NX3][CX3](=[OX1])'), ('ester_acyl_bond', '[OX2][CX3](=[OX1])'),
    ('sulfonamide_bond', '[NX3][SX4](=[OX1])(=[OX1])'), ('phosphoramide_bond', '[NX3][PX4](=[OX1])'),
    ('aryl_halide_bond', '[c][F,Cl,Br,I]'), ('alkyl_halide_bond', '[CX4][Cl,Br,I]'),
    ('michael_bond', '[C,c]=[C,c][C,c](=O)'), ('azo_bond', '[N;!R]=N')
)

_TOX_SMARTS_STRINGS = [
    ('Acyl_halide', 'C(=O)[Cl,Br,I]'), ('Aldehyde', '[CX3H1](=O)[#6]'), ('Alkyl_halide', '[CX4][Cl,Br,I]'),
    ('Anhydride', 'C(=O)OC(=O)'), ('Aziridine', 'N1CC1'), ('Azetidine', 'N1CCC1'), ('Epoxide', 'C1OC1'),
    ('Oxetane_strained', 'C1COC1'), ('Beta_lactam', 'N1C(=O)CC1'), ('Beta_lactone', 'O=C1CCO1'),
    ('Halocarbonyl', 'C(=O)[F,Cl,Br,I]'), ('Sulfonyl_halide', 'S(=O)(=O)[Cl,Br,I]'), ('Phosphonyl_halide', 'P(=O)[Cl,Br,I]'),
    ('Acyl_cyanide', 'C(=O)C#N'), ('Isocyanate', 'N=C=O'), ('Isothiocyanate', 'N=C=S'), ('Carbodiimide', 'N=C=N'),
    ('Ketene', 'C=C=O'), ('Nitro', '[N+](=O)[O-]'), ('Nitroso', '[N]=O'), ('Nitrosamine', 'N-N=O'),
    ('Alkyl_Nitrite', 'ON=O'), ('Azo', '[N;!R]=N'), ('Diazo', '[C]=[N+]=[N-]'), ('Diazonium', '[c][N+]#N'),
    ('Hydrazine', '[NX3][NX3]'), ('Hydrazide', 'C(=O)NN'), ('Semicarbazide', 'NC(=O)NN'), ('Hydroxamic_acid', 'C(=O)NO'),
    ('N_oxide', '[N+]([O-])'), ('Carbamate', 'N-C(=O)-O'), ('Urea', 'NC(=O)N'), ('Thiol', '[SX2H]'),
    ('Disulfide', 'SS'), ('Thioaldehyde', '[CX3H1](=S)[#6]'), ('Thiocarbonyl', 'C=S'), ('Sulfonamide', 'S(=O)(=O)N'),
    ('Sulfonate_ester', 'S(=O)(=O)O[CX4]'), ('Thiocarbamate', 'N-C(=S)-O'), ('Peroxide', 'OO'),
    ('Hydroperoxide', '[OX2][OX2H]'), ('Michael_acceptor', '[C,c]=[C,c][C,c](=O)'), ('Vinyl_halide', '[CX3]=[CX3][F,Cl,Br,I]'),
    ('Alpha_halo_carbonyl', 'C(=O)C[Cl,Br,I]'), ('Activated_ester', 'C(=O)O[CX3]=[CX3]'), ('Maleimide', 'N1C(=O)C=CC1=O'),
    ('Acrylamide', '[NX3][CX3](=O)[CX3]=[CX3]'), ('Phosphonate', 'P(=O)(O)O'), ('Phosphate_ester', 'OP(=O)(O)O'),
    ('Alkyl_fluoride', '[CX4]F'), ('Aniline', '[NX3;H2,H1;!$(NC=O)]c'), ('N_N_diaryl_amine', 'N(c)c'),
    ('Phenol', '[OX2H]c'), ('Catechol', 'Oc1c(O)cccc1'), ('Hydroquinone', 'Oc1ccc(O)cc1'),
    ('Aminophenol', '[NX3;H2,H1]c1ccc(O)cc1'), ('Quinone', 'O=C1C=CC(=O)[cH,cH]1'), ('Quinone_imine', 'N=C1C=CC(=O)CC1'),
    ('Polycyclic_Aromatic', 'a1aaaa2aaaa12'), ('Halo_Aromatic', 'c[F,Cl,Br,I]'), ('Nitro_Aromatic', 'c[N+](=O)[O-]'),
    ('Nitroso_Aromatic', 'cN=O'), ('Aromatic_amine_N_oxide', 'c[N+]([O-])'), ('Mustard_nitrogen', '[CX4][Cl,Br]CCN'),
    ('Mustard_sulfur', '[CX4][Cl,Br]CCS'), ('Epihalohydrin', '[C@@H]1(CO1)C[Cl,Br,I]'), ('Lactone', 'O=C1OCC1'),
    ('Propiolactone', 'O=C1CCO1'), ('Aromatic_nitro_reduct', '[cH]1[cH][cH]c([N+](=O)[O-])[cH][cH]1'),
    ('Arylamine_acetyl', 'c[NH]C(=O)C'), ('Saponin_like', '[OX2]1[CX4][CX4][CX4][CX4][CX4]1'), ('Coumarin', 'O=C1OC2=CC=CC=C2C=C1'),
    ('Furan', 'c1ccoc1'), ('Thiophene', 'c1ccsc1'), ('Purine_like', 'c1ncnc2[nH]cnc12')
]

ATOM_REACTIVITY_DIM = len(_ATOM_REACTIVITY_SMARTS)
BOND_REACTIVITY_DIM = len(_BOND_REACTIVITY_SMARTS)
BOND_FEATURE_DIM = 11 + 16 + BOND_REACTIVITY_DIM
ADVANCED_3D_DESCRIPTOR_DIM = 114 + 273 + 12 + 60 + 224 + 210 + 12

def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

def get_atom_features(atom):
    features = one_hot_embedding(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
         'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
         'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'])
    features += one_hot_embedding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_hot_embedding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    try: features += one_hot_embedding(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_UNSPECIFIED])
    except: features += [0, 0, 1, 0]
    features += one_hot_embedding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_embedding(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
    features += [1 if atom.GetIsAromatic() else 0, atom.GetMass() * 0.01]
    return np.array(features, dtype=np.float32)

def get_bond_features(bond, precomputed_reactivity_flags=None):
    bt = bond.GetBondType()
    features = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0, 1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0, 1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
        1 if bond.GetIsConjugated() else 0, 1 if bond.IsInRing() else 0
    ]
    features += one_hot_embedding(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE])
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    bz, ez = begin.GetAtomicNum(), end.GetAtomicNum()
    ben, een = _PAULING_EN.get(bz, 0.0), _PAULING_EN.get(ez, 0.0)
    try: bq = float(begin.GetProp('_GasteigerCharge'))
    except: bq = 0.0
    try: eq = float(end.GetProp('_GasteigerCharge'))
    except: eq = 0.0
    bond_order = {Chem.rdchem.BondType.SINGLE: 1.0, Chem.rdchem.BondType.DOUBLE: 2.0, Chem.rdchem.BondType.TRIPLE: 3.0, Chem.rdchem.BondType.AROMATIC: 1.5}.get(bt, 0.0)
    
    smallest_ring = 0
    if bond.IsInRing():
        for r_size in range(3, 9):
            if bond.GetOwningMol().GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), r_size):
                smallest_ring = r_size; break
    is_rotatable_like = (bt == Chem.rdchem.BondType.SINGLE and not bond.IsInRing() and bz > 1 and ez > 1 and begin.GetDegree() > 1 and end.GetDegree() > 1)
    
    advanced = [
        bond_order / 3.0, abs(ben - een) / 4.0, (ben + een) / 8.0, abs(bq - eq), bq + eq,
        abs(float(begin.GetFormalCharge() - end.GetFormalCharge())), (bz + ez) / 236.0, abs(bz - ez) / 118.0,
        float(smallest_ring) / 8.0, 1.0 if smallest_ring in (3, 4) else 0.0, 1.0 if smallest_ring in (5, 6) else 0.0,
        1.0 if begin.GetIsAromatic() and end.GetIsAromatic() else 0.0, 1.0 if is_rotatable_like else 0.0,
        1.0 if begin.GetHybridization() != end.GetHybridization() else 0.0,
        1.0 if bz in (7, 8, 15, 16) or ez in (7, 8, 15, 16) else 0.0, 1.0 if bz in (9, 17, 35, 53) or ez in (9, 17, 35, 53) else 0.0,
    ]
    features += advanced
    features += list(precomputed_reactivity_flags) if precomputed_reactivity_flags is not None else [0.0] * BOND_REACTIVITY_DIM
    return np.array(features, dtype=np.float32)

def _compute_3d_descriptor_for_mol(mol_rdkit, seed: int) -> np.ndarray:
    try:
        m = Chem.AddHs(Chem.Mol(mol_rdkit))
        if AllChem.EmbedMolecule(m, maxAttempts=2, randomSeed=int(seed), useRandomCoords=True, clearConfs=True) != 0:
            return np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m, maxIters=50)

        def _safe_vec(fn_name, length):
            fn = getattr(rdMolDescriptors, fn_name, None)
            if fn is None: return np.zeros(length, dtype=np.float32)
            try:
                v = np.nan_to_num(np.asarray(fn(m), dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            except: v = np.zeros(0, dtype=np.float32)
            out = np.zeros(length, dtype=np.float32)
            out[:min(length, len(v))] = v[:length]
            return out

        def _safe_scalar(fn_name):
            try:
                v = float(getattr(rdMolDescriptors, fn_name)(m))
                return v if not (np.isnan(v) or np.isinf(v)) else 0.0
            except: return 0.0

        shape_scalars = np.array([
            _safe_scalar('CalcPMI1'), _safe_scalar('CalcPMI2'), _safe_scalar('CalcPMI3'), _safe_scalar('CalcNPR1'),
            _safe_scalar('CalcNPR2'), _safe_scalar('CalcPBF'), _safe_scalar('CalcAsphericity'), _safe_scalar('CalcEccentricity'),
            _safe_scalar('CalcInertialShapeFactor'), _safe_scalar('CalcRadiusOfGyration'), _safe_scalar('CalcSpherocityIndex'),
            float(AllChem.ComputeMolVolume(m)) if hasattr(AllChem, 'ComputeMolVolume') else 0.0
        ], dtype=np.float32)
        
        vec = np.concatenate([_safe_vec('CalcWHIM', 114), _safe_vec('CalcGETAWAY', 273), _safe_vec('GetUSR', 12),
                              _safe_vec('GetUSRCAT', 60), _safe_vec('CalcMORSE', 224), _safe_vec('CalcRDF', 210), shape_scalars])
        if vec.shape[0] != ADVANCED_3D_DESCRIPTOR_DIM:
            fixed = np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)
            fixed[:min(len(vec), ADVANCED_3D_DESCRIPTOR_DIM)] = vec[:ADVANCED_3D_DESCRIPTOR_DIM]
            return fixed
        return vec
    except:
        return np.zeros(ADVANCED_3D_DESCRIPTOR_DIM, dtype=np.float32)


# 2. DATASET BUILDING
def featurise_dataset(dataframe: pd.DataFrame, target_columns: list) -> list:
    """Full standardisation and featurisation of a SMILES DataFrame"""
    print("Standardising...")
    lfc = rdMolStandardize.LargestFragmentChooser()
    uc = rdMolStandardize.Uncharger()
    te = rdMolStandardize.TautomerEnumerator()
    
    mols, valid_smiles, targets = [], [], []
    for idx, row in dataframe.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None: continue
            mol = te.Canonicalize(uc.uncharge(lfc.choose(mol)))
            Chem.SanitizeMol(mol)
            mols.append(mol)
            valid_smiles.append(row['smiles'])
            targets.append([row[t] for t in target_columns])
        except: pass

    # MolFormer
    print("Generating MolFormer embeddings...")
    tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model_lm = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lm.to(device)
    
    lm_matrix = []
    with torch.no_grad():
        for i in range(0, len(valid_smiles), 128):
            inputs = tokenizer(valid_smiles[i:i+128], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            out = model_lm(**inputs).pooler_output.cpu().numpy()
            lm_matrix.append(out)
    lm_matrix = np.vstack(lm_matrix)
    del model_lm, tokenizer

    # RDKit, Tox, 3D, ECFP
    print("Calculating RDKit, Tox, and 3D Descriptors...")
    mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    pharm_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    atom_react_patts = [Chem.MolFromSmarts(s) for _, s in _ATOM_REACTIVITY_SMARTS]
    bond_react_patts = [Chem.MolFromSmarts(s) for _, s in _BOND_REACTIVITY_SMARTS]
    tox_patts = [Chem.MolFromSmarts(s) for _, s in _TOX_SMARTS_STRINGS]
    
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)

    data_list = []
    for i, mol in enumerate(mols):
        mol_h = Chem.AddHs(mol)
        mol_heavy = Chem.RemoveHs(mol)
        n_all, n_heavy = mol_h.GetNumAtoms(), mol_heavy.GetNumAtoms()
        try: AllChem.ComputeGasteigerCharges(mol_h)
        except: pass
        
        # Tox
        tox_bits = [1.0 if patt and mol.HasSubstructMatch(patt) else 0.0 for patt in tox_patts]
        tox_bits.append(float(len(pains_catalog.GetMatches(mol))))
        
        # Graph Features
        atom_feats, edge_indices, edge_attrs = [], [], []
        atom_react_flags = np.zeros((n_all, ATOM_REACTIVITY_DIM), dtype=np.float32)
        for p_idx, patt in enumerate(atom_react_patts):
            if patt: 
                for match in mol_heavy.GetSubstructMatches(patt):
                    for aidx in match: atom_react_flags[aidx, p_idx] = 1.0
                    
        for a_idx, atom in enumerate(mol_h.GetAtoms()):
            base_feat = get_atom_features(atom)
            # Simplified pad for omitted RDKit complex scalars to maintain exact dimensions
            pad_feats = np.zeros(36, dtype=np.float32) 
            atom_feats.append(np.concatenate([base_feat, pad_feats, atom_react_flags[a_idx]]))
            
        bond_react_flags = np.zeros((mol_h.GetNumBonds(), BOND_REACTIVITY_DIM), dtype=np.float32)
        for bond in mol_h.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            e_feat = get_bond_features(bond, bond_react_flags[bond.GetIdx()])
            edge_indices += [[u, v], [v, u]]; edge_attrs += [e_feat, e_feat]

        x = torch.tensor(np.array(atom_feats), dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float) if edge_attrs else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

        # Global Desc
        fp = torch.tensor(mfp_gen.GetFingerprintAsNumPy(mol_heavy), dtype=torch.float)
        desc_rdkit = torch.tensor([f(mol) if not np.isnan(f(mol)) else 0.0 for _, f in Descriptors._descList], dtype=torch.float)
        desc_3d = torch.tensor(_compute_3d_descriptor_for_mol(mol, 42), dtype=torch.float)
        desc_pubchem = torch.zeros(200, dtype=torch.float) # Requires pubchem API cache, padded for structure
        
        global_feat = torch.cat([fp, desc_rdkit, torch.tensor(tox_bits, dtype=torch.float), 
                                 torch.tensor(lm_matrix[i], dtype=torch.float), desc_3d, desc_pubchem], dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([targets[i]], dtype=torch.float))
        data.global_features = global_feat
        data_list.append(data)

    return data_list


# 3. MODEL ARCHITECTURE
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x, batch):
        if not self.training or self.drop_prob == 0.0: return x
        keep = (torch.rand(int(batch.max().item()) + 1, device=x.device) >= self.drop_prob).float()
        return x * (keep / max(1.0 - self.drop_prob, 1e-6))[batch].unsqueeze(-1)

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_scores = nn.Sequential(nn.Linear(in_channels, in_channels // 2), nn.GELU(), nn.Linear(in_channels // 2, num_heads))
        self.node_transform = nn.Linear(in_channels, in_channels)
        self.final_proj = nn.Sequential(nn.Linear((num_heads * in_channels) + (2 * in_channels), out_channels), nn.LayerNorm(out_channels), nn.GELU(), nn.Linear(out_channels, out_channels))

    def forward(self, x, batch):
        scores = self.attn_scores(x)
        weights = softmax(scores, batch, dim=0)
        head_outputs = [global_add_pool(self.node_transform(x) * weights[:, h:h+1], batch) for h in range(self.num_heads)]
        return self.final_proj(torch.cat([torch.cat(head_outputs, dim=-1), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1))

class ToxLensModel(nn.Module):
    def __init__(self, in_channels=134, hidden_channels=256, global_dim=2173, edge_dim=35, num_tasks=11, n_layers=5, drop_rate=0.2):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels, bias=False)
        self.layers, self.norms1, self.norms2, self.ffns, self.vn_mlps = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.drop_paths = nn.ModuleList([DropPath(0.2 * i / (n_layers - 1)) for i in range(n_layers)])
        
        for _ in range(n_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim))
            self.norms1.append(GraphNorm(hidden_channels)); self.norms2.append(GraphNorm(hidden_channels))
            self.ffns.append(nn.Sequential(Linear(hidden_channels, hidden_channels * 2), nn.GELU(), nn.Dropout(drop_rate * 0.5), Linear(hidden_channels * 2, hidden_channels)))
            self.vn_mlps.append(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU()))
            
        self.pool = MultiHeadAttentionPooling(hidden_channels, hidden_channels)
        self.global_proj = nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU())
        self.shared_trunk = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(), nn.Dropout(drop_rate), nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU())
        self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(), nn.Dropout(drop_rate*0.25), nn.Linear(hidden_channels, 1)) for _ in range(num_tasks)])

    def forward(self, batch):
        x, edge_index, edge_attr, b_idx = self.node_emb(batch.x), batch.edge_index, batch.edge_attr, batch.batch
        num_graphs = int(b_idx.max().item() + 1)
        virtual_node = x.new_zeros((num_graphs, x.size(-1)))
        
        for i in range(len(self.layers)):
            x = x + virtual_node[b_idx]
            x = x + self.drop_paths[i](self.layers[i](self.norms1[i](x, b_idx), edge_index, edge_attr=edge_attr), b_idx)
            x = x + self.drop_paths[i](self.ffns[i](self.norms2[i](x, b_idx)), b_idx)
            virtual_node = virtual_node + self.vn_mlps[i](global_add_pool(x, b_idx))

        graph_emb = self.pool(x, b_idx)
        global_emb = self.global_proj(batch.global_features.view(num_graphs, -1))
        shared = self.shared_trunk(torch.cat([graph_emb, global_emb], dim=-1))
        return torch.cat([head(shared) for head in self.task_heads], dim=-1)


# 4. LOSS & TRAINING LOOP
class UnweightedMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.eps = 0.05
    def forward(self, preds, targets):
        losses = []
        for t in range(self.num_tasks):
            p, y = preds[:, t], targets[:, t]
            mask = (~torch.isnan(y)) & (y != -1.0)
            if mask.sum() > 0:
                y_smooth = y[mask] * (1.0 - self.eps) + 0.5 * self.eps
                losses.append(F.binary_cross_entropy_with_logits(p[mask], y_smooth))
        return torch.stack(losses).mean() if losses else preds.sum() * 0.0

def train_and_infer():
    # Setup dummy data for demonstration
    df = pd.DataFrame({
        'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O'],
        'task1': [0, 1, np.nan],
        'task2': [1, 0, 1]
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = featurise_dataset(df, target_columns=['task1', 'task2'])
    loader = DataLoader(dataset, batch_size=2, collate_fn=Batch.from_data_list)
    
    global_dim = dataset[0].global_features.shape[0]
    model = ToxLensModel(num_tasks=2, global_dim=global_dim).to(device)
    criterion = UnweightedMultiTaskLoss(num_tasks=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        
    # Infer
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    print("Predictions:", np.vstack(preds))


if __name__ == "__main__":
    train_and_infer()
