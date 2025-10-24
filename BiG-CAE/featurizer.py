import os 
import uuid
import pandas as pd
import numpy as np
import torch
import dgl
from pathlib import Path
from functools import partial, lru_cache
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from gensim.models import word2vec
from collections import defaultdict
import networkx as nx
import re
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

GENSIM_MODEL = word2vec.Word2Vec.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/mol2vec_300dim.pkl"
    )
)

CHEMICAL_PROPERTIES = {
    'C': {'electronegativity': 2.55, 'atomic_radius': 0.77, 'valence': 4, 'atomic_mass': 12.01},
    'N': {'electronegativity': 3.04, 'atomic_radius': 0.75, 'valence': 3, 'atomic_mass': 14.01},
    'O': {'electronegativity': 3.44, 'atomic_radius': 0.73, 'valence': 2, 'atomic_mass': 16.00},
    'H': {'electronegativity': 2.20, 'atomic_radius': 0.37, 'valence': 1, 'atomic_mass': 1.01},
    'S': {'electronegativity': 2.58, 'atomic_radius': 1.02, 'valence': 2, 'atomic_mass': 32.07},
    'P': {'electronegativity': 2.19, 'atomic_radius': 1.06, 'valence': 3, 'atomic_mass': 30.97},
    'F': {'electronegativity': 3.98, 'atomic_radius': 0.71, 'valence': 1, 'atomic_mass': 19.00},
    'Cl': {'electronegativity': 3.16, 'atomic_radius': 0.99, 'valence': 1, 'atomic_mass': 35.45},
    'Br': {'electronegativity': 2.96, 'atomic_radius': 1.14, 'valence': 1, 'atomic_mass': 79.90},
    'I': {'electronegativity': 2.66, 'atomic_radius': 1.33, 'valence': 1, 'atomic_mass': 126.90},
    'Si': {'electronegativity': 1.90, 'atomic_radius': 1.11, 'valence': 4, 'atomic_mass': 28.09},
}

VALID_CONNECTION_COMBINATIONS = {
    'CO_NH': {'pattern': 'CONH', 'description': 'Amide bond (C=O + N-H)'},
    'NH_CO': {'pattern': 'NHCO', 'description': 'Amide bond (N-H + C=O)'},
    'CH_N': {'pattern': 'CHN', 'description': 'Imine bond (C-H + N)'},
    'N_CH': {'pattern': 'NCH', 'description': 'Imine bond (N + C-H)'},
    'NH_CH2': {'pattern': 'NHCH2', 'description': 'Amine-alkyl bond (N-H + C-H2)'},
    'CH2_NH': {'pattern': 'CH2NH', 'description': 'Alkyl-amine bond (C-H2 + N-H)'},
}

PROP_ARRAYS = {
    elem: np.array([props['electronegativity'], props['atomic_radius'], props['valence'], props['atomic_mass']])
    for elem, props in CHEMICAL_PROPERTIES.items()
}

@lru_cache(maxsize=1)
def load_linkers_csv(csv_path="linkers.csv"):
    df = pd.read_csv(csv_path)
    linker_to_smiles = {}
    for _, row in df.iterrows():
        linker_name = row.iloc[0]
        smiles = row.iloc[1]
        linker_to_smiles[linker_name] = smiles
    return linker_to_smiles

@lru_cache(maxsize=128)
def get_structure_graph(structure_id, cif_path):
    parser = CifParser(cif_path)
    structure = parser.get_structures(primitive=False)[0]
    nn = CrystalNN()
    sg = StructureGraph.with_local_env_strategy(structure, nn)
    return structure, sg

def identify_aromatic_atoms_fast(structure):
    coords = np.array([site.coords for site in structure])
    elements = np.array([site.specie.symbol for site in structure])
    
    cn_mask = np.isin(elements, ['C', 'N'])
    cn_coords = coords[cn_mask]
    cn_indices = np.where(cn_mask)[0]
    
    if len(cn_coords) < 2:
        return set(), set(range(len(structure)))
    
    tree = cKDTree(cn_coords)
    distances = tree.sparse_distance_matrix(tree, max_distance=1.7, output_type='coo_matrix')
    
    neighbor_counts = np.zeros(len(cn_coords), dtype=int)
    for i, j, d in zip(distances.row, distances.col, distances.data):
        if i != j and 0.9 <= d <= 1.7:
            neighbor_counts[i] += 1
    
    candidate_mask = neighbor_counts >= 2
    candidate_indices = cn_indices[candidate_mask]
    candidate_coords = cn_coords[candidate_mask]
    candidate_elements = elements[cn_indices][candidate_mask]
    
    if len(candidate_coords) < 2:
        return set(), set(range(len(structure)))
    
    candidate_tree = cKDTree(candidate_coords)
    pair_distances = candidate_tree.sparse_distance_matrix(candidate_tree, max_distance=1.45, output_type='coo_matrix')
    
    bond_counts = np.zeros(len(candidate_coords), dtype=int)
    for i, j, d in zip(pair_distances.row, pair_distances.col, pair_distances.data):
        if i >= j:
            continue
        elem_i, elem_j = candidate_elements[i], candidate_elements[j]
        if elem_i == 'C' and elem_j == 'C' and 1.387 <= d <= 1.45:
            bond_counts[i] += 1
            bond_counts[j] += 1
        elif {elem_i, elem_j} == {'C', 'N'} and 1.35 <= d <= 1.37:
            bond_counts[i] += 1
            bond_counts[j] += 1
    
    aromatic_mask = bond_counts >= 2
    aromatic_indices = set(candidate_indices[aromatic_mask])
    non_aromatic_indices = set(range(len(structure))) - aromatic_indices
    
    return aromatic_indices, non_aromatic_indices

def find_functional_groups_fast(structure, group_pattern, aromatic_atoms, non_aromatic_atoms):
    if group_pattern == 'CC':
        groups = find_cc_groups_fast(structure, aromatic_atoms, non_aromatic_atoms)
    elif group_pattern in ['CONH', 'NHCO']:
        groups = find_cn_groups_fast(structure, 1.36, 1.37, 'amide')
    elif group_pattern in ['CHN', 'NCH']:
        groups = find_cn_groups_fast(structure, 1.34, 1.355, 'imine')
    elif group_pattern in ['NHCH2', 'CH2NH']:
        groups = find_cn_groups_fast(structure, 1.45, 1.49, 'amine_alkyl')
    else:
        return []

    # 新增：补齐H/O原子
    def find_nearby_atoms(structure, base_indices, target_elements, num_needed, max_dist=3.0):
        coords = np.array([structure[i].coords for i in base_indices])
        all_coords = np.array([site.coords for site in structure])
        all_elements = np.array([site.specie.symbol for site in structure])
        candidate_indices = [i for i, e in enumerate(all_elements) if e in target_elements and i not in base_indices]
        if not candidate_indices:
            return []
        candidate_coords = all_coords[candidate_indices]
        dmat = np.linalg.norm(coords[:, None, :] - candidate_coords[None, :, :], axis=-1)
        # 展平成一维，按距离排序
        flat = [(candidate_indices[j], dmat[i, j]) for i in range(len(base_indices)) for j in range(len(candidate_indices))]
        flat = sorted(flat, key=lambda x: x[1])
        selected = []
        used = set()
        for idx, dist in flat:
            if idx not in used:
                selected.append(idx)
                used.add(idx)
            if len(selected) >= num_needed:
                break
        return selected

    # pattern类型对应需要补齐的H/O个数
    pattern_to_add = {
        'CONH': {'O': 1, 'H': 1},
        'NHCO': {'O': 1, 'H': 1},
        'CHN': {'H': 1},
        'NCH': {'H': 1},
        'NHCH2': {'H': 2},
        'CH2NH': {'H': 2},
    }
    for group in groups:
        add_info = pattern_to_add.get(group_pattern, {})
        base_indices = group['atoms']
        added_indices = []
        for elem, num in add_info.items():
            found = find_nearby_atoms(structure, base_indices, [elem], num)
            added_indices.extend(found)
        group['atoms'] = base_indices + added_indices
        if added_indices:
            group['type'] = group.get('type', '') + f"_add_{'_'.join([structure[i].specie.symbol+str(i) for i in added_indices])}"
    return groups

def find_functional_groups_with_fallback(structure, group_pattern, aromatic_atoms, non_aromatic_atoms):
    found_groups = find_functional_groups_fast(structure, group_pattern, aromatic_atoms, non_aromatic_atoms)
    
    if not found_groups and group_pattern == 'CC':
        coords = np.array([site.coords for site in structure])
        elements = np.array([site.specie.symbol for site in structure])
        all_atoms = aromatic_atoms.union(non_aromatic_atoms)
        c_atoms = [i for i in all_atoms if elements[i] == 'C']
        c_coords = coords[c_atoms]
        dmat = cdist(c_coords, c_coords)
        groups2 = []
        n_c = len(c_atoms)
        for i in range(n_c):
            for j in range(i+1, n_c):
                if 1.418 <= dmat[i,j] <= 1.530:
                    groups2.append({
                        'atoms': [c_atoms[i], c_atoms[j]],
                        'pattern': 'CC',
                        'center': c_atoms[i],
                        'type': 'cc_bond_short'
                    })
                elif 1.400 <= dmat[i,j] <= 1.410:
                    groups2.append({
                        'atoms': [c_atoms[i], c_atoms[j]],
                        'pattern': 'CC',
                        'center': c_atoms[i],
                        'type': 'cc_bond_tight'
                    })
        found_groups = groups2
    
    if not found_groups and group_pattern in ['CHN', 'NCH']:
        groups1 = find_cn_groups_fast(structure, 1.360, 1.365, 'imine_long')
        groups2 = find_cn_groups_fast(structure, 1.28, 1.30, 'imine_short')
        found_groups = groups1 + groups2
        
    if not found_groups and group_pattern in ['NHCH2', 'CH2NH']:
        groups2 = find_cn_groups_fast(structure, 1.48, 1.50, 'amine_alkyl_long')
        found_groups = groups2
    
    return found_groups

def find_cc_groups_fast(structure, aromatic_atoms, non_aromatic_atoms):
    coords = np.array([site.coords for site in structure])
    elements = np.array([site.specie.symbol for site in structure])
    
    all_atoms = aromatic_atoms.union(non_aromatic_atoms)
    c_atoms = [i for i in all_atoms if elements[i] == 'C']
    
    if len(c_atoms) < 2:
        return []
    
    c_coords = coords[c_atoms]
    distances = cdist(c_coords, c_coords)
    
    found_groups = []
    n_c = len(c_atoms)
    for i in range(n_c):
        for j in range(i+1, n_c):
            if 1.425 <= distances[i, j] <= 1.530:
                atoms = [c_atoms[i], c_atoms[j]]
                found_groups.append({
                    'atoms': atoms,
                    'pattern': 'CC',
                    'center': c_atoms[i],
                    'type': 'cc_bond'
                })
    
    return found_groups

def find_cn_groups_fast(structure, min_dist, max_dist, bond_type):
    coords = np.array([site.coords for site in structure])
    elements = np.array([site.specie.symbol for site in structure])
    
    c_indices = np.where(elements == 'C')[0]
    n_indices = np.where(elements == 'N')[0]
    
    if len(c_indices) == 0 or len(n_indices) == 0:
        return []
    
    c_coords = coords[c_indices]
    n_coords = coords[n_indices]
    distances = cdist(c_coords, n_coords)
    
    found_groups = []
    for i, c_idx in enumerate(c_indices):
        for j, n_idx in enumerate(n_indices):
            if min_dist <= distances[i, j] <= max_dist:
                atoms = [c_idx, n_idx]
                found_groups.append({
                    'atoms': atoms,
                    'pattern': 'CN',
                    'center': c_idx,
                    'type': bond_type
                })
    
    return found_groups

def calculate_group_features_vectorized(structure, group_info):
    import numpy as np
    from pymatgen.core.structure import Element
    from matminer.utils.data import MatscholarElementData
    from rdkit import Chem
    from rdkit.Chem import AllChem

    atoms = group_info['atoms']
    pattern = group_info['pattern']
    group_type = group_info.get('type', 'unknown')

    # 1. matminer属性统计
    el_data = MatscholarElementData()
    elements = [structure[i].specie for i in atoms]
    prop_names = el_data.prop_names
    matminer_feats = []
    for prop in prop_names:
        values = [el_data.get_elemental_property(e, prop) for e in elements]
        matminer_feats.extend([
            np.mean(values), np.std(values), np.max(values), np.min(values)
        ])
    matminer_feats = np.array(matminer_feats)

    # 2. 距离分布统计
    coords = np.array([structure[i].coords for i in atoms])
    if len(coords) < 2:
        dist_feats = np.zeros(5)
    else:
        dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        upper = dists[np.triu_indices(len(coords), k=1)]
        dist_feats = np.array([
            np.mean(upper), np.std(upper), np.max(upper), np.min(upper), np.median(upper)
        ])

    # 3. RDKit分子指纹（Morgan指纹，半径2，nBits=64）
    # 构造SMILES
    try:
        from pymatgen.io.xyz import XYZ
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            xyz = XYZ(structure.sites)
            xyz.write_file(tmp.name)
            tmp.flush()
            # 只保留基团原子
            with open(tmp.name, 'r') as f:
                lines = f.readlines()
            atom_lines = [lines[0], lines[1]] + [lines[i+2] for i in atoms]
            smiles = None
            try:
                from openbabel import pybel
                with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp2:
                    tmp2.writelines([l.encode() for l in atom_lines])
                    tmp2.flush()
                    mols = list(pybel.readfile('xyz', tmp2.name))
                    if mols:
                        smiles = mols[0].write('can').strip()
            except Exception:
                smiles = None
    except Exception:
        smiles = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=64)
            morgan_feats = np.array(list(fp), dtype=np.float32)
        else:
            morgan_feats = np.zeros(64)
    else:
        morgan_feats = np.zeros(64)

    # 4. pattern编码（保留原有实现）
    pattern_features = encode_pattern_features_fast(pattern, group_type)

    # 5. 拼接所有特征
    features = np.concatenate([
        matminer_feats,
        dist_feats,
        morgan_feats,
        pattern_features
    ])
    # 6. 保证特征维度为200
    if len(features) < 200:
        features = np.concatenate([features, np.zeros(200 - len(features))])
    else:
        features = features[:200]
    return features.astype(np.float32)

def encode_pattern_features_fast(pattern, group_type):
    features = [len(pattern)]
    
    type_encodings = {
        'amide': 0, 'imine': 1, 'amine_alkyl': 2, 'amide_aromatic': 3,
        'imine_aromatic': 4, 'amine_alkyl_aromatic': 5, 'cc_bond': 6,'cc_bond_short': 7, 'imine_short': 8, 'amine_alkyl_long': 9,'cc_bond_tight': 10,'imine_long': 11
    }
    type_vec = np.zeros(12)
    if group_type in type_encodings:
        type_vec[type_encodings[group_type]] = 1.0
    features.extend(type_vec)
    
    common_groups = ['CH', 'NH', 'OH', 'CO', 'SH', 'CN', 'NO', 'CC']
    group_presence = [1.0 if group in pattern else 0.0 for group in common_groups]
    features.extend(group_presence)
    
    features.append(len(set(pattern)))
    
    return np.array(features)

def calculate_chemical_environment_features_fast(structure, atoms):
    coords = np.array([site.coords for site in structure])
    elements = np.array([site.specie.symbol for site in structure])
    atom_coords = coords[atoms]
    
    distances = cdist(atom_coords, coords)
    neighbor_mask = (distances < 3.0).any(axis=0)
    neighbor_mask[atoms] = False
    
    neighbor_elements = elements[neighbor_mask]
    
    element_counts = np.zeros(7)
    element_map = {'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4, 'F': 5, 'Si': 6}
    
    for elem in neighbor_elements:
        if elem in element_map:
            element_counts[element_map[elem]] += 1
    
    features = np.concatenate([element_counts, [np.sum(element_counts)]])
    return features

def calculate_geometric_features_fast(structure, atoms):
    if len(atoms) < 2:
        return np.zeros(10)
    
    coords = np.array([structure[i].coords for i in atoms])
    centroid = np.mean(coords, axis=0)
    
    distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    features = [
        np.mean(distances_to_centroid),
        np.std(distances_to_centroid),
        np.min(distances_to_centroid),
        np.max(distances_to_centroid),
        np.median(distances_to_centroid)
    ]
    
    if len(atoms) > 1:
        pairwise_distances = cdist(coords, coords)
        upper_tri = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
        
        if len(upper_tri) > 0:
            features.extend([
                np.mean(upper_tri),
                np.std(upper_tri),
                np.min(upper_tri),
                np.max(upper_tri),
                np.median(upper_tri)
            ])
        else:
            features.extend([0.0] * 5)
    else:
        features.extend([0.0] * 5)
    
    return np.array(features)

def add_charges(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    problems = Chem.DetectChemistryProblems(mol)
    if not problems:
        Chem.SanitizeMol(mol)
        return mol

    for problem in problems:
        if problem.GetType() == "AtomValenceException":
            atom = mol.GetAtomWithIdx(problem.GetAtomIdx())
            if (atom.GetAtomicNum() == 5 and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4):
                atom.SetFormalCharge(1)
            if (atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4):
                atom.SetFormalCharge(1)

    Chem.SanitizeMol(mol)
    return mol

def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)

def mol2vec_featurize_batch(smiles_linkers, model):
    def sentences2vec_batch(sentences, model, unseen=None):
        keys = set(model.wv.key_to_index.keys())
        vecs = []
        if unseen:
            unseen_vec = model.wv.get_vector(unseen)
        
        for sentence in sentences:
            if unseen:
                sentence_vecs = [
                    model.wv.get_vector(y) if y in keys else unseen_vec
                    for y in sentence
                ]
            else:
                sentence_vecs = [
                    model.wv.get_vector(y) for y in sentence if y in keys
                ]
            
            if sentence_vecs:
                vecs.append(np.sum(sentence_vecs, axis=0))
            else:
                vecs.append(np.zeros(model.wv.vector_size))
                
        return np.array(vecs)

    sentences = [mol2alt_sentence(Chem.MolFromSmiles(smi), 1) for smi in smiles_linkers]
    features = sentences2vec_batch(sentences, model, "UNK")
    return features

def parse_cof_filename(filename):
    parts = filename.split('_')
    
    connection_groups = []
    linker_names = []
    
    if len(parts) >= 4:
        linker_indices = [i for i, part in enumerate(parts) if part.startswith('linker')]
        
        if len(linker_indices) >= 2:
            linker1 = parts[linker_indices[0]]
            connection1 = parts[linker_indices[0] + 1] if linker_indices[0] + 1 < len(parts) else ""
            
            linker2 = parts[linker_indices[1]]
            connection2 = parts[linker_indices[1] + 1] if linker_indices[1] + 1 < len(parts) else ""
            
            if connection1 == 'C' and connection2 == 'C':
                linker_names = [linker1, linker2]
                connection_groups = [connection1, connection2, 'CC']
                return linker_names, connection_groups, False
            
            combo_key = f"{connection1}_{connection2}"
            reverse_combo_key = f"{connection2}_{connection1}"
            
            if combo_key in VALID_CONNECTION_COMBINATIONS:
                combined_group = VALID_CONNECTION_COMBINATIONS[combo_key]['pattern']
            elif reverse_combo_key in VALID_CONNECTION_COMBINATIONS:
                combined_group = VALID_CONNECTION_COMBINATIONS[reverse_combo_key]['pattern']
            else:
                print(f"Warning: Invalid connection combination {connection1}_{connection2}")
                return None, None, True
            
            linker_names = [linker1, linker2]
            connection_groups = [connection1, connection2, combined_group]
    
    return linker_names, connection_groups, False

def get_supra_graph_cof_fast(cif_path, linkers_csv_path="linkers.csv"):
    linker_to_smiles = load_linkers_csv(linkers_csv_path)
    
    filename = os.path.basename(cif_path)
    linker_names, connection_groups, should_skip = parse_cof_filename(filename)
    
    if should_skip or not linker_names or not connection_groups:
        return [], [], [], []
    
    try:
        structure_id = hash(cif_path)
        structure, sg = get_structure_graph(structure_id, cif_path)
    except Exception as e:
        print(f"Error reading CIF file {cif_path}: {e}")
        return [], [], [], []
    
    aromatic_atoms, non_aromatic_atoms = identify_aromatic_atoms_fast(structure)
    
    combined_group = connection_groups[2]
    found_groups = find_functional_groups_with_fallback(structure, combined_group, aromatic_atoms, non_aromatic_atoms)
    
    if not found_groups:
        print(f"No functional groups found for pattern {combined_group} in {cif_path}")
        return [], [], [], []
    
    node_features = []
    for group in found_groups:
        features = calculate_group_features_vectorized(structure, group)
        node_features.append(features)
    
    smiles_linkers = []
    for linker_name in linker_names:
        if linker_name in linker_to_smiles:
            smiles = linker_to_smiles[linker_name]
            smiles_linkers.append(smiles)
        else:
            print(f"Warning: {linker_name} not found in linkers.csv")
            smiles_linkers.append("C")
    
    linker_features = mol2vec_featurize_batch(smiles_linkers, GENSIM_MODEL)
    
    linker_src, node_dst = build_edges_with_features(found_groups, linker_features)
    
    return node_features, linker_features, linker_src, node_dst

def build_edges_with_features(groups, linker_features):
    num_groups = len(groups)
    num_linkers = len(linker_features)
    
    linker_src = np.repeat(np.arange(num_linkers), num_groups)
    node_dst = np.tile(np.arange(num_groups), num_linkers)
    
    return linker_src.tolist(), node_dst.tolist()

def get_2cg_inputs_cof(cif_path, linkers_csv_path="linkers.csv"):
    node_features, linker_features, linker_src, node_dst = get_supra_graph_cof_fast(
        cif_path, linkers_csv_path
    )

    if not node_features or len(linker_features) == 0:
        return dgl.heterograph({
            ("l", "l2n", "n"): (torch.tensor([]), torch.tensor([])),
            ("n", "n2l", "l"): (torch.tensor([]), torch.tensor([])),
        })

    node_features_tensor = torch.tensor(np.array(node_features), dtype=torch.float32)
    linker_features_tensor = torch.tensor(linker_features, dtype=torch.float32)

    graph_data = {
        ("l", "l2n", "n"): (torch.tensor(linker_src), torch.tensor(node_dst)),
        ("n", "n2l", "l"): (torch.tensor(node_dst), torch.tensor(linker_src)),
    }

    g = dgl.heterograph(graph_data)
    g.nodes["l"].data["feat"] = linker_features_tensor
    g.nodes["n"].data["feat"] = node_features_tensor

    return g

__all__ = [
    'parse_cof_filename',
    'load_linkers_csv',
    'get_2cg_inputs_cof',
    'identify_aromatic_atoms_fast',
    'find_functional_groups_fast',
    'VALID_CONNECTION_COMBINATIONS'
]

