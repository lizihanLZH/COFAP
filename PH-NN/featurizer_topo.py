import os
import csv
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import gudhi as gd
class COFFeaturizer:
    def __init__(
        self,
        struct_features_csv: str,
        topo_cache=None,
    ):
        self.struct_df = pd.read_csv(struct_features_csv)
        self.topo_cache = topo_cache if topo_cache is not None else {}
    def extract_topological_fingerprint(self, cif_path: str, topo_type: str) -> np.ndarray:
        if topo_type in self.topo_cache:
            return self.topo_cache[topo_type]
        struct = Structure.from_file(cif_path)
        coords = struct.cart_coords
        rips = gd.RipsComplex(points=coords, max_edge_length=10.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.1)
        pers_0 = [p[1][1] - p[1][0] for p in diag if p[0] == 0 and p[1][1] != float('inf')]
        pers_1 = [p[1][1] - p[1][0] for p in diag if p[0] == 1 and p[1][1] != float('inf')]
        bins = np.linspace(0, 5, 10)  
        hist_0 = np.histogram(pers_0, bins=bins)[0] if pers_0 else np.zeros(9)
        hist_1 = np.histogram(pers_1, bins=bins)[0] if pers_1 else np.zeros(9)
        pers = np.concatenate([hist_0, hist_1]).astype(np.float32)
        self.topo_cache[topo_type] = pers
        return pers
    def load_structural_features(self, cof_name: str) -> np.ndarray:
        row = self.struct_df[self.struct_df['name'] == cof_name]
        if row.empty:
            return np.zeros(5)
        vals = row[['PLD', 'LCD', 'surface_area', 'density', 'porosity']].values.flatten()
        return vals
    def featurize(self, name: str, cif_dir: str) -> dict:
        cif_path = os.path.join(cif_dir, f"{name}.cif")
        topo_type = name.split('_')[-4]
        topo_fp = self.extract_topological_fingerprint(cif_path, topo_type)
        struct_fp = self.load_structural_features(name)
        return {
            'topo': topo_fp,
            'struct': struct_fp
        }