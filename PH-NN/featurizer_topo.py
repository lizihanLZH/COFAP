import os
import csv
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import gudhi as gd

class COFFeaturizer:
    """
    Featurizer for COF structures combining:
    - Topological fingerprint (persistent homology)
    - Structural descriptors (PLD, LCD, surface area, density, porosity)
    """
    
    def __init__(
        self,
        struct_features_csv: str,
        topo_cache=None,
    ):
        # load structural features
        self.struct_df = pd.read_csv(struct_features_csv)
        
        # cache for topological fingerprints by topology type
        self.topo_cache = topo_cache if topo_cache is not None else {}
    
    def extract_topological_fingerprint(self, cif_path: str, topo_type: str) -> np.ndarray:
        """Extracts persistent homology 0D/1D persistence as fingerprint"""
        if topo_type in self.topo_cache:
            return self.topo_cache[topo_type]
        
        struct = Structure.from_file(cif_path)
        coords = struct.cart_coords
        
        # build Rips complex
        rips = gd.RipsComplex(points=coords, max_edge_length=10.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.1)
        
        # vectorize persistence diagram: use binned persistence instead of entropy for fixed dimension
        # Create bins for persistence values
        pers_0 = [p[1][1] - p[1][0] for p in diag if p[0] == 0 and p[1][1] != float('inf')]
        pers_1 = [p[1][1] - p[1][0] for p in diag if p[0] == 1 and p[1][1] != float('inf')]
        
        # Create histogram features with fixed bins
        bins = np.linspace(0, 5, 10)  # 10 bins for each homology group
        hist_0 = np.histogram(pers_0, bins=bins)[0] if pers_0 else np.zeros(9)
        hist_1 = np.histogram(pers_1, bins=bins)[0] if pers_1 else np.zeros(9)
        
        # Concatenate to get fixed 18-dimensional feature
        pers = np.concatenate([hist_0, hist_1]).astype(np.float32)
        
        self.topo_cache[topo_type] = pers
        return pers
    
    def load_structural_features(self, cof_name: str) -> np.ndarray:
        """Retrieves PLD, LCD, surface area, density, porosity for given COF"""
        row = self.struct_df[self.struct_df['name'] == cof_name]
        
        if row.empty:
            return np.zeros(5)
        
        vals = row[['PLD', 'LCD', 'surface_area', 'density', 'porosity']].values.flatten()
        return vals
    
    def featurize(self, name: str, cif_dir: str) -> dict:
        """Complete featurization pipeline for one COF, returns separate features"""
        cif_path = os.path.join(cif_dir, f"{name}.cif")
        
        # parse topology from name, e.g., XXX_qtz -> qtz
        topo_type = name.split('_')[-4]
        
        topo_fp = self.extract_topological_fingerprint(cif_path, topo_type)
        struct_fp = self.load_structural_features(name)
        
        # return as dictionary for separate processing
        return {
            'topo': topo_fp,
            'struct': struct_fp
        }

# Example usage:
# fe = COFFeaturizer('struct_features.csv')
# features = fe.featurize('linker100_CH2_linker3_NH_qtz_relaxed_interp_2', 'structures/')
# topo_feat = features['topo']
# struct_feat = features['struct']
