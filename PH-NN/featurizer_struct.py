import os
import warnings
import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from TopoFeaturizer import COFFeaturizer
from pymatgen.core import Structure
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
class CombinedCOFFeaturizer:
    def __init__(self, struct_features_csv: str, target_csv: str, cache_dir: str = "feature_cache"):
        self.cof_featurizer = COFFeaturizer(struct_features_csv)
        self.target_df = pd.read_csv(target_csv)
        self._struct_csv = struct_features_csv
        self._target_csv = target_csv
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.topo_cache_file = self.cache_dir / "topo_features.pkl"
        self.struct_cache_file = self.cache_dir / "struct_features.pkl"
        self.target_cache_file = self.cache_dir / "targets.pkl"
        self.metadata_cache_file = self.cache_dir / "cache_metadata.pkl"
        self._load_cache()
        print(f"Feature cache directory: {self.cache_dir}")
        print(f"Cache status: Topo={self._cache_loaded['topo']}, Struct={self._cache_loaded['struct']}, Targets={self._cache_loaded['targets']}")
    def _load_cache(self):
        self._cache_loaded = {
            'topo': False,
            'struct': False,
            'targets': False
        }
        try:
            if (self.topo_cache_file.exists() and 
                self.struct_cache_file.exists() and 
                self.target_cache_file.exists() and
                self.metadata_cache_file.exists()):
                with open(self.metadata_cache_file, 'rb') as f:
                    metadata = pickle.load(f)
                if self._is_cache_valid(metadata):
                    with open(self.topo_cache_file, 'rb') as f:
                        self._cached_topo = pickle.load(f)
                    with open(self.struct_cache_file, 'rb') as f:
                        self._cached_struct = pickle.load(f)
                    with open(self.target_cache_file, 'rb') as f:
                        self._cached_targets = pickle.load(f)
                    self._cache_loaded = {
                        'topo': True,
                        'struct': True,
                        'targets': True
                    }
                    print(f"✅ Loaded cached features: {len(self._cached_topo)} samples")
                else:
                    print("⚠️ Cache is outdated, will regenerate features")
            else:
                print("📝 No existing cache found, will create new cache")
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            print("📝 Will create new cache")
    def _is_cache_valid(self, metadata):
        try:
            struct_csv_mtime = os.path.getmtime(self._struct_csv)
            target_csv_mtime = os.path.getmtime(self._target_csv)
            if (metadata.get('struct_csv_mtime', 0) >= struct_csv_mtime and
                metadata.get('target_csv_mtime', 0) >= target_csv_mtime):
                return True
            else:
                return False
        except Exception:
            return False
    def _save_cache(self, topo_features, struct_features, targets):
        try:
            metadata = {
                'struct_csv_mtime': os.path.getmtime(self._struct_csv),
                'target_csv_mtime': os.path.getmtime(self._target_csv),
                'cache_created': pd.Timestamp.now(),
                'num_samples': len(topo_features),
                'feature_dims': {
                    'topo': list(topo_features.values())[0].shape[0] if topo_features else 0,
                    'struct': list(struct_features.values())[0].shape[0] if struct_features else 0
                }
            }
            with open(self.topo_cache_file, 'wb') as f:
                pickle.dump(topo_features, f)
            with open(self.struct_cache_file, 'wb') as f:
                pickle.dump(struct_features, f)
            with open(self.target_cache_file, 'wb') as f:
                pickle.dump(targets, f)
            with open(self.metadata_cache_file, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"✅ Features cached successfully: {len(topo_features)} samples")
            print(f"   Cache location: {self.cache_dir}")
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    def _get_cache_key(self, cof_name: str, cif_dir: str) -> str:
        try:
            cif_file = Path(cif_dir) / f"{cof_name}.cif"
            if cif_file.exists():
                mtime = os.path.getmtime(cif_file)
                key_data = f"{cof_name}_{mtime}"
            else:
                key_data = cof_name
        except:
            key_data = cof_name
        return hashlib.md5(key_data.encode()).hexdigest()
    def get_features_and_target(self, cof_name: str, cif_dir: str) -> dict:
        topo_features = self.cof_featurizer.featurize(cof_name, cif_dir)
        target_row = self.target_df[self.target_df.iloc[:,0] == cof_name]
        if target_row.empty:
            target = 0.0
        else:
            target = target_row.iloc[0, 1]
        return {
            'topo': topo_features['topo'],
            'struct': topo_features['struct'],
            'target': target
        }
    def batch_featurize(self, cof_names: list, cif_dir: str, num_workers: int = 8, use_cache: bool = True) -> dict:
        print(f"Starting batch featurization for {len(cof_names)} COFs...")
        if use_cache and self._cache_loaded['topo'] and self._cache_loaded['struct'] and self._cache_loaded['targets']:
            print("🔍 Checking cache compatibility...")
            cached_cofs = set(self._cached_topo.keys())
            requested_cofs = set(cof_names)
            if requested_cofs.issubset(cached_cofs):
                print(f"✅ All {len(cof_names)} COFs found in cache, using cached features")
                topo_features = []
                struct_features = []
                targets = []
                for cof_name in cof_names:
                    topo_features.append(self._cached_topo[cof_name])
                    struct_features.append(self._cached_struct[cof_name])
                    targets.append(self._cached_targets[cof_name])
                return {
                    'topo': np.array(topo_features),
                    'struct': np.array(struct_features),
                    'targets': np.array(targets)
                }
            else:
                missing_cofs = requested_cofs - cached_cofs
                print(f"⚠️ {len(missing_cofs)} COFs not in cache, will extract features for missing samples")
                missing_features = self._extract_missing_features(missing_cofs, cif_dir, num_workers)
                topo_features = []
                struct_features = []
                targets = []
                for cof_name in cof_names:
                    if cof_name in cached_cofs:
                        topo_features.append(self._cached_topo[cof_name])
                        struct_features.append(self._cached_struct[cof_name])
                        targets.append(self._cached_targets[cof_name])
                    else:
                        topo_features.append(missing_features[cof_name]['topo'])
                        struct_features.append(missing_features[cof_name]['struct'])
                        targets.append(missing_features[cof_name]['target'])
                self._update_cache(missing_features)
                return {
                    'topo': np.array(topo_features),
                    'struct': np.array(struct_features),
                    'targets': np.array(targets)
                }
        print("🔄 Extracting features for all COFs...")
        batch_features = self._extract_all_features(cof_names, cif_dir, num_workers)
        if use_cache:
            print(f"🔍 Debug: batch_features keys: {list(batch_features.keys())}")
            print(f"🔍 Debug: batch_features['topo'] type: {type(batch_features['topo'])}")
            print(f"🔍 Debug: batch_features['topo'] shape: {batch_features['topo'].shape if hasattr(batch_features['topo'], 'shape') else 'no shape'}")
            print(f"🔍 Debug: cof_names type: {type(cof_names)}")
            print(f"🔍 Debug: cof_names length: {len(cof_names)}")
            assert len(cof_names) == len(batch_features['topo']), f"COF names ({len(cof_names)}) and features ({len(batch_features['topo'])}) count mismatch"
            topo_cache = {}
            struct_cache = {}
            targets_cache = {}
            for i, cof_name in enumerate(cof_names):
                topo_cache[cof_name] = batch_features['topo'][i]
                struct_cache[cof_name] = batch_features['struct'][i]
                targets_cache[cof_name] = batch_features['targets'][i]
            print(f"💾 Saving cache for {len(cof_names)} COFs...")
            print(f"🔍 Debug: topo_cache keys: {list(topo_cache.keys())[:5]}...")
            self._save_cache(topo_cache, struct_cache, targets_cache)
        return batch_features
    def _extract_missing_features(self, missing_cofs: set, cif_dir: str, num_workers: int) -> dict:
        print(f"🔍 Extracting features for {len(missing_cofs)} missing COFs...")
        if len(missing_cofs) < 10:
            return self._extract_missing_sequential(missing_cofs, cif_dir)
        args_list = [(cof_name, cif_dir) for cof_name in missing_cofs]
        missing_features = {}
        failed_cofs = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_cof = {
                executor.submit(self._featurize_single_cof, cof_name, cif_dir): cof_name 
                for cof_name, cif_dir in args_list
            }
            for future in tqdm(as_completed(future_to_cof), total=len(missing_cofs), desc="Processing missing COFs"):
                cof_name = future_to_cof[future]
                try:
                    result = future.result()
                    if result is not None:
                        missing_features[cof_name] = result
                    else:
                        failed_cofs.append(cof_name)
                except Exception as e:
                    print(f"Error processing {cof_name}: {e}")
                    failed_cofs.append(cof_name)
        if failed_cofs:
            print(f"Failed to process {len(failed_cofs)} COFs: {failed_cofs[:5]}{'...' if len(failed_cofs) > 5 else ''}")
        print(f"Successfully processed {len(missing_features)} missing COFs")
        return missing_features
    def _extract_missing_sequential(self, missing_cofs: set, cif_dir: str) -> dict:
        missing_features = {}
        for cof_name in tqdm(missing_cofs, desc="Processing missing COFs (sequential)"):
            try:
                result = self._featurize_single_cof(cof_name, cif_dir)
                if result is not None:
                    missing_features[cof_name] = result
            except Exception as e:
                print(f"Error processing {cof_name}: {e}")
                continue
        return missing_features
    def _extract_all_features(self, cof_names: list, cif_dir: str, num_workers: int) -> dict:
        if len(cof_names) < 10:
            return self._batch_featurize_sequential(cof_names, cif_dir)
        args_list = [(cof_name, cif_dir) for cof_name in cof_names]
        topo_features = []
        struct_features = []
        targets = []
        failed_cofs = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_cof = {
                executor.submit(self._featurize_single_cof, cof_name, cif_dir): cof_name 
                for cof_name, cif_dir in args_list
            }
            for future in tqdm(as_completed(future_to_cof), total=len(cof_names), desc="Processing COFs"):
                cof_name = future_to_cof[future]
                try:
                    result = future.result()
                    if result is not None:
                        topo_features.append(result['topo'])
                        struct_features.append(result['struct'])
                        targets.append(result['target'])
                    else:
                        failed_cofs.append(cof_name)
                except Exception as e:
                    print(f"Error processing {cof_name}: {e}")
                    failed_cofs.append(cof_name)
        if failed_cofs:
            print(f"Failed to process {len(failed_cofs)} COFs: {failed_cofs[:5]}{'...' if len(failed_cofs) > 5 else ''}")
        print(f"Successfully processed {len(topo_features)} COFs")
        return {
            'topo': np.array(topo_features),
            'struct': np.array(struct_features),
            'targets': np.array(targets)
        }
    def _batch_featurize_sequential(self, cof_names: list, cif_dir: str) -> dict:
        topo_features = []
        struct_features = []
        targets = []
        for cof_name in tqdm(cof_names, desc="Processing COFs (sequential)"):
            try:
                result = self._featurize_single_cof(cof_name, cif_dir)
                if result is not None:
                    topo_features.append(result['topo'])
                    struct_features.append(result['struct'])
                    targets.append(result['target'])
            except Exception as e:
                print(f"Error processing {cof_name}: {e}")
                continue
        return {
            'topo': np.array(topo_features),
            'struct': np.array(struct_features),
            'targets': np.array(targets)
        }
    def _featurize_single_cof(self, cof_name: str, cif_dir: str) -> dict:
        try:
            featurizer = COFFeaturizer(
                struct_features_csv=getattr(self, '_struct_csv', None)
            )
            features = featurizer.featurize(cof_name, cif_dir)
            target_row = self.target_df[self.target_df.iloc[:,0] == cof_name]
            if target_row.empty:
                target = 0.0
            else:
                target = target_row.iloc[0, 1]
            return {
                'topo': features['topo'],
                'struct': features['struct'],
                'target': target
            }
        except Exception as e:
            print(f"Error in _featurize_single_cof for {cof_name}: {e}")
            return None
    def _update_cache(self, new_features: dict):
        try:
            if self._cache_loaded['topo']:
                with open(self.topo_cache_file, 'rb') as f:
                    cached_topo = pickle.load(f)
                with open(self.struct_cache_file, 'rb') as f:
                    cached_struct = pickle.load(f)
                with open(self.target_cache_file, 'rb') as f:
                    cached_targets = pickle.load(f)
            else:
                cached_topo = {}
                cached_struct = {}
                cached_targets = {}
            for cof_name, features in new_features.items():
                cached_topo[cof_name] = features['topo']
                cached_struct[cof_name] = features['struct']
                cached_targets[cof_name] = features['target']
            self._save_cache(cached_topo, cached_struct, cached_targets)
            self._cached_topo = cached_topo
            self._cached_struct = cached_struct
            self._cached_targets = cached_targets
            self._cache_loaded = {
                'topo': True,
                'struct': True,
                'targets': True
            }
            print(f"✅ Cache updated with {len(new_features)} new features")
        except Exception as e:
            print(f"⚠️ Error updating cache: {e}")
    def clear_cache(self):
        try:
            if self.topo_cache_file.exists():
                self.topo_cache_file.unlink()
            if self.struct_cache_file.exists():
                self.struct_cache_file.unlink()
            if self.target_cache_file.exists():
                self.target_cache_file.unlink()
            if self.metadata_cache_file.exists():
                self.metadata_cache_file.unlink()
            self._cache_loaded = {
                'topo': False,
                'struct': False,
                'targets': False
            }
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
    def get_cache_info(self) -> dict:
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'cache_loaded': self._cache_loaded.copy(),
            'cache_files': {}
        }
        for cache_file, desc in [
            (self.topo_cache_file, 'topo_features'),
            (self.struct_cache_file, 'struct_features'),
            (self.target_cache_file, 'targets'),
            (self.metadata_cache_file, 'metadata')
        ]:
            if cache_file.exists():
                cache_info['cache_files'][desc] = {
                    'size_bytes': cache_file.stat().st_size,
                    'size_mb': cache_file.stat().st_size / (1024 * 1024),
                    'modified': pd.Timestamp.fromtimestamp(cache_file.stat().st_mtime)
                }
            else:
                cache_info['cache_files'][desc] = None
        return cache_info