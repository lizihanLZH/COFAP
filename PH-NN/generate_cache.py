#!/usr/bin/env python3
"""
LZHNN Feature Cache Pre-generation Script

This script is designed to generate feature cache before the first training run.
It extracts all features and saves them to cache, so subsequent training runs
can use the cache directly without feature extraction.
"""

import os
import sys
import argparse
import time
import pandas as pd
from pathlib import Path

# Import our custom modules
from ElseFeaturizer import CombinedCOFFeaturizer

def parse_args():
    parser = argparse.ArgumentParser(description='Generate LZHNN feature cache')
    parser.add_argument('--struct_features_csv', type=str, required=True,
                       help='Path to structural features CSV file')
    parser.add_argument('--target_csv', type=str, required=True,
                       help='Path to target CSV file')
    parser.add_argument('--cif_dir', type=str, required=True,
                       help='Directory containing CIF files')
    parser.add_argument('--cache_dir', type=str, default='feature_cache',
                       help='Directory for feature cache (default: feature_cache)')
    parser.add_argument('--num_workers', type=int, default=12,
                       help='Number of workers for parallel processing (default: 12)')
    parser.add_argument('--clear_existing', action='store_true', default=False,
                       help='Clear existing cache before generation (default: False)')
    parser.add_argument('--validate_cache', action='store_true', default=True,
                       help='Validate generated cache (default: True)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 LZHNN Feature Cache Generator")
    print("="*50)
    print(f"Structural features: {args.struct_features_csv}")
    print(f"Target values: {args.target_csv}")
    print(f"CIF directory: {args.cif_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Clear existing: {args.clear_existing}")
    print(f"Validate cache: {args.validate_cache}")
    print("="*50)
    
    # Check if input files exist
    if not os.path.exists(args.struct_features_csv):
        print(f"❌ Error: Structural features file not found: {args.struct_features_csv}")
        return
    
    if not os.path.exists(args.target_csv):
        print(f"❌ Error: Target file not found: {args.target_csv}")
        return
    
    if not os.path.exists(args.cif_dir):
        print(f"❌ Error: CIF directory not found: {args.cif_dir}")
        return
    
    # Initialize featurizer
    print("\n🔧 Initializing featurizer...")
    featurizer = CombinedCOFFeaturizer(
        struct_features_csv=args.struct_features_csv,
        target_csv=args.target_csv,
        cache_dir=args.cache_dir
    )
    
    # Clear existing cache if requested
    if args.clear_existing:
        print("🗑️ Clearing existing cache...")
        featurizer.clear_cache()
    
    # Check current cache status
    cache_info = featurizer.get_cache_info()
    print(f"\n📊 Current cache status:")
    print(f"   Cache directory: {cache_info['cache_dir']}")
    print(f"   Cache loaded: {cache_info['cache_loaded']}")
    
    # Read target CSV to get COF names
    print(f"\n📖 Reading target file...")
    target_df = pd.read_csv(args.target_csv)
    cof_names = target_df.iloc[:, 0].tolist()
    print(f"   Found {len(cof_names)} COFs in target file")
    
    # Check if cache is already complete
    if (cache_info['cache_loaded']['topo'] and 
        cache_info['cache_loaded']['struct'] and 
        cache_info['cache_loaded']['targets']):
        
        # Check if all requested COFs are in cache
        if hasattr(featurizer, '_cached_topo') and hasattr(featurizer, '_cached_struct'):
            cached_cofs = set(featurizer._cached_topo.keys())
            requested_cofs = set(cof_names)
            
            if requested_cofs.issubset(cached_cofs):
                print(f"\n✅ Cache is already complete and contains all {len(cof_names)} COFs!")
                print("   No need to regenerate cache.")
                
                if args.validate_cache:
                    print("\n🔍 Validating existing cache...")
                    if validate_cache(featurizer, cof_names):
                        print("✅ Cache validation passed!")
                    else:
                        print("❌ Cache validation failed!")
                
                return
            else:
                missing_count = len(requested_cofs - cached_cofs)
                missing_cofs = list(requested_cofs - cached_cofs)[:5]  # Show first 5 missing
                print(f"\n⚠️ Cache exists but missing {missing_count} COFs")
                print(f"   Missing COFs: {missing_cofs}{'...' if missing_count > 5 else ''}")
                print("   Will extract features for missing samples...")
        else:
            print(f"\n⚠️ Cache files exist but not properly loaded")
            print("   Will regenerate complete cache...")
    
    # Generate cache
    print(f"\n🔄 Starting feature extraction and cache generation...")
    print(f"   This may take several minutes depending on dataset size...")
    
    start_time = time.time()
    
    try:
        # Extract features with caching enabled
        batch_features = featurizer.batch_featurize(
            cof_names, 
            args.cif_dir, 
            args.num_workers,
            use_cache=True
        )
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        print(f"\n✅ Feature extraction completed successfully!")
        print(f"   Time taken: {extraction_time:.2f} seconds")
        print(f"   Features extracted: {len(batch_features['topo'])} COFs")
        
        # Force reload cache after feature extraction
        print(f"\n🔄 Reloading cache after feature extraction...")
        featurizer._load_cache()
        
        # Display updated cache information
        updated_cache_info = featurizer.get_cache_info()
        print(f"\n📊 Updated Cache Information:")
        print(f"   Cache loaded: {updated_cache_info['cache_loaded']}")
        
        if updated_cache_info['cache_files']['topo_features']:
            print(f"   Topo features cache: {updated_cache_info['cache_files']['topo_features']['size_mb']:.2f} MB")
        if updated_cache_info['cache_files']['struct_features']:
            print(f"   Struct features cache: {updated_cache_info['cache_files']['struct_features']['size_mb']:.2f} MB")
        if updated_cache_info['cache_files']['targets']:
            print(f"   Targets cache: {updated_cache_info['cache_files']['targets']['size_mb']:.2f} MB")
        
        # Validate generated cache
        if args.validate_cache:
            print(f"\n🔍 Validating generated cache...")
            if validate_cache(featurizer, cof_names):
                print("✅ Cache validation passed!")
            else:
                print("❌ Cache validation failed!")
        
        print(f"\n🎉 Cache generation completed successfully!")
        print(f"   Cache location: {args.cache_dir}")
        print(f"   You can now run training with --use_cache flag")
        
    except Exception as e:
        print(f"\n❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return

def validate_cache(featurizer, cof_names):
    """Validate the generated cache"""
    try:
        # Check if cache is loaded
        if not (featurizer._cache_loaded['topo'] and 
                featurizer._cache_loaded['struct'] and 
                featurizer._cache_loaded['targets']):
            print("   ❌ Cache not properly loaded")
            return False
        
        # Check if all COFs are in cache
        cached_cofs = set(featurizer._cached_topo.keys())
        requested_cofs = set(cof_names)
        
        if not requested_cofs.issubset(cached_cofs):
            missing = requested_cofs - cached_cofs
            print(f"   ❌ Missing COFs in cache: {len(missing)}")
            return False
        
        # Check feature dimensions
        expected_topo_dim = 18
        expected_struct_dim = 5
        
        for cof_name in list(requested_cofs)[:5]:  # Check first 5 samples
            topo_feat = featurizer._cached_topo[cof_name]
            struct_feat = featurizer._cached_struct[cof_name]
            
            if topo_feat.shape[0] != expected_topo_dim:
                print(f"   ❌ Topo feature dimension mismatch for {cof_name}: {topo_feat.shape[0]} != {expected_topo_dim}")
                return False
            
            if struct_feat.shape[0] != expected_struct_dim:
                print(f"   ❌ Struct feature dimension mismatch for {cof_name}: {struct_feat.shape[0]} != {expected_struct_dim}")
                return False
        
        print(f"   ✅ All {len(requested_cofs)} COFs found in cache")
        print(f"   ✅ Feature dimensions correct: {expected_topo_dim} (topo), {expected_struct_dim} (struct)")
        return True
        
    except Exception as e:
        print(f"   ❌ Cache validation error: {e}")
        return False

if __name__ == "__main__":
    main() 
