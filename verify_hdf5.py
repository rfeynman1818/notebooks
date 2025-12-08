#!/usr/bin/env python3
"""
Verify HDF5 files are properly formatted for DriftLens
Run this before running the experiment to catch issues early.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def verify_hdf5_file(filepath):
    """Verify an HDF5 file has the correct structure for DriftLens."""
    
    if not Path(filepath).exists():
        return False, f"File not found: {filepath}"
    
    try:
        with h5py.File(filepath, 'r') as hf:
            # Check required keys
            if 'embeddings' not in hf:
                return False, f"Missing 'embeddings' dataset in {filepath}"
            if 'labels' not in hf:
                return False, f"Missing 'labels' dataset in {filepath}"
            
            # Get data
            embeddings = hf['embeddings']
            labels = hf['labels']
            
            # Check shapes
            if len(embeddings.shape) != 2:
                return False, f"Embeddings should be 2D, got shape {embeddings.shape}"
            
            if len(labels.shape) != 1:
                return False, f"Labels should be 1D, got shape {labels.shape}"
            
            # Check lengths match
            if embeddings.shape[0] != labels.shape[0]:
                return False, f"Mismatch: {embeddings.shape[0]} embeddings vs {labels.shape[0]} labels"
            
            # Load data to check types and values
            E = embeddings[:]
            Y = labels[:]
            
            # Check data types
            if not np.issubdtype(E.dtype, np.floating):
                return False, f"Embeddings should be float type, got {E.dtype}"
            
            if not np.issubdtype(Y.dtype, np.integer):
                return False, f"Labels should be integer type, got {Y.dtype}"
            
            # Check for NaN or Inf in embeddings
            if np.any(np.isnan(E)):
                return False, "Embeddings contain NaN values"
            if np.any(np.isinf(E)):
                return False, "Embeddings contain Inf values"
            
            # Get label statistics
            unique_labels = np.unique(Y)
            n_samples = len(Y)
            embedding_dim = E.shape[1]
            
            info = {
                'n_samples': n_samples,
                'embedding_dim': embedding_dim,
                'unique_labels': unique_labels.tolist(),
                'n_unique_labels': len(unique_labels),
                'label_range': (int(Y.min()), int(Y.max()))
            }
            
            return True, info
            
    except Exception as e:
        return False, f"Error reading {filepath}: {str(e)}"

def main():
    print("=" * 60)
    print("DriftLens HDF5 File Verification")
    print("=" * 60)
    
    files_to_check = [
        "baseline.hdf5",
        "threshold.hdf5", 
        "datastream.hdf5"
    ]
    
    all_valid = True
    file_info = {}
    
    for filepath in files_to_check:
        print(f"\nChecking {filepath}...")
        valid, result = verify_hdf5_file(filepath)
        
        if valid:
            print(f"✅ {filepath} is valid!")
            print(f"   Samples: {result['n_samples']}")
            print(f"   Embedding dimension: {result['embedding_dim']}")
            print(f"   Unique labels: {result['unique_labels']}")
            file_info[filepath] = result
        else:
            print(f"❌ {filepath} has issues: {result}")
            all_valid = False
    
    if not all_valid:
        print("\n" + "=" * 60)
        print("❌ Some files have issues. Please fix them before running DriftLens.")
        sys.exit(1)
    
    # Cross-file validation
    print("\n" + "=" * 60)
    print("Cross-file Validation")
    print("=" * 60)
    
    # Check embedding dimensions match
    embedding_dims = [info['embedding_dim'] for info in file_info.values()]
    if len(set(embedding_dims)) > 1:
        print(f"⚠️  WARNING: Embedding dimensions don't match across files: {embedding_dims}")
        all_valid = False
    else:
        print(f"✅ All files have consistent embedding dimension: {embedding_dims[0]}")
    
    # Check label overlap
    baseline_labels = set(file_info['baseline.hdf5']['unique_labels'])
    threshold_labels = set(file_info['threshold.hdf5']['unique_labels'])
    stream_labels = set(file_info['datastream.hdf5']['unique_labels'])
    
    common_labels = baseline_labels & threshold_labels
    
    print(f"\nLabel Analysis:")
    print(f"  Baseline labels: {sorted(baseline_labels)}")
    print(f"  Threshold labels: {sorted(threshold_labels)}")
    print(f"  Stream labels: {sorted(stream_labels)}")
    print(f"  Common (baseline ∩ threshold): {sorted(common_labels)}")
    
    if len(common_labels) == 0:
        print("❌ ERROR: No common labels between baseline and threshold!")
        print("   DriftLens requires at least some labels to be present in both.")
        all_valid = False
    elif len(common_labels) < len(baseline_labels):
        print(f"⚠️  WARNING: Only {len(common_labels)} labels are common.")
        print(f"   DriftLens will only use these labels: {sorted(common_labels)}")
    else:
        print(f"✅ All {len(common_labels)} baseline labels present in threshold.")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All files are valid and compatible with DriftLens!")
        print("\nYou can now run: python run_experiment_fixed.py")
        
        # Create a config file with detected parameters
        with open("detected_config.txt", "w") as f:
            f.write("# Auto-detected configuration\n")
            f.write(f"EMBEDDING_DIM = {embedding_dims[0]}\n")
            f.write(f"COMMON_LABELS = {sorted(list(common_labels))}\n")
            f.write(f"N_BASELINE = {file_info['baseline.hdf5']['n_samples']}\n")
            f.write(f"N_THRESHOLD = {file_info['threshold.hdf5']['n_samples']}\n")
            f.write(f"N_STREAM = {file_info['datastream.hdf5']['n_samples']}\n")
        print("\nConfiguration saved to: detected_config.txt")
    else:
        print("❌ Issues detected. Please fix them before running DriftLens.")
        sys.exit(1)

if __name__ == "__main__":
    main()
