import h5py
import os
import numpy as np

# --- CONFIGURATION ---
FILE_CONFIG = {
    "baseline.hdf5": 10000,     # Expected number of samples
    "threshold.hdf5": 10000,    # Expected number of samples
    "datastream.hdf5": None     # Stream size depends on the rest of the dataset
}
# --- RESNET50 EMBEDDING DIMENSION ---
# ResNet50 with its classification layer removed outputs a 2048-dimensional vector.
EMBEDDING_DIM = 2048
# -----------------------------------

def verify_hdf5_file(filepath, expected_samples=None):
    """Opens an HDF5 file and verifies its structure and contents."""
    
    print(f"\n--- Checking {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found at '{filepath}'.")
        return False
        
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Check for expected datasets
            if 'embeddings' not in f or 'labels' not in f:
                print("❌ ERROR: Missing required datasets. Expected 'embeddings' and 'labels'.")
                print(f"   Found keys: {list(f.keys())}")
                return False
                
            E = f['embeddings']
            Y = f['labels']
            
            # 2. Check shapes
            E_shape = E.shape
            Y_shape = Y.shape
            
            # Check the number of samples
            if expected_samples is not None and E_shape[0] != expected_samples:
                print(f"⚠️ WARNING: Expected {expected_samples} samples, but found {E_shape[0]} samples.")
            
            # Check the feature dimension
            if len(E_shape) != 2 or E_shape[1] != EMBEDDING_DIM:
                print(f"❌ ERROR: 'embeddings' shape is incorrect. Expected (*, {EMBEDDING_DIM}), got {E_shape}.")
                return False
                
            # Check labels shape (should be a single dimension matching the embeddings samples)
            if len(Y_shape) != 1 or Y_shape[0] != E_shape[0]:
                print(f"❌ ERROR: 'labels' shape is incorrect. Expected ({E_shape[0]},), got {Y_shape}.")
                return False
            
            # 3. Check data types
            if E.dtype not in [np.float32, np.float64]:
                print(f"⚠️ WARNING: 'embeddings' data type is unusual. Expected float, got {E.dtype}.")

            if Y.dtype not in [np.int32, np.int64]:
                print(f"⚠️ WARNING: 'labels' data type is unusual. Expected int, got {Y.dtype}.")

            print("✅ SUCCESS: File structure is correct.")
            print(f"   Embeddings Shape: {E_shape}")
            print(f"   Labels Shape: {Y_shape}")
            return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not read file due to an exception: {e}")
        return False

# --- Run Verification ---
all_ok = True
for filename, expected_samples in FILE_CONFIG.items():
    if not verify_hdf5_file(filename, expected_samples):
        all_ok = False

if all_ok:
    print("\n\nAll HDF5 files passed the basic verification checks! 🎉")
else:
    print("\n\nOne or more HDF5 files failed the verification checks. 🛑")
