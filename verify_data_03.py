import h5py
import os
import numpy as np
import sys

# --- CONFIGURATION ---
# Expected number of samples (based on your intended 10000/10000 split)
FILE_CONFIG = {
    "baseline.hdf5": 10000,     
    "threshold.hdf5": 10000,    
    "datastream.hdf5": None     # Check will only verify columns
}
# ResNet50 with classification layer removed outputs a 2048-dimensional vector.
EMBEDDING_DIM = 2048
# -----------------------------------

def verify_hdf5_file(filepath, expected_samples=None):
    """Opens an HDF5 file and verifies its structure using the actual keys found in your files (E and Y_predicted)."""
    
    print(f"\n--- Checking {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found at '{filepath}'.")
        return False

    # --- DYNAMIC KEY SELECTION (FIXED) ---
    # Based on the error log (183.png), all files contain 'E' and 'Y_predicted'.
    EMBEDDING_KEY = 'E'
    LABEL_KEY = 'Y_predicted'
    # -------------------------------------

    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Check for expected datasets
            if EMBEDDING_KEY not in f or LABEL_KEY not in f:
                print(f"❌ ERROR: Missing required datasets. Expected '{EMBEDDING_KEY}' and '{LABEL_KEY}'.")
                print(f"   Found keys: {list(f.keys())}")
                return False
                
            E = f[EMBEDDING_KEY]
            Y = f[LABEL_KEY]
            
            # 2. Check shapes
            E_shape = E.shape
            Y_shape = Y.shape
            
            # Check the number of samples (Warning is shown for baseline/threshold)
            if expected_samples is not None and E_shape[0] != expected_samples and 'datastream' not in filepath:
                print(f"⚠️ WARNING: Expected {expected_samples} samples, but found {E_shape[0]} samples in {filepath}.")
            
            # Check the feature dimension
            if len(E_shape) != 2 or E_shape[1] != EMBEDDING_DIM:
                print(f"❌ ERROR: '{EMBEDDING_KEY}' shape is incorrect. Expected (*, {EMBEDDING_DIM}), got {E_shape}.")
                return False
                
            # Check labels shape
            if len(Y_shape) != 1 or Y_shape[0] != E_shape[0]:
                print(f"❌ ERROR: '{LABEL_KEY}' shape is incorrect. Expected ({E_shape[0]},), got {Y_shape}.")
                return False
            
            # 3. Check data types
            if E.dtype not in [np.float32, np.float64]:
                print(f"⚠️ WARNING: '{EMBEDDING_KEY}' data type is unusual. Expected float, got {E.dtype}.")

            if Y.dtype not in [np.int32, np.int64]:
                print(f"⚠️ WARNING: '{LABEL_KEY}' data type is unusual. Expected int, got {Y.dtype}.")

            print(f"✅ SUCCESS: File structure is correct with keys '{EMBEDDING_KEY}' and '{LABEL_KEY}'.")
            print(f"   Embeddings Shape: {E_shape}")
            print(f"   Labels Shape: {Y_shape}")
            return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not read file due to an exception: {e}")
        return False

# --- Run Verification ---
if __name__ == "__main__":
    all_ok = True
    for filename, expected_samples in FILE_CONFIG.items():
        if not verify_hdf5_file(filename, expected_samples):
            all_ok = False

    if all_ok:
        print("\n\nAll HDF5 files passed the basic verification checks! 🎉")
    else:
        print("\n\nOne or more HDF5 files failed the verification checks. 🛑")
