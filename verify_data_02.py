import h5py
import os
import numpy as np
import sys

# ... (FILE_CONFIG and EMBEDDING_DIM remain the same) ...

def verify_hdf5_file_with_found_keys(filepath, expected_samples=None):
    """Opens an HDF5 file and verifies its structure using the keys found in the error log."""
    
    print(f"\n--- Checking {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found at '{filepath}'.")
        return False

    # Use the keys found in the error traceback (182.png)
    EMBEDDING_KEY = 'E'
    LABEL_KEY = 'Y_predicted' if 'datastream' in filepath or 'threshold' in filepath else 'Y'

    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Check for expected datasets
            if EMBEDDING_KEY not in f or LABEL_KEY not in f:
                print(f"❌ ERROR: Missing required datasets. Expected '{EMBEDDING_KEY}' and '{LABEL_KEY}'.")
                print(f"   Found keys: {list(f.keys())}")
                return False
                
            E = f[EMBEDDING_KEY]
            Y = f[LABEL_KEY]
            
            # ... (Rest of the shape and type checks are the same) ...
            E_shape = E.shape
            Y_shape = Y.shape
            
            # Check the number of samples
            if expected_samples is not None and E_shape[0] != expected_samples and 'datastream' not in filepath:
                print(f"⚠️ WARNING: Expected {expected_samples} samples, but found {E_shape[0]} samples.")
            
            # Check the feature dimension
            if len(E_shape) != 2 or E_shape[1] != EMBEDDING_DIM:
                print(f"❌ ERROR: '{EMBEDDING_KEY}' shape is incorrect. Expected (*, {EMBEDDING_DIM}), got {E_shape}.")
                return False
                
            # Check labels shape
            if len(Y_shape) != 1 or Y_shape[0] != E_shape[0]:
                print(f"❌ ERROR: '{LABEL_KEY}' shape is incorrect. Expected ({E_shape[0]},), got {Y_shape}.")
                return False
            
            # Check data types
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
        
# ... (Call the verification function with the appropriate name) ...
