#!/usr/bin/env python3
"""
Minimal test script to debug KFold_threshold_estimation parameters.
Run this to find the correct way to call the function.
"""

from driftlens.driftlens import DriftLens
import h5py
import numpy as np
import sys

print("=" * 60)
print(" Testing KFold_threshold_estimation Parameters")
print("=" * 60)

# Load minimal data
print("\n1. Loading data...")
try:
    with h5py.File('baseline.hdf5', 'r') as hf:
        E_base = hf['embeddings'][:500]  # Use only 500 samples for speed
        Y_base = hf['labels'][:500]
    
    with h5py.File('threshold.hdf5', 'r') as hf:
        E_thresh = hf['embeddings'][:500]
        Y_thresh = hf['labels'][:500]
    
    print(f"   Loaded: baseline {E_base.shape}, threshold {E_thresh.shape}")
    
except FileNotFoundError:
    print("   HDF5 files not found. Creating synthetic data...")
    np.random.seed(42)
    E_base = np.random.randn(500, 2048)
    Y_base = np.random.randint(0, 3, 500)
    E_thresh = np.random.randn(500, 2048)
    Y_thresh = np.random.randint(0, 3, 500)

# Initialize and setup baseline
print("\n2. Setting up DriftLens...")
drift_lens = DriftLens()

common_labels = list(set(Y_base) & set(Y_thresh))
print(f"   Common labels: {common_labels}")

drift_lens.estimate_baseline(
    E_base, Y_base, common_labels,
    batch_n_pc=50,
    per_label_n_pc=25
)
print("   ✅ Baseline estimated")

# Test different parameter combinations
print("\n3. Testing KFold_threshold_estimation...")
print("-" * 60)

test_cases = [
    {
        "name": "Test 1: Minimal (just E and Y)",
        "args": (E_thresh, Y_thresh),
        "kwargs": {}
    },
    {
        "name": "Test 2: With window_size as kwarg",
        "args": (E_thresh, Y_thresh),
        "kwargs": {"window_size": 100}
    },
    {
        "name": "Test 3: With window_size and alpha",
        "args": (E_thresh, Y_thresh),
        "kwargs": {"window_size": 100, "alpha": 0.05}
    },
    {
        "name": "Test 4: All as positional",
        "args": (E_thresh, Y_thresh, 100, 0.05),
        "kwargs": {}
    },
    {
        "name": "Test 5: Try with n_folds",
        "args": (E_thresh, Y_thresh),
        "kwargs": {"window_size": 100, "n_folds": 5, "alpha": 0.05}
    },
    {
        "name": "Test 6: Try with k",
        "args": (E_thresh, Y_thresh),
        "kwargs": {"window_size": 100, "k": 5, "alpha": 0.05}
    },
    {
        "name": "Test 7: Window and alpha only",
        "args": (E_thresh, Y_thresh),
        "kwargs": {"window_size": 100, "alpha": 0.05}
    }
]

successful_config = None

for test in test_cases:
    print(f"\n{test['name']}")
    print(f"   Args: {len(test['args'])} positional")
    print(f"   Kwargs: {test['kwargs']}")
    
    try:
        threshold = drift_lens.KFold_threshold_estimation(*test['args'], **test['kwargs'])
        print(f"   ✅ SUCCESS! Threshold = {threshold:.4f}")
        successful_config = test
        break  # Stop on first success
        
    except TypeError as e:
        error_msg = str(e)
        if "unexpected keyword argument" in error_msg:
            bad_param = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            print(f"   ❌ Failed: Parameter '{bad_param}' not recognized")
        elif "missing" in error_msg:
            print(f"   ❌ Failed: Missing required parameters")
        else:
            print(f"   ❌ Failed: {error_msg}")
            
    except Exception as e:
        print(f"   ⚠️  Failed: {type(e).__name__}: {str(e)[:100]}")

# Print summary
print("\n" + "=" * 60)
print(" RESULTS")
print("=" * 60)

if successful_config:
    print("\n✅ WORKING CONFIGURATION FOUND!")
    print(f"\nUse this in your code:")
    print("-" * 40)
    
    if successful_config['kwargs']:
        kwargs_str = ", ".join([f"{k}={v}" for k, v in successful_config['kwargs'].items()])
        if len(successful_config['args']) == 2:
            print(f"threshold = drift_lens.KFold_threshold_estimation(")
            print(f"    E_thresh,")
            print(f"    Y_thresh,")
            for k, v in successful_config['kwargs'].items():
                print(f"    {k}={v},")
            print(f")")
        else:
            args_str = ", ".join(["E_thresh", "Y_thresh"] + [str(a) for a in successful_config['args'][2:]])
            print(f"threshold = drift_lens.KFold_threshold_estimation({args_str})")
    else:
        args_str = ", ".join(["E_thresh", "Y_thresh"] + [str(a) for a in successful_config['args'][2:]])
        print(f"threshold = drift_lens.KFold_threshold_estimation({args_str})")
    
    print("-" * 40)
else:
    print("\n❌ No working configuration found!")
    print("\nTry running this to inspect the function:")
    print("   python -c \"from driftlens.driftlens import DriftLens; help(DriftLens.KFold_threshold_estimation)\"")
    print("\nOr check the source code directly:")
    print("   import driftlens")
    print("   print(driftlens.__file__)")

print("\n✅ Test complete!")
