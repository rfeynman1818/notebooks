#!/usr/bin/env python3
"""
Analyze label distributions and recommend appropriate PCA component settings.
This helps avoid the "n_components must be between 0 and min(n_samples)" error.
"""

import h5py
import numpy as np
from collections import Counter
import sys

print("=" * 70)
print(" Label Distribution Analysis for PCA Component Selection")
print("=" * 70)

def analyze_dataset(filepath, name):
    """Analyze a dataset and return label statistics."""
    print(f"\n📊 Analyzing {name}...")
    
    try:
        with h5py.File(filepath, 'r') as hf:
            E = hf['embeddings'][:]
            Y = hf['labels'][:]
        
        n_samples, n_features = E.shape
        print(f"   Total samples: {n_samples}")
        print(f"   Feature dimensions: {n_features}")
        
        # Count samples per label
        label_counts = Counter(Y)
        
        print(f"\n   Samples per label:")
        min_samples = float('inf')
        label_stats = {}
        
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = (count / n_samples) * 100
            print(f"     Label {label}: {count:5d} samples ({percentage:5.1f}%)")
            label_stats[label] = count
            min_samples = min(min_samples, count)
        
        # For each label, calculate max possible PCA components
        print(f"\n   Maximum PCA components per label:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            # PCA components must be less than min(n_samples, n_features)
            max_components = min(count - 1, n_features)  # -1 for safety
            print(f"     Label {label}: max {max_components} components (has {count} samples)")
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'label_counts': dict(label_counts),
            'min_samples_per_label': min_samples,
            'labels': sorted(label_counts.keys())
        }
        
    except FileNotFoundError:
        print(f"   ❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# Analyze all three datasets
datasets = {
    'baseline': analyze_dataset('baseline.hdf5', 'Baseline'),
    'threshold': analyze_dataset('threshold.hdf5', 'Threshold'),
    'stream': analyze_dataset('datastream.hdf5', 'Stream')
}

# Find common labels and minimum samples
print("\n" + "=" * 70)
print(" CROSS-DATASET ANALYSIS")
print("=" * 70)

if datasets['baseline'] and datasets['threshold']:
    baseline_labels = set(datasets['baseline']['labels'])
    threshold_labels = set(datasets['threshold']['labels'])
    common_labels = baseline_labels & threshold_labels
    
    print(f"\nCommon labels: {sorted(common_labels)}")
    
    # Find minimum samples across common labels
    min_samples_overall = float('inf')
    critical_label = None
    
    for label in common_labels:
        baseline_count = datasets['baseline']['label_counts'].get(label, 0)
        threshold_count = datasets['threshold']['label_counts'].get(label, 0)
        min_count = min(baseline_count, threshold_count)
        
        if min_count < min_samples_overall:
            min_samples_overall = min_count
            critical_label = label
    
    print(f"\nCritical constraint:")
    print(f"  Label {critical_label} has the fewest samples: {min_samples_overall}")
    print(f"  This limits your PCA components!")

# Calculate recommended settings
print("\n" + "=" * 70)
print(" RECOMMENDED SETTINGS")
print("=" * 70)

if datasets['baseline'] and datasets['threshold']:
    # Conservative recommendations
    min_samples = min_samples_overall
    
    # For batch PCA (uses all samples)
    total_baseline = datasets['baseline']['n_samples']
    total_threshold = datasets['threshold']['n_samples']
    min_total = min(total_baseline, total_threshold)
    
    # Batch PCA can use more components (based on total samples)
    max_batch_pc = min(min_total - 1, datasets['baseline']['n_features'])
    recommended_batch_pc = min(150, max_batch_pc)  # Cap at 150 or max possible
    
    # Per-label PCA must respect the smallest label group
    max_per_label_pc = min_samples - 1  # Must be less than smallest label group
    recommended_per_label_pc = min(75, max_per_label_pc)  # Cap at 75 or max possible
    
    print(f"\n✅ SAFE CONFIGURATION:")
    print(f"   batch_n_pc = {recommended_batch_pc}")
    print(f"   per_label_n_pc = {recommended_per_label_pc}")
    
    print(f"\n📝 Add this to your config:")
    print(f"""
config = {{
    "window_size": 500,
    "batch_n_pc": {recommended_batch_pc},      # Safe for batch PCA
    "per_label_n_pc": {recommended_per_label_pc},    # Safe for per-label PCA  
    "alpha": 0.05
}}
""")
    
    # Warnings
    if recommended_per_label_pc < 25:
        print(f"\n⚠️  WARNING:")
        print(f"   Per-label PCA is limited to {recommended_per_label_pc} components")
        print(f"   because label {critical_label} only has {min_samples} samples.")
        print(f"   This might reduce drift detection sensitivity.")
        print(f"\n   Consider:")
        print(f"   1. Using more data for better label representation")
        print(f"   2. Removing rare labels from analysis")
        print(f"   3. Using data augmentation for minority classes")
    
    # Alternative configurations
    print(f"\n🔧 ALTERNATIVE CONFIGURATIONS:")
    
    print(f"\n1. Minimal (guaranteed to work):")
    minimal_per_label = min(10, max_per_label_pc)
    print(f"   batch_n_pc = 50")
    print(f"   per_label_n_pc = {minimal_per_label}")
    
    print(f"\n2. Balanced (good performance/stability):")
    balanced_per_label = min(30, max_per_label_pc)
    balanced_batch = min(100, max_batch_pc)
    print(f"   batch_n_pc = {balanced_batch}")
    print(f"   per_label_n_pc = {balanced_per_label}")
    
    print(f"\n3. Maximum possible:")
    print(f"   batch_n_pc = {max_batch_pc}")
    print(f"   per_label_n_pc = {max_per_label_pc}")
    
else:
    print("\n❌ Could not analyze datasets. Please ensure HDF5 files exist.")

# Create a config file
print("\n" + "=" * 70)
print(" GENERATING CONFIG FILE")
print("=" * 70)

if datasets['baseline'] and datasets['threshold']:
    config_content = f"""# Auto-generated DriftLens configuration
# Based on your data analysis

# Safe configuration that will work with your data
DRIFTLENS_CONFIG = {{
    "window_size": 500,
    "batch_n_pc": {recommended_batch_pc},
    "per_label_n_pc": {recommended_per_label_pc},
    "alpha": 0.05
}}

# Data statistics for reference
DATA_STATS = {{
    "min_samples_per_label": {min_samples_overall},
    "critical_label": {critical_label},
    "common_labels": {sorted(common_labels)},
    "baseline_samples": {total_baseline},
    "threshold_samples": {total_threshold}
}}

# Alternative configurations
MINIMAL_CONFIG = {{
    "window_size": 500,
    "batch_n_pc": 50,
    "per_label_n_pc": {min(10, max_per_label_pc)},
    "alpha": 0.05
}}

BALANCED_CONFIG = {{
    "window_size": 500,
    "batch_n_pc": {min(100, max_batch_pc)},
    "per_label_n_pc": {min(30, max_per_label_pc)},
    "alpha": 0.05
}}
"""
    
    with open("driftlens_config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration saved to: driftlens_config.py")
    print("   Import it in your script:")
    print("   from driftlens_config import DRIFTLENS_CONFIG")

print("\n✅ Analysis complete!")
