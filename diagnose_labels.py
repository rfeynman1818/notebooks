import h5py
import numpy as np
from collections import Counter

def analyze_labels(filepath, dataset_name):
    """Analyze label distribution in an HDF5 file."""
    with h5py.File(filepath, 'r') as hf:
        labels = hf['labels'][:]
        
    label_counts = Counter(labels)
    unique_labels = sorted(label_counts.keys())
    
    print(f"\n=== {dataset_name} ===")
    print(f"Total samples: {len(labels)}")
    print(f"Unique labels present: {unique_labels}")
    print(f"Label distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(labels)) * 100
        print(f"  Label {label}: {count:5d} samples ({percentage:5.2f}%)")
    
    return unique_labels

# Analyze each dataset
print("FireRisk Dataset Label Analysis")
print("=" * 50)

baseline_labels = analyze_labels("baseline.hdf5", "BASELINE")
threshold_labels = analyze_labels("threshold.hdf5", "THRESHOLD")
stream_labels = analyze_labels("datastream.hdf5", "DATASTREAM")

# Find common labels across all datasets
common_labels = list(set(baseline_labels) & set(threshold_labels))
print(f"\n=== COMMON LABELS ===")
print(f"Labels present in both baseline and threshold: {sorted(common_labels)}")

# Identify missing labels
all_expected_labels = list(range(7))
missing_in_baseline = [l for l in all_expected_labels if l not in baseline_labels]
missing_in_threshold = [l for l in all_expected_labels if l not in threshold_labels]

if missing_in_baseline:
    print(f"\n⚠️  WARNING: Labels {missing_in_baseline} are MISSING from baseline!")
    print("   This is causing the DriftLens error.")
    
if missing_in_threshold:
    print(f"⚠️  WARNING: Labels {missing_in_threshold} are MISSING from threshold!")

print(f"\n=== SOLUTION ===")
print(f"Use only these labels in DriftLens: {sorted(common_labels)}")
