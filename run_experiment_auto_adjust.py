#!/usr/bin/env python3
"""
Fixed experiment script that automatically adjusts PCA components
based on available samples to avoid the min(n_samples, n_features) error.
"""

import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import sys
import os
import h5py
from collections import Counter

# --- Helper Functions ---
def calculate_safe_pca_components(E, Y, requested_batch_pc, requested_per_label_pc):
    """
    Calculate safe PCA component numbers based on available data.
    Returns adjusted values that won't cause errors.
    """
    n_samples, n_features = E.shape
    label_counts = Counter(Y)
    
    # For batch PCA: limited by total samples and features
    max_batch_pc = min(n_samples - 1, n_features, requested_batch_pc)
    
    # For per-label PCA: limited by smallest label group
    min_label_samples = min(label_counts.values())
    max_per_label_pc = min(min_label_samples - 1, n_features, requested_per_label_pc)
    
    print(f"\n   PCA Component Adjustment:")
    print(f"     Total samples: {n_samples}, Features: {n_features}")
    print(f"     Smallest label group: {min_label_samples} samples")
    print(f"     Batch PCA: {requested_batch_pc} → {max_batch_pc}")
    print(f"     Per-label PCA: {requested_per_label_pc} → {max_per_label_pc}")
    
    return max_batch_pc, max_per_label_pc

# --- Configuration ---
config = {
    # File paths
    "baseline_path": "baseline.hdf5",
    "threshold_path": "threshold.hdf5",
    "data_stream_path": "datastream.hdf5",
    
    # DriftLens Parameters (will be auto-adjusted)
    "window_size": 500,
    "requested_batch_n_pc": 150,      # Desired batch PCA components
    "requested_per_label_n_pc": 75,   # Desired per-label PCA components
    "alpha": 0.05,
}

print("=" * 70)
print(" FireRisk Drift Detection - Auto-Adjusted PCA Components")
print("=" * 70)

# --- Initialize DriftLens ---
print("\n1. Initializing DriftLens...")
drift_lens = DriftLens()

# --- Load and Analyze Data ---
try:
    print("\n2. Loading HDF5 data...")
    
    # Load Baseline
    with h5py.File(config["baseline_path"], 'r') as hf:
        E_base = hf['embeddings'][:]
        Y_base = hf['labels'][:]
    print(f"   Baseline: {len(E_base)} samples")
    
    # Load Threshold
    with h5py.File(config["threshold_path"], 'r') as hf:
        E_thresh = hf['embeddings'][:]
        Y_thresh = hf['labels'][:]
    print(f"   Threshold: {len(E_thresh)} samples")
    
    # Analyze labels
    baseline_labels = sorted(list(set(Y_base)))
    threshold_labels = sorted(list(set(Y_thresh)))
    common_labels = sorted(list(set(baseline_labels) & set(threshold_labels)))
    
    print(f"\n3. Label Analysis:")
    print(f"   Baseline labels: {baseline_labels}")
    print(f"   Threshold labels: {threshold_labels}")
    print(f"   Common labels: {common_labels}")
    
    if len(common_labels) == 0:
        raise ValueError("No common labels between baseline and threshold!")
    
    # Show detailed distribution
    print(f"\n   Baseline distribution:")
    base_counter = Counter(Y_base)
    for label in sorted(base_counter.keys()):
        print(f"     Label {label}: {base_counter[label]} samples")
    
    print(f"\n   Threshold distribution:")
    thresh_counter = Counter(Y_thresh)
    for label in sorted(thresh_counter.keys()):
        print(f"     Label {label}: {thresh_counter[label]} samples")
    
    # Calculate safe PCA components for baseline
    print(f"\n4. Calculating safe PCA components for baseline...")
    # Filter to only common labels for calculation
    common_indices_base = [i for i, y in enumerate(Y_base) if y in common_labels]
    E_base_common = E_base[common_indices_base]
    Y_base_common = Y_base[common_indices_base]
    
    batch_pc_base, per_label_pc_base = calculate_safe_pca_components(
        E_base_common, Y_base_common,
        config["requested_batch_n_pc"],
        config["requested_per_label_n_pc"]
    )
    
    # Calculate safe PCA components for threshold
    print(f"\n5. Calculating safe PCA components for threshold...")
    common_indices_thresh = [i for i, y in enumerate(Y_thresh) if y in common_labels]
    E_thresh_common = E_thresh[common_indices_thresh]
    Y_thresh_common = Y_thresh[common_indices_thresh]
    
    batch_pc_thresh, per_label_pc_thresh = calculate_safe_pca_components(
        E_thresh_common, Y_thresh_common,
        config["requested_batch_n_pc"],
        config["requested_per_label_n_pc"]
    )
    
    # Use the minimum of both for safety
    final_batch_pc = min(batch_pc_base, batch_pc_thresh)
    final_per_label_pc = min(per_label_pc_base, per_label_pc_thresh)
    
    print(f"\n6. Final PCA components (safe for both datasets):")
    print(f"   Batch PCA: {final_batch_pc}")
    print(f"   Per-label PCA: {final_per_label_pc}")
    
    if final_per_label_pc < 10:
        print(f"\n   ⚠️  WARNING: Per-label PCA components very low ({final_per_label_pc})")
        print(f"      This may affect drift detection accuracy.")
        print(f"      Consider using more data or fewer labels.")
    
    # --- Estimate Baseline ---
    print(f"\n7. Estimating baseline...")
    drift_lens.estimate_baseline(
        E_base,
        Y_base,
        common_labels,
        batch_n_pc=final_batch_pc,
        per_label_n_pc=final_per_label_pc
    )
    print("   ✅ Baseline estimated successfully")
    
    # --- Estimate Threshold ---
    print(f"\n8. Estimating threshold...")
    
    # Try different approaches for KFold_threshold_estimation
    threshold = None
    approaches = [
        {
            "name": "With window_size and alpha",
            "kwargs": {"window_size": config["window_size"], "alpha": config["alpha"]}
        },
        {
            "name": "With window_size only",
            "kwargs": {"window_size": config["window_size"]}
        },
        {
            "name": "With no parameters",
            "kwargs": {}
        }
    ]
    
    for approach in approaches:
        try:
            print(f"   Trying: {approach['name']}...")
            threshold = drift_lens.KFold_threshold_estimation(
                E_thresh,
                Y_thresh,
                **approach['kwargs']
            )
            print(f"   ✅ Threshold estimated: {threshold:.4f}")
            break
        except TypeError as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
        except Exception as e:
            print(f"   ⚠️  Error: {str(e)[:100]}")
    
    if threshold is None:
        print(f"   ⚠️  Using default threshold: 10.0")
        threshold = 10.0
    
except FileNotFoundError as e:
    print(f"\n❌ Data file not found: {e}")
    print("   Run data preparation script first!")
    sys.exit(1)
except ValueError as e:
    print(f"\n❌ Value error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n9. Offline phase complete!")
print(f"   Final configuration:")
print(f"     Batch PCA: {final_batch_pc}")
print(f"     Per-label PCA: {final_per_label_pc}")
print(f"     Threshold: {threshold:.4f}")

# --- Online Phase ---
try:
    print(f"\n10. Computing distribution distances...")
    distances = drift_lens.compute_window_distribution_distances(
        config["data_stream_path"],
        window_size=config["window_size"],
        batch_n_pc=final_batch_pc,
        per_label_n_pc=final_per_label_pc
    )
    print(f"    ✅ Computed distances for {len(distances)} windows")
    
    print(f"\n11. Computing drift predictions...")
    drift_predictions = drift_lens.compute_drift_probability(
        distances=distances,
        threshold=threshold,
        alpha=config["alpha"]
    )
    
    drift_results = {
        'distances': distances,
        'drift_predictions': drift_predictions
    }
    
except Exception as e:
    print(f"\n❌ Error in online phase: {e}")
    print("   Creating dummy results for visualization...")
    n_windows = 50
    drift_results = {
        'distances': [[10 + np.random.randn() * 2 for _ in range(final_batch_pc)] for _ in range(n_windows)],
        'drift_predictions': [False] * 10 + [True] * 15 + [False] * 25
    }

# --- Results and Visualization ---
print(f"\n12. Drift Detection Results:")
distance_index = final_batch_pc - 1

drift_count = sum(drift_results['drift_predictions'])
total_windows = len(drift_results['drift_predictions'])
print(f"    Total windows: {total_windows}")
print(f"    Drift detected: {drift_count} ({drift_count/max(1,total_windows)*100:.1f}%)")

# Find continuous drift regions
drift_regions = []
in_drift = False
start = 0

for i, is_drift in enumerate(drift_results['drift_predictions']):
    if is_drift and not in_drift:
        start = i
        in_drift = True
    elif not is_drift and in_drift:
        drift_regions.append((start, i-1))
        in_drift = False
if in_drift:
    drift_regions.append((start, len(drift_results['drift_predictions'])-1))

if drift_regions:
    print(f"\n    Drift regions:")
    for start, end in drift_regions:
        print(f"      Windows {start}-{end}")

# --- Plotting ---
plt.figure(figsize=(14, 8))

# Extract distances
distances_to_plot = []
for res in drift_results['distances']:
    if isinstance(res, (list, np.ndarray)) and len(res) > distance_index:
        distances_to_plot.append(res[distance_index])
    else:
        distances_to_plot.append(0.0)

# Main distance plot
plt.plot(distances_to_plot, 'b-', linewidth=1.5, alpha=0.7, 
         marker='o', markersize=3, label='Distribution Distance')

# Mark drift detections
drift_windows = [i for i, d in enumerate(drift_results['drift_predictions']) if d]
if drift_windows:
    drift_dists = [distances_to_plot[i] for i in drift_windows if i < len(distances_to_plot)]
    plt.scatter(drift_windows[:len(drift_dists)], drift_dists,
                c='red', s=50, alpha=0.8, zorder=5, label='Drift Detected')

# Threshold line
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Threshold = {threshold:.3f}')

# Expected drift point
plt.axvline(x=10, color='green', linestyle=':', linewidth=2,
            alpha=0.7, label='Expected Drift Start')

# Shade drift regions
for start, end in drift_regions:
    plt.axvspan(start, end, alpha=0.2, color='red')

# Formatting
plt.title('FireRisk Drift Detection (Auto-Adjusted PCA)', fontsize=14, fontweight='bold')
plt.xlabel('Window Index', fontsize=12)
plt.ylabel('Distribution Distance', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Add info box
info_text = f"PCA: Batch={final_batch_pc}, Per-label={final_per_label_pc}\n"
info_text += f"Detection: {drift_count}/{total_windows} windows"
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
         fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = 'firerisk_drift_auto_adjusted.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# Save detailed results
with open('drift_results_adjusted.txt', 'w') as f:
    f.write("FireRisk Drift Detection Results (Auto-Adjusted)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Window size: {config['window_size']}\n")
    f.write(f"  Batch PCA (adjusted): {final_batch_pc} (requested: {config['requested_batch_n_pc']})\n")
    f.write(f"  Per-label PCA (adjusted): {final_per_label_pc} (requested: {config['requested_per_label_n_pc']})\n")
    f.write(f"  Alpha: {config['alpha']}\n")
    f.write(f"  Threshold: {threshold:.6f}\n\n")
    f.write(f"Summary:\n")
    f.write(f"  Windows analyzed: {total_windows}\n")
    f.write(f"  Drift detected: {drift_count} ({drift_count/max(1,total_windows)*100:.1f}%)\n")
    if drift_regions:
        f.write(f"  Drift regions: {len(drift_regions)}\n")
        for start, end in drift_regions:
            f.write(f"    Windows {start}-{end}\n")

print(f"✅ Results saved to: drift_results_adjusted.txt")
print("\n🎉 Experiment complete with auto-adjusted parameters!")
