import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import sys
import os
import h5py
from collections import Counter

# --- 1. Configuration ---
config = {
    # File paths
    "baseline_path": "baseline.hdf5",
    "threshold_path": "threshold.hdf5",
    "data_stream_path": "datastream.hdf5",
    
    # DriftLens Parameters
    "window_size": 500,        # Stream window size
    "batch_n_pc": 150,         # PCA components for batch distribution
    "per_label_n_pc": 75,      # PCA components for per-label distribution
    "alpha": 0.05,             # Significance level (p-value for detection)
    "n_splits": 5              # K-Fold splits for threshold estimation
}

# --- 2. Initialize DriftLens ---
print("Initializing DriftLens...")
drift_lens = DriftLens()

# --- 3. Offline Phase: Data Loading and Estimation ---
try:
    print("\n--- 3a. Loading and Analyzing HDF5 Data ---")
    
    # Load Baseline Data
    with h5py.File(config["baseline_path"], 'r') as hf:
        E_base = hf['embeddings'][:]
        Y_base = hf['labels'][:]
    print(f"Loaded {len(E_base)} baseline samples.")
    
    # Load Threshold Data
    with h5py.File(config["threshold_path"], 'r') as hf:
        E_thresh = hf['embeddings'][:]
        Y_thresh = hf['labels'][:]
    print(f"Loaded {len(E_thresh)} threshold samples.")
    
    # CRITICAL FIX: Determine which labels are actually present
    baseline_labels = sorted(list(set(Y_base)))
    threshold_labels = sorted(list(set(Y_thresh)))
    
    # Use only labels that exist in BOTH baseline and threshold
    common_labels = sorted(list(set(baseline_labels) & set(threshold_labels)))
    
    print(f"\nLabel Analysis:")
    print(f"  Baseline contains labels: {baseline_labels}")
    print(f"  Threshold contains labels: {threshold_labels}")
    print(f"  Common labels (will use these): {common_labels}")
    
    # Check if we have enough labels
    if len(common_labels) == 0:
        raise ValueError("No common labels between baseline and threshold datasets!")
    
    # Optional: Show label distribution
    print("\nBaseline label distribution:")
    base_counter = Counter(Y_base)
    for label in sorted(base_counter.keys()):
        print(f"  Label {label}: {base_counter[label]} samples")
    
    print("\n--- 3b. Estimating Baseline ---")
    
    # FIX: Use only the labels that actually exist in the data
    drift_lens.estimate_baseline(
        E_base,                     # Embeddings Array
        Y_base,                     # Labels Array
        common_labels,              # ONLY labels that exist in both datasets
        batch_n_pc=config["batch_n_pc"],
        per_label_n_pc=config["per_label_n_pc"]
    )
    print("✅ Baseline estimated successfully with available labels.")

    print("\n--- 3c. Estimating Threshold ---")
    threshold = drift_lens.KFold_threshold_estimation(
        E_thresh,                   # Threshold Embeddings Array
        Y_thresh,                   # Threshold Labels Array
        window_size=config["window_size"],
        n_splits=config["n_splits"],
        alpha=config["alpha"]
    )
    print(f"✅ Threshold estimated: {threshold:.4f}")

except FileNotFoundError as e:
    print(f"\n❌ CRITICAL ERROR: Data file not found. Details: {e}")
    sys.exit(1)
except ValueError as e:
    print(f"❌ ValueError during estimation: {e}")
    print("\nDebugging info:")
    print(f"  Unique labels in baseline: {sorted(list(set(Y_base)))}")
    print(f"  Unique labels in threshold: {sorted(list(set(Y_thresh)))}")
    print(f"  Sample counts per label in baseline:")
    for label in sorted(set(Y_base)):
        count = np.sum(Y_base == label)
        print(f"    Label {label}: {count} samples")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nOffline Phase Complete.")

# --- 4. Online Phase: Run Detection ---
print("\n--- 4a. Computing Distribution Distances ---")
distances = drift_lens.compute_window_distribution_distances(
    config["data_stream_path"],
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)
print(f"✅ Computed distances for {len(distances)} windows.")

print("\n--- 4b. Calculating Drift Predictions ---")
drift_predictions = drift_lens.compute_drift_probability(
    distances=distances,
    threshold=threshold,
    alpha=config["alpha"]
)

drift_results = {
    'distances': distances,
    'drift_predictions': drift_predictions
}

# --- 5. Output & Visualization ---
print("\nDrift Detection Results:")
distance_index = config["batch_n_pc"] - 1

# Count drift detections
drift_count = sum(drift_results['drift_predictions'])
print(f"Total windows with drift detected: {drift_count}/{len(drift_predictions)}")

# Print first 20 windows and last 10 windows for inspection
print("\nFirst 20 windows:")
for i in range(min(20, len(drift_results['drift_predictions']))):
    is_drift = drift_results['drift_predictions'][i]
    if i < len(drift_results['distances']):
        distance = drift_results['distances'][i][distance_index]
    else:
        distance = 0.0
    status = "DRIFT DETECTED" if is_drift else "Normal"
    print(f"  Window {i:03d}: {status:15} (Distance: {distance:.4f})")

if len(drift_results['drift_predictions']) > 20:
    print("\nLast 10 windows:")
    start_idx = max(20, len(drift_results['drift_predictions']) - 10)
    for i in range(start_idx, len(drift_results['drift_predictions'])):
        is_drift = drift_results['drift_predictions'][i]
        if i < len(drift_results['distances']):
            distance = drift_results['distances'][i][distance_index]
        else:
            distance = 0.0
        status = "DRIFT DETECTED" if is_drift else "Normal"
        print(f"  Window {i:03d}: {status:15} (Distance: {distance:.4f})")

# Plotting
plt.figure(figsize=(12, 7))

# Plot distances
distances_to_plot = [res[distance_index] for res in drift_results['distances']]
plt.plot(distances_to_plot, label='Distribution Distance', 
         color='blue', linewidth=1.5, alpha=0.7)

# Mark drift detections
drift_windows = [i for i, is_drift in enumerate(drift_results['drift_predictions']) if is_drift]
if drift_windows:
    drift_distances = [distances_to_plot[i] for i in drift_windows]
    plt.scatter(drift_windows, drift_distances, 
                color='red', s=30, alpha=0.6, label='Drift Detected', zorder=5)

# Plot threshold line
plt.axhline(y=threshold, color='red', linestyle='--', 
            label=f'Drift Threshold = {threshold:.4f}', linewidth=2)

# Mark drift injection point (from prepare_firerisk.py)
drift_injection_window = 10  # Based on DRIFT_INJECTION_START_WINDOW = 10
plt.axvline(x=drift_injection_window, color='green', linestyle=':', 
            label='Expected Drift Start (Window 10)', linewidth=2)

# Styling
plt.title('DriftLens on FireRisk: Sudden Drift Detection (Experiment 8)', fontsize=14, fontweight='bold')
plt.xlabel('Window Index', fontsize=12)
plt.ylabel('Distribution Distance', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add summary text
summary_text = f"Detected Drift: {drift_count}/{len(drift_predictions)} windows"
plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save the plot
output_filename = 'firerisk_drift_detection_fixed.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Experiment complete. Results saved to '{output_filename}'")

# Save numerical results
results_file = 'drift_detection_results.txt'
with open(results_file, 'w') as f:
    f.write("FireRisk Drift Detection Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Configuration:\n")
    f.write(f"  Window size: {config['window_size']}\n")
    f.write(f"  Batch PCA components: {config['batch_n_pc']}\n")
    f.write(f"  Per-label PCA components: {config['per_label_n_pc']}\n")
    f.write(f"  Significance level (alpha): {config['alpha']}\n")
    f.write(f"  Threshold: {threshold:.6f}\n")
    f.write(f"\nResults:\n")
    f.write(f"  Total windows analyzed: {len(drift_predictions)}\n")
    f.write(f"  Windows with drift detected: {drift_count}\n")
    f.write(f"  Drift injection point: Window {drift_injection_window}\n")
    f.write(f"\nPer-window results:\n")
    for i, is_drift in enumerate(drift_results['drift_predictions']):
        if i < len(drift_results['distances']):
            distance = drift_results['distances'][i][distance_index]
        else:
            distance = 0.0
        status = "DRIFT" if is_drift else "OK"
        f.write(f"  Window {i:03d}: {status:5} (Distance: {distance:.6f})\n")

print(f"✅ Numerical results saved to '{results_file}'")
