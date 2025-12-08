import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import sys
import os
import h5py # Necessary for manual data loading

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

# The complete list of unique integer labels (0 to 6) 
FULL_LABEL_LIST = list(range(7)) 

# --- 2. Initialize DriftLens ---
print("Initializing DriftLens...")
drift_lens = DriftLens()


# --- 3. Offline Phase: Data Loading and Estimation (HDF5 KEY FIX) ---
try:
    print("\n--- 3a. Manually Loading All HDF5 Data ---")
    
    # FIX: Use the actual keys found in the HDF5 files: 'E' and 'Y_predicted' (from 183.png)
    
    # 1. Load Baseline Data
    with h5py.File(config["baseline_path"], 'r') as hf:
        E_base = hf['E'][:] # Corrected key
        Y_base = hf['Y_predicted'][:] # Corrected key
    print(f"Loaded {len(E_base)} baseline samples.")

    # 2. Load Threshold Data
    with h5py.File(config["threshold_path"], 'r') as hf:
        E_thresh = hf['E'][:] # Corrected key
        Y_thresh = hf['Y_predicted'][:] # Corrected key
    print(f"Loaded {len(E_thresh)} threshold samples.")


    print("\n--- 3b. Estimating Baseline ---")
    
    # E and Y arrays are passed positionally
    drift_lens.estimate_baseline(
        E_base,                     # 1st Positional: Embeddings Array (E)
        Y_base,                     # 2nd Positional: Labels Array (Y)
        FULL_LABEL_LIST,            # 3rd Positional: All possible labels
        batch_n_pc=config["batch_n_pc"],
        per_label_n_pc=config["per_label_n_pc"]
    )
    print("✅ Baseline estimated and internal state set successfully.")

    print("\n--- 3c. Estimating Threshold ---")
    # E and Y arrays are passed positionally
    threshold = drift_lens.KFold_threshold_estimation(
        E_thresh,                   # 1st Positional: Threshold Embeddings Array (E)
        Y_thresh,                   # 2nd Positional: Threshold Labels Array (Y)
        window_size=config["window_size"],
        n_splits=config["n_splits"],
        alpha=config["alpha"]
    )
    print(f"✅ Threshold estimated: {threshold:.4f}")

except FileNotFoundError as e:
    print(f"\n❌ CRITICAL ERROR: Data file not found. Ensure all HDF5 files exist. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred during resource estimation: {e}") 
    sys.exit(1)

print("\nOffline Phase Complete.")


# --- 4. Online Phase: Run Detection ---
print("\n--- 4a. Computing Distribution Distances ---")
# The datastream path must be passed positionally.
distances = drift_lens.compute_window_distribution_distances(
    config["data_stream_path"], # 1st Positional: Stream HDF5 Path
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

# Manually construct the expected 'drift_results' dictionary for plotting
drift_results = {
    'distances': distances,
    'drift_predictions': drift_predictions
}


# --- 5. Output & Visualization ---
print("\nDrift Detection Results:")
distance_index = config["batch_n_pc"] - 1

# Print results for each window
for i, is_drift in enumerate(drift_results['drift_predictions']):
    if i < len(drift_results['distances']):
        distance = drift_results['distances'][i][distance_index] 
    else:
        distance = 0.0
        
    status = "DRIFT DETECTED" if is_drift else "Normal"
    print(f"Window {i:03d}: {status:15} (Distance: {distance:.4f})")

# Basic Plotting of Drift Distance
plt.figure(figsize=(10, 6))
distances_to_plot = [res[distance_index] for res in drift_results['distances']]
plt.plot(distances_to_plot, label='Drift Distance', marker='o', linestyle='-', markersize=2)

# Plot the Drift Threshold
plt.axhline(
    y=threshold, 
    color='r', 
    linestyle='--', 
    label=f'Drift Threshold (p={config["alpha"]})'
)

# Plot the Drift Injection Start (Assumed at window index 10)
drift_injection_window = 10
plt.axvline(
    x=drift_injection_window, 
    color='g', 
    linestyle=':', 
    label='Drift Injection Start (Window 10)'
)

# Set labels and title
plt.title('DRIFTLENS on FireRisk: Experiment 8 (Sudden Drift)')
plt.xlabel("Window Index")
plt.ylabel("Distribution Distance")

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
output_filename = 'experiment8_firerisk_results_final.png'
plt.savefig(output_filename)
print(f"\nExperiment complete. Results saved to '{output_filename}'.")
