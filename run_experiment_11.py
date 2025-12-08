import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import sys
import os

# --- 1. Configuration ---
config = {
    # File paths (must be the HDF5 files created by the preparation script)
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


# --- 3. Offline Phase: Estimate Baseline and Threshold (FINAL FIX) ---
try:
    print("\n--- 3a. Estimating Baseline ---")
    
    # CRITICAL FIX: The error in 192.png shows 'baseline_path' must be positional.
    drift_lens.estimate_baseline(
        config["baseline_path"],  # Path passed as the required first positional argument
        batch_n_pc=config["batch_n_pc"],
        per_label_n_pc=config["per_label_n_pc"]
    )
    print("✅ Baseline estimated and internal state set successfully.")

    print("\n--- 3b. Estimating Threshold ---")
    # This method also likely requires a positional argument for the path.
    threshold = drift_lens.KFold_threshold_estimation(
        config["threshold_path"],  # Path passed as the required first positional argument
        window_size=config["window_size"],
        n_splits=config["n_splits"],
        alpha=config["alpha"]
    )
    print(f"✅ Threshold estimated: {threshold:.4f}")

except FileNotFoundError as e:
    print(f"\n❌ CRITICAL ERROR: Data file not found. Ensure all HDF5 files exist. Details: {e}")
    sys.exit(1)
except Exception as e:
    # This captures the TypeError from 192.png if it still occurs, but with the positional fix, it shouldn't.
    print(f"❌ An unexpected error occurred during resource estimation: {e}") 
    sys.exit(1)

print("\nOffline Phase Complete.")


# --- 4. Online Phase: Run Detection (Flow remains the same) ---
print("\n--- 4a. Computing Distribution Distances ---")
# This method also likely requires a positional argument for the data_stream_path
distances = drift_lens.compute_window_distribution_distances(
    config["data_stream_path"], # Path passed as the required first positional argument
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)
print(f"✅ Computed distances for {len(distances)} windows.")

print("\n--- 4b. Calculating Drift Predictions ---")
# Step 4b: Compare distances against the calculated threshold.
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
plt.title('DriftLens on FireRisk: Experiment 8 (Sudden Drift)')
plt.xlabel("Window Index")
plt.ylabel("Distribution Distance")

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
output_filename = 'experiment8_firerisk_results_final.png'
plt.savefig(output_filename)
print(f"\nExperiment complete. Results saved to '{output_filename}'.")
