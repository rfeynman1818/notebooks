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
    
    # Required positional argument for load_baseline()
    "baseline_name": "fire_risk_baseline",  
    
    # DriftLens Parameters
    "window_size": 500,        # Stream window size
    "batch_n_pc": 150,         # PCA components for batch distribution
    "per_label_n_pc": 75,      # PCA components for per-label distribution
    "alpha": 0.05              # Significance level (p-value for detection)
}

# --- 2. Initialize DriftLens ---
print("Initializing DriftLens...")
# FIX: Initialize the class without arguments (Fixes original TypeError)
drift_lens = DriftLens()


# --- 3. Offline Phase: Load Baseline and Threshold ---
try:
    print("Loading Baseline and Threshold resources...")
    
    # FIX: load_baseline requires two positional arguments: baseline_name and baseline_path 
    drift_lens.load_baseline(config["baseline_name"], config["baseline_path"])
    
    # load_threshold sets the internal 'threshold' property
    drift_lens.load_threshold(config["threshold_path"])
    
    # Retrieve the loaded threshold value for plotting
    threshold = drift_lens.threshold 

except FileNotFoundError as e:
    print(f"\nCRITICAL ERROR: Data file not found. Ensure all HDF5 files exist. Details: {e}")
    sys.exit(1)
except Exception as e:
    # Handles errors during resource loading
    print(f"An error occurred during resource loading: {e}") 
    sys.exit(1)

print("Resources loaded successfully.")


# --- 4. Online Phase: Run Detection (Sequential Steps) ---
# FIX: Using sequential public methods instead of failed single-call methods (.run(), .execute(), .detect())

print("\nComputing Distribution Distances...")
# Step 4a: Calculate the distribution distances for each window in the data stream.
# This requires the internal baseline state to be set by load_baseline().
distances = drift_lens.compute_window_distribution_distances(
    data_stream_path=config["data_stream_path"],
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)
print(f"Computed distances for {len(distances)} windows.")

print("Calculating Drift Predictions...")
# Step 4b: Compare distances against the loaded threshold to get binary predictions.
drift_predictions = drift_lens.compute_drift_probability(
    distances=distances,
    threshold=threshold,
    alpha=config["alpha"]
)

# Step 4c: Manually construct the expected 'drift_results' dictionary for plotting
# (This simulates the output of the failed single-call methods).
drift_results = {
    'distances': distances,
    'drift_predictions': drift_predictions
}


# --- 5. Output & Visualization ---
print("\nDrift Detection Results:")
# The distance index is the index within the distance vector that corresponds to the batch distribution (full-dataset drift)
distance_index = config["batch_n_pc"] - 1

# Print results for each window
for i, is_drift in enumerate(drift_results['drift_predictions']):
    # Safety check: ensure distance list is not empty
    if i < len(drift_results['distances']):
        distance = drift_results['distances'][i][distance_index] 
    else:
        distance = 0.0 # Placeholder if somehow mismatched
        
    status = "DRIFT DETECTED" if is_drift else "Normal"
    print(f"Window {i:03d}: {status:15} (Distance: {distance:.4f})")

# Basic Plotting of Drift Distance
plt.figure(figsize=(10, 6))
# Extract the relevant distance metric (the batch distance) for plotting
distances_to_plot = [res[distance_index] for res in drift_results['distances']]
plt.plot(distances_to_plot, label='Drift Distance', marker='o', linestyle='-', markersize=2)

# Plot the Drift Threshold
plt.axhline(
    y=threshold, 
    color='r', 
    linestyle='--', 
    label=f'Drift Threshold (p={config["alpha"]})'
)

# Plot the Drift Injection Start (Assumed at window index 10, i.e., at x=10)
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
