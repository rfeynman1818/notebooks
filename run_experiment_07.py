import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import sys

# --- 1. Configuration ---
config = {
    # File paths (must be pre-generated HDF5 files)
    "baseline_path": "baseline.hdf5",
    "threshold_path": "threshold.hdf5",
    "data_stream_path": "datastream.hdf5",
    
    "baseline_name": "fire_risk_baseline",  # New required argument for load_baseline()
    
    "window_size": 500,        # Stream window size
    "batch_n_pc": 150,         # PCA components for batch distribution
    "per_label_n_pc": 75,      # PCA components for per-label distribution
    "alpha": 0.05              # Significance level (p-value for detection)
}

# --- 2. Initialize DriftLens ---
print("Initializing DriftLens...")
# CORRECT: Initialize the class without arguments (Fixes TypeError from 164.png)
drift_lens = DriftLens()


# --- 3. Offline Phase: Load Baseline and Threshold ---
try:
    print("Loading Baseline and Threshold...")
    # CORRECT: load_baseline requires two positional arguments: baseline_name and baseline_path (Fixes TypeError from 176.png)
    drift_lens.load_baseline(config["baseline_name"], config["baseline_path"])
    
    # load_threshold likely takes threshold_path and also sets the internal 'threshold' property
    drift_lens.load_threshold(config["threshold_path"])
    
    # Retrieve the loaded threshold value for plotting
    threshold = drift_lens.threshold 

except FileNotFoundError as e:
    print(f"\nCRITICAL ERROR: Data file not found. Ensure all HDF5 files exist. Details: {e}")
    sys.exit(1)
except Exception as e:
    # This block handles the original TypeError from 176.png and potential other errors
    print(f"An error occurred during resource loading: {e}") 
    sys.exit(1)


# --- 4. Online Phase: Run Detection (Sequential Steps) ---
# This sequential chain uses the public methods from 172.png since single-call methods (.run(), .execute()) failed.

print("Computing Distribution Distances...")
# Step 4a: Calculate the distribution distances for each window in the data stream.
# The internal state (baseline) must be set by load_baseline() for this to work.
distances = drift_lens.compute_window_distribution_distances(
    data_stream_path=config["data_stream_path"],
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)

print("Calculating Drift Predictions...")
# Step 4b: Compare distances against the loaded threshold to get binary predictions.
drift_predictions = drift_lens.compute_drift_probability(
    distances=distances,
    threshold=threshold,
    alpha=config["alpha"]
)

# Step 4c: Manually construct the expected 'drift_results' dictionary for plotting 
# (This matches the structure expected in 165.png and 168.png).
drift_results = {
    'distances': distances,
    'drift_predictions': drift_predictions
}


# --- 5. Output & Visualization ---
print("\nDrift Detection Results:")
# Use the last element in the distances array (distance_index) for the plot and printout
distance_index = config["batch_n_pc"] - 1

# Print results for each window
for i, is_drift in enumerate(drift_results['drift_predictions']):
    status = "DRIFT DETECTED" if is_drift else "Normal"
    distance = drift_results['distances'][i][distance_index] 
    print(f"Window {i:03d}: {status:15} (Distance: {distance:.4f})")

# Basic Plotting of Drift Distance
plt.figure(figsize=(10, 6))
distances_to_plot = [res[distance_index] for res in drift_results['distances']]
plt.plot(distances_to_plot, label='Drift Distance', marker='o', linestyle='-')

# Plot the Drift Threshold
plt.axhline(
    y=threshold, 
    color='r', 
    linestyle='--', 
    label='Drift Threshold'
)

# Plot the Drift Injection Start (Assumed at window index 10)
plt.axvline(
    x=10, 
    color='g', 
    linestyle=':', 
    label='Drift Injection Start'
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
