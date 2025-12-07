import matplotlib.pyplot as plt
from driftlens.driftlens import DriftLens
import numpy as np
import os
import sys

# --- 1. Configuration ---
config = {
    # NOTE: These HDF5 files must be pre-generated from the preparation script.
    "baseline_path": "baseline.hdf5",
    "threshold_path": "threshold.hdf5",
    "data_stream_path": "datastream.hdf5",
    
    "window_size": 500,        # Stream window size
    "batch_n_pc": 150,         # PCA components for batch distribution
    "per_label_n_pc": 75,      # PCA components for per-label distribution
    "alpha": 0.05              # Significance level for detection (e.g., 0.05)
}

# --- 2. Initialize DriftLens ---
print("Initializing DriftLens...")
drift_lens = DriftLens()


# --- 3. Offline Phase: Load Baseline and Threshold ---
# Since DriftLens is designed for sequential use, we load the required resources.
try:
    # Load Baseline and Threshold files, which sets internal properties 
    # (self.baseline and self.threshold) needed for the detection step.
    drift_lens.load_baseline(config["baseline_path"])
    drift_lens.load_threshold(config["threshold_path"])
    
    threshold = drift_lens.threshold # Retrieve the loaded threshold value

except FileNotFoundError as e:
    print(f"\nCRITICAL ERROR: Data file not found. Ensure the following files exist:")
    print(f"- {config['baseline_path']}")
    print(f"- {config['threshold_path']}")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during resource loading: {e}")
    sys.exit(1)


# --- 4. Online Phase: Run Detection (Sequential Steps) ---
print("Computing Distribution Distances...")

# Step 4a: Calculate the distribution distances for each window in the data stream 
# against the loaded baseline.
# This function requires the path to the data stream and the PCA settings.
distances = drift_lens.compute_window_distribution_distances(
    data_stream_path=config["data_stream_path"],
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)

print("Calculating Drift Predictions...")
# Step 4b: Compare the calculated distances against the loaded threshold to get 
# binary drift predictions.
drift_predictions = drift_lens.compute_drift_probability(
    distances=distances,
    threshold=threshold,
    alpha=config["alpha"]
)

# Step 4c: Manually construct the expected 'drift_results' dictionary for plotting
drift_results = {
    'distances': distances,
    'drift_predictions': drift_predictions
}


# --- 5. Output & Visualization (Matches the successful plot structure) ---
print("\nDrift Detection Results:")

# Print results for each window
for i, is_drift in enumerate(drift_results['drift_predictions']):
    status = "DRIFT DETECTED" if is_drift else "Normal"
    # Use the per-batch distance, which is the full vector distance
    # The last element in the distances array usually corresponds to the full batch comparison
    distance_index = config["batch_n_pc"] - 1
    distance = drift_results['distances'][i][distance_index] 
    print(f"Window {i:03d}: {status:15} (Distance: {distance:.4f})")

# Basic Plotting of Drift Distance
plt.figure(figsize=(10, 6))
# Plot the computed drift distances
distances_to_plot = [res[distance_index] for res in drift_results['distances']]
plt.plot(distances_to_plot, label='Drift Distance', marker='o', linestyle='-')

# Plot the Drift Threshold (Horizontal Red Line)
plt.axhline(
    y=threshold, 
    color='r', 
    linestyle='--', 
    label='Drift Threshold'
)

# Plot the Drift Injection Start (Vertical Green Line)
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
output_filename = 'experiment8_firerisk_results_corrected.png'
plt.savefig(output_filename)
print(f"\nExperiment complete. Results saved to '{output_filename}'.")
