from driftlens.driftlens import DriftLens
import matplotlib.pyplot as plt

# Configuration (Standard DriftLens settings)
config = {
    # File paths are kept in the config
    "baseline_path": "baseline.hdf5",
    "threshold_path": "threshold.hdf5",
    "data_stream_path": "datastream.hdf5",
    "window_size": 500,           # Matches our simulation
    "batch_n_pc": 150,            # PCA components for batch distribution
    "per_label_n_pc": 75,         # PCA components for per-label distribution
    "alpha": 0.05                 # Significance level
}

print("Initializing DriftLens...")
# FIX: Initialize the class without any arguments.
drift_lens = DriftLens() 

print("Running Drift Detection...")

# FIX: Pass all required configuration and paths to the run() method.
drift_results = drift_lens.run(
    baseline_path=config["baseline_path"],
    threshold_path=config["threshold_path"],
    data_stream_path=config["data_stream_path"],
    window_size=config["window_size"],
    batch_n_pc=config["batch_n_pc"],
    per_label_n_pc=config["per_label_n_pc"]
)

# --- Output & Visualization ---
print("\nDrift Detection Results:")
# Note: The threshold attribute is accessed after run()
threshold = drift_lens.threshold

for i, is_drift in enumerate(drift_results['drift_predictions']):
    status = "DRIFT DETECTED" if is_drift else "Normal"
    print(f"Window {i}: {status} (Distance: {drift_results['distances'][i]:.4f})")

# Basic Plotting of Drift Distance
plt.figure(figsize=(10, 6))
plt.plot(drift_results['distances'], label='Drift Distance', marker='o')
# Use the threshold calculated during the run() call
plt.axhline(y=threshold, color='r', linestyle='--', label='Drift Threshold') 
plt.axvline(x=10, color='g', linestyle=':', label='Drift Injection Start')
plt.title("DriftLens on FireRisk: Experiment 8 (Sudden Drift)")
plt.xlabel("Window Index")
plt.ylabel("Distribution Distance")
plt.legend()
plt.grid(True)
plt.savefig("experiment8_firerisk_results.png")
print("\nExperiment complete. Results saved to 'experiment8_firerisk_results.png'.")
