# ... (Part 1, 2, and 3 - Model setup and feature extraction function - remain the same) ...

# --- HDF5 Saving Helper Function (FIXED KEYS) ---
def save_hdf5(filepath, E_data, Y_data):
    """Saves embedding and label arrays using the standard expected keys."""
    with h5py.File(filepath, 'w') as hf:
        # **CRITICAL FIX:** Use 'embeddings' and 'labels' keys for detection script compatibility
        hf.create_dataset('embeddings', data=E_data) 
        hf.create_dataset('labels', data=Y_data)     
    print(f"Successfully saved data to {filepath}")
# ---------------------

# 3. Load Local FireRisk Dataset (This part is correct)
# ...
dataset = dataset_dict["train"]
total_samples = len(dataset)
print(f"Loaded {total_samples} total samples.")

# 4. Split Dataset into Baseline, Threshold, and Stream (CORRECTED SLICING)
print("\nSplitting dataset (10k baseline, 10k threshold)...")

# Define the boundaries
N_BASELINE = 10000 
N_THRESHOLD = 10000 
idx_thresh_start = N_BASELINE
idx_stream_start = N_BASELINE + N_THRESHOLD

# Baseline split (First 10,000 samples)
ds_baseline = dataset.select(range(N_BASELINE))

# Threshold split (Samples 10,000 to 19,999)
ds_threshold = dataset.select(range(idx_thresh_start, idx_stream_start))

# Stream source split (Remaining samples from 20,000 onwards)
ds_stream_source = dataset.select(range(idx_stream_start, total_samples))


# 5. Extract Embeddings (CORRECTED)

# Extract Baseline Embeddings
print(f"Extracting Baseline Embeddings ({len(ds_baseline)} samples)...")
E_base, Y_base = extract_features(ds_baseline)

# Extract Threshold Embeddings
print(f"Extracting Threshold Embeddings ({len(ds_threshold)} samples)...")
E_thresh, Y_thresh = extract_features(ds_threshold)
# Check samples count to confirm fix
print(f"Baseline extracted: {len(E_base)} | Threshold extracted: {len(E_thresh)}")

# --- DRIFT SIMULATION (Part of the script that creates the final stream) ---
# ... (The rest of the script remains the same, ensuring it uses E_thresh/Y_thresh)
E_stream_source, Y_stream_source = extract_features(ds_stream_source)
# ... (The rest of the stream construction logic remains the same) ...


# 6. Save HDF5 files (Uses the corrected save_hdf5 function)
print("\nSaving HDF5 files...")
# These files will now contain the correct keys: 'embeddings' and 'labels'
save_hdf5("baseline.hdf5", E_base, Y_base)
save_hdf5("threshold.hdf5", E_thresh, Y_thresh)
save_hdf5("datastream.hdf5", E_final_stream, Y_final_stream)
