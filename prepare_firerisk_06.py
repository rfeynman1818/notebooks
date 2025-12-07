import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_from_disk
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image

# --- CONFIGURATION ---
LOCAL_DATASET_PATH = "FireRisk" 
# --- DATA SPLIT SIZES (FIXED) ---
N_BASELINE = 10000 
N_THRESHOLD = 10000 

# --- LABEL MAPPING (CORRECT) ---
# Map string labels to 0-indexed integers.
LABEL_MAPPING = {
    'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4,
    'Non-burnable': 5, 'Water': 6
}
# ---------------------

# --- HDF5 Saving Helper Function (FIXED to use standard keys) ---
def save_hdf5(filepath, E_data, Y_data):
    """Saves embedding and label arrays using the standard expected keys."""
    with h5py.File(filepath, 'w') as hf:
        # CRITICAL FIX: Use 'embeddings' and 'labels' for DriftLens compatibility
        hf.create_dataset('embeddings', data=E_data) 
        hf.create_dataset('labels', data=Y_data)     
    print(f"Successfully saved data to {filepath}")
# ---------------------

# 1. Setup Device and Model (ResNet50 for Embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Identity() # Remove classification layer to get embeddings
model.to(device)
model.eval()

# 2. Define Transforms (CORRECT)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Standard ImageNet normalization values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(dataset_split):
    embeddings = []
    labels = []
    
    # Collate function (CORRECT)
    def collate_fn(batch):
        imgs = []
        lbls = []
        for item in batch:
            img_data = item['image']
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
            else:
                img = img_data.convert("RGB") 
                
            imgs.append(preprocess(img))
            
            lbl_str = item['label']
            if lbl_str in LABEL_MAPPING:
                lbls.append(LABEL_MAPPING[lbl_str])
            else:
                raise ValueError(f"Unknown label: {lbl_str}.")

        return torch.stack(imgs), torch.tensor(lbls) 

    loader = DataLoader(dataset_split, batch_size=32, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.vstack(embeddings), np.concatenate(labels)

# 3. Load Local FireRisk Dataset
if not os.path.exists(LOCAL_DATASET_PATH):
    raise FileNotFoundError(f"Could not find dataset at '{LOCAL_DATASET_PATH}'.")

print(f"Loading local dataset from: {LOCAL_DATASET_PATH}...")
dataset_dict = load_from_disk(LOCAL_DATASET_PATH)
dataset = dataset_dict["train"]
total_samples = len(dataset)
print(f"Loaded {total_samples} total samples.")

# 4. Split Dataset into Baseline, Threshold, and Stream (CORRECTED SLICING)
print("\nSplitting dataset (10k baseline, 10k threshold)...")

idx_thresh_start = N_BASELINE
idx_stream_start = N_BASELINE + N_THRESHOLD

# Baseline split (First 10,000 samples)
ds_baseline = dataset.select(range(N_BASELINE))

# Threshold split (Samples 10,000 to 19,999)
ds_threshold = dataset.select(range(idx_thresh_start, idx_stream_start))

# Stream source split (Remaining samples from 20,000 onwards)
ds_stream_source = dataset.select(range(idx_stream_start, total_samples))

print(f"Baseline samples allocated: {len(ds_baseline)}")
print(f"Threshold samples allocated: {len(ds_threshold)}")
print(f"Stream source samples allocated: {len(ds_stream_source)}")


# 5. Extract Embeddings 
print(f"Extracting Baseline Embeddings ({len(ds_baseline)} samples)...")
E_base, Y_base = extract_features(ds_baseline)

print(f"Extracting Threshold Embeddings ({len(ds_threshold)} samples)...")
E_thresh, Y_thresh = extract_features(ds_threshold)

# --- DRIFT SIMULATION (Experiment 8: Sudden Drift) ---
print("\nExtracting Stream Source Embeddings...")
E_stream_source, Y_stream_source = extract_features(ds_stream_source)

# Select the target label for the drift (e.g., 'Very High' is ID 4)
DRIFT_LABEL_ID = 4 
drift_indices = np.where(Y_stream_source == DRIFT_LABEL_ID)[0]

# Define the start and end of the drift injection
WINDOW_SIZE = 500
DRIFT_INJECTION_START_WINDOW = 10 
DRIFT_INJECTION_INDEX = DRIFT_INJECTION_START_WINDOW * WINDOW_SIZE 

# Construct the final stream array
E_stream_normal = E_stream_source[:DRIFT_INJECTION_INDEX]
Y_stream_normal = Y_stream_source[:DRIFT_INJECTION_INDEX]

E_stream_drift_section = E_stream_source[DRIFT_INJECTION_INDEX:]
required_drift_samples = len(E_stream_drift_section)

# Create the drifting population by sampling from the target label (ID 4)
E_drifted_population = E_stream_source[drift_indices]
Y_drifted_population = Y_stream_source[drift_indices]

# Pad the stream with drifting samples (using numpy's 'take' for wrapping if not enough)
# This simulates a persistent drift of the "Very High" class
E_final_drift = np.take(E_drifted_population, np.arange(required_drift_samples) % len(E_drifted_population), axis=0)
Y_final_drift = np.take(Y_drifted_population, np.arange(required_drift_samples) % len(Y_drifted_population), axis=0)

# Final stream concatenation
E_final_stream = np.vstack((E_stream_normal, E_final_drift))
Y_final_stream = np.concatenate((Y_stream_normal, Y_final_drift))

print(f"Total Stream Samples: {len(E_final_stream)}. Drift injected after index {DRIFT_INJECTION_INDEX}.")


# 6. Save HDF5 files (Uses the corrected save_hdf5 function with standard keys)
print("\nSaving HDF5 files...")
save_hdf5("baseline.hdf5", E_base, Y_base)
save_hdf5("threshold.hdf5", E_thresh, Y_thresh)
save_hdf5("datastream.hdf5", E_final_stream, Y_final_stream)

print("\nData Preparation Complete. All HDF5 files created with correct splits and standard keys.")
