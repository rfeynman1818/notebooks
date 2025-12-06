import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_from_disk
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# Update this path to point to the folder containing 'train' and 'val'
# Based on your screenshot, if your script is in '~', this is likely correct:
LOCAL_DATASET_PATH = "datasets/FireRisk" 
# ---------------------

# 1. Setup Device and Model (ResNet50 for Embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Identity() # Remove classification layer
model.to(device)
model.eval()

# 2. Define Transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    weights.transforms().normalize,
])

def extract_features(dataset_split):
    embeddings = []
    labels = []
    
    # Collate function to handle the decoded images from Arrow format
    def collate_fn(batch):
        # Convert images to tensors
        # Note: load_from_disk usually handles PIL decoding automatically
        imgs = [preprocess(item['image'].convert("RGB")) for item in batch]
        lbls = [item['label'] for item in batch]
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
    raise FileNotFoundError(f"Could not find dataset at {LOCAL_DATASET_PATH}. Please check the path.")

print(f"Loading local dataset from: {LOCAL_DATASET_PATH}...")
dataset_dict = load_from_disk(LOCAL_DATASET_PATH)

# The screenshot shows 'train' and 'val' folders, so it loads as a DatasetDict
# We will use 'train' for the experiment to ensure we have enough data
dataset = dataset_dict["train"]

print(f"Loaded {len(dataset)} samples.")

# 4. Split into Baseline, Threshold, and Stream
# Shuffle to ensure random distribution before we artificially inject drift
dataset = dataset.shuffle(seed=42)

n = len(dataset)
idx_base = int(n * 0.4)     # 40% for Baseline
idx_thresh = int(n * 0.5)   # 10% for Threshold (validation)
# Remaining 50% for Stream Source

ds_baseline = dataset.select(range(0, idx_base))
ds_threshold = dataset.select(range(idx_base, idx_thresh))
ds_stream_source = dataset.select(range(idx_thresh, n))

print("Extracting Baseline Embeddings...")
E_base, Y_base = extract_features(ds_baseline)

print("Extracting Threshold Embeddings...")
E_thresh, Y_thresh = extract_features(ds_threshold)

print("Extracting Stream Source Embeddings...")
E_stream_raw, Y_stream_raw = extract_features(ds_stream_source)

# 5. Save Baseline and Threshold files
def save_hdf5(filename, embeddings, labels):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("E", data=embeddings)
        f.create_dataset("Y_predicted", data=labels)
    print(f"Saved {filename}")

save_hdf5("baseline.hdf5", E_base, Y_base)
save_hdf5("threshold.hdf5", E_thresh, Y_thresh)

# --- DRIFT SIMULATION (Experiment 8: Sudden Drift) ---
print("Simulating Sudden Drift Stream...")

# Setup windows
window_size = 500  
n_windows = 20
stream_embeddings = []
stream_labels = []

# Identify Risk Classes (Assuming 0-indexed: 3=High, 4=Very High)
# Adjust these indices if your specific version of FireRisk uses different integers
high_risk_indices = np.where((Y_stream_raw == 3) | (Y_stream_raw == 4))[0]
low_risk_indices = np.where((Y_stream_raw != 3) & (Y_stream_raw != 4))[0]

rng = np.random.default_rng(seed=42)

for w in range(n_windows):
    if w < 10:
        # No Drift: Random sample from general population
        # Use modulo or replacement to avoid running out of data
        idx = rng.choice(len(Y_stream_raw), window_size)
        e_win, y_win = E_stream_raw[idx], Y_stream_raw[idx]
    else:
        # Sudden Drift: 80% High Risk injection
        n_drift = int(window_size * 0.8)
        n_norm = window_size - n_drift
        
        # Safe sampling with replacement ensures we don't crash if counts are low
        idx_d = rng.choice(high_risk_indices, n_drift, replace=True)
        idx_n = rng.choice(low_risk_indices, n_norm, replace=True)
        
        e_win = np.vstack([E_stream_raw[idx_d], E_stream_raw[idx_n]])
        y_win = np.concatenate([Y_stream_raw[idx_d], Y_stream_raw[idx_n]])
        
        # Shuffle within window so it looks like a real stream
        p = rng.permutation(window_size)
        e_win, y_win = e_win[p], y_win[p]
        
    stream_embeddings.append(e_win)
    stream_labels.append(y_win)

E_final_stream = np.vstack(stream_embeddings)
Y_final_stream = np.concatenate(stream_labels)

save_hdf5("datastream.hdf5", E_final_stream, Y_final_stream)
print("Data Preparation Complete. You can now run the DriftLens experiment script.")
