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
# Based on your screenshot, your notebook is in '~/datasets'
# and the 'FireRisk' folder is in the same directory.
LOCAL_DATASET_PATH = "FireRisk" 
# ---------------------

# 1. Setup Device and Model (ResNet50 for Embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Identity() # Remove classification layer
model.to(device)
model.eval()

# 2. Define Transforms (FIXED)
# We manually define the standard ImageNet normalization here
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(dataset_split):
    embeddings = []
    labels = []
    
    # Collate function to handle the decoded images from Arrow format
    def collate_fn(batch):
        # Convert images to tensors
        # .convert("RGB") ensures we handle grayscale/RGBA correctly if present
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
    # Fallback: try absolute path if running from a different location
    alt_path = os.path.join(os.getcwd(), "datasets", "FireRisk")
    if os.path.exists(alt_path):
        LOCAL_DATASET_PATH = alt_path
    else:
        raise FileNotFoundError(f"Could not find dataset at '{LOCAL_DATASET_PATH}'. Ensure you are in the correct directory.")

print(f"Loading local dataset from: {LOCAL_DATASET_PATH}...")
dataset_dict = load_from_disk(LOCAL_DATASET_PATH)

# Your screenshot shows 'dataset_dict.json', so this loads as a dict containing 'train' and 'val'
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

# Identify Risk Classes (Assuming 0-indexed labels: 3 and 4 are high risk)
high_risk_indices = np.where((Y_stream_raw == 3) | (Y_stream_raw == 4))[0]
low_risk_indices = np.where((Y_stream_raw != 3) & (Y_stream_raw != 4))[0]

rng = np.random.default_rng(seed=42)

for w in range(n_windows):
    if w < 10:
        # No Drift: Random sample from general population
        idx = rng.choice(len(Y_stream_raw), window_size)
        e_win, y_win = E_stream_raw[idx], Y_stream_raw[idx]
    else:
        # Sudden Drift: 80% High Risk injection
        n_drift = int(window_size * 0.8)
        n_norm = window_size - n_drift
        
        # Use replace=True to avoid errors if we run out of high/low risk samples
        idx_d = rng.choice(high_risk_indices, n_drift, replace=True)
        idx_n = rng.choice(low_risk_indices, n_norm, replace=True)
        
        e_win = np.vstack([E_stream_raw[idx_d], E_stream_raw[idx_n]])
        y_win = np.concatenate([Y_stream_raw[idx_d], Y_stream_raw[idx_n]])
        
        # Shuffle within window
        p = rng.permutation(window_size)
        e_win, y_win = e_win[p], y_win[p]
        
    stream_embeddings.append(e_win)
    stream_labels.append(y_win)

E_final_stream = np.vstack(stream_embeddings)
Y_final_stream = np.concatenate(stream_labels)

save_hdf5("datastream.hdf5", E_final_stream, Y_final_stream)
print("Data Preparation Complete. You can now run the DriftLens experiment script.")
