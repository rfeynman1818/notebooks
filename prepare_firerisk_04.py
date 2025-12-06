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
# --- MAPPING FIX (NEW) ---
# Map string labels to 0-indexed integers. 
# This ensures the 'High' (3) and 'Very High' (4) risk classes 
# are correctly used for the drift simulation.
LABEL_MAPPING = {
    'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4,
    'Non-burnable': 5, 'Water': 6
}
# ---------------------

# 1. Setup Device and Model (ResNet50 for Embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Identity() # Remove classification layer
model.to(device)
model.eval()

# 2. Define Transforms (FIXED from previous error)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Standard ImageNet normalization values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(dataset_split):
    embeddings = []
    labels = []
    
    # Collate function (FIXED label retrieval)
    def collate_fn(batch):
        imgs = []
        lbls = []
        for item in batch:
            # Check if image is a string path and load it, or assume it's a PIL object
            img_data = item['image']
            if isinstance(img_data, str):
                img_data = Image.open(img_data).convert("RGB")
            else:
                img_data = img_data.convert("RGB")
                
            imgs.append(preprocess(img_data))
            
            # FIX: Look up the integer ID from the string label
            lbl_str = item['label']
            if lbl_str in LABEL_MAPPING:
                lbls.append(LABEL_MAPPING[lbl_str])
            else:
                raise ValueError(f"Unknown label: {lbl_str}. Please check the LABEL_MAPPING.")

        return torch.stack(imgs), torch.tensor(lbls) # This now receives integers

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
    raise FileNotFoundError(f"Could not find dataset at '{LOCAL_DATASET_PATH}'. Please ensure your script and the '{LOCAL_DATASET_PATH}' folder are in the same directory.")

print(f"Loading local dataset from: {LOCAL_DATASET_PATH}...")
dataset_dict = load_from_disk(LOCAL_DATASET_PATH)

dataset = dataset_dict["train"]

print(f"Loaded {len(dataset)} samples.")

# 4. Split and extract features (rest of the script remains valid)
# ... (Splitting logic is omitted for brevity, but it is the same as the previous response)

# 5. Save HDF5 files (omitted for brevity)
# ...

# --- DRIFT SIMULATION (Experiment 8: Sudden Drift) ---
# ... (The rest of the script for drift simulation remains the same)
