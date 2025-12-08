#!/usr/bin/env python3
"""
Alternative FireRisk data preparation script with flexible arrow file loading.
This version provides multiple loading strategies for different arrow file structures.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import Dataset, load_dataset, load_from_disk
import h5py
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
import glob
import pyarrow.dataset as ds
import pyarrow as pa
import json

# --- CONFIGURATION ---
# Update this path to match your FireRisk location
FIRERISK_PATH = "/datasets/FireRisk"

# Alternative paths to try (modify as needed)
ALTERNATIVE_PATHS = [
    "./FireRisk",
    "../FireRisk",
    "~/datasets/FireRisk",
    "/data/FireRisk"
]

# Data split configuration
N_BASELINE = 10000 
N_THRESHOLD = 10000 

# Label mapping
LABEL_MAPPING = {
    'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4,
    'Non-burnable': 5, 'Water': 6
}

def find_firerisk_path():
    """Try to locate the FireRisk dataset."""
    paths_to_try = [FIRERISK_PATH] + ALTERNATIVE_PATHS
    
    for path in paths_to_try:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            # Check if it contains expected subdirectories
            if any(os.path.exists(os.path.join(expanded_path, subdir)) 
                   for subdir in ['train$', 'val$', 'val', 'train']):
                print(f"✅ Found FireRisk dataset at: {expanded_path}")
                return expanded_path
    
    # If not found, list what we can see
    print("❌ Could not find FireRisk dataset automatically.")
    print("\nSearching for arrow files in current directory structure...")
    
    # Search for arrow files
    arrow_files = glob.glob("**/*.arrow", recursive=True)[:10]  # Show first 10
    if arrow_files:
        print(f"Found arrow files at:")
        for f in arrow_files:
            print(f"  - {f}")
        
        # Try to infer the base path
        if arrow_files:
            base_path = os.path.dirname(os.path.dirname(arrow_files[0]))
            print(f"\nInferred base path: {base_path}")
            return base_path
    
    return None

def load_arrow_dataset_method1(base_path):
    """Method 1: Using datasets.load_from_disk (for Hugging Face format)"""
    print("  Method 1: Trying load_from_disk...")
    datasets = {}
    
    for split_name in ['train$', 'train', 'val$', 'val', 'test$', 'test']:
        split_path = os.path.join(base_path, split_name)
        if os.path.exists(split_path):
            try:
                dataset = load_from_disk(split_path)
                datasets[split_name] = dataset
                print(f"    ✅ Loaded {split_name}: {len(dataset)} samples")
            except Exception as e:
                print(f"    ⚠️  Could not load {split_name}: {str(e)[:100]}")
    
    return datasets

def load_arrow_dataset_method2(base_path):
    """Method 2: Using load_dataset with arrow files"""
    print("  Method 2: Trying load_dataset with arrow files...")
    datasets = {}
    
    for split_name in ['train$', 'train', 'val$', 'val', 'test$', 'test']:
        split_path = os.path.join(base_path, split_name)
        if os.path.exists(split_path):
            arrow_files = glob.glob(os.path.join(split_path, "*.arrow"))
            if arrow_files:
                try:
                    dataset = load_dataset("arrow", data_files=arrow_files, split="train")
                    datasets[split_name] = dataset
                    print(f"    ✅ Loaded {split_name}: {len(dataset)} samples")
                except Exception as e:
                    print(f"    ⚠️  Could not load {split_name}: {str(e)[:100]}")
    
    return datasets

def load_arrow_dataset_method3(base_path):
    """Method 3: Using PyArrow dataset API directly"""
    print("  Method 3: Trying PyArrow dataset API...")
    datasets = {}
    
    for split_name in ['train$', 'train', 'val$', 'val', 'test$', 'test']:
        split_path = os.path.join(base_path, split_name)
        if os.path.exists(split_path):
            try:
                # Load as PyArrow dataset
                arrow_dataset = ds.dataset(split_path, format='arrow')
                # Convert to pandas then to Hugging Face Dataset
                df = arrow_dataset.to_table().to_pandas()
                dataset = Dataset.from_pandas(df)
                datasets[split_name] = dataset
                print(f"    ✅ Loaded {split_name}: {len(dataset)} samples")
            except Exception as e:
                print(f"    ⚠️  Could not load {split_name}: {str(e)[:100]}")
    
    return datasets

def combine_datasets(datasets_dict):
    """Combine multiple dataset splits into one."""
    if not datasets_dict:
        return None
    
    all_datasets = list(datasets_dict.values())
    
    if len(all_datasets) == 1:
        return all_datasets[0]
    
    print(f"\n  Combining {len(all_datasets)} splits...")
    
    # Try to concatenate
    try:
        from datasets import concatenate_datasets
        combined = concatenate_datasets(all_datasets)
        print(f"  ✅ Combined into single dataset with {len(combined)} samples")
        return combined
    except Exception as e:
        print(f"  ❌ Could not concatenate: {e}")
        # Return the largest one
        largest = max(all_datasets, key=len)
        print(f"  Using largest split with {len(largest)} samples")
        return largest

def load_firerisk_dataset():
    """Main function to load FireRisk dataset with multiple fallback methods."""
    print("\n🔍 Attempting to load FireRisk dataset...")
    
    # Find the dataset path
    base_path = find_firerisk_path()
    
    if base_path is None:
        print("\n❌ ERROR: Could not find FireRisk dataset!")
        print("\nPlease update FIRERISK_PATH in the script to point to your dataset location.")
        print("Expected structure:")
        print("  FireRisk/")
        print("    ├── train$/  (or train/)")
        print("    │   ├── data-00000-of-00024.arrow")
        print("    │   ├── data-00001-of-00024.arrow")
        print("    │   └── ...")
        print("    └── val$/    (or val/)")
        print("        ├── data-00000-of-00008.arrow")
        print("        └── ...")
        return None
    
    # Try different loading methods
    print(f"\n📂 Loading from: {base_path}")
    
    # Method 1
    datasets = load_arrow_dataset_method1(base_path)
    if datasets:
        return combine_datasets(datasets)
    
    # Method 2
    datasets = load_arrow_dataset_method2(base_path)
    if datasets:
        return combine_datasets(datasets)
    
    # Method 3
    datasets = load_arrow_dataset_method3(base_path)
    if datasets:
        return combine_datasets(datasets)
    
    print("\n❌ All loading methods failed!")
    return None

def save_hdf5(filepath, E_data, Y_data):
    """Save embeddings and labels to HDF5."""
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset('embeddings', data=E_data) 
        hf.create_dataset('labels', data=Y_data)
    
    print(f"\n💾 Saved to {filepath}:")
    print(f"   Samples: {len(Y_data)}")
    print(f"   Embedding dim: {E_data.shape[1]}")
    
    # Show label distribution
    label_counts = Counter(Y_data)
    print(f"   Labels present: {sorted(label_counts.keys())}")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = (count / len(Y_data)) * 100
        label_name = [k for k, v in LABEL_MAPPING.items() if v == label][0]
        print(f"     {label} ({label_name:12s}): {count:5d} ({pct:5.1f}%)")

def extract_features(indices, dataset, model, device, preprocess, batch_size=32, desc="Processing"):
    """Extract ResNet50 features for given indices."""
    embeddings = []
    labels = []
    
    for i in tqdm(range(0, len(indices), batch_size), desc=desc):
        batch_idx = indices[i:i+batch_size]
        batch_imgs = []
        batch_labels = []
        
        for idx in batch_idx:
            try:
                item = dataset[int(idx)]
                
                # Handle image
                img_data = item['image']
                if isinstance(img_data, str):
                    img = Image.open(img_data).convert("RGB")
                elif isinstance(img_data, Image.Image):
                    img = img_data.convert("RGB")
                else:
                    img = Image.fromarray(np.array(img_data)).convert("RGB")
                
                batch_imgs.append(preprocess(img))
                
                # Handle label
                lbl_str = item.get('label', 'Low')  # Default to 'Low' if missing
                batch_labels.append(LABEL_MAPPING.get(lbl_str, 1))
                
            except Exception as e:
                print(f"    Error processing index {idx}: {e}")
                continue
        
        if batch_imgs:
            imgs_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                emb = model(imgs_tensor)
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch_labels)
    
    return np.vstack(embeddings) if embeddings else np.array([]), np.array(labels)

# --- MAIN EXECUTION ---
def main():
    print("="*70)
    print(" FireRisk Arrow Dataset Preparation")
    print("="*70)
    
    # Load dataset
    dataset = load_firerisk_dataset()
    if dataset is None:
        print("\n❌ Failed to load dataset. Exiting.")
        return 1
    
    print(f"\n✅ Successfully loaded dataset with {len(dataset)} samples")
    
    # Check dataset structure
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Features: {list(sample.keys())}")
    
    # Setup model
    print("\n🧠 Setting up ResNet50 feature extractor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract labels for stratification
    print("\n📊 Analyzing dataset labels...")
    all_indices = []
    all_labels = []
    
    for idx in tqdm(range(len(dataset)), desc="Scanning"):
        item = dataset[idx]
        lbl_str = item.get('label', 'Unknown')
        if lbl_str in LABEL_MAPPING:
            all_indices.append(idx)
            all_labels.append(LABEL_MAPPING[lbl_str])
    
    all_indices = np.array(all_indices)
    all_labels = np.array(all_labels)
    
    # Show distribution
    label_counts = Counter(all_labels)
    print(f"\n   Total valid samples: {len(all_indices)}")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = (count / len(all_labels)) * 100
        label_name = [k for k, v in LABEL_MAPPING.items() if v == label][0]
        print(f"   {label} ({label_name:12s}): {count:6d} ({pct:5.1f}%)")
    
    # Stratified splitting
    print("\n✂️  Creating stratified splits...")
    
    # Adjust sizes if needed
    n_baseline = min(N_BASELINE, len(all_indices) // 3)
    n_threshold = min(N_THRESHOLD, len(all_indices) // 3)
    
    try:
        # Split 1: Baseline vs Rest
        baseline_idx, rest_idx, baseline_lbl, rest_lbl = train_test_split(
            all_indices, all_labels, train_size=n_baseline, 
            stratify=all_labels, random_state=42
        )
        
        # Split 2: Threshold vs Stream
        threshold_idx, stream_idx, threshold_lbl, stream_lbl = train_test_split(
            rest_idx, rest_lbl, train_size=n_threshold,
            stratify=rest_lbl, random_state=42
        )
        
        print(f"   Baseline:  {len(baseline_idx)} samples")
        print(f"   Threshold: {len(threshold_idx)} samples")
        print(f"   Stream:    {len(stream_idx)} samples")
        
    except ValueError as e:
        print(f"   ⚠️  Stratification failed: {e}")
        print(f"   Using random splitting instead...")
        
        np.random.shuffle(all_indices)
        baseline_idx = all_indices[:n_baseline]
        threshold_idx = all_indices[n_baseline:n_baseline + n_threshold]
        stream_idx = all_indices[n_baseline + n_threshold:]
    
    # Extract features
    print("\n⚙️  Extracting features...")
    
    E_base, Y_base = extract_features(
        baseline_idx, dataset, model, device, preprocess, desc="Baseline"
    )
    
    E_thresh, Y_thresh = extract_features(
        threshold_idx, dataset, model, device, preprocess, desc="Threshold"
    )
    
    E_stream, Y_stream = extract_features(
        stream_idx, dataset, model, device, preprocess, desc="Stream"
    )
    
    # Create drift in stream
    print("\n🌊 Injecting drift into stream...")
    DRIFT_LABEL = 4  # Very High
    WINDOW_SIZE = 500
    DRIFT_START = 10 * WINDOW_SIZE
    
    drift_idx = np.where(Y_stream == DRIFT_LABEL)[0]
    if len(drift_idx) == 0:
        counts = Counter(Y_stream)
        DRIFT_LABEL = counts.most_common(1)[0][0]
        drift_idx = np.where(Y_stream == DRIFT_LABEL)[0]
        print(f"   Using label {DRIFT_LABEL} for drift ({len(drift_idx)} samples)")
    
    # Inject drift
    if DRIFT_START < len(E_stream):
        E_drift_pool = E_stream[drift_idx]
        Y_drift_pool = Y_stream[drift_idx]
        
        E_normal = E_stream[:DRIFT_START]
        Y_normal = Y_stream[:DRIFT_START]
        
        remaining = len(E_stream) - DRIFT_START
        reps = (remaining // len(drift_idx)) + 1
        
        E_drift = np.tile(E_drift_pool, (reps, 1))[:remaining]
        Y_drift = np.tile(Y_drift_pool, reps)[:remaining]
        
        E_final = np.vstack([E_normal, E_drift])
        Y_final = np.concatenate([Y_normal, Y_drift])
    else:
        E_final = E_stream
        Y_final = Y_stream
        print(f"   ⚠️  Stream too short for drift at index {DRIFT_START}")
    
    # Save files
    save_hdf5("baseline.hdf5", E_base, Y_base)
    save_hdf5("threshold.hdf5", E_thresh, Y_thresh)
    save_hdf5("datastream.hdf5", E_final, Y_final)
    
    # Final check
    common = set(Y_base) & set(Y_thresh)
    print(f"\n✅ Common labels: {sorted(common)}")
    
    if len(common) == 0:
        print("❌ WARNING: No common labels! DriftLens will fail!")
        return 1
    
    print("\n✅ Success! Ready for drift detection.")
    print("   Next: python run_experiment_fixed.py")
    return 0

if __name__ == "__main__":
    exit(main())
