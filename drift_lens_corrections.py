#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# DriftLens + FireRisk + Vision Transformer (folder-structured)
# ===============================================================
# This script runs two DriftLens experiments on the FireRisk dataset:
# 1) New-class drift: "Water" is unseen in training and appears later.
# 2) Gaussian blur drift: all classes, with blur injected after some time.
#
# Assumes FireRisk is laid out as:
#   DATA_ROOT/
#       train/
#           High/
#           Low/
#           Moderate/
#           Non-burnable/
#           Very_High/
#           Very_Low/
#           Water/
#       val/
#           High/
#           ...
#
# You’ll need:
#   pip install driftlens timm torch torchvision matplotlib scikit-learn
# ===============================================================

import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import timm
import matplotlib.pyplot as plt

from driftlens.driftlens import DriftLens


# ===============================================================
# Configuration
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# CHANGE THIS to your FireRisk root path
DATA_ROOT = "/home/joyvan/datasets/FireRisk"

assert os.path.isdir(DATA_ROOT), f"FireRisk root not found: {DATA_ROOT}"
assert os.path.isdir(os.path.join(DATA_ROOT, "train")), "Expected 'train' subfolder"
assert os.path.isdir(os.path.join(DATA_ROOT, "val")), "Expected 'val' subfolder"

batch_size = 64
num_workers = 4
default_window_size = 1000  # for DriftLens windows
default_epochs = 10         # adjust as needed


# ===============================================================
# Dataset: folder-based FireRisk
# ===============================================================

class FireRiskDataset(Dataset):
    """
    Folder-based FireRisk dataset.

    Expects:
        root/
          train/
            High/
            Low/
            Moderate/
            Non-burnable/
            Very_High/
            Very_Low/
            Water/
          val/
            High/
            ...
    """
    def __init__(self, root, split="train", transform=None, classes_subset=None):
        self.root = root
        self.split = split
        self.transform = transform

        train_root = os.path.join(root, "train")
        assert os.path.isdir(train_root), f"Missing train folder: {train_root}"
        all_classes = sorted(
            d for d in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, d))
        )
        self.all_classes = all_classes

        if classes_subset is not None:
            selected_classes = sorted(classes_subset)
            for c in selected_classes:
                assert c in all_classes, f"Unknown class in classes_subset: {c}"
        else:
            selected_classes = all_classes

        self.selected_classes = selected_classes
        self.class_to_idx = {c: i for i, c in enumerate(selected_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        split_root = os.path.join(root, split)
        assert os.path.isdir(split_root), f"Missing split folder: {split_root}"

        paths, labels = [], []
        exts = ("*.png", "*.jpg", "*.jpeg")
        for cls_name in selected_classes:
            cls_dir = os.path.join(split_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for ext in exts:
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    paths.append(p)
                    labels.append(self.class_to_idx[cls_name])

        assert paths, f"No images found in {split_root} for classes {selected_classes}"
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path = self.paths[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[i]
        return img, label


# ===============================================================
# Transforms and base loaders
# ===============================================================

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds_full = FireRiskDataset(DATA_ROOT, split="train",
                                transform=img_transform, classes_subset=None)
val_ds_full   = FireRiskDataset(DATA_ROOT, split="val",
                                transform=img_transform, classes_subset=None)
# use val as "test"
test_ds_full  = val_ds_full

train_loader_full = DataLoader(train_ds_full, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
val_loader_full   = DataLoader(val_ds_full, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
test_loader_full  = DataLoader(test_ds_full, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)

print("Full-train size:", len(train_ds_full),
      "val/test size:", len(val_ds_full))
print("Discovered classes:", train_ds_full.selected_classes)


# ===============================================================
# Model + training helpers
# ===============================================================

def create_vit(num_classes: int) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Unexpected ViT model structure.")
    model.to(device)
    return model


def run_epoch(model, loader, optimizer=None, criterion=None):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad()
            feats = model.forward_features(x)
            logits = model.head(feats)
            loss = criterion(logits, y) if criterion is not None else None

            if train:
                loss.backward()
                optimizer.step()

            if loss is not None:
                total_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = total_correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_model(model, train_loader, val_loader,
                epochs=default_epochs, lr=3e-4, wd=0.05):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, val_loader)
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return history


@torch.no_grad()
def get_embeddings_and_preds(model, loader):
    model.eval()
    all_E, all_Y_hat, all_Y_true = [], [], []
    for x, y in loader:
        x = x.to(device)
        feats = model.forward_features(x)
        logits = model.head(feats)
        preds = logits.argmax(dim=1)
        all_E.append(feats.cpu())
        all_Y_hat.append(preds.cpu())
        all_Y_true.append(y)
    E = torch.cat(all_E, dim=0).numpy()
    Y_hat = torch.cat(all_Y_hat, dim=0).numpy()
    Y_true = torch.cat(all_Y_true, dim=0).numpy()
    return E, Y_hat, Y_true


# ===============================================================
# DriftLens helpers
# ===============================================================

def fit_driftlens_baseline(E_train, Y_pred_train,
                           E_thr, Y_pred_thr,
                           batch_n_pc=150,
                           per_label_n_pc=75,
                           window_size=default_window_size,
                           n_samples=10000):
    label_list = sorted(np.unique(Y_pred_train))
    dl = DriftLens()
    baseline = dl.estimate_baseline(
        E=E_train,
        Y=Y_pred_train,
        label_list=label_list,
        batch_n_pc=batch_n_pc,
        per_label_n_pc=per_label_n_pc,
    )
    per_batch_sorted, per_label_sorted = dl.random_sampling_threshold_estimation(
        label_list=label_list,
        E=E_thr,
        Y=Y_pred_thr,
        batch_n_pc=batch_n_pc,
        per_label_n_pc=per_label_n_pc,
        window_size=window_size,
        n_samples=n_samples,
        flag_shuffle=True,
        flag_replacement=True,
    )
    return dl, baseline, per_batch_sorted, per_label_sorted


def compute_window_distances(dl, E_stream, Yp_stream, window_size):
    n = E_stream.shape[0]
    n_windows = n // window_size
    distances = []
    for w in range(n_windows):
        s = w * window_size
        e = s + window_size
        Ew = E_stream[s:e]
        Ypw = Yp_stream[s:e]
        dist = dl.compute_window_distribution_distances(Ew, Ypw)
        distances.append(dist)
    return np.array(distances)


# ===============================================================
# Experiment A: New-class drift (Water unseen in training)
# ===============================================================

def experiment_new_class():
    print("\n========== Experiment A: New-class drift (Water) ==========")

    all_classes = train_ds_full.selected_classes
    print("All classes:", all_classes)

    # treat Water as the unseen new class
    classes_train_nc = {c for c in all_classes if c != "Water"}
    classes_new_nc = {"Water"}

    print("Train classes (no new class):", classes_train_nc)
    print("New-class only in drift   :", classes_new_nc)

    train_ds_nc = FireRiskDataset(DATA_ROOT, split="train",
                                  transform=img_transform,
                                  classes_subset=classes_train_nc)
    val_ds_nc   = FireRiskDataset(DATA_ROOT, split="val",
                                  transform=img_transform,
                                  classes_subset=classes_train_nc)
    test_ds_nc  = val_ds_nc  # reuse val as test

    train_loader_nc = DataLoader(train_ds_nc, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    val_loader_nc   = DataLoader(val_ds_nc, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    test_loader_nc  = DataLoader(test_ds_nc, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    print("New-class train size:", len(train_ds_nc),
          "val/test size:", len(val_ds_nc))

    num_classes_nc = len(train_ds_nc.selected_classes)
    model_nc = create_vit(num_classes=num_classes_nc)
    _ = train_model(model_nc, train_loader_nc, val_loader_nc)

    E_train_nc, Yp_train_nc, _ = get_embeddings_and_preds(model_nc, train_loader_nc)
    E_thr_nc,   Yp_thr_nc,   _ = get_embeddings_and_preds(model_nc, test_loader_nc)

    dl_nc, baseline_nc, per_batch_sorted_nc, per_label_sorted_nc = fit_driftlens_baseline(
        E_train_nc, Yp_train_nc, E_thr_nc, Yp_thr_nc,
        batch_n_pc=150, per_label_n_pc=75,
        window_size=default_window_size, n_samples=10000,
    )

    print("Baseline fitted. Labels:", sorted(np.unique(Yp_train_nc)))

    # ----- Build stream with Water gradually appearing -----

    test_ds_all = FireRiskDataset(DATA_ROOT, split="val",
                                  transform=img_transform,
                                  classes_subset=None)
    labels_all = test_ds_all.labels
    classes_all = test_ds_all.selected_classes

    indices_by_class = {cls_name: [] for cls_name in classes_all}
    for idx, lab in enumerate(labels_all):
        cls_name = classes_all[lab]
        indices_by_class[cls_name].append(idx)

    for c in indices_by_class:
        random.shuffle(indices_by_class[c])

    window_size_nc = default_window_size
    # target windows; may shrink if data is small
    target_windows_pre = 10
    target_windows_post = 10
    target_pre_len = target_windows_pre * window_size_nc
    target_post_len = target_windows_post * window_size_nc

    pre_indices = []
    while len(pre_indices) < target_pre_len:
        exhausted = True
        for c in classes_train_nc:
            if indices_by_class[c]:
                pre_indices.append(indices_by_class[c].pop())
                exhausted = False
                if len(pre_indices) >= target_pre_len:
                    break
        if exhausted:
            break

    post_indices = []
    while len(post_indices) < target_post_len:
        exhausted = True
        for c in classes_train_nc | classes_new_nc:
            if indices_by_class[c]:
                post_indices.append(indices_by_class[c].pop())
                exhausted = False
                if len(post_indices) >= target_post_len:
                    break
        if exhausted:
            break

    stream_indices_nc = pre_indices + post_indices
    stream_ds_nc = Subset(test_ds_all, stream_indices_nc)
    stream_loader_nc = DataLoader(stream_ds_nc, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    print("Stream size (samples):", len(stream_ds_nc))

    E_stream_nc, Yp_stream_nc, Ytrue_stream_nc = get_embeddings_and_preds(
        model_nc, stream_loader_nc
    )
    distances_nc = compute_window_distances(
        dl_nc, E_stream_nc, Yp_stream_nc, window_size=window_size_nc
    )
    print("Number of windows:", len(distances_nc))

    # approximate drift point at half the stream
    approx_drift_window = (len(stream_indices_nc) // 2) // window_size_nc

    plt.figure(figsize=(10, 4))
    plt.plot(distances_nc, marker="o")
    plt.axvline(x=approx_drift_window - 0.5, linestyle="--", label="drift start (approx)")
    plt.xlabel("Window index")
    plt.ylabel("DriftLens distance")
    plt.title("New-class drift (Water) – FireRisk + ViT")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# Experiment B: Gaussian blur drift (all classes)
# ===============================================================

def experiment_blur():
    print("\n========== Experiment B: Gaussian blur drift ==========")

    num_classes_full = len(train_ds_full.selected_classes)
    model_full = create_vit(num_classes=num_classes_full)
    _ = train_model(model_full, train_loader_full, val_loader_full)

    E_train_full, Yp_train_full, _ = get_embeddings_and_preds(model_full, train_loader_full)
    E_thr_full,   Yp_thr_full,   _ = get_embeddings_and_preds(model_full, test_loader_full)

    dl_full, baseline_full, per_batch_sorted_full, per_label_sorted_full = fit_driftlens_baseline(
        E_train_full, Yp_train_full, E_thr_full, Yp_thr_full,
        batch_n_pc=150, per_label_n_pc=75,
        window_size=default_window_size, n_samples=10000,
    )

    print("Baseline fitted (blur experiment). Labels:",
          sorted(np.unique(Yp_train_full)))

    blur_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.GaussianBlur(kernel_size=11, sigma=3.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    base_ds_clean = FireRiskDataset(DATA_ROOT, split="val",
                                    transform=img_transform, classes_subset=None)
    base_ds_blur  = FireRiskDataset(DATA_ROOT, split="val",
                                    transform=blur_transform, classes_subset=None)

    assert base_ds_clean.paths == base_ds_blur.paths
    assert base_ds_clean.labels == base_ds_blur.labels

    clean_loader = DataLoader(base_ds_clean, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    blur_loader  = DataLoader(base_ds_blur, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    E_clean, Yp_clean, _ = get_embeddings_and_preds(model_full, clean_loader)
    E_blur,  Yp_blur,  _ = get_embeddings_and_preds(model_full, blur_loader)

    window_size_blur = default_window_size
    n_clean = (len(E_clean) // window_size_blur) * window_size_blur
    n_blur  = (len(E_blur) // window_size_blur) * window_size_blur
    n_clean = min(n_clean, 20 * window_size_blur)  # cap #windows if huge
    n_blur  = min(n_blur, 20 * window_size_blur)

    E_stream_blur = np.concatenate([E_clean[:n_clean], E_blur[:n_blur]], axis=0)
    Yp_stream_blur = np.concatenate([Yp_clean[:n_clean], Yp_blur[:n_blur]], axis=0)

    print("Stream sizes (clean, blur):", n_clean, n_blur)

    distances_blur = compute_window_distances(
        dl_full, E_stream_blur, Yp_stream_blur, window_size=window_size_blur
    )
    print("Number of windows:", len(distances_blur))

    clean_windows = n_clean // window_size_blur

    plt.figure(figsize=(10, 4))
    plt.plot(distances_blur, marker="o")
    plt.axvline(x=clean_windows - 0.5, linestyle="--", label="blur start")
    plt.xlabel("Window index")
    plt.ylabel("DriftLens distance")
    plt.title("Gaussian blur drift – FireRisk + ViT")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# Main
# ===============================================================

def main():
    experiment_new_class()
    experiment_blur()


if __name__ == "__main__":
    main()
