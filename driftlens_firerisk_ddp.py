#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DriftLens + FireRisk + Vision Transformer with DDP support.

Run either as:
    python driftlens_firerisk_ddp.py
or (multi-GPU, recommended):
    torchrun --nproc_per_node=4 driftlens_firerisk_ddp.py

It runs two experiments:

1. New-class drift: 'Water' unseen in training, appears later in the stream.
2. Gaussian blur drift: all classes present, drift is blur on all classes.
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from torchvision import transforms

import timm
import matplotlib.pyplot as plt

from driftlens.driftlens import DriftLens


# ============================================================
# DDP helpers
# ============================================================

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """Initialize distributed process group if run under torchrun.

    Returns a torch.device to use for this process.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        if is_main_process():
            print(f"[DDP] World size: {get_world_size()}")
        print(f"[Rank {get_rank()}] Using device {device}")
    else:
        # single-process / notebook mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Single process] Using device", device)
    return device


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def _reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


# ============================================================
# Dataset
# ============================================================

class FireRiskDataset(Dataset):
    """Folder-based FireRisk dataset.

    Expected layout:

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

    Args:
        root: dataset root directory.
        split: 'train' or 'val'.
        transform: torchvision transform.
        classes_subset: optional iterable of class names to restrict to.
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
            # case-insensitive match to be robust
            lower_map = {c.lower(): c for c in all_classes}
            selected = []
            for c in classes_subset:
                key = c.lower()
                assert key in lower_map, f"Unknown class in classes_subset: {c}"
                selected.append(lower_map[key])
            selected_classes = sorted(selected)
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


# ============================================================
# Model + training helpers
# ============================================================

def create_vit(num_classes: int, device: torch.device) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Unexpected ViT model structure.")

    model.to(device)

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[device.index], output_device=device.index)
        if is_main_process():
            print("Wrapped ViT in DistributedDataParallel")
    return model


def run_epoch(model, loader, device, optimizer=None, criterion=None,
              epoch: int = 0, sampler=None):
    train = optimizer is not None
    if train:
        model.train()
        if sampler is not None and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
    else:
        model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            core = model.module if isinstance(model, DDP) else model
            feats = core.forward_features(x)
            if feats.ndim == 3:
                feats = feats[:, 0]  # CLS token
            logits = core.head(feats)

            loss = criterion(logits, y) if criterion is not None else None

            if train:
                loss.backward()
                optimizer.step()

            if loss is not None:
                total_loss += loss.detach() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum()
            total += x.size(0)

    total_loss = _reduce_tensor(total_loss)
    total_correct = _reduce_tensor(total_correct)
    total = _reduce_tensor(total)

    avg_loss = (total_loss / total).item() if total.item() > 0 else 0.0
    acc = (total_correct / total).item() if total.item() > 0 else 0.0
    return avg_loss, acc


def train_model(model, device, train_loader, val_loader,
                train_sampler=None, val_sampler=None,
                epochs: int = 10, lr: float = 3e-4, wd: float = 0.05):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, optimizer, criterion,
            epoch=epoch, sampler=train_sampler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, device, optimizer=None, criterion=criterion,
            epoch=epoch, sampler=val_sampler
        )
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        if is_main_process():
            print(f"Epoch {epoch:02d} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if is_main_process():
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None and is_main_process():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


@torch.no_grad()
def get_embeddings_and_preds(model, device, loader):
    """Collect embeddings & predictions.

    Use a non-distributed loader (no DistributedSampler).
    In DDP, call only on rank 0.
    """
    model.eval()
    core = model.module if isinstance(model, DDP) else model

    all_E, all_Y_hat, all_Y_true = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats = core.forward_features(x)
        if feats.ndim == 3:
            feats = feats[:, 0]
        logits = core.head(feats)
        preds = logits.argmax(dim=1)

        all_E.append(feats.cpu())
        all_Y_hat.append(preds.cpu())
        all_Y_true.append(y)

    E = torch.cat(all_E, dim=0).numpy()
    Y_hat = torch.cat(all_Y_hat, dim=0).numpy()
    Y_true = torch.cat(all_Y_true, dim=0).numpy()
    return E, Y_hat, Y_true


# ============================================================
# DriftLens helpers
# ============================================================

def fit_driftlens_baseline(E_train, Y_pred_train,
                           E_thr, Y_pred_thr,
                           window_size: int,
                           batch_n_pc: int = 150,
                           per_label_n_pc: int = 75,
                           n_samples: int = 10000):
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


def compute_window_distances(dl, E_stream, Yp_stream, window_size: int):
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


# ============================================================
# Experiment A: New-class drift (Water unseen in training)
# ============================================================

def run_experiment_new_class(device, data_root, batch_size=64, num_workers=4,
                             window_size=1000, epochs=10):
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # full datasets just to get class list
    train_ds_full = FireRiskDataset(data_root, split="train",
                                    transform=img_transform, classes_subset=None)
    val_ds_full   = FireRiskDataset(data_root, split="val",
                                    transform=img_transform, classes_subset=None)

    all_classes = train_ds_full.selected_classes
    if is_main_process():
        print("All classes:", all_classes)

    classes_train_nc = [c for c in all_classes if c != "Water"]
    classes_new_nc = ["Water"]
    if is_main_process():
        print("Train classes (no new class):", classes_train_nc)
        print("New-class only in drift   :", classes_new_nc)

    train_ds_nc = FireRiskDataset(data_root, split="train",
                                  transform=img_transform,
                                  classes_subset=classes_train_nc)
    val_ds_nc   = FireRiskDataset(data_root, split="val",
                                  transform=img_transform,
                                  classes_subset=classes_train_nc)
    test_ds_nc  = val_ds_nc  # reuse val as test

    # DDP-aware samplers
    if is_dist_avail_and_initialized():
        train_sampler_nc = DistributedSampler(train_ds_nc, shuffle=True)
        val_sampler_nc   = DistributedSampler(val_ds_nc,   shuffle=False)
        shuffle_train = False
    else:
        train_sampler_nc = None
        val_sampler_nc   = None
        shuffle_train = True

    train_loader_nc = DataLoader(
        train_ds_nc,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler_nc,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader_nc = DataLoader(
        val_ds_nc,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler_nc,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader_nc = DataLoader(
        test_ds_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if is_main_process():
        print("New-class train size:", len(train_ds_nc),
              "val/test size:", len(val_ds_nc))
        print("Selected classes:", train_ds_nc.selected_classes)

    num_classes_nc = len(train_ds_nc.selected_classes)
    model_nc = create_vit(num_classes_nc, device)

    _ = train_model(model_nc, device, train_loader_nc, val_loader_nc,
                    train_sampler=train_sampler_nc, val_sampler=val_sampler_nc,
                    epochs=epochs)

    # Only main process runs DriftLens + plotting
    if not is_main_process():
        return

    # For embeddings, use non-distributed loaders on rank 0
    eval_train_loader_nc = DataLoader(
        train_ds_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_test_loader_nc = DataLoader(
        test_ds_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    E_train_nc, Yp_train_nc, _ = get_embeddings_and_preds(model_nc, device, eval_train_loader_nc)
    E_thr_nc,   Yp_thr_nc,   _ = get_embeddings_and_preds(model_nc, device, eval_test_loader_nc)

    dl_nc, baseline_nc, per_batch_sorted_nc, per_label_sorted_nc = fit_driftlens_baseline(
        E_train_nc, Yp_train_nc, E_thr_nc, Yp_thr_nc,
        window_size=window_size, batch_n_pc=150, per_label_n_pc=75, n_samples=10000,
    )

    print("Baseline fitted. Labels:", sorted(np.unique(Yp_train_nc)))

    # ----- build stream with Water appearing -----
    test_ds_all = FireRiskDataset(data_root, split="val",
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

    target_windows_pre = 10
    target_windows_post = 10
    target_pre_len = target_windows_pre * window_size
    target_post_len = target_windows_post * window_size

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
        for c in classes_train_nc + classes_new_nc:
            if indices_by_class[c]:
                post_indices.append(indices_by_class[c].pop())
                exhausted = False
                if len(post_indices) >= target_post_len:
                    break
        if exhausted:
            break

    stream_indices_nc = pre_indices + post_indices
    stream_ds_nc = Subset(test_ds_all, stream_indices_nc)
    stream_loader_nc = DataLoader(
        stream_ds_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Stream size (samples):", len(stream_ds_nc))

    E_stream_nc, Yp_stream_nc, _ = get_embeddings_and_preds(model_nc, device, stream_loader_nc)
    distances_nc = compute_window_distances(dl_nc, E_stream_nc, Yp_stream_nc, window_size=window_size)
    print("Number of windows:", len(distances_nc))

    approx_drift_window = (len(stream_indices_nc) // 2) // window_size

    plt.figure(figsize=(10, 4))
    plt.plot(distances_nc, marker="o")
    plt.axvline(x=approx_drift_window - 0.5, linestyle="--", label="drift start (approx)")
    plt.xlabel("Window index")
    plt.ylabel("DriftLens distance")
    plt.title("New-class drift (Water) – FireRisk + ViT")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Experiment B: Gaussian blur drift
# ============================================================

def run_experiment_blur(device, data_root, batch_size=64, num_workers=4,
                        window_size=1000, epochs=10):
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds_full = FireRiskDataset(data_root, split="train",
                                    transform=img_transform, classes_subset=None)
    val_ds_full   = FireRiskDataset(data_root, split="val",
                                    transform=img_transform, classes_subset=None)
    test_ds_full  = val_ds_full

    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_ds_full, shuffle=True)
        val_sampler   = DistributedSampler(val_ds_full,   shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True

    train_loader_full = DataLoader(
        train_ds_full,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader_full = DataLoader(
        test_ds_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if is_main_process():
        print("Full-train size:", len(train_ds_full),
              "val/test size:", len(val_ds_full))
        print("Discovered classes:", train_ds_full.selected_classes)

    num_classes_full = len(train_ds_full.selected_classes)
    model_full = create_vit(num_classes_full, device)

    _ = train_model(model_full, device, train_loader_full, val_loader_full,
                    train_sampler=train_sampler, val_sampler=val_sampler,
                    epochs=epochs)

    if not is_main_process():
        return

    # non-distributed loaders for embeddings on rank 0
    eval_train_loader_full = DataLoader(
        train_ds_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_test_loader_full = DataLoader(
        test_ds_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    E_train_full, Yp_train_full, _ = get_embeddings_and_preds(model_full, device, eval_train_loader_full)
    E_thr_full,   Yp_thr_full,   _ = get_embeddings_and_preds(model_full, device, eval_test_loader_full)

    dl_full, baseline_full, per_batch_sorted_full, per_label_sorted_full = fit_driftlens_baseline(
        E_train_full, Yp_train_full, E_thr_full, Yp_thr_full,
        window_size=window_size, batch_n_pc=150, per_label_n_pc=75, n_samples=10000,
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

    base_ds_clean = FireRiskDataset(data_root, split="val",
                                    transform=img_transform, classes_subset=None)
    base_ds_blur  = FireRiskDataset(data_root, split="val",
                                    transform=blur_transform, classes_subset=None)

    assert base_ds_clean.paths == base_ds_blur.paths
    assert base_ds_clean.labels == base_ds_blur.labels

    clean_loader = DataLoader(
        base_ds_clean,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    blur_loader = DataLoader(
        base_ds_blur,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    E_clean, Yp_clean, _ = get_embeddings_and_preds(model_full, device, clean_loader)
    E_blur,  Yp_blur,  _ = get_embeddings_and_preds(model_full, device, blur_loader)

    n_clean = (len(E_clean) // window_size) * window_size
    n_blur  = (len(E_blur)  // window_size) * window_size
    n_clean = min(n_clean, 20 * window_size)
    n_blur  = min(n_blur, 20 * window_size)

    E_stream_blur = np.concatenate([E_clean[:n_clean], E_blur[:n_blur]], axis=0)
    Yp_stream_blur = np.concatenate([Yp_clean[:n_clean], Yp_blur[:n_blur]], axis=0)

    print("Stream sizes (clean, blur):", n_clean, n_blur)

    distances_blur = compute_window_distances(dl_full, E_stream_blur, Yp_stream_blur,
                                              window_size=window_size)
    print("Number of windows:", len(distances_blur))

    clean_windows = n_clean // window_size

    plt.figure(figsize=(10, 4))
    plt.plot(distances_blur, marker="o")
    plt.axvline(x=clean_windows - 0.5, linestyle="--", label="blur start")
    plt.xlabel("Window index")
    plt.ylabel("DriftLens distance")
    plt.title("Gaussian blur drift – FireRisk + ViT")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    DATA_ROOT = os.environ.get("FIRERISK_ROOT", "/home/joyvan/datasets/FireRisk")
    assert os.path.isdir(DATA_ROOT), f"FireRisk root not found: {DATA_ROOT}"

    batch_size = 64
    num_workers = 4
    window_size = 1000
    epochs = 10

    device = setup_distributed()

    try:
        if is_main_process():
            print("=== Experiment A: New-class drift (Water) ===")
        run_experiment_new_class(device, DATA_ROOT,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 window_size=window_size,
                                 epochs=epochs)

        if is_main_process():
            print("=== Experiment B: Gaussian blur drift ===")
        run_experiment_blur(device, DATA_ROOT,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            window_size=window_size,
                            epochs=epochs)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
