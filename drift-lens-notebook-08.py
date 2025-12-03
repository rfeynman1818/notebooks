# ================================================================
# 0. Imports & global config (run once near the top of your notebook)
# ================================================================
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# assumes DATA_ROOT already points to the FireRisk dataset root
# DATA_ROOT = "/datasets/FireRisk"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32      # use same as training
num_workers = 4      # tune for your machine

# directory where train_model already saves checkpoints
MODEL_SAVE_DIR = "./saved_models"

# directory where we will save embeddings for DriftLens
EMB_SAVE_BASE_DIR = "/home/path-to/drift-baseline-data/saved_embeddings"
TAG_BLUR = "gaussian-blur-all-classes"

# ================================================================
# 1. Full-class datasets & loaders (no blur)
#    - Train labels include *all* FireRisk classes, including "Water"
# ================================================================
train_ds_full = FireRiskDataset(
    DATA_ROOT,
    split="train",
    transform=img_transform,     # standard transform, no blur
    classes_subset=None,         # all classes
)

val_ds_full = FireRiskDataset(
    DATA_ROOT,
    split="val",
    transform=img_transform,     # standard transform, no blur
    classes_subset=None,         # all classes
)

print("Full-class train size:", len(train_ds_full))
print("Full-class val size:  ", len(val_ds_full))
print("Classes:", train_ds_full.selected_classes)

train_loader_full = DataLoader(
    train_ds_full,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader_full = DataLoader(
    val_ds_full,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# ================================================================
# 2. Train full-class model (run ONCE; reuse checkpoint afterward)
#    - This is analogous to STL-10 use case 8 training.
# ================================================================
num_classes_full = len(train_ds_full.selected_classes)

# create a fresh ViT model for the full-class experiment
model_full = make_vit_model(num_classes=num_classes_full).to(device)

epochs = 10  # or whatever you used before

_ = train_model(
    model_full,
    device,
    train_loader_full,
    val_loader_full,
    train_sampler=None,
    val_sampler=None,
    epochs=epochs,
)

# train_model will have saved the best checkpoints into MODEL_SAVE_DIR.
# Pick the best path from its logs and set it below.

# ================================================================
# 3. Load best full-class checkpoint
#    - Fill in FULL_MODEL_CKPT with the filename printed by train_model.
# ================================================================
FULL_MODEL_CKPT = os.path.join(
    MODEL_SAVE_DIR,
    "epoch_10_acc_0.90xx.pt"     # <-- REPLACE with your actual best file
)

print("Loading full-class model from:", FULL_MODEL_CKPT)

model_full = make_vit_model(num_classes=num_classes_full)
model_full.load_state_dict(torch.load(FULL_MODEL_CKPT, map_location=device))
model_full.to(device).eval()

# ================================================================
# 4. Define Gaussian-blur transform & blurred dataset/loader
#    - Drift stream: same labels, but images are blurred.
# ================================================================
img_transform_blur = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=9, sigma=2.0),  # adjust strength if desired
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# baseline / threshold loaders (no blur) reuse train_ds_full / val_ds_full
eval_train_loader_full = DataLoader(
    train_ds_full,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

eval_test_loader_full = DataLoader(
    val_ds_full,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# blurred drift stream: same split as val_ds_full but with blur transform
blur_ds_full = FireRiskDataset(
    DATA_ROOT,
    split="val",
    transform=img_transform_blur,
    classes_subset=None,   # all classes
)

eval_stream_loader_blur = DataLoader(
    blur_ds_full,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

print("Blurred stream size:", len(blur_ds_full))

# ================================================================
# 5. Compute embeddings & predictions for:
#    - baseline train
#    - threshold test (no blur)
#    - blurred drift stream
#    Uses your existing get_embeddings_and_preds and
#    save_embeddings_and_predictions utilities.
# ================================================================
# baseline / train
train_embeddings_full, train_predictions_full, train_labels_full = \
    get_embeddings_and_preds(model_full, eval_train_loader_full)

# threshold / test (no blur)
test_embeddings_full, test_predictions_full, test_labels_full = \
    get_embeddings_and_preds(model_full, eval_test_loader_full)

# drift stream / blurred
blur_embeddings, blur_predictions, blur_labels = \
    get_embeddings_and_preds(model_full, eval_stream_loader_blur)

# paths for DriftLens-style .npz files
emb_base = os.path.join(EMB_SAVE_BASE_DIR, TAG_BLUR)
train_path  = os.path.join(emb_base, "train_embeddings_and_predictions.npz")
test_path   = os.path.join(emb_base, "test_embeddings_and_predictions.npz")
stream_path = os.path.join(emb_base, "stream_embeddings_and_predictions.npz")

# save them
save_embeddings_and_predictions(train_embeddings_full, train_labels_full, train_predictions_full, train_path)
save_embeddings_and_predictions(test_embeddings_full,  test_labels_full,  test_predictions_full,  test_path)
save_embeddings_and_predictions(blur_embeddings,       blur_labels,       blur_predictions,       stream_path)

# label list for DriftLens
training_label_list_full = train_ds_full.selected_classes
print("Label list for DriftLens:", training_label_list_full)

# ================================================================
# 6. (Optional) Skeleton for plugging into DriftLens
#    - This mirrors the code in the drift-lens repo; fill in parameters as needed.
# ================================================================
# from driftlens.driftlens import DriftLens

# dl = DriftLens()

# # load back the .npz if running in a separate script
# train_npz = np.load(train_path)
# test_npz  = np.load(test_path)
# stream_npz = np.load(stream_path)

# E_train = train_npz["embeddings"]
# Y_train_pred = train_npz["preds"]
# E_test = test_npz["embeddings"]
# Y_test_pred = test_npz["preds"]
# E_stream = stream_npz["embeddings"]
# Y_stream_pred = stream_npz["preds"]

# # baseline
# baseline = dl.estimate_baseline(
#     E=E_train,
#     Y=Y_train_pred,
#     label_list=training_label_list_full,
#     batch_n_pc=150,
#     per_label_n_pc=75,
# )

# # threshold estimation
# per_batch_sorted, per_label_sorted = dl.random_sampling_threshold_estimation(
#     label_list=training_label_list_full,
#     E=E_test,
#     Y=Y_test_pred,
#     batch_n_pc=150,
#     per_label_n_pc=75,
#     window_size=1000,
#     n_samples=1000,
#     flag_shuffle=True,
#     flag_replacement=True,
# )

# # online phase: slide windows over E_stream / Y_stream_pred and call
# # dl.compute_window_distribution_distances(...) compared against thresholds.

