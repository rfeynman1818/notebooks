from collections import Counter
from torch.utils.data import DataLoader

# --- get raw labels ---
label_ids = train_ds_full.hf_ds["label"]
counts = Counter(label_ids)

print("Raw label values in train split:", list(counts.keys()))
print("Counts:", counts)

# --- detect label type: string or int ---
if isinstance(label_ids[0], str):
    # labels are class names already
    class_counts = counts
    all_classes = list(train_ds_full.classes)
else:
    # labels are numeric IDs
    id2name = {i: name for i, name in enumerate(train_ds_full.classes)}
    class_counts = {id2name[i]: c for i, c in counts.items()}
    all_classes = list(train_ds_full.classes)

print("\nTrain class counts:", class_counts)

total_train = len(train_ds_full)

# --- choose NC class ---
preferred_nc = "Water"
if preferred_nc in class_counts:
    nc_name = preferred_nc
else:
    # fallback: pick the least frequent class
    nc_name = min(class_counts.items(), key=lambda x: x[1])[0]

print("Chosen NC class:", nc_name)

classes_train_nc = [c for c in all_classes if c != nc_name]

print("Classes kept for training:", classes_train_nc)

non_nc_samples = sum(class_counts[c] for c in classes_train_nc)
print("Non-NC sample count:", non_nc_samples)

if non_nc_samples == 0:
    raise ValueError(
        f"Cannot remove NC class '{nc_name}', no samples remain. "
        f"Class counts: {class_counts}"
    )

# --- build NC datasets ---
train_ds_nc = FireRiskDataset(
    DATA_ROOT,
    split="train",
    transform=img_transform,
    classes_subset=classes_train_nc,
)

val_ds_nc = FireRiskDataset(
    DATA_ROOT,
    split="val",
    transform=img_transform,
    classes_subset=classes_train_nc,
)

test_ds_nc = val_ds_nc

print("Filtered train size:", len(train_ds_nc))
print("Filtered val size:", len(val_ds_nc))

if len(train_ds_nc) == 0:
    raise ValueError("Dataset filtering produced an empty train set — investigate label format.")

# --- DataLoaders ---
batch_size = 32
num_workers = 4

train_loader_nc = DataLoader(
    train_ds_nc,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader_nc = DataLoader(
    val_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_loader_nc = DataLoader(
    test_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

print("DataLoaders created successfully.")
