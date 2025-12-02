from collections import Counter
from torch.utils.data import DataLoader

# --- inspect class distribution in the train split ---
label_ids = train_ds_full.hf_ds["label"]  # list of ints
counts = Counter(label_ids)
id2name = {i: name for i, name in enumerate(train_ds_full.classes)}

class_counts = {id2name[i]: c for i, c in counts.items()}
print("Train class counts:", class_counts)

total_train = len(train_ds_full)

# --- choose NC class ---
# Prefer 'Water' if it exists and is not the only class
if "Water" in train_ds_full.classes:
    nc_name = "Water"
else:
    # fallback: pick the class with the smallest count as NC
    nc_id, _ = min(counts.items(), key=lambda kv: kv[1])
    nc_name = id2name[nc_id]

nc_id = train_ds_full.class_to_idx[nc_name]
non_nc_samples = total_train - counts[nc_id]

if non_nc_samples == 0:
    # If removing that class kills the dataset, pick another class as NC
    print(f"Cannot use '{nc_name}' as NC: it is the only class with samples.")
    # choose some other class with samples
    alt_id, _ = min(
        [(i, c) for i, c in counts.items() if i != nc_id],
        key=lambda kv: kv[1]
    )
    nc_name = id2name[alt_id]
    nc_id = alt_id
    non_nc_samples = total_train - counts[nc_id]
    print(f"Using '{nc_name}' as NC instead.")

print("New-class (held-out):", nc_name)

classes_train_nc = [c for c in train_ds_full.classes if c != nc_name]
print("Train classes (kept):", classes_train_nc)
print("Non-NC train samples:", non_nc_samples)

if non_nc_samples == 0:
    raise ValueError(
        "No samples left after NC selection. "
        "Check class_counts and choose a different NC strategy."
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
    raise ValueError(
        "Filtered training dataset is empty even though non_nc_samples > 0. "
        "This would indicate a mismatch between labels and class names."
    )

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
