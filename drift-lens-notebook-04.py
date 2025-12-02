from datasets import load_from_disk, ClassLabel
from torch.utils.data import Dataset
from torchvision import transforms

DATA_ROOT = "/datasets/FireRisk"   # directory containing dataset_dict.json

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FireRiskDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None, classes_subset=None):
        ds_dict = load_from_disk(data_root)
        self.hf_ds = ds_dict[split]          # full HF dataset for this split
        self.transform = transform

        # Raw labels as they are stored (strings in your case)
        labels = self.hf_ds["label"]

        # Determine global class list (prefer the ClassLabel feature if present)
        label_feat = self.hf_ds.features.get("label", None)
        if isinstance(label_feat, ClassLabel) and label_feat.names is not None:
            all_classes = list(label_feat.names)
        else:
            # preserve order of first appearance
            seen = {}
            for lab in labels:
                if lab not in seen:
                    seen[lab] = None
            all_classes = list(seen.keys())

        self.classes = all_classes

        if classes_subset is None:
            # Use all classes; keep all indices
            self.selected_classes = list(self.classes)
            self.indices = list(range(len(self.hf_ds)))
        else:
            # Use only the requested subset of class names
            self.selected_classes = list(classes_subset)
            allowed = set(self.selected_classes)
            self.indices = [i for i, lab in enumerate(labels) if lab in allowed]

        # Map selected class name -> contiguous int ID for training
        self.class_to_idx = {name: idx for idx, name in enumerate(self.selected_classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.hf_ds[real_idx]

        img = ex["image"]
        label_name = ex["label"]           # string like "High"
        label = self.class_to_idx[label_name]  # int 0..K-1

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# Full (unfiltered) train/val datasets
train_ds_full = FireRiskDataset(
    DATA_ROOT, split="train",
    transform=img_transform, classes_subset=None
)
val_ds_full = FireRiskDataset(
    DATA_ROOT, split="val",
    transform=img_transform, classes_subset=None
)

print("Class → index (global):", {c: i for i, c in enumerate(train_ds_full.classes)})
print("Full classes:", train_ds_full.selected_classes)
print("Train size:", len(train_ds_full), "Val size:", len(val_ds_full))


=========================================

from collections import Counter
from torch.utils.data import DataLoader

# Raw labels from FULL train split (strings)
label_names = train_ds_full.hf_ds["label"]
counts = Counter(label_names)

print("Train class counts:", counts)

all_classes = list(train_ds_full.classes)

# Prefer 'Water' as NC if present, else least frequent class
preferred_nc = "Water"
if preferred_nc in counts:
    nc_name = preferred_nc
else:
    nc_name = min(counts.items(), key=lambda x: x[1])[0]

print("Chosen NC class:", nc_name)

classes_train_nc = [c for c in all_classes if c != nc_name]
non_nc_samples = sum(counts[c] for c in classes_train_nc)

print("Classes kept for training:", classes_train_nc)
print("Non-NC sample count:", non_nc_samples)

if non_nc_samples == 0:
    raise ValueError(
        f"No samples left after removing NC class '{nc_name}'. "
        f"Class counts: {counts}"
    )

# Build NC datasets (filtering happens inside FireRiskDataset via indices)
train_ds_nc = FireRiskDataset(
    DATA_ROOT, split="train",
    transform=img_transform, classes_subset=classes_train_nc
)
val_ds_nc = FireRiskDataset(
    DATA_ROOT, split="val",
    transform=img_transform, classes_subset=classes_train_nc
)
test_ds_nc = val_ds_nc

print("Filtered train size:", len(train_ds_nc))
print("Filtered val size:", len(val_ds_nc))

if len(train_ds_nc) == 0:
    raise ValueError("Dataset filtering produced an empty train set — this should not happen now.")

# DataLoaders
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
