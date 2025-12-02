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
        ds_dict = load_from_disk(data_root)        # HuggingFace DatasetDict
        hf_ds = ds_dict[split]

        self.transform = transform

        # ----- get full class list -----
        label_feat = hf_ds.features["label"]
        if isinstance(label_feat, ClassLabel) and label_feat.names is not None:
            self.classes = list(label_feat.names)
        else:
            unique_ids = sorted(set(hf_ds["label"]))
            self.classes = [str(i) for i in unique_ids]

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # ----- no filtering: keep all classes -----
        if classes_subset is None:
            self.selected_classes = list(self.classes)
            self.hf_ds = hf_ds
            return

        # ----- filtering: keep only selected class names -----
        self.selected_classes = list(classes_subset)
        selected_ids = [self.class_to_idx[c] for c in self.selected_classes]

        # mapping old_label → new_label (0..K-1)
        idmap = {old: new for new, old in enumerate(selected_ids)}

        def keep(ex):
            return ex["label"] in selected_ids

        def relabel(ex):
            ex["label"] = idmap[ex["label"]]
            return ex

        hf_ds = hf_ds.filter(keep)
        hf_ds = hf_ds.map(relabel)

        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        ex = self.hf_ds[idx]
        img = ex["image"]
        label = ex["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---- load full train/val sets with no filtering ----
train_ds_full = FireRiskDataset(DATA_ROOT, split="train",
                                transform=img_transform, classes_subset=None)
val_ds_full   = FireRiskDataset(DATA_ROOT, split="val",
                                transform=img_transform, classes_subset=None)

print("Class → index:", train_ds_full.class_to_idx)
print("Full classes:", train_ds_full.selected_classes)


==========================================================================================

from torch.utils.data import DataLoader

# remove Water = NC scenario
all_classes = train_ds_full.selected_classes
classes_train_nc = [c for c in all_classes if c != "Water"]
classes_new_nc = ["Water"]

# re-create datasets filtering out Water
train_ds_nc = FireRiskDataset(DATA_ROOT, split="train",
                              transform=img_transform,
                              classes_subset=classes_train_nc)

val_ds_nc = FireRiskDataset(DATA_ROOT, split="val",
                            transform=img_transform,
                            classes_subset=classes_train_nc)

test_ds_nc = val_ds_nc   # same as before

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

print("Train size:", len(train_ds_nc))
print("Val size:", len(val_ds_nc))
print("Selected (kept) classes:", train_ds_nc.selected_classes)
