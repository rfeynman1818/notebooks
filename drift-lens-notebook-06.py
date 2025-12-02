from datasets import load_from_disk, ClassLabel
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

DATA_ROOT = "/datasets/FireRisk"

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
        self.hf_ds = ds_dict[split]
        self.transform = transform

        raw_labels = self.hf_ds["label"]        # strings: 'High', 'Low', ...

        # global class list (from features if available)
        label_feat = self.hf_ds.features.get("label", None)
        if isinstance(label_feat, ClassLabel) and label_feat.names is not None:
            all_classes = list(label_feat.names)
        else:
            seen = {}
            for lab in raw_labels:
                if lab not in seen:
                    seen[lab] = None
            all_classes = list(seen.keys())

        self.classes = all_classes

        # indices to keep for this dataset (for NC vs full)
        if classes_subset is None:
            self.selected_classes = list(self.classes)
            self.indices = list(range(len(self.hf_ds)))
        else:
            self.selected_classes = list(classes_subset)
            allowed = set(self.selected_classes)
            self.indices = [i for i, lab in enumerate(raw_labels) if lab in allowed]

        # name → contiguous int id for this dataset
        self.class_to_idx = {name: idx for idx, name in enumerate(self.selected_classes)}

        # precompute integer labels aligned with indices
        self.labels = np.array([
            self.class_to_idx[self.hf_ds[i]["label"]] for i in self.indices
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.hf_ds[real_idx]

        img = ex["image"]
        label_name = ex["label"]
        label = self.class_to_idx[label_name]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
