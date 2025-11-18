import os, glob
from PIL import Image
from torch.utils.data import Dataset

class FireRiskDataset(Dataset):
    def __init__(self, root, split="train", transform=None, classes_subset=None):
        self.root = root
        self.transform = transform

        # collect all PNGs from train/* and val/* (we ignore their current split)
        patterns = [
            os.path.join(root, "train", "*", "*.png"),
            os.path.join(root, "val",   "*", "*.png"),
        ]
        all_paths = []
        for pat in patterns:
            all_paths.extend(glob.glob(pat))
        all_paths = sorted(all_paths)

        paths, labels = [], []
        for p in all_paths:
            fname = os.path.basename(p)
            grid_code = int(fname.split("_")[1])  # 1..7
            if classes_subset is not None and grid_code not in classes_subset:
                continue
            paths.append(p)
            labels.append(grid_code - 1)          # 0..6

        # 70/15/15 split, stratified by label
        from sklearn.model_selection import train_test_split
        n = len(paths)
        idx = list(range(n))

        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, stratify=labels, random_state=42
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=temp_labels, random_state=42
        )

        if split == "train":   use_idx = train_idx
        elif split == "val":   use_idx = val_idx
        elif split == "test":  use_idx = test_idx
        else: raise ValueError(f"Unknown split: {split}")

        self.paths  = [paths[i] for i in use_idx]
        self.labels = [labels[i] for i in use_idx]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

#########

from torchvision import transforms
from torch.utils.data import DataLoader

root = "datasets/FireRisk"

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_ds = FireRiskDataset(root, split="train", transform=train_tf)
val_ds   = FireRiskDataset(root, split="val",   transform=eval_tf)
test_ds  = FireRiskDataset(root, split="test",  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

