### [1] Setup & Imports

#########################

%load_ext autoreload
%autoreload 2

%matplotlib inline

import nest_asyncio
nest_asyncio.apply()

#########################

import os
import json
import numpy as np
from pathlib import Path
import copy
import random
import torch
import tqdm
from torchvision import transforms
import tqdm

#########################

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

#########################

from data.firerisk import FireRiskDataset
from torch.utils.data import DataLoader
from utils.metrics import get_confusion_matrix, get_scores

#########################

DATA_ROOT = os.environ.get("FIRERISK_ROOT", f"{Path.home()}/datasets/FireRisk/")
DATA_ROOT

#########################
### [2] Image Transform & Dataset Loading 

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

#########################

train_ds_full = FireRiskDataset(DATA_ROOT, split="train",
                                transform=img_transform,
                                classes_subset=None)
val_ds_full = FireRiskDataset(DATA_ROOT, split="val",
                              transform=img_transform,
                              classes_subset=None)
print(train_ds_full.class_to_idx)

#########################
### [3] Class Filtering & New Datasets

all_classes = train_ds_full.selected_classes
classes_train_nc = [c for c in all_classes if c != "Water"]
classes_new_nc = ["Water"]

#########################

train_ds_nc = FireRiskDataset(DATA_ROOT, split="train",
                              transform=img_transform,
                              classes_subset=classes_train_nc)
val_ds_nc = FireRiskDataset(DATA_ROOT, split="val",
                            transform=img_transform,
                            classes_subset=classes_train_nc)
test_ds_nc = val_ds_nc

#########################
### [4] Dataloaders

batch_size = 32
num_workers = 4
train_sampler_nc = None
val_sampler_nc = None
shuffle_train = True

#########################

train_loader_nc = DataLoader(
    train_ds_nc,
    batch_size=batch_size,
    shuffle=shuffle_train,
    sampler=train_sampler_nc,
    num_workers=num_workers,
    pin_memory=device == "cuda",
)

val_loader_nc = DataLoader(
    val_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler_nc,
    num_workers=num_workers,
    pin_memory=device == "cuda",
)

test_loader_nc = DataLoader(
    test_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device == "cuda",
)

print("New-class train size:", len(train_ds_nc),
      "val/test size:", len(val_ds_nc))
print("Selected classes:", train_ds_nc.selected_classes)

#########################
### [5] ViT Model Definition 

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import vit_b_16

#########################

class ViTForImageClassification(nn.Module):
    def __init__(self):
        super(ViTForImageClassification, self).__init__()
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 7)
        # self.class_token = create_feature_extractor(self.model, return_nodes={"encoder.ln": "getitem_5"})

    def forward(self, image):
        output = self.vit(image)
        return output

    def get_embedding(self, image):
        feats = self.vit._process_input(image)
        batch_class_token = self.vit.class_token.expand(image.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        output = self.vit.encoder(feats)
        output = output[:, 0]
        return output

#########################
### [6] Instantiate Model 

num_classes_nc = len(train_ds_nc.selected_classes)
model_nc = ViTForImageClassification()
model_nc.to(device)

#########################
### [7] Training Helpers 
## run_epoch()

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
        with tqdm.tqdm(loader, unit="batch", mininterval=0) as bar:
            for data in bar:
                x, y = data
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if train:
                    optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)

                if train:
                    loss.backward()
                    optimizer.step()

                if loss is not None:
                    total_loss += loss.detach() * x.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum()
                total += x.size(0)

                bar.set_postfix(loss=loss.item(), acc=(total_correct / total))

    avg_loss = (total_loss / total).item() if total.item() > 0 else 0.0
    acc = (total_correct / total).item() if total.item() > 0 else 0.0

    return avg_loss, acc

#########################
## train_model()

def train_model(model, device, train_loader, val_loader,
                train_sampler=None, val_sampler=None,
                epochs: int = 10, lr: float = 5e-4, wd: float = 0.05):

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

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), f'./saved_models/epoch_{epoch}_acc_{best_val_acc}')

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history

#########################
### [8] Example Training Run

epochs = 10
_ = train_model(
    model_nc,
    device,
    train_loader_nc,
    val_loader_nc,
    train_sampler=train_sampler_nc,
    val_sampler=val_sampler_nc,
    epochs=epochs
)

#########################
### [9] Load Trained Weights

model_path = '/home/jovyan/driftDSS1/saved_models/epoch_10_acc_0.58500736951828'
model_nc.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model_nc.eval()

#########################
### [10] Embedding Extraction 

def get_embeddings_and_preds(m, loader):
    m.eval()
    embedding_list = []
    prediction_list = []
    label_list = []

    pbar = tqdm.tqdm(loader, total=len(loader), unit='batch', leave=True)

    for image, label in pbar:
        with torch.no_grad():
            image = image.to(device, non_blocking=True)

            embedding = m.get_embedding(image)
            embedding_list.append(embedding)

            logits = m(image)
            pred = logits.argmax(dim=1)

            prediction_list.append(pred)
            label_list.append(label)

    embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
    predictions = torch.cat(prediction_list, dim=0).cpu().numpy()
    labels = torch.cat(label_list, dim=0).cpu().numpy()

    print(embeddings.shape, predictions.shape, labels.shape)
    return embeddings, predictions, labels

#########################
### [11] Evaluation Dataloaders & Save Embeddings 

eval_train_loader_nc = DataLoader(
    train_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device == "cuda",
)

eval_test_loader_nc = DataLoader(
    test_ds_nc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device == "cuda",
)

train_embeddings, train_predictions, train_labels =
    get_embeddings_and_preds(model_nc, eval_train_loader_nc)

test_embeddings, test_predictions, test_labels =
    get_embeddings_and_preds(model_nc, eval_test_loader_nc)

#########################
def save_embeddings_and_predictions(embeddings, labels, predictions, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, truth=labels, preds=predictions)
    print(f"embeddings saved at {path}")

#########################
tag = "missing-class"
train_path = f"/home/jovyan/drift-baseline-data/saved_embeddings/{tag}/train_embeddings_and_predictions.npz"
test_path  = f"/home/jovyan/drift-baseline-data/saved_embeddings/{tag}/test_embeddings_and_predictions.npz"

save_embeddings_and_predictions(train_embeddings, train_predictions, train_labels, train_path)
save_embeddings_and_predictions(test_embeddings, test_predictions, test_labels, test_path)

#########################
###  [12] Retrieve All Test Data (Full validation set)

from torch.utils.data import Subset

test_ds_all = FireRiskDataset(DATA_ROOT, split="val",
                              transform=img_transform,
                              classes_subset=None)
labels_all = test_ds_all.labels
classes_all = test_ds_all.selected_classes

#########################
### [13] Simulate Drift (Water Appearing)

indices_by_class = {cls_name: [] for cls_name in classes_all}
for idx, lab in enumerate(labels_all):
    cls_name = test_ds_all.idx_to_class[lab]
    indices_by_class[cls_name].append(idx)

for c in indices_by_class:
    random.shuffle(indices_by_class[c])

window_size = 1000
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

#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
