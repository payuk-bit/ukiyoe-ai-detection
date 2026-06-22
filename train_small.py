import os
import time
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DATASET_ROOT, IMG_SIZE, BATCH_SIZE

# Small custom CNN used as a deliberately weaker baseline so the contribution
# of the FFT and Gabor layers is visible, instead of being hidden by the
# EfficientNet-B0 ceiling effect. One fixed early insertion point is used; the
# early/middle/late position question is answered by the EfficientNet runs.
# Each variant is trained over several seeds and reported as mean +/- std.
#
# FFT layer:   Bammey, Q. (2024). Synthbuster. arXiv (frequency-domain motivation)
# Gabor layer: Luan, S. et al. (2018). Gabor Convolutional Networks.

# 1.Configuration
EPOCHS = 20              # maximum training epochs
LEARNING_RATE = 1e-4     # adam learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
PATIENCE = 5             # early stopping patience (epochs without improvement)
SEEDS = [42, 1, 7]       # one training run per seed, per variant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("./results/small_baseline")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")


def set_seed(seed):
    # seed all RNGs so data sampling and weight init vary per run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



#2.Model
# NOTE: if your repo already defines the corrected FFTLayer and the GaborLayer
# used in the EfficientNet experiments, import those instead so the small-CNN
# variants use identical layer code:  from layers import FFTLayer, GaborLayer
class FFTLayer(nn.Module):
    # corrected spectral layer: FFT -> learnable weighting (real & imaginary
    # parts kept, so phase is preserved) -> inverse FFT back to the spatial
    # domain, then concatenate-and-reduce
    def __init__(self, in_channels):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_i = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.reduce = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        freq = torch.fft.fft2(x, norm="ortho")                       # to frequency domain
        weighted = torch.complex(self.conv_r(freq.real), self.conv_i(freq.imag))
        x_freq = torch.fft.ifft2(weighted, norm="ortho").real        # back to spatial domain
        z = torch.cat([x, x_freq], dim=1)                            # concatenate
        return self.act(self.bn(self.reduce(z)))                     # reduce to C channels


class GaborLayer(nn.Module):
    # learnable Gabor filter bank (16 filters, 7x7) with five learnable
    # parameters per filter, followed by concatenate-and-reduce
    def __init__(self, in_channels, n_filters=16, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        # orientations initialised evenly over [0, pi) to cover all textures
        thetas = torch.linspace(0, math.pi, n_filters + 1)[:-1]
        self.theta = nn.Parameter(thetas.clone())
        self.sigma = nn.Parameter(torch.full((n_filters,), 2.0))
        self.lambd = nn.Parameter(torch.full((n_filters,), 4.0))
        self.gamma = nn.Parameter(torch.full((n_filters,), 0.5))
        self.psi = nn.Parameter(torch.zeros(n_filters))
        self.project = nn.Conv2d(n_filters, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.reduce = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def _build_filters(self, device):
        k = self.kernel_size
        coords = torch.arange(k, device=device).float() - k // 2
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        filters = []
        for i in range(self.n_filters):
            x_t = x * torch.cos(self.theta[i]) + y * torch.sin(self.theta[i])
            y_t = -x * torch.sin(self.theta[i]) + y * torch.cos(self.theta[i])
            envelope = torch.exp(-(x_t ** 2 + (self.gamma[i] ** 2) * y_t ** 2)
                                 / (2 * self.sigma[i] ** 2 + 1e-6))
            carrier = torch.cos(2 * math.pi * x_t / (self.lambd[i] + 1e-6) + self.psi[i])
            filters.append(envelope * carrier)
        return torch.stack(filters)  # (n_filters, k, k)

    def forward(self, x):
        B, C, H, W = x.shape
        gabor = self._build_filters(x.device)                  # (Nf, k, k)
        weight = gabor.unsqueeze(1).repeat(1, C, 1, 1) / C      # (Nf, C, k, k)
        pad = self.kernel_size // 2
        G = F.conv2d(x, weight, padding=pad)                   # (B, Nf, H, W)
        G = self.act(self.bn1(self.project(G)))                # (B, C, H, W)
        z = torch.cat([x, G], dim=1)                           # concatenate
        return self.act(self.bn2(self.reduce(z)))              # reduce to C channels


class SmallCNN(nn.Module):
    # weak baseline: very narrow channels, low dropout (so train/val track
    # together), coarse spatial path -> competent but clearly below ceiling
    def __init__(self, variant="baseline"):   # "baseline" | "fft" | "gabor"
        super().__init__()
        self.variant = variant
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),    # 224 -> 112
            nn.BatchNorm2d(4), nn.ReLU(inplace=True),
        )
        # early insertion: added layer operates on 4-channel 112x112 maps
        if variant == "fft":
            self.extra = FFTLayer(4)
        elif variant == "gabor":
            self.extra = GaborLayer(4)
        else:
            self.extra = nn.Identity()
        self.block2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),    # 112 -> 56
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),         # low dropout so train and val track together
            nn.Linear(8, 1),           # single output logit for binary
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.extra(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(variant: str = "baseline", num_classes: int = 1) -> nn.Module:
    # build the small CNN for the requested variant (baseline / fft / gabor)
    model = SmallCNN(variant=variant)
    return model


#3.Training
def train_one_epoch(model, loader, criterion, optimizer, device):
    #Train the model for one epoch. Returns average loss and accuracy
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().to(device)    #BCEWithLogitsLoss expects float
        optimizer.zero_grad()                  # reset gradients
        outputs = model(images).squeeze(1)     #forward pass and squeeze
        loss = criterion(outputs, labels)      # calculate  binary cross-entropy loss
        loss.backward()                        # backpropagation
        optimizer.step()                       # update weights
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()  # put treshold at 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc



#4.Validation
@torch.no_grad() # disable gradient computation for efficiency
# evaluate model on validation/test set. Returns loss, acc, f1, preds, labels.
def validate(model, loader, criterion, device): # model to evaluation mode
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels



#5.Full training loop
def train(model, loaders, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
    model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    # halve LR when validation loss plateaus for 2 epochs
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = SAVE_DIR / "best_model.pth"

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)
     # print history for plotting
    for epoch in range(1, epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        #Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, DEVICE
        )

        #validate
        val_loss, val_acc, val_f1, _, _ = validate(
            model, loaders["val"], criterion, DEVICE
        )

        elapsed = time.time() - start

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  "
              f"|  Val F1: {val_f1:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        #record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        #learning rate scheduler
        scheduler.step(val_loss)

        #early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  \u2713 Best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

    #save training history
    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history



#6.test evaluation
@torch.no_grad()
def evaluate_test(model, loader, device=DEVICE):
    # run full evaluation on the test set and print metrics.
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_acc, test_f1, preds, labels = validate(
        model, loader, criterion, device
    )

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\n{classification_report(labels, preds, target_names=['Human', 'AI'])}")

    #save test results
    results = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": test_f1,
        "loss": test_loss,
        "confusion_matrix": cm.tolist(),
    }
    with open(SAVE_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results



#7.plot training curves
def plot_history(history: dict, save_dir: Path = SAVE_DIR):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.show()
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")



#8.Main
if __name__ == "__main__":
    # train all three small-CNN variants over several seeds, then report
    # mean +/- std for the added rows in Table 3
    # NOTE: if your FFT/Gabor EfficientNet experiments used reduced augmentation
    # (no rotation for FFT, flip-only for Gabor), pass the same profile to
    # get_dataloaders() here so the small-CNN comparison mirrors them.
    name = {"baseline": "Small CNN baseline",
            "fft": "Small CNN + FFT",
            "gabor": "Small CNN + Gabor"}

    # all_runs[variant] = list of per-seed metric dicts
    all_runs = {v: [] for v in ["baseline", "fft", "gabor"]}

    for variant in ["baseline", "fft", "gabor"]:
        for seed in SEEDS:
            print("\n" + "#" * 60)
            print(f"# SMALL CNN  --  variant: {variant}  |  seed: {seed}")
            print("#" * 60)

            set_seed(seed)   # reseed before data + model so each run is independent

            #set per-variant, per-seed output directory
            SAVE_DIR = Path(f"./results/small_{variant}_seed{seed}")
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            #load data
            loaders = get_dataloaders()

            #build model
            model = build_model(variant=variant)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel: SmallCNN ({variant}) seed {seed}  params: {total_params:,}")

            #train
            model, history = train(model, loaders)

            #load best model and evaluate on test set
            model.load_state_dict(torch.load(SAVE_DIR / "best_model.pth",
                                              map_location=DEVICE))
            results = evaluate_test(model, loaders["test"])
            all_runs[variant].append(results)

            #plot the training curves
            plot_history(history, SAVE_DIR)

    #aggregate mean +/- std across seeds
    summary = {}
    print("\n--- small-CNN results: mean +/- std over seeds ---")
    for variant in ["baseline", "fft", "gabor"]:
        runs = all_runs[variant]
        agg = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            vals = np.array([r[metric] for r in runs], dtype=float)
            agg[metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
        summary[f"small_{variant}"] = {"seeds": SEEDS, "runs": runs, "aggregate": agg}

        a = agg
        print(f"{name[variant]:20s} "
              f"acc {a['accuracy']['mean']:.4f}\u00b1{a['accuracy']['std']:.4f}  "
              f"prec {a['precision']['mean']:.4f}\u00b1{a['precision']['std']:.4f}  "
              f"rec {a['recall']['mean']:.4f}\u00b1{a['recall']['std']:.4f}  "
              f"f1 {a['f1']['mean']:.4f}\u00b1{a['f1']['std']:.4f}")

    #save combined summary for Table 3
    with open("./results/small_cnn_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved ./results/small_cnn_seed_summary.json")