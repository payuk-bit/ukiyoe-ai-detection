#This experiment integrates an FFT layer directly into the EfficientNet-B0
#architecture at different positions to determine where frequency-domain
#information most effectively contributes to classification.

#The FFT layer computes the 2D Fast Fourier Transform of the feature maps,
#extracts the magnitude spectrum, and concatenates it with the original
#spatial features. This allows the network to learn from both spatial and
#frequency-domain representations simultaneously.

#Positions tested:
# - 'early'  : After block 2 (low-level features)
#  - 'middle' : After block 5 (mid-level features)
#  - 'late'   : After block 7 (high-level features)

# The use of 2D FFT magnitude spectrum for detecting AI-generated images is based on:
#   Bammey, Q. (2023). Synthbuster: Towards Detection of Diffusion Model Generated Images. IEEE Open J. Signal Processing.
#   doi: 10.1109/OJSP.2023.3337714

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DATASET_ROOT, IMG_SIZE, BATCH_SIZE

#1.Configuration same as Effecienet.py
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Positions to test
FFT_POSITIONS = ["early", "middle", "late"]

print(f"Using device: {DEVICE}")



#2.FFT Layer
class FFTLayer(nn.Module):
    
    #Learnable FFT integration layer.

    #Computes the 2D FFT magnitude spectrum of the input feature maps,
    #passes it through a learnable 1x1 convolution to weight frequency components, and concatenates the result with the original spatial features. 
    #A 1x1 convolution then reduces the doubled channels back  to the original channel count.
    #This allows the network to learn from spatial and spectral representations.

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        #learnable weighting of frequency components
        self.freq_weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        #channel reduction: concatenated spatial + freq, back to original
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #calculate 2D FFT and shift zero-frequency to center
        # (Bammey, 2023)
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)

        #extract magnitude spectrum 
        # (Bammey, 2023)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log1p(magnitude)

        #learn which frequency components matter
        freq_features = self.freq_weight(magnitude)

        #concatenate spatial + frequency features
        combined = torch.cat([x, freq_features], dim=1)

        #reduce back to original channel count
        out = self.channel_reduce(combined)

        return out



#3.Model with FFT integration
# base architecture: Tan & Le (2019)
class EfficientNetFFT(nn.Module):
    
    POSITION_MAP = {
        "early":  (2, 24),
        "middle": (5, 112),
        "late":   (7, 320),
    }

    def __init__(self, position: str = "middle", num_classes: int = 1):
        super().__init__()

        if position not in self.POSITION_MAP:
            raise ValueError(f"Invalid position '{position}'. "
                             f"Choose from {list(self.POSITION_MAP.keys())}")

        self.position = position
        block_idx, channels = self.POSITION_MAP[position]

        base_model = models.efficientnet_b0(weights=None)  # load EfficientNet-B0 without pretrained weights (Tan & Le, 2019)

        #split features into before/after the insertion point
        self.features_before = nn.Sequential(*list(base_model.features[:block_idx + 1]))
        self.fft_layer = FFTLayer(in_channels=channels)
        self.features_after = nn.Sequential(*list(base_model.features[block_idx + 1:]))

        #pooling and classifier
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_before(x)
        x = self.fft_layer(x)
        x = self.features_after(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



#4. Training  (same  as Effecienet.py)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)}  "
                  f"loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc



#5.Validation
@torch.no_grad()
def validate(model, loader, criterion, device):
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



#6.Training loop (same as Effecienet.py)
def train(model, loaders, save_dir, epochs=EPOCHS, lr=LEARNING_RATE,
          patience=PATIENCE):
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

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = save_dir / "best_model.pth"

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, DEVICE
        )
        val_loss, val_acc, val_f1, _, _ = validate(
            model, loaders["val"], criterion, DEVICE
        )

        elapsed = time.time() - start

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  "
              f"|  Val F1: {val_f1:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history



#7.Test evaluation (same as Effecienet.py)
@torch.no_grad()
def evaluate_test(model, loader, save_dir, device=DEVICE):
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

    results = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": test_f1,
        "loss": test_loss,
        "confusion_matrix": cm.tolist(),
    }
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results



#8.Plot training curves 
def plot_history(history: dict, title: str, save_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.show()



#9.Compare all positions
def plot_position_comparison(all_results: dict, save_dir: Path):
    """Compare test metrics across FFT insertion positions."""
    positions = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(positions))
    width = 0.18
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

    for i, metric in enumerate(metrics):
        values = [all_results[pos][metric] for pos in positions]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("FFT Layer Position")
    ax.set_ylabel("Score")
    ax.set_title("FFT Position Comparison — Test Set Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"FFT {p}" for p in positions])
    ax.set_ylim(0.9, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "position_comparison.png", dpi=150)
    plt.show()
    print(f"Saved position comparison to {save_dir / 'position_comparison.png'}")



#10.Main 
if __name__ == "__main__":
    #load data with FFT-specific augmentations
    loaders = get_dataloaders(experiment="fft")

    all_results = {}

    for position in FFT_POSITIONS:
        print("\n" + "#" * 60)
        print(f"  EXPERIMENT 2: EfficientNet-B0 + FFT @ {position.upper()}")
        print("#" * 60)

        
        save_dir = Path(f"./results/fft_{position}_scratch")
        save_dir.mkdir(parents=True, exist_ok=True)

        #build model
        model = EfficientNetFFT(position=position)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"\nModel: EfficientNet-B0 + FFT @ {position}")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

        #train
        model, history = train(model, loaders, save_dir)

        #load best model and evaluate
        model.load_state_dict(torch.load(save_dir / "best_model.pth",
                                          map_location=DEVICE))
        results = evaluate_test(model, loaders["test"], save_dir)
        all_results[position] = results

        #plot training curves
        plot_history(history, f"FFT @ {position}", save_dir)

    #summary across all positions
    print("\n" + "=" * 60)
    print("FFT POSITION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Position':<12} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 48)
    for pos, res in all_results.items():
        print(f"{pos:<12} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
              f"{res['recall']:>8.4f} {res['f1']:>8.4f}")
    print("=" * 60)

    #save combined results
    combined_dir = Path("./results/fft_combined_scratch")
    combined_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_dir / "all_positions.json", "w") as f:
        json.dump(all_results, f, indent=2)

    #plot comparisons
    plot_position_comparison(all_results, combined_dir)
