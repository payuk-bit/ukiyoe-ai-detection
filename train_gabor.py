#This experiment integrates learnable Gabor filter layers directly into the 
# EfficientNet-B0 architecture at different positions to determine where texture-based information most effectively contributes to classification.

#Gabor filters are parameterized by orientation and spatial frequency, making
#them well-suited for capturing the structured, repetitive texture patterns
#characteristic of woodblock printing (Luan et al., 2019). Unlike fixed external features like LBP, these filters are implemented as convolutional layers enabling end-to-end optimization.

#Positions tested:
# - 'early'  : After block 2 (low-level features)
#  - 'middle' : After block 5 (mid-level features)
#  - 'late'   : After block 7 (high-level features)

# The learnable Gabor filter implementation is based on:
#   Luan, S., Chen, C., Zhang, B., Han, J., & Liu, J. (2019). Gabor Convolutional Networks. IEEE Trans. Image Processing, 27(9), 4357-4366.
#    doi: 10.1109/TIP.2018.2835143


import os
import time
import json
import math
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
GABOR_POSITIONS = ["early", "middle", "late"]

print(f"Using device: {DEVICE}")



#2.Gabor Filter Layer (based on Luan et al. (2019))
class GaborConv2d(nn.Module):
    #Learnable Gabor filter bank implemented as a convolutional layer.

    #Each filter is parameterized by:
        #- theta:     Orientation of the filter (radians)
        #- sigma:     Standard deviation of the Gaussian envelope
        #- lambd:     Wavelength of the sinusoidal component
        #- gamma:     Spatial aspect ratio (ellipticity)
        #- psi:       Phase offset

    #All parameters are learnable via backpropagation, following the approach of Luan et al. (2018) — GaborNet.
    #The filters are initialized with evenly spaced orientations to provide broad initial coverage of texture directions.

    def __init__(self, in_channels: int, num_filters: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 # have the same-padding to save spatial dimensions

        #total output filters = num_filters per input channel
        #apply the same Gabor bank to each input channel
        total_filters = num_filters

        #learnable Gabor parameters
        #Initialize theta with evenly spaced orientations [0, pi)
        # (Luan et al., 2019,)
        theta_init = torch.linspace(0, math.pi, total_filters + 1)[:-1]
        self.theta = nn.Parameter(theta_init)

        #sigma: controls Gaussian envelope width
        self.sigma = nn.Parameter(torch.ones(total_filters) * 2.0)

        #lambd: wavelength (controls frequency)
        self.lambd = nn.Parameter(torch.ones(total_filters) * 4.0)

        #gamma: spatial aspect ratio
        self.gamma = nn.Parameter(torch.ones(total_filters) * 0.5)

        #psi: phase offset
        self.psi = nn.Parameter(torch.zeros(total_filters))

        # pre-calculate coordinate grids 
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        self.register_buffer('grid_y', y.view(-1, 1).repeat(1, kernel_size))
        self.register_buffer('grid_x', x.view(1, -1).repeat(kernel_size, 1))

    def _make_gabor_kernels(self) -> torch.Tensor:
        # generate Gabor filter kernels from current parameters.
        kernels = []
        for i in range(self.num_filters):
            theta = self.theta[i]
            sigma = self.sigma[i].clamp(min=0.5)
            lambd = self.lambd[i].clamp(min=1.0)
            gamma = self.gamma[i].clamp(min=0.1)
            psi = self.psi[i]

            #rotate coordinates
            x_theta = self.grid_x * torch.cos(theta) + self.grid_y * torch.sin(theta)
            y_theta = -self.grid_x * torch.sin(theta) + self.grid_y * torch.cos(theta)

            #gabor formula  (Luan et al., 2019)
            # g(x,y) = exp(-(x'^2 + gamma^2 * y'^2) / 2sigma^2) * cos(2pi*x'/lambda + psi)
            gaussian = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
            sinusoid = torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernel = gaussian * sinusoid

            #normalize
            kernel = kernel / (kernel.norm() + 1e-8)
            kernels.append(kernel)

        # hape is num_filters, 1, kernel_size, kernel_size
        return torch.stack(kernels).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernels = self._make_gabor_kernels()  

        #apply each Gabor filter to each channel and sum across channels
        # expand kernels to match input channels
        kernels_expanded = kernels.repeat(1, C, 1, 1) 

        out = nn.functional.conv2d(x, kernels_expanded, padding=self.padding)
        return out



#3.Gabor Integration Layer
class GaborLayer(nn.Module):
    #integrates Gabor filter responses with the original feature maps.
    #Concat-and-reduce strategy to combine texture + spatial features
    #Pipeline:
        #1. Apply learnable Gabor filter bank → texture response maps
        #2. Pass through BatchNorm + ReLU
        #3. Project Gabor responses to match original channel count
        #4. Concatenate with original features
        #5. Reduce back to original channel count via 1x1 conv
  

    def __init__(self, in_channels: int, num_gabor_filters: int = 16,
                 kernel_size: int = 7):
        super().__init__()
       #Gabor filter bank adapted from Luan et al., 2019.
        self.gabor_conv = GaborConv2d(
            in_channels=in_channels,
            num_filters=num_gabor_filters,
            kernel_size=kernel_size,
        )

        #Process Gabor responses
        self.gabor_process = nn.Sequential(
            nn.BatchNorm2d(num_gabor_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_gabor_filters, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        #Channel reduction: concatenated spatial + gabor, back to original
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Gabor texture features
        gabor_out = self.gabor_conv(x)
        gabor_features = self.gabor_process(gabor_out)

        #concatenate spatial + texture features
        combined = torch.cat([x, gabor_features], dim=1)

        #reduce back to original channel count
        out = self.channel_reduce(combined)

        return out



# 4.Model with Gabor integration
#Base architecture: Tan & Le (2019)
class EfficientNetGabor(nn.Module):
    POSITION_MAP = {
        "early":  (2, 24),
        "middle": (5, 112),
        "late":   (7, 320),
    }

    def __init__(self, position: str = "middle", num_classes: int = 1,
                 num_gabor_filters: int = 16, gabor_kernel_size: int = 7):
        super().__init__()

        if position not in self.POSITION_MAP:
            raise ValueError(f"Invalid position '{position}'. "
                             f"Choose from {list(self.POSITION_MAP.keys())}")

        self.position = position
        block_idx, channels = self.POSITION_MAP[position]

        
        base_model = models.efficientnet_b0(weights=None)

        #split features into before/after the insertion point
        self.features_before = nn.Sequential(*list(base_model.features[:block_idx + 1]))
        self.gabor_layer = GaborLayer(
            in_channels=channels,
            num_gabor_filters=num_gabor_filters,
            kernel_size=gabor_kernel_size,
        )
        self.features_after = nn.Sequential(*list(base_model.features[block_idx + 1:]))

        #pooling and classifier
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_before(x)
        x = self.gabor_layer(x)
        x = self.features_after(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#5.Training  (same as Effecienet.py)
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



#6.Validation (same as Effecienet.py)
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



#7.Training (same as Effecienet.py)
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



#8.Test evaluation (same as Effecienet.py)
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



#9.Plot training curves
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



#10.Compare all positions
def plot_position_comparison(all_results: dict, save_dir: Path):
    #compare test metrics across Gabor insertion positions.
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

    ax.set_xlabel("Gabor Layer Position")
    ax.set_ylabel("Score")
    ax.set_title("Gabor Position Comparison — Test Set Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Gabor {p}" for p in positions])
    ax.set_ylim(0.9, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "position_comparison.png", dpi=150)
    plt.show()
    print(f"Saved position comparison to {save_dir / 'position_comparison.png'}")



#11.Main
if __name__ == "__main__":
    #load data 
    loaders = get_dataloaders(experiment="gabor")

    all_results = {}

    for position in GABOR_POSITIONS:
        print("\n" + "#" * 60)
        print(f"  EXPERIMENT 3: EfficientNet-B0 + Gabor @ {position.upper()}")
        print("#" * 60)

        save_dir = Path(f"./results/gabor_{position}_scratch")
        save_dir.mkdir(parents=True, exist_ok=True)

        #build model
        model = EfficientNetGabor(position=position)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"\nModel: EfficientNet-B0 + Gabor @ {position}")
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
        plot_history(history, f"Gabor @ {position}", save_dir)

    #summary across all positions
    print("\n" + "=" * 60)
    print("GABOR POSITION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Position':<12} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 48)
    for pos, res in all_results.items():
        print(f"{pos:<12} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
              f"{res['recall']:>8.4f} {res['f1']:>8.4f}")
    print("=" * 60)

    #save combined results
    combined_dir = Path("./results/gabor_combined_scratch")
    combined_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_dir / "all_positions.json", "w") as f:
        json.dump(all_results, f, indent=2)

    #plot comparisons
    plot_position_comparison(all_results, combined_dir)
