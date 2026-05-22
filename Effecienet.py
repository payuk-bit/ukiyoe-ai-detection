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

# Architecture loaded from torchvision.models.efficientnet_b0()
#   Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model
#   Scaling for Convolutional Neural Networks. arXiv:1905.11946.

# 1.Configuration
EPOCHS = 20              # maximum training epochs
LEARNING_RATE = 1e-4     # adam learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
PATIENCE = 5             # early stopping patience (epochs without improvement)        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("./results/baseline")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")



#2.Model
def build_model(num_classes: int = 1) -> nn.Module:
    # load architecture without pretrained weights (Tan & Le, 2019)
    model = models.efficientnet_b0(weights=None)
     # get the number of input features to the original classifier
    #all layers are trainable (no freezing)
    #replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),  # same dropout as original
        nn.Linear(in_features, num_classes), # single output logit for binary
    )
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
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
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
    #load data
    loaders = get_dataloaders()

    #build model
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-B0")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    #train
    model, history = train(model, loaders)

    #load best model and evaluate on test set
    model.load_state_dict(torch.load(SAVE_DIR / "best_model.pth",
                                      map_location=DEVICE))
    evaluate_test(model, loaders["test"])

    #plot the training curves
    plot_history(history)