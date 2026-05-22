import io
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from dataset import (
    UkiyoeDataset, DATASET_ROOT, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    UKIYOE_MEAN, UKIYOE_STD,
)

from Effecienet import build_model
from train_fft import EfficientNetFFT
from train_gabor import EfficientNetGabor

# Evaluation methodology follows:
#   Bammey (2023), Synthbuster; Yan et al. (2024), Sanity Check

#1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("./results/robustness_scratch")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Baseline",     "baseline",     Path("./results/baseline/best_model.pth")),
    ("FFT early",    "fft_early",    Path("./results/fft_early_scratch/best_model.pth")),
    ("FFT middle",   "fft_middle",   Path("./results/fft_middle_scratch/best_model.pth")),
    ("FFT late",     "fft_late",     Path("./results/fft_late_scratch/best_model.pth")),
    ("Gabor early",  "gabor_early",  Path("./results/gabor_early_scratch/best_model.pth")),
    ("Gabor middle", "gabor_middle", Path("./results/gabor_middle_scratch/best_model.pth")),
    ("Gabor late",   "gabor_late",   Path("./results/gabor_late_scratch/best_model.pth")),
]

#Perturbation levels
JPEG_QUALITIES = [95, 75, 50, 30, 10]
NOISE_SIGMAS = [0.01, 0.05, 0.10, 0.20, 0.30]
RESOLUTION_FACTORS = [0.75, 0.50, 0.25, 0.125]

print(f"Using device: {DEVICE}")



#2.Model loader
def load_model(model_type: str, model_path: Path) -> nn.Module:
    """Load a trained model by type."""
    if model_type == "baseline":
        model = build_model()
    elif model_type.startswith("fft_"):
        position = model_type.split("_")[1]  # early, middle, late
        model = EfficientNetFFT(position=position)
    elif model_type.startswith("gabor_"):
        position = model_type.split("_")[1]
        model = EfficientNetGabor(position=position)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model



#3.perturbations
#compress and decompress via JPEG at given quality level.
def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# add zero-mean Gaussian noise with  sigma, clamp to [0,1].
def apply_gaussian_noise(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    
    noise = torch.randn_like(tensor) * sigma
    return (tensor + noise).clamp(0.0, 1.0)

# downscale then upscale back to target size.
def apply_resolution_shift(image: Image.Image, factor: float,
                            target_size: int = IMG_SIZE) -> Image.Image:
    w, h = image.size
    small_size = (max(1, int(w * factor)), max(1, int(h * factor)))
    image = image.resize(small_size, Image.BILINEAR)
    image = image.resize((target_size, target_size), Image.BILINEAR)
    return image



#4.Perturbed Dataset
class PerturbedDataset(Dataset):
    def __init__(self, base_dataset, perturbation="none", level=0,
                 img_size=IMG_SIZE):
        self.samples = base_dataset.samples
        self.perturbation = perturbation
        self.level = level
        self.img_size = img_size

        self.resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=UKIYOE_MEAN, std=UKIYOE_STD
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # apply the perturbations before preprocessing
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.perturbation == "jpeg":
            image = apply_jpeg_compression(image, quality=int(self.level))

        if self.perturbation == "resolution":
            image = apply_resolution_shift(image, factor=self.level,
                                           target_size=self.img_size)

        image = self.resize(image)
        tensor = self.to_tensor(image)
        # noise is applied to tensor after ToTensor, and before normalize.
        if self.perturbation == "noise":
            tensor = apply_gaussian_noise(tensor, sigma=self.level)

        tensor = self.normalize(tensor)
        return tensor, label



#5.Evaluation
@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}



# 6.  Run robustness tests 
def run_robustness_tests(model, model_name, test_dataset):
    results = {"clean": None, "jpeg": {}, "noise": {}, "resolution": {}}

    #Clean
    print(f"\n  Clean test set...")
    clean_ds = PerturbedDataset(test_dataset)
    clean_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True)
    results["clean"] = evaluate(model, clean_loader)
    print(f"    Acc: {results['clean']['accuracy']:.4f}  "
          f"F1: {results['clean']['f1']:.4f}")

    #JPEG
    print(f"  JPEG compression...")
    for q in JPEG_QUALITIES:
        ds = PerturbedDataset(test_dataset, perturbation="jpeg", level=q)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)
        results["jpeg"][q] = evaluate(model, loader)
        print(f"    Q={q:3d}:  Acc={results['jpeg'][q]['accuracy']:.4f}  "
              f"F1={results['jpeg'][q]['f1']:.4f}")

    #Noise
    print(f"  Gaussian noise...")
    for sigma in NOISE_SIGMAS:
        ds = PerturbedDataset(test_dataset, perturbation="noise", level=sigma)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)
        results["noise"][sigma] = evaluate(model, loader)
        print(f"    σ={sigma:.2f}:  Acc={results['noise'][sigma]['accuracy']:.4f}  "
              f"F1={results['noise'][sigma]['f1']:.4f}")

    #Resolution
    print(f"  Resolution shifts...")
    for factor in RESOLUTION_FACTORS:
        ds = PerturbedDataset(test_dataset, perturbation="resolution", level=factor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)
        results["resolution"][factor] = evaluate(model, loader)
        eff = int(IMG_SIZE * factor)
        print(f"    {eff}px:  Acc={results['resolution'][factor]['accuracy']:.4f}  "
              f"F1={results['resolution'][factor]['f1']:.4f}")

    return results



#7.Comparison plots
def plot_robustness_comparison(all_model_results: dict, SAVE_DIR: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    #Color scheme for models
    colors = {
        "Baseline":     "#333333",
        "FFT early":    "#1a73e8",
        "FFT middle":   "#5b9bd5",
        "FFT late":     "#a3c4e9",
        "Gabor early":  "#e8421a",
        "Gabor middle": "#e88a6e",
        "Gabor late":   "#f0b8a8",
    }

    #JPEG
    ax = axes[0]
    for model_name, results in all_model_results.items():
        qualities = sorted(results["jpeg"].keys(), reverse=True)
        accs = [results["jpeg"][q]["accuracy"] for q in qualities]
        x_labels = [str(q) for q in qualities]
        ax.plot(x_labels, accs, "-o", label=model_name,
                color=colors.get(model_name, "#888"), markersize=5)
    ax.set_xlabel("JPEG Quality")
    ax.set_ylabel("Accuracy")
    ax.set_title("JPEG Compression Robustness")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    #Noise
    ax = axes[1]
    for model_name, results in all_model_results.items():
        sigmas = sorted(results["noise"].keys())
        accs = [results["noise"][s]["accuracy"] for s in sigmas]
        x_labels = [f"{s:.2f}" for s in sigmas]
        ax.plot(x_labels, accs, "-o", label=model_name,
                color=colors.get(model_name, "#888"), markersize=5)
    ax.set_xlabel("Noise Sigma (σ)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gaussian Noise Robustness")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    #Resolution
    ax = axes[2]
    for model_name, results in all_model_results.items():
        factors = sorted(results["resolution"].keys(), reverse=True)
        accs = [results["resolution"][f]["accuracy"] for f in factors]
        x_labels = [f"{int(IMG_SIZE * f)}px" for f in factors]
        ax.plot(x_labels, accs, "-o", label=model_name,
                color=colors.get(model_name, "#888"), markersize=5)
    ax.set_xlabel("Effective Resolution")
    ax.set_ylabel("Accuracy")
    ax.set_title("Resolution Shift Robustness")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Comparison — All Models (From Scratch)", fontsize=14)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "robustness_comparison.png", dpi=150)
    plt.show()
    print(f"\nSaved comparison plot to {SAVE_DIR / 'robustness_comparison.png'}")



#8. Summary table
def print_summary_table(all_model_results: dict):
    """Print a formatted summary comparing all models."""

    print("\n" + "=" * 90)
    print("ROBUSTNESS COMPARISON — ALL MODELS")
    print("=" * 90)

    #Header
    model_names = list(all_model_results.keys())
    header = f"{'Perturbation':<22} {'Level':<10}"
    for name in model_names:
        header += f" {name:>10}"
    print(header)
    print("-" * len(header))

    #Clean
    line = f"{'Clean':<22} {'—':<10}"
    for name in model_names:
        line += f" {all_model_results[name]['clean']['accuracy']:>10.4f}"
    print(line)
    print("-" * len(header))

    #JPEG
    for q in sorted(JPEG_QUALITIES, reverse=True):
        line = f"{'JPEG':<22} {'Q=' + str(q):<10}"
        for name in model_names:
            line += f" {all_model_results[name]['jpeg'][q]['accuracy']:>10.4f}"
        print(line)
    print("-" * len(header))

    #Noise
    for s in sorted(NOISE_SIGMAS):
        line = f"{'Noise':<22} {'σ=' + f'{s:.2f}':<10}"
        for name in model_names:
            line += f" {all_model_results[name]['noise'][s]['accuracy']:>10.4f}"
        print(line)
    print("-" * len(header))

    #Resolution
    for f in sorted(RESOLUTION_FACTORS, reverse=True):
        eff = f"{int(IMG_SIZE * f)}px"
        line = f"{'Resolution':<22} {eff:<10}"
        for name in model_names:
            line += f" {all_model_results[name]['resolution'][f]['accuracy']:>10.4f}"
        print(line)

    print("=" * len(header))



# 9.  Main
if __name__ == "__main__":
    #Load test dataset
    test_dataset = UkiyoeDataset(DATASET_ROOT, split="test", transform=None)

    all_model_results = {}

    for model_name, model_type, model_path in MODELS:
        print("\n" + "#" * 60)
        print(f"  Testing: {model_name}")
        print("#" * 60)

        if not model_path.exists():
            print(f"  ⚠ Model not found at {model_path}, skipping...")
            continue

        model = load_model(model_type, model_path)
        results = run_robustness_tests(model, model_name, test_dataset)
        all_model_results[model_name] = results

        #Free the memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #Print summary
    print_summary_table(all_model_results)

    #Save results
    json_results = {}
    for name, results in all_model_results.items():
        json_results[name] = {
            "clean": results["clean"],
            "jpeg": {str(k): v for k, v in results["jpeg"].items()},
            "noise": {str(k): v for k, v in results["noise"].items()},
            "resolution": {str(k): v for k, v in results["resolution"].items()},
        }
    with open(SAVE_DIR / "all_robustness_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {SAVE_DIR / 'all_robustness_results.json'}")

    #Plot comparisons
    plot_robustness_comparison(all_model_results, SAVE_DIR)
