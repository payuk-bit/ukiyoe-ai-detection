import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import shap

from dataset import (
    UkiyoeDataset, DATASET_ROOT, IMG_SIZE,
    UKIYOE_MEAN, UKIYOE_STD,
)
from Effecienet import build_model
from train_fft import EfficientNetFFT
from train_gabor import EfficientNetGabor

warnings.filterwarnings("ignore")

#   use SHAP GradientExplainer for pixel-level feature attribution:
#   Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.

#1.Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("./results/shap_analysis")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_BACKGROUND = 50
N_EXPLAIN = 20

MODELS = [
    ("Baseline",     "baseline",     Path("./results/baseline/best_model.pth")),
    ("FFT early",    "fft_early",    Path("./results/fft_early_scratch/best_model.pth")),
    ("Gabor early",  "gabor_early",  Path("./results/gabor_early_scratch/best_model.pth")),
]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")


#2.Model wrapper
class SHAPModelWrapper(nn.Module):
#wraps any model to guarantee output shape (N, 1).
#SHAP GradientExplainer needs consistent output dimensions.
   
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        #make sure output is (N, 1)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        if out.ndim == 2 and out.shape[1] != 1:
            out = out[:, :1]
        return out



#3.Model loader
#load model and wrap for SHAP compatibility.
def load_model(model_type: str, model_path: Path) -> nn.Module:
    if model_type == "baseline":
        model = build_model()
    elif model_type.startswith("fft_"):
        position = model_type.split("_")[1]
        model = EfficientNetFFT(position=position)
    elif model_type.startswith("gabor_"):
        position = model_type.split("_")[1]
        model = EfficientNetGabor(position=position)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Wrap for SHAP compatibility
    wrapped = SHAPModelWrapper(model).to(DEVICE)
    wrapped.eval()
    return wrapped



#4.Data prep
def get_eval_transform(): #Eval transform: resize, tensor, normalize.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=UKIYOE_MEAN, std=UKIYOE_STD),
    ])


def load_images_as_tensor(dataset, indices):  #load specific images by index and return as a stacked tensor."""
    transform = get_eval_transform()
    images = []
    labels = []
    for idx in indices:
        path, label = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(label)
    return torch.stack(images), labels


def denormalize(tensor):  #reverse normalization for display.
    mean = torch.tensor(UKIYOE_MEAN).view(3, 1, 1)
    std = torch.tensor(UKIYOE_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)



#5.Calcuting SHAP
def compute_shap_values(model, background_data, explain_data):
    
    #calculateompute SHAP values using GradientExplainer (Lundberg & Lee, 2017).
    #processes images in small batches to avoid memory issues.
    #returns numpy array of shape (N, 3, H, W).
    
    background_data = background_data.to(DEVICE)

    explainer = shap.GradientExplainer(model, background_data)

    all_shap = []
    batch_size = 4  #Small batches to avoid memory issues

    for i in range(0, explain_data.shape[0], batch_size):
        batch = explain_data[i:i + batch_size].to(DEVICE)
        sv = explainer.shap_values(batch)

        #handle list output 
        if isinstance(sv, list):
            sv = sv[0]

        #convert to numpy
        if isinstance(sv, torch.Tensor):
            sv = sv.cpu().numpy()
        else:
            sv = np.array(sv)

        
       #print shape on first batch for debugging
        if i == 0:
            print(f"    Raw SHAP batch shape: {sv.shape}")

        #remove trailing dimension of 1 
        sv = sv.squeeze(-1)

        # fixes shape 
        if sv.ndim == 5:
            sv = sv[0]
        if sv.ndim == 4:
            
            if sv.shape[1] == 3 and sv.shape[2] == IMG_SIZE:
                pass  
            elif sv.shape[0] == 3 and sv.shape[2] == IMG_SIZE:
                sv = np.transpose(sv, (1, 0, 2, 3))  
            elif sv.shape[3] == 3:
                sv = np.transpose(sv, (0, 3, 1, 2)) 
        elif sv.ndim == 3:
            
            sv = sv[np.newaxis, ...]

        all_shap.append(sv)
        print(f"    Processed {min(i + batch_size, explain_data.shape[0])}/{explain_data.shape[0]}")

    result = np.concatenate(all_shap, axis=0)
    print(f"  Final SHAP shape: {result.shape}")
    return result


#6.Visualizations
def plot_shap_examples(shap_values, images, labels, model_name, save_dir,
                       n_show=8):
    n_show = min(n_show, shap_values.shape[0])
 #plot original, SHAP heatmap, and overlay for sample images.
    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 8))

    for i in range(n_show):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title("Human" if labels[i] == 0 else "AI", fontsize=9)
        axes[0, i].axis("off")

        shap_map = np.abs(shap_values[i]).mean(axis=0)
        axes[1, i].imshow(shap_map, cmap="hot")
        axes[1, i].set_title("SHAP importance", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(img)
        axes[2, i].imshow(shap_map, cmap="hot", alpha=0.5)
        axes[2, i].set_title("Overlay", fontsize=9)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("SHAP", fontsize=10)
    axes[2, 0].set_ylabel("Overlay", fontsize=10)

    plt.suptitle(f"SHAP Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_examples_{model_name.lower().replace(' ', '_')}.png",
                dpi=150)
    plt.show()
    print(f"  Saved SHAP examples for {model_name}")


def plot_mean_importance(shap_values, model_name, save_dir):
    mean_shap = np.abs(shap_values).mean(axis=(0, 1))
 #plot mean SHAP importance map averaged across all images.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_shap, cmap="hot")
    ax.set_title(f"Mean SHAP Importance — {model_name}", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_mean_{model_name.lower().replace(' ', '_')}.png",
                dpi=150)
    plt.show()


def plot_by_class(shap_values, labels, model_name, save_dir):  #Plot SHAP maps separated by class with difference map.
    labels_arr = np.array(labels)
    human_idx = labels_arr == 0
    ai_idx = labels_arr == 1

    human_shap = np.abs(shap_values[human_idx]).mean(axis=(0, 1))
    ai_shap = np.abs(shap_values[ai_idx]).mean(axis=(0, 1))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    im1 = ax1.imshow(human_shap, cmap="hot")
    ax1.set_title("Human images", fontsize=11)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(ai_shap, cmap="hot")
    ax2.set_title("AI images", fontsize=11)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    diff = ai_shap - human_shap
    max_val = max(abs(float(diff.min())), abs(float(diff.max())))
    if max_val == 0:
        max_val = 1.0
    im3 = ax3.imshow(diff, cmap="RdBu_r", vmin=-max_val, vmax=max_val)
    ax3.set_title("Difference (AI − Human)", fontsize=11)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(f"SHAP by Class — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_by_class_{model_name.lower().replace(' ', '_')}.png",
                dpi=150)
    plt.show()
    print(f"  Saved class comparison for {model_name}")


def plot_channel_importance(shap_values, model_name, save_dir): # make bar chart of mean |SHAP| per RGB channel."""
    r_imp = float(np.abs(shap_values[:, 0, :, :]).mean())
    g_imp = float(np.abs(shap_values[:, 1, :, :]).mean())
    b_imp = float(np.abs(shap_values[:, 2, :, :]).mean())

    channels = ["Red", "Green", "Blue"]
    values = [r_imp, g_imp, b_imp]
    colors_rgb = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(channels, values, color=colors_rgb, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"Channel Importance — {model_name}", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / f"shap_channels_{model_name.lower().replace(' ', '_')}.png",
                dpi=150)
    plt.show()
    print(f"  Saved channel importance for {model_name}")



#7.Model comparisons
def plot_model_comparison(all_results: dict, save_dir: Path):
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
 # make side-by-side mean SHAP maps for all models.
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, shap_vals) in zip(axes, all_results.items()):
        mean_shap = np.abs(shap_vals).mean(axis=(0, 1))
        im = ax.imshow(mean_shap, cmap="hot")
        ax.set_title(model_name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Mean SHAP Importance — Model Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_model_comparison.png", dpi=150)
    plt.show()
    print(f"\nSaved model comparison to {save_dir / 'shap_model_comparison.png'}")


def plot_region_analysis(all_results: dict, save_dir: Path):
    region_names = [
        "Top-left", "Top-center", "Top-right",
        "Mid-left", "Center", "Mid-right",
        "Bot-left", "Bot-center", "Bot-right",
    ]
#3x3 spatial grid analysis comparing models.
    h, w = IMG_SIZE, IMG_SIZE
    h3, w3 = h // 3, w // 3

    all_region_data = {}

    for model_name, shap_vals in all_results.items():
        mean_shap = np.abs(shap_vals).mean(axis=(0, 1))
        regions = []
        for r in range(3):
            for c in range(3):
                region = mean_shap[r * h3:(r + 1) * h3, c * w3:(c + 1) * w3]
                regions.append(float(region.mean()))
        all_region_data[model_name] = regions

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(9)
    width = 0.8 / len(all_region_data)
    colors = ["#333333", "#1a73e8", "#e8421a"]

    for i, (model_name, regions) in enumerate(all_region_data.items()):
        ax.bar(x + i * width, regions, width, label=model_name,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(all_region_data) - 1) / 2)
    ax.set_xticklabels(region_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Spatial Region Importance — Model Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "shap_region_analysis.png", dpi=150)
    plt.show()
    print(f"Saved region analysis to {save_dir / 'shap_region_analysis.png'}")

    with open(save_dir / "region_importance.json", "w") as f:
        json.dump(all_region_data, f, indent=2)



#8.Exucute
if __name__ == "__main__":
    print("=" * 60)
    print("SHAP Explainability Analysis (GradientExplainer)")
    print("=" * 60)

    #load test dataset
    test_dataset = UkiyoeDataset(DATASET_ROOT, split="test", transform=None)

    #select the balanced samples
    human_indices = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 0]
    ai_indices = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 1]

    np.random.shuffle(human_indices)
    np.random.shuffle(ai_indices)

    bg_indices = human_indices[:N_BACKGROUND // 2] + ai_indices[:N_BACKGROUND // 2]
    explain_indices = (human_indices[N_BACKGROUND // 2:N_BACKGROUND // 2 + N_EXPLAIN // 2] +
                       ai_indices[N_BACKGROUND // 2:N_BACKGROUND // 2 + N_EXPLAIN // 2])

    print(f"\nLoading {N_BACKGROUND} background samples...")
    background_data, _ = load_images_as_tensor(test_dataset, bg_indices)

    print(f"Loading {N_EXPLAIN} images to explain...")
    explain_data, explain_labels = load_images_as_tensor(test_dataset, explain_indices)
    print(f"  Images shape: {explain_data.shape}")

    #run SHAP for each model
    all_results = {}

    for model_name, model_type, model_path in MODELS:
        print("\n" + "#" * 60)
        print(f"  Analysing: {model_name}")
        print("#" * 60)

        if not model_path.exists():
            print(f"  ⚠ Model not found at {model_path}, skipping...")
            continue

        model = load_model(model_type, model_path)

        # calculate SHAP values
        print(f"  Computing SHAP values...")
        shap_values = compute_shap_values(model, background_data, explain_data)

        all_results[model_name] = shap_values

        #visualizations
        plot_shap_examples(shap_values, explain_data, explain_labels,
                           model_name, SAVE_DIR)
        plot_mean_importance(shap_values, model_name, SAVE_DIR)
        plot_by_class(shap_values, explain_labels, model_name, SAVE_DIR)
        plot_channel_importance(shap_values, model_name, SAVE_DIR)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #model comparisons
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Cross-model comparison")
        print("=" * 60)
        plot_model_comparison(all_results, SAVE_DIR)
        plot_region_analysis(all_results, SAVE_DIR)

    print(f"\nAll results saved to {SAVE_DIR}")
    print("Done!")
