import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Dataset: AI-ArtBench (Ukiyo-e subset)
#   Silva, R. (n.d.). AI-ArtBench: Real vs AI-generated artwork.
#   https://www.kaggle.com/datasets/ravidussilva/real-ai-art

#1.Configuration
DATASET_ROOT = Path("C:/Users/deowo/OneDrive/Documents/thesis/ai-artbench")

HUMAN_DIR = "human"
AI_DIRS = ["latent_diffusion", "stable_diffusion"]  

IMG_SIZE = 224                   # EfficientNet-B0 default input size
BATCH_SIZE = 32
NUM_WORKERS = 4                  
VALID_SPLIT = 0.3            
SEED = 42


#2.Transforms
UKIYOE_MEAN = [0.575, 0.536, 0.456]
UKIYOE_STD  = [0.282, 0.243, 0.200]
 
def get_train_transforms(img_size: int = IMG_SIZE,
                         experiment: str = "baseline") -> transforms.Compose:

    #resize with anti-aliasing
    base = [
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
    ]
 
    if experiment == "baseline":
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.02),
        ]
 
    elif experiment == "fft":
        #no rotation  
        #minimal color 
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05),
        ]
 
    elif experiment == "gabor":
        # no rotation 
        # no color jitter
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
 
    else:
        raise ValueError(f"Unknown experiment: {experiment}. "
                         f"Use 'baseline', 'fft', or 'gabor'.")
 
    finalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=UKIYOE_MEAN, std=UKIYOE_STD),
    ]
 
    return transforms.Compose(base + augment + finalize)
 
 
def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    #Validation / test transforms (no augmentation).
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=UKIYOE_MEAN, std=UKIYOE_STD),
    ])
 
 

#3.Dataset class
class UkiyoeDataset(Dataset):
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root:      Path to the dataset root (contains train/ and test/).
            split:     'train' or 'test'.
            transform: torchvision transforms to apply.
        """
        self.root = Path(root) / split
        self.transform = transform
        self.samples: list[tuple[str, int]] = []  # (filepath, label)
 
        # human images  →  label 0
        human_dir = self.root / HUMAN_DIR
        if human_dir.is_dir():
            for fp in sorted(human_dir.iterdir()):
                if fp.suffix.lower() in self.SUPPORTED_EXT:
                    self.samples.append((str(fp), 0))
 
        # AI images  →  label 1
        for ai_dir_name in AI_DIRS:
            ai_dir = self.root / ai_dir_name
            if ai_dir.is_dir():
                for fp in sorted(ai_dir.iterdir()):
                    if fp.suffix.lower() in self.SUPPORTED_EXT:
                        self.samples.append((str(fp), 1))
 
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                f"Check that subfolders '{HUMAN_DIR}' and {AI_DIRS} exist."
            )
 
        print(f"[{split.upper()}] Loaded {len(self.samples)} images  "
              f"(human={self.label_counts()[0]}, AI={self.label_counts()[1]})")
 
    # helpers 
    def label_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
 
 
#4.Splits
def split_train_val(
    dataset: UkiyoeDataset,
    val_fraction: float = VALID_SPLIT,
    seed: int = SEED,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    # split of the training set into train and validation subsets.
    from sklearn.model_selection import train_test_split
 
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))
 
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
 
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)
 
    print(f"  → train subset: {len(train_subset)}  |  val subset: {len(val_subset)}")
    return train_subset, val_subset
 
 

#5.Weighted sampler handles class imbalance
def make_weighted_sampler(subset: torch.utils.data.Subset) -> WeightedRandomSampler:
    #creates a WeightedRandomSampler so each batch is roughly balanced.

    labels = [subset.dataset.samples[i][1] for i in subset.indices]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
 
 

#6.DataLoading
def get_dataloaders(
    root: Path = DATASET_ROOT,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_fraction: float = VALID_SPLIT,
    use_weighted_sampler: bool = True,
    seed: int = SEED,
    experiment: str = "baseline",
) -> dict[str, DataLoader]:
    
    #returns a dict with 'train', 'val', and 'test' DataLoaders.
    
    #full training set 
    full_train = UkiyoeDataset(root, split="train",
                               transform=get_train_transforms(img_size, experiment))
    train_subset, val_subset = split_train_val(full_train, val_fraction, seed)
 
    #validation subset should use eval transforms .
    #wrap it so the transform is swapped at __getitem__ time.
    val_subset.dataset_transform_override = get_eval_transforms(img_size)
 
    #test set
    test_set = UkiyoeDataset(root, split="test",
                             transform=get_eval_transforms(img_size))
 
    #sampler 
    train_sampler = (make_weighted_sampler(train_subset)
                     if use_weighted_sampler else None)
 
    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),   # shuffle only if no sampler
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
 
    return loaders
 
 

#7.Validation-transform wrapper
class _ValTransformSubset(torch.utils.data.Dataset):
    
    #wraps a Subset so that evaluation transforms are applied instead of
    #the training transforms stored on the underlying dataset.
    
    def __init__(self, subset: torch.utils.data.Subset,
                 eval_transform: transforms.Compose):
        self.subset = subset
        self.eval_transform = eval_transform
 
    def __len__(self):
        return len(self.subset)
 
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        image = self.eval_transform(image)
        return image, label
 
 
#updated get_dataloaders using the wrapper
def get_dataloaders(
    root: Path = DATASET_ROOT,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_fraction: float = VALID_SPLIT,
    use_weighted_sampler: bool = True,
    seed: int = SEED,
    experiment: str = "baseline",
    #Create train, val, and test DataLoaders.
) -> dict[str, DataLoader]:
   
    # returns a dict with 'train', 'val', and 'test' DataLoaders.
 
    # arguments:
            #experiment: One of 'baseline', 'fft', 'gabor'.
            #controls which augmentations are applied during training.
   
    print(f"  Using '{experiment}' augmentation profile")
    full_train = UkiyoeDataset(root, split="train",
                               transform=get_train_transforms(img_size, experiment))
    train_subset, val_subset = split_train_val(full_train, val_fraction, seed)
 
    # wrap val subset with eval transforms
    val_dataset = _ValTransformSubset(val_subset, get_eval_transforms(img_size))
 
    test_set = UkiyoeDataset(root, split="test",
                             transform=get_eval_transforms(img_size))
 
    train_sampler = (make_weighted_sampler(train_subset)
                     if use_weighted_sampler else None)
 
    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
 
    return loaders
 
 

# 8.sanity-check & visualization
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(UKIYOE_MEAN).view(3, 1, 1)
    std  = torch.tensor(UKIYOE_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
 
 
def show_batch(loader: DataLoader, n: int = 8):
    """Display a grid of images from a DataLoader."""
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]
 
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i] if n > 1 else axes
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title("Human" if lbl == 0 else "AI", fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample batch", fontsize=13)
    plt.tight_layout()
    plt.savefig("sample_batch.png", dpi=150)
    plt.show()
    print("Saved sample_batch.png")
 
 

# 9.Exuctute
if __name__ == "__main__":
    print("=" * 60)
    print("Ukiyo-e Dataset  –  Sanity Check")
    print("=" * 60)
 
    loaders = get_dataloaders()
 
    #print shapes & label distribution
    for split_name, loader in loaders.items():
        imgs, lbls = next(iter(loader))
        print(f"\n{split_name}: batch shape = {imgs.shape}, "
              f"labels = {lbls.tolist()[:8]}...")
 
    #show a sample batch from the training set
    show_batch(loaders["train"])