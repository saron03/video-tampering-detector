# modern_methods/utils.py

# WHAT THIS FILE DOES:
# Handles everything related to data loading for the CNN.
#
# RESPONSIBILITIES:
#   1. Walk through FF++ dataset folders recursively
#   2. Collect all image paths with their labels (0=real, 1=fake)
#   3. Apply image transformations (resize, normalize, augment)
#   4. Split data into train/val/test sets
#   5. Create PyTorch DataLoaders for efficient batch loading
#
# WHY THIS IS SEPARATE FROM cnn_model.py:
#   Data loading logic is completely independent of model
#   architecture. Keeping them separate makes both cleaner
#   and easier to modify independently.
#
# DATASET STRUCTURE IT HANDLES:
#   FF++ fake/
#     ├── Deepfakes/
#     │   ├── 000_003/
#     │   │   ├── frame001.png
#     │   │   └── ...
#     │   └── ...
#     ├── Face2Face/
#     └── ...
#   FF++ real/
#     ├── 000/
#     │   ├── frame001.png
#     │   └── ...
#     └── ...

import os
import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FF_FAKE_DIR, FF_REAL_DIR,
    FF_FAKE_SUBDIRS,
    MAX_FAKE_SAMPLES, MAX_REAL_SAMPLES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, NUM_WORKERS,
    CNN_INPUT_SIZE,
    DEVICE
)



# IMAGE TRANSFORMS

# Training transforms — includes augmentation
# Augmentation = randomly flip, rotate, adjust brightness
# This makes the model more robust and reduces overfitting
TRAIN_TRANSFORMS = transforms.Compose([
    # Resize to CNN input size (224x224)
    transforms.Resize(CNN_INPUT_SIZE),

    # Randomly flip horizontally — fake faces are still fake mirrored
    transforms.RandomHorizontalFlip(p=0.5),

    # Randomly adjust brightness/contrast/saturation
    # Helps model not rely on lighting conditions
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    # Randomly rotate up to 10 degrees
    transforms.RandomRotation(degrees=10),

    # Convert PIL Image to PyTorch tensor (0-255 → 0.0-1.0)
    transforms.ToTensor(),

    # Normalize with ImageNet mean and std
    # Required because EfficientNet was pretrained on ImageNet
    # These are the standard values used for ImageNet training
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

# Validation/Test transforms — NO augmentation
# We don't augment during evaluation, just resize and normalize
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(CNN_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

# Inference transform — for single image prediction at runtime
# Same as val transforms
INFERENCE_TRANSFORMS = VAL_TRANSFORMS


# FUNCTION 1 — Collect all image paths
def collect_image_paths(fake_dir, real_dir,
                        max_fake=None, max_real=None):
    """
    Recursively walks through fake/ and real/ directories
    and collects all image file paths with their labels.

    Label 1 = fake (tampered)
    Label 0 = real (authentic)

    Args:
        fake_dir : Path to fake frames directory
        real_dir : Path to real frames directory
        max_fake : Max fake images to collect (None = all)
        max_real : Max real images to collect (None = all)
    Returns:
        List of (image_path, label) tuples
        e.g. [('/path/to/fake.png', 1), ('/path/to/real.png', 0)]
    """
    supported = ('.png', '.jpg', '.jpeg')
    fake_paths = []
    real_paths = []

    # Collect fake images 
    print(f"[DataLoader] Scanning fake directory: {fake_dir}")
    fake_dir = Path(fake_dir)

    # Walk ALL subdirectories recursively
    # fake/ → Deepfakes/ → 000_003/ → *.png
    for img_path in fake_dir.rglob('*'):
        if img_path.suffix.lower() in supported:
            fake_paths.append(str(img_path))

    print(f"[DataLoader] Found {len(fake_paths)} fake images")

    # Collect real images 
    print(f"[DataLoader] Scanning real directory: {real_dir}")
    real_dir = Path(real_dir)

    for img_path in real_dir.rglob('*'):
        if img_path.suffix.lower() in supported:
            real_paths.append(str(img_path))

    print(f"[DataLoader] Found {len(real_paths)} real images")

    #  Apply limits 
    # Shuffle before limiting so we get diverse samples
    random.shuffle(fake_paths)
    random.shuffle(real_paths)

    if max_fake and len(fake_paths) > max_fake:
        fake_paths = fake_paths[:max_fake]
        print(f"[DataLoader] Limited fake to {max_fake} images")

    if max_real and len(real_paths) > max_real:
        real_paths = real_paths[:max_real]
        print(f"[DataLoader] Limited real to {max_real} images")

    #  Combine with labels 
    # Label 1 = fake, Label 0 = real
    all_data = (
        [(path, 1) for path in fake_paths] +
        [(path, 0) for path in real_paths]
    )

    # Shuffle combined data
    random.shuffle(all_data)

    print(f"[DataLoader] Total dataset size: {len(all_data)} images")
    print(f"  Fake : {len(fake_paths)}")
    print(f"  Real : {len(real_paths)}")

    return all_data



# CLASS — PyTorch Dataset
class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.

    PyTorch needs data in a specific format:
      - A class with __len__() and __getitem__() methods
      - __len__ returns total number of samples
      - __getitem__ returns (image_tensor, label) for index i

    This class handles loading, converting, and transforming
    individual images on demand (lazy loading).

    Args:
        data      : List of (image_path, label) tuples
        transform : Torchvision transforms to apply
    """

    def __init__(self, data, transform=None):
        self.data      = data
        self.transform = transform

    def __len__(self):
        # Total number of images in this dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return one image and its label.
        Called automatically by DataLoader during training.

        Args:
            idx : Index of the image to load
        Returns:
            (image_tensor, label) tuple
        """
        img_path, label = self.data[idx]

        # Load image using PIL
        # PIL is required by torchvision transforms
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', CNN_INPUT_SIZE, color=0)

        # Apply transforms (resize, normalize, augment etc.)
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label



# FUNCTION 2 — Create train/val/test splits

def create_data_loaders(fake_dir=FF_FAKE_DIR,
                        real_dir=FF_REAL_DIR,
                        max_fake=MAX_FAKE_SAMPLES,
                        max_real=MAX_REAL_SAMPLES,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS):
    """
    Full pipeline: collect paths → split → create DataLoaders.

    DataLoaders handle:
      - Batching (grouping images into batches of 32)
      - Shuffling (randomizing order each epoch)
      - Parallel loading (multiple CPU workers load images)

    Args:
        fake_dir    : Path to fake frames
        real_dir    : Path to real frames
        max_fake    : Max fake samples
        max_real    : Max real samples
        batch_size  : Images per batch
        num_workers : Parallel loading threads
    Returns:
        train_loader, val_loader, test_loader
    """

    # Step 1 — Collect all image paths
    all_data = collect_image_paths(
        fake_dir, real_dir,
        max_fake, max_real
    )

    total = len(all_data)

    # Step 2 — Calculate split sizes
    train_size = int(TRAIN_RATIO * total)   # 70%
    val_size   = int(VAL_RATIO   * total)   # 15%
    test_size  = total - train_size - val_size  # 15% (remainder)

    print(f"\n[DataLoader] Split sizes:")
    print(f"  Train : {train_size}")
    print(f"  Val   : {val_size}")
    print(f"  Test  : {test_size}")

    # Step 3 — Split the data list
    train_data = all_data[:train_size]
    val_data   = all_data[train_size:train_size + val_size]
    test_data  = all_data[train_size + val_size:]

    # Step 4 — Create Dataset objects with appropriate transforms
    # Training gets augmentation, val/test do not
    train_dataset = DeepfakeDataset(train_data, TRAIN_TRANSFORMS)
    val_dataset   = DeepfakeDataset(val_data,   VAL_TRANSFORMS)
    test_dataset  = DeepfakeDataset(test_data,  VAL_TRANSFORMS)

    # Step 5 — Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle every epoch
        num_workers=num_workers,
        pin_memory=True         # faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # no need to shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n[DataLoader] DataLoaders created successfully")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader



# FUNCTION 3 — Load single image for inference

def load_single_image(image_path):
    """
    Loads and transforms a single image for CNN inference.
    Used by evaluate_cnn.py when predicting on user uploads.

    Args:
        image_path : Path to image file
    Returns:
        image_tensor : Shape (1, 3, 224, 224) ready for CNN
    """
    image = Image.open(image_path).convert('RGB')
    tensor = INFERENCE_TRANSFORMS(image)

    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# QUICK TEST

if __name__ == "__main__":
    print("=== Testing modern_methods/utils.py ===\n")

    # Test with small sample to verify everything works
    train_loader, val_loader, test_loader = create_data_loaders(
        max_fake=100,       # just 100 images for quick test
        max_real=100,
        batch_size=8,
        num_workers=0       # 0 workers for testing (avoids multiprocessing issues)
    )

    # Test loading one batch
    print("\n[Test] Loading one batch...")
    images, labels = next(iter(train_loader))

    print(f"  Batch image shape : {images.shape}")
    print(f"  Batch label shape : {labels.shape}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Label values      : {labels[:8].tolist()}")
    print(f"  Image min/max     : {images.min():.3f} / {images.max():.3f}")

    print("\n=== utils.py test passed ✅ ===")