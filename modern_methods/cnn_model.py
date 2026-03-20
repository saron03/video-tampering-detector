
# modern_methods/cnn_model.py
# WHAT THIS FILE DOES:
# Defines the CNN architecture for deepfake detection.
#
# We use Transfer Learning with EfficientNet-B0.
# Transfer Learning means we take a model already trained
# on millions of images (ImageNet) and fine-tune it for
# our specific task (real vs fake face detection).
#
# WHY EfficientNet-B0:
#   - Small and fast, good for your GPU
#   - Very accurate for binary classification
#   - Already understands edges, textures, faces
#   - Much better than training from scratch
#
# ARCHITECTURE:
#   Input image (224x224x3)
#       ↓
#   EfficientNet-B0 backbone (pretrained, frozen initially)
#       ↓
#   Custom classifier head
#       ↓
#   Output: [real_prob, fake_prob]


import torch
import torch.nn as nn
from torchvision import models
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, DEVICE, MODEL_PATH


# MAIN MODEL CLASS

class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model based on EfficientNet-B0.

    nn.Module is the base class for all PyTorch models.
    Every model must inherit from it and implement:
        __init__  : define layers
        forward   : define how data flows through layers
    """

    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=False):
        """
        Initialize the model.

        Args:
            num_classes     : Number of output classes (2 for real/fake)
            freeze_backbone : If True, only train the classifier head
                              If False, train the whole network
                              Start with False for better accuracy
        """
        super(DeepfakeDetector, self).__init__()

        # Load pretrained EfficientNet-B0
        # weights=DEFAULT means load ImageNet pretrained weights
        print("[Model] Loading pretrained EfficientNet-B0...")
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Optionally freeze backbone layers
        # Freezing means those weights won't change during training
        # Useful if you have very little data
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[Model] Backbone frozen, only classifier will train")

        # Get the number of features output by EfficientNet backbone
        # EfficientNet-B0 outputs 1280 features before its classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the default EfficientNet classifier with our own
        # Original: Linear(1280, 1000) for ImageNet 1000 classes
        # Ours    : Custom head for binary classification
        self.backbone.classifier = nn.Sequential(

            # Dropout randomly zeros out neurons during training
            # Prevents overfitting (memorizing training data)
            nn.Dropout(p=0.3, inplace=True),

            # First fully connected layer
            # Reduces 1280 features to 256
            nn.Linear(in_features, 256),

            # ReLU activation function
            # Adds non-linearity so model can learn complex patterns
            nn.ReLU(),

            # Another dropout layer
            nn.Dropout(p=0.2),

            # Final output layer
            # 256 features → num_classes (2) outputs
            # Output is raw scores (logits), not probabilities yet
            # Softmax is applied later during prediction
            nn.Linear(256, num_classes)
        )

        print(f"[Model] Classifier head replaced for {num_classes} classes")
        print(f"[Model] Model ready on device: {DEVICE}")

    def forward(self, x):
        """
        Defines forward pass — how input flows through the model.
        PyTorch calls this automatically during training/inference.

        Args:
            x : Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            logits : Raw output scores shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def predict_proba(self, x):
        """
        Returns probability scores instead of raw logits.
        Used during inference when we want a confidence score.

        Args:
            x : Input tensor (batch_size, 3, 224, 224)
        Returns:
            probs : Probability tensor (batch_size, num_classes)
                    probs[:, 0] = probability of real
                    probs[:, 1] = probability of fake
        """
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=1)
        return probs


# HELPER FUNCTIONS

def get_model(freeze_backbone=False):
    """
    Creates and returns the model on the correct device.
    This is the main function called by train_cnn.py
    and evaluate_cnn.py.

    Args:
        freeze_backbone : Whether to freeze backbone layers
    Returns:
        model : DeepfakeDetector instance on GPU/CPU
    """
    model = DeepfakeDetector(
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze_backbone
    )
    model = model.to(DEVICE)
    return model


def save_model(model, path=MODEL_PATH):
    """
    Saves model weights to disk.
    Only saves weights (state_dict), not the full model.
    This is the standard PyTorch practice.

    Args:
        model : Trained DeepfakeDetector instance
        path  : Where to save the weights file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Model] Weights saved to: {path}")


def load_model(path=MODEL_PATH):
    """
    Loads saved model weights from disk.
    Used by evaluate_cnn.py and pipeline at runtime.

    Args:
        path  : Path to saved weights file (.pth)
    Returns:
        model : DeepfakeDetector with loaded weights
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")

    model = DeepfakeDetector(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    print(f"[Model] Weights loaded from: {path}")
    return model


def count_parameters(model):
    """
    Counts total and trainable parameters in the model.
    Useful to understand model size and what's being trained.

    Args:
        model : DeepfakeDetector instance
    Returns:
        total, trainable parameter counts
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# QUICK TEST
if __name__ == "__main__":
    print("=== Testing cnn_model.py ===\n")

    # Create model
    model = get_model(freeze_backbone=False)

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n[Model] Total parameters     : {total:,}")
    print(f"[Model] Trainable parameters : {trainable:,}")

    # Test forward pass with a dummy batch
    print("\n[Test] Running dummy forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)
    output      = model(dummy_input)
    probs       = model.predict_proba(dummy_input)

    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Probs shape  : {probs.shape}")
    print(f"  Sample probs : {probs[0].tolist()}")
    print(f"  Probs sum    : {probs[0].sum().item():.4f} (should be 1.0)")

    print("\n=== cnn_model.py test passed ✅ ===")