# train_cnn.py
# Trains the DeepfakeDetector model on the FF++ dataset.
# Saves the best model weights based on validation loss.

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE,
    MODEL_PATH, RESULTS_DIR,
    MAX_FAKE_SAMPLES, MAX_REAL_SAMPLES
)
from modern_methods.cnn_model import get_model, save_model
from modern_methods.utils import create_data_loaders


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Runs one full pass through the training data.
    Updates model weights after each batch.

    Args:
        model     : DeepfakeDetector instance
        loader    : Training DataLoader
        optimizer : Adam optimizer
        criterion : CrossEntropyLoss
    Returns:
        avg_loss : Average loss for this epoch
        accuracy : Training accuracy for this epoch
    """
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # tqdm wraps the loader to show a progress bar
    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:
        # Move data to GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Zero gradients from previous batch
        # If we don't do this gradients accumulate incorrectly
        optimizer.zero_grad()

        # Forward pass: get model predictions
        outputs = model(images)

        # Calculate loss: how wrong were the predictions?
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients
        loss.backward()

        # Update weights using gradients
        optimizer.step()

        # Track metrics
        total_loss    += loss.item()
        predicted      = outputs.argmax(dim=1)
        correct       += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples * 100

    return avg_loss, accuracy


def validate(model, loader, criterion):
    """
    Evaluates model on validation data.
    No weight updates happen here.

    Args:
        model     : DeepfakeDetector instance
        loader    : Validation DataLoader
        criterion : CrossEntropyLoss
    Returns:
        avg_loss : Average validation loss
        accuracy : Validation accuracy
    """
    # eval mode disables dropout and batch norm updates
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # torch.no_grad() disables gradient calculation
    # Saves memory and speeds up validation
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item()
            predicted      = outputs.argmax(dim=1)
            correct       += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples * 100

    return avg_loss, accuracy


def train(num_epochs=NUM_EPOCHS, max_fake=MAX_FAKE_SAMPLES, max_real=MAX_REAL_SAMPLES):
    """
    Full training loop.
    Trains for num_epochs, saves best model based on val loss.

    Args:
        num_epochs : How many epochs to train
        max_fake   : Max fake samples to use
        max_real   : Max real samples to use
    """
    print("=== Starting CNN Training ===\n")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {num_epochs}")
    print(f"Fake samples : {max_fake}")
    print(f"Real samples : {max_real}\n")

    # Step 1: Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        max_fake=max_fake,
        max_real=max_real
    )

    # Step 2: Create model
    model = get_model(freeze_backbone=False)

    # Step 3: Define loss function
    # CrossEntropyLoss is standard for classification
    # It combines softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss()

    # Step 4: Define optimizer
    # Adam is the most popular optimizer, works well out of the box
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Step 5: Learning rate scheduler
    # Reduces learning rate when validation loss stops improving
    # This helps the model fine-tune in later epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,
        factor=0.5
    )

    # Step 6: Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}   | Val Acc   : {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, MODEL_PATH)
            print(f"New best model saved! Val loss: {val_loss:.4f}")

    print("\n=== Training Complete ===")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    # Quick test run with small dataset
    # Change max_fake and max_real to larger numbers for real training
    # Recommended for full training: max_fake=10000, max_real=10000
    train(
    num_epochs=10,
    max_fake=10000,
    max_real=10000
    )