# evaluate_cnn.py
# Tests the trained CNN on unseen data and provides
# the predict() function used by the pipeline at runtime.

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, MODEL_PATH, RESULTS_DIR
from modern_methods.cnn_model import load_model
from modern_methods.utils import create_data_loaders, load_single_image


def evaluate_on_test_set(model, test_loader):
    """
    Runs the model on the test set and collects
    all predictions and true labels.

    Args:
        model       : Loaded DeepfakeDetector instance
        test_loader : Test DataLoader
    Returns:
        all_preds  : List of predicted labels
        all_labels : List of true labels
        all_probs  : List of fake probabilities
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            probs     = model.predict_proba(images)
            predicted = probs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # probs[:, 1] is the probability of being fake
            all_probs.extend(probs[:, 1].cpu().numpy())

    return all_preds, all_labels, all_probs


def predict_single_frame(model, image_path):
    """
    Predicts whether a single frame is real or fake.
    Used internally by predict_video().

    Args:
        model      : Loaded DeepfakeDetector instance
        image_path : Path to a single frame image
    Returns:
        fake_prob  : Probability of being fake (0.0 to 1.0)
        prediction : 'Tampered' or 'Authentic'
    """
    model.eval()

    with torch.no_grad():
        image_tensor = load_single_image(image_path)
        probs        = model.predict_proba(image_tensor)

        fake_prob  = float(probs[0, 1].item())
        prediction = 'Tampered' if fake_prob > 0.5 else 'Authentic'

    return fake_prob, prediction


def predict(video_frames_dir, model=None):
    """
    Main prediction function called by pipeline/detect_video.py
    at runtime when a user uploads a video.

    Takes a directory of extracted frames, runs CNN on each frame,
    averages the probabilities, returns final prediction.

    Args:
        video_frames_dir : Directory containing extracted frame images
        model            : Optional pre-loaded model (avoids reloading)
    Returns:
        dict with:
            score      : Average fake probability (0.0 to 1.0)
            prediction : 'Tampered' or 'Authentic'
            frame_probs: Per-frame fake probabilities
    """
    # Load model if not provided
    if model is None:
        model = load_model(MODEL_PATH)

    model.eval()

    # Get all frame image paths
    supported  = ('.png', '.jpg', '.jpeg')
    frame_paths = sorted([
        os.path.join(video_frames_dir, f)
        for f in os.listdir(video_frames_dir)
        if os.path.splitext(f)[1].lower() in supported
    ])

    if not frame_paths:
        return {
            'score':       0.0,
            'prediction':  'Authentic',
            'frame_probs': []
        }

    print(f"[CNN] Running inference on {len(frame_paths)} frames...")

    frame_probs = []

    for frame_path in tqdm(frame_paths, desc="CNN Inference", leave=False):
        fake_prob, _ = predict_single_frame(model, frame_path)
        frame_probs.append(fake_prob)

    # Average probability across all frames
    avg_prob   = float(np.mean(frame_probs))
    prediction = 'Tampered' if avg_prob > 0.5 else 'Authentic'

    print(f"[CNN] Average fake probability : {avg_prob:.4f}")
    print(f"[CNN] Prediction               : {prediction}")

    return {
        'score':       avg_prob,
        'prediction':  prediction,
        'frame_probs': frame_probs
    }


if __name__ == "__main__":
    print("=== Testing evaluate_cnn.py ===\n")

    # Load trained model
    model = load_model(MODEL_PATH)

    # Create data loaders to get test set
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(
        max_fake=1000,
        max_real=1000,
        num_workers=0
    )

    # Run evaluation
    print("\nRunning evaluation on test set...")
    preds, labels, probs = evaluate_on_test_set(model, test_loader)

    # Basic accuracy
    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean() * 100

    print(f"\n=== CNN Test Results ===")
    print(f"Test Accuracy : {acc:.2f}%")
    print(f"Total samples : {len(labels)}")
    print(f"Correct       : {(preds == labels).sum()}")
    print(f"Wrong         : {(preds != labels).sum()}")

    print("\n=== evaluate_cnn.py test passed ✅ ===")