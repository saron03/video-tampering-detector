# frame_correlation.py

# WHAT THIS FILE DOES:
# In a real, untampered video, consecutive frames look very
# similar to each other because motion is smooth and gradual.
# If someone DELETES frames (to remove an event) or INSERTS
# frames (to add fake content), there will be a sudden jump
# in similarity between consecutive frames.
#
# This file measures that similarity using two methods:
#   1. Mean Squared Error (MSE)   — pixel-level difference
#   2. SSIM (Structural Similarity) — perceptual similarity
#
# A sudden spike in MSE or drop in SSIM = possible tampering.
#
# DATASET USED: Video Forgery Dataset (Insertion, Deletion types)

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traditional_methods.utils import normalize_scores, smooth_signal


def compute_mse(frame1, frame2):
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    return float(np.mean((f1 - f2) ** 2))


def compute_ssim(frame1, frame2):
    score, _ = ssim(frame1, frame2, full=True)
    return float(score)


def analyze(frames, threshold_mse=3000.0, threshold_ssim=0.50, visualize=False):
    """
    Analyzes frames for temporal inconsistencies using MSE.
    Decision is based purely on MSE spikes above threshold_mse.
    A single spike above 5000 is treated as strong evidence of tampering.
    SSIM is computed and returned but does not affect the decision.

    Args:
        frames        : List of BGR numpy arrays
        threshold_mse : MSE above this flags a frame pair as anomalous
        threshold_ssim: Kept for compatibility, not used in decision
        visualize     : If True plots the MSE graph
    Returns:
        dict with score, prediction, anomalies, mse_values, ssim_values
    """
    if len(frames) < 2:
        return {
            'score'     : 0.0,
            'prediction': 'Authentic',
            'anomalies' : [],
            'mse_values': [],
            'ssim_values': []
        }

    mse_values  = []
    ssim_values = []
    anomalies   = []

    print(f"[FrameCorrelation] Analyzing {len(frames)} frames...")

    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i],     cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        mse        = compute_mse(gray1, gray2)
        ssim_score = compute_ssim(gray1, gray2)

        mse_values.append(mse)
        ssim_values.append(ssim_score)

        if mse > threshold_mse:
            anomalies.append(i)
            print(f"  ⚠ Anomaly at frame {i}-{i+1} | MSE: {mse:.2f} | SSIM: {ssim_score:.4f}")

    anomaly_ratio = len(anomalies) / len(mse_values) if mse_values else 0.0
    max_mse       = max(mse_values) if mse_values else 0

    normalized  = normalize_scores(mse_values)
    final_score = float(np.mean(normalized)) if len(normalized) > 0 else 0.0

    # Decision based on MSE spikes only
    # A single spike above 5000 is very strong evidence of tampering
    # High anomaly ratio also indicates tampering
    prediction = 'Tampered' if (
        max_mse > 5000 or anomaly_ratio > 0.40
    ) else 'Authentic'

    print(f"[FrameCorrelation] Anomaly ratio : {anomaly_ratio:.2%}")
    print(f"[FrameCorrelation] Max MSE       : {max_mse:.2f}")
    print(f"[FrameCorrelation] Final score   : {final_score:.4f}")
    print(f"[FrameCorrelation] Prediction    : {prediction}")

    if visualize:
        plot_correlation_graph(mse_values, ssim_values, anomalies)

    return {
        'score'      : final_score,
        'prediction' : prediction,
        'anomalies'  : anomalies,
        'mse_values' : mse_values,
        'ssim_values': ssim_values
    }


def plot_correlation_graph(mse_values, ssim_values, anomalies):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(mse_values, color='steelblue', linewidth=1, label='MSE')
    ax1.set_title('Frame-to-Frame MSE')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Frame Pair Index')
    for idx in anomalies:
        ax1.axvline(x=idx, color='red', alpha=0.4, linewidth=1)
    ax1.legend()

    ax2.plot(ssim_values, color='green', linewidth=1, label='SSIM')
    ax2.set_title('Frame-to-Frame SSIM')
    ax2.set_ylabel('SSIM Score')
    ax2.set_xlabel('Frame Pair Index')
    ax2.set_ylim(0, 1)
    for idx in anomalies:
        ax2.axvline(x=idx, color='red', alpha=0.4, linewidth=1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('frame_correlation_analysis.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from preprocessing.extract_frames import extract_frames

    for path, label in [
        ("/home/saron03/Downloads/archive (1)/Testing/Forged/Forgery_insertion/vid100.avi", "FORGED"),
        ("/home/saron03/Downloads/archive (1)/Testing/Original/vid1.avi", "ORIGINAL"),
    ]:
        print(f"\n=== Testing {label} ===")
        frame_paths   = extract_frames(path, output_dir="../temp/fc_test", frame_rate=1, max_frames=50)
        loaded_frames = [cv2.imread(f) for f in frame_paths if cv2.imread(f) is not None]
        result        = analyze(loaded_frames)
        print(f"Prediction : {result['prediction']}")
        print(f"Score      : {result['score']:.4f}")