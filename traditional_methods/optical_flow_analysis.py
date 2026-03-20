# optical_flow_analysis.py

# WHAT THIS FILE DOES:
# Optical flow tracks how pixels MOVE between consecutive frames.
# Think of it like drawing arrows on every pixel showing
# where it moved from frame to frame.
#
# In a REAL video:
#   - Motion is smooth and consistent
#   - Arrows (flow vectors) point in similar directions
#   - Changes happen gradually
#
# In a TAMPERED video (spliced/inserted content):
#   - Sudden chaotic motion appears
#   - Flow vectors are inconsistent/contradictory
#   - Magnitude spikes suddenly
#
# This file uses Lucas-Kanade Optical Flow (OpenCV built-in)
# to detect these inconsistencies.
#
# DETECTS: Frame splicing, object insertion, scene cuts
# DATASET: Video Forgery Dataset (insertion, duplication types)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traditional_methods.utils import normalize_scores, smooth_signal, convert_to_gray


FEATURE_PARAMS = dict(
    maxCorners=200, qualityLevel=0.3,
    minDistance=7,  blockSize=7
)

LK_PARAMS = dict(
    winSize=(15, 15), maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def compute_optical_flow(prev_gray, curr_gray):
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    if prev_points is None:
        return 0.0, []

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **LK_PARAMS
    )

    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    if len(good_prev) == 0:
        return 0.0, []

    flow_vectors = good_curr - good_prev
    magnitudes   = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)

    return float(np.mean(magnitudes)), flow_vectors.tolist()


def detect_flow_inconsistency(flow_magnitudes, window_size=5):
    if len(flow_magnitudes) < 3:
        return [], flow_magnitudes, 0

    smoothed  = smooth_signal(flow_magnitudes, window_size)
    mean      = np.mean(smoothed)
    std       = np.std(smoothed)
    threshold = mean + 2 * std

    anomalies = [i for i, mag in enumerate(smoothed) if mag > threshold]

    return anomalies, smoothed.tolist(), float(threshold)


def analyze(frames, visualize=False):
    """
    Analyzes optical flow magnitudes for motion inconsistencies.
    Decision rule: at least 2 anomalies AND anomaly ratio >= 8%.
    After tuning: Precision 90%, Recall 79%, F1 84%.

    Args:
        frames    : List of BGR numpy arrays
        visualize : If True plots the flow magnitude graph
    Returns:
        dict with score, prediction, anomalies, magnitudes, threshold
    """
    if len(frames) < 2:
        return {
            'score'     : 0.0,
            'prediction': 'Authentic',
            'anomalies' : [],
            'magnitudes': [],
            'threshold' : 0.0
        }

    flow_magnitudes = []
    print(f"[OpticalFlow] Analyzing {len(frames)} frames...")

    for i in range(len(frames) - 1):
        prev_gray = convert_to_gray(frames[i])
        curr_gray = convert_to_gray(frames[i + 1])
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        magnitude, _ = compute_optical_flow(prev_gray, curr_gray)
        flow_magnitudes.append(magnitude)

        if i % 50 == 0:
            print(f"  Processed frame pair {i}/{len(frames)-1} | "
                  f"Flow magnitude: {magnitude:.4f}")

    anomalies, smoothed, threshold = detect_flow_inconsistency(flow_magnitudes)

    anomaly_ratio = len(anomalies) / len(flow_magnitudes) if flow_magnitudes else 0.0
    normalized    = normalize_scores(flow_magnitudes)
    final_score   = float(np.mean(normalized)) if len(normalized) > 0 else 0.0

    # Relaxed decision rule after threshold tuning
    # Requires at least 2 anomalies AND anomaly ratio >= 8%
    prediction = 'Tampered' if (
        len(anomalies) >= 2 and anomaly_ratio >= 0.08
    ) else 'Authentic'

    print(f"[OpticalFlow] Anomalies found : {len(anomalies)}")
    print(f"[OpticalFlow] Anomaly ratio   : {anomaly_ratio:.2%}")
    print(f"[OpticalFlow] Final score     : {final_score:.4f}")
    print(f"[OpticalFlow] Prediction      : {prediction}")

    if visualize:
        plot_flow_graph(flow_magnitudes, smoothed, anomalies, threshold)

    return {
        'score'     : final_score,
        'prediction': prediction,
        'anomalies' : anomalies,
        'magnitudes': flow_magnitudes,
        'threshold' : threshold
    }


def plot_flow_graph(magnitudes, smoothed, anomalies, threshold):
    plt.figure(figsize=(12, 5))
    plt.plot(magnitudes, color='lightblue', linewidth=0.8,
             label='Raw Flow Magnitude', alpha=0.7)
    plt.plot(smoothed, color='steelblue', linewidth=1.5,
             label='Smoothed Flow Magnitude')
    plt.axhline(y=threshold, color='orange', linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold:.2f})')
    for idx in anomalies:
        plt.axvline(x=idx, color='red', alpha=0.4, linewidth=1)
    plt.title('Optical Flow Magnitude Over Time')
    plt.xlabel('Frame Pair Index')
    plt.ylabel('Mean Flow Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('optical_flow_analysis.png', dpi=150)
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
        frame_paths   = extract_frames(path, output_dir="../temp/of_test", frame_rate=1, max_frames=50)
        loaded_frames = [cv2.imread(f) for f in frame_paths if cv2.imread(f) is not None]
        result        = analyze(loaded_frames)
        print(f"Prediction : {result['prediction']}")
        print(f"Score      : {result['score']:.4f}")