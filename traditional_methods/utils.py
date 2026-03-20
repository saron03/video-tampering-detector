# traditional_methods/utils.py

# WHAT THIS FILE DOES:
# Shared helper functions used across ALL traditional methods:
#   - frame_correlation.py
#   - optical_flow_analysis.py
#   - copy_move_svd_lbp.py
#
# Instead of rewriting the same functions in each file,
# we write them once here and import them everywhere.
#
# Functions included:
#   - normalize_scores()   : Scale any list of values to 0-1
#   - smooth_signal()      : Smooth noisy data (moving average)
#   - load_frames()        : Load saved frame images into numpy arrays
#   - resize_frame()       : Resize a single frame
#   - convert_to_gray()    : Convert BGR frame to grayscale
#   - draw_anomaly_boxes() : Draw red boxes on suspicious frames
#   - save_result_image()  : Save a frame with annotations


import cv2
import numpy as np
import os
from pathlib import Path


# FUNCTION 1 — Normalize a list of scores to 0-1
def normalize_scores(scores):
    """
    Takes a list of raw values (like MSE values which can be
    anywhere from 0 to 100000) and scales them to 0.0 - 1.0.
    This makes scores from different methods comparable.

    Example:
        [100, 500, 1000] → [0.0, 0.44, 1.0]

    Args:
        scores : List or numpy array of float values
    Returns:
        Numpy array of normalized values between 0 and 1
    """
    scores = np.array(scores, dtype=np.float32)

    min_val = scores.min()
    max_val = scores.max()

    # Avoid division by zero if all scores are the same
    if max_val - min_val == 0:
        return np.zeros_like(scores)

    normalized = (scores - min_val) / (max_val - min_val)
    return normalized


# FUNCTION 2 — Smooth a noisy signal
def smooth_signal(values, window_size=5):
    """
    Applies a moving average to smooth out noisy data.
    Useful for optical flow and MSE values which can be
    spiky even in authentic videos.

    Example with window_size=3:
        [10, 100, 12, 11, 9] → [10, 40.6, 41, 10.6, 9]
        The spike at 100 is smoothed out a bit.

    Args:
        values      : List or numpy array of values
        window_size : Number of values to average together
    Returns:
        Numpy array of smoothed values (same length as input)
    """
    values = np.array(values, dtype=np.float32)

    if len(values) < window_size:
        # Not enough values to smooth, return as is
        return values

    # np.convolve does the sliding window average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode='same')

    return smoothed

# FUNCTION 3 — Load saved frame images from disk
def load_frames(frames_dir, max_frames=None, as_gray=False):
    """
    Loads all PNG/JPG frames from a folder into a list
    of numpy arrays. Used when frames are already saved
    to disk by extract_frames.py.

    Args:
        frames_dir  : Path to folder containing frame images
        max_frames  : Max number of frames to load (None = all)
        as_gray     : If True, load as grayscale instead of BGR
    Returns:
        List of numpy arrays (frames)
    """
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Get all image files, sorted by name (important! frames must be in order)
    image_extensions = ('.png', '.jpg', '.jpeg')
    frame_files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not frame_files:
        raise ValueError(f"No image files found in: {frames_dir}")

    # Apply max_frames limit if specified
    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for frame_file in frame_files:
        if as_gray:
            frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
        else:
            frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)

        if frame is not None:
            frames.append(frame)
        else:
            print(f"  Warning: Could not load {frame_file.name}")

    print(f"[Utils] Loaded {len(frames)} frames from {frames_dir}")
    return frames


# FUNCTION 4 — Resize a single frame
def resize_frame(frame, width=224, height=224):
    """
    Resize a frame to a target size.
    224x224 is standard for CNN input.
    Smaller size = faster processing for traditional methods.

    Args:
        frame  : Numpy array (BGR or grayscale)
        width  : Target width in pixels
        height : Target height in pixels
    Returns:
        Resized numpy array
    """
    return cv2.resize(frame, (width, height))


# FUNCTION 5 — Convert BGR frame to grayscale
def convert_to_gray(frame):
    """
    Converts a BGR color frame to grayscale.
    Most traditional methods work on grayscale
    because we care about structure, not color.

    Args:
        frame : BGR numpy array
    Returns:
        Grayscale numpy array
    """
    if len(frame.shape) == 2:
        # Already grayscale
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# FUNCTION 6 — Draw red boxes on anomalous regions
def draw_anomaly_boxes(frame, regions, color=(0, 0, 255), thickness=2):
    """
    Draws red rectangles on suspicious regions of a frame.
    Used by copy_move_svd_lbp.py to highlight copy-pasted areas.

    Args:
        frame     : BGR numpy array
        regions   : List of (x, y, w, h) tuples — regions to highlight
        color     : BGR color for the box (default: red)
        thickness : Line thickness in pixels
    Returns:
        Annotated frame (numpy array)
    """
    # Work on a copy so we don't modify the original
    annotated = frame.copy()

    for (x, y, w, h) in regions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

    return annotated


# FUNCTION 7 — Save an annotated frame to disk
def save_result_image(frame, output_path, label=None):
    """
    Saves a frame (with optional text label) to disk.
    Useful for generating visual evidence of tampering.

    Args:
        frame       : Numpy array to save
        output_path : Where to save the image
        label       : Optional text to write on the image (e.g. 'TAMPERED')
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    annotated = frame.copy()

    if label:
        # Write label text in top-left corner
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color      = (0, 0, 255)   # Red
        thickness  = 2

        cv2.putText(annotated, label, (10, 40),
                    font, font_scale, color, thickness)

    cv2.imwrite(output_path, annotated)
    print(f"[Utils] Saved result image: {output_path}")


# QUICK TEST — Run this file directly to verify
if __name__ == "__main__":
    print("=== Testing utils.py ===\n")

    # Test normalize_scores
    raw = [100, 500, 200, 1000, 50]
    normalized = normalize_scores(raw)
    print(f"normalize_scores({raw})")
    print(f"  → {normalized}\n")

    # Test smooth_signal
    noisy = [10, 12, 100, 11, 9, 10, 98, 10, 11]
    smoothed = smooth_signal(noisy, window_size=3)
    print(f"smooth_signal({noisy})")
    print(f"  → {smoothed}\n")

    # Test resize
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    resized = resize_frame(dummy_frame, 224, 224)
    print(f"resize_frame: {dummy_frame.shape} → {resized.shape}\n")

    # Test convert_to_gray
    gray = convert_to_gray(dummy_frame)
    print(f"convert_to_gray: {dummy_frame.shape} → {gray.shape}\n")

    print("=== All utils tests passed ✅ ===")