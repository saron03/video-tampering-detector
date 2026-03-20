# copy_move_svd_lbp.py

# WHAT THIS FILE DOES:
# Detects copy-move forgery WITHIN individual frames.
# Someone copied a region of the frame and pasted it
# elsewhere in the same frame to hide/replace content.
#
# TWO DETECTION TECHNIQUES:
#
# 1. SVD (Singular Value Decomposition)
#    - Divides frame into small blocks (e.g. 16x16 pixels)
#    - Computes SVD fingerprint for each block
#    - Finds blocks with suspiciously similar fingerprints
#    - Similar blocks in different locations = copy-move
#
# 2. LBP (Local Binary Pattern)
#    - Computes texture fingerprint for each block
#    - Compares texture histograms across blocks
#    - Matching textures in different locations = copy-move
#
# DETECTS: Copy-paste forgery, region cloning, content hiding
# DATASET: Video Forgery Dataset (duplication type)

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os
import sys
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traditional_methods.utils import (
    normalize_scores,
    convert_to_gray,
    draw_anomaly_boxes,
    save_result_image
)


# PARAMETERS

# Block size for dividing the frame into patches
# Smaller = more precise but slower
# 16x16 is a good balance
BLOCK_SIZE = 16

# LBP parameters
# radius  : how far to look for neighbors
# n_points: number of neighbors to sample
LBP_RADIUS   = 1
LBP_N_POINTS = 8 * LBP_RADIUS

# Similarity threshold
# How similar two blocks must be to count as copy-move
# Lower = more strict (fewer false positives)
# Higher = more sensitive (more detections)
SVD_THRESHOLD = 0.95
LBP_THRESHOLD = 0.90



# HELPER — Divide frame into blocks

def get_blocks(gray_frame, block_size=BLOCK_SIZE):
    """
    Divides a grayscale frame into non-overlapping blocks.
    Returns blocks with their (row, col) positions.

    Args:
        gray_frame : Grayscale numpy array
        block_size : Size of each square block in pixels
    Returns:
        List of (block, row, col) tuples
    """
    blocks = []
    h, w = gray_frame.shape

    for row in range(0, h - block_size, block_size):
        for col in range(0, w - block_size, block_size):
            # Extract this block from the frame
            block = gray_frame[row:row+block_size, col:col+block_size]
            blocks.append((block, row, col))

    return blocks


# TECHNIQUE 1 — SVD Fingerprint

def compute_svd_fingerprint(block):
    """
    Computes SVD fingerprint for a single block.
    SVD decomposes the block matrix into components.
    The singular values (S) act as a unique fingerprint.

    Two blocks with very similar S values are likely
    copies of each other.

    Args:
        block : 2D numpy array (grayscale block)
    Returns:
        Normalized singular values as fingerprint vector
    """
    block = block.astype(np.float32)

    # SVD decomposition: block = U * S * V^T
    # We only care about S (singular values)
    _, S, _ = np.linalg.svd(block, full_matrices=False)

    # Normalize so we can compare across blocks
    norm = np.linalg.norm(S)
    if norm == 0:
        return S
    return S / norm


def svd_similarity(fp1, fp2):
    """
    Computes cosine similarity between two SVD fingerprints.
    Range: 0 to 1. Higher = more similar.

    Args:
        fp1, fp2 : SVD fingerprint vectors
    Returns:
        Float similarity score
    """
    # Cosine similarity = dot product of normalized vectors
    return float(np.dot(fp1, fp2))


# TECHNIQUE 2 — LBP Texture Fingerprint

def compute_lbp_fingerprint(block):
    """
    Computes LBP (Local Binary Pattern) texture fingerprint.

    LBP works by:
    1. For each pixel, compare it to its N neighbors
    2. If neighbor > center pixel → 1, else → 0
    3. This gives an 8-bit binary number per pixel
    4. Build histogram of these numbers = texture fingerprint

    Args:
        block : 2D numpy array (grayscale block)
    Returns:
        Normalized histogram as fingerprint vector
    """
    # Compute LBP for the block
    lbp = local_binary_pattern(
        block,
        LBP_N_POINTS,
        LBP_RADIUS,
        method='uniform'
    )

    # Build histogram of LBP values
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))

    # Normalize histogram
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm == 0:
        return hist
    return hist / norm


def lbp_similarity(fp1, fp2):
    """
    Computes similarity between two LBP histograms
    using correlation method.
    Range: 0 to 1. Higher = more similar.

    Args:
        fp1, fp2 : LBP histogram vectors
    Returns:
        Float similarity score
    """
    # Reshape for OpenCV compareHist
    h1 = fp1.reshape(-1, 1).astype(np.float32)
    h2 = fp2.reshape(-1, 1).astype(np.float32)

    # HISTCMP_CORREL returns -1 to 1, we shift to 0-1
    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float((score + 1) / 2)


# CORE — Detect copy-move in a single frame

def detect_copy_move_in_frame(frame, use_svd=True, use_lbp=True):
    """
    Runs copy-move detection on a single frame.
    Divides into blocks, computes fingerprints,
    compares all block pairs, flags suspicious matches.

    Args:
        frame   : BGR numpy array
        use_svd : Whether to use SVD method
        use_lbp : Whether to use LBP method
    Returns:
        dict with:
            - 'score'           : Anomaly score 0-1
            - 'suspicious_pairs': List of ((r1,c1),(r2,c2)) matched block pairs
            - 'annotated_frame' : Frame with boxes drawn on suspicious regions
    """
    gray = convert_to_gray(frame)

    # Resize for consistent processing
    # (smaller = faster, 256x256 is enough for block analysis)
    gray = cv2.resize(gray, (256, 256))

    # Get all blocks
    blocks = get_blocks(gray, BLOCK_SIZE)

    if len(blocks) < 2:
        return {'score': 0.0, 'suspicious_pairs': [], 'annotated_frame': frame}

    # Compute fingerprints for all blocks
    svd_fingerprints = []
    lbp_fingerprints = []

    for block, row, col in blocks:
        if use_svd:
            svd_fingerprints.append(compute_svd_fingerprint(block))
        if use_lbp:
            lbp_fingerprints.append(compute_lbp_fingerprint(block))

    # Compare ALL pairs of blocks
    suspicious_pairs = []
    n = len(blocks)

    for i, j in combinations(range(n), 2):
        _, r1, c1 = blocks[i]
        _, r2, c2 = blocks[j]

        # Skip adjacent blocks — they're naturally similar
        if abs(r1 - r2) <= BLOCK_SIZE and abs(c1 - c2) <= BLOCK_SIZE:
            continue

        svd_match = False
        lbp_match = False

        # Check SVD similarity
        if use_svd:
            sim = svd_similarity(svd_fingerprints[i], svd_fingerprints[j])
            if sim >= SVD_THRESHOLD:
                svd_match = True

        # Check LBP similarity
        if use_lbp:
            sim = lbp_similarity(lbp_fingerprints[i], lbp_fingerprints[j])
            if sim >= LBP_THRESHOLD:
                lbp_match = True

        # Both methods must agree (reduces false positives)
        if use_svd and use_lbp:
            if svd_match and lbp_match:
                suspicious_pairs.append(((r1, c1), (r2, c2)))
        elif use_svd and svd_match:
            suspicious_pairs.append(((r1, c1), (r2, c2)))
        elif use_lbp and lbp_match:
            suspicious_pairs.append(((r1, c1), (r2, c2)))

    # ── Calculate score ──────────────────────────────────────
    # Score = ratio of suspicious block pairs to total pairs
    total_pairs  = n * (n - 1) / 2
    score = len(suspicious_pairs) / total_pairs if total_pairs > 0 else 0.0

    # ── Draw boxes on suspicious regions ────────────────────
    # Scale coordinates back to original frame size
    scale_x = frame.shape[1] / 256
    scale_y = frame.shape[0] / 256

    regions = []
    for (r1, c1), (r2, c2) in suspicious_pairs[:20]:  # limit to 20 boxes
        x1 = int(c1 * scale_x)
        y1 = int(r1 * scale_y)
        x2 = int(c2 * scale_x)
        y2 = int(r2 * scale_y)
        bs = int(BLOCK_SIZE * scale_x)
        regions.extend([(x1, y1, bs, bs), (x2, y2, bs, bs)])

    annotated = draw_anomaly_boxes(frame, regions)

    return {
        'score':            score,
        'suspicious_pairs': suspicious_pairs,
        'annotated_frame':  annotated
    }



# MAIN ANALYSIS FUNCTION

def analyze(frames, sample_rate=10, visualize=False):
    """
    Runs copy-move detection across ALL frames of a video.
    We don't check every frame (too slow) — we sample
    every Nth frame instead.

    Args:
        frames      : List of BGR numpy arrays
        sample_rate : Check every Nth frame (default: 10)
        visualize   : If True, save annotated suspicious frames
    Returns:
        dict with:
            - 'score'      : Final anomaly score (0.0 to 1.0)
            - 'prediction' : 'Tampered' or 'Authentic'
            - 'anomalies'  : List of suspicious frame indices
            - 'frame_scores: Per-frame copy-move scores
    """

    if len(frames) == 0:
        return {
            'score':       0.0,
            'prediction':  'Authentic',
            'anomalies':   [],
            'frame_scores': []
        }

    frame_scores = []
    anomalies    = []

    # Sample every Nth frame to keep processing fast
    sampled_indices = list(range(0, len(frames), sample_rate))
    print(f"[CopyMove] Analyzing {len(sampled_indices)} sampled frames "
          f"(every {sample_rate} frames)...")

    for idx in sampled_indices:
        frame = frames[idx]

        result = detect_copy_move_in_frame(frame)
        score  = result['score']
        frame_scores.append(score)

        # Flag frame if score is above 0.01
        # (even a small number of matching blocks is suspicious)
        if score > 0.01:
            anomalies.append(idx)
            print(f"  ⚠ Copy-move detected at frame {idx} | "
                  f"Score: {score:.6f} | "
                  f"Pairs: {len(result['suspicious_pairs'])}")

            # Save annotated frame if visualize is on
            if visualize:
                save_result_image(
                    result['annotated_frame'],
                    output_path=f"../temp/copymove_frame_{idx}.png",
                    label=f"COPY-MOVE DETECTED (score={score:.4f})"
                )

    # ── Final score ──────────────────────────────────────────
    final_score = float(np.mean(frame_scores)) if frame_scores else 0.0

    # ── Prediction ───────────────────────────────────────────
    anomaly_ratio = len(anomalies) / len(sampled_indices) \
                    if sampled_indices else 0.0
    prediction = 'Tampered' if anomaly_ratio > 0.05 else 'Authentic'

    print(f"[CopyMove] Frames checked  : {len(sampled_indices)}")
    print(f"[CopyMove] Anomalies found : {len(anomalies)}")
    print(f"[CopyMove] Final score     : {final_score:.6f}")
    print(f"[CopyMove] Prediction      : {prediction}")

    return {
        'score':        final_score,
        'prediction':   prediction,
        'anomalies':    anomalies,
        'frame_scores': frame_scores
    }



# QUICK TEST

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from preprocessing.extract_frames import extract_frames

    print("=== Copy-Move Detection Test ===\n")

    # Test 1 — Forged video (duplication is best for copy-move)
    print("--- Testing FORGED video ---")
    TEST_VIDEO_FORGED = "/home/saron03/Downloads/archive (1)/Testing/Forged/Forgery_duplication/vid100.avi"

    frame_paths = extract_frames(
        TEST_VIDEO_FORGED,
        output_dir="../temp/cm_test_forged",
        frame_rate=1,
        max_frames=30       # less frames, copy-move is slower
    )
    loaded_forged = [
        cv2.imread(f) for f in frame_paths
        if cv2.imread(f) is not None
    ]
    print(f"Loaded {len(loaded_forged)} frames\n")
    result_forged = analyze(loaded_forged, sample_rate=5, visualize=False)

    print("\n=== FORGED RESULT ===")
    print(f"Prediction : {result_forged['prediction']}")
    print(f"Score      : {result_forged['score']:.6f}")
    print(f"Anomalies  : {len(result_forged['anomalies'])} suspicious frames")

    print("\n" + "="*50 + "\n")

    # Test 2 — Original video
    print("--- Testing ORIGINAL video ---")
    TEST_VIDEO_ORIGINAL = "/home/saron03/Downloads/archive (1)/Testing/Original/vid1.avi"

    frame_paths = extract_frames(
        TEST_VIDEO_ORIGINAL,
        output_dir="../temp/cm_test_original",
        frame_rate=1,
        max_frames=30
    )
    loaded_original = [
        cv2.imread(f) for f in frame_paths
        if cv2.imread(f) is not None
    ]
    print(f"Loaded {len(loaded_original)} frames\n")
    result_original = analyze(loaded_original, sample_rate=5, visualize=False)

    print("\n=== ORIGINAL RESULT ===")
    print(f"Prediction : {result_original['prediction']}")
    print(f"Score      : {result_original['score']:.6f}")
    print(f"Anomalies  : {len(result_original['anomalies'])} suspicious frames")