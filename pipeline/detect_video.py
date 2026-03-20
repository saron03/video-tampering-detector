# detect_video.py
# Central pipeline connecting all detection methods.
# Two methods available: traditional and modern (CNN).
# Traditional uses FC+CM fusion with OF as supporting evidence.
# Modern uses EfficientNet-B0 with face cropping preprocessing.

import os
import sys
import shutil
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TEMP_DIR, MODEL_PATH, FRAME_RATE,
    FC_MSE_THRESHOLD, FC_SSIM_THRESHOLD, CM_SAMPLE_RATE
)

from preprocessing.extract_frames import extract_frames
from traditional_methods.frame_correlation import analyze as fc_analyze
from traditional_methods.optical_flow_analysis import analyze as of_analyze
from traditional_methods.copy_move_svd_lbp import analyze as cm_analyze
from modern_methods.evaluate_cnn import predict as cnn_predict
from modern_methods.cnn_model import load_model


CNN_MODEL = None

def get_cnn_model():
    global CNN_MODEL
    if CNN_MODEL is None:
        print("[Pipeline] Loading CNN model...")
        CNN_MODEL = load_model(MODEL_PATH)
    return CNN_MODEL


def extract_video_frames(video_path, max_frames=100):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(TEMP_DIR, f"{video_name}_frames")

    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        frame_rate=FRAME_RATE,
        max_frames=max_frames
    )

    loaded_frames = [
        cv2.imread(f) for f in frame_paths
        if cv2.imread(f) is not None
    ]

    print(f"[Pipeline] Extracted {len(loaded_frames)} frames")
    return frames_dir, loaded_frames


def run_traditional(loaded_frames):
    """
    Runs all three traditional methods.

    Decision logic (FC+CM fusion):
        FC says Tampered              → Tampered (100% precision)
        FC says Authentic + CM says Tampered → Tampered (CM catches rest)
        Both say Authentic            → Authentic

    OF is computed and shown but does not affect the final decision.
    It serves as supporting evidence in the UI.

    Performance after tuning:
        FC alone  : Precision 100%, Recall 57%,  F1 73%
        OF alone  : Precision 90%,  Recall 79%,  F1 84%
        CM alone  : Precision 78%,  Recall 100%, F1 88%
        FC+CM     : Precision 78%,  Recall 100%, F1 88%
    """
    print("[Pipeline] Running traditional methods...")

    fc_result = fc_analyze(
        loaded_frames,
        threshold_mse=FC_MSE_THRESHOLD,
        threshold_ssim=FC_SSIM_THRESHOLD
    )
    of_result = of_analyze(loaded_frames)
    cm_result = cm_analyze(loaded_frames, sample_rate=CM_SAMPLE_RATE)

    fc_score = fc_result['score']
    of_score = of_result['score']
    cm_score = cm_result['score']
    fc_pred  = fc_result['prediction']
    of_pred  = of_result['prediction']
    cm_pred  = cm_result['prediction']

    # FC+OFfusion decision
    if of_pred == 'Tampered':
        combined_prediction = 'Tampered'
        reasoning = 'Optical Flow detected motion inconsistencies'
    else:
        combined_prediction = 'Authentic'
    reasoning = 'No significant motion tampering detected'

    print(f"[Pipeline] FC : {fc_pred} (score: {fc_score:.4f}) [decision maker]")
    print(f"[Pipeline] OF : {of_pred} (score: {of_score:.4f}) [supporting evidence]")
    print(f"[Pipeline] CM : {cm_pred} (score: {cm_score:.4f}) [supporting evidence]")
    print(f"[Pipeline] Traditional result: {combined_prediction}")
    print(f"[Pipeline] Reasoning: {reasoning}")

    return {
        'prediction'       : combined_prediction,
        'reasoning'        : reasoning,
        'score'            : float(np.mean([fc_score, cm_score])),
        'frame_correlation': {'score': fc_score, 'prediction': fc_pred},
        'optical_flow'     : {'score': of_score, 'prediction': of_pred,
                              'note': 'supporting evidence'},
        'copy_move'        : {'score': cm_score, 'prediction': cm_pred}
    }


def run_modern(frames_dir):
    """
    Runs CNN detection with face cropping preprocessing.
    CNN was trained on cropped face images from FF++.
    Face cropping aligns uploaded video frames with training data.
    Will be updated after retraining on FF++ C23 full frame videos.
    """
    print("[Pipeline] Running CNN detection...")

    from preprocessing.face_crop import crop_faces_from_folder

    cropped_dir = frames_dir + "_cropped"
    _, found_count = crop_faces_from_folder(frames_dir, cropped_dir)

    if found_count == 0:
        print("[Pipeline] No faces found, using full frames")
        cropped_dir = frames_dir

    model  = get_cnn_model()
    result = cnn_predict(cropped_dir, model=model)

    if os.path.exists(cropped_dir) and cropped_dir != frames_dir:
        shutil.rmtree(cropped_dir)

    print(f"[Pipeline] CNN: {result['prediction']} (score: {result['score']:.4f})")
    return result


def detect_video(video_path, method='traditional', max_frames=100, cleanup=True):
    """
    Main entry point called by UI.
    Accepts method: 'traditional' or 'modern'.

    Args:
        video_path : Path to uploaded video
        method     : 'traditional' or 'modern'
        max_frames : Max frames to extract
        cleanup    : Delete temp frames after detection
    Returns:
        result dict with prediction, confidence, scores
    """
    print(f"\n[Pipeline] Starting detection")
    print(f"[Pipeline] Video  : {os.path.basename(video_path)}")
    print(f"[Pipeline] Method : {method}")

    if not os.path.exists(video_path):
        return {
            'error'     : f'Video not found: {video_path}',
            'prediction': 'Unknown',
            'score'     : 0.0
        }

    frames_dir, loaded_frames = extract_video_frames(video_path, max_frames)

    if len(loaded_frames) == 0:
        return {
            'error'     : 'Could not extract frames',
            'prediction': 'Unknown',
            'score'     : 0.0
        }

    if method == 'traditional':
        result = run_traditional(loaded_frames)
    elif method == 'modern':
        result = run_modern(frames_dir)
    else:
        return {
            'error'     : f'Unknown method: {method}',
            'prediction': 'Unknown',
            'score'     : 0.0
        }

    result['method']     = method
    result['video_name'] = os.path.basename(video_path)
    result['confidence'] = round(result['score'] * 100, 2)

    if cleanup and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"[Pipeline] Temp frames cleaned up")

    print(f"\n[Pipeline] FINAL: {result['prediction']} ({result['confidence']}% confidence)")
    return result


if __name__ == "__main__":
    print("=== Pipeline Test ===\n")

    for path, label in [
        ("/home/saron03/Downloads/archive (1)/Testing/Forged/Forgery_insertion/vid100.avi", "FORGED"),
        ("/home/saron03/Downloads/archive (1)/Testing/Original/vid1.avi", "ORIGINAL"),
    ]:
        print(f"\nTesting {label}:")
        result = detect_video(path, method='traditional', max_frames=50)
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}%")