# metrics.py
# Evaluates all detection methods on the full test dataset.
# Produces accuracy, precision, recall and F1 scores for each method.


import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VF_TEST_FORGED, VF_TEST_ORIGINAL,
    VF_FORGERY_TYPES, FC_MSE_THRESHOLD,
    FC_SSIM_THRESHOLD, CM_SAMPLE_RATE, RESULTS_DIR
)

from preprocessing.extract_frames import extract_frames
from traditional_methods.frame_correlation import analyze as fc_analyze
from traditional_methods.optical_flow_analysis import analyze as of_analyze
from traditional_methods.copy_move_svd_lbp import analyze as cm_analyze


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total     = len(y_true)
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp)    if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0
    f1        = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0

    return {
        'accuracy' : round(accuracy  * 100, 2),
        'precision': round(precision * 100, 2),
        'recall'   : round(recall    * 100, 2),
        'f1'       : round(f1        * 100, 2),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'total': total
    }


def collect_test_videos(max_per_type=20, max_original=50):
    test_videos = []

    for forgery_type in VF_FORGERY_TYPES:
        folder = os.path.join(VF_TEST_FORGED, forgery_type)
        if not os.path.exists(folder):
            print(f"[Metrics] Folder not found: {folder}")
            continue

        videos = sorted([
            f for f in os.listdir(folder)
            if f.endswith('.avi')
        ])[:max_per_type]

        for v in videos:
            test_videos.append((
                os.path.join(folder, v),
                1,
                forgery_type.replace('Forgery_', '')
            ))
        print(f"[Metrics] {forgery_type}: {len(videos)} videos")

    originals = sorted([
        f for f in os.listdir(VF_TEST_ORIGINAL)
        if f.endswith('.avi')
    ])[:max_original]

    for v in originals:
        test_videos.append((
            os.path.join(VF_TEST_ORIGINAL, v),
            0,
            'original'
        ))

    print(f"[Metrics] Original   : {len(originals)} videos")
    print(f"[Metrics] Total      : {len(test_videos)} videos")
    return test_videos


def load_frames_for_video(video_path, max_frames=50):
    try:
        temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'temp', 'metrics_frames'
        )
        frame_paths = extract_frames(
            video_path, output_dir=temp_dir,
            frame_rate=1, max_frames=max_frames
        )
        return [cv2.imread(f) for f in frame_paths if cv2.imread(f) is not None]
    except Exception as e:
        print(f"[Metrics] Error: {e}")
        return []


def run_all_methods(test_videos, max_frames=50):
    """
    Runs all methods on all videos in one pass.
    More efficient than running separate loops per method.
    Returns predictions for all methods.
    """
    fc_true, fc_pred = [], []
    of_true, of_pred = [], []
    cm_true, cm_pred = [], []
    fu_true, fu_pred = [], []

    for video_path, true_label, forgery_type in tqdm(test_videos, desc="Evaluating"):
        frames = load_frames_for_video(video_path, max_frames)
        if not frames:
            continue

        fc = fc_analyze(frames, threshold_mse=FC_MSE_THRESHOLD)
        of = of_analyze(frames)
        cm = cm_analyze(frames, sample_rate=CM_SAMPLE_RATE)

        fc_p = 1 if fc['prediction'] == 'Tampered' else 0
        of_p = 1 if of['prediction'] == 'Tampered' else 0
        cm_p = 1 if cm['prediction'] == 'Tampered' else 0

        # FC+CM fusion
        if fc_p == 1 or cm_p == 1:
            fu_p = 1
        else:
            fu_p = 0

        fc_true.append(true_label); fc_pred.append(fc_p)
        of_true.append(true_label); of_pred.append(of_p)
        cm_true.append(true_label); cm_pred.append(cm_p)
        fu_true.append(true_label); fu_pred.append(fu_p)

    return {
        'Frame Correlation': calculate_metrics(fc_true, fc_pred),
        'Optical Flow'     : calculate_metrics(of_true, of_pred),
        'Copy-Move'        : calculate_metrics(cm_true, cm_pred),
        'FC+CM Fusion'     : calculate_metrics(fu_true, fu_pred)
    }


def run_per_type(test_videos, max_frames=50):
    """
    Evaluates FC+CM fusion broken down by forgery type.
    Correctly separates original videos per comparison.
    """
    forgery_types   = list(set([
        t for _, _, t in test_videos if t != 'original'
    ]))
    original_videos = [(p, l, t) for p, l, t in test_videos if t == 'original']
    results         = {}

    for ftype in forgery_types:
        type_forged = [(p, l, t) for p, l, t in test_videos if t == ftype]
        combined    = type_forged + original_videos

        y_true, y_pred = [], []

        for video_path, true_label, _ in tqdm(
            combined, desc=f"  {ftype}", leave=False
        ):
            frames = load_frames_for_video(video_path, max_frames)
            if not frames:
                continue

            fc = fc_analyze(frames, threshold_mse=FC_MSE_THRESHOLD)
            cm = cm_analyze(frames, sample_rate=CM_SAMPLE_RATE)

            fc_p = 1 if fc['prediction'] == 'Tampered' else 0
            cm_p = 1 if cm['prediction'] == 'Tampered' else 0
            fu_p = 1 if (fc_p == 1 or cm_p == 1) else 0

            y_true.append(true_label)
            y_pred.append(fu_p)

        if y_true:
            results[ftype] = calculate_metrics(y_true, y_pred)

    return results


def print_results_table(results):
    print("\n" + "="*65)
    print(f"{'Method':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("="*65)
    for method, m in results.items():
        print(
            f"{method:<25} "
            f"{m['accuracy']:>8.1f}% "
            f"{m['precision']:>9.1f}% "
            f"{m['recall']:>7.1f}% "
            f"{m['f1']:>5.1f}%"
        )
    print("="*65)


def run_evaluation(max_per_type=15, max_original=30, max_frames=50):
    print("="*65)
    print("VIDEO TAMPERING DETECTION — EVALUATION")
    print("="*65)
    print(f"Max per forgery type : {max_per_type}")
    print(f"Max original videos  : {max_original}")
    print(f"Frames per video     : {max_frames}")

    test_videos = collect_test_videos(max_per_type, max_original)

    print(f"\nForged   : {sum(1 for _,l,_ in test_videos if l==1)}")
    print(f"Original : {sum(1 for _,l,_ in test_videos if l==0)}")

    print("\n[Metrics] Running all methods...")
    results = run_all_methods(test_videos, max_frames)
    print_results_table(results)

    print("\n[Metrics] Running per forgery type (FC+CM Fusion)...")
    per_type = run_per_type(test_videos, max_frames)

    print("\n" + "="*65)
    print(f"{'Forgery Type':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("="*65)
    for ftype, m in per_type.items():
        print(
            f"{ftype:<25} "
            f"{m['accuracy']:>8.1f}% "
            f"{m['precision']:>9.1f}% "
            f"{m['recall']:>7.1f}% "
            f"{m['f1']:>5.1f}%"
        )
    print("="*65)

    output_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump({'overall': results, 'per_type': per_type}, f, indent=2)
    print(f"\n[Metrics] Saved to: {output_path}")

    return results, per_type


if __name__ == "__main__":
    results, per_type = run_evaluation(
        max_per_type=10,
        max_original=20,
        max_frames=50
    )
    print("\n=== EVALUATION COMPLETE ===")