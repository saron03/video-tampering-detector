# face_crop.py
# Detects and crops faces from frames before CNN inference.
# CNN was trained on cropped face images from FF++
# so we must crop faces from uploaded videos to match.

import cv2
import numpy as np
import os

# OpenCV's pretrained face detector
# This file comes with opencv, no download needed
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def crop_face(frame, padding=0.2):
    """
    Detects the largest face in a frame and returns
    a cropped and padded version of it.

    If no face is detected returns the original frame resized.
    This way CNN still gets input even if face detection fails.

    Args:
        frame   : BGR numpy array (full video frame)
        padding : Extra padding around face as fraction of face size
    Returns:
        cropped : BGR numpy array of face region (or full frame if no face)
        found   : True if face was detected, False otherwise
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        # No face found, return resized full frame
        return cv2.resize(frame, (224, 224)), False

    # Take the largest face if multiple detected
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    # Add padding around face
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    # Clamp to image boundaries
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    cropped = frame[y1:y2, x1:x2]

    # Resize to CNN input size
    cropped = cv2.resize(cropped, (224, 224))

    return cropped, True


def crop_faces_from_folder(frames_dir, output_dir):
    """
    Processes all frames in a folder.
    Detects and crops faces from each frame.
    Saves cropped faces to output_dir.
    Used by pipeline before passing frames to CNN.

    Args:
        frames_dir : Folder containing extracted video frames
        output_dir : Folder to save cropped face images
    Returns:
        output_dir : Same output dir (for chaining)
        found_count: How many frames had faces detected
    """
    os.makedirs(output_dir, exist_ok=True)

    supported    = ('.png', '.jpg', '.jpeg')
    frame_files  = sorted([
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in supported
    ])

    found_count = 0

    for fname in frame_files:
        frame_path = os.path.join(frames_dir, fname)
        frame      = cv2.imread(frame_path)

        if frame is None:
            continue

        cropped, found = crop_face(frame)

        if found:
            found_count += 1

        # Save cropped face with same filename
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, cropped)

    print(f"[FaceCrop] Processed {len(frame_files)} frames")
    print(f"[FaceCrop] Faces detected in {found_count}/{len(frame_files)} frames")

    return output_dir, found_count


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('..')
    from extract_frames import extract_frames

    TEST_VIDEO = "/home/saron03/Downloads/archive/fake/Deepfakes/000_003/../../../real/000/000_0001.png"

    print("=== Face Crop Test ===")

    # Test on a single FF++ frame
    test_img_path = "/home/saron03/Downloads/archive/fake/Deepfakes/000_003/000.png"
    frame = cv2.imread(test_img_path)

    if frame is not None:
        cropped, found = crop_face(frame)
        print(f"Input shape  : {frame.shape}")
        print(f"Output shape : {cropped.shape}")
        print(f"Face found   : {found}")
        cv2.imwrite("../temp/test_crop.png", cropped)
        print("Saved to temp/test_crop.png")
    else:
        print("Could not load test image")
