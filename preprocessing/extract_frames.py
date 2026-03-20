import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, frame_rate=1, max_frames=None, resize=None):
    """
    Extract frames from a video file.

    Args:
        video_path   : Path to the input video file
        output_dir   : Directory to save extracted frames
        frame_rate   : Extract every N frames (1 = every frame, 5 = every 5th frame)
        max_frames   : Max number of frames to extract (None = all)
        resize       : Tuple (width, height) to resize frames, or None to keep original
    
    Returns:
        List of saved frame paths
    """
    video_path = str(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_name = Path(video_path).stem
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video     : {video_name}")
    print(f"FPS       : {fps:.2f}")
    print(f"Total frames : {total_frames}")

    saved_paths = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Only extract every Nth frame
        if frame_idx % frame_rate == 0:

            if resize:
                frame = cv2.resize(frame, resize)

            frame_filename = f"{video_name}_{frame_idx:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                print(f"Reached max_frames limit: {max_frames}")
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_count} frames to: {output_dir}\n")
    return saved_paths


def extract_from_folder(input_folder, output_folder, frame_rate=1, max_frames=None, resize=None):
    """
    Extract frames from all videos inside a folder.
    Useful for batch processing your dataset (authentic/ and tampered/ folders).

    Args:
        input_folder  : Folder containing video files
        output_folder : Root folder to save frames (subfolders per video)
        frame_rate    : Extract every N frames
        max_frames    : Max frames per video
        resize        : Resize tuple or None
    """
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    video_files = [
        f for f in Path(input_folder).iterdir()
        if f.suffix.lower() in supported_formats
    ]

    if not video_files:
        print(f"No video files found in: {input_folder}")
        return

    print(f"Found {len(video_files)} videos in {input_folder}\n")

    all_paths = []
    for video_file in video_files:
        # Each video gets its own subfolder
        video_output_dir = os.path.join(output_folder, video_file.stem)
        try:
            paths = extract_frames(
                video_path=video_file,
                output_dir=video_output_dir,
                frame_rate=frame_rate,
                max_frames=max_frames,
                resize=resize
            )
            all_paths.extend(paths)
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")

    print(f"Done. Total frames extracted: {len(all_paths)}")
    return all_paths


# ──────────────────────────────────────────────
# CLI usage:
#   python extract_frames.py --video path/to/video.mp4 --output frames/
#   python extract_frames.py --folder dataset/authentic --output frames/authentic
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video(s)")

    parser.add_argument("--video",      type=str, help="Path to a single video file")
    parser.add_argument("--folder",     type=str, help="Path to a folder of videos")
    parser.add_argument("--output",     type=str, required=True, help="Output directory for frames")
    parser.add_argument("--frame_rate", type=int, default=1,    help="Extract every N frames (default: 1)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to extract per video")
    parser.add_argument("--width",      type=int, default=None, help="Resize width")
    parser.add_argument("--height",     type=int, default=None, help="Resize height")

    args = parser.parse_args()

    resize = (args.width, args.height) if args.width and args.height else None

    if args.video:
        extract_frames(args.video, args.output, args.frame_rate, args.max_frames, resize)
    elif args.folder:
        extract_from_folder(args.folder, args.output, args.frame_rate, args.max_frames, resize)
    else:
        print("Provide either --video or --folder")