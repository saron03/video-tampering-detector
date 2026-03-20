
# config.py

# WHAT THIS FILE DOES:
# Central configuration file for the entire project.
# ALL paths, settings, and hyperparameters live here.
#
# WHY THIS IS IMPORTANT:
# Instead of hardcoding paths in every file like:
#   "/home/saron03/Downloads/archive/fake"
# We define it once here and import it everywhere.
# If you move your dataset, you only change ONE file.
#
# USAGE in other files:
#   from config import FF_FAKE_DIR, MODEL_PATH, BATCH_SIZE


import os

# BASE DIRECTORY
# This is the root of your project folder
# Everything else is relative to this

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# DATASET PATHS

# ── FF++ Dataset (used for CNN training) ─────
# These are your extracted frames from FaceForensics++
# Located in your Downloads based on your terminal output
FF_DATASET_DIR = "/home/saron03/Downloads/archive"
FF_FAKE_DIR    = os.path.join(FF_DATASET_DIR, "fake")
FF_REAL_DIR    = os.path.join(FF_DATASET_DIR, "real")

# Subdirectories inside fake/ (all manipulation types)
FF_FAKE_SUBDIRS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

# ── Video Forgery Dataset (used for traditional methods) ──
# Located in your Downloads as "archive (1)"
VF_DATASET_DIR = "/home/saron03/Downloads/archive (1)"

VF_TRAIN_DIR          = os.path.join(VF_DATASET_DIR, "Training")
VF_TRAIN_FORGED       = os.path.join(VF_TRAIN_DIR,   "Forged")
VF_TRAIN_ORIGINAL     = os.path.join(VF_TRAIN_DIR,   "Original")

VF_TEST_DIR           = os.path.join(VF_DATASET_DIR, "Testing")
VF_TEST_FORGED        = os.path.join(VF_TEST_DIR,    "Forged")
VF_TEST_ORIGINAL      = os.path.join(VF_TEST_DIR,    "Original")

# Forgery type subdirectories inside Forged/
VF_FORGERY_TYPES = [
    "Forgery_deletion",
    "Forgery_insertion",
    "Forgery_duplication",
    "Forgery_horizontal_flipping",
    "Forgery_vertical_flipping",
    "Forgery_other_rotations",
    "Forgery_zooming"
]


# PROJECT FOLDER PATHS

# Temp folder for runtime frame extraction
# (when user uploads video via UI)
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Where CNN model weights are saved after training
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "cnn_weights.pth")

# Where evaluation results and plots are saved
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create these directories if they don't exist yet
for directory in [TEMP_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# FRAME EXTRACTION SETTINGS

# How many frames to extract per second of video
# 1 = every frame, 5 = every 5th frame
FRAME_RATE = 1

# Resize all frames to this size
# 224x224 is standard CNN input size
FRAME_WIDTH  = 224
FRAME_HEIGHT = 224
FRAME_SIZE   = (FRAME_WIDTH, FRAME_HEIGHT)

# Max frames to extract per video
# None = extract all frames
# Set a number to limit (e.g. 100 for quick testing)
MAX_FRAMES_PER_VIDEO = None


# CNN TRAINING SETTINGS

# How many images to use for training
# None = use all available images
# Set lower number for faster prototyping
MAX_FAKE_SAMPLES = 10000    # from all fake subfolders combined
MAX_REAL_SAMPLES = 10000    # from real folder

# Train/validation/test split ratios
# Must add up to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Training hyperparameters
BATCH_SIZE   = 32           # images per training step
NUM_EPOCHS   = 10           # how many times to go through data
LEARNING_RATE = 0.001       # how fast the model learns
NUM_WORKERS  = 2            # parallel data loading threads

# CNN input size (must match FRAME_SIZE)
CNN_INPUT_SIZE = (224, 224)

# Number of output classes
# 2 = binary (real vs fake)
NUM_CLASSES = 2


# TRADITIONAL METHOD SETTINGS

# Frame Correlation thresholds
FC_MSE_THRESHOLD  = 2500.0  # MSE above this = suspicious
FC_SSIM_THRESHOLD = 0.50    # SSIM below this = suspicious

# Optical Flow settings
OF_WINDOW_SIZE = 5          # smoothing window for flow signal

# Copy-Move settings
CM_BLOCK_SIZE     = 16      # block size for SVD/LBP analysis
CM_SVD_THRESHOLD  = 0.95    # SVD similarity threshold
CM_LBP_THRESHOLD  = 0.90    # LBP similarity threshold
CM_SAMPLE_RATE    = 10      # check every Nth frame


# HYBRID METHOD SETTINGS

# Weights for combining traditional and CNN scores
# Must add up to 1.0
# Adjust based on which method performs better
TRADITIONAL_WEIGHT = 0.4    # 40% traditional
CNN_WEIGHT         = 0.6    # 60% CNN

# Final decision threshold
# Score above this = Tampered
HYBRID_THRESHOLD = 0.5

# DEVICE SETTINGS

import torch

# Automatically use GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# QUICK VERIFY — Run this file to check paths
if __name__ == "__main__":
    print("=== Config Verification ===\n")

    paths_to_check = {
        "FF Fake Dir"         : FF_FAKE_DIR,
        "FF Real Dir"         : FF_REAL_DIR,
        "VF Train Forged"     : VF_TRAIN_FORGED,
        "VF Train Original"   : VF_TRAIN_ORIGINAL,
        "VF Test Forged"      : VF_TEST_FORGED,
        "VF Test Original"    : VF_TEST_ORIGINAL,
        "Temp Dir"            : TEMP_DIR,
        "Models Dir"          : MODELS_DIR,
        "Results Dir"         : RESULTS_DIR,
    }

    all_good = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status}  {name}: {path}")
        if not exists:
            all_good = False

    print(f"\nDevice : {DEVICE}")
    print(f"\n{'✅ All paths verified!' if all_good else '❌ Some paths missing — fix them above'}")