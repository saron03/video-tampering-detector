# Video Tampering Detection System

A video forensics system that automatically determines whether a video has been tampered with. The system supports two detection approaches: traditional signal processing algorithms and a modern deep learning CNN model.

---

## What It Does

Users upload a video through a web interface and select a detection method. The system extracts frames from the video, runs the chosen detection algorithm, and returns a verdict of **Authentic** or **Tampered** along with confidence scores and a visual breakdown of the analysis.

**Traditional Method** — analyzes temporal and spatial properties of video frames using three classical algorithms. Best suited for detecting frame insertion, deletion, duplication, zooming, flipping, and rotation forgeries.

**Modern Method (CNN)** — uses a fine-tuned EfficientNet-B0 convolutional neural network trained on the FaceForensics++ dataset. Best suited for detecting deepfake face manipulations such as face swapping and expression transfer.

---

## Detection Methods

### Traditional
Three algorithms run on the extracted frames:

**Frame Correlation** — measures Mean Squared Error (MSE) and Structural Similarity (SSIM) between consecutive frames. A sudden spike in MSE above a threshold indicates a frame was inserted or deleted. This method achieved 100% precision on the Video Forgery Dataset — when it fires, it is always correct.

**Optical Flow** — tracks pixel movement between frames using the Lucas-Kanade algorithm. In authentic videos, motion is smooth and consistent. Sudden chaotic motion vectors indicate spliced content or motion-based forgeries. After threshold tuning this method achieved 90% precision and 79% recall.

**Copy-Move Detection** — divides each frame into blocks and computes SVD and LBP texture fingerprints. Blocks with suspiciously similar fingerprints in different locations indicate copy-paste forgery. Note: this method has known limitations on compressed AVI files due to compression artifacts producing false positives. It is shown in the UI as supporting evidence but does not affect the final decision.

The final traditional decision is made by Optical Flow. Frame Correlation and Copy-Move are shown as supporting evidence.

### Modern (CNN)
A pretrained EfficientNet-B0 model fine-tuned on 20,000 frames from the FaceForensics++ dataset (fake and real). The model achieved 92.33% accuracy on an unseen test set. Face detection and cropping is applied before inference to align uploaded video frames with the training data format.

---

## Datasets Used

**FaceForensics++ (FF++) Extracted Frames**
Used for CNN training. Contains approximately 192,000 pre-extracted image frames organized into fake/ and real/ directories. The fake directory contains five manipulation types: Deepfakes, Face2Face, FaceShifter, FaceSwap, and NeuralTextures.

Source: Kaggle — FaceForensics++ extracted frames
Citation: Rössler et al., FaceForensics++: Learning to Detect Manipulated Facial Images, ICCV 2019.

**Video Forgery Dataset**
Used for traditional method evaluation. Contains 9,448 training and 2,904 testing videos across seven forgery types: insertion, deletion, duplication, horizontal flipping, vertical flipping, rotation, and zooming.

Source: Kaggle — Video Forgery Dataset
Credit: Dr. Premanand Ghadekar et al.

---

## Evaluation Results

### Traditional Methods (Video Forgery Dataset, 90 test videos)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Frame Correlation | 66.67% | 100.00% | 57.14% | 72.73% |
| Optical Flow | 76.67% | 90.16% | 78.57% | 83.97% |
| Copy-Move | 77.78% | 77.78% | 100.00% | 87.50% |

### CNN (FaceForensics++ test set, 300 frames)

| Method | Accuracy | Notes |
|---|---|---|
| EfficientNet-B0 | 92.33% | Trained on 20k images, 10 epochs |

---

## Limitations

**Traditional methods on compressed video** — the Video Forgery Dataset uses compressed AVI files. Compression artifacts create artificially high frame differences, making threshold-based detection unreliable for some forgery types. Copy-Move detection in particular produces false positives on compressed videos and is therefore excluded from the final decision.

**CNN generalization** — the CNN was trained exclusively on FaceForensics++ style manipulations (face swapping, expression transfer). It does not generalize well to other deepfake types or to spatially/temporally tampered videos outside the FF++ distribution. This is a known limitation of dataset-specific deepfake detectors.

**Method mismatch** — traditional methods perform best on temporal forgeries (insertion, deletion). The CNN performs best on deepfake face manipulations. Neither method reliably detects the other's target forgery type.

---

## Folder Structure

```
video-tampering-detector/
│
├── config.py                       # Central configuration — all paths, thresholds,
│                                   # hyperparameters defined here. Edit this file
│                                   # to point to your dataset locations.
│
├── requirements.txt                # Python dependencies. Install with:
│                                   # pip install -r requirements.txt
│
├── README.md                       # This file
│
├── dataset/                        # Reference folder — actual data lives in Downloads
│   └── .gitkeep                    # Data is never pushed to GitHub
│
├── preprocessing/
│   ├── extract_frames.py           # Extracts frames from any video file using OpenCV.
│   │                               # Used by pipeline at runtime and for dataset prep.
│   │                               # Supports single video and batch folder processing.
│   └── face_crop.py                # Detects and crops faces from frames using
│                                   # OpenCV Haar cascade. Applied before CNN inference
│                                   # to match training data format (cropped faces).
│
├── traditional_methods/
│   ├── frame_correlation.py        # Frame deletion/insertion detection.
│   │                               # Compares consecutive frames using MSE and SSIM.
│   │                               # Flags frame pairs with MSE spikes above threshold.
│   │                               # Performance: Precision 100%, Recall 57%, F1 73%.
│   │
│   ├── optical_flow_analysis.py    # Motion inconsistency detection.
│   │                               # Tracks pixel movement using Lucas-Kanade method.
│   │                               # Detects chaotic or inconsistent motion vectors.
│   │                               # Performance: Precision 90%, Recall 79%, F1 84%.
│   │
│   ├── copy_move_svd_lbp.py        # Copy-paste forgery detection within frames.
│   │                               # Divides frames into blocks, computes SVD and LBP
│   │                               # fingerprints, flags suspiciously similar blocks.
│   │                               # Note: informational only on compressed AVI files.
│   │                               # Performance: Precision 78%, Recall 100%, F1 88%.
│   │
│   └── utils.py                    # Shared helper functions for all traditional methods.
│                                   # Includes: normalize_scores, smooth_signal,
│                                   # load_frames, resize_frame, convert_to_gray,
│                                   # draw_anomaly_boxes, save_result_image.
│
├── modern_methods/
│   ├── cnn_model.py                # EfficientNet-B0 model architecture.
│   │                               # Defines the model blueprint with a custom
│   │                               # classifier head for binary (real/fake) output.
│   │                               # Includes save_model and load_model helpers.
│   │
│   ├── train_cnn.py                # Model training script.
│   │                               # Loads FF++ dataset, trains for N epochs,
│   │                               # saves best weights based on validation loss.
│   │                               # Run once: python modern_methods/train_cnn.py
│   │
│   ├── evaluate_cnn.py             # Model evaluation and inference.
│   │                               # Tests trained model on unseen data.
│   │                               # Contains predict() function used by pipeline
│   │                               # at runtime when user uploads a video.
│   │
│   └── utils.py                    # Data loading utilities for CNN.
│                                   # Walks FF++ dataset folders, applies transforms,
│                                   # creates train/val/test splits, builds DataLoaders.
│
├── pipeline/
│   └── detect_video.py             # Central pipeline connecting everything.
│                                   # Called by UI when user uploads a video.
│                                   # Steps: extract frames → run chosen method
│                                   # → combine results → return verdict to UI.
│                                   # Traditional decision maker: Optical Flow.
│                                   # Modern decision maker: CNN probability > 0.5.
│
├── ui/
│   └── app.py                      # Streamlit web interface.
│                                   # User uploads video, selects method, clicks analyze.
│                                   # Displays verdict, confidence score, and
│                                   # visual breakdown of detection scores.
│                                   # Run with: streamlit run ui/app.py
│
├── evaluation/
│   └── metrics.py                  # Formal evaluation on test datasets.
│                                   # Computes accuracy, precision, recall, F1
│                                   # for all traditional methods across all
│                                   # forgery types. Results saved to results/.
│
├── models/                         # Saved CNN weights (not pushed to GitHub)
│   └── cnn_weights.pth             # Trained EfficientNet-B0 weights.
│                                   # Generated by train_cnn.py.
│
├── results/                        # Evaluation outputs
│   └── evaluation_results.json     # Full metrics breakdown by method and forgery type.
│
└── temp/                           # Runtime temporary files (not pushed to GitHub)
                                    # Frame extractions are saved here during detection
                                    # and deleted automatically after analysis.
```

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/video-tampering-detector.git
cd video-tampering-detector
```

**2. Create a conda environment**
```bash
conda create -n deepfake python=3.10
conda activate deepfake
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure dataset paths**

Open `config.py` and update the dataset paths to match your system:
```python
FF_FAKE_DIR = "/path/to/ff_plus_plus/fake"
FF_REAL_DIR = "/path/to/ff_plus_plus/real"
VF_DATASET_DIR = "/path/to/video_forgery_dataset"
```

**5. Train the CNN (optional — skip if using pretrained weights)**
```bash
python modern_methods/train_cnn.py
```

**6. Run the application**
```bash
streamlit run ui/app.py
```

---

## Usage

1. Open the app in your browser at `http://localhost:8501`
2. Upload a video file (MP4, AVI, MOV, MKV supported)
3. Select a detection method:
   - **Traditional** — for spatially and temporally tampered videos
   - **Modern (CNN)** — for deepfake face manipulation videos
4. Adjust the max frames slider (more frames = more accurate but slower)
5. Click **Run Analysis**
6. View the verdict, confidence score, and per-method breakdown

---

## Tech Stack

- **Python 3.10**
- **PyTorch** — CNN training and inference
- **OpenCV** — frame extraction, optical flow, face detection
- **scikit-image** — SSIM calculation, LBP texture features
- **Streamlit** — web interface
- **Plotly** — interactive charts in UI
- **scikit-learn** — evaluation metrics
- **torchvision** — EfficientNet-B0 pretrained model

---

Developed under the mentorship of iCog Labs.
