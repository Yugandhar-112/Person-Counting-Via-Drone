# AeroStreamID — Project Context for Claude

> Upload this file as context in your Claude project so it understands your full project.

---

## 1. Project Identity

- **Title**: AeroStreamID: Off-Board Real-Time Person Identification Pipeline for Wi-Fi Connected UAVs
- **Type**: B.Tech 7th Semester Major Project (Dissertation)
- **University**: Amity University Punjab, Mohali, India
- **School**: Amity School of Engineering and Technology
- **Timeline**: August 2026 – December 2026

---

## 2. Project Overview

A low-cost drone (ESP32-CAM module, Wi-Fi MJPEG stream) feeds live video to a laptop, which runs a computer vision pipeline to detect, track, and count unique people in the frame in real time.

### Core Thesis / USP

> Off-the-shelf multi-object tracking silently overcounts under camera ego-motion; this project adds motion-compensated association to the tracking layer and proves, with a real evaluation harness, that it measurably reduces ID-switch-driven overcounting compared to the naive baseline.

This is the load-bearing claim of the project. Everything else is secondary polish that supports this claim.

---

## 3. The Ego-Motion Problem (Why This Project Matters)

Tracking-by-detection algorithms (SORT, DeepSORT, ByteTrack) associate detections across frames using IoU and/or appearance similarity, implicitly assuming the camera is roughly static. A flying drone violates this constantly — panning, altitude changes, vibration shift the entire scene between frames. The practical effect: tracks get dropped and re-spawned as new IDs, which **silently inflates the unique-person count**.

The solution: estimate camera motion between frames (ORB features + RANSAC homography), warp previous-frame track positions into the current frame's coordinate space, then run the standard association. This is a modular addition to the tracker, not a rewrite.

---

## 4. System Architecture

```
ESP32-CAM (on drone)
    │ Wi-Fi MJPEG stream
    ▼
StreamReader (stream_reader.py)
    │ frames + drop info
    ▼
PersonDetector (detector.py) ←── optional: TiledDetector (tiled_detector.py)
    │ bounding boxes + confidence
    ▼
MotionCompensator (motion_compensation.py)  [optional, toggleable]
    │ homography matrix
    ▼
CentroidTracker (tracker.py)
    │ persistent track IDs
    ▼
PersonCounter (counter.py)
    │ unique count
    ▼
Display + Logger (main.py, logger.py)
```

Optional enhancements (all toggleable via CLI flags):
- **Tiled Detection** — SAHI-style frame slicing for small/distant people
- **Count Confidence** — reliability score based on motion magnitude, track overlap, frame drop rate
- **Gap Interpolation** — constant-velocity track prediction during Wi-Fi frame drops
- **Privacy Blur** — Gaussian blur on head regions

---

## 5. Codebase Structure

```
Person-Counting-Via-Drone/
├── src/
│   ├── main.py                # Full pipeline orchestration + CLI
│   ├── config.py              # Dataclass-based config, all toggles default OFF
│   ├── stream_reader.py       # Video input with FrameResult (drop tracking)
│   ├── detector.py            # YOLOv8 person detection
│   ├── tracker.py             # Centroid tracker + homography warp + gap interpolation
│   ├── counter.py             # Unique ID accumulator
│   ├── motion_compensation.py # ORB + BFMatcher + RANSAC homography
│   ├── tiled_detector.py      # SAHI-style sliced detection + NMS merge
│   ├── confidence.py          # Three-signal count reliability scoring
│   ├── privacy.py             # Head-region Gaussian blur
│   ├── logger.py              # Per-frame JSONL logging
│   ├── evaluation.py          # MOT metrics: MOTA, IDF1, ID switches
│   ├── annotate_gt.py         # Interactive ground-truth annotation (CV2 GUI)
│   ├── run_evaluation.py      # Batch eval runner + baseline vs enhanced comparison
│   └── benchmark_models.py    # YOLO model-size Pareto benchmark
├── test_clips/                # Test videos (gitignored)
├── ground_truth/              # Hand-annotated GT JSONs (gitignored)
├── logs/                      # Pipeline run logs (gitignored)
├── docs/                      # Project proposal PDF
├── requirements.txt           # opencv-python, ultralytics, numpy, matplotlib
└── README.md
```

---

## 6. Key Algorithms

### 6.1 Motion Compensation
1. Convert consecutive frames to grayscale
2. Detect ORB keypoints + compute descriptors (up to 1000 features)
3. Match descriptors using BFMatcher with Hamming distance, kNN (k=2)
4. Filter matches via Lowe's ratio test (ratio = 0.75)
5. Estimate homography via RANSAC (minimum 10 inlier matches required)
6. Warp all previous-frame track centroids and bounding boxes using the homography
7. Fallback: if estimation fails → skip compensation, use naive association

### 6.2 Centroid Tracker
- Greedy nearest-neighbor association (Euclidean distance < max_distance threshold)
- Missed-frame counter per track (removed after max_missed_frames)
- Recently-deleted track memory (3 frames) for ID-switch detection
- Velocity estimation for gap interpolation
- Accepts optional homography to warp track positions before association

### 6.3 Tiled Detection (SAHI-style)
- Slice frame into overlapping tiles (default 640×640, 20% overlap)
- Run YOLO on each tile + full frame
- Map tile-local detections back to full-frame coordinates
- Merge via NMS (IoU threshold 0.5) to remove cross-tile duplicates

### 6.4 Evaluation Metrics
- **MOTA** = 1 − (FN + FP + ID_switches) / total_GT
- **IDF1** = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
- **ID Switches**: count of track ID changes for the same ground-truth person
- **Count Error**: |predicted_unique_count − GT_unique_count|

---

## 7. Hardware

- **ESP32-CAM**: ESP32-S module + OV2640 camera sensor (2MP, up to 1600×1200), 802.11 b/g/n Wi-Fi, broadcasts MJPEG stream over its own AP
- **Quadcopter**: Self-assembled frame, brushless motors, ESCs, flight controller, LiPo battery, RC receiver — final parts TBD with faculty
- **Laptop**: For running the CV pipeline (needs discrete GPU for real-time YOLO inference)

---

## 8. Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.13 | Rapid prototyping, rich CV/ML ecosystem |
| Detection | Ultralytics YOLOv8 (nano/small) | Real-time speed, COCO pretrained, no retraining needed |
| Video/CV | OpenCV 4.9+ | Video I/O, feature matching, homography, GUI |
| Features | ORB (not SIFT/SURF) | Patent-free, faster, good enough for ego-motion estimation |
| Math | NumPy 1.26+ | Matrix operations, coordinate transforms |
| Plotting | Matplotlib 3.8+ | Benchmark visualization, Pareto plots |

---

## 9. Feature Toggles (CLI Flags)

All enhancements default OFF so baseline runs are clean:

| Flag | Feature |
|------|---------|
| `--motion-compensation` | ORB+homography ego-motion correction |
| `--tiling` | SAHI-style tiled detection |
| `--confidence-flag` | Count reliability indicator |
| `--gap-interpolation` | Dropped-frame track prediction |
| `--privacy-blur` | Head-region Gaussian blur |
| `--headless` | No display window (for eval/CI) |
| `--log-dir DIR` | Per-frame JSONL logging |

---

## 10. Priority Order (What Matters Most)

### Tier 1 — Primary Thesis (must ship)
1. Baseline instrumentation on naive pipeline (measure the problem)
2. Motion-compensated tracking (fix the problem)
3. Evaluation harness (prove the fix with numbers)

### Tier 2 — Secondary (in order of cost-to-value)
4. Tiled detection for small-object recall
5. Confidence-flagged counts
6. Stream frame-gap interpolation
7. Model-size Pareto benchmark
8. Privacy blurring
9. Environmental characterization (stretch)

---

## 11. Current Status (as of November 2026)

- ✅ All 14 source modules implemented and functional
- ✅ Full pipeline runs end-to-end (webcam + video files tested)
- ✅ All feature toggles wired and working
- ✅ Evaluation infrastructure complete (annotation tool + metrics + batch runner)
- 🔲 Hardware (drone) assembly in progress
- 🔲 Test flight footage not yet captured
- 🔲 Quantitative evaluation (baseline vs. enhanced numbers) pending test clips
- 🔲 Mid-term report and synopsis being prepared

---

## 12. Known Risks & Constraints

- Ego-motion estimation fails under fast rotation, low-texture scenes, or motion blur → fallback to naive association
- Ground truth annotation is manual and small-scale → don't overstate statistical significance
- Wi-Fi stream reliability is hardware-dependent → characterize, don't try to "fix" the physical link
- Do NOT let Tier 2 work start before Tier 1 baseline exists
- Do NOT pursue fine-tuning on VisDrone or custom detector training — that's a separate thesis

---

## 13. What I Need Help With

- Writing/editing dissertation report and synopsis
- Code improvements and bug fixes
- Evaluation strategy and metrics interpretation
- Presentation preparation
- Hardware integration debugging (ESP32-CAM streaming issues)
- Any technical challenges that arise
