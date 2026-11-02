<div align="center">

# AeroStreamID 🛸
### Off-Board Real-Time Person Identification & Counting Pipeline for Wi-Fi Connected UAVs

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%204.9%2B-orange.svg)](https://opencv.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A robust aerial visual surveillance pipeline addressing camera ego-motion and identity fragmentation for accurate crowd monitoring and person counting.*

</div>

---

## 📌 Executive Summary

Traditional Multi-Object Tracking (MOT) frameworks (e.g., SORT, DeepSORT, ByteTrack) implicitly assume a stationary or slow-moving camera. When deployed on low-cost unmanned aerial vehicles (UAVs), unpredictable rotational and translational **camera ego-motion** causes bounding boxes to shift rapidly across consecutive frames. This results in severe **identity fragmentation (ID switching)** and leads naive tracking pipelines to **silently overcount** individuals.

**AeroStreamID** solves this by establishing a lightweight, off-board video analytics architecture:
1. An on-drone **ESP32-CAM module** broadcasts live MJPEG video over an ad-hoc Wi-Fi access point.
2. An off-board ground station (laptop with GPU) ingests the video feed in real time.
3. An **ORB + RANSAC homography motion-compensation engine** rectifies inter-frame perspective shifts *prior* to data association.
4. An association layer tracks unique persons and maintains a reliable, persistent headcount.

---

## 🏗️ System Architecture

```
                       [ AERIAL PLATFORM ]
                    ┌─────────────────────────┐
                    │  ESP32-CAM (OV2640)     │
                    │  Self-Assembled Quad    │
                    └────────────┬────────────┘
                                 │ Wi-Fi 802.11n (MJPEG Stream)
                                 ▼
                    [ GROUND STATION (LAPTOP) ]
                    ┌─────────────────────────┐
                    │   StreamReader          │ (Drop tracking, gap detection)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   PersonDetector        │ (YOLOv8n / optional SAHI Slicing)
                    └────────────┬────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   MotionCompensator (ORB + Lowe's Ratio + RANSAC)       │ ◄── Inter-frame Ego-Motion
       └─────────────────────────┬──────────────────────────────┘
                                 │ Warped Prior Track Coordinates
                                 ▼
                    ┌─────────────────────────┐
                    │   CentroidTracker       │ (Greedy matching + gap interpolation)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   PersonCounter         │ (Unique cumulative headcount)
                    └────────────┬────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
            ┌─────────────────┐     ┌─────────────────┐
            │ Live Display    │     │ Pipeline Logger │
            │ (Overlays & UI) │     │ (JSONL / MOT)   │
            └─────────────────┘     └─────────────────┘
```

---

## 🌟 Key Features & Modular Enhancements

Every feature in AeroStreamID is modular and toggleable via CLI parameters, enabling controlled ablation studies:

- **Ego-Motion Compensation (Thesis Core)**: ORB feature extraction and RANSAC homography estimation project previous-frame track coordinates into the current coordinate space, preventing motion-induced track drops.
- **Tiled / Sliced Inference (SAHI-style)**: Slices frames into overlapping tiles for enhanced small/distant person detection at elevated flight altitudes.
- **Count-Confidence Flagging**: Continuously computes a reliability metric based on estimated camera velocity, track spatial overlap, and wireless frame-drop rates.
- **Frame-Gap Interpolation**: Employs constant-velocity kinematic projection during wireless transmission interruptions, preventing track re-initialization.
- **Privacy-by-Design Filter**: Automated Gaussian head-region blurring ensures privacy compliance without degrading counting fidelity.
- **Full MOT Evaluation Suite**: Standalone evaluation harness supporting MOTA, IDF1, False Positives/Negatives, and ID-switch counts against ground truth.

---

## 📂 Repository Structure

```
AeroStreamID/
├── .github/
│   └── workflows/ci.yml       # Automated GitHub Actions CI workflow
├── docs/                      # Academic proposals and project documentation
├── ground_truth/              # Annotated ground-truth JSON files (.gitkeep)
├── logs/                      # Run logs and JSONL trajectory outputs (.gitkeep)
├── test_clips/                # Benchmark aerial video sequences (.gitkeep)
├── src/
│   ├── __init__.py            # Package entrypoint
│   ├── annotate_gt.py         # Interactive GUI for hand-annotating ground truth
│   ├── benchmark_models.py    # YOLO Pareto frontier benchmark (nano/small/medium)
│   ├── confidence.py          # Three-signal count reliability scoring
│   ├── config.py              # Dataclass-based centralized configuration
│   ├── counter.py             # Unique ID accumulation logic
│   ├── detector.py            # YOLOv8 person detection wrapper
│   ├── evaluation.py          # MOT metric computation (MOTA, IDF1, ID switches)
│   ├── logger.py              # Per-frame structured JSONL logger
│   ├── main.py                # Pipeline orchestration and live visualizer
│   ├── motion_compensation.py # ORB + RANSAC homography estimation
│   ├── privacy.py             # Head-region Gaussian anonymization
│   ├── run_evaluation.py      # Automated benchmark and ablation comparison tool
│   ├── stream_reader.py       # Resilient video ingestion with drop monitoring
│   ├── tiled_detector.py      # SAHI-style sliced inference implementation
│   └── tracker.py             # Motion-compensated centroid tracker
├── .gitignore                 # Publication-ready git ignore configuration
├── LICENSE                    # MIT License
├── pyproject.toml             # Python package configuration (PEP 517/518)
├── requirements.txt           # Python library dependencies
└── README.md                  # Comprehensive project documentation
```

---

## ⚡ Installation & Setup

### 1. Prerequisites
- Python 3.9 to 3.13
- CUDA-compatible GPU (recommended for real-time YOLOv8 inference) or modern multi-core CPU

### 2. Clone Repository
```bash
git clone https://github.com/Yugandhar-112/AeroStreamID.git
cd AeroStreamID
```

### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Usage

### 1. Live Local Testing (Webcam Baseline)
Test the naive pipeline using your integrated laptop webcam:
```bash
python src/main.py --source 0
```

### 2. Live UAV Stream (ESP32-CAM)
Connect your laptop to the ESP32-CAM Wi-Fi Access Point and stream the camera feed:
```bash
python src/main.py --source http://192.168.4.1:81/stream --motion-compensation
```

### 3. Offline Benchmark Clip
Process a pre-recorded aerial flight video with motion compensation and logging:
```bash
python src/main.py --source test_clips/flight_01.mp4 --motion-compensation --log-dir logs
```

### 4. Full Production Pipeline
Enable all advanced modules (motion compensation, sliced tiling, reliability scoring, privacy blurring):
```bash
python src/main.py --source test_clips/flight_01.mp4 \
    --motion-compensation \
    --tiling \
    --confidence-flag \
    --privacy-blur
```

---

## 🎛️ Command-Line Interface Reference

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--source` | `str` | `0` | Device index (`0`), HTTP stream URL, or video path |
| `--model` | `str` | `yolov8n.pt` | YOLO weights path (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `--confidence`| `float` | `0.5` | Minimum detection confidence threshold |
| `--motion-compensation` | `flag` | `False` | Enable ORB + RANSAC homography compensation |
| `--tiling` | `flag` | `False` | Enable SAHI-style sliced inference for small persons |
| `--confidence-flag` | `flag` | `False` | Render real-time count reliability indicator |
| `--gap-interpolation`| `flag` | `False` | Predict track positions across dropped frames |
| `--privacy-blur` | `flag` | `False` | Apply Gaussian blur to person head regions |
| `--headless` | `flag` | `False` | Disable UI window (optimized for headless evaluation) |
| `--log-dir` | `str` | `None` | Directory to output detailed per-frame JSONL logs |

---

## 📊 Ground Truth Annotation & Evaluation

AeroStreamID provides an integrated evaluation workflow to quantify tracking accuracy and prove ego-motion compensation gains:

### Step 1: Annotate Ground Truth
Launch the built-in OpenCV annotation tool on your test footage:
```bash
python src/annotate_gt.py test_clips/flight_01.mp4 --step 5
```
- **Left Click & Drag**: Draw person bounding box
- **Number Keys (0-9)**: Specify persistent ID
- **`n` / `Right Arrow`**: Next frame
- **`s`**: Save ground truth JSON to `ground_truth/flight_01.json`

### Step 2: Run Comparative Ablation (Baseline vs. Compensated)
```bash
python src/run_evaluation.py --clip test_clips/flight_01.mp4 --gt ground_truth/flight_01.json --compare
```
*Generates a side-by-side performance table showing MOTA, IDF1, and ID Switches.*

### Step 3: Model-Size Pareto Benchmark
```bash
python src/benchmark_models.py --clip test_clips/flight_01.mp4 --plot
```
*Plots throughput (FPS) versus accuracy across YOLOv8 Nano, Small, and Medium variants.*

---

## 🔬 Academic Evaluation & Citation

This project is submitted as part of the **B.Tech Dissertation (Major Project)** at **Amity School of Engineering and Technology, Amity University Punjab, Mohali, India**.

If you use or build upon this work, please cite:
```bibtex
@misc{aerostreamid2026,
  title={AeroStreamID: Off-Board Real-Time Person Identification Pipeline for Wi-Fi Connected UAVs},
  author={Yugandhar and Contributors},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Yugandhar-112/AeroStreamID}}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
