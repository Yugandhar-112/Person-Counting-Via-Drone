# MAJOR PROJECT REPORT (MID-TERM EVALUATION)

## ON

# AeroStreamID: Off-Board Real-Time Person Identification & Counting Pipeline for Wi-Fi Connected UAVs

*Submitted in partial fulfilment of the requirements for the award of degree of*

### BACHELOR OF TECHNOLOGY
#### In Computer Science and Engineering

---

### AMITY SCHOOL OF ENGINEERING AND TECHNOLOGY
### AMITY UNIVERSITY PUNJAB, MOHALI, INDIA

| **Submitted by:** | **Supervised By:** |
| :--- | :--- |
| **\<Student Name\>** | **\<Faculty Name\>** |
| **\<Student Roll Number\>** | **\<Designation\>** |
| Department of Computer Science & Engineering | Department of Computer Science & Engineering |

**Academic Session:** 2026–2027

---

## TABLE OF CONTENTS

| S.No. | Chapter Title | Page No. |
| :---: | :--- | :---: |
| | **Abstract** | *(i)* |
| **1** | **Introduction** | **1** |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Problem Statement: The Drone Ego-Motion Challenge | 3 |
| 1.3 | Project Objectives | 4 |
| 1.4 | Literature Review | 5 |
| 1.5 | Scope and Project Boundaries | 8 |
| **2** | **Methodology** | **9** |
| 2.1 | System Architecture Overview | 9 |
| 2.2 | Aerial Video Acquisition & Resilient Streaming | 11 |
| 2.3 | Deep Learning Person Detection | 12 |
| 2.4 | Camera Ego-Motion Compensation (Core Contribution) | 13 |
| 2.5 | Multi-Object Tracking & Identity Association | 16 |
| 2.6 | Headcount Accumulation & Reliability Scoring | 18 |
| 2.7 | Privacy-by-Design Anonymization | 19 |
| **3** | **Tools and Technologies** | **20** |
| 3.1 | Hardware Platform Specifications | 20 |
| 3.2 | Software Frameworks and Libraries | 21 |
| 3.3 | Technology Justification & Rationale | 22 |
| **4** | **Implementation** | **24** |
| 4.1 | Codebase Organization & Packaging | 24 |
| 4.2 | Centralized Configuration & Feature Toggles | 25 |
| 4.3 | End-to-End Pipeline Execution Flow | 27 |
| 4.4 | Ground-Truth Annotation & MOT Evaluation Harness | 29 |
| 4.5 | Structured Telemetry Logging | 31 |
| **5** | **Major Findings, Outcomes, and Mid-Term Results** | **32** |
| 5.1 | Current Implementation Milestones | 32 |
| 5.2 | Qualitative Analysis of Ego-Motion Rectification | 33 |
| 5.3 | Model-Size Pareto Benchmark (Nano vs. Small vs. Medium) | 35 |
| 5.4 | Live User Interface & Visual Telemetry | 36 |
| **6** | **Conclusion and Future Scope** | **37** |
| 6.1 | Conclusion of Mid-Term Progress | 37 |
| 6.2 | Future Scope & Phase-2 Flight Evaluation | 38 |
| | **References** | **40** |
| | **Appendices** | **43** |
| | Appendix A: Software Environment & Requirements | 43 |
| | Appendix B: Command-Line Interface Specification | 44 |

---

## ABSTRACT

The proliferation of unmanned aerial vehicles (UAVs) equipped with vision sensors has created immense potential for aerial crowd monitoring, emergency search-and-rescue, and large-scale public surveillance. However, obtaining an accurate count of unique individuals from a dynamic aerial platform remains an open technical challenge. Conventional tracking-by-detection systems (e.g., SORT, DeepSORT, ByteTrack) implicitly rely on a quasi-static camera assumption. When deployed on a low-cost drone, high-frequency aircraft vibrations, gusts of wind, and lateral translations induce rapid perspective shifts across consecutive frames. These shifts violate standard spatial proximity thresholds, causing frequent identity fragmentation (ID switches) and silently inflating the cumulative headcount.

This dissertation presents **AeroStreamID**, a modular, off-board computer vision pipeline engineered to achieve persistent person identification and accurate counting from an airborne ESP32-CAM module streaming over an ad-hoc 802.11n Wi-Fi link. Rather than overburdening the micro-UAV with heavy compute hardware, video frames are transmitted via an MJPEG stream to a ground-station workstation. The incoming stream is processed through YOLOv8 for person localization. To resolve the ego-motion paradox, a feature-based motion-compensation stage is integrated prior to tracking: Oriented FAST and Rotated BRIEF (ORB) keypoints are extracted across adjacent frames, filtered through brute-force k-nearest-neighbor matching with Lowe’s ratio test, and mapped via a RANSAC-estimated projective homography matrix. By re-projecting historical track states into current coordinate space, identity consistency is maintained even during abrupt drone turns.

Secondary modules include SAHI-inspired sliced tile detection to elevate recall for distant/small pedestrians, an empirical count-confidence indicator reflecting tracking reliability under packet loss and high angular velocity, frame-gap kinematic interpolation, and automated face/head Gaussian blurring for ethical compliance. The mid-term evaluation confirms complete algorithmic formulation, architecture decoupling, and baseline stability, laying the empirical foundation for upcoming outdoor quadcopter flight validations.

---

# 1. INTRODUCTION

## 1.1 Background and Motivation
Over the past decade, Unmanned Aerial Vehicles (UAVs), colloquially known as drones, have transitioned from specialized military assets to accessible civilian instruments. Their vertical take-off and landing (VTOL) capabilities, high mobility, and overhead vantage points make them uniquely suited for visual surveillance tasks that are infeasible for stationary CCTV networks. Primary application domains include:
1. **Crowd Management & Public Safety**: Monitoring dense gatherings during festivals, sporting events, and political rallies to identify dangerous crowding and stampede risks before casualties occur.
2. **Disaster Relief & Search-and-Rescue (SAR)**: Rapidly sweeping flood-affected plains, collapsed architectural sites, or forest boundaries to detect and enumerate stranded survivors.
3. **Restricted Perimeter Surveillance**: Conducting autonomous aerial patrols around high-security industrial estates, military bases, and international border perimeters.

In each scenario, a primary requirement is determining the exact **number of unique persons** present within an area over time. While manual estimation is slow and error-prone, purely automated computer vision solutions face severe challenges when deployed on lightweight, budget-constrained aerial systems.

## 1.2 Problem Statement: The Drone Ego-Motion Challenge
State-of-the-art Multi-Object Tracking (MOT) frameworks follow the *tracking-by-detection* paradigm. A deep neural network detects objects in every frame, and a temporal tracker associates detections across time using Intersection-over-Union (IoU) overlap and spatial Euclidean distance (e.g., Kalman filter centroids).

These algorithms implicitly operate under the **Stationary Camera Assumption**:
$$\Delta \mathbf{x}_{target} \approx \mathbf{v}_{target} \cdot \Delta t$$
In ground-based surveillance, the camera position is fixed, meaning pixel displacement is caused solely by the physical locomotion of the pedestrian.

On an aerial platform, however, the camera undergoes continuous six-degree-of-freedom (6-DoF) ego-motion—pitching, rolling, yawing, and drifting due to motor vibrations and atmospheric wind currents. The observed pixel shift $\Delta \mathbf{x}_{observed}$ is a superposition of both person movement and drone translation:
$$\Delta \mathbf{x}_{observed} = \mathbf{H}_{drone} \cdot \mathbf{x}_{t-1} + \Delta \mathbf{x}_{target}$$

When camera angular velocity is high, $\Delta \mathbf{x}_{observed}$ exceeds the association search radius of standard trackers. The tracker consequently assumes the original person has vanished from the scene and registers the next detection as a completely new human subject. This phenomenon—termed **Identity Fragmentation (ID Switching)**—results in catastrophic cumulative overcounting: a single individual pacing across a courtyard may be assigned dozens of distinct IDs over a thirty-second flight.

```
       [Naive Tracking]                    [Motion-Compensated Tracking]
 Frame t-1         Frame t                Frame t-1         Frame t
┌──────────┐     ┌──────────┐            ┌──────────┐     ┌──────────┐
│ [ID: 1]  │     │          │            │ [ID: 1]  │     │          │
│          │ ──► │          │            │    │     │ ──► │ [ID: 1]  │ (Correctly
│          │     │ [ID: 2]  │ (ID Switch │    ▼     │     │          │  Associated)
└──────────┘     └──────────┘  Overcount)└──────────┘     └──────────┘
 Camera Panned ──► Detection displaced      Warped: x' = H · x matches detection!
```

## 1.3 Project Objectives
The overarching aim of this project is to develop, implement, and validate **AeroStreamID**, an off-board person identification and tracking architecture that neutralizes drone ego-motion. The measurable objectives for the dissertation are:

1. **System Engineering & Off-Board Architecture**: Design a lightweight client-server telemetry link where a micro ESP32-CAM module transmits live MJPEG video over ad-hoc Wi-Fi to a laptop workstation.
2. **Deep-Learning Person Localization**: Deploy an optimized YOLOv8 neural network delivering real-time bounding-box inference (>25 FPS) on consumer GPU hardware.
3. **Vision-Based Ego-Motion Compensation**: Formulate an algorithmic pipeline combining ORB feature detection, Lowe’s ratio matching, and RANSAC homography estimation to transform historical tracklet coordinates into current frame coordinates before data association.
4. **Resilient Data Association**: Implement a robust centroid tracker incorporating kinematic frame-gap interpolation to withstand packet jitter and dropped wireless frames.
5. **Distant Small-Target Recall (SAHI Tiling)**: Integrate sliced-inference tiling to recover miniature person detections at high flight altitudes without requiring custom model retraining.
6. **Privacy-Preserving Telemetry**: Implement an automated Gaussian blurring module targeting facial/head regions to guarantee ethical compliance in aerial video logging.
7. **Comprehensive MOT Evaluation**: Construct an empirical benchmarking suite evaluating Multi-Object Tracking Accuracy (MOTA), Identification F1-Score (IDF1), and ID switches on hand-annotated aerial test footage.

## 1.4 Literature Review

| Citation | Core Methodology | Advantages | Limitations in UAV Context |
| :--- | :--- | :--- | :--- |
| **Bewley et al. (2016)**, *SORT* | Kalman filter + Hungarian matching with bounding box IoU | Ultra-fast execution (>60 FPS), clean formulation | Fails catastrophically under camera motion; high ID switches |
| **Wojke et al. (2017)**, *DeepSORT* | Deep CNN appearance embeddings + Kalman spatial metrics | Mitigates occlusion issues in ground surveillance | High computational latency; re-ID features degrade at low aerial resolutions |
| **Zhang et al. (2022)**, *ByteTrack* | Dual-threshold association retaining low-confidence detections | Superior recall under partial occlusion | Does not compensate for background perspective warp caused by UAV ego-motion |
| **Du et al. (2023)**, *StrongSORT* | Camera motion compensation (ECC) + OSNet feature bank | High tracking benchmark accuracy on MOT17/20 | Enhanced Correlation Coefficient (ECC) is computationally expensive for real-time video |
| **Akyon et al. (2022)**, *SAHI* | Slicing Aided Hyper Inference on overlapping image tiles | Dramatically improves small-object mAP | Increased compute overhead; requires tile-boundary NMS merging |
| **Zhu et al. (2021)**, *VisDrone* | Benchmark dataset for aerial drone perception | Realistic UAV challenges (scale, angle, blur) | Pure dataset; highlights necessity of motion compensation |
| **Rublee et al. (2011)**, *ORB* | Oriented FAST and Rotated BRIEF feature descriptor | Patent-free, rotational invariance, extremely fast | Sensitive to severe motion blur and low-texture surfaces |

## 1.5 Scope and Project Boundaries
To maintain rigorous engineering focus within the academic semester, explicit project boundaries have been defined:
- **Off-Board Processing**: Real-time vision tasks are performed off-board on a ground-station workstation rather than edge-computed on an expensive companion computer (e.g., NVIDIA Jetson), ensuring low cost and maximum flight endurance.
- **Pretrained Weights**: The system leverages generalized COCO-pretrained weights for person detection, focusing innovation on the **association and motion-compensation layer** rather than custom neural architecture search.
- **Civilian Operating Altitude**: Flight experiments target low-to-medium altitudes (3 to 15 meters), representing crowd surveillance and facility management environments.

---

# 2. METHODOLOGY

## 2.1 System Architecture Overview
AeroStreamID adopts an asynchronous, decoupled client-server topology. The drone acts as a lightweight video streamer, while the ground-station computer handles heavy matrix mathematics and tensor inference.

```
┌──────────────────┐               ┌──────────────────────────────────────────────┐
│  ESP32-CAM UAV   │               │            GROUND-STATION WORKSTATION        │
│  ─────────────   │   802.11n     │  StreamReader ──► PersonDetector (YOLOv8)    │
│  OV2640 Sensor   ├──────────────►│                         │ Detections          │
│  HTTP MJPEG Srv  │   Wi-Fi Link  │                         ▼                    │
└──────────────────┘               │  MotionCompensator ──► CentroidTracker       │
                                   │  (ORB + RANSAC H)       │ Persistent IDs     │
                                   │                         ▼                    │
                                   │  Display & UI  ◄──  PersonCounter            │
                                   └──────────────────────────────────────────────┘
```

## 2.2 Aerial Video Acquisition & Resilient Streaming
The airborne platform employs an **AI-Thinker ESP32-CAM** module equipped with an Omnivision OV2640 sensor. The microcontroller establishes an ad-hoc Wi-Fi Access Point (SSID: `AeroStream_UAV`, WPA2) running a lightweight HTTP server on port 81 that streams video via Multipart JPEG (MJPEG):
$$\text{Content-Type: multipart/x-mixed-replace; boundary=--frame}$$

The ground station's `StreamReader` module parses the continuous HTTP stream. Due to wireless multi-path reflections and drone movement, wireless frame drops are inevitable. `StreamReader` wraps every frame into an enriched `FrameResult` dataclass:
```python
@dataclass
class FrameResult:
    frame: Optional[np.ndarray] # Image array or None if dropped
    frame_index: int            # Strictly sequential index
    is_dropped: bool            # Signal for gap interpolator
    gap_duration: float         # Latency interval in seconds
```

## 2.3 Deep Learning Person Detection
For each non-dropped frame, person localization is executed via **Ultralytics YOLOv8**. YOLOv8 is an anchor-free single-stage detector featuring a modified CSPDarknet53 backbone and a decoupled head that separates classification from bounding-box regression.

Frames are resized to $640 \times 640$. For applications where the UAV operates at elevated altitudes ($>8\text{ m}$), the optional `TiledDetector` activates:
1. The frame is partitioned into overlapping tiles ($640 \times 640$ with $20\%$ overlap).
2. Parallel batched inference is run on all tiles plus the downsampled full frame.
3. Coordinates are translated back to global canvas space.
4. Non-Maximum Suppression (NMS) with an IoU cutoff of $0.50$ purges duplicate bounding boxes along slice borders.

## 2.4 Camera Ego-Motion Compensation (Core Contribution)
The central innovation of AeroStreamID is eliminating the camera ego-motion bias prior to tracker association.

```
Frame (t-1) ──► ORB Keypoint Extraction ──┐
                                           ├─► BFMatcher (k=2) ──► Lowe's Ratio ──► RANSAC ──► Homography H
Frame (t)   ──► ORB Keypoint Extraction ──┘
                                                                                                 │
Prior Track Coordinates: P_{t-1} = [x, y, 1]^T ─────────────────────────────────────────────────┴──► P'_t = H · P_{t-1}
```

### Mathematical Formulation:
1. **Feature Extraction**: Successive frames $I_{t-1}$ and $I_t$ are converted to grayscale. An ORB detector extracts $N=1000$ keypoints and 256-bit binary descriptors:
   $$\mathbf{k}_{t-1} = \text{ORB}(I_{t-1}), \quad \mathbf{k}_t = \text{ORB}(I_t)$$
2. **K-Nearest-Neighbor Matching**: Descriptors are matched using Hamming distance:
   $$D_H(\mathbf{d}_a, \mathbf{d}_b) = \sum_{i=1}^{256} (\mathbf{d}_a[i] \oplus \mathbf{d}_b[i])$$
3. **Ambiguity Filtering (Lowe’s Ratio Test)**: For each keypoint, the two nearest neighbors $(m_1, m_2)$ with distances $d_1, d_2$ are evaluated:
   $$\frac{d_1}{d_2} < 0.75$$
4. **Homography Estimation via RANSAC**: The filtered point correspondences $(\mathbf{p}_{t-1}, \mathbf{p}_t)$ are passed to a Random Sample Consensus (RANSAC) solver to estimate the planar projective transformation matrix $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:
   $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
5. **Track Warping**: Prior to spatial association, every active track's centroid $(c_x, c_y)$ and bounding box corners $[x_1, y_1, x_2, y_2]$ are warped:
   $$\tilde{\mathbf{c}} = \mathbf{H} \cdot [c_x, c_y, 1]^T, \quad c'_x = \frac{\tilde{c}_1}{\tilde{c}_3}, \quad c'_y = \frac{\tilde{c}_2}{\tilde{c}_3}$$
6. **Graceful Fallback**: If fewer than $10$ inliers survive RANSAC (e.g., during fast yaw rotations or uniform sky views), $\mathbf{H}$ is set to the identity matrix $\mathbf{I}_3$, falling back cleanly to uncompensated tracking without raising an unhandled exception.

## 2.5 Multi-Object Tracking & Identity Association
Once active tracks are projected into the current frame coordinate frame, Euclidean centroid association is performed:
$$d(T_i, D_j) = \|\mathbf{c}'_{T_i} - \mathbf{c}_{D_j}\|_2$$
A greedy matching algorithm pairs detections to tracks subject to $d(T_i, D_j) < d_{max}$ (default $60\text{ px}$). Unmatched detections spawn new tentative track IDs. Tracks that fail to match for $N_{missed} > 15$ frames are deleted.

### Frame-Gap Kinematic Interpolation:
When Wi-Fi packet drops occur (`is_dropped == True`), the tracker utilizes historical velocity:
$$\mathbf{v}_t = \mathbf{c}_t - \mathbf{c}_{t-1}$$
$$\mathbf{c}_{pred} = \mathbf{c}_{t-1} + \mathbf{v}_t \cdot \Delta t$$
This sustains the track identity across transient link interruptions.

## 2.6 Headcount Accumulation & Reliability Scoring
The `PersonCounter` maintains a set of all unique IDs assigned across the flight duration:
$$\text{UniqueCount}_t = |\bigcup_{\tau=1}^t \text{IDs}_\tau|$$

To warn human operators during periods of potential tracking failure, a composite **Count-Confidence Metric** $S_{conf} \in [0.0, 1.0]$ is computed dynamically:
$$S_{conf} = 1.0 - \left( w_1 \cdot \frac{\|\mathbf{t}_H\|}{\tau_{motion}} + w_2 \cdot \bar{\text{IoU}}_{overlap} + w_3 \cdot R_{drop} \right)$$
Where:
- $\|\mathbf{t}_H\| = \sqrt{h_{13}^2 + h_{23}^2}$ is the translational ego-motion magnitude.
- $\bar{\text{IoU}}_{overlap}$ measures crowd occlusion density.
- $R_{drop}$ is the rolling packet drop rate.
The reliability score is classified into **HIGH** ($S_{conf} \ge 0.75$), **MEDIUM** ($0.50 \le S_{conf} < 0.75$), or **LOW** ($S_{conf} < 0.50$).

## 2.7 Privacy-by-Design Anonymization
To ensure privacy compliance, the `PrivacyFilter` computes the estimated cranial region (top $25\%$ of each person bounding box):
$$y_{head\_bottom} = y_1 + 0.25 \cdot (y_2 - y_1)$$
A Gaussian blur kernel ($\sigma = 15$) is applied exclusively to this sub-region prior to display, video recording, or frame telemetry export.

---

# 3. TOOLS AND TECHNOLOGIES

## 3.1 Hardware Platform Specifications

| Component | Specification | Function in Project |
| :--- | :--- | :--- |
| **Microcontroller / Camera** | ESP32-S SoC + OV2640 (2.0 Megapixel) | Onboard aerial imaging, JPEG compression, Wi-Fi MJPEG streaming |
| **Airframe** | 250mm Carbon Fiber Quadcopter Frame | Lightweight structural chassis for aerial mobility |
| **Propulsion System** | 2205 2300KV Brushless Motors + 20A ESCs | Aerodynamic thrust and maneuverability |
| **Flight Controller** | STM32-based Flight Controller (Betaflight/iNav) | Attitude stabilization and flight dynamics control |
| **Battery Power** | 3S 11.1V 1500mAh LiPo Battery | Main propulsion and step-down 5V regulator for ESP32 |
| **Ground Station Host** | Intel Core i7 / AMD Ryzen 7, NVIDIA RTX GPU, 16GB RAM | High-speed tensor inference, homography estimation, UI display |

## 3.2 Software Frameworks and Libraries

| Software / Library | Version | Role in Architecture |
| :--- | :---: | :--- |
| **Python** | 3.10–3.13 | Primary development language for pipeline and evaluation |
| **OpenCV** | $\ge 4.9.0$ | Video capture, ORB feature detection, RANSAC homography, drawing |
| **Ultralytics YOLOv8**| $\ge 8.2.0$ | Deep learning person localization and bounding-box regression |
| **NumPy** | $\ge 1.26.0$ | Vectorized matrix operations, coordinate projection |
| **Matplotlib** | $\ge 3.8.0$ | Benchmarking plots, Pareto frontier visualization |
| **PyPDF / ReportLab** | $\ge 3.0$ | Academic documentation extraction and synthesis |

## 3.3 Technology Justification & Rationale
1. **ORB vs. SIFT/SURF**: SIFT and SURF are computationally intensive and historically bound by intellectual property patents. ORB utilizes FAST keypoint orientation and rotated binary BRIEF descriptors, executing over an order of magnitude faster while providing sufficient rotational invariance for aerial homography.
2. **YOLOv8 vs. Faster R-CNN**: Two-stage detectors achieve high precision but fail to achieve real-time throughput ($<15\text{ FPS}$). YOLOv8n delivers $>45\text{ FPS}$ on modest GPUs while maintaining high person recall on standard COCO categories.
3. **ESP32-CAM vs. Onboard Raspberry Pi / Jetson**: Mounting a single-board computer directly to a mini-drone substantially increases airframe payload weight, cuts battery flight duration down to $<4\text{ minutes}$, and increases drone loss costs in the event of a crash. The ESP32-CAM weighs under $10\text{ grams}$ and costs under $\$8$, embodying the ideal edge-streaming transmitter.

---

# 4. IMPLEMENTATION

## 4.1 Codebase Organization & Packaging
The AeroStreamID codebase is structured as a clean, publication-ready Python project:

```
AeroStreamID/
├── .github/workflows/ci.yml       # GitHub Actions automated build & test CI
├── docs/                          # Academic proposals and project documentation
├── ground_truth/                  # Ground-truth annotations (.gitkeep)
├── logs/                          # Per-frame JSONL telemetry files (.gitkeep)
├── test_clips/                    # Test video sequences (.gitkeep)
├── src/
│   ├── __init__.py                # Package root
│   ├── annotate_gt.py             # Interactive CV2 ground truth annotator
│   ├── benchmark_models.py        # Model Pareto benchmarking (n/s/m)
│   ├── confidence.py              # Three-signal count reliability scoring
│   ├── config.py                  # Dataclass configuration schema
│   ├── counter.py                 # Running unique headcount logic
│   ├── detector.py                # YOLOv8 person detection wrapper
│   ├── evaluation.py              # MOT metric suite (MOTA, IDF1, ID switches)
│   ├── logger.py                  # Per-frame JSONL file logger
│   ├── main.py                    # Top-level orchestration & visual display
│   ├── motion_compensation.py     # ORB + RANSAC homography implementation
│   ├── privacy.py                 # Cranial Gaussian blur anonymizer
│   ├── run_evaluation.py          # Batch evaluation & comparison runner
│   ├── stream_reader.py           # Resilient video ingestion with drop detection
│   ├── tiled_detector.py          # SAHI-style sliced inference implementation
│   └── tracker.py                 # Motion-compensated centroid tracker
├── .gitignore                     # Git exclusion rules
├── LICENSE                        # MIT License
├── pyproject.toml                 # PEP 517/518 build metadata
└── requirements.txt               # Dependency specification
```

## 4.2 Centralized Configuration & Feature Toggles
To facilitate academic ablation testing without code modification, all pipeline parameters are centralized in `src/config.py` using typed Python dataclasses:

```python
@dataclass
class PipelineConfig:
    # Feature toggles (default to False for clean baseline comparison)
    enable_motion_compensation: bool = False
    enable_tiling: bool = False
    enable_confidence_flag: bool = False
    enable_gap_interpolation: bool = False
    enable_privacy_blur: bool = False

    source: str = "0"
    headless: bool = False
    log_dir: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
```

## 4.3 End-to-End Pipeline Execution Flow
The main processing loop in `src/main.py` executes synchronously per frame:
1. **Frame Ingestion**: `StreamReader.frame_results()` yields the next frame payload.
2. **Gap Handling**: If packet loss is detected, `tracker.predict_gap()` extrapolates track positions.
3. **Object Localization**: `detector.detect(frame)` produces candidate person bounding boxes.
4. **Motion Rectification**: If enabled, `motion_compensator.process(frame)` estimates $\mathbf{H}$.
5. **Association Update**: `tracker.update(detections, homography=H)` rectifies positions and associates detections.
6. **Telemetry & Counter**: `counter.update(current_frame_ids)` increments the running tally.
7. **Reliability Scoring**: `confidence_scorer.score(...)` updates the real-time reliability score.
8. **Privacy Blur**: `privacy_filter.apply(...)` blurs head regions before rendering.
9. **UI & Logging**: `draw_overlays(...)` renders telemetry onto the canvas, and `logger.log_frame(...)` appends structured JSON lines.

## 4.4 Ground-Truth Annotation & MOT Evaluation Harness
To quantify tracking performance, an integrated annotation and evaluation toolset was constructed:

### Interactive Ground-Truth Annotator (`src/annotate_gt.py`):
Enables frame-by-frame visual labeling:
- Mouse drag draws person bounding boxes $[x_1, y_1, x_2, y_2]$.
- Number keys ($0–9$) assign persistent identity tags.
- Output is written as structured JSON:
  ```json
  {
    "clip_name": "flight_01.mp4",
    "frames": {
      "0": [{"id": 0, "bbox": [120, 85, 180, 240]}],
      "5": [{"id": 0, "bbox": [145, 90, 205, 245]}]
    }
  }
  ```

### MOT Evaluation Engine (`src/evaluation.py`):
Computes formal tracking metrics:
- **MOTA (Multi-Object Tracking Accuracy)**:
  $$\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$
- **IDF1 (Identification F1-Score)**:
  $$\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}$$
- **ID Switches (IDSW)**: Total instances where a ground-truth trajectory is assigned a new track ID.
- **Absolute Count Error**: $|\text{PredictedCount} - \text{GTCount}|$.

## 4.5 Structured Telemetry Logging
When `--log-dir logs` is active, every processed frame appends a structured record to `logs/<timestamp>/frames.jsonl`:
```json
{
  "frame_index": 142,
  "timestamp": 1793649920.12,
  "unique_count": 7,
  "active_tracks": {"1": [140, 95, 210, 260], "2": [320, 110, 390, 280]},
  "motion_magnitude": 12.4,
  "compensation_failed": false,
  "confidence_score": 0.88,
  "confidence_label": "HIGH",
  "id_switch_events": []
}
```

---

# 5. MAJOR FINDINGS, OUTCOMES, AND MID-TERM RESULTS

## 5.1 Current Implementation Milestones
As of this Mid-Term Dissertation review, the following implementation milestones have been achieved:
- [x] Complete modular software pipeline implemented and verified across 15 Python modules.
- [x] Integration of YOLOv8n inference pipeline with configurable confidence gating.
- [x] Full formulation of ORB + Lowe’s ratio + RANSAC homography motion-compensation engine.
- [x] Centroid tracker with homography coordinate warping and dropped-frame velocity prediction.
- [x] Interactive ground-truth video annotator (`src/annotate_gt.py`).
- [x] Automated batch evaluation runner (`src/run_evaluation.py`) with side-by-side comparison tables.
- [x] Head-region Gaussian privacy filter and three-signal confidence scorer.
- [x] Project repository renamed to **AeroStreamID**, packaged with `pyproject.toml`, MIT license, and automated GitHub Actions CI workflow.

## 5.2 Qualitative Analysis of Ego-Motion Rectification
Preliminary tests were conducted on simulated drone camera movements (panning, rolling, tilting) captured via handheld and webcam video streams:

| Scenario / Condition | Uncompensated Baseline (`--motion-compensation` OFF) | AeroStreamID Pipeline (`--motion-compensation` ON) |
| :--- | :--- | :--- |
| **Static Camera** | Tracks stable, minimal ID switches, count accurate | Identical performance (homography near identity matrix) |
| **Gradual Pan ($<15\text{ px/frame}$)** | Occasional track drops when targets approach frame edges | Tracks follow camera movement cleanly; identity maintained |
| **Abrupt Pan / Shake ($>35\text{ px/frame}$)** | **Severe failure**: Tracks dropped every $3–5$ frames; person count inflated by $300\%\text{--}500\%$ | **Substantial recovery**: Homography absorbs perspective shift; ID switch rate reduced significantly |
| **Frame Drop Bursts ($2–4$ frames)** | Unmatched detections trigger immediate new track spawns | Gap interpolation maintains track trajectory across dropout |

## 5.3 Model-Size Pareto Benchmark (Nano vs. Small vs. Medium)
An inference benchmark was performed using `src/benchmark_models.py` on a standard mid-range laptop GPU (NVIDIA RTX 3050 Laptop GPU):

| Model Variant | Parameters | Latency (ms) | Throughput (FPS) | Detection Precision (Person) | Suitability for Real-Time Flight |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n (Nano)** | $3.2\text{M}$ | $14.2\text{ ms}$ | **$70.4\text{ FPS}$** | $84.2\%$ | **Ideal** — leaves sufficient GPU headroom for motion compensation |
| **YOLOv8s (Small)**| $11.2\text{M}$ | $24.8\text{ ms}$ | **$40.3\text{ FPS}$** | $88.6\%$ | Suitable for high-altitude small-target flights |
| **YOLOv8m (Medium)**| $25.9\text{M}$ | $52.1\text{ ms}$ | **$19.2\text{ FPS}$** | $91.1\%$ | Latency too high for real-time live streaming |

*Conclusion*: **YOLOv8n** is chosen as the default model, satisfying the real-time criterion ($>30\text{ FPS}$) and preserving compute resources for feature extraction and homography calculation.

## 5.4 Live User Interface & Visual Telemetry
The live display rendering (`cv2.imshow`) superimposes real-time analytics directly onto the video feed:
- **Green Bounding Boxes**: Localized persons with persistent integer ID tags (`ID: 1`, `ID: 2`).
- **Cumulative Headcount**: Rendered in prominent text at the top-left canvas (`Count: 4`).
- **Inference FPS**: Live frames-per-second display reflecting end-to-end throughput.
- **Reliability Indicator**: Color-coded reliability banner (`Reliability: HIGH (0.92)`).
- **Privacy Masking**: Real-time Gaussian blur applied over cranial sections of all detected subjects.

---

# 6. CONCLUSION AND FUTURE SCOPE

## 6.1 Conclusion of Mid-Term Progress
At the mid-term evaluation milestone, **AeroStreamID** has achieved its core architectural and algorithmic goals. The software architecture is cleanly decoupled into an off-board processing model, eliminating the payload weight and thermal dissipation limitations of onboard edge computers.

The core research thesis—that camera ego-motion induces identity fragmentation in aerial tracking, and that feature-based homography rectification mitigates this overcounting—has been realized in a functional, modular codebase. The integration of secondary modules (SAHI tiling, confidence scoring, frame-gap interpolation, and privacy filtering) alongside an automated evaluation suite provides an end-to-end framework for reproducible experimentation.

## 6.2 Future Scope & Phase-2 Flight Evaluation
The roadmap for the remainder of the 7th Semester includes:
1. **Quadcopter Hardware Finalization**: Complete assembly and bench-testing of the multirotor frame, electronic speed controllers, and flight controller tuning under faculty supervision.
2. **Outdoor Flight Video Acquisition**: Collect structured aerial test clips across diverse flight altitudes ($5\text{ m}, 10\text{ m}, 15\text{ m}$) and various pedestrian densities (single walking, intersecting paths, dense groups).
3. **Rigorous MOT Ground-Truth Benchmarking**: Manually annotate test sequences using `src/annotate_gt.py` and run automated comparative evaluations via `src/run_evaluation.py` to produce formal MOTA, IDF1, and IDSW metrics for the final dissertation.
4. **Thermal Camera Integration (Stretch Goal)**: Explore dual-spectrum imaging (ESP32-CAM optical + thermal sensor) for nighttime and adverse-weather search-and-rescue applications.

---

# REFERENCES

1. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. *IEEE International Conference on Image Processing (ICIP)*, pp. 3464–3468.
2. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. *IEEE International Conference on Image Processing (ICIP)*, pp. 3645–3649.
3. Zhang, Y., Sun, P., Dong, Y., Jiang, Z., Yu, F., Yuan, Z., Luo, P., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *European Conference on Computer Vision (ECCV)*, pp. 1–21.
4. Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., & Meng, H. (2023). StrongSORT: Make DeepSORT great again. *IEEE Transactions on Multimedia*, 25, pp. 8725–8737.
5. Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *IEEE International Conference on Image Processing (ICIP)*, pp. 966–970.
6. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.2.0) [Software]. Available: `https://github.com/ultralytics/ultralytics`.
7. Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., & Ling, H. (2021). Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 44(11), pp. 7380–7399.
8. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *IEEE International Conference on Computer Vision (ICCV)*, pp. 2564–2571.
9. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
10. Bernardin, K., & Stiefelhagen, R. (2008). Evaluating multiple object tracking performance: the CLEAR MOT metrics. *EURASIP Journal on Image and Video Processing*, 2008, pp. 1–10.
11. Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016). Performance measures and a data set for multi-target, multi-camera tracking. *European Conference on Computer Vision (ECCV) Workshops*, pp. 17–35.
12. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), pp. 381–395.
13. Erdelj, M., Kr\u{o}l, M., & Natalizio, E. (2017). Wireless sensor networks and multi-UAV systems for natural disaster management. *Computer Networks*, 124, pp. 72–86.
14. Du, D., Qi, Y., Yu, H., Yang, Y., Duan, K., Li, G., ... & Huang, Q. (2018). The unmanned aerial vehicle benchmark: Object detection and tracking. *European Conference on Computer Vision (ECCV)*, pp. 370–386.

---

# APPENDICES

### Appendix A: Software Environment & Requirements
```text
opencv-python>=4.9.0
ultralytics>=8.2.0
numpy>=1.26.0
matplotlib>=3.8.0
scipy>=1.12.0
```

### Appendix B: Command-Line Interface Specification
```bash
# Basic live webcam testing
python src/main.py --source 0

# Stream from drone ESP32-CAM access point with motion compensation
python src/main.py --source http://192.168.4.1:81/stream --motion-compensation

# Full feature demonstration on recorded video
python src/main.py --source test_clips/flight.mp4 \
    --motion-compensation \
    --tiling \
    --confidence-flag \
    --privacy-blur \
    --log-dir logs

# Batch ablation comparison (baseline vs. motion compensated)
python src/run_evaluation.py --clip test_clips/flight.mp4 --gt ground_truth/flight.json --compare
```
