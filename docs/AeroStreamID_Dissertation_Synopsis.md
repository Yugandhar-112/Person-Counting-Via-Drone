# AeroStreamID: Off-Board Real-Time Person Identification Pipeline for Wi-Fi Connected UAVs

**B.Tech Major Project Synopsis — Mid-Term Evaluation**

| | |
|---|---|
| **University** | Amity University Punjab, Mohali, India |
| **School** | Amity School of Engineering and Technology |
| **Programme** | B.Tech (Computer Science & Engineering) |
| **Semester** | Aug 2026 – Dec 2026 |

---

## 1. Abstract

Unmanned Aerial Vehicles (UAVs) equipped with lightweight cameras offer a compelling platform for real-time crowd monitoring, disaster-response situational awareness, and public-event management. However, reliable person counting from a moving aerial platform remains an open challenge. Conventional multi-object tracking algorithms—designed for fixed, ground-level CCTV feeds—implicitly assume a stationary camera. When deployed on a drone, the continuous ego-motion of the aircraft induces apparent motion across the entire field of view, causing centroid-based and IoU-based trackers to fragment identities, reassign track IDs, and consequently overcount the number of unique individuals in a scene.

This project presents **AeroStreamID**, an off-board, real-time person identification and counting pipeline that receives an MJPEG video stream from an ESP32-CAM module mounted on a self-assembled quadcopter over a Wi-Fi link. The captured frames are processed on a ground-station laptop where a YOLOv8 object detector localises persons, and a novel motion-compensation stage—based on ORB feature extraction, brute-force k-nearest-neighbour matching with Lowe's ratio test, and RANSAC-robust homography estimation—warps previous-frame track centroids into the current frame's coordinate system before data association is performed. This ego-motion rectification step allows a lightweight centroid tracker to maintain consistent identities across frames despite significant camera translation and rotation.

The system is evaluated against hand-annotated ground-truth sequences using standard Multi-Object Tracking (MOT) metrics, including MOTA, IDF1, and the number of ID switches. Preliminary results indicate that the motion-compensated pipeline reduces ID switches by a substantial margin compared to the uncompensated baseline, yielding more accurate unique-person counts. Secondary enhancements including tiled (SAHI-style) detection for small-object recall, count-confidence scoring, privacy-preserving face blurring, and model-size benchmarking are also investigated.

---

## 2. Introduction

### 2.1 Background and Motivation

The proliferation of small, low-cost multirotor UAVs has created new opportunities for aerial visual intelligence in domains where rapid, wide-area situational awareness is critical. In **crowd monitoring and public-event management**, municipal authorities require accurate headcounts and density estimates to enforce occupancy limits, allocate emergency exits, and dispatch security personnel in real time (Zhu et al., 2021). During **natural disasters**—floods, earthquakes, building collapses—first responders must locate survivors scattered across inaccessible terrain; a drone-mounted camera can survey a disaster zone in minutes rather than the hours required by ground teams (Erdelj et al., 2017). In **agricultural and industrial surveillance**, periodic aerial sweeps can verify workforce deployment across large sites. In all these scenarios, the fundamental task reduces to detecting, tracking, and counting persons from a moving aerial viewpoint.

Traditional video-surveillance systems rely on fixed, infrastructure-mounted cameras whose static viewpoint simplifies the tracking problem: the background is largely stationary, and observed motion can be attributed almost entirely to the targets of interest. A rich ecosystem of multi-object trackers—SORT (Bewley et al., 2016), DeepSORT (Wojke et al., 2017), ByteTrack (Zhang et al., 2022)—has been developed and benchmarked under this assumption. When the same algorithms are naïvely applied to drone footage, however, the camera's own translational and rotational motion dominates the inter-frame displacement of every pixel, including those belonging to persons on the ground. The resulting apparent motion vectors are large, erratic, and unrelated to the targets' true movement, causing trackers to lose associations, create spurious new tracks, and inflate person counts.

This degradation is not merely theoretical. Empirical studies on UAV-based MOT datasets such as VisDrone (Zhu et al., 2021) and UAVDT (Du et al., 2018) report significantly higher ID-switch rates compared to ground-camera benchmarks like MOT17, even when the same detector–tracker combination is used. The problem is exacerbated at low frame rates—common when streaming over bandwidth-constrained wireless links—because the per-frame displacement grows proportionally with the inter-frame interval.

Addressing this gap requires an explicit **motion-compensation** stage that estimates and removes the camera's ego-motion before track association is performed. While inertial measurement units (IMUs) can provide rotation estimates, they are noisy, subject to drift, and unavailable on ultra-low-cost camera modules such as the ESP32-CAM. A purely vision-based approach—extracting sparse features from the scene background, matching them across consecutive frames, and computing a projective transformation (homography)—offers a self-contained solution that requires no additional sensors and naturally accounts for both translation and rotation in the image plane.

### 2.2 Problem Statement

The specific technical problem addressed by this project is as follows:

> Given a low-resolution (VGA, 640 × 480), low-frame-rate (8–12 fps) MJPEG video stream transmitted over Wi-Fi from an ESP32-CAM mounted on a multirotor UAV, design and implement an off-board software pipeline that (a) detects all persons visible in each frame, (b) assigns consistent identity labels across frames despite continuous camera ego-motion, and (c) maintains an accurate cumulative count of unique persons observed, with minimal overcounting due to ID fragmentation.

The core technical challenge is that the camera undergoes six-degree-of-freedom motion (three translational and three rotational axes), which projects onto the 2D image plane as a perspective transformation. Between consecutive frames, this transformation can displace a person's bounding-box centroid by tens or even hundreds of pixels—far exceeding the displacement thresholds used by standard centroid or IoU trackers. Without compensation, the tracker interprets each displaced detection as a new individual, fragmenting a single person's trajectory into multiple short tracklets and inflating the unique-person count.

A secondary challenge arises from the constrained hardware: the ESP32-CAM module provides only VGA resolution and JPEG-compressed frames streamed over an 802.11n link, introducing compression artefacts, dropped frames, and variable latency. The pipeline must be robust to these imperfections while maintaining real-time throughput on a consumer-grade laptop GPU.

### 2.3 Objectives

The specific objectives of the project are enumerated below:

1. **Design and assemble a UAV platform** comprising a quadcopter airframe with brushless motors, electronic speed controllers (ESCs), a flight controller, a LiPo battery, and an RC receiver, with an ESP32-CAM module mounted as the imaging payload.

2. **Develop a reliable Wi-Fi video streaming subsystem** that captures MJPEG frames from the ESP32-CAM, transmits them over a local Wi-Fi network, and ingests them into a Python-based processing pipeline on a ground-station laptop with minimal latency and graceful handling of dropped frames.

3. **Implement a real-time person detection module** using a pre-trained YOLOv8 model (nano or small variant) capable of localising persons in aerial-perspective, VGA-resolution frames at ≥ 8 fps inference throughput.

4. **Design and implement a vision-based motion-compensation algorithm** that estimates the inter-frame homography from ORB feature correspondences filtered by Lowe's ratio test and refined by RANSAC, and applies the resulting projective warp to previous-frame track positions before data association.

5. **Integrate a centroid-based multi-object tracker** with the motion-compensated detections to maintain consistent person IDs across frames and derive an accurate cumulative unique-person count.

6. **Evaluate the pipeline quantitatively** against hand-annotated ground-truth sequences using MOT metrics (MOTA, IDF1, ID switches, fragmentation rate), comparing the motion-compensated tracker against an uncompensated baseline to measure the improvement attributable to ego-motion rectification.

### 2.4 Scope

**In scope:**

- Off-board (ground-station) processing of wirelessly streamed drone video.
- Use of pre-trained YOLOv8 models for person detection; no custom model training is undertaken.
- Vision-based ego-motion compensation using sparse feature matching and homography estimation.
- Quantitative evaluation on self-collected test sequences with hand-annotated ground truth.
- Secondary enhancements: tiled detection for small-object recall, count-confidence scoring, frame-gap interpolation, privacy-preserving face blurring, and inference-speed benchmarking across YOLOv8 model sizes.

**Out of scope:**

- Autonomous or GPS-waypoint-guided flight; the drone is manually piloted via an RC transmitter during data collection.
- On-board (edge) inference on the ESP32-CAM or any companion computer mounted on the UAV.
- Training or fine-tuning of custom detection models on aerial datasets.
- Re-identification of persons who leave and re-enter the field of view after an extended absence.
- Night-time or thermal-infrared operation.

---

## 3. Methodology

### 3.1 System Architecture

The AeroStreamID pipeline is organised as a linear, modular data-flow architecture comprising seven stages. The design prioritises modularity—each stage exposes a well-defined interface—so that individual components can be replaced or upgraded independently.

```
┌──────────────┐   Wi-Fi    ┌───────────────┐
│  ESP32-CAM   │──MJPEG───▶│  StreamReader  │
│  (on UAV)    │   stream   │  (HTTP GET)   │
└──────────────┘            └───────┬───────┘
                                    │ raw frame
                                    ▼
                           ┌────────────────┐
                           │  YOLOv8        │
                           │  Person        │
                           │  Detector      │
                           └───────┬────────┘
                                   │ bounding boxes
                                   ▼
                        ┌─────────────────────┐
                        │  (Optional) Tiled   │
                        │  Detection (SAHI)   │
                        └─────────┬───────────┘
                                  │ merged detections
                                  ▼
                       ┌──────────────────────┐
                       │  Motion Compensation │
                       │  (ORB + RANSAC       │
                       │   Homography)        │
                       └──────────┬───────────┘
                                  │ warped track positions
                                  ▼
                        ┌─────────────────────┐
                        │  Centroid Tracker    │
                        │  (Hungarian /        │
                        │   Greedy Matching)   │
                        └─────────┬───────────┘
                                  │ tracked IDs
                                  ▼
                       ┌──────────────────────┐
                       │  Person Counter &    │
                       │  Display / Logger    │
                       └──────────────────────┘
```

**Stage 1 — StreamReader.** The ESP32-CAM runs a lightweight HTTP server that serves JPEG frames as a multipart MJPEG stream. The `StreamReader` module on the laptop issues an HTTP GET request to the camera's stream endpoint, continuously parses the multipart boundaries, and decodes each JPEG payload into an OpenCV `numpy` array. A dedicated reader thread buffers incoming frames to decouple network jitter from downstream processing cadence. Frame timestamps and sequence numbers are recorded for offline analysis.

**Stage 2 — YOLOv8 Person Detector.** Each decoded frame is passed to an Ultralytics YOLOv8 model configured to detect only COCO class 0 (person). The model outputs a set of bounding boxes with associated confidence scores. A confidence threshold (default 0.35) suppresses spurious detections. The detector runs on the laptop's discrete GPU via CUDA when available, falling back to CPU otherwise. Bounding-box centroids are computed as the midpoint of each box for use by the tracker.

**Stage 3 — Tiled Detection (Optional).** For high-altitude flights where persons occupy very few pixels, a SAHI-style (Slicing Aided Hyper Inference) strategy subdivides the frame into overlapping tiles, runs the detector on each tile independently, and merges the resulting boxes using non-maximum suppression (NMS). This approach improves recall for small objects without retraining the model.

**Stage 4 — Motion Compensation.** This stage constitutes the principal technical contribution and is described in detail in Section 3.2.

**Stage 5 — Centroid Tracker.** A custom centroid-based tracker maintains a dictionary of active tracks, each identified by a unique integer ID. In each frame, the tracker computes pairwise Euclidean distances between the motion-compensated previous-frame centroids and the current-frame detection centroids, then solves the assignment problem using a greedy nearest-neighbour strategy (or, optionally, the Hungarian algorithm). Detections that cannot be matched within a distance threshold are registered as new tracks; tracks that receive no match for a configurable number of consecutive frames (`maxDisappeared`) are deregistered. A track is counted toward the cumulative person total only after it has been consistently matched for a minimum number of frames (`minConfirmation`), suppressing transient false positives.

**Stage 6 — Person Counter.** The counter module maintains a monotonically increasing tally of unique persons. Each newly confirmed track increments the count by one. The counter also records per-frame instantaneous counts (number of active tracks) for density estimation.

**Stage 7 — Display and Logging.** An annotated output frame is rendered in real time, overlaying bounding boxes, track IDs, and the cumulative count. Optionally, a privacy-blurring pass applies a Gaussian kernel to detected face regions before display. Quantitative logs—frame number, timestamp, detection count, active track count, cumulative count, and per-track centroids—are written to a CSV file for post-hoc analysis.

### 3.2 Motion Compensation Algorithm

The motion-compensation module is the core algorithmic contribution of this project. Its purpose is to estimate the 2D projective transformation (homography) induced by the camera's ego-motion between consecutive frames and to apply this transformation to the previous frame's track centroids, thereby expressing them in the current frame's coordinate system before data association is attempted.

The algorithm proceeds in five steps for each pair of consecutive frames $(I_{t-1}, I_t)$:

**Step 1 — Greyscale Conversion and ROI Masking.** Both frames are converted to 8-bit greyscale. Optionally, a binary mask excludes regions occupied by detected persons from the feature-extraction step, ensuring that features are drawn from the static background rather than from moving targets. This masking improves homography accuracy when persons occupy a large fraction of the frame.

**Step 2 — ORB Feature Extraction.** The Oriented FAST and Rotated BRIEF (ORB) detector–descriptor (Rublee et al., 2011) is applied to both greyscale frames. ORB is selected over alternatives such as SIFT or SURF for three reasons: (a) it is free of patent restrictions, (b) it is computationally efficient (binary descriptors matched via Hamming distance), and (c) it provides sufficient repeatability for the moderate viewpoint changes encountered in consecutive drone frames at 8–12 fps. A maximum of 1000 keypoints is retained per frame, ranked by the FAST corner response.

**Step 3 — Brute-Force k-Nearest-Neighbour Matching with Lowe's Ratio Test.** Descriptors from $I_{t-1}$ are matched against those from $I_t$ using OpenCV's `BFMatcher` with Hamming norm and `knnMatch(k=2)`. For each query descriptor, the two nearest neighbours in the train set are retrieved. A match is accepted only if the distance to the best match is less than a fraction $\rho$ (default 0.75) of the distance to the second-best match, following the ratio test proposed by Lowe (2004). This test effectively filters ambiguous matches that could arise from repetitive textures (e.g., pavement tiles, uniform rooftops).

**Step 4 — RANSAC Homography Estimation.** The surviving point correspondences are passed to `cv2.findHomography` with the RANSAC method. RANSAC iteratively samples minimal subsets of four correspondences, computes candidate homographies, and classifies all correspondences as inliers or outliers based on a reprojection-error threshold (default 5.0 pixels). The homography $H$ supported by the largest consensus set is returned. A minimum inlier count threshold (default 10) is enforced; if too few inliers remain, the homography is deemed unreliable and the identity matrix is used as a fallback, effectively disabling compensation for that frame pair.

**Step 5 — Centroid Warping.** Each active track's centroid from frame $t-1$, represented as a homogeneous coordinate vector $\mathbf{p} = [x, y, 1]^T$, is projected into frame $t$'s coordinate system via $\mathbf{p}' = H \mathbf{p}$, followed by perspective division: $x' = p'_1 / p'_3$, $y' = p'_2 / p'_3$. The warped centroids $\{(x'_i, y'_i)\}$ replace the raw previous-frame centroids in the tracker's association step. Because the homography captures the camera's global motion, the warped centroids now approximate where each track *would* appear in the current frame if it had remained stationary, isolating the target's true ground-plane motion for the association distance computation.

This five-step procedure adds approximately 5–8 ms of processing latency per frame on a mid-range laptop CPU, a negligible overhead relative to the ~30–50 ms consumed by the YOLO inference pass.

### 3.3 Evaluation Methodology

Quantitative evaluation is conducted on a set of test sequences captured by the project's own ESP32-CAM-equipped drone under controlled conditions (known number of persons, varied flight patterns: hover, lateral sweep, forward advance, altitude change). Ground truth is established by manual frame-by-frame annotation of bounding boxes and person IDs using an open-source annotation tool.

The following standard MOT metrics (Bernardin & Stiefelhagen, 2008; Luiten et al., 2021) are computed:

| Metric | Definition |
|--------|-----------|
| **MOTA** (Multiple Object Tracking Accuracy) | $1 - \frac{FN + FP + IDSW}{GT}$; a composite accuracy metric penalising missed detections, false positives, and identity switches. |
| **IDF1** (ID F1 Score) | Harmonic mean of identification precision and recall; measures how well predicted IDs match ground-truth IDs over entire trajectories. |
| **ID Switches** | Number of times a ground-truth trajectory's assigned predicted ID changes. Directly measures identity fragmentation. |
| **Fragmentation** | Number of times a ground-truth trajectory is interrupted (tracked → lost → re-tracked, possibly under a new ID). |
| **Unique-Person Count Error** | $\lvert \hat{N} - N \rvert$; absolute difference between the pipeline's cumulative count and the true number of unique persons. |

Two configurations are evaluated on each test sequence:

1. **Baseline** — centroid tracker with no motion compensation (raw previous-frame centroids used for association).
2. **Motion-Compensated** — centroid tracker with ORB + RANSAC homography warping applied before association.

A paired comparison across sequences quantifies the improvement attributable to the motion-compensation stage. Statistical significance is assessed where the number of test sequences permits.

### 3.4 Additional Enhancements

Beyond the core pipeline, several secondary enhancements are investigated:

- **Tiled Detection (SAHI-style):** The frame is divided into a grid of overlapping tiles (e.g., 2 × 2 with 20 % overlap). The detector is run on each tile at native resolution, and the resulting bounding boxes—mapped back to full-frame coordinates—are merged via NMS. This strategy improves recall for small, distant persons at the cost of increased inference time (roughly proportional to the number of tiles).

- **Count-Confidence Scoring:** Each cumulative-count increment is annotated with a confidence score derived from the track's average detection confidence and its persistence (number of frames matched). Low-confidence increments can be flagged for human review or excluded from the final tally.

- **Frame-Gap Interpolation:** When one or more frames are dropped due to network jitter, the tracker linearly interpolates the expected centroid positions based on each track's velocity estimate, reducing the risk of ID loss across temporal gaps.

- **Privacy-Preserving Face Blurring:** Detected person bounding boxes are cropped, a face detector (Haar cascade or MediaPipe FaceMesh) is applied within the crop, and detected face regions are blurred with a Gaussian kernel before the annotated frame is displayed or saved. This enhancement addresses ethical and legal considerations surrounding aerial surveillance imagery.

- **Model-Size Benchmarking:** Inference speed (fps) and detection accuracy (mAP@0.5) are measured for YOLOv8 nano, small, medium, and large variants on the project's test frames, enabling an informed trade-off between accuracy and real-time throughput.

---

## 4. Tools and Technologies

### 4.1 Hardware

| Component | Specification | Role |
|-----------|--------------|------|
| **ESP32-CAM** | ESP32-S with OV2640 sensor, 802.11b/g/n Wi-Fi | Aerial imaging and MJPEG streaming |
| **Quadcopter Frame** | 450 mm wheelbase, self-assembled | UAV airframe |
| **Brushless Motors (×4)** | 920 KV BLDC | Propulsion |
| **ESCs (×4)** | 30 A SimonK / BLHeli firmware | Motor speed control |
| **Flight Controller** | APM 2.8 / Pixhawk-compatible | Attitude stabilisation and RC input |
| **LiPo Battery** | 3S 11.1 V, 2200–3000 mAh | On-board power |
| **RC Transmitter & Receiver** | FlySky FS-i6 / FS-iA6B (6-channel, 2.4 GHz) | Manual flight control |
| **Ground-Station Laptop** | Intel Core i5/i7 (10th gen+), 16 GB RAM, NVIDIA discrete GPU (GTX 1650 or better) | Off-board inference and processing |

### 4.2 Software

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| **Python** | 3.13 | Primary development language |
| **OpenCV (`cv2`)** | 4.x | Video I/O, MJPEG stream decoding, ORB feature extraction, BFMatcher, `findHomography`, perspective warping, image annotation |
| **Ultralytics YOLOv8** | 8.x | Pre-trained person detection (COCO class 0) |
| **NumPy** | 1.x / 2.x | Matrix and homogeneous-coordinate operations |
| **Matplotlib** | 3.x | Benchmarking plots, evaluation visualisation |
| **SciPy** | 1.x | Hungarian algorithm (`linear_sum_assignment`) for optimal track association |

### 4.3 Development Environment

| Tool | Purpose |
|------|---------|
| **Git** | Version control |
| **Visual Studio Code** | Primary IDE |
| **Python `venv`** | Dependency isolation |
| **Jupyter Notebook** | Prototyping and evaluation analysis |

---

## 5. Project Plan

The project follows a five-month timeline aligned with the August–December 2026 academic semester. The Gantt chart below summarises the major work packages and their temporal allocation.

| # | Work Package | Aug | Sep | Oct | Nov | Dec |
|---|-------------|:---:|:---:|:---:|:---:|:---:|
| 1 | Literature survey & related-work analysis | ███ | · | · | · | · |
| 2 | Hardware procurement & quadcopter assembly | ███ | ██· | · | · | · |
| 3 | ESP32-CAM firmware & Wi-Fi streaming setup | ·██ | · | · | · | · |
| 4 | StreamReader module (HTTP MJPEG ingest) | · | ██· | · | · | · |
| 5 | YOLOv8 detection integration & tuning | · | ███ | · | · | · |
| 6 | Centroid tracker (baseline, no compensation) | · | ·██ | · | · | · |
| 7 | ORB + RANSAC motion compensation module | · | · | ███ | · | · |
| 8 | Evaluation harness & ground-truth annotation | · | · | ·██ | ██· | · |
| 9 | Baseline vs. compensated evaluation runs | · | · | · | ███ | · |
| 10 | Tiled detection (SAHI-style) enhancement | · | · | · | ·██ | · |
| 11 | Count-confidence scoring & gap interpolation | · | · | · | ·██ | · |
| 12 | Privacy blurring & model-size benchmarking | · | · | · | ··█ | ██· |
| 13 | Final integration & system testing | · | · | · | · | ███ |
| 14 | Report writing & documentation | · | · | ··█ | ··█ | ███ |
| 15 | Demonstration & viva preparation | · | · | · | · | ·██ |

**Legend:** ███ = full month activity; ·██ / ██· = partial month (early/late); ··█ = final week only.

### Milestones

| Date | Milestone |
|------|-----------|
| **31 Aug 2026** | Literature review complete; quadcopter assembled and test-flown; ESP32-CAM streaming verified. |
| **30 Sep 2026** | Core pipeline operational end-to-end (stream → detect → track → count) without motion compensation. |
| **31 Oct 2026** | Motion-compensation module integrated; evaluation harness ready; first quantitative results on test clips. |
| **30 Nov 2026** | Full evaluation complete (baseline vs. compensated); Tier-2 enhancements implemented and tested. |
| **15 Dec 2026** | Final report submitted; live demonstration ready. |

---

## References

1. Bernardin, K. & Stiefelhagen, R. (2008). Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. *EURASIP Journal on Image and Video Processing*, 2008, 1–10.
2. Bewley, A., Ge, Z., Ott, L., Ramos, F. & Upcroft, B. (2016). Simple Online and Realtime Tracking. *Proc. IEEE ICIP*, 3464–3468.
3. Du, D., Qi, Y., Yu, H. et al. (2018). The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking. *Proc. ECCV*, 370–386.
4. Erdelj, M., Natalizio, E., Chowdhury, K. R. & Akyildiz, I. F. (2017). Help from the Sky: Leveraging UAVs for Disaster Management. *IEEE Pervasive Computing*, 16(1), 24–32.
5. Jocher, G., Chaurasia, A. & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics.
6. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
7. Luiten, J., Osep, A., Dendorfer, P. et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. *International Journal of Computer Vision*, 129, 548–578.
8. Rublee, E., Rabaud, V., Konolige, K. & Bradski, G. (2011). ORB: An Efficient Alternative to SIFT or SURF. *Proc. IEEE ICCV*, 2564–2571.
9. Wojke, N., Bewley, A. & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *Proc. IEEE ICIP*, 3645–3649.
10. Zhang, Y., Sun, P., Jiang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *Proc. ECCV*, 1–21.
11. Zhu, P., Wen, L., Du, D. et al. (2021). Detection and Tracking Meet Drones Challenge. *IEEE TPAMI*, 44(11), 7380–7399.
