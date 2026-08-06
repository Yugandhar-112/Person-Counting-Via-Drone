# Person Counting Via Drone

A low-cost drone that streams its camera feed live to a laptop over Wi-Fi, where
a computer-vision pipeline detects and counts the number of people present in
the room or area in real time.

## Project Goal

The video is pulled directly into the software as the drone flies (live
stream), rather than recorded and analyzed afterward. Detection, tracking,
and counting happen live, and a running headcount is shown on screen during
the flight.

See [`docs/`](docs/) for the full project proposal.

## Proposed Hardware

- Self-assembled quadcopter frame (motors, ESCs, flight controller, battery, receiver) — final parts TBD with faculty guidance
- ESP32-CAM module for Wi-Fi video (broadcasts an MJPEG stream over its own Wi-Fi access point)

## Software Pipeline

1. **Connect** — laptop connects to the ESP32-CAM's Wi-Fi access point and opens its MJPEG stream URL
2. **Detect** — a pretrained person-detection model (YOLO) runs on each incoming frame
3. **Track** — a multi-object tracker assigns each detected person a persistent ID across frames, to avoid double-counting
4. **Count** — a running count of unique tracked IDs is maintained and displayed live
5. **Report** — methodology, accuracy checks, and results are summarized after the demo flight

## Project Structure

```
Person-Counting-Via-Drone/
├── docs/                  # Project proposal and related documents
├── src/
│   ├── stream_reader.py   # Connects to the ESP32-CAM MJPEG stream
│   ├── detector.py        # Person detection (YOLO)
│   ├── tracker.py         # Multi-object tracking / ID assignment
│   ├── counter.py         # Running unique-person count logic
│   └── main.py            # Ties the pipeline together, live display
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py --stream-url http://<esp32-cam-ip>:81/stream
```

## Status

🚧 Early stage — hardware and detection pipeline in progress.
