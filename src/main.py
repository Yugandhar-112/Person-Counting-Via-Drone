"""
main.py

Ties the full pipeline together:
  video source (webcam / ESP32-CAM stream / video file)
    -> detect -> (optional motion compensation) -> track -> count -> display

Usage:
    # Test with laptop webcam (default):
    python src/main.py --source 0

    # Pre-recorded video file:
    python src/main.py --source test_clips/sample.mp4 --headless --log-dir logs

    # Full-featured run:
    python src/main.py --source 0 --motion-compensation --tiling \\
        --confidence-flag --privacy-blur

    # Once you have the drone's camera:
    python src/main.py --source http://192.168.4.1:81/stream
"""

import argparse
import os
import sys
import time

# Ensure local module imports work regardless of working directory
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import cv2

from config import PipelineConfig
from stream_reader import StreamReader
from detector import PersonDetector
from tracker import CentroidTracker
from counter import PersonCounter
from logger import PipelineLogger


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlays(frame, current_frame_ids, count, fps=0.0,
                  confidence_label=None, confidence_score=None):
    """Draw bounding boxes, IDs, count, FPS, and optional confidence on frame."""
    for track_id, (x1, y1, x2, y2) in current_frame_ids.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Count
    cv2.putText(frame, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # Confidence flag
    if confidence_label is not None:
        color_map = {"HIGH": (0, 255, 0), "MEDIUM": (0, 200, 255), "LOW": (0, 0, 255)}
        color = color_map.get(confidence_label, (200, 200, 200))
        text = f"Reliability: {confidence_label} ({confidence_score:.2f})"
        cv2.putText(frame, text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# ---------------------------------------------------------------------------
# CLI argument parsing -> PipelineConfig
# ---------------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    """Parse CLI arguments and return a populated PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Live person counting pipeline with optional enhancements"
    )
    # Input / output
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index (e.g. 0), stream URL, "
                             "or path to a video file.")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model weights to use.")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without cv2.imshow (for evaluation / CI).")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for per-frame JSON-lines logs.")

    # Feature toggles
    parser.add_argument("--motion-compensation", action="store_true",
                        help="Enable ORB+homography motion compensation.")
    parser.add_argument("--tiling", action="store_true",
                        help="Enable tiled (SAHI-style) detection.")
    parser.add_argument("--confidence-flag", action="store_true",
                        help="Show count-reliability confidence indicator.")
    parser.add_argument("--gap-interpolation", action="store_true",
                        help="Enable frame-gap interpolation in the tracker.")
    parser.add_argument("--privacy-blur", action="store_true",
                        help="Blur head regions of detected people.")

    args = parser.parse_args()

    cfg = PipelineConfig(
        source=args.source,
        headless=args.headless,
        log_dir=args.log_dir,
        enable_motion_compensation=args.motion_compensation,
        enable_tiling=args.tiling,
        enable_confidence_flag=args.confidence_flag,
        enable_gap_interpolation=args.gap_interpolation,
        enable_privacy_blur=args.privacy_blur,
    )
    cfg.model.model_path = args.model
    cfg.model.confidence = args.confidence
    return cfg


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    # Resolve source (int for webcam, str for file/URL)
    source = int(cfg.source) if cfg.source.isdigit() else cfg.source

    reader = StreamReader(source)
    if not reader.open():
        print(f"Could not open source: {source}")
        sys.exit(1)

    # --- Detector (plain or tiled) ---
    base_detector = PersonDetector(
        model_path=cfg.model.model_path,
        confidence=cfg.model.confidence,
    )

    if cfg.enable_tiling:
        from tiled_detector import TiledDetector
        detector = TiledDetector(
            base_detector,
            tile_size=cfg.tiling.tile_size,
            overlap_ratio=cfg.tiling.overlap_ratio,
            nms_iou_threshold=cfg.tiling.nms_iou_threshold,
        )
    else:
        detector = base_detector

    # --- Motion compensator ---
    motion_compensator = None
    if cfg.enable_motion_compensation:
        from motion_compensation import MotionCompensator
        motion_compensator = MotionCompensator()

    # --- Tracker ---
    tracker = CentroidTracker(
        max_distance=cfg.tracker.max_distance,
        max_missed_frames=cfg.tracker.max_missed_frames,
        enable_gap_interpolation=cfg.enable_gap_interpolation,
    )

    # --- Counter ---
    counter = PersonCounter()

    # --- Confidence scorer ---
    confidence_scorer = None
    if cfg.enable_confidence_flag:
        from confidence import CountConfidence
        confidence_scorer = CountConfidence(
            high_motion_threshold=cfg.confidence.high_motion_threshold,
            high_overlap_threshold=cfg.confidence.high_overlap_threshold,
            high_drop_rate_threshold=cfg.confidence.high_drop_rate_threshold,
        )

    # --- Privacy filter ---
    privacy_filter = None
    if cfg.enable_privacy_blur:
        from privacy import PrivacyFilter
        privacy_filter = PrivacyFilter()

    # --- Logger ---
    logger = None
    if cfg.log_dir:
        logger = PipelineLogger(cfg.log_dir)

    # --- Run ---
    print("Pipeline started. Press 'q' to quit." if not cfg.headless else "Pipeline started (headless).")
    fps_value = 0.0
    frame_time = time.perf_counter()

    try:
        for fr in reader.frame_results():
            # Handle dropped frames
            if fr.is_dropped:
                if cfg.enable_gap_interpolation:
                    tracker.predict_gap()
                if logger:
                    logger.log_frame(
                        frame_index=fr.frame_index,
                        detections=[],
                        tracks={tid: list(info["bbox"]) for tid, info in tracker.tracked.items() if "bbox" in info},
                        unique_count=counter.count,
                        is_dropped_frame=True,
                    )
                continue

            frame = fr.frame

            # ── Detect ──
            detections = detector.detect(frame)

            # ── Motion compensation ──
            homography = None
            motion_mag = None
            comp_failed = None
            if motion_compensator is not None:
                homography, motion_mag, comp_failed = motion_compensator.process(frame)

            # ── Track ──
            current_frame_ids = tracker.update(
                detections,
                homography=homography,
                is_gap_frame=False,
            )
            id_switch_events = tracker.pop_id_switch_events()

            # ── Count ──
            count = counter.update(current_frame_ids)

            # ── Confidence ──
            conf_score, conf_label = None, None
            if confidence_scorer is not None:
                conf_score, conf_label = confidence_scorer.score(
                    motion_magnitude=motion_mag,
                    tracks=tracker.tracked,
                    drop_rate=reader.drop_rate,
                )

            # ── Privacy blur (before display / save) ──
            if privacy_filter is not None:
                frame = privacy_filter.apply(frame, detections)

            # ── Overlay + display ──
            if not cfg.headless:
                frame = draw_overlays(frame, current_frame_ids, count,
                                      fps=fps_value,
                                      confidence_label=conf_label,
                                      confidence_score=conf_score)
                cv2.imshow("Person Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Log ──
            if logger:
                logger.log_frame(
                    frame_index=fr.frame_index,
                    detections=detections,
                    tracks=current_frame_ids,
                    unique_count=count,
                    motion_magnitude=motion_mag,
                    compensation_failed=comp_failed,
                    id_switch_events=id_switch_events if id_switch_events else None,
                    confidence_score=conf_score,
                    confidence_label=conf_label,
                )

            # ── FPS ──
            now = time.perf_counter()
            dt = now - frame_time
            fps_value = 1.0 / dt if dt > 0 else 0.0
            frame_time = now

    finally:
        print(f"\nFinal unique person count: {counter.count}")
        print(f"Frames processed: {reader.frame_index}, drops: {reader.drop_count}")
        if logger:
            logger.close()
            print(f"Logs saved to: {logger.run_dir}")
        reader.release()
        if not cfg.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()