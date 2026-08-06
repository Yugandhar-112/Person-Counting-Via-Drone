"""
main.py

Ties the full pipeline together:
  video source (webcam or ESP32-CAM stream) -> detect -> track -> count -> display

Usage:
    # Test with laptop webcam (default):
    python src/main.py --source 0

    # Once you have the drone's camera:
    python src/main.py --source http://192.168.4.1:81/stream
"""

import argparse
import cv2

from stream_reader import StreamReader
from detector import PersonDetector
from tracker import CentroidTracker
from counter import PersonCounter


def draw_overlays(frame, current_frame_ids, count):
    for track_id, (x1, y1, x2, y2) in current_frame_ids.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live person counting pipeline")
    parser.add_argument("--source", default="0",
                         help="Video source: webcam index (e.g. 0) or a stream URL "
                              "(e.g. http://192.168.4.1:81/stream)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights to use")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    reader = StreamReader(source)
    if not reader.open():
        print(f"Could not open source: {source}")
        return

    detector = PersonDetector(model_path=args.model, confidence=args.confidence)
    tracker = CentroidTracker()
    counter = PersonCounter()

    print("Streaming started. Press 'q' to quit.")
    for frame in reader.frames():
        detections = detector.detect(frame)
        current_frame_ids = tracker.update(detections)
        count = counter.update(current_frame_ids)

        frame = draw_overlays(frame, current_frame_ids, count)
        cv2.imshow("Person Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Final unique person count: {counter.count}")
    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()