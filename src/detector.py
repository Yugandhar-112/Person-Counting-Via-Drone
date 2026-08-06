"""
detector.py

Runs person detection on individual video frames using a pretrained
YOLO model (via the ultralytics package).
"""

from ultralytics import YOLO


class PersonDetector:
    """Detects people in a frame using a pretrained YOLO model."""

    # COCO class index for "person"
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Args:
            model_path: path or name of the YOLO model weights.
                "yolov8n.pt" is the smallest/fastest model, good for a first pass.
            confidence: minimum confidence score to keep a detection.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run detection on a single frame.

        Returns:
            List of detections, each a dict with keys:
                "bbox": (x1, y1, x2, y2)
                "confidence": float
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id != self.PERSON_CLASS_ID or conf < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })

        return detections
