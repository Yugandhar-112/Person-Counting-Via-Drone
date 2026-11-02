"""
confidence.py

Assigns a reliability score (0.0–1.0) and human-readable flag to the
current person count based on real-time quality signals:

  - Camera motion magnitude (high motion = less reliable tracking)
  - Track overlap ratio (overlapping tracks = potential mis-associations)
  - Frame-drop rate (dropped frames = gaps in observation)

Displayed on the live overlay alongside the count so the operator knows
how much to trust the current number.
"""

from typing import Dict, Optional, Tuple


class CountConfidence:
    """Scores count reliability from real-time pipeline signals."""

    def __init__(
        self,
        high_motion_threshold: float = 30.0,
        high_overlap_threshold: float = 0.3,
        high_drop_rate_threshold: float = 0.05,
    ):
        """
        Args:
            high_motion_threshold: motion magnitude above which tracking
                becomes less reliable (translation-component norm).
            high_overlap_threshold: mean IoU among active tracks above
                which counting becomes less reliable.
            high_drop_rate_threshold: frame-drop fraction above which
                counting becomes less reliable.
        """
        self.high_motion = high_motion_threshold
        self.high_overlap = high_overlap_threshold
        self.high_drop_rate = high_drop_rate_threshold

    def score(
        self,
        motion_magnitude: Optional[float],
        tracks: Dict[int, dict],
        drop_rate: float = 0.0,
    ) -> Tuple[float, str]:
        """Compute a confidence score and label.

        Args:
            motion_magnitude: scalar camera-motion estimate from
                MotionCompensator (None if not enabled or first frame).
            tracks: the tracker's internal tracked dict
                (track_id -> {"centroid": ..., "bbox": ...}).
            drop_rate: fraction of frames dropped so far (0.0–1.0).

        Returns:
            (score, label) where score ∈ [0.0, 1.0] and
            label ∈ {"HIGH", "MEDIUM", "LOW"}.
        """
        penalties = []

        # --- Motion penalty ---
        if motion_magnitude is not None:
            ratio = min(motion_magnitude / self.high_motion, 1.0)
            penalties.append(ratio * 0.4)  # max 0.4 penalty from motion
        else:
            penalties.append(0.0)

        # --- Overlap penalty ---
        overlap = self._mean_overlap(tracks)
        ratio = min(overlap / self.high_overlap, 1.0) if self.high_overlap > 0 else 0.0
        penalties.append(ratio * 0.3)  # max 0.3 penalty from overlap

        # --- Drop-rate penalty ---
        ratio = min(drop_rate / self.high_drop_rate, 1.0) if self.high_drop_rate > 0 else 0.0
        penalties.append(ratio * 0.3)  # max 0.3 penalty from drops

        total_penalty = sum(penalties)
        score = max(0.0, 1.0 - total_penalty)

        # Label
        if score >= 0.7:
            label = "HIGH"
        elif score >= 0.4:
            label = "MEDIUM"
        else:
            label = "LOW"

        return round(score, 3), label

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_overlap(tracks: Dict[int, dict]) -> float:
        """Compute mean pairwise IoU among active tracks' bounding boxes."""
        bboxes = [
            info["bbox"] for info in tracks.values()
            if "bbox" in info and info.get("missed", 0) == 0
        ]
        if len(bboxes) < 2:
            return 0.0

        ious = []
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                ious.append(_iou(bboxes[i], bboxes[j]))
        return sum(ious) / len(ious) if ious else 0.0


def _iou(box_a, box_b) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
