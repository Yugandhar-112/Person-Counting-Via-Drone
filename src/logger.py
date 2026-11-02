"""
logger.py

Writes per-frame JSON-lines (.jsonl) logs during pipeline runs.

Each line is a self-contained JSON object capturing the full pipeline
state for one frame: raw detections, tracker output, motion estimate,
running count, and timing.  A separate log is created per run inside
the configured log directory.

These logs are consumed by the evaluation harness (evaluation.py) and
are useful for post-hoc debugging of tracking failures.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class PipelineLogger:
    """Writes structured per-frame logs to a JSONL file."""

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: root directory for logs.  A timestamped sub-directory
                is created automatically for each run.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._filepath = os.path.join(self.run_dir, "frames.jsonl")
        self._file = open(self._filepath, "w", encoding="utf-8")
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame_index: int,
        detections: List[Dict[str, Any]],
        tracks: Dict[int, Any],
        unique_count: int,
        *,
        motion_magnitude: Optional[float] = None,
        compensation_failed: Optional[bool] = None,
        id_switch_events: Optional[List[Dict]] = None,
        confidence_score: Optional[float] = None,
        confidence_label: Optional[str] = None,
        is_dropped_frame: bool = False,
    ) -> None:
        """Write one log line for a single frame.

        Args:
            frame_index: zero-based frame counter.
            detections: raw detector output (list of {"bbox": ..., "confidence": ...}).
            tracks: current tracker output (track_id -> bbox).
            unique_count: running unique person count.
            motion_magnitude: scalar camera-motion estimate (None before Phase 2).
            compensation_failed: True if homography estimation failed this frame.
            id_switch_events: list of {"old_id": ..., "new_id": ..., "centroid": ...}.
            confidence_score: 0.0-1.0 reliability score (None before Phase 5).
            confidence_label: "HIGH" / "MEDIUM" / "LOW" (None before Phase 5).
            is_dropped_frame: True if this frame was interpolated / dropped.
        """
        record = {
            "frame_index": frame_index,
            "timestamp_sec": round(time.perf_counter() - self._start_time, 4),
            "detections": [
                {"bbox": list(d["bbox"]), "confidence": round(d["confidence"], 4)}
                for d in detections
            ],
            "tracks": {str(tid): list(bbox) for tid, bbox in tracks.items()},
            "unique_count": unique_count,
            "motion_magnitude": motion_magnitude,
            "compensation_failed": compensation_failed,
            "id_switch_events": id_switch_events,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "is_dropped_frame": is_dropped_frame,
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def run_dir(self) -> str:
        return self._run_dir

    @run_dir.setter
    def run_dir(self, value: str) -> None:
        self._run_dir = value

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
