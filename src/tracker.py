"""
tracker.py

Assigns a persistent ID to each detected person across frames, so the same
person isn't counted more than once as the drone moves through a space.

Enhanced with:
  - Optional homography-based motion compensation (Phase 2): warps stored
    centroids/bboxes into current-frame space before distance matching.
  - Bbox storage alongside centroids (needed by evaluation + confidence).
  - ID-switch event detection and logging.
  - Optional constant-velocity gap interpolation (Phase 6): predicts track
    positions during dropped frames.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class CentroidTracker:
    """Assigns/maintains IDs for detections based on centroid proximity."""

    def __init__(
        self,
        max_distance: int = 60,
        max_missed_frames: int = 15,
        enable_gap_interpolation: bool = False,
    ):
        """
        Args:
            max_distance: max pixel distance between frames to treat two
                detections as the same person.
            max_missed_frames: how many consecutive frames a tracked person
                can go undetected before their ID is dropped.
            enable_gap_interpolation: if True, predict track positions
                during dropped frames using constant-velocity model.
        """
        self.next_id = 0
        self.tracked: Dict[int, dict] = {}
        # Each entry: {"centroid": (x, y), "bbox": (x1,y1,x2,y2),
        #              "missed": int, "velocity": (vx, vy)}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.enable_gap_interpolation = enable_gap_interpolation

        # Accumulates ID-switch events for the current frame; consumed by
        # main.py after each update() call via pop_id_switch_events().
        self._id_switch_events: List[dict] = []

        # Tracks recently deleted IDs (centroid + frame of deletion) for
        # ID-switch heuristic detection.
        self._recently_deleted: List[dict] = []
        self._deletion_memory_frames: int = 3  # how long to remember deletions

    @staticmethod
    def _centroid(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(
        self,
        detections: list,
        homography: Optional[np.ndarray] = None,
        is_gap_frame: bool = False,
    ) -> dict:
        """
        Update tracks with the current frame's detections.

        Args:
            detections: list of dicts with a "bbox" key (from PersonDetector).
            homography: 3x3 matrix mapping prev-frame coords → current-frame
                coords.  When provided, all stored centroids are warped before
                distance matching, correcting for camera ego-motion.
            is_gap_frame: reserved for gap-interpolation (not used directly
                in update — see predict_gap() instead).

        Returns:
            dict mapping track_id -> bbox for currently visible people.
        """
        self._id_switch_events.clear()

        # ── Warp stored positions if we have a valid homography ──
        if homography is not None:
            self._apply_homography(homography)

        input_centroids = [self._centroid(d["bbox"]) for d in detections]
        unmatched_inputs = set(range(len(input_centroids)))
        current_frame_ids = {}

        # ── Match existing tracks to new detections ──
        for track_id, info in list(self.tracked.items()):
            best_match, best_dist = None, self.max_distance
            for i in unmatched_inputs:
                dist = math.dist(info["centroid"], input_centroids[i])
                if dist < best_dist:
                    best_match, best_dist = i, dist

            if best_match is not None:
                prev_centroid = info["centroid"]
                new_centroid = input_centroids[best_match]
                velocity = (
                    new_centroid[0] - prev_centroid[0],
                    new_centroid[1] - prev_centroid[1],
                )
                self.tracked[track_id] = {
                    "centroid": new_centroid,
                    "bbox": detections[best_match]["bbox"],
                    "missed": 0,
                    "velocity": velocity,
                }
                current_frame_ids[track_id] = detections[best_match]["bbox"]
                unmatched_inputs.discard(best_match)
            else:
                info["missed"] += 1
                if info["missed"] > self.max_missed_frames:
                    # Remember deletion for ID-switch detection
                    self._recently_deleted.append({
                        "id": track_id,
                        "centroid": info["centroid"],
                        "age": 0,
                    })
                    del self.tracked[track_id]

        # ── Age and prune deletion memory ──
        for entry in self._recently_deleted:
            entry["age"] += 1
        self._recently_deleted = [
            e for e in self._recently_deleted
            if e["age"] <= self._deletion_memory_frames
        ]

        # ── New tracks for unmatched detections ──
        for i in unmatched_inputs:
            new_centroid = input_centroids[i]

            # ID-switch heuristic: if a new track spawns very close to a
            # recently deleted track, flag it as a likely re-identification.
            for deleted in self._recently_deleted:
                if math.dist(new_centroid, deleted["centroid"]) < self.max_distance:
                    self._id_switch_events.append({
                        "old_id": deleted["id"],
                        "new_id": self.next_id,
                        "centroid": new_centroid,
                    })
                    break

            self.tracked[self.next_id] = {
                "centroid": new_centroid,
                "bbox": detections[i]["bbox"],
                "missed": 0,
                "velocity": (0.0, 0.0),
            }
            current_frame_ids[self.next_id] = detections[i]["bbox"]
            self.next_id += 1

        return current_frame_ids

    def predict_gap(self) -> None:
        """Advance all tracks by one frame using constant-velocity prediction.

        Called during dropped frames when gap interpolation is enabled.
        Does NOT spawn new IDs or delete old ones — just extrapolates
        existing tracks' centroids.
        """
        if not self.enable_gap_interpolation:
            return

        for tid, info in self.tracked.items():
            vx, vy = info.get("velocity", (0.0, 0.0))
            cx, cy = info["centroid"]
            info["centroid"] = (cx + vx, cy + vy)

            # Also shift bbox
            if "bbox" in info:
                x1, y1, x2, y2 = info["bbox"]
                info["bbox"] = (
                    int(x1 + vx), int(y1 + vy),
                    int(x2 + vx), int(y2 + vy),
                )
            # Don't increment missed — this is an interpolated frame

    def pop_id_switch_events(self) -> List[dict]:
        """Return and clear any ID-switch events flagged during the last update()."""
        events = list(self._id_switch_events)
        self._id_switch_events.clear()
        return events

    @property
    def total_unique_ids_seen(self) -> int:
        """Total number of unique people ever assigned an ID this session."""
        return self.next_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_homography(self, H: np.ndarray) -> None:
        """Warp all stored centroids and bboxes from previous-frame space
        into current-frame space using the given homography."""
        try:
            from motion_compensation import MotionCompensator
        except ImportError:
            from src.motion_compensation import MotionCompensator

        # Warp centroids
        centroids = []
        tids = []
        for tid, info in self.tracked.items():
            centroids.append(info["centroid"])
            tids.append(tid)

        if not centroids:
            return

        warped = MotionCompensator.warp_points(np.array(centroids, dtype=np.float32), H)
        for idx, tid in enumerate(tids):
            self.tracked[tid]["centroid"] = tuple(warped[idx])

        # Warp bboxes
        bboxes = [self.tracked[tid].get("bbox") for tid in tids]
        valid_bboxes = [b for b in bboxes if b is not None]
        valid_tids = [tids[i] for i, b in enumerate(bboxes) if b is not None]

        if valid_bboxes:
            warped_bboxes = MotionCompensator.warp_bboxes(valid_bboxes, H)
            for tid, wbbox in zip(valid_tids, warped_bboxes):
                self.tracked[tid]["bbox"] = wbbox
