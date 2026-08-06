"""
tracker.py

Assigns a persistent ID to each detected person across frames, so the same
person isn't counted more than once as the drone moves through a space.

This starter version uses a simple centroid-distance tracker, which is
lightweight and dependency-free. Swap in DeepSORT later if more robust
re-identification is needed (e.g. after occlusion).
"""

import math


class CentroidTracker:
    """Assigns/maintains IDs for detections based on centroid proximity."""

    def __init__(self, max_distance: int = 60, max_missed_frames: int = 15):
        """
        Args:
            max_distance: max pixel distance between frames to treat two
                detections as the same person.
            max_missed_frames: how many consecutive frames a tracked person
                can go undetected before their ID is dropped.
        """
        self.next_id = 0
        self.tracked = {}  # id -> {"centroid": (x, y), "missed": int}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections):
        """
        Update tracks with the current frame's detections.

        Args:
            detections: list of dicts with a "bbox" key (from PersonDetector).

        Returns:
            dict mapping track_id -> bbox for currently visible people.
        """
        input_centroids = [self._centroid(d["bbox"]) for d in detections]
        unmatched_inputs = set(range(len(input_centroids)))
        current_frame_ids = {}

        # Try to match existing tracks to new detections.
        for track_id, info in list(self.tracked.items()):
            best_match, best_dist = None, self.max_distance
            for i in unmatched_inputs:
                dist = math.dist(info["centroid"], input_centroids[i])
                if dist < best_dist:
                    best_match, best_dist = i, dist

            if best_match is not None:
                self.tracked[track_id] = {"centroid": input_centroids[best_match], "missed": 0}
                current_frame_ids[track_id] = detections[best_match]["bbox"]
                unmatched_inputs.discard(best_match)
            else:
                info["missed"] += 1
                if info["missed"] > self.max_missed_frames:
                    del self.tracked[track_id]

        # Any leftover detections become new tracks (new people).
        for i in unmatched_inputs:
            self.tracked[self.next_id] = {"centroid": input_centroids[i], "missed": 0}
            current_frame_ids[self.next_id] = detections[i]["bbox"]
            self.next_id += 1

        return current_frame_ids

    @property
    def total_unique_ids_seen(self) -> int:
        """Total number of unique people ever assigned an ID this session."""
        return self.next_id
