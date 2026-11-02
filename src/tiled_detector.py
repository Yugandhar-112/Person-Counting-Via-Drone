"""
tiled_detector.py

SAHI-style tiled detection: slices the input frame into overlapping tiles,
runs the base PersonDetector on each tile, maps tile-local coordinates back
to full-frame coordinates, and runs NMS across all merged detections to
suppress duplicates at tile boundaries.

Improves recall on small/distant people that YOLO would otherwise miss at
full-frame resolution.

Toggle: when config.enable_tiling is False, main.py uses PersonDetector
directly — this module is never instantiated.
"""

from typing import List, Tuple

import cv2
import numpy as np

from detector import PersonDetector


class TiledDetector:
    """Wraps PersonDetector with overlapping-tile slicing + cross-tile NMS."""

    def __init__(
        self,
        base_detector: PersonDetector,
        tile_size: int = 640,
        overlap_ratio: float = 0.2,
        nms_iou_threshold: float = 0.5,
    ):
        """
        Args:
            base_detector: the underlying PersonDetector instance.
            tile_size: width and height of each tile (square).
            overlap_ratio: fraction of tile_size that adjacent tiles overlap.
            nms_iou_threshold: IoU threshold for cross-tile NMS.
        """
        self.detector = base_detector
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.nms_iou_threshold = nms_iou_threshold

    def detect(self, frame) -> list:
        """Run tiled detection on a full frame.

        Returns:
            List of detections, same format as PersonDetector.detect():
            [{"bbox": (x1, y1, x2, y2), "confidence": float}, ...]
        """
        h, w = frame.shape[:2]
        tiles = self._compute_tiles(w, h)

        all_detections = []
        for (tx, ty, tw, th) in tiles:
            tile_img = frame[ty:ty + th, tx:tx + tw]
            tile_dets = self.detector.detect(tile_img)

            # Map tile-local coords to full-frame coords
            for det in tile_dets:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (x1 + tx, y1 + ty, x2 + tx, y2 + ty)
                all_detections.append(det)

        # Also run on the full frame at native resolution to catch
        # large people that span multiple tiles
        full_dets = self.detector.detect(frame)
        all_detections.extend(full_dets)

        # Cross-tile NMS
        if len(all_detections) > 0:
            all_detections = self._nms(all_detections)

        return all_detections

    # ------------------------------------------------------------------
    # Tile computation
    # ------------------------------------------------------------------

    def _compute_tiles(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Compute tile positions (x, y, width, height) covering the image.

        Tiles overlap by overlap_ratio * tile_size on each side.
        """
        step = int(self.tile_size * (1 - self.overlap_ratio))
        tiles = []

        y = 0
        while y < img_h:
            x = 0
            th = min(self.tile_size, img_h - y)
            while x < img_w:
                tw = min(self.tile_size, img_w - x)
                tiles.append((x, y, tw, th))
                x += step
                if x + self.tile_size >= img_w and x < img_w:
                    # Ensure last column covers the right edge
                    tiles.append((max(0, img_w - self.tile_size), y,
                                  min(self.tile_size, img_w), th))
                    break
            y += step
            if y + self.tile_size >= img_h and y < img_h:
                # Ensure last row covers the bottom edge
                # (will be handled in next iteration)
                pass

        # Deduplicate tiles with identical origins
        seen = set()
        unique_tiles = []
        for t in tiles:
            key = (t[0], t[1])
            if key not in seen:
                seen.add(key)
                unique_tiles.append(t)

        return unique_tiles

    # ------------------------------------------------------------------
    # Non-Maximum Suppression
    # ------------------------------------------------------------------

    def _nms(self, detections: list) -> list:
        """Apply NMS to merged detections from all tiles."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= self.nms_iou_threshold)[0]
            order = order[remaining + 1]

        return [detections[i] for i in keep]
