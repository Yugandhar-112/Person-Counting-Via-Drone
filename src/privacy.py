"""
privacy.py

Applies Gaussian blur to estimated head regions of detected people,
protecting their visual identity before any frame is displayed or saved.

The head region is estimated as the top ~25% of each person's bounding box.
This is a simple heuristic that works well for upright people as viewed
from typical drone altitudes.

Toggle: --privacy-blur CLI flag / config.enable_privacy_blur.
"""

import cv2
import numpy as np
from typing import List


class PrivacyFilter:
    """Blurs head regions of detected people in a frame."""

    def __init__(
        self,
        head_fraction: float = 0.25,
        blur_kernel_size: int = 51,
    ):
        """
        Args:
            head_fraction: fraction of the bbox height to treat as the
                head region (measured from the top).
            blur_kernel_size: size of the Gaussian blur kernel (must be odd).
        """
        self.head_fraction = head_fraction
        # Ensure kernel size is odd
        self.blur_kernel = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1

    def apply(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Apply privacy blur to all detected people in the frame.

        Args:
            frame: BGR image (modified in-place and returned).
            detections: list of dicts with "bbox" key (from PersonDetector).

        Returns:
            The frame with head regions blurred.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_height = y2 - y1
            head_height = int(bbox_height * self.head_fraction)

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            hx1 = max(0, x1)
            hy1 = max(0, y1)
            hx2 = min(w, x2)
            hy2 = min(h, y1 + head_height)

            if hx2 <= hx1 or hy2 <= hy1:
                continue  # degenerate region, skip

            # Apply Gaussian blur to the head region
            roi = frame[hy1:hy2, hx1:hx2]
            blurred = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
            frame[hy1:hy2, hx1:hx2] = blurred

        return frame
