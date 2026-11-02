"""
motion_compensation.py

Estimates camera ego-motion between consecutive frames using ORB feature
matching and RANSAC homography estimation.  The resulting homography
matrix is used by the tracker to warp previous-frame track positions
into current-frame coordinates before running association, dramatically
reducing ID switches caused by camera panning / altitude changes.

Toggle: entirely skipped when config.enable_motion_compensation is False
(the homography is never computed and tracker receives None).
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class MotionCompensator:
    """Estimates frame-to-frame camera motion via ORB + homography."""

    def __init__(
        self,
        n_features: int = 500,
        match_ratio: float = 0.75,
        ransac_reproj_thresh: float = 5.0,
        min_matches: int = 10,
    ):
        """
        Args:
            n_features: number of ORB keypoints to detect per frame.
            match_ratio: Lowe's ratio test threshold for BFMatcher.
            ransac_reproj_thresh: RANSAC reprojection error threshold (pixels).
            min_matches: minimum good matches required to estimate homography.
        """
        self.orb = cv2.ORB_create(nFeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.match_ratio = match_ratio
        self.ransac_thresh = ransac_reproj_thresh
        self.min_matches = min_matches

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_des = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
        """Compute homography mapping previous frame → current frame.

        Args:
            frame: BGR image (current frame).

        Returns:
            (homography, motion_magnitude, compensation_failed)
            - homography: 3x3 ndarray or None (first frame / failure).
            - motion_magnitude: scalar summarizing camera displacement
              (Frobenius norm of the translation component), or None.
            - compensation_failed: True if estimation was attempted but
              failed (too few matches, RANSAC failure).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        # First frame — nothing to compare against
        if self._prev_gray is None or self._prev_des is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return None, None, False

        # Not enough features in the current frame
        if des is None or len(des) < self.min_matches:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return None, None, True

        # Match features
        try:
            raw_matches = self.bf.knnMatch(self._prev_des, des, k=2)
        except cv2.error:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return None, None, True

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_matches:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return None, None, True

        # Extract matched point pairs
        src_pts = np.float32([self._prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate homography (prev → current)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        if H is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return None, None, True

        motion_mag = self._motion_magnitude(H)

        # Update state for next frame
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des

        return H, motion_mag, False

    # ------------------------------------------------------------------
    # Geometric helpers (usable externally by the tracker)
    # ------------------------------------------------------------------

    @staticmethod
    def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp an Nx2 array of (x, y) points through a 3x3 homography.

        Args:
            points: shape (N, 2) float array.
            H: 3x3 homography matrix.

        Returns:
            Warped points, shape (N, 2).
        """
        if len(points) == 0:
            return points
        pts = np.float32(points).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)

    @staticmethod
    def warp_bboxes(bboxes, H: np.ndarray):
        """Warp axis-aligned bounding boxes through a homography.

        Each bbox is (x1, y1, x2, y2). The four corners are transformed,
        then a new axis-aligned bbox is fitted around the warped corners.

        Args:
            bboxes: list of (x1, y1, x2, y2) tuples.
            H: 3x3 homography matrix.

        Returns:
            List of warped (x1, y1, x2, y2) tuples (ints).
        """
        if not bboxes:
            return []
        result = []
        for (x1, y1, x2, y2) in bboxes:
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            nx1, ny1 = warped.min(axis=0).astype(int)
            nx2, ny2 = warped.max(axis=0).astype(int)
            result.append((int(nx1), int(ny1), int(nx2), int(ny2)))
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_magnitude(H: np.ndarray) -> float:
        """Scalar summarizing the translation component of the homography."""
        # H[0,2] and H[1,2] are the translational offsets
        tx, ty = H[0, 2], H[1, 2]
        return float(np.sqrt(tx * tx + ty * ty))
