"""
evaluation.py

MOT (Multiple Object Tracking) evaluation metrics computed from pipeline
logs and hand-annotated ground truth.

Metrics:
  - ID switches:  count of frames where a GT person's matched prediction ID changes.
  - MOTA:         1 - (FN + FP + ID_switches) / total_GT
  - IDF1:         F1 over ID-consistent matches.
  - Count error:  |predicted_unique_count - GT_unique_count| and percentage.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def _greedy_match(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Greedy bipartite matching based on IoU.

    Returns:
        matches: list of (gt_idx, pred_idx) pairs.
        unmatched_gt: set of gt indices with no match.
        unmatched_pred: set of pred indices with no match.
    """
    gt_indices = list(range(len(gt_boxes)))
    pred_indices = list(range(len(pred_boxes)))

    # Compute all IoUs
    pairs = []
    for gi in gt_indices:
        for pi in pred_indices:
            iou = _iou(gt_boxes[gi], pred_boxes[pi])
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))

    # Greedy: pick highest IoU first
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_gt = set()
    matched_pred = set()
    matches = []
    for _, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matches.append((gi, pi))
            matched_gt.add(gi)
            matched_pred.add(pi)

    unmatched_gt = set(gt_indices) - matched_gt
    unmatched_pred = set(pred_indices) - matched_pred
    return matches, unmatched_gt, unmatched_pred


class MOTEvaluator:
    """Compute MOT metrics from pipeline prediction logs and ground truth."""

    def __init__(
        self,
        prediction_log_path: str,
        ground_truth_path: str,
        iou_threshold: float = 0.5,
    ):
        """
        Args:
            prediction_log_path: path to the JSONL prediction log (from PipelineLogger).
            ground_truth_path: path to the ground-truth JSON file.
            iou_threshold: IoU threshold for a detection to count as a true positive.
        """
        self.pred_log_path = prediction_log_path
        self.gt_path = ground_truth_path
        self.iou_threshold = iou_threshold

        self._pred_frames: Dict[int, dict] = {}  # frame_idx -> log record
        self._gt_frames: Dict[int, List[dict]] = {}  # frame_idx -> list of GT objects
        self._gt_unique_ids: set = set()

        self._load_predictions()
        self._load_ground_truth()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run the full evaluation and return a results dict.

        Returns:
            {
                "id_switches": int,
                "mota": float,
                "idf1": float,
                "count_error_abs": int,
                "count_error_pct": float,
                "predicted_unique_count": int,
                "gt_unique_count": int,
                "total_gt": int,
                "total_fp": int,
                "total_fn": int,
            }
        """
        # Per-frame matching
        total_gt = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # For IDF1: track which GT-ID was matched to which pred-ID
        gt_to_pred_id: Dict[str, Optional[str]] = {}  # gt_id -> last matched pred_id

        # For IDF1 numerator
        idtp = 0  # ID-consistent true positives

        all_frame_indices = sorted(
            set(self._gt_frames.keys()) | set(self._pred_frames.keys())
        )

        for fidx in all_frame_indices:
            gt_objects = self._gt_frames.get(fidx, [])
            pred_record = self._pred_frames.get(fidx)

            gt_boxes = [obj["bbox"] for obj in gt_objects]
            gt_ids = [obj["id"] for obj in gt_objects]

            if pred_record is None:
                # No prediction for this frame — all GT are false negatives
                total_gt += len(gt_objects)
                total_fn += len(gt_objects)
                continue

            pred_tracks = pred_record.get("tracks", {})
            pred_ids = list(pred_tracks.keys())
            pred_boxes = [pred_tracks[pid] for pid in pred_ids]

            total_gt += len(gt_objects)

            matches, unmatched_gt, unmatched_pred = _greedy_match(
                gt_boxes, pred_boxes, self.iou_threshold
            )

            total_fn += len(unmatched_gt)
            total_fp += len(unmatched_pred)

            for gi, pi in matches:
                gt_id = str(gt_ids[gi])
                pred_id = str(pred_ids[pi])

                prev_pred_id = gt_to_pred_id.get(gt_id)
                if prev_pred_id is not None and prev_pred_id != pred_id:
                    id_switches += 1

                gt_to_pred_id[gt_id] = pred_id

                # ID-consistent TP
                if prev_pred_id is None or prev_pred_id == pred_id:
                    idtp += 1

        # MOTA
        mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0.0

        # IDF1 (simplified: F1 of ID-consistent TPs)
        total_matches = sum(
            len(_greedy_match(
                [obj["bbox"] for obj in self._gt_frames.get(f, [])],
                list((self._pred_frames.get(f, {}).get("tracks", {})).values()),
                self.iou_threshold,
            )[0])
            for f in all_frame_indices
        )
        precision_id = idtp / total_matches if total_matches > 0 else 0.0
        recall_id = idtp / total_gt if total_gt > 0 else 0.0
        idf1 = (
            2 * precision_id * recall_id / (precision_id + recall_id)
            if (precision_id + recall_id) > 0 else 0.0
        )

        # Count error
        pred_unique = self._predicted_unique_count()
        gt_unique = len(self._gt_unique_ids)
        count_err_abs = abs(pred_unique - gt_unique)
        count_err_pct = (count_err_abs / gt_unique * 100) if gt_unique > 0 else 0.0

        return {
            "id_switches": id_switches,
            "mota": round(mota, 4),
            "idf1": round(idf1, 4),
            "count_error_abs": count_err_abs,
            "count_error_pct": round(count_err_pct, 2),
            "predicted_unique_count": pred_unique,
            "gt_unique_count": gt_unique,
            "total_gt": total_gt,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

    def print_results(self, results: Optional[dict] = None) -> None:
        """Print a formatted summary table."""
        if results is None:
            results = self.evaluate()

        print("\n" + "=" * 50)
        print("  MOT Evaluation Results")
        print("=" * 50)
        print(f"  ID Switches:          {results['id_switches']}")
        print(f"  MOTA:                 {results['mota']:.4f}")
        print(f"  IDF1:                 {results['idf1']:.4f}")
        print(f"  Count Error (abs):    {results['count_error_abs']}")
        print(f"  Count Error (%):      {results['count_error_pct']:.1f}%")
        print(f"  Pred Unique Count:    {results['predicted_unique_count']}")
        print(f"  GT Unique Count:      {results['gt_unique_count']}")
        print(f"  Total GT Instances:   {results['total_gt']}")
        print(f"  Total FP:             {results['total_fp']}")
        print(f"  Total FN:             {results['total_fn']}")
        print("=" * 50 + "\n")

    def save_results(self, output_path: str, results: Optional[dict] = None) -> None:
        """Write results to a JSON file."""
        if results is None:
            results = self.evaluate()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_predictions(self) -> None:
        """Load the JSONL prediction log."""
        with open(self.pred_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fidx = record["frame_index"]
                self._pred_frames[fidx] = record

    def _load_ground_truth(self) -> None:
        """Load the ground-truth JSON file.

        Expected format:
        {
            "frames": {
                "0": [{"id": 1, "bbox": [x1, y1, x2, y2]}, ...],
                "5": [...]
            },
            "metadata": {"video": "...", "annotator": "...", ...}
        }
        """
        with open(self.gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for frame_key, objects in data.get("frames", {}).items():
            fidx = int(frame_key)
            self._gt_frames[fidx] = objects
            for obj in objects:
                self._gt_unique_ids.add(obj["id"])

    def _predicted_unique_count(self) -> int:
        """Count unique prediction track IDs across all frames."""
        all_ids = set()
        for record in self._pred_frames.values():
            all_ids.update(record.get("tracks", {}).keys())
        return len(all_ids)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluation.py <prediction_log.jsonl> <ground_truth.json>")
        sys.exit(1)

    evaluator = MOTEvaluator(sys.argv[1], sys.argv[2])
    results = evaluator.evaluate()
    evaluator.print_results(results)
