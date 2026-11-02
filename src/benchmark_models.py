"""
benchmark_models.py

Benchmarks multiple YOLO model sizes (nano, small, medium) on a test clip
to produce a Pareto analysis of speed vs. accuracy.

Measures:
  - FPS (inference time per frame)
  - Detection count per frame (mean)
  - mAP against ground truth (if available)

Outputs a summary table and optionally a matplotlib Pareto plot.

Usage:
    python src/benchmark_models.py --clip test_clips/sample.mp4
    python src/benchmark_models.py --clip test_clips/sample.mp4 \\
        --gt ground_truth/sample.json --plot
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

# Ensure local module imports work regardless of working directory
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import cv2
import numpy as np

from detector import PersonDetector


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


def benchmark_model(
    model_name: str,
    clip_path: str,
    max_frames: int = 300,
    gt_data: Optional[dict] = None,
) -> dict:
    """Benchmark a single YOLO model on a video clip.

    Returns:
        {
            "model": str,
            "avg_fps": float,
            "avg_detections": float,
            "total_frames": int,
            "avg_inference_ms": float,
            "precision": float (if GT provided),
            "recall": float (if GT provided),
        }
    """
    print(f"  Benchmarking {model_name}...")
    detector = PersonDetector(model_path=model_name, confidence=0.5)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"    Cannot open: {clip_path}")
        return {}

    inference_times = []
    detection_counts = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()

        inference_times.append(t1 - t0)
        detection_counts.append(len(detections))

        # Evaluate against GT if available
        if gt_data is not None:
            gt_objects = gt_data.get("frames", {}).get(str(frame_idx), [])
            if gt_objects:
                gt_boxes = [obj["bbox"] for obj in gt_objects]
                pred_boxes = [d["bbox"] for d in detections]
                tp, fp, fn = _match_frame(gt_boxes, pred_boxes)
                tp_total += tp
                fp_total += fp
                fn_total += fn

        frame_idx += 1

    cap.release()

    avg_time = np.mean(inference_times) if inference_times else 0
    result = {
        "model": model_name,
        "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
        "avg_detections": np.mean(detection_counts) if detection_counts else 0,
        "total_frames": frame_idx,
        "avg_inference_ms": avg_time * 1000,
    }

    if gt_data is not None:
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        result["precision"] = round(precision, 4)
        result["recall"] = round(recall, 4)

    return result


def _match_frame(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Simple greedy matching, returns (TP, FP, FN) counts."""
    matched_gt = set()
    matched_pred = set()

    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            iou = _iou(gb, pb)
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    for _, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def print_results_table(results: List[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  Model Benchmark Results")
    print("=" * 80)

    has_gt = any("precision" in r for r in results)

    header = f"{'Model':<15} {'FPS':>8} {'Infer(ms)':>10} {'Avg Dets':>10} {'Frames':>8}"
    if has_gt:
        header += f" {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = (
            f"{r['model']:<15} "
            f"{r['avg_fps']:>8.1f} "
            f"{r['avg_inference_ms']:>10.1f} "
            f"{r['avg_detections']:>10.1f} "
            f"{r['total_frames']:>8}"
        )
        if has_gt:
            row += f" {r.get('precision', 0):>10.4f} {r.get('recall', 0):>8.4f}"
        print(row)

    print("=" * 80 + "\n")


def make_pareto_plot(results: List[dict], output_path: str = "benchmark_pareto.png"):
    """Generate a Pareto plot of FPS vs recall."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    models = [r["model"] for r in results]
    fps_vals = [r["avg_fps"] for r in results]
    recall_vals = [r.get("recall", 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fps_vals, recall_vals, s=100, zorder=5)
    for i, model in enumerate(models):
        ax.annotate(model, (fps_vals[i], recall_vals[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)

    ax.set_xlabel("FPS (higher = faster)", fontsize=12)
    ax.set_ylabel("Recall (higher = more people found)", fontsize=12)
    ax.set_title("Model Size Pareto: Speed vs. Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Pareto plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model sizes")
    parser.add_argument("--clip", required=True, help="Path to test video clip.")
    parser.add_argument("--gt", default=None, help="Path to ground-truth JSON.")
    parser.add_argument("--models", nargs="+",
                        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                        help="Model names to benchmark.")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to process per model.")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a matplotlib Pareto plot.")
    args = parser.parse_args()

    gt_data = None
    if args.gt:
        import json
        with open(args.gt, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

    results = []
    for model_name in args.models:
        r = benchmark_model(model_name, args.clip, max_frames=args.max_frames, gt_data=gt_data)
        if r:
            results.append(r)

    print_results_table(results)

    if args.plot and any("recall" in r for r in results):
        make_pareto_plot(results)
    elif args.plot:
        print("No GT provided — cannot generate Pareto plot without recall data.")


if __name__ == "__main__":
    main()
