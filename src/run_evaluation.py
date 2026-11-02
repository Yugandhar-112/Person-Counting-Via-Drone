"""
run_evaluation.py

CLI script that:
  1. Runs the person-counting pipeline on test clips in headless mode.
  2. Runs the MOT evaluator against ground-truth annotations.
  3. Optionally runs with and without motion compensation, printing a
     side-by-side comparison table.

Usage:
    # Single clip evaluation:
    python src/run_evaluation.py --clip test_clips/sample.mp4 \\
        --gt ground_truth/sample.json

    # All clips in a directory:
    python src/run_evaluation.py --clips test_clips/ --gt-dir ground_truth/

    # Side-by-side comparison (baseline vs motion compensation):
    python src/run_evaluation.py --clip test_clips/sample.mp4 \\
        --gt ground_truth/sample.json --compare
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

# Ensure local module imports work regardless of working directory
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from evaluation import MOTEvaluator


def _find_latest_log(log_dir: str) -> Optional[str]:
    """Find the most recently created frames.jsonl in a log directory."""
    if not os.path.isdir(log_dir):
        return None
    run_dirs = sorted(
        [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))],
        reverse=True,
    )
    for rd in run_dirs:
        candidate = os.path.join(log_dir, rd, "frames.jsonl")
        if os.path.isfile(candidate):
            return candidate
    return None


def run_pipeline(
    clip_path: str,
    log_dir: str,
    enable_motion_compensation: bool = False,
    extra_flags: Optional[List[str]] = None,
) -> str:
    """Run main.py on a clip in headless mode, return path to the log file."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "main.py"),
        "--source", clip_path,
        "--headless",
        "--log-dir", log_dir,
    ]
    if enable_motion_compensation:
        cmd.append("--motion-compensation")
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Pipeline failed (exit code {result.returncode}):")
        print(result.stderr)
        return ""

    if result.stdout:
        print(result.stdout)

    log_file = _find_latest_log(log_dir)
    if log_file is None:
        print(f"  Warning: no log file found in {log_dir}")
        return ""
    return log_file


def evaluate_clip(
    clip_path: str,
    gt_path: str,
    log_base_dir: str,
    compare: bool = False,
) -> Dict[str, dict]:
    """Evaluate a single clip. Returns {config_name: results_dict}."""
    results = {}
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]

    configs = [("baseline", False)]
    if compare:
        configs.append(("motion_comp", True))

    for config_name, mc_enabled in configs:
        print(f"\n{'─' * 50}")
        print(f"  {clip_name} — config: {config_name}")
        print(f"{'─' * 50}")

        run_log_dir = os.path.join(log_base_dir, clip_name, config_name)
        log_file = run_pipeline(clip_path, run_log_dir, enable_motion_compensation=mc_enabled)

        if not log_file:
            print(f"  Skipping evaluation (no log).")
            continue

        evaluator = MOTEvaluator(log_file, gt_path)
        res = evaluator.evaluate()
        evaluator.print_results(res)

        # Save per-config results
        eval_out = os.path.join(run_log_dir, "eval_results.json")
        evaluator.save_results(eval_out, res)
        results[config_name] = res

    return results


def print_comparison_table(all_results: Dict[str, Dict[str, dict]]):
    """Print a side-by-side comparison table across clips and configs."""
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)

    metrics = ["id_switches", "mota", "idf1", "count_error_abs", "count_error_pct"]
    header = f"{'Clip':<20} {'Config':<15}" + "".join(f" {m:<18}" for m in metrics)
    print(header)
    print("-" * len(header))

    for clip_name, configs in sorted(all_results.items()):
        for config_name, res in sorted(configs.items()):
            row = f"{clip_name:<20} {config_name:<15}"
            for m in metrics:
                val = res.get(m, "N/A")
                if isinstance(val, float):
                    row += f" {val:<18.4f}"
                else:
                    row += f" {val!s:<18}"
            print(row)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline evaluation on test clips")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip", help="Path to a single test clip.")
    group.add_argument("--clips", help="Path to a directory of test clips.")

    parser.add_argument("--gt", default=None, help="Path to GT JSON (for single clip).")
    parser.add_argument("--gt-dir", default="ground_truth",
                        help="Directory of GT JSONs (for batch mode).")
    parser.add_argument("--log-dir", default="logs",
                        help="Base directory for pipeline logs.")
    parser.add_argument("--compare", action="store_true",
                        help="Run both baseline and motion-compensated, show comparison.")

    args = parser.parse_args()

    all_results: Dict[str, Dict[str, dict]] = {}

    if args.clip:
        # Single clip mode
        gt_path = args.gt
        if gt_path is None:
            basename = os.path.splitext(os.path.basename(args.clip))[0]
            gt_path = os.path.join(args.gt_dir, f"{basename}.json")

        if not os.path.isfile(gt_path):
            print(f"Ground truth not found: {gt_path}")
            sys.exit(1)

        clip_name = os.path.splitext(os.path.basename(args.clip))[0]
        all_results[clip_name] = evaluate_clip(
            args.clip, gt_path, args.log_dir, compare=args.compare
        )
    else:
        # Batch mode
        clip_dir = args.clips
        for fname in sorted(os.listdir(clip_dir)):
            if not fname.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                continue
            clip_path = os.path.join(clip_dir, fname)
            basename = os.path.splitext(fname)[0]
            gt_path = os.path.join(args.gt_dir, f"{basename}.json")

            if not os.path.isfile(gt_path):
                print(f"Skipping {fname}: no GT found at {gt_path}")
                continue

            all_results[basename] = evaluate_clip(
                clip_path, gt_path, args.log_dir, compare=args.compare
            )

    if args.compare and len(all_results) > 0:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
