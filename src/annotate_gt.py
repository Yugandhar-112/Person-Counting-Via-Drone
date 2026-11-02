"""
annotate_gt.py

Simple interactive ground-truth annotation tool.

Opens a video file, lets you click to mark person bounding boxes per frame,
assign persistent IDs, and saves to the JSON format expected by evaluation.py.

Controls:
  - 'n' / Right Arrow  : next frame (or skip N frames with --step)
  - 'p' / Left Arrow   : previous frame
  - 'd'                : delete the last bbox drawn on the current frame
  - 's'                : save annotations to disk
  - 'q' / Esc          : save and quit
  - Mouse drag         : draw a bounding box (left-click and drag)

ID assignment: each new bbox is auto-assigned the next ID. You can override
by pressing a digit key (0–9) before drawing to set the current ID.

This is intentionally not polished — just enough to produce ground truth
for 5-10 clips.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


class AnnotationTool:
    """Interactive bounding-box annotator."""

    def __init__(self, video_path: str, output_path: str, step: int = 1):
        self.video_path = video_path
        self.output_path = output_path
        self.step = step

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Cannot open video: {video_path}")
            sys.exit(1)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Annotation state: frame_index -> list of {"id": int, "bbox": [x1,y1,x2,y2]}
        self.annotations: dict = {}
        self.current_frame_idx = 0
        self.next_id = 0
        self.current_id = None  # override via digit keys

        # Drawing state
        self._drawing = False
        self._start_pt = None
        self._end_pt = None
        self._frame_display = None

        # Load existing annotations if file exists
        if os.path.exists(output_path):
            self._load_existing()

    def run(self):
        """Main event loop."""
        cv2.namedWindow("Annotate GT", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate GT", self._mouse_callback)

        while True:
            frame = self._read_frame(self.current_frame_idx)
            if frame is None:
                print("End of video.")
                break

            self._frame_display = frame.copy()
            self._draw_annotations(self._frame_display)
            self._draw_hud(self._frame_display)
            cv2.imshow("Annotate GT", self._frame_display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q") or key == 27:  # q or Esc
                self._save()
                break
            elif key == ord("n") or key == 83:  # n or right arrow
                self.current_frame_idx = min(
                    self.current_frame_idx + self.step, self.total_frames - 1
                )
            elif key == ord("p") or key == 81:  # p or left arrow
                self.current_frame_idx = max(self.current_frame_idx - self.step, 0)
            elif key == ord("d"):
                # Delete last bbox on current frame
                frame_key = str(self.current_frame_idx)
                if frame_key in self.annotations and self.annotations[frame_key]:
                    removed = self.annotations[frame_key].pop()
                    print(f"Removed ID {removed['id']} from frame {self.current_frame_idx}")
            elif key == ord("s"):
                self._save()
            elif ord("0") <= key <= ord("9"):
                self.current_id = key - ord("0")
                print(f"Next bbox will use ID {self.current_id}")

        cv2.destroyAllWindows()
        self.cap.release()

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_pt = (x, y)
            self._end_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end_pt = (x, y)
            # Live preview of the rectangle
            if self._frame_display is not None:
                preview = self._frame_display.copy()
                cv2.rectangle(preview, self._start_pt, self._end_pt, (0, 255, 255), 2)
                cv2.imshow("Annotate GT", preview)

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            self._end_pt = (x, y)

            x1 = min(self._start_pt[0], self._end_pt[0])
            y1 = min(self._start_pt[1], self._end_pt[1])
            x2 = max(self._start_pt[0], self._end_pt[0])
            y2 = max(self._start_pt[1], self._end_pt[1])

            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return  # too small, ignore

            # Assign ID
            if self.current_id is not None:
                use_id = self.current_id
                self.current_id = None  # consume the override
            else:
                use_id = self.next_id
                self.next_id += 1

            frame_key = str(self.current_frame_idx)
            if frame_key not in self.annotations:
                self.annotations[frame_key] = []
            self.annotations[frame_key].append({
                "id": use_id,
                "bbox": [x1, y1, x2, y2],
            })
            # Ensure next_id is always ahead
            if use_id >= self.next_id:
                self.next_id = use_id + 1

            print(f"Frame {self.current_frame_idx}: added ID {use_id} bbox [{x1},{y1},{x2},{y2}]")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_annotations(self, frame):
        frame_key = str(self.current_frame_idx)
        for obj in self.annotations.get(frame_key, []):
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj['id']}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        text = (
            f"Frame: {self.current_frame_idx}/{self.total_frames - 1}  |  "
            f"Step: {self.step}  |  "
            f"Annotations: {len(self.annotations.get(str(self.current_frame_idx), []))}  |  "
            f"[n/p] nav  [d] del  [s] save  [0-9] set ID  [q] quit"
        )
        cv2.putText(frame, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _save(self):
        data = {
            "metadata": {
                "video": self.video_path,
                "total_frames": self.total_frames,
                "fps": self.fps,
                "step": self.step,
            },
            "frames": self.annotations,
        }
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        total_boxes = sum(len(v) for v in self.annotations.values())
        print(f"Saved {len(self.annotations)} annotated frames "
              f"({total_boxes} boxes) to {self.output_path}")

    def _load_existing(self):
        with open(self.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.annotations = data.get("frames", {})
        # Restore next_id
        max_id = -1
        for objs in self.annotations.values():
            for obj in objs:
                if obj["id"] > max_id:
                    max_id = obj["id"]
        self.next_id = max_id + 1
        print(f"Loaded existing annotations ({len(self.annotations)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Interactive ground-truth annotation tool")
    parser.add_argument("video", help="Path to the video file to annotate.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: ground_truth/<video_basename>.json)")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame step size for navigation (annotate every Nth frame).")
    args = parser.parse_args()

    if args.output is None:
        basename = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join("ground_truth", f"{basename}.json")

    tool = AnnotationTool(args.video, args.output, step=args.step)
    tool.run()


if __name__ == "__main__":
    main()
