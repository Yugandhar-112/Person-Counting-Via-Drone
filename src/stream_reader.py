"""
stream_reader.py

Connects to a video source and yields frames for the rest of the pipeline
to consume. Works with either:
  - a laptop webcam (pass an integer device index, e.g. 0)
  - a live network stream, e.g. an ESP32-CAM MJPEG stream URL
  - a pre-recorded video file path

Enhanced with:
  - frame_index counter (for logging / evaluation alignment)
  - fps property (playback-rate-aware evaluation)
  - frame-drop tracking
  - FrameResult dataclass (for gap-interpolation support in Phase 6)
"""

import time
from dataclasses import dataclass
from typing import Generator, Optional

import cv2


@dataclass
class FrameResult:
    """Enriched frame payload yielded by StreamReader.frame_results()."""
    frame: object               # numpy ndarray (the actual image)
    frame_index: int            # zero-based sequential counter
    is_dropped: bool = False    # True if this was a failed read (gap)
    gap_duration: float = 0.0   # seconds elapsed since last good frame


class StreamReader:
    """Wraps cv2.VideoCapture to read frames from a webcam or a stream URL."""

    def __init__(self, source):
        """
        Args:
            source: int (webcam device index, e.g. 0) or str (stream URL,
                e.g. "http://192.168.4.1:81/stream" for a typical ESP32-CAM,
                or a path to a .mp4 file).
        """
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self._frame_index: int = 0
        self._drop_count: int = 0
        self._total_reads: int = 0

    def open(self) -> bool:
        """Open the connection to the source. Returns True on success."""
        self.capture = cv2.VideoCapture(self.source)
        return self.capture.isOpened()

    @property
    def fps(self) -> float:
        """Frames-per-second reported by the capture backend.

        Returns 0.0 if the capture is not opened or the value is
        unavailable (common for live webcams).
        """
        if self.capture is None:
            return 0.0
        val = self.capture.get(cv2.CAP_PROP_FPS)
        return val if val > 0 else 0.0

    @property
    def frame_index(self) -> int:
        """Zero-based index of the *next* frame that will be read."""
        return self._frame_index

    @property
    def drop_count(self) -> int:
        """Total number of dropped (failed read) frames so far."""
        return self._drop_count

    @property
    def drop_rate(self) -> float:
        """Fraction of reads that failed (0.0 – 1.0)."""
        if self._total_reads == 0:
            return 0.0
        return self._drop_count / self._total_reads

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        return self.capture.read()

    def frames(self) -> Generator:
        """Generator that yields raw frames until the source ends or errors.

        This is the original simple interface — frame_index is still
        incremented internally so it stays in sync.
        """
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        while True:
            self._total_reads += 1
            success, frame = self.capture.read()
            if not success:
                # For video files, end-of-file is expected.
                # For live streams, a single failure is a drop.
                if self._is_live_source():
                    self._drop_count += 1
                    continue  # skip and retry on live sources
                break
            self._frame_index += 1
            yield frame

    def frame_results(self) -> Generator[FrameResult, None, None]:
        """Generator that yields FrameResult objects, including drop info.

        Use this instead of frames() when gap-interpolation or detailed
        drop tracking is needed.
        """
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")

        last_good_time = time.perf_counter()
        consecutive_drops = 0
        max_consecutive_drops = 30  # give up after this many in a row

        while True:
            self._total_reads += 1
            success, frame = self.capture.read()
            now = time.perf_counter()

            if not success:
                if self._is_live_source() and consecutive_drops < max_consecutive_drops:
                    self._drop_count += 1
                    consecutive_drops += 1
                    gap = now - last_good_time
                    yield FrameResult(
                        frame=None,
                        frame_index=self._frame_index,
                        is_dropped=True,
                        gap_duration=gap,
                    )
                    self._frame_index += 1
                    continue
                break  # end of file or too many consecutive drops

            consecutive_drops = 0
            gap = now - last_good_time
            last_good_time = now

            result = FrameResult(
                frame=frame,
                frame_index=self._frame_index,
                is_dropped=False,
                gap_duration=gap,
            )
            self._frame_index += 1
            yield result

    def release(self):
        """Release the video capture resource."""
        if self.capture is not None:
            self.capture.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_live_source(self) -> bool:
        """Heuristic: webcam indices and URLs are 'live'; file paths are not."""
        if isinstance(self.source, int):
            return True
        if isinstance(self.source, str) and self.source.startswith(("http://", "https://", "rtsp://")):
            return True
        return False


if __name__ == "__main__":
    # Quick manual test: defaults to webcam 0. Pass a stream URL instead to
    # test against an ESP32-CAM, e.g.:
    #   python stream_reader.py http://192.168.4.1:81/stream
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(arg) if arg.isdigit() else arg
    reader = StreamReader(source)

    if not reader.open():
        print(f"Failed to open source: {source}")
        sys.exit(1)

    print(f"Connected to {source} (FPS: {reader.fps:.1f}). Press 'q' to quit.")
    for frame in reader.frames():
        cv2.imshow("Video Source", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Frames read: {reader.frame_index}, drops: {reader.drop_count}")
    reader.release()
    cv2.destroyAllWindows()