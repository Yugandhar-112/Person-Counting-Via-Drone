"""
stream_reader.py

Connects to a video source and yields frames for the rest of the pipeline
to consume. Works with either:
  - a laptop webcam (pass an integer device index, e.g. 0)
  - a live network stream, e.g. an ESP32-CAM MJPEG stream URL
"""

import cv2


class StreamReader:
    """Wraps cv2.VideoCapture to read frames from a webcam or a stream URL."""

    def __init__(self, source):
        """
        Args:
            source: int (webcam device index, e.g. 0) or str (stream URL,
                e.g. "http://192.168.4.1:81/stream" for a typical ESP32-CAM).
        """
        self.source = source
        self.capture = None

    def open(self) -> bool:
        """Open the connection to the source. Returns True on success."""
        self.capture = cv2.VideoCapture(self.source)
        return self.capture.isOpened()

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        return self.capture.read()

    def frames(self):
        """Generator that yields frames until the source ends or errors."""
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        while True:
            success, frame = self.capture.read()
            if not success:
                break
            yield frame

    def release(self):
        """Release the video capture resource."""
        if self.capture is not None:
            self.capture.release()


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

    print(f"Connected to {source}. Press 'q' to quit.")
    for frame in reader.frames():
        cv2.imshow("Video Source", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()