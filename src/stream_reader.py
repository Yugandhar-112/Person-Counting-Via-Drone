"""
stream_reader.py

Connects to the drone's Wi-Fi camera (e.g. an ESP32-CAM MJPEG stream) and
yields video frames for the rest of the pipeline to consume.
"""

import cv2


class StreamReader:
    """Wraps cv2.VideoCapture to read a live MJPEG/HTTP video stream."""

    def __init__(self, stream_url: str):
        """
        Args:
            stream_url: URL of the camera's video stream,
                e.g. "http://192.168.4.1:81/stream" for a typical ESP32-CAM.
        """
        self.stream_url = stream_url
        self.capture = None

    def open(self) -> bool:
        """Open the connection to the stream. Returns True on success."""
        self.capture = cv2.VideoCapture(self.stream_url)
        return self.capture.isOpened()

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.capture is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        return self.capture.read()

    def frames(self):
        """Generator that yields frames until the stream ends or errors."""
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
    # Quick manual test: point this at your ESP32-CAM stream URL and confirm
    # frames are coming through before wiring in detection/tracking.
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.4.1:81/stream"
    reader = StreamReader(url)

    if not reader.open():
        print(f"Failed to open stream at {url}")
        sys.exit(1)

    print(f"Connected to {url}. Press 'q' to quit.")
    for frame in reader.frames():
        cv2.imshow("Drone Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()
