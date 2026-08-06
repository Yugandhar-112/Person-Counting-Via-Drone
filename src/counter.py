"""
counter.py

Thin wrapper that turns tracked IDs into a simple running headcount,
kept separate from the tracker so counting rules (e.g. minimum frames
seen before counting someone) can evolve independently.
"""


class PersonCounter:
    """Maintains a running unique-person count from a tracker's output."""

    def __init__(self):
        self.seen_ids = set()

    def update(self, current_frame_ids: dict) -> int:
        """
        Args:
            current_frame_ids: dict of track_id -> bbox for this frame,
                as returned by CentroidTracker.update().

        Returns:
            The current total unique-person count.
        """
        self.seen_ids.update(current_frame_ids.keys())
        return len(self.seen_ids)

    @property
    def count(self) -> int:
        return len(self.seen_ids)
