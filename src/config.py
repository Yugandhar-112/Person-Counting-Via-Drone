"""
config.py

Centralized configuration for the person-counting pipeline.

Every enhancement (motion compensation, tiling, confidence flags, etc.)
is controlled by a toggle here, defaulting to OFF so the baseline
pipeline behaviour is preserved unless the user explicitly enables a
feature via CLI flags.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """YOLO model parameters."""
    model_path: str = "yolov8n.pt"
    confidence: float = 0.5
    input_size: int = 640  # YOLO input resolution (square)


@dataclass
class TrackerConfig:
    """Centroid tracker parameters."""
    max_distance: int = 60
    max_missed_frames: int = 15


@dataclass
class TilingConfig:
    """Tiled / SAHI-style detection parameters."""
    tile_size: int = 640
    overlap_ratio: float = 0.2
    nms_iou_threshold: float = 0.5


@dataclass
class ConfidenceConfig:
    """Count-confidence flag thresholds."""
    high_motion_threshold: float = 30.0    # homography translation magnitude
    high_overlap_threshold: float = 0.3    # mean IOU among active tracks
    high_drop_rate_threshold: float = 0.05 # fraction of frames dropped


@dataclass
class PipelineConfig:
    """Top-level configuration — passed around the whole pipeline."""

    # --- Feature toggles (all default OFF) ---
    enable_motion_compensation: bool = False
    enable_tiling: bool = False
    enable_confidence_flag: bool = False
    enable_gap_interpolation: bool = False
    enable_privacy_blur: bool = False

    # --- Input / output ---
    source: str = "0"          # webcam index or file path or URL
    headless: bool = False     # skip cv2.imshow
    log_dir: Optional[str] = None  # if set, write per-frame JSON-lines logs

    # --- Sub-configs ---
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
