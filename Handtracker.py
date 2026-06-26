"""
hand_tracker.py

Thin wrapper around MediaPipe HandLandmarker.
Responsible for:
  - Auto-downloading the model asset if not present in assets/models/
  - Initialising the landmarker with the settings from HandConfig
  - Running per-frame inference and returning structured landmark data
  - Converting normalised landmarks → pixel coordinates

Design note:
  This class knows nothing about gestures or physics.
  It is a pure "perception" layer: raw landmark data in, nothing else out.
  All interpretation lives in gesture_detector.py.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import HandConfig


# ── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class HandLandmarks:
    """
    Processed result for a single detected hand in one frame.

    px_coords  : list of (x, y, z) tuples in pixel space.
                 z is the MediaPipe depth estimate (still normalised, not in px).
    norm_coords: list of (x, y, z) tuples in [0,1] normalised image space.
    handedness : 'Left' or 'Right' (after horizontal flip, if applied).
    world_coords: 3-D metric coordinates provided by the model (metres,
                  origin at wrist). Useful for joint-angle calculation.
    """
    px_coords:    List[Tuple[float, float, float]]
    norm_coords:  List[Tuple[float, float, float]]
    handedness:   str
    world_coords: Optional[List[Tuple[float, float, float]]] = None


# ── MediaPipe landmark index constants ───────────────────────────────────────
WRIST         = 0
THUMB_CMC     = 1;  THUMB_MCP   = 2;  THUMB_IP    = 3;  THUMB_TIP   = 4
INDEX_MCP     = 5;  INDEX_PIP   = 6;  INDEX_DIP   = 7;  INDEX_TIP   = 8
MIDDLE_MCP    = 9;  MIDDLE_PIP  = 10; MIDDLE_DIP  = 11; MIDDLE_TIP  = 12
RING_MCP      = 13; RING_PIP    = 14; RING_DIP    = 15; RING_TIP    = 16
PINKY_MCP     = 17; PINKY_PIP   = 18; PINKY_DIP   = 19; PINKY_TIP   = 20

# Skeleton edges used for drawing (index pairs)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (WRIST, PINKY_MCP),
]

# Palm reference points used for palm-centre calculation
PALM_REFS: Tuple[int, ...] = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Fingertip indices (order: thumb, index, middle, ring, pinky)
FINGERTIPS: Tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# MCP (knuckle) indices matching the fingertip order
FINGER_MCPS: Tuple[int, ...] = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) for per-frame VIDEO mode.

    Usage:
        tracker = HandTracker(cfg.hand, model_dir="assets/models")
        tracker.open()
        ...
        results = tracker.process(bgr_frame, timestamp_ms)
        ...
        tracker.close()

    Or as a context manager:
        with HandTracker(cfg.hand) as tracker:
            results = tracker.process(frame, ts)
    """

    def __init__(self, cfg: HandConfig, model_dir: str = "assets/models"):
        self._cfg       = cfg
        self._model_dir = model_dir
        self._landmarker: Optional[mp_vision.HandLandmarker] = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def open(self) -> None:
        """Download model if needed, then create the HandLandmarker."""
        model_path = self._ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self._cfg.max_hands,
            min_hand_detection_confidence=self._cfg.detection_confidence,
            min_hand_presence_confidence=self._cfg.presence_confidence,
            min_tracking_confidence=self._cfg.tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    def close(self) -> None:
        """Release the underlying MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── inference ─────────────────────────────────────────────────────────
    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        """
        Run hand landmark detection on one BGR frame.

        Args:
            bgr_frame    : OpenCV BGR image.
            timestamp_ms : Monotonic timestamp in milliseconds.
                           Must be strictly increasing across calls.

        Returns:
            List of HandLandmarks, one per detected hand (may be empty).
        """
        if self._landmarker is None:
            raise RuntimeError("HandTracker.open() must be called before process().")

        h, w = bgr_frame.shape[:2]
        rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # MediaPipe requires a contiguous array
        rgb   = np.ascontiguousarray(rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        hands: List[HandLandmarks] = []
        for idx, lm_list in enumerate(result.hand_landmarks):
            px  = [(lm.x * w, lm.y * h, lm.z) for lm in lm_list]
            nrm = [(lm.x,     lm.y,     lm.z)  for lm in lm_list]

            # world landmarks (metric, origin at wrist)
            wld = None
            if result.hand_world_landmarks:
                wl = result.hand_world_landmarks[idx]
                wld = [(lm.x, lm.y, lm.z) for lm in wl]

            handedness = "Unknown"
            if result.handedness:
                cats = result.handedness[idx]
                if cats:
                    handedness = cats[0].display_name   # 'Left' or 'Right'

            hands.append(HandLandmarks(
                px_coords=px,
                norm_coords=nrm,
                handedness=handedness,
                world_coords=wld,
            ))

        return hands

    # ── model management ──────────────────────────────────────────────────
    def _ensure_model(self) -> str:
        """
        Return the absolute path to the model file.
        Downloads it from the official bucket if it doesn't exist yet.
        The download path is: <model_dir>/<model_filename>
        """
        os.makedirs(self._model_dir, exist_ok=True)
        model_path = os.path.join(self._model_dir, self._cfg.model_filename)
        if not os.path.exists(model_path):
            print(f"[HandTracker] Downloading hand-landmark model …")
            print(f"  URL  : {self._cfg.model_url}")
            print(f"  Dest : {model_path}")
            urllib.request.urlretrieve(self._cfg.model_url, model_path)
            print("[HandTracker] Download complete.")
        return model_path


# ── Utility functions ─────────────────────────────────────────────────────────

def palm_center_px(lm: HandLandmarks) -> np.ndarray:
    """
    Return the 2-D pixel centroid of the palm reference points.
    This gives a more stable position than the wrist alone.
    """
    pts = [lm.px_coords[i][:2] for i in PALM_REFS]
    return np.mean(pts, axis=0)


def draw_skeleton(
    frame:     np.ndarray,
    lm:        HandLandmarks,
    bone_color: Tuple[int,int,int] = (0, 200, 60),
    joint_color:Tuple[int,int,int] = (0, 140, 255),
    bone_thickness: int = 1,
    joint_radius:   int = 3,
) -> None:
    """
    Draw the hand skeleton (bones + joints) onto `frame` in place.
    Kept here because it needs the landmark index constants defined above.
    Controlled by DebugConfig.show_skeleton in the main loop.
    """
    pts = [(int(x), int(y)) for x, y, *_ in lm.px_coords]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], bone_color, bone_thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, joint_radius, joint_color, -1, cv2.LINE_AA)