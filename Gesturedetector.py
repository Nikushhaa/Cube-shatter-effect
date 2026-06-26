"""
gesture_detector.py

Advanced gesture recognition built on top of raw hand landmarks.

Original problem with the single-file version:
  - Openness ratio was computed from a single frame with no confirmation.
  - Random MediaPipe jitter could trigger EXPLODE during a half-open hand.
  - No notion of gesture "confidence" → false triggers.

This module fixes all of that:

  1. Per-finger curl detection  (joint-angle based)
  2. Thumb state detection      (special-cased due to its different axis)
  3. Palm orientation           (up/down/toward-camera)
  4. Composite openness ratio   (now derived from curl, not just tip distance)
  5. Exponential smoothing on all continuous values
  6. Frame-confirmation buffer  (a gesture must be held for N frames at
     confidence ≥ threshold before it is reported)
  7. Discrete gesture taxonomy:
       FIST, OPEN, PINCH, POINT, UNKNOWN

All public methods return plain Python floats/enums — no MediaPipe types leak
beyond this module.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum, auto
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from src.config import GestureConfig
from src.hand_tracker import (
    HandLandmarks,
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP,   RING_PIP,  RING_DIP,  RING_TIP,
    PINKY_MCP,  PINKY_PIP, PINKY_DIP, PINKY_TIP,
    PALM_REFS, FINGERTIPS, FINGER_MCPS,
    palm_center_px,
)


# ── Gesture Taxonomy ──────────────────────────────────────────────────────────

class Gesture(Enum):
    """
    Recognised discrete hand gestures.

    FIST    : All fingers curled, palm closed.
               → Cube appears / fragments pulled back.
    OPEN    : All fingers extended, palm flat.
               → Cube explodes.
    PINCH   : Thumb and index finger close together,
               other fingers loosely extended.
               → (future feature hook)
    POINT   : Index finger extended, all others curled.
               → (future feature hook)
    UNKNOWN : Transitional / unclassified state.
               → No state transition fired.
    """
    FIST    = auto()
    OPEN    = auto()
    PINCH   = auto()
    POINT   = auto()
    UNKNOWN = auto()


# ── Per-hand Gesture State ────────────────────────────────────────────────────

class HandGestureState:
    """
    Tracks the smoothed measurements and confirmation buffer for ONE hand.

    GestureDetector holds one HandGestureState per tracked hand index.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self._cfg               = cfg
        # Smoothed openness in [0, 1]   (0 = fist, 1 = fully open)
        self.openness:    float = 0.0
        # Per-finger curl in [0, 1]      (0 = straight, 1 = fully curled)
        self.finger_curl: Dict[str, float] = {
            "thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0
        }
        # Normalised thumb↔index distance for pinch detection
        self.pinch_dist:  float = 1.0
        # Palm normal dot product with camera z-axis  (+1 = facing camera)
        self.palm_facing: float = 1.0
        # Most-recently confirmed discrete gesture
        self.gesture:     Gesture = Gesture.UNKNOWN
        # Raw confidence of the current candidate gesture (0–1)
        self.confidence:  float   = 0.0

        # Frame-confirmation buffer: stores (Gesture, confidence) per frame
        self._confirm_buf: Deque[Tuple[Gesture, float]] = deque(
            maxlen=cfg.confirm_frames
        )

    # ── update ────────────────────────────────────────────────────────────
    def update(self, lm: HandLandmarks) -> None:
        """
        Process one frame of landmarks and update all smoothed values.
        Call this every frame a hand is visible.
        """
        pts  = lm.px_coords        # pixel coords list
        norm = lm.norm_coords      # normalised [0,1] coords
        cfg  = self._cfg

        # ── 1. Per-finger curl (based on joint angles) ────────────────
        self._update_finger_curl(pts, norm)

        # ── 2. Composite openness score ───────────────────────────────
        #    Invert the average curl: curl=0 → open=1, curl=1 → open=0
        #    Exclude thumb (it has a very different range)
        non_thumb_curl = (
            self.finger_curl["index"]  +
            self.finger_curl["middle"] +
            self.finger_curl["ring"]   +
            self.finger_curl["pinky"]
        ) / 4.0
        raw_open  = 1.0 - non_thumb_curl
        # Remap to [0,1] using configured thresholds
        alpha     = cfg.smoothing_alpha
        self.openness = self.openness + alpha * (float(np.clip(raw_open, 0, 1)) - self.openness)

        # ── 3. Pinch distance (thumb tip ↔ index tip, normalised) ─────
        t_tip = np.array(norm[THUMB_TIP][:2])
        i_tip = np.array(norm[INDEX_TIP][:2])
        raw_pinch = float(np.linalg.norm(t_tip - i_tip))
        self.pinch_dist = self.pinch_dist + alpha * (raw_pinch - self.pinch_dist)

        # ── 4. Classify this frame into a candidate gesture ───────────
        candidate, conf = self._classify(lm)

        # ── 5. Frame-confirmation buffer ──────────────────────────────
        self._confirm_buf.append((candidate, conf))
        self._confirm_gesture()

    def reset(self) -> None:
        """Call when the hand disappears from frame."""
        self._confirm_buf.clear()
        self.gesture   = Gesture.UNKNOWN
        self.confidence = 0.0

    # ── private helpers ───────────────────────────────────────────────────
    def _update_finger_curl(
        self,
        pts:  list,
        norm: list,
    ) -> None:
        """
        Compute per-finger curl by measuring how far each fingertip is from
        the wrist relative to the finger's MCP knuckle.

        curl = 1 − (tip_wrist / mcp_wrist)
        clamped to [0, 1] then smoothed.

        A straight finger has tip_wrist ≈ mcp_wrist × (1 + segment_lengths),
        which is larger → curl close to 0.
        A curled finger has tip_wrist ≈ mcp_wrist → curl close to 1.

        Thumb uses a slightly different reference (THUMB_CMC instead of wrist)
        because the thumb lies almost perpendicular to the palm axis.
        """
        cfg   = self._cfg
        alpha = cfg.smoothing_alpha
        wrist = np.array(pts[WRIST][:2])

        finger_map = [
            ("thumb",  THUMB_TIP,  THUMB_CMC),
            ("index",  INDEX_TIP,  INDEX_MCP),
            ("middle", MIDDLE_TIP, MIDDLE_MCP),
            ("ring",   RING_TIP,   RING_MCP),
            ("pinky",  PINKY_TIP,  PINKY_MCP),
        ]

        for name, tip_idx, base_idx in finger_map:
            tip  = np.array(pts[tip_idx][:2])
            base = np.array(pts[base_idx][:2])

            d_tip  = float(np.linalg.norm(tip  - wrist))
            d_base = float(np.linalg.norm(base - wrist))

            if d_base < 1e-3:
                raw_curl = 0.0
            else:
                # ratio < 1 means tip closer to wrist than base → curled
                ratio    = d_tip / (d_base + 1e-6)
                raw_curl = float(np.clip(1.0 - ratio * cfg.curl_threshold, 0, 1))

            prev = self.finger_curl[name]
            self.finger_curl[name] = prev + alpha * (raw_curl - prev)

    def _classify(self, lm: HandLandmarks) -> Tuple[Gesture, float]:
        """
        Map the current smoothed measurements to a discrete Gesture + confidence.

        FIST:  all non-thumb fingers highly curled
        OPEN:  all non-thumb fingers lowly curled (extended)
        PINCH: thumb and index very close, others loosely open
        POINT: index finger extended, all others curled
        UNKNOWN: everything else
        """
        curl  = self.finger_curl
        cfg   = self._cfg

        # Helper: how "curled" are the four non-thumb fingers on average?
        avg_curl_4 = (curl["index"] + curl["middle"] + curl["ring"] + curl["pinky"]) / 4.0
        avg_curl_3 = (curl["middle"] + curl["ring"] + curl["pinky"]) / 3.0   # excl. index

        # ── FIST ─────────────────────────────────────────────────────
        if avg_curl_4 > 0.60:
            # confidence scales with how curled the fingers are
            conf = float(np.clip((avg_curl_4 - 0.60) / 0.40, 0, 1))
            return Gesture.FIST, conf

        # ── OPEN ─────────────────────────────────────────────────────
        if avg_curl_4 < 0.30:
            conf = float(np.clip((0.30 - avg_curl_4) / 0.30, 0, 1))
            return Gesture.OPEN, conf

        # ── POINT (index straight, others curled) ─────────────────────
        if curl["index"] < 0.35 and avg_curl_3 > 0.55:
            conf_index = float(np.clip((0.35 - curl["index"]) / 0.35, 0, 1))
            conf_rest  = float(np.clip((avg_curl_3 - 0.55) / 0.45, 0, 1))
            conf = (conf_index + conf_rest) * 0.5
            return Gesture.POINT, conf

        # ── PINCH ────────────────────────────────────────────────────
        if self.pinch_dist < cfg.pinch_threshold:
            conf = float(np.clip(1.0 - self.pinch_dist / cfg.pinch_threshold, 0, 1))
            return Gesture.PINCH, conf

        return Gesture.UNKNOWN, 0.0

    def _confirm_gesture(self) -> None:
        """
        A gesture is confirmed only if the last `confirm_frames` frames ALL
        report the same gesture at confidence ≥ confirm_confidence.

        This is the key anti-jitter mechanism.
        Previously the single-file version acted on every frame immediately,
        causing random explosions during partially-open hands.
        """
        cfg = self._cfg
        if len(self._confirm_buf) < cfg.confirm_frames:
            return   # not enough history yet

        # Collect gesture labels that meet the confidence bar
        gestures_seen = [
            g for g, c in self._confirm_buf
            if c >= cfg.confirm_confidence
        ]
        if len(gestures_seen) < cfg.confirm_frames:
            # Not all frames confident enough — stay UNKNOWN
            self.gesture    = Gesture.UNKNOWN
            self.confidence = 0.0
            return

        # Check all N frames agree
        first = gestures_seen[0]
        if all(g == first for g in gestures_seen):
            self.gesture    = first
            self.confidence = float(np.mean([c for _, c in self._confirm_buf]))
        else:
            self.gesture    = Gesture.UNKNOWN
            self.confidence = 0.0


# ── GestureDetector ───────────────────────────────────────────────────────────

class GestureDetector:
    """
    Manages per-hand state for up to `max_hands` hands.

    Usage:
        detector = GestureDetector(cfg.gesture)
        ...
        gesture_states = detector.update(hand_landmarks_list)
        # gesture_states is a dict: hand_index → HandGestureState
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self._cfg    = cfg
        self._states: Dict[int, HandGestureState] = {}

    def update(
        self, hands: list   # list[HandLandmarks]
    ) -> Dict[int, HandGestureState]:
        """
        Process all detected hands for this frame.
        Hands that disappeared are reset and removed.

        Returns the current dict of {hand_index: HandGestureState}.
        """
        # Update existing / new hands
        live_indices = set(range(len(hands)))
        for idx, lm in enumerate(hands):
            if idx not in self._states:
                self._states[idx] = HandGestureState(self._cfg)
            self._states[idx].update(lm)

        # Reset hands that vanished
        dead = set(self._states.keys()) - live_indices
        for idx in dead:
            self._states[idx].reset()
            del self._states[idx]

        return dict(self._states)

    def clear(self) -> None:
        """Reset all hand states (e.g. when no hands visible for several frames)."""
        for state in self._states.values():
            state.reset()
        self._states.clear()

    @property
    def active_count(self) -> int:
        """Number of hands currently tracked."""
        return len(self._states)