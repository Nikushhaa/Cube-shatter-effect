"""
hand_shatter_effect.py

Webcam hand tracking where a sparkle-cube "object" reacts to how OPEN
or CLOSED your hand is:

  - Closed fist  -> the object collapses into a small bright cluster,
                    like you're holding a tiny glowing object.
  - Open hand    -> the object EXPLODES outward into a scattered
                    particle cloud, like the reference screenshot.

Built with OpenCV + MediaPipe Tasks API only.

INSTALL
-------
    pip install opencv-python mediapipe numpy

RUN
---
    python hand_shatter_effect.py

First run auto-downloads a small model file (hand_landmarker.task)
into the same folder. Needs internet once.

Press 'q' to quit.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ===========================================================================
# STEP 0 — CONFIG
# These are the values you'll most likely want to tweak.
# ===========================================================================

CAMERA_INDEX = 0
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6

# --- Hand-openness calibration ---
# We measure: average distance from fingertips to palm center, divided
# by a hand-size reference (wrist -> middle-knuckle distance). This
# ratio is roughly camera-distance-independent, but DOES vary a bit
# between people/hands. If the effect feels "always exploded" or
# "never opens", print(raw) (see below) and adjust these two numbers.
CLOSED_RATIO = 0.55   # typical ratio for a fully closed fist
OPEN_RATIO = 2.0       # typical ratio for a fully open, spread hand
SMOOTHING = 0.25       # 0..1, higher = snappier, lower = smoother

# --- Object / cloud look ---
CLOUD_POINTS = 450     # more = denser cloud, but slower
MIN_SCALE = 10          # px — size of the "held object" when fist is closed
MAX_SCALE = 160         # px — half-size of the cube when hand is fully open

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

WRIST = 0
PALM_REF_POINTS = (0, 5, 9, 13, 17)   # wrist + 4 knuckles -> palm center
FINGERTIPS = (4, 8, 12, 16, 20)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-landmark model (one-time, a few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to:", MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not auto-download the model file. Download it manually "
            f"from:\n  {MODEL_URL}\nand save it as:\n  {MODEL_PATH}\n"
            f"Original error: {e}"
        )


# ===========================================================================
# STEP 1 — LANDMARK HELPERS
# ===========================================================================

def landmarks_to_pixels(hand_landmarks, w, h):
    return [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks]


def draw_hand_skeleton(frame, hand_landmarks, w, h):
    pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts_px[a], pts_px[b], (0, 200, 0), 1, cv2.LINE_AA)
    for (x, y) in pts_px:
        cv2.circle(frame, (x, y), 3, (0, 140, 255), -1, cv2.LINE_AA)


def hand_openness_raw(pts):
    """
    Returns a single number: average (fingertip-to-palm distance) /
    (hand-size reference). Small when the fist is closed, big when the
    hand is fully open and fingers are spread.
    """
    wrist = np.array(pts[WRIST][:2])
    mcp_mid = np.array(pts[9][:2])
    hand_size = np.linalg.norm(mcp_mid - wrist)
    if hand_size < 1e-3:
        return 0.0

    palm_center = np.mean([pts[i][:2] for i in PALM_REF_POINTS], axis=0)
    dists = [np.linalg.norm(np.array(pts[t][:2]) - palm_center) for t in FINGERTIPS]
    return float(np.mean(dists) / hand_size)


def palm_center_px(pts):
    return np.mean([pts[i][:2] for i in PALM_REF_POINTS], axis=0)


# ===========================================================================
# STEP 2 — THE "OBJECT": a fixed cloud of points + a wireframe cube
# Each point gets ONE random direction + reach, chosen once at startup.
# Every frame we just re-scale this fixed pattern by the openness-driven
# `scale` value — that's what makes it look like the SAME object
# shrinking into your fist or exploding outward, instead of random noise.
# ===========================================================================

class ShatterCloud:
    def __init__(self, n_points=CLOUD_POINTS, seed=42):
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, n_points)
        self.reach = rng.uniform(0.15, 1.0, n_points)   # 0=core, 1=outer edge
        self.dirx = np.cos(angles)
        self.diry = np.sin(angles)
        self.size = rng.uniform(1.0, 3.0, n_points)
        palette = np.array([
            [255, 255, 255], [220, 255, 230], [200, 255, 180], [255, 240, 200],
        ])
        self.color = palette[rng.integers(0, len(palette), n_points)]

    def draw(self, frame, cx, cy, scale, t):
        scale_frac = np.clip(scale / MAX_SCALE, 0.15, 1.0)
        # subtle shimmer so it doesn't look frozen, more shimmer the more "exploded"
        jitter = 6.0 * scale_frac
        jx = np.sin(t * 3.0 + self.reach * 30) * jitter
        jy = np.cos(t * 2.7 + self.reach * 17) * jitter

        x = cx + self.dirx * self.reach * scale + jx
        y = cy + self.diry * self.reach * scale + jy

        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        for i in range(len(x)):
            px, py = int(x[i]), int(y[i])
            if 0 <= px < w and 0 <= py < h:
                brightness = (1.0 - 0.4 * self.reach[i]) * scale_frac
                color = tuple(int(c * brightness) for c in self.color[i])
                radius = max(1, int(self.size[i] * scale_frac + 1))
                cv2.circle(layer, (px, py), radius, color, -1)
        cv2.add(frame, layer, dst=frame)


def draw_scaled_cube(frame, cx, cy, scale, color=(255, 255, 255)):
    """Wireframe cube (front face + skewed back face) sized by `scale`,
    with corner dots — matches the look of the reference screenshot."""
    x0, y0 = cx - scale, cy - scale
    x1, y1 = cx + scale, cy + scale
    skew = 0.25 * scale

    front = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    back = [(x0 + skew, y0 - skew), (x1 + skew, y0 - skew),
             (x1 + skew, y1 - skew), (x0 + skew, y1 - skew)]

    pts_f = np.array(front, dtype=int)
    pts_b = np.array(back, dtype=int)
    cv2.polylines(frame, [pts_f], True, color, 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts_b], True, color, 1, cv2.LINE_AA)
    for a, b in zip(front, back):
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, cv2.LINE_AA)
    for (px, py) in front + back:
        cv2.circle(frame, (int(px), int(py)), 5, color, -1, cv2.LINE_AA)


# ===========================================================================
# STEP 3 — MAIN LOOP
# ===========================================================================

def main():
    ensure_model_downloaded()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    # cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows fix if laggy

    cloud = ShatterCloud()
    prev_openness = {}   # one smoothed value per hand index (0, 1)
    t0 = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(now * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    pts = landmarks_to_pixels(hand_landmarks, w, h)

                    raw = hand_openness_raw(pts)
                    target = np.clip(
                        (raw - CLOSED_RATIO) / (OPEN_RATIO - CLOSED_RATIO), 0.0, 1.0
                    )
                    prev = prev_openness.get(idx, target)
                    smoothed = prev + (target - prev) * SMOOTHING
                    prev_openness[idx] = smoothed

                    cx, cy = palm_center_px(pts)
                    scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * smoothed

                    draw_hand_skeleton(frame, hand_landmarks, w, h)
                    draw_scaled_cube(frame, cx, cy, scale)
                    cloud.draw(frame, cx, cy, scale, now - t0)

                    # Debug readout -- handy for tuning CLOSED_RATIO / OPEN_RATIO.
                    cv2.putText(
                        frame, f"openness: {smoothed:.2f} (raw {raw:.2f})",
                        (10, 25 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )

            cv2.imshow("Hand Shatter Effect (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
