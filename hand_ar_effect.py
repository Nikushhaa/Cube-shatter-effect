"""
hand_ar_effect.py

Real-time webcam hand tracking with an attached "point cloud / cube" AR
effect, built with ONLY Python (OpenCV + MediaPipe). No external apps
(no TouchDesigner, no Unity) — everything runs in this one script.

IMPORTANT — API NOTE
---------------------
Newer MediaPipe releases (0.10.31+) removed the old `mediapipe.solutions`
API (the `mp.solutions.hands` you may see in older tutorials). This
script uses MediaPipe's current "Tasks" API instead
(`mediapipe.tasks.python.vision.HandLandmarker`), which is what recent
`pip install mediapipe` actually gives you.

WHAT IT DOES
-------------
1. Opens your webcam.
2. Detects up to 2 hands per frame and tracks all 21 landmarks per hand
   (palm, knuckles, fingertips).
3. Draws the hand skeleton (the connected dots/lines) by hand, with
   OpenCV — since the old built-in drawing helper no longer exists.
4. Draws a pseudo-3D wireframe cube around the hand, whose size and
   "depth skew" change as you move your hand closer/farther from the
   camera.
5. Spawns a glowing particle / point-cloud effect attached to your
   fingertips, mimicking the sparkle effect from motion-graphics
   software.

INSTALL
-------
    pip install opencv-python mediapipe numpy

RUN
---
    python hand_ar_effect.py

The FIRST run will auto-download a small model file
(hand_landmarker.task, a few MB) into the same folder as this script.
You need an internet connection for that one-time download.

Press 'q' in the video window to quit.

WINDOWS NOTE: if the webcam feels laggy or slow to open, change the
cv2.VideoCapture(0) line in main() to cv2.VideoCapture(0, cv2.CAP_DSHOW).
"""

import os
import time
import random
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ===========================================================================
# STEP 0 — CONFIG
# ===========================================================================

CAMERA_INDEX = 0
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6

PARTICLES_PER_FRAME_PER_HAND = 18
MAX_PARTICLES = 3000
PARTICLE_LIFETIME_RANGE = (0.5, 1.4)  # seconds, randomised per particle

# Where the hand-landmark model file lives / gets downloaded to.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices MediaPipe gives us for each hand (0-20).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# The 20 bone connections between the 21 landmarks (palm + 4 fingers + thumb).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                               # palm base
]


# ===========================================================================
# STEP 1 — MAKE SURE WE HAVE THE MODEL FILE (auto-download once)
# ===========================================================================

def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-landmark model (one-time, a few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to:", MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not auto-download the model file. Check your internet "
            "connection, or manually download it from:\n"
            f"  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}\n"
            f"Original error: {e}"
        )


# ===========================================================================
# STEP 2 — PARTICLE SYSTEM
# A tiny physics simulation: each particle has a position, velocity and
# a limited lifetime. This is what creates the "sparkle cloud" look.
# ===========================================================================

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    PALETTE = [
        (255, 255, 255),   # white
        (220, 255, 230),   # pale mint
        (200, 255, 180),   # pale green
        (255, 240, 200),   # warm white
    ]

    def __init__(self, x, y, depth_scale=1.0):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.2, 1.4) * depth_scale
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        self.max_life = random.uniform(*PARTICLE_LIFETIME_RANGE)
        self.life = self.max_life
        self.size = random.uniform(1, 3) * depth_scale
        self.color = random.choice(self.PALETTE)

    def update(self, dt):
        self.vx += random.uniform(-0.15, 0.15)
        self.vy += random.uniform(-0.15, 0.15)
        self.vx *= 0.96
        self.vy *= 0.96
        self.x += self.vx
        self.y += self.vy
        self.life -= dt
        return self.life > 0

    def alpha(self):
        return max(0.0, self.life / self.max_life)


class ParticleSystem:
    def __init__(self, max_particles=MAX_PARTICLES):
        self.max_particles = max_particles
        self.particles = deque(maxlen=max_particles)

    def spawn(self, x, y, n, depth_scale=1.0):
        for _ in range(n):
            self.particles.append(Particle(x, y, depth_scale))

    def update(self, dt):
        alive = deque(maxlen=self.max_particles)
        for p in self.particles:
            if p.update(dt):
                alive.append(p)
        self.particles = alive

    def draw(self, frame):
        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        for p in self.particles:
            a = p.alpha()
            if a <= 0:
                continue
            color = tuple(int(c * a) for c in p.color)
            px, py = int(p.x), int(p.y)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(layer, (px, py), max(1, int(p.size)), color, -1)
        cv2.add(frame, layer, dst=frame)  # additive blend -> glow look


# ===========================================================================
# STEP 3 — HELPERS THAT TURN LANDMARKS INTO PIXELS / SHAPES
# ===========================================================================

def draw_hand_skeleton(frame, hand_landmarks, w, h):
    """
    hand_landmarks here is the list of 21 landmark objects MediaPipe's
    Tasks API gives us for ONE hand (each with .x, .y, .z, normalized
    0..1). We replicate the classic 'dots + lines' skeleton drawing
    that used to come from mp.solutions.drawing_utils.
    """
    pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts_px[a], pts_px[b], (0, 220, 0), 2, cv2.LINE_AA)
    for (x, y) in pts_px:
        cv2.circle(frame, (x, y), 4, (0, 140, 255), -1, cv2.LINE_AA)


def landmarks_to_pixels(hand_landmarks, w, h):
    """
    Converts the 21 normalized landmarks (x, y in 0..1, z = relative
    depth) into pixel coordinates we can use for drawing / spawning
    particles, keeping z as-is for depth calculations.
    """
    return [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks]


def hand_depth_scale(pts):
    """
    Turn the wrist's z value into a 'how close is the hand' multiplier.
    MediaPipe's z is unitless, roughly centred on 0, negative = closer
    to the camera. We squash it into a friendly 0.6 - 1.6 range so a
    closer hand -> bigger cube / particles.
    """
    wrist_z = pts[WRIST][2]
    scale = 1.0 - wrist_z * 4.0
    return float(np.clip(scale, 0.6, 1.6))


def hand_bbox(pts, w, h, pad=40):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    return x0, y0, x1, y1


def draw_pseudo_cube(frame, x0, y0, x1, y1, depth_scale, color=(255, 255, 255)):
    """
    OpenCV has no real 3D renderer, so we FAKE a cube: draw the 'front
    face' rectangle, draw a second 'back face' rectangle offset by a
    skew amount, then connect matching corners. The skew amount is
    driven by depth_scale, so the cube visibly distorts as your hand
    moves toward/away from the camera.
    """
    skew = 20 * depth_scale
    front = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    back = [(x0 + skew, y0 - skew), (x1 + skew, y0 - skew),
             (x1 + skew, y1 - skew), (x0 + skew, y1 - skew)]

    pts_f = np.array(front, dtype=int)
    pts_b = np.array(back, dtype=int)
    cv2.polylines(frame, [pts_f], True, color, 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts_b], True, color, 1, cv2.LINE_AA)
    for a, b in zip(front, back):
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, cv2.LINE_AA)


# ===========================================================================
# STEP 4 — MAIN LOOP
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
    # If the webcam is slow/laggy on Windows, try this instead:
    # cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    particles = ParticleSystem()
    last_t = time.time()
    fps_smooth = 0.0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror, feels more natural
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - last_t
            last_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0)

            # MediaPipe expects RGB, OpenCV gives us BGR -> convert.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(now * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:

                    draw_hand_skeleton(frame, hand_landmarks, w, h)

                    pts = landmarks_to_pixels(hand_landmarks, w, h)
                    depth_scale = hand_depth_scale(pts)

                    x0, y0, x1, y1 = hand_bbox(pts, w, h)
                    draw_pseudo_cube(frame, x0, y0, x1, y1, depth_scale)

                    for tip_idx in FINGERTIPS:
                        tx, ty, _ = pts[tip_idx]
                        particles.spawn(
                            tx, ty,
                            n=PARTICLES_PER_FRAME_PER_HAND // len(FINGERTIPS),
                            depth_scale=depth_scale,
                        )

            particles.update(dt)
            particles.draw(frame)

            cv2.putText(frame, f"FPS: {fps_smooth:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Hand AR Effect (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()