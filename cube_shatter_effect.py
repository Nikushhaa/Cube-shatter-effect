"""
cube_shatter_effect.py  –  Holographic Hand Shatter Effect
═══════════════════════════════════════════════════════════
Real-time AR cube shatter effect controlled by hand gestures.

  - Closed fist  → futuristic holographic cube "charges up" and appears in hand
  - Open hand    → cube explodes into hundreds of glowing fragments
  - Close again  → fragments arc back and reassemble the cube

Gesture detection uses finger-joint angle analysis + temporal confirmation
so the cube will NOT randomly trigger from tracking noise.

Requirements:
    pip install opencv-python mediapipe numpy
    python cube_shatter_effect.py

Keyboard controls:
    q  =  quit
    r  =  reset all effects
    d  =  toggle debug overlay

Author: adapted & rewritten from Nikusha Nakashidze's original concept
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ── IMPORTS ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import time
import math
import urllib.request
import collections

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═══════════════════════════════════════════════════════════════════════════════
# ── CONFIG  (tweak everything here, never use magic numbers below) ────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 60          # requested FPS (camera may deliver less)

# ── MediaPipe ────────────────────────────────────────────────────────────────
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.65
TRACKING_CONFIDENCE  = 0.65

# ── Gesture thresholds ───────────────────────────────────────────────────────
# A finger is "extended" when its tip-to-palm angle exceeds this (degrees)
FINGER_EXTEND_ANGLE  = 35.0
# Minimum fraction of fingers extended to count as "open hand"
OPEN_FINGER_RATIO    = 0.75
# Confidence must exceed this to trigger a state change (0–1)
GESTURE_CONFIDENCE   = 0.85
# Number of consecutive frames a gesture must hold before triggering
CONFIRM_FRAMES       = 6
# Exponential-smoothing factor for openness readout (lower = smoother)
SMOOTHING            = 0.12

# ── Cube ─────────────────────────────────────────────────────────────────────
CUBE_HALF     = 90           # half-edge in pixels  (scales with hand size)
CUBE_SKEW     = 0.38         # isometric skew factor
CHARGE_SECS   = 0.9          # duration of "charging" animation before cube appears
BUILD_SECS    = 0.55         # duration of rebuild animation

# ── Fragments ────────────────────────────────────────────────────────────────
FRAG_DIVS      = 4           # NxN subdivisions per face  (4→ 6×16 = 96 face frags)
FRAG_EXTRA     = 24          # extra edge-sliver fragments
GRAVITY        = 0.45        # pixels / frame²  downward acceleration
AIR_RESISTANCE = 0.93        # velocity multiplier per frame
ROT_DECAY      = 0.96        # rotational speed multiplier per frame
EXPLODE_MIN_V  = 10.0        # min explosion speed (px/frame)
EXPLODE_MAX_V  = 28.0        # max explosion speed (px/frame)
FLOAT_SECS     = 1.8         # how long fragments drift before entering FLOATING state
PULL_SECS      = 0.55        # duration of "pull back" animation

# ── Particles ────────────────────────────────────────────────────────────────
SPARK_COUNT    = 140         # ambient sparkles around hand
TRAIL_COUNT    = 60          # fragment motion-trail dots
SHOCKWAVE_SECS = 0.45        # duration of explosion shockwave ring

# ── Colors  (BGR) ────────────────────────────────────────────────────────────
CLR_CUBE_FRONT  = (230, 220, 210)   # warm white-silver front face
CLR_CUBE_TOP    = (255, 240, 200)   # slightly golden top
CLR_CUBE_LEFT   = (200, 195, 185)   # darker sides
CLR_GLOW        = (180, 240, 255)   # cyan glow
CLR_CHARGE      = (100, 200, 255)   # orange-white charge
CLR_SHOCKWAVE   = ( 80, 200, 255)   # shockwave ring
CLR_SKELETON    = (  0, 180,  80)   # hand skeleton green
CLR_JOINT       = (  0, 120, 255)   # joint dots

# ── MediaPipe landmark model ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── Landmark indices ─────────────────────────────────────────────────────────
WRIST           = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP           = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP           = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP      = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP              = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP         = 17, 18, 19, 20

PALM_REF    = (0, 5, 9, 13, 17)         # landmarks that define the palm centre
FINGERTIPS  = (4, 8, 12, 16, 20)        # one tip per finger

# Finger chains: (mcp, pip, dip, tip)  — thumb uses a slightly different chain
FINGER_CHAINS = [
    (THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP),
    (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
    (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── MODEL DOWNLOAD ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    """Download the MediaPipe hand-landmark model if it's not already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-landmark model …")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done:", MODEL_PATH)
    except Exception as exc:
        print(f"[ERROR] Could not download model: {exc}", file=sys.stderr)
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# ── HAND TRACKING UTILITIES ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def lm_to_px(lms, w, h):
    """Convert normalised MediaPipe landmarks to pixel coordinates (x, y, z)."""
    return [(lm.x * w, lm.y * h, lm.z) for lm in lms]


def palm_centre(pts):
    """Return the (x, y) centroid of the palm reference landmarks."""
    return np.mean([pts[i][:2] for i in PALM_REF], axis=0)


def palm_size(pts):
    """
    Return an approximate hand scale in pixels:
    distance from wrist to middle-finger MCP.
    Used to normalise cube size and gesture thresholds.
    """
    wrist  = np.array(pts[WRIST][:2])
    mid_mc = np.array(pts[MIDDLE_MCP][:2])
    return float(np.linalg.norm(mid_mc - wrist)) + 1e-6


def finger_angle(pts, mcp_i, pip_i, tip_i):
    """
    Compute the bend angle (degrees) at the PIP joint of a finger.
    Larger angle  →  finger more extended.
    0°  =  fully curled.
    """
    mcp = np.array(pts[mcp_i][:2])
    pip = np.array(pts[pip_i][:2])
    tip = np.array(pts[tip_i][:2])
    v1  = mcp - pip
    v2  = tip - pip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def fingers_extended(pts):
    """
    Return a bool array [thumb, index, middle, ring, pinky] –
    True when the finger is visibly extended using joint-angle analysis.
    """
    extended = []
    for mcp_i, pip_i, _dip_i, tip_i in FINGER_CHAINS:
        ang = finger_angle(pts, mcp_i, pip_i, tip_i)
        extended.append(ang > FINGER_EXTEND_ANGLE)
    return extended


def draw_skeleton(frame, lms, w, h):
    """Render a semi-transparent hand skeleton overlay."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    overlay = frame.copy()
    for a, b in HAND_CONNECTIONS:
        cv2.line(overlay, pts[a], pts[b], CLR_SKELETON, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(overlay, p, 3, CLR_JOINT, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

# ═══════════════════════════════════════════════════════════════════════════════
# ── GESTURE RECOGNISER ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class GestureRecogniser:
    """
    Robust gesture classification with confidence scoring and
    frame-based temporal confirmation to prevent false triggers.

    Attributes
    ----------
    gesture   : "open" | "closed" | "neutral"
    confidence: float 0–1
    """

    def __init__(self):
        self._history  = collections.deque(maxlen=CONFIRM_FRAMES * 2)
        self.gesture   = "neutral"
        self.confidence = 0.0
        # smoothed openness ratio (0 = fist, 1 = open)
        self._smooth   = 0.0
        # counters for frame-confirmation
        self._open_cnt  = 0
        self._close_cnt = 0

    def update(self, pts):
        """
        Feed one frame's worth of pixel landmarks.
        Returns (gesture, confidence) tuple.
        """
        ext    = fingers_extended(pts)
        n_ext  = sum(ext[1:])  # exclude thumb for open/close count
        ratio  = n_ext / 4.0  # 4 non-thumb fingers

        # Exponential smoothing of the open ratio
        self._smooth += (ratio - self._smooth) * SMOOTHING

        # Per-frame raw label
        raw = "open" if ratio >= OPEN_FINGER_RATIO else "closed"
        self._history.append(raw)

        # Count consecutive matching frames
        if raw == "open":
            self._open_cnt  = min(self._open_cnt + 1, CONFIRM_FRAMES * 2)
            self._close_cnt = max(self._close_cnt - 1, 0)
        else:
            self._close_cnt = min(self._close_cnt + 1, CONFIRM_FRAMES * 2)
            self._open_cnt  = max(self._open_cnt - 1, 0)

        # Confidence: fraction of recent history that matches raw
        recent = list(self._history)[-CONFIRM_FRAMES:]
        match  = sum(1 for g in recent if g == raw)
        self.confidence = match / max(len(recent), 1)

        # Only commit a new gesture if confidence threshold is met
        if self.confidence >= GESTURE_CONFIDENCE:
            if raw == "open"   and self._open_cnt  >= CONFIRM_FRAMES:
                self.gesture = "open"
            elif raw == "closed" and self._close_cnt >= CONFIRM_FRAMES:
                self.gesture = "closed"
        # else keep current gesture

        return self.gesture, self.confidence

    @property
    def openness(self):
        """Smoothed openness value (0 = fist, 1 = fully open)."""
        return self._smooth

# ═══════════════════════════════════════════════════════════════════════════════
# ── CUBE GEOMETRY ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def cube_verts(cx, cy, s, skew=CUBE_SKEW):
    """
    Return 8 vertices of an isometric-perspective cube centred at (cx, cy).
    Vertices 0-3 = front face, 4-7 = back face.
    """
    sk = s * skew
    front = np.array([
        [cx - s, cy - s],
        [cx + s, cy - s],
        [cx + s, cy + s],
        [cx - s, cy + s],
    ], dtype=float)
    back = front + [sk, -sk]
    return np.vstack([front, back])


# Each face: (vertex_indices, base_BGR_color)
CUBE_FACES = [
    ([0, 1, 2, 3], CLR_CUBE_FRONT),
    ([4, 5, 6, 7], tuple(int(c * 0.55) for c in CLR_CUBE_FRONT)),   # back (darker)
    ([0, 1, 5, 4], CLR_CUBE_TOP),
    ([3, 2, 6, 7], tuple(int(c * 0.50) for c in CLR_CUBE_TOP)),     # bottom
    ([0, 3, 7, 4], CLR_CUBE_LEFT),
    ([1, 2, 6, 5], tuple(int(c * 0.65) for c in CLR_CUBE_LEFT)),    # right
]


def draw_cube(frame, cx, cy, s, alpha=1.0, glow_strength=0.0):
    """
    Draw the intact holographic cube at screen position (cx, cy) with
    optional glow and alpha transparency.

    Parameters
    ----------
    alpha         : 0–1  overall opacity
    glow_strength : 0–1  extra cyan glow halo intensity
    """
    if alpha <= 0.01:
        return

    v   = cube_verts(cx, cy, s)
    ov  = frame.copy()

    # Optional glow halo (drawn behind the solid cube)
    if glow_strength > 0.01:
        glow = frame.copy()
        for vidx, col in CUBE_FACES:
            pts = np.array([v[i] for i in vidx], dtype=np.int32)
            gc  = tuple(min(255, int(CLR_GLOW[k] * glow_strength)) for k in range(3))
            cv2.fillPoly(glow, [pts], gc)
        # blur the glow layer for a soft bloom effect
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=s * 0.15)
        cv2.addWeighted(glow, 0.55 * glow_strength, ov, 1.0, 0, ov)

    # Solid cube faces
    for vidx, col in CUBE_FACES:
        pts = np.array([v[i] for i in vidx], dtype=np.int32)
        fc  = tuple(int(c * alpha) for c in col)
        cv2.fillPoly(ov, [pts], fc)
        cv2.polylines(ov, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # Edge highlight lines (brighten top edges for depth illusion)
    edge_pairs = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for a, b in edge_pairs:
        cv2.line(ov, v[a].astype(int), v[b].astype(int),
                 tuple(min(255, int(CLR_GLOW[k] * 0.9 * alpha)) for k in range(3)),
                 2, cv2.LINE_AA)

    # Corner dot accents
    for vx in v:
        r = max(3, int(s * 0.065))
        cv2.circle(ov, (int(vx[0]), int(vx[1])), r,
                   tuple(min(255, int(c * alpha)) for c in CLR_GLOW), -1, cv2.LINE_AA)

    blend = min(0.90, 0.45 + 0.45 * alpha)
    cv2.addWeighted(ov, blend, frame, 1 - blend, 0, frame)

# ═══════════════════════════════════════════════════════════════════════════════
# ── PHYSICS  –  FRAGMENT ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class Fragment:
    """
    One shard of the shattered cube.

    Attributes
    ----------
    local       : (N, 2) polygon in local coordinates, centred at origin
    color       : BGR tuple
    pos         : world position (px)
    vel         : velocity (px/frame)
    rot         : rotation angle (radians)
    rot_spd     : angular velocity (radians/frame)
    scale       : pixel scale factor
    trail       : deque of recent world positions for motion-trail rendering
    frozen_pos  : position snapshot taken at start of PULLING phase
    frozen_rot  : rotation snapshot for rebuilding
    """
    __slots__ = (
        'local', 'color', 'pos', 'vel', 'rot', 'rot_spd',
        'scale', 'trail', 'frozen_pos', 'frozen_rot',
    )

    def __init__(self, local, color):
        self.local      = np.array(local, dtype=float)
        self.color      = color
        self.pos        = np.zeros(2, dtype=float)
        self.vel        = np.zeros(2, dtype=float)
        self.rot        = 0.0
        self.rot_spd    = 0.0
        self.scale      = 1.0
        self.trail      = collections.deque(maxlen=8)
        self.frozen_pos = None
        self.frozen_rot = None

    def world_pts(self):
        """Return the rotated, scaled, translated polygon vertices."""
        c = math.cos(self.rot)
        s = math.sin(self.rot)
        R = np.array([[c, -s], [s, c]])
        return (R @ (self.local * self.scale).T).T + self.pos

    def step(self):
        """Advance physics by one frame (gravity + air resistance)."""
        self.trail.append(self.pos.copy())
        self.vel[1] += GRAVITY            # downward gravity
        self.vel    *= AIR_RESISTANCE     # air drag
        self.pos    += self.vel
        self.rot    += self.rot_spd
        self.rot_spd *= ROT_DECAY

# ═══════════════════════════════════════════════════════════════════════════════
# ── SHATTER SYSTEM ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Effect states
INTACT, CHARGING, EXPLODING, FLOATING, PULLING, BUILDING = range(6)

_STATE_NAMES = {
    INTACT:    "INTACT",
    CHARGING:  "CHARGING",
    EXPLODING: "EXPLODING",
    FLOATING:  "FLOATING",
    PULLING:   "PULLING",
    BUILDING:  "BUILDING",
}


class ShatterSystem:
    """
    Manages the full lifecycle of the cube effect for one detected hand.

    States
    ------
    INTACT    →  solid holographic cube sits in the closed fist
    CHARGING  →  hand opens; charging animation plays
    EXPLODING →  fragments fly outward with gravity
    FLOATING  →  fragments drift gently after the main explosion
    PULLING   →  fist closes; fragments arc back toward palm
    BUILDING  →  fragments arrived; cube fades back in
    """

    def __init__(self, seed=7):
        self.rng      = np.random.default_rng(seed)
        self.state    = INTACT
        self.t0       = 0.0
        # current hand palm centre (updated every frame)
        self.hx = self.hy = 0.0
        # scale of cube (set from palm_size each frame)
        self.cube_s   = float(CUBE_HALF)
        # position where the explosion happened (stays fixed during drift)
        self.ex = self.ey = 0.0
        # shockwave radius (grows after explosion)
        self.shockwave_r  = 0.0
        self.shockwave_t0 = -999.0
        # pre-built fragment list
        self.frags = self._make_frags()

    # ── Fragment factory ──────────────────────────────────────────────────
    def _make_frags(self):
        """
        Subdivide each cube face into FRAG_DIVS×FRAG_DIVS sub-quads and
        generate FRAG_EXTRA edge-sliver fragments for visual richness.
        All coordinates are in normalised local space (±1 units).
        """
        rng   = self.rng
        frags = []

        # Local-space corners for each face in (u,v) parameterisation
        sk = CUBE_SKEW * 2
        face_corners = [
            # front face
            [(-1,-1), (1,-1), (1,1), (-1,1)],
            # back face (offset by skew)
            [(-1+sk,-1-sk), (1+sk,-1-sk), (1+sk,1-sk), (-1+sk,1-sk)],
            # top
            [(-1,-1), (1,-1), (1+sk,-1-sk), (-1+sk,-1-sk)],
            # bottom
            [(-1,1), (1,1), (1+sk,1-sk), (-1+sk,1-sk)],
            # left
            [(-1,-1), (-1,1), (-1+sk,1-sk), (-1+sk,-1-sk)],
            # right
            [(1,-1), (1,1), (1+sk,1-sk), (1+sk,-1-sk)],
        ]
        base_colors = [col for _, col in CUBE_FACES]

        D = FRAG_DIVS
        for fi, corners in enumerate(face_corners):
            c  = np.array(corners, dtype=float)
            bc = base_colors[fi]

            for gi in range(D):
                for gj in range(D):
                    u0, u1 = gi / D, (gi + 1) / D
                    v0, v1 = gj / D, (gj + 1) / D

                    # Bilinear interpolation of sub-quad corners
                    def blerp(u, v):
                        return (c[0] * (1-u) * (1-v) + c[1] * u * (1-v) +
                                c[3] * (1-u) * v      + c[2] * u * v)

                    pts  = np.array([blerp(u0,v0), blerp(u1,v0),
                                     blerp(u1,v1), blerp(u0,v1)])
                    ctr  = pts.mean(axis=0)
                    local = pts - ctr
                    # Tiny organic jitter so edges don't look too mechanical
                    local += rng.uniform(-0.035, 0.035, local.shape)

                    # Slight random tint per fragment
                    noise = rng.integers(-25, 25, 3)
                    color = tuple(int(np.clip(bc[k] + noise[k], 0, 255)) for k in range(3))
                    frags.append(Fragment(local, color))

        # Extra randomly oriented edge slivers for sparkling debris
        for _ in range(FRAG_EXTRA):
            ang = rng.uniform(0, math.pi * 2)
            r   = rng.uniform(0.25, 1.0)
            w2  = rng.uniform(0.04, 0.12)
            sliver = np.array([
                [0, 0],
                [math.cos(ang) * r, math.sin(ang) * r],
                [math.cos(ang + 0.25) * r + w2, math.sin(ang + 0.25) * r + w2],
                [w2, w2],
            ])
            sliver -= sliver.mean(axis=0)
            # White-cyan slivers for sparkle
            frags.append(Fragment(sliver, (220, 245, 255)))

        return frags

    # ── Trigger helpers ───────────────────────────────────────────────────
    def _launch_frags(self):
        """
        Initialise fragment velocities for the explosion.
        Fragments launch from the explosion origin with random outward
        velocities, plus some upward bias for a dramatic look.
        """
        rng = self.rng
        for frag in self.frags:
            frag.pos     = np.array([self.ex, self.ey], dtype=float)
            frag.scale   = float(self.cube_s)
            ang           = rng.uniform(0, math.pi * 2)
            spd           = rng.uniform(EXPLODE_MIN_V, EXPLODE_MAX_V)
            # Upward bias:  reduce downward components
            vy = math.sin(ang) * spd - rng.uniform(3, 10)
            frag.vel     = np.array([math.cos(ang) * spd, vy])
            frag.rot     = rng.uniform(0, math.pi * 2)
            frag.rot_spd = rng.uniform(-0.28, 0.28)
            frag.trail.clear()
            frag.frozen_pos = None
            frag.frozen_rot = None

    def _freeze_frags(self):
        """Snapshot fragment positions and rotations for PULLING interpolation."""
        for frag in self.frags:
            frag.frozen_pos = frag.pos.copy()
            frag.frozen_rot = frag.rot

    # ── State machine update ──────────────────────────────────────────────
    def update(self, hx, hy, gesture, openness, now, cube_s):
        """
        Called every frame with the current hand state.

        Parameters
        ----------
        hx, hy   : palm centre in pixels
        gesture  : "open" | "closed" | "neutral"  (confirmed gesture)
        openness : float 0–1  (smoothed, for visual interpolation)
        now      : current time (seconds)
        cube_s   : pixel half-size of cube (from palm_size)
        """
        self.hx, self.hy = hx, hy
        self.cube_s = cube_s

        # ── State transitions ──────────────────────────────────────────
        if gesture == "open" and self.state in (INTACT, CHARGING):
            # Begin explosion
            self.ex, self.ey = hx, hy
            self.state = EXPLODING
            self.t0    = now
            self._launch_frags()
            self.shockwave_r  = 0.0
            self.shockwave_t0 = now

        elif gesture == "open" and self.state == BUILDING:
            # Re-explode immediately if hand opened during rebuild
            self.ex, self.ey = hx, hy
            self.state = EXPLODING
            self.t0    = now
            self._launch_frags()
            self.shockwave_r  = 0.0
            self.shockwave_t0 = now

        elif gesture == "closed" and self.state in (EXPLODING, FLOATING):
            # Begin pulling fragments back
            self.state = PULLING
            self.t0    = now
            self._freeze_frags()

        # ── Physics update per state ────────────────────────────────────
        if self.state == EXPLODING:
            for f in self.frags:
                f.step()
            if now - self.t0 > FLOAT_SECS:
                self.state = FLOATING
                self._freeze_frags()

        elif self.state == FLOATING:
            for f in self.frags:
                # Gentle continued drift (less gravity)
                f.trail.append(f.pos.copy())
                f.vel    *= 0.97
                f.pos    += f.vel * 0.35
                f.rot    += f.rot_spd * 0.35
                f.rot_spd *= 0.98

        elif self.state == PULLING:
            dt   = now - self.t0
            ease = min(dt / PULL_SECS, 1.0)
            ease = ease * ease * (3 - 2 * ease)   # smoothstep
            hp   = np.array([hx, hy])
            for f in self.frags:
                f.pos = f.frozen_pos + (hp - f.frozen_pos) * ease
                f.rot = f.frozen_rot * (1 - ease)
            if ease >= 1.0:
                self.state = BUILDING
                self.t0    = now

        elif self.state == BUILDING:
            if now - self.t0 > BUILD_SECS:
                self.state = INTACT

    # ── Draw ─────────────────────────────────────────────────────────────
    def draw(self, frame, now):
        """Render the cube effect in its current state onto `frame`."""
        h, w = frame.shape[:2]

        if self.state == INTACT:
            draw_cube(frame, self.hx, self.hy, self.cube_s,
                      alpha=1.0, glow_strength=0.7)

        elif self.state == CHARGING:
            # Pulsing charge-up glow (used by ChargeEffect class separately)
            dt    = now - self.t0
            alpha = min(dt / CHARGE_SECS, 1.0)
            draw_cube(frame, self.hx, self.hy, self.cube_s,
                      alpha=alpha * 0.4, glow_strength=alpha)

        elif self.state == BUILDING:
            # Ghost cube fades in from translucent to solid
            dt    = now - self.t0
            alpha = min(dt / BUILD_SECS, 1.0)
            alpha = alpha * alpha * (3 - 2 * alpha)   # smoothstep
            draw_cube(frame, self.hx, self.hy, self.cube_s,
                      alpha=alpha, glow_strength=1.0 - alpha * 0.5)

        else:
            # EXPLODING / FLOATING / PULLING  →  draw fragments
            self._draw_frags(frame, w, h, now)

            # During PULLING: ghost cube grows at hand position
            if self.state == PULLING:
                dt    = now - self.t0
                ghost = min(dt / PULL_SECS, 1.0)
                ghost = ghost * ghost * (3 - 2 * ghost)
                draw_cube(frame, self.hx, self.hy, self.cube_s,
                          alpha=ghost * 0.55, glow_strength=ghost)

        # Shockwave ring after explosion
        self._draw_shockwave(frame, now)

    def _draw_frags(self, frame, w, h, now):
        """Render all fragments with motion trails and glow."""
        layer = np.zeros_like(frame)
        ex    = np.array([self.ex, self.ey])

        for f in self.frags:
            wp  = f.world_pts().astype(np.int32)
            in_f = ((wp[:, 0] >= 0) & (wp[:, 0] < w) &
                    (wp[:, 1] >= 0) & (wp[:, 1] < h)).any()
            if not in_f:
                continue

            # Fade based on distance from explosion origin
            dist = np.linalg.norm(f.pos - ex)
            fade = float(np.clip(1.0 - dist / (self.cube_s * 5.5), 0.08, 1.0))
            col  = tuple(int(c * fade) for c in f.color)

            # Motion trail  (draw older positions as fading dots)
            trail_list = list(f.trail)
            for ti, tp in enumerate(trail_list):
                tf   = (ti / max(len(trail_list), 1)) * fade * 0.55
                tpx, tpy = int(tp[0]), int(tp[1])
                if 0 <= tpx < w and 0 <= tpy < h:
                    tr = max(1, int(self.cube_s * 0.02 * tf * 3))
                    tc = tuple(int(c * tf * 0.8) for c in f.color)
                    cv2.circle(layer, (tpx, tpy), tr, tc, -1, cv2.LINE_AA)

            # Fragment polygon
            edge = wp.reshape(-1, 1, 2)
            cv2.fillPoly(layer, [edge], col)
            cv2.polylines(layer, [edge], True, (200, 230, 255), 1, cv2.LINE_AA)

            # Bright centre dot
            px, py = int(f.pos[0]), int(f.pos[1])
            if 0 <= px < w and 0 <= py < h:
                cr = max(2, int(self.cube_s * 0.03 * fade))
                cc = tuple(min(255, int(c * 1.5 * fade)) for c in f.color)
                cv2.circle(layer, (px, py), cr, cc, -1, cv2.LINE_AA)

        cv2.add(frame, layer, dst=frame)

    def _draw_shockwave(self, frame, now):
        """Expanding translucent ring emitted at the moment of explosion."""
        dt = now - self.shockwave_t0
        if dt > SHOCKWAVE_SECS or dt < 0:
            return
        progress = dt / SHOCKWAVE_SECS
        radius   = int(self.cube_s * 0.5 + self.cube_s * 3.5 * progress)
        alpha_sw  = max(0.0, 1.0 - progress)
        col_sw    = tuple(int(c * alpha_sw) for c in CLR_SHOCKWAVE)
        thickness = max(1, int(4 * (1.0 - progress)))
        overlay   = frame.copy()
        cv2.circle(overlay, (int(self.ex), int(self.ey)), radius, col_sw, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha_sw * 0.75, frame, 1.0 - alpha_sw * 0.75, 0, frame)

# ═══════════════════════════════════════════════════════════════════════════════
# ── EFFECTS  –  PARTICLES ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class AmbientSparkles:
    """
    Floating sparkle halo around the hand.
    Uses pre-computed angular positions that oscillate over time
    for a cheap but convincing energy-field effect.
    """

    def __init__(self, n=SPARK_COUNT, seed=42):
        rng       = np.random.default_rng(seed)
        self.ang  = rng.uniform(0, math.pi * 2, n)
        self.reach = rng.uniform(0.05, 1.0, n)
        self.sz   = rng.uniform(1.0, 3.5, n)
        self.spd  = rng.uniform(0.5, 2.0, n)   # individual oscillation speeds
        # Palette: white, cyan-white, green-white, soft blue
        pal = [
            (255, 255, 255), (230, 255, 240),
            (200, 255, 180), (180, 210, 255),
            (255, 230, 200),
        ]
        self.col = np.array(pal)[rng.integers(0, len(pal), n)]

    def draw(self, frame, cx, cy, spread, t):
        if spread < 5:
            return
        h, w  = frame.shape[:2]
        layer = np.zeros_like(frame)

        # Oscillate each sparkle at its own speed
        jx = np.sin(t * self.spd * 2.8 + self.reach * 28) * 6
        jy = np.cos(t * self.spd * 2.3 + self.reach * 15) * 6
        x  = cx + np.cos(self.ang) * self.reach * spread + jx
        y  = cy + np.sin(self.ang) * self.reach * spread + jy

        for i in range(len(x)):
            px, py = int(x[i]), int(y[i])
            if not (0 <= px < w and 0 <= py < h):
                continue
            br  = 1.0 - 0.6 * self.reach[i]
            col = tuple(int(c * br) for c in self.col[i])
            cv2.circle(layer, (px, py), max(1, int(self.sz[i])), col, -1, cv2.LINE_AA)

        cv2.add(frame, layer, dst=frame)


class ChargeEffect:
    """
    Radial energy-gathering animation shown while the cube is forming
    (state == INTACT and system hasn't exploded yet, or during BUILDING).
    Shows energy lines converging toward the palm centre.
    """

    def __init__(self, n=20, seed=9):
        rng        = np.random.default_rng(seed)
        self.angs  = rng.uniform(0, math.pi * 2, n)
        self.dists = rng.uniform(0.5, 1.5, n)
        self.spds  = rng.uniform(0.8, 2.2, n)

    def draw(self, frame, cx, cy, spread, t, intensity):
        """
        intensity: 0–1  (tied to cube glow strength / state)
        """
        if intensity < 0.02 or spread < 5:
            return
        h, w  = frame.shape[:2]
        layer = np.zeros_like(frame)
        for i, ang in enumerate(self.angs):
            phase  = (t * self.spds[i]) % 1.0
            d_out  = self.dists[i] * spread * (1.0 - phase)
            d_in   = max(0, d_out - spread * 0.18)
            sx     = int(cx + math.cos(ang) * d_out)
            sy     = int(cy + math.sin(ang) * d_out)
            ex_    = int(cx + math.cos(ang) * d_in)
            ey_    = int(cy + math.sin(ang) * d_in)
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            alpha_line = intensity * (0.3 + 0.7 * phase)
            col = tuple(int(CLR_CHARGE[k] * alpha_line) for k in range(3))
            cv2.line(layer, (sx, sy), (ex_, ey_), col, 1, cv2.LINE_AA)
        cv2.add(frame, layer, dst=frame)

# ═══════════════════════════════════════════════════════════════════════════════
# ── FPS COUNTER ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-window frames-per-second counter."""

    def __init__(self, window=30):
        self._times  = collections.deque(maxlen=window)
        self._last   = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

# ═══════════════════════════════════════════════════════════════════════════════
# ── OVERLAY  –  HUD / DEBUG ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps, debug, hand_data):
    """
    Draw the heads-up display:
      - FPS counter (always shown)
      - Key hints (always shown)
      - Per-hand debug info (only in debug mode)

    Parameters
    ----------
    hand_data : list of dicts with keys: gesture, confidence, openness, state
    """
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 160), 1, cv2.LINE_AA)

    # Key hints
    hints = "  q=quit   r=reset   d=debug"
    cv2.putText(frame, hints, (w - 310, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    if not debug or not hand_data:
        return

    # Per-hand debug rows
    for idx, hd in enumerate(hand_data):
        y = 60 + idx * 28
        txt = (f"Hand {idx+1}  gest={hd['gesture']:<7}"
               f"  conf={hd['confidence']:.2f}"
               f"  open={hd['openness']:.2f}"
               f"  [{hd['state']}]")
        # Background pill
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        pill = frame.copy()
        cv2.rectangle(pill, (6, y - th - 4), (14 + tw, y + 6), (0, 0, 0), -1)
        cv2.addWeighted(pill, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 255, 220), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN LOOP ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point.
    Sets up the camera, MediaPipe hand-landmark detector, and runs
    the per-frame update/render loop until the user presses 'q'.
    """
    ensure_model()

    # ── MediaPipe hand landmarker setup ──────────────────────────────────
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    # ── Camera setup ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    # ── Per-hand state objects ────────────────────────────────────────────
    systems    = {}   # idx → ShatterSystem
    gestures   = {}   # idx → GestureRecogniser

    # ── Shared visual systems ─────────────────────────────────────────────
    sparkles   = AmbientSparkles()
    charge_fx  = ChargeEffect()
    fps_ctr    = FPSCounter()
    t0         = time.time()

    # ── Runtime flags ─────────────────────────────────────────────────────
    debug_mode = False

    print("Hand Shatter Effect  –  q=quit  r=reset  d=debug")

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Empty camera frame, retrying …")
                time.sleep(0.01)
                continue

            # Mirror the image so it feels like a mirror
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now    = time.time()
            t      = now - t0

            # ── Hand detection ────────────────────────────────────────────
            rgb    = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = lmk.detect_for_video(mp_img, int(now * 1000))

            hand_data_hud = []   # collected for HUD display

            if res.hand_landmarks:
                active_ids = set()

                for idx, hand_lm in enumerate(res.hand_landmarks):
                    pts = lm_to_px(hand_lm, fw, fh)
                    active_ids.add(idx)

                    # Initialise per-hand objects on first sight
                    if idx not in systems:
                        systems[idx]  = ShatterSystem(seed=idx * 31 + 7)
                        gestures[idx] = GestureRecogniser()

                    gr   = gestures[idx]
                    sys_ = systems[idx]

                    # Gesture recognition
                    gesture, confidence = gr.update(pts)

                    # Hand geometry
                    hx, hy  = palm_centre(pts)
                    ps      = palm_size(pts)
                    cube_s  = np.clip(ps * 0.85, 50, 160)  # scale cube with hand

                    # Update cube/fragment physics
                    sys_.update(hx, hy, gesture, gr.openness, now, cube_s)

                    # ── Render layers (back to front) ─────────────────────
                    # 1. Hand skeleton
                    draw_skeleton(frame, hand_lm, fw, fh)

                    # 2. Ambient sparkles (energy halo around palm)
                    sp_spread = cube_s * (0.55 + 0.55 * gr.openness)
                    sparkles.draw(frame, hx, hy, sp_spread, t)

                    # 3. Charge energy lines (visible in INTACT/BUILDING states)
                    if sys_.state in (INTACT, BUILDING, CHARGING):
                        charge_intensity = (
                            0.6 if sys_.state == INTACT else
                            min((now - sys_.t0) / BUILD_SECS, 1.0)
                        )
                        charge_fx.draw(frame, hx, hy,
                                       cube_s * 1.4, t, charge_intensity)

                    # 4. Cube / fragment effect (main draw)
                    sys_.draw(frame, now)

                    # Collect HUD data
                    hand_data_hud.append({
                        "gesture":    gesture,
                        "confidence": confidence,
                        "openness":   gr.openness,
                        "state":      _STATE_NAMES[sys_.state],
                    })

                # Remove trackers for hands that left the frame
                for gone in set(systems.keys()) - active_ids:
                    del systems[gone]
                    del gestures[gone]

            else:
                # No hands detected → clear all trackers
                systems.clear()
                gestures.clear()

            # ── FPS & HUD ─────────────────────────────────────────────────
            fps_ctr.tick()
            draw_hud(frame, fps_ctr.fps, debug_mode, hand_data_hud)

            # ── Show ─────────────────────────────────────────────────────
            cv2.imshow("Hand Shatter  [q=quit  r=reset  d=debug]", frame)

            # ── Keyboard input ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset: destroy all active hand systems
                systems.clear()
                gestures.clear()
                print("[INFO] Effect reset.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug mode {'ON' if debug_mode else 'OFF'}.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()

## Run with:
## C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe cube_shatter_effect.py