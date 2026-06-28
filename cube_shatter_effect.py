"""
cube_shatter_effect.py  –  Holographic Hand Shatter Effect  (v2 — Performance & 3D Upgrade)
═══════════════════════════════════════════════════════════════════════════════════════════
Real-time AR cube shatter effect controlled by hand gestures.

  - Closed fist  → a real 3D holographic cube charges up, appears, and floats/rotates in hand
  - Open hand    → cube disintegrates into thousands of tiny glowing holographic particles
  - Close again  → particles orbit, slow down, and reassemble the cube layer by layer

Gesture detection uses finger-joint angle analysis + temporal confirmation
so the cube will NOT randomly trigger from tracking noise.

──────────────────────────────────────────────────────────────────────────────
 WHAT CHANGED IN v2  (architecture notes, see ARCHITECTURE.md-style summary below)
──────────────────────────────────────────────────────────────────────────────
This version keeps the original idea (MediaPipe hand tracking → gesture state
machine → cube / particle effect) but rebuilds the rendering + particle systems
for real-time performance and a genuine 3D look:

PERFORMANCE
  • MediaPipe now runs on a downscaled copy of the frame (MP_DETECT_SCALE) —
    landmark detection cost scales with pixel count, so detecting at ~half
    resolution and re-projecting coordinates back to full-res is a large,
    accuracy-preserving win.
  • Particles are stored as **structure-of-arrays NumPy buffers** (positions,
    velocities, colors, sizes, alphas, life, speed-class, ...) instead of a
    list of Python `Fragment` objects. All physics (gravity, drag, rotation,
    fades) is updated with a handful of vectorized NumPy ops instead of a
    Python for-loop touching one object at a time.
  • Particle **object pooling**: every hand owns one fixed-size pool
    (PARTICLE_POOL_SIZE slots) allocated once. Explosions/rebuilds just
    reset/activate slots — nothing is re-instantiated per explosion.
  • Particle rendering no longer calls `cv2.circle` once per particle.
    Instead, alive particle screen-positions are scatter-written directly
    into a small local ROI buffer (sized to the explosion's bounding
    spread, not the full frame) using vectorized NumPy indexing, then a
    *single* Gaussian blur over that small ROI produces the bloom — this
    replaces thousands of antialiased `cv2.circle` calls and a full-frame
    blur with a handful of cheap array ops.
  • Per-frame `frame.copy()` / `np.zeros_like(frame)` allocations (one full
    1280×720×3 buffer per hand per effect layer in v1) are gone. Layers are
    pre-allocated once and cleared in-place; ROI buffers are small.
  • Cube geometry (vertices, faces, projected screen points) is computed
    once per frame from a cached rotation, not rebuilt from scratch with
    redundant trig.
  • Update (physics/state) and render (drawing) are now cleanly separated
    methods on each system: `update(dt, ...)` then `draw(frame)`.

3D HOLOGRAM
  • The cube is now a true 3D mesh: 8 vertices in 3D object space, rotated
    with a real 3×3 rotation matrix (continuously animating spin), then
    perspective-projected to screen space every frame — no more fixed
    isometric skew.
  • Per-face lighting is derived from each face's normal vector dotted with
    a light direction, giving real shading/depth instead of flat fill
    colors.
  • Faces are alpha-blended (semi-transparent "glass") and back-faces are
    rendered first so the cube reads as a translucent volumetric object
    you can partially see through, with edges glowing brighter than faces.
  • A subtle floating bob + slow idle rotation runs even at rest, so the
    cube never looks like a static 2D picture.
  • Animated scan-line / energy-grid pattern is drawn across each face
    using a cheap per-row alpha sine wave (no extra geometry).

PARTICLE / SHATTER UPGRADE
  • Explosion density raised substantially (configurable pool size).
  • Each particle has independently randomised size, brightness, alpha,
    rotation, and a speed-class (fast / normal / slow) for varied movement.
  • Realistic-feeling physics: gravity + air resistance + angular damping,
    all vectorized.
  • Cinematic, multi-phase reassembly: ORBIT (particles swirl toward the
    hand on converging spiral paths) → CONVERGE (smoothstep pull-in) →
    LAYERED BUILD (cube fades in face-by-face with a charging pulse) →
    STABILIZE (brief hologram flicker-settle) → INTACT.

Requirements:
    pip install opencv-python mediapipe numpy

Keyboard controls:
    q  =  quit
    r  =  reset all effects
    d  =  toggle debug / perf overlay

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
# Landmark detection runs on a downscaled copy of the frame for speed.
# Coordinates are re-projected to full resolution afterwards, so visual
# accuracy is effectively unchanged but detector cost drops a lot
# (cost scales roughly with pixel count).
MP_DETECT_SCALE      = 0.6

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

# ── Cube (3D) ─────────────────────────────────────────────────────────────────
CUBE_HALF        = 90          # half-edge in pixels (scales with hand size)
CHARGE_SECS      = 0.9         # duration of "charging" animation before cube appears
# Idle motion — keeps the hologram alive even when nothing is happening
CUBE_SPIN_SPEED  = 0.45        # rad/sec idle auto-rotation around Y axis
CUBE_TILT_SPEED  = 0.28        # rad/sec slow wobble around X axis
CUBE_TILT_AMOUNT = 0.18        # radians, amplitude of the X-axis wobble
CUBE_FLOAT_AMP   = 6.0         # px, vertical bobbing amplitude
CUBE_FLOAT_SPEED = 1.1         # rad/sec bobbing speed
CUBE_FOV         = 620.0       # perspective focal length (bigger = less fisheye)
# Lighting
LIGHT_DIR        = np.array([0.35, -0.55, -0.75])      # normalized below
LIGHT_DIR        = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
LIGHT_AMBIENT    = 0.35         # minimum lit fraction even on unlit faces
FACE_ALPHA       = 0.42        # base translucency of cube faces ("glass" look)
EDGE_GLOW_BOOST  = 1.0          # multiplier for edge brightness vs face brightness

# Reassembly (cinematic, multi-phase)
ORBIT_SECS       = 0.65        # swirling orbit phase duration
CONVERGE_SECS    = 0.45        # final pull-to-center phase duration
LAYER_BUILD_SECS = 0.6         # cube fades in face-by-face
STABILIZE_SECS   = 0.35        # brief flicker/settle after full rebuild
ORBIT_RADIUS_MUL = 2.6         # orbit radius relative to cube_s

# ── Particle pool / shatter ──────────────────────────────────────────────────
# All particles for one hand live in a fixed-size pool allocated once
# (object pooling). FRAG_DIVS / FRAG_PER_CELL / FRAG_EXTRA only control how
# many of the pool's slots get *activated* per explosion, not how much
# memory gets allocated — that happens once at startup.
FRAG_DIVS         = 9           # NxN subdivision grid per cube face (spawn seeding)
FRAG_PER_CELL     = 2           # particles seeded per grid cell
FRAG_EXTRA        = 220         # extra free-floating sparkle particles
PARTICLE_POOL_SIZE = 2200       # fixed pool size per hand (object pooling)

GRAVITY          = 0.10         # px/frame² gentle downward drift
AIR_RESISTANCE   = 0.95         # velocity multiplier per frame
ROT_DECAY        = 0.96         # angular velocity multiplier per frame
EXPLODE_MIN_V    = 3.0
EXPLODE_MAX_V    = 15.0
FLOAT_SECS       = 1.5          # explode → floating transition time

PARTICLE_MIN_PX   = 1.0
PARTICLE_MAX_PX   = 3.2
PARTICLE_ALPHA_LO = 0.35
PARTICLE_ALPHA_HI = 0.95
SLOW_FRACTION     = 0.35
FAST_FRACTION     = 0.25

# ── Particles (ambient) ───────────────────────────────────────────────────────
SPARK_COUNT    = 90           # ambient sparkles around hand (reduced for perf)
SHOCKWAVE_SECS = 0.45         # duration of explosion shockwave ring

# ── Colors  (BGR) ────────────────────────────────────────────────────────────
CLR_CUBE_BASE   = (235, 225, 215)   # base hologram tint (warm silver-white)
CLR_GLOW        = (180, 240, 255)   # cyan glow
CLR_CHARGE      = (100, 200, 255)   # orange-white charge
CLR_SHOCKWAVE   = ( 80, 200, 255)   # shockwave ring
CLR_SKELETON    = (  0, 180,  80)   # hand skeleton green
CLR_JOINT       = (  0, 120, 255)   # joint dots

PARTICLE_PALETTE = np.array([
    (255, 255, 255),
    (255, 250, 225),
    (255, 235, 180),
    (255, 215, 140),
    (245, 245, 255),
    (255, 200, 120),
], dtype=np.float32)

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


def draw_skeleton(frame, pts_px, skeleton_layer):
    """
    Render a semi-transparent hand skeleton overlay.

    Performance note: instead of `frame.copy()` + `addWeighted` (a full
    frame-sized allocation + blend per hand per frame, as in v1), the
    caller passes in one pre-allocated `skeleton_layer` buffer (sized once,
    cleared in-place) that gets reused across hands and frames.
    """
    for a, b in HAND_CONNECTIONS:
        cv2.line(skeleton_layer, pts_px[a], pts_px[b], CLR_SKELETON, 1, cv2.LINE_AA)
    for p in pts_px:
        cv2.circle(skeleton_layer, p, 3, CLR_JOINT, -1, cv2.LINE_AA)

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
# ── 3D HOLOGRAPHIC CUBE ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# This replaces the old flat isometric-skew drawing with a genuine 3D mesh:
# 8 object-space vertices are rotated with a real rotation matrix (animated
# every frame), perspective-projected to screen space, and each face is
# shaded from its normal vs. a fixed light direction. Faces are drawn
# back-to-front with alpha blending so the cube reads as a translucent
# "glass hologram" instead of a flat opaque drawing.

# Unit cube corners in object space (±1), indices match the old layout
# for readability: 0-3 = "front-ish" ring, 4-7 = "back-ish" ring.
_CUBE_OBJ_VERTS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],   # near face (z = -1)
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],   # far face  (z = +1)
], dtype=np.float64)

# Faces as (vertex indices in winding order) — normals computed once.
_CUBE_FACE_IDX = [
    (0, 1, 2, 3),   # near   (-Z)
    (5, 4, 7, 6),   # far    (+Z)
    (4, 0, 3, 7),   # left   (-X)
    (1, 5, 6, 2),   # right  (+X)
    (4, 5, 1, 0),   # bottom (-Y)
    (3, 2, 6, 7),   # top    (+Y)
]
# Static object-space face normals (axis-aligned cube → trivial to precompute)
_CUBE_FACE_NORMALS_OBJ = np.array([
    [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
], dtype=np.float64)

# Cube edges (vertex index pairs) for the glowing wireframe overlay
_CUBE_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


def rotation_matrix(rx, ry, rz):
    """Build a combined XYZ rotation matrix (cheap, called once per frame per cube)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


class HoloCube:
    """
    A real 3D holographic cube: rotates, floats, has per-face lighting,
    glowing edges, transparent "glass" faces, and an animated energy-grid
    pattern. Geometry is recomputed once per frame (cheap: 8 verts, 6 faces)
    — no per-frame Python object churn beyond a few small NumPy arrays.
    """

    def __init__(self):
        self.ry = 0.0     # current spin angle (accumulated)
        self.t_alive = 0.0

    def project(self, cx, cy, s, t, extra_tilt=0.0):
        """
        Compute this frame's projected 2D vertices, per-face screen
        polygons, per-face brightness, per-face depth (for sorting), and
        per-face/edge alpha — everything the draw step needs, bundled so
        it's only computed once per frame even though draw() may reuse it
        for faces + edges + glow.

        Returns a dict with: verts2d, faces (sorted back-to-front: list of
        (poly_pts, color, alpha)), edges (list of (p0, p1, alpha)).
        """
        # Idle animation: continuous spin + gentle wobble + float bob.
        self.ry = (CUBE_SPIN_SPEED * t) % (2 * math.pi)
        rx = CUBE_TILT_AMOUNT * math.sin(t * CUBE_TILT_SPEED) + extra_tilt
        rz = 0.05 * math.sin(t * 0.7)
        R = rotation_matrix(rx, self.ry, rz)

        bob = math.sin(t * CUBE_FLOAT_SPEED) * CUBE_FLOAT_AMP

        # Rotate object-space verts, then perspective-project.
        verts_cam = _CUBE_OBJ_VERTS @ R.T            # (8,3)
        # Push the cube "into" the screen a bit so perspective divide is stable.
        z = verts_cam[:, 2] * s * 0.9 + (CUBE_FOV)
        scale = CUBE_FOV / np.clip(z, 1.0, None)
        sx = cx + verts_cam[:, 0] * s * scale
        sy = (cy + bob) + verts_cam[:, 1] * s * scale
        verts2d = np.stack([sx, sy], axis=1)

        # Rotate face normals the same way, for lighting + back-face order.
        normals_cam = _CUBE_FACE_NORMALS_OBJ @ R.T    # (6,3)

        # Depth of each face = mean Z of its 4 verts (camera space) — used
        # to draw back-to-front (painter's algorithm) for correct alpha
        # blending of a translucent object.
        face_depths = np.array([
            verts_cam[list(idx), 2].mean() for idx in _CUBE_FACE_IDX
        ])
        # Painter's algorithm: draw faces back-to-front by camera-space
        # depth so alpha-blended faces composite correctly (farthest face
        # first, nearest face last, on top).
        order = np.argsort(-face_depths)

        faces = []
        for fi in order:
            idx = _CUBE_FACE_IDX[fi]
            poly = verts2d[list(idx)]
            # Lighting: dot of face normal with light dir → brightness.
            ndotl = float(np.dot(normals_cam[fi], -LIGHT_DIR))
            lit = LIGHT_AMBIENT + (1.0 - LIGHT_AMBIENT) * max(0.0, ndotl)
            color = tuple(min(255, int(c * lit)) for c in CLR_CUBE_BASE)
            # Faces angled toward the viewer get slightly more opacity so
            # the silhouette reads clearly; grazing faces are more see-through.
            face_alpha = FACE_ALPHA * (0.55 + 0.45 * max(0.0, ndotl))
            faces.append((poly.astype(np.int32), color, face_alpha, lit))

        edges = []
        for a, b in _CUBE_EDGES:
            edges.append((verts2d[a].astype(int), verts2d[b].astype(int)))

        return {
            "verts2d": verts2d,
            "faces": faces,
            "edges": edges,
            "center": (cx, cy + bob),
        }

    def draw(self, frame, cx, cy, s, t, alpha=1.0, glow_strength=0.7,
              extra_tilt=0.0, build_progress=1.0):
        """
        Render the holographic cube.

        Parameters
        ----------
        alpha          : overall opacity multiplier (0–1)
        glow_strength  : intensity of the cyan bloom/edge-glow (0–1)
        build_progress : 0–1, used during the LAYERED BUILD reassembly
                         phase so faces "fill in" one by one instead of
                         all fading in together (see ShatterSystem).
        """
        if alpha <= 0.01 or s <= 1:
            return
        geo = self.project(cx, cy, s, t, extra_tilt=extra_tilt)

        h, w = frame.shape[:2]
        n_faces = len(geo["faces"])
        for i, (poly, color, face_alpha, lit) in enumerate(geo["faces"]):
            # Layer-by-layer build: faces appear progressively as
            # build_progress advances (used during cinematic reassembly).
            face_reveal = np.clip(build_progress * n_faces - i, 0.0, 1.0)
            if face_reveal <= 0.01:
                continue
            fa = face_alpha * alpha * face_reveal
            fc = tuple(int(c) for c in color)

            # Performance: blend only the face's small bounding-box ROI
            # instead of copying/blending the entire frame per face (the
            # cube has just 6 faces, but a full-frame copy+blend per face
            # was the dominant cost in cube rendering). Painter's-algorithm
            # back-to-front ordering is preserved since faces are still
            # composited one at a time in depth order.
            x0 = max(0, int(poly[:, 0].min()) - 2)
            y0 = max(0, int(poly[:, 1].min()) - 2)
            x1 = min(w, int(poly[:, 0].max()) + 3)
            y1 = min(h, int(poly[:, 1].max()) + 3)
            if x1 <= x0 or y1 <= y0:
                continue

            roi = frame[y0:y1, x0:x1]
            local_poly = poly - [x0, y0]
            face_layer = roi.copy()
            cv2.fillPoly(face_layer, [local_poly], fc, lineType=cv2.LINE_AA)
            cv2.addWeighted(face_layer, fa, roi, 1 - fa, 0, roi)

        # Animated energy-grid scanlines across the silhouette (cheap:
        # a handful of horizontal lines whose alpha pulses with time).
        if glow_strength > 0.05:
            self._draw_energy_grid(frame, geo, t, alpha * glow_strength)

        # Glowing wireframe edges (brighter than faces → reads as the
        # "frame" of the hologram).
        glow_col_base = CLR_GLOW
        for p0, p1 in geo["edges"]:
            col = tuple(min(255, int(c * glow_strength * EDGE_GLOW_BOOST * alpha))
                         for c in glow_col_base)
            cv2.line(frame, tuple(p0), tuple(p1), col, 2, cv2.LINE_AA)

        # Corner glow dots (sharper highlight at vertices)
        for vx in geo["verts2d"]:
            r = max(2, int(s * 0.05))
            col = tuple(min(255, int(c * alpha)) for c in CLR_GLOW)
            cv2.circle(frame, (int(vx[0]), int(vx[1])), r, col, -1, cv2.LINE_AA)

    def _draw_energy_grid(self, frame, geo, t, strength):
        """
        Cheap animated "data lines" across the cube's front-most face:
        a few horizontal scanlines that sweep vertically over time. This
        reads as a holographic energy pattern without any extra geometry
        or per-pixel cost — just a handful of `cv2.line` calls.
        """
        # Use the face that is currently most "front facing" (last in the
        # painter's-algorithm order = nearest to camera).
        poly = geo["faces"][-1][0]
        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
        x1, y1 = poly[:, 0].max(), poly[:, 1].max()
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        n_lines = 5
        sweep = (t * 0.6) % 1.0
        for i in range(n_lines):
            frac = (i / n_lines + sweep) % 1.0
            yy = int(y0 + frac * (y1 - y0))
            line_alpha = strength * (0.25 + 0.55 * (1 - abs(frac - 0.5) * 2))
            col = tuple(min(255, int(c * line_alpha)) for c in CLR_GLOW)
            cv2.line(frame, (int(x0), yy), (int(x1), yy), col, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# ── PARTICLE POOL  (object pooling + structure-of-arrays, vectorized) ────────
# ═══════════════════════════════════════════════════════════════════════════════

class ParticlePool:
    """
    Fixed-size pool of holographic dust particles for one hand, stored as
    parallel NumPy arrays (structure-of-arrays) instead of a list of
    Python objects.

    This is the core performance upgrade for the shatter effect:
      • Allocated ONCE (object pooling) — explosions/rebuilds just
        activate/reset a slice of the arrays, no per-explosion allocation.
      • Physics update (gravity, drag, rotation decay) is a few vectorized
        NumPy expressions over the whole "alive" slice, not a Python loop.
      • Rendering scatters all alive particles into a small ROI buffer in
        one batched pass instead of one `cv2.circle` call per particle.

    Array layout (all shape (POOL_SIZE,) unless noted):
      pos        (N,2) float32   world position
      vel        (N,2) float32   velocity (px/frame)
      anchor     (N,2) float32   local cube-space anchor (±1ish), used both
                                  as the pre-explosion position (relative to
                                  cube center) and as the outward launch
                                  direction hint
      rot, rot_spd        float32   per-particle spin (visual glint only)
      radius     (N,) float32    base render radius (px)
      alpha0     (N,) float32    base opacity
      brightness (N,) float32    per-particle brightness multiplier
      color      (N,3) float32   BGR base color
      speed_class(N,) int8       0=normal 1=fast 2=slow
      alive      (N,) bool       whether this slot is currently active
      orbit_phase(N,) float32    random phase offset used during the ORBIT
                                  reassembly phase so particles swirl with
                                  varied timing instead of marching in sync
      orbit_radius_mul (N,) float32  per-particle orbit radius variance
    """

    def __init__(self, size, seed=0):
        self.size = size
        rng = np.random.default_rng(seed)
        self.rng = rng

        self.pos        = np.zeros((size, 2), dtype=np.float32)
        self.vel         = np.zeros((size, 2), dtype=np.float32)
        self.anchor      = np.zeros((size, 2), dtype=np.float32)
        self.rot         = np.zeros(size, dtype=np.float32)
        self.rot_spd     = np.zeros(size, dtype=np.float32)
        self.radius      = rng.uniform(PARTICLE_MIN_PX, PARTICLE_MAX_PX, size).astype(np.float32)
        self.alpha0      = rng.uniform(PARTICLE_ALPHA_LO, PARTICLE_ALPHA_HI, size).astype(np.float32)
        self.brightness  = rng.uniform(0.6, 1.6, size).astype(np.float32)
        pal_idx          = rng.integers(0, len(PARTICLE_PALETTE), size)
        jitter           = rng.integers(-12, 12, (size, 3)).astype(np.float32)
        self.color       = np.clip(PARTICLE_PALETTE[pal_idx] + jitter, 0, 255).astype(np.float32)
        # speed class: 0 normal, 1 fast, 2 slow
        roll = rng.uniform(size=size)
        speed_class = np.zeros(size, dtype=np.int8)
        speed_class[roll < FAST_FRACTION] = 1
        speed_class[(roll >= FAST_FRACTION) & (roll < FAST_FRACTION + SLOW_FRACTION)] = 2
        self.speed_class = speed_class

        self.alive       = np.zeros(size, dtype=bool)
        self.orbit_phase = rng.uniform(0, 2 * math.pi, size).astype(np.float32)
        self.orbit_radius_mul = rng.uniform(0.6, 1.3, size).astype(np.float32)

        # How many of the pool's slots get used for the *current* cube
        # (depends on FRAG_DIVS/FRAG_PER_CELL/FRAG_EXTRA). Pre-compute the
        # anchors for those slots once at construction (cube shape doesn't
        # change), the remaining pool slots simply stay unused/inactive.
        self.active_count = self._seed_anchors()

        # Snapshot buffers reused across PULLING/ORBIT phases (avoid
        # per-call allocation).
        self._frozen_pos = np.zeros((size, 2), dtype=np.float32)
        self._frozen_rot = np.zeros(size, dtype=np.float32)
        self._orbit_anchor = np.zeros((size, 2), dtype=np.float32)

    def _seed_anchors(self):
        """
        Fill `self.anchor` for the first N pool slots with positions
        sampled across the cube's surface (high-res grid) plus extra
        scattered volume particles — mirrors the v1 density logic but
        writes directly into pre-allocated arrays instead of building a
        Python list of objects.
        """
        rng = self.rng
        anchors = []

        sk = 0.0  # no isometric skew needed in pure local (pre-3D) anchor space;
        # anchors are only used as an outward-direction *hint* + initial
        # position relative to cube center, not as exact 3D coordinates.
        D = FRAG_DIVS
        for _face in range(6):
            for gi in range(D):
                for gj in range(D):
                    u0, u1 = gi / D, (gi + 1) / D
                    v0, v1 = gj / D, (gj + 1) / D
                    for _ in range(FRAG_PER_CELL):
                        uu = rng.uniform(u0, u1) * 2 - 1
                        vv = rng.uniform(v0, v1) * 2 - 1
                        anchors.append((uu, vv))

        for _ in range(FRAG_EXTRA):
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.0, 1.15)
            anchors.append((math.cos(ang) * r, math.sin(ang) * r))

        n = min(len(anchors), self.size)
        arr = np.array(anchors[:n], dtype=np.float32)
        self.anchor[:n] = arr
        return n

    # ── Explosion ──────────────────────────────────────────────────────
    def explode(self, origin, cube_s):
        """
        Activate all seeded slots and launch them outward from `origin`.
        Fully vectorized — no per-particle Python loop.
        """
        n = self.active_count
        rng = self.rng
        self.alive[:n] = True
        self.pos[:n] = origin

        anchor = self.anchor[:n]
        anchor_norm = np.linalg.norm(anchor, axis=1, keepdims=True)
        anchor_norm = np.clip(anchor_norm, 1e-5, None)
        anchor_dir = anchor / anchor_norm

        rand_ang = rng.uniform(0, 2 * math.pi, n)
        rand_dir = np.stack([np.cos(rand_ang), np.sin(rand_ang)], axis=1)

        blend = rng.uniform(0.35, 0.85, (n, 1))
        direction = anchor_dir * blend + rand_dir * (1 - blend)
        dn = np.linalg.norm(direction, axis=1, keepdims=True)
        dn = np.clip(dn, 1e-5, None)
        direction = direction / dn

        sc = self.speed_class[:n]
        speed = np.empty(n, dtype=np.float32)
        normal_mask = sc == 0
        fast_mask   = sc == 1
        slow_mask   = sc == 2
        speed[normal_mask] = rng.uniform(EXPLODE_MIN_V, EXPLODE_MAX_V, normal_mask.sum())
        speed[fast_mask]   = rng.uniform(EXPLODE_MAX_V * 0.7, EXPLODE_MAX_V * 1.3, fast_mask.sum())
        speed[slow_mask]   = rng.uniform(EXPLODE_MIN_V * 0.3, EXPLODE_MIN_V * 1.1, slow_mask.sum())

        vy_bias = rng.uniform(0.5, 2.5, n)
        self.vel[:n, 0] = direction[:, 0] * speed
        self.vel[:n, 1] = direction[:, 1] * speed - vy_bias

        self.rot[:n] = rng.uniform(0, 2 * math.pi, n)
        self.rot_spd[:n] = rng.uniform(-0.22, 0.22, n)
        # Re-roll orbit phase each explosion for varied reassembly motion.
        self.orbit_phase[:n] = rng.uniform(0, 2 * math.pi, n)

    # ── Physics phases (all vectorized) ──────────────────────────────────
    def step_explode(self):
        """EXPLODING phase physics: gravity + drag, slow-class extra damping."""
        n = self.active_count
        self.vel[:n, 1] += GRAVITY
        self.vel[:n] *= AIR_RESISTANCE
        slow_mask = self.speed_class[:n] == 2
        self.vel[:n][slow_mask] *= 0.995
        self.pos[:n] += self.vel[:n]
        self.rot[:n] += self.rot_spd[:n]
        self.rot_spd[:n] *= ROT_DECAY

    def step_float(self):
        """FLOATING phase physics: gentle drift, almost no gravity."""
        n = self.active_count
        self.vel[:n] *= 0.985
        self.pos[:n] += self.vel[:n] * 0.30
        self.rot[:n] += self.rot_spd[:n] * 0.30
        self.rot_spd[:n] *= 0.98

    def freeze(self):
        """Snapshot current positions/rotations (used before ORBIT/PULL phases)."""
        n = self.active_count
        self._frozen_pos[:n] = self.pos[:n]
        self._frozen_rot[:n] = self.rot[:n]

    def step_orbit(self, hand_xy, ease, t):
        """
        ORBIT phase: particles spiral inward around the hand position
        instead of moving in a straight line — gives the "particles orbit
        around the hand" cinematic feel before final convergence.

        `ease` 0→1 drives both the orbit radius shrink and the blend from
        the frozen explosion position toward the orbit path, fully
        vectorized over all alive particles.
        """
        n = self.active_count
        hx, hy = hand_xy
        phase = self.orbit_phase[:n] + t * 3.2
        radius = (1.0 - ease) * np.linalg.norm(self._frozen_pos[:n] - np.array([hx, hy]), axis=1)
        radius = radius * 0.6 + 14.0 * self.orbit_radius_mul[:n] * (1.0 - ease * 0.5)
        target_x = hx + np.cos(phase) * radius
        target_y = hy + np.sin(phase) * radius * 0.6  # flatten orbit a bit (elliptical)

        blend = ease
        self.pos[:n, 0] = self._frozen_pos[:n, 0] * (1 - blend) + target_x * blend
        self.pos[:n, 1] = self._frozen_pos[:n, 1] * (1 - blend) + target_y * blend
        self.rot[:n] += self.rot_spd[:n] * 0.5

    def step_converge(self, hand_xy, ease):
        """CONVERGE phase: smoothstep pull directly into the hand/cube center."""
        n = self.active_count
        hp = np.array(hand_xy, dtype=np.float32)
        self.pos[:n] = self._orbit_anchor[:n] * (1 - ease) + hp * ease
        self.rot[:n] = self._frozen_rot[:n] * (1 - ease)

    def snapshot_orbit_anchor(self):
        """Store current positions as the start point for the CONVERGE phase."""
        n = self.active_count
        self._orbit_anchor[:n] = self.pos[:n]

    def deactivate_all(self):
        self.alive[:] = False

# ═══════════════════════════════════════════════════════════════════════════════
# ── SHATTER SYSTEM  (state machine: drives HoloCube + ParticlePool) ─────────
# ═══════════════════════════════════════════════════════════════════════════════

# Effect states — ORBIT/CONVERGE/STABILIZE replace the old single PULLING
# state with a multi-phase cinematic reassembly.
(INTACT, CHARGING, EXPLODING, FLOATING,
 ORBIT, CONVERGE, BUILDING, STABILIZE) = range(8)

_STATE_NAMES = {
    INTACT:    "INTACT",
    CHARGING:  "CHARGING",
    EXPLODING: "EXPLODING",
    FLOATING:  "FLOATING",
    ORBIT:     "ORBIT",
    CONVERGE:  "CONVERGE",
    BUILDING:  "BUILDING",
    STABILIZE: "STABILIZE",
}


class ShatterSystem:
    """
    Manages the full lifecycle of the cube effect for one detected hand:
    a real 3D HoloCube plus a pooled particle system, driven by a small
    state machine.

    States
    ------
    INTACT     →  solid holographic cube floats/rotates in the closed fist
    CHARGING   →  hand opens; charging animation plays
    EXPLODING  →  particles fly outward with gravity/drag
    FLOATING   →  particles drift gently after the main explosion
    ORBIT      →  fist closes; particles swirl around the hand (cinematic)
    CONVERGE   →  particles pull smoothly into the center
    BUILDING   →  cube fades in face-by-face (layered) with charging pulse
    STABILIZE  →  brief hologram flicker-settle before returning to INTACT
    """

    def __init__(self, seed=7):
        self.state    = INTACT
        self.t0       = 0.0
        self.hx = self.hy = 0.0
        self.cube_s   = float(CUBE_HALF)
        self.ex = self.ey = 0.0
        self.shockwave_t0 = -999.0

        self.cube = HoloCube()
        self.pool = ParticlePool(PARTICLE_POOL_SIZE, seed=seed)

    # ── Trigger helpers ───────────────────────────────────────────────────
    def _begin_explosion(self, now):
        self.state = EXPLODING
        self.t0 = now
        self.pool.explode((self.ex, self.ey), self.cube_s)
        self.shockwave_t0 = now

    # ── State machine update ──────────────────────────────────────────────
    def update(self, hx, hy, gesture, openness, now, cube_s, t):
        """
        Called every frame with the current hand state.

        Parameters
        ----------
        hx, hy   : palm centre in pixels
        gesture  : "open" | "closed" | "neutral"  (confirmed gesture)
        openness : float 0–1  (smoothed, for visual interpolation)
        now      : current time (seconds, wall clock — used for durations)
        cube_s   : pixel half-size of cube (from palm_size)
        t        : continuous animation clock (seconds since start — used
                   for idle spin/float/grid animation so it never resets)
        """
        self.hx, self.hy = hx, hy
        self.cube_s = cube_s
        self.t_anim = t

        # ── State transitions ──────────────────────────────────────────
        if gesture == "open" and self.state in (INTACT, CHARGING, BUILDING, STABILIZE):
            self.ex, self.ey = hx, hy
            self._begin_explosion(now)

        elif gesture == "closed" and self.state in (EXPLODING, FLOATING):
            self.state = ORBIT
            self.t0 = now
            self.pool.freeze()

        # ── Physics update per state ────────────────────────────────────
        if self.state == EXPLODING:
            self.pool.step_explode()
            if now - self.t0 > FLOAT_SECS:
                self.state = FLOATING
                self.pool.freeze()

        elif self.state == FLOATING:
            self.pool.step_float()

        elif self.state == ORBIT:
            dt = now - self.t0
            ease = min(dt / ORBIT_SECS, 1.0)
            ease_s = ease * ease * (3 - 2 * ease)
            self.pool.step_orbit((hx, hy), ease_s, t)
            if ease >= 1.0:
                self.pool.snapshot_orbit_anchor()
                self.state = CONVERGE
                self.t0 = now

        elif self.state == CONVERGE:
            dt = now - self.t0
            ease = min(dt / CONVERGE_SECS, 1.0)
            ease_s = ease * ease * (3 - 2 * ease)
            self.pool.step_converge((hx, hy), ease_s)
            if ease >= 1.0:
                self.state = BUILDING
                self.t0 = now
                self.pool.deactivate_all()

        elif self.state == BUILDING:
            if now - self.t0 > LAYER_BUILD_SECS:
                self.state = STABILIZE
                self.t0 = now

        elif self.state == STABILIZE:
            if now - self.t0 > STABILIZE_SECS:
                self.state = INTACT

    # ── Draw ─────────────────────────────────────────────────────────────
    def draw(self, frame, now, particle_renderer):
        """
        Render the cube/particle effect in its current state onto `frame`.

        particle_renderer : a shared `ParticleRenderer` instance (one per
        program run, not per hand) that batches the scatter+bloom drawing
        — passed in so its small ROI scratch buffers are reused across
        hands/frames instead of being allocated here.
        """
        t = self.t_anim

        if self.state == INTACT:
            self.cube.draw(frame, self.hx, self.hy, self.cube_s, t,
                            alpha=1.0, glow_strength=0.7)

        elif self.state == CHARGING:
            dt = now - self.t0
            alpha = min(dt / CHARGE_SECS, 1.0)
            self.cube.draw(frame, self.hx, self.hy, self.cube_s, t,
                            alpha=alpha * 0.4, glow_strength=alpha)

        elif self.state == BUILDING:
            dt = now - self.t0
            progress = min(dt / LAYER_BUILD_SECS, 1.0)
            eased = progress * progress * (3 - 2 * progress)
            self.cube.draw(frame, self.hx, self.hy, self.cube_s, t,
                            alpha=eased, glow_strength=1.0,
                            build_progress=eased)
            # particles still gently converge visually under the cube fade
            particle_renderer.draw(frame, self.pool, self.ex, self.ey,
                                    self.cube_s, fade_out=1.0 - eased)

        elif self.state == STABILIZE:
            dt = now - self.t0
            flicker = 0.85 + 0.15 * math.sin(dt * 40.0) * (1.0 - dt / STABILIZE_SECS)
            self.cube.draw(frame, self.hx, self.hy, self.cube_s, t,
                            alpha=1.0, glow_strength=flicker)

        else:
            # EXPLODING / FLOATING / ORBIT / CONVERGE → draw particles
            particle_renderer.draw(frame, self.pool, self.ex, self.ey, self.cube_s)

            if self.state in (ORBIT, CONVERGE):
                dt = now - self.t0
                dur = ORBIT_SECS if self.state == ORBIT else CONVERGE_SECS
                ghost = min(dt / dur, 1.0)
                base = 0.15 if self.state == ORBIT else 0.15 + 0.4 * ghost
                self.cube.draw(frame, self.hx, self.hy, self.cube_s, t,
                                alpha=base, glow_strength=0.5 + 0.5 * ghost)

        self._draw_shockwave(frame, now)

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
        cv2.circle(frame, (int(self.ex), int(self.ey)), radius, col_sw, thickness, cv2.LINE_AA)


class ParticleRenderer:
    """
    Shared, reusable renderer for ParticlePool instances.

    Performance design: instead of allocating a full-frame buffer and
    calling `cv2.circle` per particle (v1 behaviour), this renderer:
      1. Computes a small ROI (region of interest) around the explosion
         origin, sized to the current particle spread — NOT the full
         frame.
      2. Vectorizes all alive particles' screen coordinates + colors +
         alphas with NumPy, clips them to the ROI, and scatter-writes
         them directly into the ROI's pixel buffer using fancy indexing
         (no per-particle OpenCV call at all for the base dots).
      3. Runs a single Gaussian blur over the small ROI (cheap, since the
         ROI is much smaller than the full frame) to get the bloom, then
         additively composites the ROI back onto the frame.

    One instance is shared across all hands/frames; its scratch buffers
    are reused (allocated lazily, resized only when the ROI grows).
    """

    def __init__(self):
        self._roi_buf = None
        self._roi_cap = (0, 0)   # allocated capacity (>= any requested size so far)

    def _get_roi_buffer(self, h, w):
        """
        Return a (h, w, 3) float32 scratch buffer, cleared and ready to
        use. The underlying allocation is only grown (never shrunk) and
        only the requested (h, w) sub-region is cleared each call — this
        avoids both per-frame reallocation AND avoids paying to zero out
        a buffer bigger than what's actually needed this frame.
        """
        cap_h, cap_w = self._roi_cap
        if self._roi_buf is None or h > cap_h or w > cap_w:
            new_h, new_w = max(h, cap_h), max(w, cap_w)
            self._roi_buf = np.zeros((new_h, new_w, 3), dtype=np.float32)
            self._roi_cap = (new_h, new_w)
        else:
            self._roi_buf[:h, :w] = 0.0
        return self._roi_buf[:h, :w]

    def draw(self, frame, pool, ex, ey, cube_s, fade_out=None):
        """
        Draw all alive particles in `pool` onto `frame`.

        fade_out : optional 0–1 extra fade multiplier (used while the cube
                   is fading back in during BUILDING, so leftover particles
                   dim out instead of popping off instantly).
        """
        n = pool.active_count
        alive = pool.alive[:n]
        if not np.any(alive):
            return

        h, w = frame.shape[:2]
        pos = pool.pos[:n][alive]

        # ── Compute a tight ROI around the alive particles ──────────────
        margin = 24
        pmin = pos.min(axis=0)
        pmax = pos.max(axis=0)
        x0 = max(0, int(pmin[0]) - margin)
        y0 = max(0, int(pmin[1]) - margin)
        x1 = min(w, int(pmax[0]) + margin)
        y1 = min(h, int(pmax[1]) + margin)
        if x1 <= x0 or y1 <= y0:
            return
        roi_w, roi_h = x1 - x0, y1 - y0

        roi = self._get_roi_buffer(roi_h, roi_w)

        # ── Vectorized per-particle visual params (all float32 to avoid
        # implicit float64 upcasts / extra astype churn) ─────────────────
        ox = np.float32(x0); oy = np.float32(y0)
        local_x = pos[:, 0] - ox
        local_y = pos[:, 1] - oy
        radius = pool.radius[:n][alive]
        alpha0 = pool.alpha0[:n][alive]
        brightness = pool.brightness[:n][alive]
        color = pool.color[:n][alive]
        rot = pool.rot[:n][alive]

        dx = pos[:, 0] - np.float32(ex)
        dy = pos[:, 1] - np.float32(ey)
        dist = np.sqrt(dx * dx + dy * dy)
        inv_range = np.float32(1.0 / (cube_s * 6.5 + 1e-5))
        dist_fade = 1.0 - dist * inv_range
        np.clip(dist_fade, 0.05, 1.0, out=dist_fade)
        size_wobble = 0.85 + 0.3 * np.sin(rot)

        total_fade = dist_fade * brightness
        if fade_out is not None:
            total_fade = total_fade * np.float32(fade_out)
        alpha = alpha0 * total_fade
        np.clip(alpha, 0.0, 1.0, out=alpha)

        keep = alpha > 0.02
        if not np.any(keep):
            return
        local_x = local_x[keep]
        local_y = local_y[keep]
        radius = radius[keep] * size_wobble[keep]
        color = color[keep]
        total_fade = total_fade[keep]

        # Final per-particle BGR weighted by fade — computed once, vectorized.
        weighted_color = color * total_fade[:, None]

        # ── Scatter-write particles into the ROI ────────────────────────
        # Particles are tiny (1-3px), so we approximate each as a filled
        # square via vectorized index splatting rather than calling
        # cv2.circle per particle. This is the key cost reduction: one
        # NumPy fancy-index write for potentially thousands of points
        # instead of thousands of Python-level OpenCV calls.
        ix = local_x.astype(np.int32)
        iy = local_y.astype(np.int32)
        np.clip(ix, 0, roi_w - 1, out=ix)
        np.clip(iy, 0, roi_h - 1, out=iy)

        # Base 1px scatter (covers all particles cheaply)
        roi[iy, ix] = np.maximum(roi[iy, ix], weighted_color)

        # For particles whose radius rounds to >1px, thicken with a few
        # cheap offset writes (still fully vectorized, no Python loop over
        # particles — only over a tiny fixed set of neighbour offsets).
        # Coordinates are clamped once up front; since the offset is only
        # ±1px and the ROI has a >=24px margin around all particles, the
        # offset can never leave the buffer, so no per-offset clip is
        # needed (saves 4 extra clip calls per frame).
        w_m1, h_m1 = roi_w - 1, roi_h - 1
        big = radius >= 1.5
        if np.any(big):
            bx, by = ix[big], iy[big]
            bcol = weighted_color[big] * np.float32(0.85)
            bx_p = np.minimum(bx + 1, w_m1); bx_m = np.maximum(bx - 1, 0)
            by_p = np.minimum(by + 1, h_m1); by_m = np.maximum(by - 1, 0)
            roi[by, bx_p] = np.maximum(roi[by, bx_p], bcol)
            roi[by, bx_m] = np.maximum(roi[by, bx_m], bcol)
            roi[by_p, bx] = np.maximum(roi[by_p, bx], bcol)
            roi[by_m, bx] = np.maximum(roi[by_m, bx], bcol)

        # Bright tiny core highlight (extra emphasis pass, still vectorized)
        core_big = radius >= 2.2
        if np.any(core_big):
            cx_, cy_ = ix[core_big], iy[core_big]
            roi[cy_, cx_] = np.minimum(255.0, roi[cy_, cx_] + weighted_color[core_big] * 0.6)

        # ── Bloom: one cheap blur over the small ROI only ───────────────
        # The downscale factor adapts to the ROI's size: a small ROI (a
        # tight cluster of particles) only downscales a little (keeps
        # detail), while a large ROI (particles spread across much of the
        # frame after a big explosion) downscales more aggressively so
        # the blur cost stays roughly constant either way. Bloom is
        # inherently low-frequency, so this loses no visible quality.
        roi_max_dim = max(roi_w, roi_h)
        if roi_max_dim > 480:
            ds = 4
        elif roi_max_dim > 220:
            ds = 3
        else:
            ds = 2
        sigma = max(1.0, cube_s * 0.08)
        small_w, small_h = max(1, roi_w // ds), max(1, roi_h // ds)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        small_blur = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.6, sigma * 0.5 / (ds / 2)))
        bloom = cv2.resize(small_blur, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        combined = roi + bloom * 0.9
        np.clip(combined, 0, 255, out=combined)

        # ── Composite ROI back onto the frame ───────────────────────────
        frame_roi = frame[y0:y1, x0:x1]
        cv2.add(frame_roi, combined.astype(np.uint8), dst=frame_roi)

# ═══════════════════════════════════════════════════════════════════════════════
# ── EFFECTS  –  AMBIENT PARTICLES ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class AmbientSparkles:
    """
    Floating sparkle halo around the hand.
    Uses pre-computed angular positions that oscillate over time
    for a cheap but convincing energy-field effect.

    Performance: draws into a caller-provided shared layer buffer (cleared
    in-place once per frame) instead of allocating `np.zeros_like(frame)`
    per hand per frame.
    """

    def __init__(self, n=SPARK_COUNT, seed=42):
        rng       = np.random.default_rng(seed)
        self.ang  = rng.uniform(0, math.pi * 2, n)
        self.reach = rng.uniform(0.05, 1.0, n)
        self.sz   = rng.uniform(1.0, 3.5, n)
        self.spd  = rng.uniform(0.5, 2.0, n)   # individual oscillation speeds
        pal = [
            (255, 255, 255), (230, 255, 240),
            (200, 255, 180), (180, 210, 255),
            (255, 230, 200),
        ]
        self.col = np.array(pal)[rng.integers(0, len(pal), n)]

    def draw(self, layer, cx, cy, spread, t):
        if spread < 5:
            return
        h, w  = layer.shape[:2]

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


class ChargeEffect:
    """
    Radial energy-gathering animation shown while the cube is forming.
    Draws directly onto the frame (lines are cheap; no separate layer
    buffer needed since there's no blur/blend step for this effect).
    """

    def __init__(self, n=20, seed=9):
        rng        = np.random.default_rng(seed)
        self.angs  = rng.uniform(0, math.pi * 2, n)
        self.dists = rng.uniform(0.5, 1.5, n)
        self.spds  = rng.uniform(0.8, 2.2, n)

    def draw(self, frame, cx, cy, spread, t, intensity):
        if intensity < 0.02 or spread < 5:
            return
        h, w  = frame.shape[:2]
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
            cv2.line(frame, (sx, sy), (ex_, ey_), col, 1, cv2.LINE_AA)

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

def draw_hud(frame, fps, debug, hand_data, hud_layer):
    """
    Draw the heads-up display (FPS, key hints, optional per-hand debug
    rows). Uses a pre-allocated `hud_layer` for the semi-transparent bar
    instead of `frame.copy()` every frame.
    """
    h, w = frame.shape[:2]

    hud_layer[:36, :] = 0
    cv2.rectangle(hud_layer, (0, 0), (w, 36), (255, 255, 255), -1)
    cv2.addWeighted(hud_layer[:36], 0.45, frame[:36], 0.55, 0, frame[:36])

    cv2.putText(frame, f"FPS {fps:.0f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 160), 1, cv2.LINE_AA)

    hints = "  q=quit   r=reset   d=debug"
    cv2.putText(frame, hints, (w - 310, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    if not debug or not hand_data:
        return

    for idx, hd in enumerate(hand_data):
        y = 60 + idx * 28
        txt = (f"Hand {idx+1}  gest={hd['gesture']:<7}"
               f"  conf={hd['confidence']:.2f}"
               f"  open={hd['openness']:.2f}"
               f"  [{hd['state']}]")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        y0c, y1c = max(0, y - th - 4), min(h, y + 6)
        x1c = min(w, 14 + tw)
        region = frame[y0c:y1c, 6:x1c]
        black = np.zeros_like(region)
        cv2.addWeighted(black, 0.45, region, 0.55, 0, region)
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 255, 220), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN LOOP ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point.
    Sets up the camera, MediaPipe hand-landmark detector, and runs the
    per-frame update/render loop until the user presses 'q'.

    Performance notes specific to this loop:
      • MediaPipe detection runs on a downscaled frame (MP_DETECT_SCALE);
        results are re-projected to full-res coordinates.
      • Shared scratch buffers (skeleton layer, sparkle layer, HUD layer,
        ParticleRenderer ROI buffer) are allocated once outside the loop
        and cleared in-place, never reallocated per frame.
      • update() (physics/state) and draw() (rendering) are kept as
        separate calls per hand, per the "separate update from render"
        requirement — this also makes it easy to, e.g., update all hands
        before drawing any of them if that's ever needed.
    """
    ensure_model()

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    # Smaller internal buffer reduces latency (avoids the driver queuing
    # up stale frames when our loop briefly falls behind the camera).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    systems    = {}   # idx → ShatterSystem
    gestures   = {}   # idx → GestureRecogniser

    sparkles        = AmbientSparkles()
    charge_fx       = ChargeEffect()
    particle_renderer = ParticleRenderer()   # shared across all hands
    fps_ctr         = FPSCounter()
    t0              = time.time()

    debug_mode = False

    # ── Pre-allocated shared scratch buffers (object-pooling for layers) ──
    skeleton_layer = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    sparkle_layer  = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    hud_layer      = np.zeros((36, CAMERA_WIDTH, 3), dtype=np.uint8)

    print("Hand Shatter Effect v2  –  q=quit  r=reset  d=debug")

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Empty camera frame, retrying …")
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now    = time.time()
            t      = now - t0

            # ── Hand detection on a DOWNSCALED copy (perf) ─────────────────
            det_w = max(1, int(fw * MP_DETECT_SCALE))
            det_h = max(1, int(fh * MP_DETECT_SCALE))
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = lmk.detect_for_video(mp_img, int(now * 1000))

            hand_data_hud = []

            # Reset reusable overlay layers in-place (no reallocation).
            skeleton_layer.fill(0)
            sparkle_layer.fill(0)
            any_skeleton = False
            any_sparkle = False

            if res.hand_landmarks:
                active_ids = set()

                for idx, hand_lm in enumerate(res.hand_landmarks):
                    # Landmarks were detected on the downscaled image; since
                    # MediaPipe landmarks are normalized (0-1), converting to
                    # pixels with the FULL resolution re-projects them back
                    # automatically — no extra math needed.
                    pts = lm_to_px(hand_lm, fw, fh)
                    pts_px_int = [(int(p[0]), int(p[1])) for p in pts]
                    active_ids.add(idx)

                    if idx not in systems:
                        systems[idx]  = ShatterSystem(seed=idx * 31 + 7)
                        gestures[idx] = GestureRecogniser()

                    gr   = gestures[idx]
                    sys_ = systems[idx]

                    gesture, confidence = gr.update(pts)

                    hx, hy  = palm_centre(pts)
                    ps      = palm_size(pts)
                    cube_s  = np.clip(ps * 0.85, 50, 160)

                    # ── UPDATE (physics/state) — separated from render ────
                    sys_.update(hx, hy, gesture, gr.openness, now, cube_s, t)

                    # ── RENDER layers (back to front) ─────────────────────
                    draw_skeleton(frame, pts_px_int, skeleton_layer)
                    any_skeleton = True

                    sp_spread = cube_s * (0.55 + 0.55 * gr.openness)
                    sparkles.draw(sparkle_layer, hx, hy, sp_spread, t)
                    any_sparkle = True

                    if sys_.state in (INTACT, BUILDING, CHARGING, STABILIZE):
                        charge_intensity = (
                            0.6 if sys_.state == INTACT else
                            min((now - sys_.t0) / LAYER_BUILD_SECS, 1.0) if sys_.state == BUILDING
                            else 0.5
                        )
                        charge_fx.draw(frame, hx, hy, cube_s * 1.4, t, charge_intensity)

                    sys_.draw(frame, now, particle_renderer)

                    hand_data_hud.append({
                        "gesture":    gesture,
                        "confidence": confidence,
                        "openness":   gr.openness,
                        "state":      _STATE_NAMES[sys_.state],
                    })

                for gone in set(systems.keys()) - active_ids:
                    del systems[gone]
                    del gestures[gone]
            else:
                systems.clear()
                gestures.clear()

            # Composite the skeleton/sparkle overlay layers in one shot
            # (single blend each, instead of per-hand frame.copy()+blend).
            if any_skeleton:
                cv2.addWeighted(skeleton_layer, 0.6, frame, 1.0, 0, frame)
            if any_sparkle:
                cv2.add(frame, sparkle_layer, dst=frame)

            fps_ctr.tick()
            draw_hud(frame, fps_ctr.fps, debug_mode, hand_data_hud, hud_layer)

            cv2.imshow("Hand Shatter v2  [q=quit  r=reset  d=debug]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                systems.clear()
                gestures.clear()
                print("[INFO] Effect reset.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug mode {'ON' if debug_mode else 'OFF'}.")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()

## Run with:
## C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe cube_shatter_effect.py