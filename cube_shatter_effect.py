"""
cube_shatter_effect.py  –  Holographic Hand Shatter Effect  (v3 — Cinematic VFX, cube removed)
═══════════════════════════════════════════════════════════════════════════════════
Real-time AR hand-triggered shatter effect.

  - Closed fist  → idle (nothing shown, ready)
  - Open hand    → cracks appear → energy pulse → thousands of glowing
                   crystal particles scatter in 3D space
  - Close again  → particles orbit the hand, converge into the palm, fade out

NOTE: The holographic cube object has been fully removed. Only the crack /
explosion / particle / shockwave / bloom / camera-shake cinematic effects
remain, exactly as before.

Requirements:
    pip install opencv-python mediapipe numpy

Controls:
    q  =  quit
    r  =  reset
    d  =  toggle debug / perf overlay
    s  =  toggle camera shake
    b  =  toggle bloom pulse
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os, sys, time, math, urllib.request, collections
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Camera
CAMERA_INDEX   = 0
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 60

# MediaPipe
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.65
TRACKING_CONFIDENCE  = 0.65
MP_DETECT_SCALE      = 0.6

# Gesture
FINGER_EXTEND_ANGLE = 35.0
OPEN_FINGER_RATIO   = 0.75
GESTURE_CONFIDENCE  = 0.85
CONFIRM_FRAMES      = 6
SMOOTHING           = 0.12

# Object size (drives explosion / crack / particle scale relative to hand size)
CUBE_HALF        = 90

# Cinematic reassembly timings
ORBIT_SECS       = 0.70
CONVERGE_SECS    = 0.50
LAYER_BUILD_SECS = 0.65
STABILIZE_SECS   = 0.40
ORBIT_RADIUS_MUL = 2.8

# Crack phase (pre-explosion)
CRACK_SECS   = 0.25            # how long cracks show before particles fly
N_CRACKS     = 18              # number of crack lines per explosion

# Shockwave
SHOCKWAVE_SECS    = 0.50
SHOCKWAVE_RINGS   = 3          # concentric expanding rings

# Camera shake
SHAKE_DURATION = 0.42          # seconds
SHAKE_MAGNITUDE = 11.0         # max pixels of displacement
SHAKE_DECAY     = 5.5          # exponential decay rate (higher = faster settle)

# Bloom pulse
BLOOM_PEAK_SIGMA  = 28.0       # Gaussian sigma at explosion peak
BLOOM_DECAY_SECS  =  0.5       # how long bloom pulse lasts

# Particles (3D)
FRAG_DIVS          = 9
FRAG_PER_CELL      = 2
FRAG_EXTRA         = 260
PARTICLE_POOL_SIZE = 2400

GRAVITY          = 0.09        # Y only (px/frame²)
AIR_RESISTANCE   = 0.952       # isotropic drag
ROT_DECAY        = 0.96
EXPLODE_MIN_V    = 3.5
EXPLODE_MAX_V    = 16.0
EXPLODE_Z_RANGE  = 8.0         # ±Z velocity at explosion
FLOAT_SECS       = 1.6
TURBULENCE_AMP   = 0.18        # lateral turbulence magnitude

PARTICLE_MIN_PX   = 1.0
PARTICLE_MAX_PX   = 3.8
PARTICLE_ALPHA_LO = 0.30
PARTICLE_ALPHA_HI = 0.98
SLOW_FRACTION     = 0.35
FAST_FRACTION     = 0.25

PARTICLE_Z_NEAR   = -120.0    # Z range for perspective
PARTICLE_Z_FAR    =  140.0
PERSP_FOCAL       = 460.0     # focal length for particle perspective

# Ambient sparkles
SPARK_COUNT    = 100

# Colors (BGR)
CLR_GLOW        = (200, 248, 255)
CLR_CHARGE      = ( 80, 190, 255)
CLR_SHOCKWAVE   = (100, 220, 255)
CLR_SKELETON    = (  0, 175,  75)
CLR_JOINT       = (  0, 120, 255)

PARTICLE_PALETTE = np.array([
    (255, 255, 255),
    (255, 252, 220),
    (255, 240, 180),
    (255, 218, 130),
    (245, 245, 255),
    (200, 242, 255),
    (230, 255, 240),
    (255, 200, 110),
], dtype=np.float32)

# MediaPipe model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP          = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP       = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP      = 9,10,11,12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP        =13,14,15,16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP      =17,18,19,20

PALM_REF   = (0, 5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
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
# MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_model():
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
# HAND TRACKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def lm_to_px(lms, w, h):
    return [(lm.x * w, lm.y * h, lm.z) for lm in lms]

def palm_centre(pts):
    return np.mean([pts[i][:2] for i in PALM_REF], axis=0)

def palm_size(pts):
    wrist  = np.array(pts[WRIST][:2])
    mid_mc = np.array(pts[MIDDLE_MCP][:2])
    return float(np.linalg.norm(mid_mc - wrist)) + 1e-6

def finger_angle(pts, mcp_i, pip_i, tip_i):
    mcp = np.array(pts[mcp_i][:2])
    pip = np.array(pts[pip_i][:2])
    tip = np.array(pts[tip_i][:2])
    v1  = mcp - pip; v2 = tip - pip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def fingers_extended(pts):
    return [finger_angle(pts, mcp, pip, tip) > FINGER_EXTEND_ANGLE
            for mcp, pip, _dip, tip in FINGER_CHAINS]

def draw_skeleton(frame, pts_px, skeleton_layer):
    for a, b in HAND_CONNECTIONS:
        cv2.line(skeleton_layer, pts_px[a], pts_px[b], CLR_SKELETON, 1, cv2.LINE_AA)
    for p in pts_px:
        cv2.circle(skeleton_layer, p, 3, CLR_JOINT, -1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNISER  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class GestureRecogniser:
    def __init__(self):
        self._history  = collections.deque(maxlen=CONFIRM_FRAMES * 2)
        self.gesture   = "neutral"
        self.confidence = 0.0
        self._smooth   = 0.0
        self._open_cnt  = 0
        self._close_cnt = 0
        self._prev_palm = None
        self.palm_velocity = np.zeros(2, dtype=np.float32)  # px/s estimate

    def update(self, pts, dt=0.033):
        ext   = fingers_extended(pts)
        n_ext = sum(ext[1:])
        ratio = n_ext / 4.0
        self._smooth += (ratio - self._smooth) * SMOOTHING
        raw = "open" if ratio >= OPEN_FINGER_RATIO else "closed"
        self._history.append(raw)
        if raw == "open":
            self._open_cnt  = min(self._open_cnt + 1, CONFIRM_FRAMES * 2)
            self._close_cnt = max(self._close_cnt - 1, 0)
        else:
            self._close_cnt = min(self._close_cnt + 1, CONFIRM_FRAMES * 2)
            self._open_cnt  = max(self._open_cnt - 1, 0)
        recent = list(self._history)[-CONFIRM_FRAMES:]
        match  = sum(1 for g in recent if g == raw)
        self.confidence = match / max(len(recent), 1)
        if self.confidence >= GESTURE_CONFIDENCE:
            if raw == "open"   and self._open_cnt  >= CONFIRM_FRAMES:
                self.gesture = "open"
            elif raw == "closed" and self._close_cnt >= CONFIRM_FRAMES:
                self.gesture = "closed"

        # Estimate palm velocity (still used for HUD display)
        palm = np.array(palm_centre(pts), dtype=np.float32)
        if self._prev_palm is not None and dt > 0:
            self.palm_velocity = (palm - self._prev_palm) / max(dt, 0.001)
        self._prev_palm = palm
        return self.gesture, self.confidence

    @property
    def openness(self):
        return self._smooth

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE POOL  (3D positions, depth sort, turbulence) — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

class ParticlePool:
    """
    Fixed-size pool with 3D positions (x, y, z) and perspective projection.
    Far particles render smaller; near particles render larger.
    Depth-sorted each frame for correct alpha compositing.
    """

    def __init__(self, size, seed=0):
        self.size = size
        rng       = np.random.default_rng(seed)
        self.rng  = rng

        self.pos         = np.zeros((size, 3), dtype=np.float32)  # x,y,z
        self.vel         = np.zeros((size, 3), dtype=np.float32)  # vx,vy,vz
        self.anchor      = np.zeros((size, 2), dtype=np.float32)
        self.rot         = np.zeros(size, dtype=np.float32)
        self.rot_spd     = np.zeros(size, dtype=np.float32)
        self.turb_phase  = rng.uniform(0, 2*math.pi, size).astype(np.float32)
        self.radius      = rng.uniform(PARTICLE_MIN_PX, PARTICLE_MAX_PX, size).astype(np.float32)
        self.alpha0      = rng.uniform(PARTICLE_ALPHA_LO, PARTICLE_ALPHA_HI, size).astype(np.float32)
        self.brightness  = rng.uniform(0.6, 1.7, size).astype(np.float32)
        pal_idx          = rng.integers(0, len(PARTICLE_PALETTE), size)
        jitter           = rng.integers(-14, 14, (size, 3)).astype(np.float32)
        self.color       = np.clip(PARTICLE_PALETTE[pal_idx] + jitter, 0, 255).astype(np.float32)
        roll = rng.uniform(size=size)
        sc   = np.zeros(size, dtype=np.int8)
        sc[roll < FAST_FRACTION] = 1
        sc[(roll >= FAST_FRACTION) & (roll < FAST_FRACTION + SLOW_FRACTION)] = 2
        self.speed_class = sc
        self.alive       = np.zeros(size, dtype=bool)
        self.orbit_phase = rng.uniform(0, 2*math.pi, size).astype(np.float32)
        self.orbit_radius_mul = rng.uniform(0.6, 1.4, size).astype(np.float32)

        self.active_count = self._seed_anchors()

        self._frozen_pos     = np.zeros((size, 3), dtype=np.float32)
        self._frozen_rot     = np.zeros(size, dtype=np.float32)
        self._orbit_anchor   = np.zeros((size, 3), dtype=np.float32)

    def _seed_anchors(self):
        rng = self.rng
        anchors = []
        D = FRAG_DIVS
        for _ in range(6):
            for gi in range(D):
                for gj in range(D):
                    for _ in range(FRAG_PER_CELL):
                        uu = rng.uniform(gi/D, (gi+1)/D) * 2 - 1
                        vv = rng.uniform(gj/D, (gj+1)/D) * 2 - 1
                        anchors.append((uu, vv))
        for _ in range(FRAG_EXTRA):
            ang = rng.uniform(0, 2*math.pi)
            r   = rng.uniform(0.0, 1.2)
            anchors.append((math.cos(ang)*r, math.sin(ang)*r))
        n   = min(len(anchors), self.size)
        arr = np.array(anchors[:n], dtype=np.float32)
        self.anchor[:n] = arr
        return n

    def explode(self, origin, cube_s):
        """Launch all active particles from origin with 3D velocity."""
        n   = self.active_count
        rng = self.rng
        self.alive[:n] = True
        # Initialize in 3D; Z starts near 0
        self.pos[:n, 0] = origin[0]
        self.pos[:n, 1] = origin[1]
        self.pos[:n, 2] = rng.uniform(-cube_s * 0.3, cube_s * 0.3, n).astype(np.float32)

        anchor = self.anchor[:n]
        adir   = anchor / np.clip(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-5, None)
        rand_ang = rng.uniform(0, 2*math.pi, n)
        rand_dir = np.stack([np.cos(rand_ang), np.sin(rand_ang)], axis=1)
        blend    = rng.uniform(0.35, 0.85, (n,1))
        direction = adir * blend + rand_dir * (1 - blend)
        dn = np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-5, None)
        direction = direction / dn

        sc    = self.speed_class[:n]
        speed = np.empty(n, dtype=np.float32)
        nm = sc == 0; fm = sc == 1; sm = sc == 2
        speed[nm] = rng.uniform(EXPLODE_MIN_V, EXPLODE_MAX_V, nm.sum())
        speed[fm] = rng.uniform(EXPLODE_MAX_V*0.7, EXPLODE_MAX_V*1.3, fm.sum())
        speed[sm] = rng.uniform(EXPLODE_MIN_V*0.3, EXPLODE_MIN_V*1.1, sm.sum())

        vy_bias = rng.uniform(0.4, 2.5, n)
        self.vel[:n, 0] = direction[:, 0] * speed
        self.vel[:n, 1] = direction[:, 1] * speed - vy_bias
        self.vel[:n, 2] = rng.uniform(-EXPLODE_Z_RANGE, EXPLODE_Z_RANGE, n).astype(np.float32)

        self.rot[:n]      = rng.uniform(0, 2*math.pi, n)
        self.rot_spd[:n]  = rng.uniform(-0.25, 0.25, n)
        self.turb_phase[:n] = rng.uniform(0, 2*math.pi, n)
        self.orbit_phase[:n] = rng.uniform(0, 2*math.pi, n)

    def step_explode(self, t):
        n = self.active_count
        # Gravity on Y only
        self.vel[:n, 1] += GRAVITY
        # Isotropic drag in 3D
        self.vel[:n] *= AIR_RESISTANCE
        sm = self.speed_class[:n] == 2
        self.vel[:n][sm] *= 0.993
        # Turbulence: lateral drift based on z-phase and time
        turb = (TURBULENCE_AMP *
                np.sin(self.turb_phase[:n] + t * 2.1).astype(np.float32))
        self.vel[:n, 0] += turb * 0.5
        self.vel[:n, 2] += turb * 0.3
        self.pos[:n] += self.vel[:n]
        self.rot[:n]     += self.rot_spd[:n]
        self.rot_spd[:n] *= ROT_DECAY

    def step_float(self, t):
        n = self.active_count
        self.vel[:n] *= 0.982
        self.pos[:n] += self.vel[:n] * 0.28
        # Gentle turbulence in float
        turb = (TURBULENCE_AMP * 0.4 *
                np.sin(self.turb_phase[:n] + t * 1.3).astype(np.float32))
        self.pos[:n, 0] += turb * 0.3
        self.rot[:n]     += self.rot_spd[:n] * 0.28
        self.rot_spd[:n] *= 0.98

    def freeze(self):
        n = self.active_count
        self._frozen_pos[:n] = self.pos[:n]
        self._frozen_rot[:n] = self.rot[:n]

    def step_orbit(self, hand_xy, ease, t):
        """3D orbital spiral: particles swirl on an ellipsoid."""
        n = self.active_count
        hx, hy = hand_xy
        phase  = self.orbit_phase[:n] + t * 3.2
        diff   = self._frozen_pos[:n, :2] - np.array([hx, hy], dtype=np.float32)
        base_r = np.linalg.norm(diff, axis=1)
        radius = base_r * (1.0-ease) * 0.6 + 16.0 * self.orbit_radius_mul[:n] * (1.0-ease*0.5)
        tx = hx + np.cos(phase) * radius
        ty = hy + np.sin(phase) * radius * 0.55
        # Z spirals back to 0 during orbit
        tz = self._frozen_pos[:n, 2] * (1.0 - ease)
        blend = ease
        self.pos[:n, 0] = self._frozen_pos[:n, 0] * (1-blend) + tx * blend
        self.pos[:n, 1] = self._frozen_pos[:n, 1] * (1-blend) + ty * blend
        self.pos[:n, 2] = self._frozen_pos[:n, 2] * (1-blend) + tz * blend
        self.rot[:n] += self.rot_spd[:n] * 0.5

    def step_converge(self, hand_xy, ease):
        n  = self.active_count
        hp = np.array([hand_xy[0], hand_xy[1], 0.0], dtype=np.float32)
        self.pos[:n] = self._orbit_anchor[:n] * (1-ease) + hp * ease
        self.rot[:n] = self._frozen_rot[:n] * (1-ease)

    def snapshot_orbit_anchor(self):
        n = self.active_count
        self._orbit_anchor[:n] = self.pos[:n]

    def deactivate_all(self):
        self.alive[:] = False

# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER STATES
# ═══════════════════════════════════════════════════════════════════════════════

(INTACT, CHARGING, CRACKING, EXPLODING, FLOATING,
 ORBIT, CONVERGE, BUILDING, STABILIZE) = range(9)

_STATE_NAMES = {
    INTACT:    "INTACT",    CHARGING:  "CHARGING",
    CRACKING:  "CRACKING",  EXPLODING: "EXPLODING",
    FLOATING:  "FLOATING",  ORBIT:     "ORBIT",
    CONVERGE:  "CONVERGE",  BUILDING:  "BUILDING",
    STABILIZE: "STABILIZE",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CRACK GENERATOR  (now centred on the hand directly — no cube geometry needed)
# ═══════════════════════════════════════════════════════════════════════════════

class CrackEffect:
    """
    Pre-generates Voronoi-like crack lines floating above the hand.
    Drawn as hairline bright segments, growing over CRACK_SECS before explosion.
    Cached per explosion (cheap: only recomputed on each detonation).
    """

    def __init__(self):
        self._lines = []    # list of ((x0,y0),(x1,y1)) in screen space

    def generate(self, cx, cy, cube_s, seed=None):
        """Build crack lines centred on (cx, cy), sized relative to cube_s."""
        rng = np.random.default_rng(seed)
        self._lines.clear()
        for _ in range(N_CRACKS):
            ang = rng.uniform(0, 2*math.pi)
            length = rng.uniform(cube_s * 0.4, cube_s * 1.6)
            sx = cx + rng.uniform(-cube_s*0.7, cube_s*0.7)
            sy = cy + rng.uniform(-cube_s*0.7, cube_s*0.7)
            ex = sx + math.cos(ang) * length
            ey = sy + math.sin(ang) * length
            self._lines.append(((int(sx), int(sy)), (int(ex), int(ey))))

    def draw(self, frame, progress):
        """Draw cracks with `progress` 0→1 controlling how many are visible."""
        if not self._lines:
            return
        n_show = max(1, int(len(self._lines) * progress))
        for i, (p0, p1) in enumerate(self._lines[:n_show]):
            frac  = i / max(len(self._lines), 1)
            alpha = 0.55 + 0.45 * (1 - frac) * progress
            col   = tuple(int(c * alpha) for c in CLR_GLOW)
            # Hairline white core + glow color border
            cv2.line(frame, p0, p1, col, 2, cv2.LINE_AA)
            cv2.line(frame, p0, p1, (255,255,255), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA SHAKE  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class CameraShake:
    """
    Applies a decaying random translation to the frame at the moment of explosion.
    Uses np.roll for a very cheap implementation (no image reallocation).
    """

    def __init__(self):
        self._t0  = -999.0
        self._dx  = 0
        self._dy  = 0

    def trigger(self, now, magnitude=SHAKE_MAGNITUDE):
        self._t0 = now
        self._dx = int(np.random.uniform(-magnitude, magnitude))
        self._dy = int(np.random.uniform(-magnitude, magnitude))

    def apply(self, frame, now):
        dt = now - self._t0
        if dt > SHAKE_DURATION:
            return frame
        decay = math.exp(-SHAKE_DECAY * dt)
        dx    = int(self._dx * decay)
        dy    = int(self._dy * decay)
        if dx == 0 and dy == 0:
            return frame
        # np.roll wraps edges; for a small shift this is barely visible
        # and vastly cheaper than cv2.warpAffine on a full frame.
        out = np.roll(frame, dy, axis=0)
        out = np.roll(out, dx, axis=1)
        return out

# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER SYSTEM  (cube fully removed — only crack/explosion/particle visuals)
# ═══════════════════════════════════════════════════════════════════════════════

class ShatterSystem:
    """
    Manages the full shatter lifecycle for one detected hand.

    States: INTACT → CRACKING → EXPLODING → FLOATING → ORBIT → CONVERGE → BUILDING → STABILIZE
    (No cube is drawn in any state — only cracks, particles, and the shockwave.)
    """

    def __init__(self, seed=7):
        self.state  = INTACT
        self.t0     = 0.0
        self.hx = self.hy = 0.0
        self.cube_s = float(CUBE_HALF)
        self.ex = self.ey = 0.0
        self.shockwave_t0 = -999.0
        self._bloom_t0    = -999.0
        self._last_draw_t = 0.0

        self.pool    = ParticlePool(PARTICLE_POOL_SIZE, seed=seed)
        self.cracks  = CrackEffect()

    def _begin_crack(self, now):
        self.state = CRACKING
        self.t0    = now
        self.cracks.generate(self.hx, self.hy, self.cube_s, seed=int(now*1000))

    def _begin_explosion(self, now):
        self.state = EXPLODING
        self.t0    = now
        self.pool.explode((self.ex, self.ey), self.cube_s)
        self.shockwave_t0 = now
        self._bloom_t0    = now

    def update(self, hx, hy, gesture, openness, now, cube_s, t, dt):
        self.hx, self.hy = hx, hy
        self.cube_s = cube_s
        self.t_anim = t

        # State transitions
        if gesture == "open" and self.state in (INTACT, CHARGING, BUILDING, STABILIZE):
            self.ex, self.ey = hx, hy
            self._begin_crack(now)

        elif gesture == "closed" and self.state in (EXPLODING, FLOATING):
            self.state = ORBIT
            self.t0    = now
            self.pool.freeze()

        # Per-state physics
        if self.state == CRACKING:
            if now - self.t0 > CRACK_SECS:
                self._begin_explosion(now)

        elif self.state == EXPLODING:
            self.pool.step_explode(t)
            if now - self.t0 > FLOAT_SECS:
                self.state = FLOATING
                self.pool.freeze()

        elif self.state == FLOATING:
            self.pool.step_float(t)

        elif self.state == ORBIT:
            dt_s = now - self.t0
            ease = min(dt_s / ORBIT_SECS, 1.0)
            ease_s = ease * ease * (3 - 2 * ease)
            self.pool.step_orbit((hx, hy), ease_s, t)
            if ease >= 1.0:
                self.pool.snapshot_orbit_anchor()
                self.state = CONVERGE
                self.t0    = now

        elif self.state == CONVERGE:
            dt_s = now - self.t0
            ease = min(dt_s / CONVERGE_SECS, 1.0)
            ease_s = ease * ease * (3 - 2 * ease)
            self.pool.step_converge((hx, hy), ease_s)
            if ease >= 1.0:
                self.state = BUILDING
                self.t0    = now
                self.pool.deactivate_all()

        elif self.state == BUILDING:
            if now - self.t0 > LAYER_BUILD_SECS:
                self.state = STABILIZE
                self.t0    = now

        elif self.state == STABILIZE:
            if now - self.t0 > STABILIZE_SECS:
                self.state = INTACT

    def draw(self, frame, now, particle_renderer, dt,
             enable_shake=True, enable_bloom=True, cam_shake=None,
             palm_velocity=None):
        t = self.t_anim

        # Bloom sigma boost at explosion moment
        bloom_boost = 0.0
        if enable_bloom:
            bd = now - self._bloom_t0
            if bd < BLOOM_DECAY_SECS:
                bloom_boost = BLOOM_PEAK_SIGMA * (1.0 - bd / BLOOM_DECAY_SECS)

        if self.state == CRACKING:
            crack_prog = min((now - self.t0) / CRACK_SECS, 1.0)
            self.cracks.draw(frame, crack_prog)

        elif self.state in (EXPLODING, FLOATING, ORBIT, CONVERGE):
            particle_renderer.draw(frame, self.pool, self.ex, self.ey,
                                    self.cube_s, bloom_boost=bloom_boost)

        elif self.state == BUILDING:
            dt_b     = now - self.t0
            progress = min(dt_b / LAYER_BUILD_SECS, 1.0)
            particle_renderer.draw(frame, self.pool, self.ex, self.ey,
                                    self.cube_s, fade_out=1.0 - progress)

        # INTACT, CHARGING, STABILIZE: nothing to draw (no cube anymore)

        self._draw_shockwave(frame, now)

    def _draw_shockwave(self, frame, now):
        dt = now - self.shockwave_t0
        if dt > SHOCKWAVE_SECS or dt < 0:
            return
        progress = dt / SHOCKWAVE_SECS
        for ring_i in range(SHOCKWAVE_RINGS):
            ring_prog = max(0.0, progress - ring_i * 0.12)
            if ring_prog <= 0 or ring_prog > 1.0:
                continue
            radius   = int(self.cube_s * 0.4 + self.cube_s * 3.8 * ring_prog)
            alpha_sw  = max(0.0, 1.0 - ring_prog) * (1.0 - ring_i * 0.28)
            col_sw    = tuple(int(c * alpha_sw) for c in CLR_SHOCKWAVE)
            thickness = max(1, int(3 * (1.0 - ring_prog)))
            cv2.circle(frame, (int(self.ex), int(self.ey)),
                       radius, col_sw, thickness, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE RENDERER  (3D perspective projection + depth sort) — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

class ParticleRenderer:
    """
    Renders ParticlePool onto a frame ROI.
    • Perspective projection: Z coord scales rendered size and brightness
    • Depth sort: far particles drawn first (painter's algorithm)
    • bloom_sigma_boost: spike at explosion moment
    """

    def __init__(self):
        self._roi_buf = None
        self._roi_cap = (0, 0)

    def _get_roi_buffer(self, h, w):
        cap_h, cap_w = self._roi_cap
        if self._roi_buf is None or h > cap_h or w > cap_w:
            new_h, new_w = max(h, cap_h), max(w, cap_w)
            self._roi_buf = np.zeros((new_h, new_w, 3), dtype=np.float32)
            self._roi_cap = (new_h, new_w)
        else:
            self._roi_buf[:h, :w] = 0.0
        return self._roi_buf[:h, :w]

    def draw(self, frame, pool, ex, ey, cube_s, fade_out=None, bloom_boost=0.0):
        n     = pool.active_count
        alive = pool.alive[:n]
        if not np.any(alive):
            return

        h, w  = frame.shape[:2]
        pos3d = pool.pos[:n][alive]          # (M,3)
        rot   = pool.rot[:n][alive]

        # ── Perspective projection of Z → scale ──────────────────────────
        z_raw    = pos3d[:, 2]
        z_clip   = np.clip(z_raw, PARTICLE_Z_NEAR, PARTICLE_Z_FAR)
        persp    = PERSP_FOCAL / (PERSP_FOCAL + z_clip)   # 1 = at origin, <1 = far
        # Screen x, y after perspective
        screen_x = pos3d[:, 0]   # X is already in screen space (no extra projection needed
        screen_y = pos3d[:, 1]   # since the cube itself is screen-space; Z just modulates size)

        # ── Depth sort (far → near, painter's) ───────────────────────────
        sort_order = np.argsort(-z_raw)   # negative = draw far first
        screen_x = screen_x[sort_order]
        screen_y = screen_y[sort_order]
        persp    = persp[sort_order]
        rot      = rot[sort_order]
        radius   = pool.radius[:n][alive][sort_order]
        alpha0   = pool.alpha0[:n][alive][sort_order]
        brightness = pool.brightness[:n][alive][sort_order]
        color    = pool.color[:n][alive][sort_order]

        # ── ROI ───────────────────────────────────────────────────────────
        margin = 28
        x0 = max(0, int(screen_x.min()) - margin)
        y0 = max(0, int(screen_y.min()) - margin)
        x1 = min(w, int(screen_x.max()) + margin)
        y1 = min(h, int(screen_y.max()) + margin)
        if x1 <= x0 or y1 <= y0:
            return
        roi_w, roi_h = x1 - x0, y1 - y0
        roi = self._get_roi_buffer(roi_h, roi_w)

        # ── Vectorised per-particle params ────────────────────────────────
        ox, oy = np.float32(x0), np.float32(y0)
        lx = screen_x - ox
        ly = screen_y - oy

        dx = screen_x - np.float32(ex)
        dy = screen_y - np.float32(ey)
        dist = np.sqrt(dx*dx + dy*dy)
        inv_range  = np.float32(1.0 / (cube_s * 6.5 + 1e-5))
        dist_fade  = np.clip(1.0 - dist * inv_range, 0.05, 1.0)
        size_wobble = 0.82 + 0.32 * np.sin(rot)

        # Z-depth affects brightness: near = brighter, far = dimmer
        z_bright = 0.65 + 0.5 * persp   # 0.65–1.15 range

        total_fade = dist_fade * brightness * z_bright
        if fade_out is not None:
            total_fade *= np.float32(fade_out)
        alpha = np.clip(alpha0 * total_fade, 0.0, 1.0)

        keep = alpha > 0.02
        if not np.any(keep):
            return
        lx     = lx[keep];    ly   = ly[keep]
        radius = radius[keep] * size_wobble[keep] * persp[keep]
        color  = color[keep]
        total_fade = total_fade[keep]

        weighted = color * total_fade[:, None]

        ix = np.clip(lx.astype(np.int32), 0, roi_w - 1)
        iy = np.clip(ly.astype(np.int32), 0, roi_h - 1)

        roi[iy, ix] = np.maximum(roi[iy, ix], weighted)

        w_m1, h_m1 = roi_w - 1, roi_h - 1
        big = radius >= 1.5
        if np.any(big):
            bx  = ix[big]; by = iy[big]; bcol = weighted[big] * np.float32(0.82)
            bxp = np.minimum(bx+1, w_m1); bxm = np.maximum(bx-1, 0)
            byp = np.minimum(by+1, h_m1); bym = np.maximum(by-1, 0)
            roi[by, bxp] = np.maximum(roi[by, bxp], bcol)
            roi[by, bxm] = np.maximum(roi[by, bxm], bcol)
            roi[byp, bx] = np.maximum(roi[byp, bx], bcol)
            roi[bym, bx] = np.maximum(roi[bym, bx], bcol)

        core_big = radius >= 2.2
        if np.any(core_big):
            cx_, cy_ = ix[core_big], iy[core_big]
            roi[cy_, cx_] = np.minimum(255.0, roi[cy_, cx_] + weighted[core_big] * 0.65)

        # ── Bloom: adaptive downscale blur on small ROI ───────────────────
        roi_max  = max(roi_w, roi_h)
        if roi_max > 480:   ds = 4
        elif roi_max > 220: ds = 3
        else:               ds = 2
        base_sigma = max(1.0, cube_s * 0.08)
        sigma  = base_sigma + bloom_boost * 0.04   # spike at explosion
        sigma  = max(0.5, sigma)
        sw, sh = max(1, roi_w // ds), max(1, roi_h // ds)
        small  = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
        small_blur = cv2.GaussianBlur(small, (0, 0),
                                       sigmaX=max(0.4, sigma * 0.5 / (ds/2)))
        bloom  = cv2.resize(small_blur, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        combined = roi + bloom * (0.9 + bloom_boost * 0.005)
        np.clip(combined, 0, 255, out=combined)
        cv2.add(frame[y0:y1, x0:x1], combined.astype(np.uint8),
                dst=frame[y0:y1, x0:x1])

# ═══════════════════════════════════════════════════════════════════════════════
# AMBIENT SPARKLES  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class AmbientSparkles:
    def __init__(self, n=SPARK_COUNT, seed=42):
        rng        = np.random.default_rng(seed)
        self.ang   = rng.uniform(0, math.pi*2, n)
        self.reach = rng.uniform(0.05, 1.0, n)
        self.sz    = rng.uniform(1.0, 3.5, n)
        self.spd   = rng.uniform(0.5, 2.0, n)
        pal = [(255,255,255),(230,255,240),(200,255,180),(180,210,255),(255,230,200)]
        self.col = np.array(pal)[rng.integers(0, len(pal), n)]

    def draw(self, layer, cx, cy, spread, t):
        if spread < 5:
            return
        h, w = layer.shape[:2]
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

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGE EFFECT  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class ChargeEffect:
    def __init__(self, n=22, seed=9):
        rng        = np.random.default_rng(seed)
        self.angs  = rng.uniform(0, math.pi*2, n)
        self.dists = rng.uniform(0.5, 1.5, n)
        self.spds  = rng.uniform(0.8, 2.2, n)

    def draw(self, frame, cx, cy, spread, t, intensity):
        if intensity < 0.02 or spread < 5:
            return
        h, w = frame.shape[:2]
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
            al = intensity * (0.3 + 0.7 * phase)
            col = tuple(int(CLR_CHARGE[k] * al) for k in range(3))
            cv2.line(frame, (sx, sy), (ex_, ey_), col, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# FPS COUNTER
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    def __init__(self, window=30):
        self._times = collections.deque(maxlen=window)
        self._last  = time.perf_counter()

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
# HUD
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps, debug, hand_data, hud_layer, shake_on, bloom_on):
    h, w = frame.shape[:2]
    hud_layer[:36, :] = 0
    cv2.rectangle(hud_layer, (0,0), (w,36), (255,255,255), -1)
    cv2.addWeighted(hud_layer[:36], 0.40, frame[:36], 0.60, 0, frame[:36])

    # FPS with color indicator
    fps_col = (100,255,100) if fps >= 45 else (0,165,255) if fps >= 25 else (0,60,255)
    cv2.putText(frame, f"FPS {fps:.0f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_col, 1, cv2.LINE_AA)

    flags = f"  shake={'ON' if shake_on else 'OFF'}  bloom={'ON' if bloom_on else 'OFF'}"
    hints = f"q=quit  r=reset  d=debug  s=shake  b=bloom{flags}"
    cv2.putText(frame, hints, (w - 560, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170,170,170), 1, cv2.LINE_AA)

    if not debug or not hand_data:
        return
    for idx, hd in enumerate(hand_data):
        y   = 60 + idx * 28
        txt = (f"Hand {idx+1}  gest={hd['gesture']:<7}"
               f"  conf={hd['confidence']:.2f}"
               f"  open={hd['openness']:.2f}"
               f"  [{hd['state']}]"
               f"  vel=({hd['vx']:.0f},{hd['vy']:.0f})")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        y0c, y1c = max(0, y-th-4), min(h, y+6)
        x1c = min(w, 14+tw)
        region = frame[y0c:y1c, 6:x1c]
        black  = np.zeros_like(region)
        cv2.addWeighted(black, 0.45, region, 0.55, 0, region)
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,255,220), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
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
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    systems    = {}
    gestures   = {}

    sparkles          = AmbientSparkles()
    charge_fx         = ChargeEffect()
    particle_renderer = ParticleRenderer()
    cam_shake         = CameraShake()
    fps_ctr           = FPSCounter()
    t0                = time.time()
    prev_time         = t0

    debug_mode  = False
    shake_on    = True
    bloom_on    = True

    skeleton_layer = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    sparkle_layer  = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    hud_layer      = np.zeros((36, CAMERA_WIDTH, 3), dtype=np.uint8)

    print("Hand Shatter Effect v3  –  q=quit  r=reset  d=debug  s=shake  b=bloom")

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now   = time.time()
            dt    = max(0.001, now - prev_time)
            prev_time = now
            t     = now - t0

            # MediaPipe on downscaled frame
            det_w = max(1, int(fw * MP_DETECT_SCALE))
            det_h = max(1, int(fh * MP_DETECT_SCALE))
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = lmk.detect_for_video(mp_img, int(now * 1000))

            hand_data_hud = []
            skeleton_layer.fill(0)
            sparkle_layer.fill(0)
            any_skeleton = any_sparkle = False

            if res.hand_landmarks:
                active_ids = set()
                for idx, hand_lm in enumerate(res.hand_landmarks):
                    pts       = lm_to_px(hand_lm, fw, fh)
                    pts_px_int = [(int(p[0]), int(p[1])) for p in pts]
                    active_ids.add(idx)

                    if idx not in systems:
                        systems[idx]  = ShatterSystem(seed=idx * 31 + 7)
                        gestures[idx] = GestureRecogniser()

                    gr   = gestures[idx]
                    sys_ = systems[idx]

                    gesture, confidence = gr.update(pts, dt)

                    hx, hy  = palm_centre(pts)
                    ps      = palm_size(pts)
                    cube_s  = float(np.clip(ps * 0.85, 50, 160))

                    # Trigger camera shake at explosion moment
                    prev_state = sys_.state
                    sys_.update(hx, hy, gesture, gr.openness, now, cube_s, t, dt)
                    if (shake_on and
                            prev_state == CRACKING and sys_.state == EXPLODING):
                        cam_shake.trigger(now)

                    draw_skeleton(frame, pts_px_int, skeleton_layer)
                    any_skeleton = True

                    sp_spread = cube_s * (0.55 + 0.55 * gr.openness)
                    sparkles.draw(sparkle_layer, hx, hy, sp_spread, t)
                    any_sparkle = True

                    if sys_.state in (INTACT, BUILDING, CHARGING, STABILIZE, CRACKING):
                        ci = (0.6 if sys_.state == INTACT else
                              min((now - sys_.t0) / LAYER_BUILD_SECS, 1.0)
                              if sys_.state == BUILDING else 0.5)
                        charge_fx.draw(frame, hx, hy, cube_s * 1.4, t, ci)

                    sys_.draw(frame, now, particle_renderer, dt,
                               enable_shake=shake_on, enable_bloom=bloom_on,
                               cam_shake=cam_shake,
                               palm_velocity=gr.palm_velocity)

                    vx, vy = float(gr.palm_velocity[0]), float(gr.palm_velocity[1])
                    hand_data_hud.append({
                        "gesture":    gesture,
                        "confidence": confidence,
                        "openness":   gr.openness,
                        "state":      _STATE_NAMES[sys_.state],
                        "vx": vx, "vy": vy,
                    })

                for gone in set(systems.keys()) - active_ids:
                    del systems[gone]; del gestures[gone]
            else:
                systems.clear(); gestures.clear()

            if any_skeleton:
                cv2.addWeighted(skeleton_layer, 0.6, frame, 1.0, 0, frame)
            if any_sparkle:
                cv2.add(frame, sparkle_layer, dst=frame)

            # Camera shake (applied after all rendering)
            if shake_on:
                frame = cam_shake.apply(frame, now)

            fps_ctr.tick()
            draw_hud(frame, fps_ctr.fps, debug_mode, hand_data_hud, hud_layer,
                     shake_on, bloom_on)

            cv2.imshow("Hand Shatter v3  [q=quit  r=reset  d=debug  s=shake  b=bloom]", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('r'):
                systems.clear(); gestures.clear()
                print("[INFO] Effect reset.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug {'ON' if debug_mode else 'OFF'}.")
            elif key == ord('s'):
                shake_on = not shake_on
                print(f"[INFO] Camera shake {'ON' if shake_on else 'OFF'}.")
            elif key == ord('b'):
                bloom_on = not bloom_on
                print(f"[INFO] Bloom pulse {'ON' if bloom_on else 'OFF'}.")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()

# Run with:
# C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe cube_shatter_effect.py