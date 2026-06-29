"""
cube_shatter_effect.py  –  Holographic Hand Shatter Effect  (v4 — True Plasma Edition)
══════════════════════════════════════════════════════════════════════════════════════════
Real-time AR hand-triggered shatter effect.

  - Closed fist  → idle (nothing shown, ready)
  - Open hand    → cracks appear → ENERGY SPHERE erupts — glowing electric
                   cyan tendrils form a writhing procedural plasma ball
  - Close again  → sphere collapses into the palm, shrinks and fades

PARTICLE VISUAL: Dense electric-blue filament tendrils that extend freely
beyond the core (NO sphere clamping), producing jagged lightning-like strands
with bright white-hot accumulation at the center — matches the reference
energy-sphere VFX style exactly.

Requirements:
    pip install opencv-python mediapipe numpy

Controls:
    q  =  quit
    r  =  reset
    d  =  toggle debug / perf overlay
    s  =  toggle camera shake
    b  =  toggle bloom pulse
"""

import os, sys, time, math, urllib.request, collections
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX   = 0
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 60

MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.65
TRACKING_CONFIDENCE  = 0.65
MP_DETECT_SCALE      = 0.6

FINGER_EXTEND_ANGLE = 35.0
OPEN_FINGER_RATIO   = 0.75
GESTURE_CONFIDENCE  = 0.85
CONFIRM_FRAMES      = 6
SMOOTHING           = 0.12

CUBE_HALF = 90

ORBIT_SECS       = 0.70
CONVERGE_SECS    = 0.50
LAYER_BUILD_SECS = 0.65
STABILIZE_SECS   = 0.40

CRACK_SECS = 0.25
N_CRACKS   = 18

SHAKE_DURATION  = 0.42
SHAKE_MAGNITUDE = 11.0
SHAKE_DECAY     = 5.5

BLOOM_PEAK_SIGMA = 28.0
BLOOM_DECAY_SECS = 0.5

# ── Energy Sphere Particle Config ──────────────────────────────────────────────

# Each "tendril" is a chain of points traced through noise space
N_TENDRILS          = 220    # number of independent filaments (more = denser plasma)
TENDRIL_PTS         = 22     # points per filament (longer strands)
TENDRIL_STEP        = 7.0    # px between points — bigger = longer reach

# Sphere geometry — defines the SPAWN region only, NOT a clamp boundary
SPHERE_RADIUS_MUL   = 1.15   # spawn radius = cube_s * this
SPHERE_INNER_GLOW   = 0.50   # inner bright core radius fraction

# How far beyond the sphere tendrils can wander (1.0 = sphere edge, 2.5 = 2.5x radius)
TENDRIL_OUTER_LIMIT = 2.2    # soft outer boundary multiplier — allows wild exterior strands

# Noise / motion params
CURL_FREQ           = 0.014  # spatial frequency — lower = bigger, lazier swirls
CURL_AMP            = 2.8    # amplitude — higher = more chaotic/jagged bends
TIME_SPEED          = 0.65   # how fast tendrils writhe
RADIAL_DRIFT_SPEED  = 0.28   # how fast tendril roots drift on sphere surface

# Inward pull: lower = more chaotic outward tendrils, higher = neat inward arcs
INWARD_BASE         = 0.10   # base inward pull (was 0.35 — much weaker now)
INWARD_TIP          = 0.30   # inward pull at tip (was 0.85)

# Color: white-hot core → bright cyan → deep blue (BGR format for OpenCV)
ENERGY_COLORS = [
    (255, 255, 255),   # pure white-hot (core filaments)
    (255, 248, 180),   # white-cyan
    (255, 220,  60),   # bright cyan
    (240, 180,  10),   # medium cyan
    (200, 130,   0),   # deep cyan-blue
    (160,  80,   0),   # dark blue edge
]

# Weight toward brighter colors for more white-hot appearance
COLOR_WEIGHTS = [0.20, 0.20, 0.22, 0.18, 0.12, 0.08]

SPARK_COUNT = 120   # more ambient sparks

CLR_GLOW     = (200, 248, 255)
CLR_CHARGE   = ( 80, 190, 255)
CLR_SKELETON = (  0, 175,  75)
CLR_JOINT    = (  0, 120, 255)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP    = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,10,11,12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   =13,14,15,16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  =17,18,19,20

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
    mcp = np.array(pts[mcp_i][:2]); pip = np.array(pts[pip_i][:2])
    tip = np.array(pts[tip_i][:2])
    v1 = mcp - pip; v2 = tip - pip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2),-1.,1.)))

def fingers_extended(pts):
    return [finger_angle(pts, mcp, pip, tip) > FINGER_EXTEND_ANGLE
            for mcp, pip, _dip, tip in FINGER_CHAINS]

def draw_skeleton(frame, pts_px, layer):
    for a, b in HAND_CONNECTIONS:
        cv2.line(layer, pts_px[a], pts_px[b], CLR_SKELETON, 1, cv2.LINE_AA)
    for p in pts_px:
        cv2.circle(layer, p, 3, CLR_JOINT, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNISER
# ═══════════════════════════════════════════════════════════════════════════════

class GestureRecogniser:
    def __init__(self):
        self._history   = collections.deque(maxlen=CONFIRM_FRAMES * 2)
        self.gesture    = "neutral"
        self.confidence = 0.0
        self._smooth    = 0.0
        self._open_cnt  = 0
        self._close_cnt = 0
        self._prev_palm = None
        self.palm_velocity = np.zeros(2, dtype=np.float32)

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
        palm = np.array(palm_centre(pts), dtype=np.float32)
        if self._prev_palm is not None and dt > 0:
            self.palm_velocity = (palm - self._prev_palm) / max(dt, 0.001)
        self._prev_palm = palm
        return self.gesture, self.confidence

    @property
    def openness(self): return self._smooth


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY SPHERE PARTICLE POOL  (v4 — unclamped plasma tendrils)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyParticlePool:
    """
    Procedural energy plasma: N_TENDRILS filaments driven by curl noise.

    KEY CHANGE vs v3: tendrils are NO LONGER clamped inside the sphere.
    They start near the core and are advected freely outward by curl noise,
    producing jagged lightning-like strands that extend well beyond the centre —
    exactly like the reference image.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.alive    = False
        self.collapse = 0.0
        self._t_birth = 0.0

        # Tendril root positions — spawn anywhere within inner core region
        # (not just on sphere surface — roots are scattered through the volume)
        self._root_r_frac = rng.uniform(0.0, 0.55, N_TENDRILS).astype(np.float32)  # 0=centre, 0.55=mid
        self._root_theta  = rng.uniform(0, math.pi,   N_TENDRILS).astype(np.float32)
        self._root_phi    = rng.uniform(0, 2*math.pi, N_TENDRILS).astype(np.float32)
        self._root_drift_spd = rng.uniform(0.04, 0.22, (N_TENDRILS, 2)).astype(np.float32)
        self._root_drift_off = rng.uniform(0, 2*math.pi,(N_TENDRILS, 2)).astype(np.float32)

        # Per-tendril noise offsets
        self._noise_ox = rng.uniform(0, 300.0, N_TENDRILS).astype(np.float32)
        self._noise_oy = rng.uniform(0, 300.0, N_TENDRILS).astype(np.float32)
        self._noise_oz = rng.uniform(0, 300.0, N_TENDRILS).astype(np.float32)

        # Per-tendril properties
        self._brightness = rng.uniform(0.55, 1.0, N_TENDRILS).astype(np.float32)
        # Weighted color selection — more white/bright cyan
        col_weights = np.array(COLOR_WEIGHTS, dtype=np.float64)
        col_weights /= col_weights.sum()
        self._color_idx = rng.choice(len(ENERGY_COLORS), N_TENDRILS, p=col_weights)
        self._thickness = rng.choice([1, 1, 1, 2, 2], N_TENDRILS)

        # Initial outward direction bias per tendril (so they radiate outward)
        self._out_angle  = rng.uniform(0, 2*math.pi, N_TENDRILS).astype(np.float32)
        self._out_bias   = rng.uniform(0.3, 1.2,     N_TENDRILS).astype(np.float32)

        # Pre-allocated point buffer
        self.pts = np.zeros((N_TENDRILS, TENDRIL_PTS, 2), dtype=np.float32)

        # Orbit/converge state
        self._frozen_pts  = np.zeros_like(self.pts)
        self._orbit_pts   = np.zeros_like(self.pts)
        self._orbit_phase = rng.uniform(0, 2*math.pi, N_TENDRILS).astype(np.float32)
        self._orbit_r_mul = rng.uniform(0.5, 1.3, N_TENDRILS).astype(np.float32)

    @staticmethod
    def _curl2d(x, y, ox, oy, oz, t, freq=CURL_FREQ):
        """
        2D curl noise via layered sine potential.
        Returns (nx, ny) curl direction — used to advect tendril points.
        """
        px = x * freq + ox + t * TIME_SPEED
        py = y * freq + oy + t * TIME_SPEED * 0.7
        pz = oz + t * TIME_SPEED * 0.5
        eps = 0.5
        def F(u, v):
            return (math.sin(u + pz) * math.cos(v * 0.9 + pz * 0.8) +
                    0.5  * math.sin(u * 2.1 + pz * 1.3) * math.cos(v * 1.8) +
                    0.25 * math.sin(u * 3.7 + pz * 0.6) +
                    0.12 * math.sin(u * 6.3 + pz * 1.1) * math.cos(v * 5.1))  # extra octave for jag
        dFdy = (F(px, py + eps) - F(px, py - eps)) / (2 * eps)
        dFdx = (F(px + eps, py) - F(px - eps, py)) / (2 * eps)
        return dFdy, -dFdx

    def update(self, cx, cy, sphere_r, t):
        if not self.alive:
            return

        scale = 1.0 - self.collapse
        r = sphere_r * scale
        outer_limit = r * TENDRIL_OUTER_LIMIT  # soft outer boundary

        for i in range(N_TENDRILS):
            # Drifting root within inner core volume
            dth = math.sin(self._root_drift_off[i,0] + t * self._root_drift_spd[i,0]) * RADIAL_DRIFT_SPEED
            dph = math.cos(self._root_drift_off[i,1] + t * self._root_drift_spd[i,1]) * RADIAL_DRIFT_SPEED
            theta = self._root_theta[i] + dth
            phi   = self._root_phi[i]   + dph

            st   = math.sin(theta)
            r0   = r * self._root_r_frac[i]   # root at variable depth in core
            px_  = cx + r0 * st * math.cos(phi)
            py_  = cy + r0 * math.cos(theta) * 0.88

            ox = self._noise_ox[i]
            oy = self._noise_oy[i]
            oz = self._noise_oz[i]

            self.pts[i, 0, 0] = px_
            self.pts[i, 0, 1] = py_

            for j in range(1, TENDRIL_PTS):
                frac = j / (TENDRIL_PTS - 1)

                # Curl noise direction
                nx, ny = self._curl2d(px_, py_, ox, oy, oz, t)

                # Outward radial bias (pushes tendrils away from centre)
                ddx = px_ - cx
                ddy = py_ - cy
                dist = math.sqrt(ddx*ddx + ddy*ddy) + 1e-5
                out_x = ddx / dist
                out_y = ddy / dist

                # Very weak inward pull only near outer limit (soft boundary)
                dist_frac = dist / max(outer_limit, 1.0)
                if dist_frac > 0.85:
                    # Soft repulsion back inward when approaching limit
                    pull = (dist_frac - 0.85) / 0.15  # 0→1 as approaching limit
                    inward_x = -out_x * pull * 1.5
                    inward_y = -out_y * pull * 1.5
                else:
                    inward_x = inward_y = 0.0

                # Combine: outward bias + curl + soft boundary
                out_bias = self._out_bias[i] * (1.0 - frac * 0.6)  # bias fades along tendril
                step_x = (out_x * out_bias * 0.4 + nx * CURL_AMP + inward_x) * TENDRIL_STEP
                step_y = (out_y * out_bias * 0.4 + ny * CURL_AMP + inward_y) * TENDRIL_STEP

                px_ += step_x
                py_ += step_y

                self.pts[i, j, 0] = px_
                self.pts[i, j, 1] = py_

    def explode(self, t):
        self.alive    = True
        self.collapse = 0.0
        self._t_birth = t

    def freeze(self):
        self._frozen_pts[:] = self.pts

    def step_orbit(self, hand_xy, ease, t):
        hx, hy = hand_xy
        phase  = self._orbit_phase + t * 2.8
        radius = 80.0 * self._orbit_r_mul * (1.0 - ease * 0.8)
        for i in range(N_TENDRILS):
            tx = hx + math.cos(phase[i]) * radius[i]
            ty = hy + math.sin(phase[i]) * radius[i] * 0.55
            b  = ease
            for j in range(TENDRIL_PTS):
                self.pts[i, j, 0] = self._frozen_pts[i, j, 0]*(1-b) + tx*b
                self.pts[i, j, 1] = self._frozen_pts[i, j, 1]*(1-b) + ty*b

    def snapshot_orbit_anchor(self):
        self._orbit_pts[:] = self.pts

    def step_converge(self, hand_xy, ease):
        hx, hy = hand_xy
        for i in range(N_TENDRILS):
            for j in range(TENDRIL_PTS):
                self.pts[i, j, 0] = self._orbit_pts[i, j, 0]*(1-ease) + hx*ease
                self.pts[i, j, 1] = self._orbit_pts[i, j, 1]*(1-ease) + hy*ease
        self.collapse = ease

    def deactivate_all(self):
        self.alive    = False
        self.collapse = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY SPHERE RENDERER  (v4 — brighter, more additive, white-hot core)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyRenderer:
    """
    Renders EnergyParticlePool as glowing filament lines.

    Pipeline (v4):
    1. Draw all tendril segments onto float32 accumulation buffer (additive)
    2. Three-pass Gaussian bloom (wide soft glow, mid halo, tight bright)
    3. Additive composite onto frame
    4. White-hot radial core glow
    5. Bright white centre point burst
    """

    def __init__(self):
        self._buf1 = None
        self._cap  = (0, 0)

    def _ensure_bufs(self, h, w):
        if self._buf1 is None or self._cap != (h, w):
            self._buf1 = np.zeros((h, w, 3), dtype=np.float32)
            self._cap  = (h, w)

    def draw(self, frame, pool: EnergyParticlePool,
             cx, cy, sphere_r, fade=1.0, bloom_boost=0.0):
        if not pool.alive:
            return

        h, w = frame.shape[:2]
        self._ensure_bufs(h, w)
        buf = self._buf1
        buf[:] = 0.0

        scale = 1.0 - pool.collapse
        r_eff = sphere_r * max(scale, 0.01)

        # ── Draw tendrils (additive) ──────────────────────────────────────
        for i in range(N_TENDRILS):
            br      = float(pool._brightness[i]) * fade
            col_bgr = ENERGY_COLORS[pool._color_idx[i]]
            th      = int(pool._thickness[i])

            for j in range(TENDRIL_PTS - 1):
                p0 = (int(pool.pts[i, j,   0]), int(pool.pts[i, j,   1]))
                p1 = (int(pool.pts[i, j+1, 0]), int(pool.pts[i, j+1, 1]))

                # Brightness: highest near root (j=0), fades toward tip
                seg_br = br * (1.0 - 0.65 * (j / (TENDRIL_PTS - 1)))
                c = (col_bgr[0] * seg_br, col_bgr[1] * seg_br, col_bgr[2] * seg_br)
                cv2.line(buf, p0, p1,
                         (float(min(c[0],255)), float(min(c[1],255)), float(min(c[2],255))),
                         th, cv2.LINE_AA)

            # White-hot root dot (additive — will overbright to white at centre)
            rp = (int(pool.pts[i, 0, 0]), int(pool.pts[i, 0, 1]))
            if 0 <= rp[0] < w and 0 <= rp[1] < h:
                buf[rp[1], rp[0]] = np.minimum(
                    500.0,   # allow overbright — clamped at composite stage
                    buf[rp[1], rp[0]] + np.array([255.0, 255.0, 255.0], dtype=np.float32)
                )

        # ── Three-pass bloom ─────────────────────────────────────────────
        # Pass 1: wide soft glow
        ds1 = 6
        sw1, sh1 = max(1, w//ds1), max(1, h//ds1)
        s1  = cv2.resize(buf, (sw1,sh1), interpolation=cv2.INTER_LINEAR)
        sig1 = max(2.0, r_eff * 0.22 + bloom_boost * 0.07)
        b1  = cv2.GaussianBlur(s1, (0,0), sigmaX=sig1)
        bloom1 = cv2.resize(b1, (w,h), interpolation=cv2.INTER_LINEAR)

        # Pass 2: mid bright halo
        ds2 = 3
        sw2, sh2 = max(1, w//ds2), max(1, h//ds2)
        s2  = cv2.resize(buf, (sw2,sh2), interpolation=cv2.INTER_LINEAR)
        sig2 = max(1.0, r_eff * 0.09 + bloom_boost * 0.04)
        b2  = cv2.GaussianBlur(s2, (0,0), sigmaX=sig2)
        bloom2 = cv2.resize(b2, (w,h), interpolation=cv2.INTER_LINEAR)

        # Pass 3: tight sharp halo (full-res, small sigma)
        sig3 = max(0.5, r_eff * 0.03)
        bloom3 = cv2.GaussianBlur(buf, (0,0), sigmaX=sig3)

        combined = (buf * 1.2          # raw lines
                  + bloom1 * 1.8       # wide glow (dominant — creates the plasma cloud)
                  + bloom2 * 1.1       # mid halo
                  + bloom3 * 0.6)      # tight detail
        if bloom_boost > 0:
            combined += bloom1 * (bloom_boost * 0.015)
        np.clip(combined, 0, 255, out=combined)
        cv2.add(frame, combined.astype(np.uint8), dst=frame)

        # ── White-hot radial core glow ────────────────────────────────────
        if r_eff > 10 and fade > 0.05:
            core_r   = int(r_eff * SPHERE_INNER_GLOW)
            core_img = np.zeros((h, w, 3), dtype=np.uint8)
            icx, icy = int(cx), int(cy)

            # Layers from largest (dim blue outer) to smallest (white-hot centre)
            layers = [
                (core_r,         (  5,  15,  10), int(22 * fade)),   # outer dim blue
                (core_r * 3//4,  ( 20,  60,  20), int(35 * fade)),   # mid cyan
                (core_r // 2,    ( 60, 150,  60), int(50 * fade)),   # bright cyan
                (core_r // 3,    (140, 220, 140), int(65 * fade)),   # near-white cyan
                (core_r // 5,    (220, 255, 220), int(80 * fade)),   # white-hot
                (max(4, core_r//8), (255,255,255), int(120 * fade)), # pure white burst
            ]
            for radius_c, col_c, alpha_c in layers:
                if radius_c < 2: continue
                overlay = core_img.copy()
                cv2.circle(overlay, (icx, icy), radius_c, col_c, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, min(alpha_c/255.0, 1.0), core_img, 1.0, 0, core_img)
            cv2.add(frame, core_img, dst=frame)


# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER STATES
# ═══════════════════════════════════════════════════════════════════════════════

(INTACT, CHARGING, CRACKING, EXPLODING, FLOATING,
 ORBIT, CONVERGE, BUILDING, STABILIZE) = range(9)

_STATE_NAMES = {
    INTACT: "INTACT", CHARGING: "CHARGING", CRACKING: "CRACKING",
    EXPLODING: "EXPLODING", FLOATING: "FLOATING", ORBIT: "ORBIT",
    CONVERGE: "CONVERGE", BUILDING: "BUILDING", STABILIZE: "STABILIZE",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CRACK EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class CrackEffect:
    def __init__(self):
        self._lines = []

    def generate(self, cx, cy, cube_s, seed=None):
        rng = np.random.default_rng(seed)
        self._lines.clear()
        for _ in range(N_CRACKS):
            ang    = rng.uniform(0, 2*math.pi)
            length = rng.uniform(cube_s * 0.4, cube_s * 1.6)
            sx = cx + rng.uniform(-cube_s*0.7, cube_s*0.7)
            sy = cy + rng.uniform(-cube_s*0.7, cube_s*0.7)
            ex = sx + math.cos(ang) * length
            ey = sy + math.sin(ang) * length
            self._lines.append(((int(sx), int(sy)), (int(ex), int(ey))))

    def draw(self, frame, progress):
        if not self._lines: return
        n_show = max(1, int(len(self._lines) * progress))
        for i, (p0, p1) in enumerate(self._lines[:n_show]):
            frac  = i / max(len(self._lines), 1)
            alpha = 0.55 + 0.45 * (1 - frac) * progress
            col   = tuple(int(c * alpha) for c in CLR_GLOW)
            cv2.line(frame, p0, p1, col, 2, cv2.LINE_AA)
            cv2.line(frame, p0, p1, (255,255,255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA SHAKE
# ═══════════════════════════════════════════════════════════════════════════════

class CameraShake:
    def __init__(self):
        self._t0 = -999.0
        self._dx = self._dy = 0

    def trigger(self, now, magnitude=SHAKE_MAGNITUDE):
        self._t0 = now
        self._dx = int(np.random.uniform(-magnitude, magnitude))
        self._dy = int(np.random.uniform(-magnitude, magnitude))

    def apply(self, frame, now):
        dt = now - self._t0
        if dt > SHAKE_DURATION: return frame
        decay = math.exp(-SHAKE_DECAY * dt)
        dx, dy = int(self._dx*decay), int(self._dy*decay)
        if dx == 0 and dy == 0: return frame
        out = np.roll(frame, dy, axis=0)
        out = np.roll(out, dx, axis=1)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ShatterSystem:
    def __init__(self, seed=7):
        self.state  = INTACT
        self.t0     = 0.0
        self.hx = self.hy = 0.0
        self.cube_s = float(CUBE_HALF)
        self.ex = self.ey = 0.0
        self._bloom_t0 = -999.0
        self.t_anim    = 0.0

        self.pool   = EnergyParticlePool(seed=seed)
        self.cracks = CrackEffect()

    def _begin_crack(self, now):
        self.state = CRACKING
        self.t0    = now
        self.cracks.generate(self.hx, self.hy, self.cube_s, seed=int(now*1000))

    def _begin_explosion(self, now, t):
        self.state     = EXPLODING
        self.t0        = now
        self._bloom_t0 = now
        self.pool.explode(t)

    def update(self, hx, hy, gesture, openness, now, cube_s, t, dt):
        self.hx, self.hy = hx, hy
        self.cube_s  = cube_s
        self.t_anim  = t

        if gesture == "open" and self.state in (INTACT, CHARGING, BUILDING, STABILIZE):
            self.ex, self.ey = hx, hy
            self._begin_crack(now)

        elif gesture == "closed" and self.state in (EXPLODING, FLOATING):
            self.state = ORBIT
            self.t0    = now
            self.pool.freeze()

        if self.state == CRACKING:
            if now - self.t0 > CRACK_SECS:
                self._begin_explosion(now, t)

        elif self.state in (EXPLODING, FLOATING):
            sphere_r = self.cube_s * SPHERE_RADIUS_MUL
            self.pool.update(hx, hy, sphere_r, t)
            if self.state == EXPLODING and now - self.t0 > 1.6:
                self.state = FLOATING

        elif self.state == ORBIT:
            dt_s  = now - self.t0
            ease  = min(dt_s / ORBIT_SECS, 1.0)
            ease_s = ease*ease*(3 - 2*ease)
            self.pool.step_orbit((hx, hy), ease_s, t)
            if ease >= 1.0:
                self.pool.snapshot_orbit_anchor()
                self.state = CONVERGE
                self.t0    = now

        elif self.state == CONVERGE:
            dt_s  = now - self.t0
            ease  = min(dt_s / CONVERGE_SECS, 1.0)
            ease_s = ease*ease*(3 - 2*ease)
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

    def draw(self, frame, now, renderer: EnergyRenderer, dt,
             enable_shake=True, enable_bloom=True, cam_shake=None,
             palm_velocity=None):
        t = self.t_anim

        bloom_boost = 0.0
        if enable_bloom:
            bd = now - self._bloom_t0
            if bd < BLOOM_DECAY_SECS:
                bloom_boost = BLOOM_PEAK_SIGMA * (1.0 - bd / BLOOM_DECAY_SECS)

        sphere_r = self.cube_s * SPHERE_RADIUS_MUL

        if self.state == CRACKING:
            prog = min((now - self.t0) / CRACK_SECS, 1.0)
            self.cracks.draw(frame, prog)

        elif self.state in (EXPLODING, FLOATING):
            renderer.draw(frame, self.pool, self.ex, self.ey,
                          sphere_r, fade=1.0, bloom_boost=bloom_boost)

        elif self.state == ORBIT:
            renderer.draw(frame, self.pool, self.ex, self.ey,
                          sphere_r, fade=1.0)

        elif self.state == CONVERGE:
            fade = max(0.0, 1.0 - (now - self.t0) / CONVERGE_SECS * 0.7)
            renderer.draw(frame, self.pool, self.ex, self.ey,
                          sphere_r * (1.0 - self.pool.collapse),
                          fade=fade)

        elif self.state == BUILDING:
            prog = min((now - self.t0) / LAYER_BUILD_SECS, 1.0)
            fade = max(0.0, 1.0 - prog)
            if fade > 0.01:
                renderer.draw(frame, self.pool, self.ex, self.ey,
                              sphere_r, fade=fade)


# ═══════════════════════════════════════════════════════════════════════════════
# AMBIENT SPARKLES
# ═══════════════════════════════════════════════════════════════════════════════

class AmbientSparkles:
    def __init__(self, n=SPARK_COUNT, seed=42):
        rng        = np.random.default_rng(seed)
        self.ang   = rng.uniform(0, math.pi*2, n)
        self.reach = rng.uniform(0.05, 1.6, n)   # reach further out (was 1.0)
        self.sz    = rng.uniform(1.0, 3.5, n)
        self.spd   = rng.uniform(0.5, 2.0, n)
        pal = [(255,255,255),(230,255,240),(200,255,180),(180,210,255),(255,230,200)]
        self.col = np.array(pal)[rng.integers(0, len(pal), n)]

    def draw(self, layer, cx, cy, spread, t):
        if spread < 5: return
        h, w = layer.shape[:2]
        jx = np.sin(t * self.spd * 2.8 + self.reach * 28) * 8
        jy = np.cos(t * self.spd * 2.3 + self.reach * 15) * 8
        x  = cx + np.cos(self.ang) * self.reach * spread + jx
        y  = cy + np.sin(self.ang) * self.reach * spread + jy
        for i in range(len(x)):
            px, py = int(x[i]), int(y[i])
            if not (0 <= px < w and 0 <= py < h): continue
            br  = max(0.1, 1.0 - 0.55 * self.reach[i])
            col = tuple(int(c * br) for c in self.col[i])
            cv2.circle(layer, (px, py), max(1, int(self.sz[i])), col, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGE EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class ChargeEffect:
    def __init__(self, n=22, seed=9):
        rng        = np.random.default_rng(seed)
        self.angs  = rng.uniform(0, math.pi*2, n)
        self.dists = rng.uniform(0.5, 1.5, n)
        self.spds  = rng.uniform(0.8, 2.2, n)

    def draw(self, frame, cx, cy, spread, t, intensity):
        if intensity < 0.02 or spread < 5: return
        h, w = frame.shape[:2]
        for i, ang in enumerate(self.angs):
            phase  = (t * self.spds[i]) % 1.0
            d_out  = self.dists[i] * spread * (1.0 - phase)
            d_in   = max(0, d_out - spread * 0.18)
            sx     = int(cx + math.cos(ang) * d_out)
            sy     = int(cy + math.sin(ang) * d_out)
            ex_    = int(cx + math.cos(ang) * d_in)
            ey_    = int(cy + math.sin(ang) * d_in)
            if not (0 <= sx < w and 0 <= sy < h): continue
            al  = intensity * (0.3 + 0.7 * phase)
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
        if len(self._times) < 2: return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ═══════════════════════════════════════════════════════════════════════════════
# HUD
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps, debug, hand_data, hud_layer, shake_on, bloom_on):
    h, w = frame.shape[:2]
    hud_layer[:36, :] = 0
    cv2.rectangle(hud_layer, (0,0), (w,36), (255,255,255), -1)
    cv2.addWeighted(hud_layer[:36], 0.40, frame[:36], 0.60, 0, frame[:36])
    fps_col = (100,255,100) if fps >= 45 else (0,165,255) if fps >= 25 else (0,60,255)
    cv2.putText(frame, f"FPS {fps:.0f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_col, 1, cv2.LINE_AA)
    flags = f"  shake={'ON' if shake_on else 'OFF'}  bloom={'ON' if bloom_on else 'OFF'}"
    hints = f"q=quit  r=reset  d=debug  s=shake  b=bloom{flags}"
    cv2.putText(frame, hints, (w - 560, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170,170,170), 1, cv2.LINE_AA)
    if not debug or not hand_data: return
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
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    systems  = {}
    gestures = {}

    sparkles  = AmbientSparkles()
    charge_fx = ChargeEffect()
    renderer  = EnergyRenderer()
    cam_shake = CameraShake()
    fps_ctr   = FPSCounter()

    t0        = time.time()
    prev_time = t0
    debug_mode = False
    shake_on   = True
    bloom_on   = True

    skeleton_layer = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    sparkle_layer  = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    hud_layer      = np.zeros((36, CAMERA_WIDTH, 3), dtype=np.uint8)

    print("Hand Shatter Effect — True Plasma Edition (v4)")
    print("  q=quit  r=reset  d=debug  s=shake  b=bloom")
    print("  Open hand → plasma orb  |  Close fist → collapse")

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now    = time.time()
            dt     = max(0.001, now - prev_time)
            prev_time = now
            t      = now - t0

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
                    pts        = lm_to_px(hand_lm, fw, fh)
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

                    prev_state = sys_.state
                    sys_.update(hx, hy, gesture, gr.openness, now, cube_s, t, dt)

                    if shake_on and prev_state == CRACKING and sys_.state == EXPLODING:
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

                    sys_.draw(frame, now, renderer, dt,
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

            if shake_on:
                frame = cam_shake.apply(frame, now)

            fps_ctr.tick()
            draw_hud(frame, fps_ctr.fps, debug_mode, hand_data_hud,
                     hud_layer, shake_on, bloom_on)

            cv2.imshow("Hand Shatter — True Plasma  [q=quit  r=reset  d=debug  s=shake  b=bloom]", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('r'):
                systems.clear(); gestures.clear()
                print("[INFO] Reset.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug {'ON' if debug_mode else 'OFF'}.")
            elif key == ord('s'):
                shake_on = not shake_on
                print(f"[INFO] Shake {'ON' if shake_on else 'OFF'}.")
            elif key == ord('b'):
                bloom_on = not bloom_on
                print(f"[INFO] Bloom {'ON' if bloom_on else 'OFF'}.")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()

# Run with:
# python cube_shatter_effect.py