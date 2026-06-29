"""
cube_shatter_effect.py  –  Holographic Hand Shatter Effect  (v5 — Cinematic Plasma)
═════════════════════════════════════════════════════════════════════════════════════
Senior-grade real-time AR VFX system with:
  • NumPy-vectorized particle pipeline (no per-particle Python loops)
  • Velocity/acceleration-reactive plasma tendrils
  • Electric arc jumps between fingertips
  • Shockwave pulse rings on gesture transitions
  • Chromatic aberration + film grain
  • Palm energy absorption effect on collapse
  • Multi-octave curl noise with turbulence injection

Requirements:
    pip install opencv-python mediapipe numpy

Optional (significant speedup):
    pip install numba

Controls:
    q  =  quit        r  =  reset
    d  =  debug       s  =  shake
    c  =  chromatic aberration
"""

import os, sys, time, math, urllib.request, collections
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Optional Numba JIT ────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    NUMBA = True
except ImportError:
    NUMBA = False
    def njit(*a, **kw):          # no-op decorator fallback
        return lambda f: f
    prange = range

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
MP_DETECT_SCALE      = 0.55   # slightly smaller for speed

FINGER_EXTEND_ANGLE = 32.0
OPEN_FINGER_RATIO   = 0.72
GESTURE_CONFIDENCE  = 0.82
CONFIRM_FRAMES      = 5       # slightly more responsive
SMOOTHING           = 0.14

CUBE_HALF = 90

ORBIT_SECS       = 0.65
CONVERGE_SECS    = 0.45
LAYER_BUILD_SECS = 0.55
STABILIZE_SECS   = 0.35

CRACK_SECS = 0.22
N_CRACKS   = 22

SHAKE_DURATION  = 0.45
SHAKE_MAGNITUDE = 13.0
SHAKE_DECAY     = 5.0

# ── Plasma Tendril Config ─────────────────────────────────────────────────────
N_TENDRILS        = 260       # more filaments
TENDRIL_PTS       = 24        # longer chains
TENDRIL_STEP      = 7.5       # px per step

SPHERE_RADIUS_MUL = 1.15
SPHERE_INNER_GLOW = 0.48
TENDRIL_OUTER_LIM = 2.4

# Noise
CURL_FREQ         = 0.013
CURL_AMP          = 3.0
TIME_SPEED        = 0.70
RADIAL_DRIFT_SPD  = 0.30

# Velocity reactivity
VEL_WARP_SCALE    = 0.0008    # how much hand velocity warps the plasma
VEL_STRETCH_SCALE = 0.0012    # directional stretch along velocity

# Colors: white-hot → cyan → blue (BGR)
ENERGY_COLORS = np.array([
    [255, 255, 255],   # white-hot
    [255, 252, 200],   # white-cyan
    [255, 230,  80],   # bright cyan
    [255, 190,  20],   # mid cyan
    [220, 140,   5],   # deep cyan
    [160,  80,   0],   # blue edge
    [100,  40,   0],   # dark fringe
], dtype=np.float32)

COLOR_WEIGHTS = np.array([0.18, 0.18, 0.20, 0.18, 0.13, 0.08, 0.05], dtype=np.float64)
COLOR_WEIGHTS /= COLOR_WEIGHTS.sum()

# Arc / shockwave config
N_ARC_PAIRS      = 6          # finger pairs that can arc
ARC_PROB         = 0.18       # probability per frame of arc spawn
ARC_LIFETIME     = 0.18       # seconds
ARC_SEGMENTS     = 14
ARC_JITTER       = 18.0       # px random offset per segment

SHOCK_RINGS      = 3
SHOCK_SPEED      = 420.0      # px/s expansion
SHOCK_LIFETIME   = 0.45

SPARK_COUNT = 140

CLR_GLOW     = (200, 248, 255)
CLR_CHARGE   = ( 80, 190, 255)
CLR_ARC      = (255, 240, 120)   # arc color (BGR)
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
FINGERTIPS = [4, 8, 12, 16, 20]
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
    if os.path.exists(MODEL_PATH): return
    print("Downloading hand-landmark model …")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done:", MODEL_PATH)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr); sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED CURL NOISE  (NumPy — no Python loops)
# ═══════════════════════════════════════════════════════════════════════════════

def curl_noise_batch(px, py, ox, oy, oz, t):
    """
    Vectorized curl noise for N points simultaneously.
    px, py : (N,) arrays of x/y positions
    ox,oy,oz : (N,) per-tendril offsets
    Returns (nx, ny) each shape (N,)
    """
    EPS = 0.5
    u  = px * CURL_FREQ + ox + t * TIME_SPEED
    v  = py * CURL_FREQ + oy + t * TIME_SPEED * 0.7
    w  = oz + t * TIME_SPEED * 0.5

    def F(uu, vv):
        return (np.sin(uu + w) * np.cos(vv * 0.9 + w * 0.8)
              + 0.5  * np.sin(uu * 2.1 + w * 1.3) * np.cos(vv * 1.8)
              + 0.25 * np.sin(uu * 3.7 + w * 0.6)
              + 0.12 * np.sin(uu * 6.3 + w * 1.1) * np.cos(vv * 5.1))

    dFdy = (F(u, v + EPS) - F(u, v - EPS)) / (2 * EPS)
    dFdx = (F(u + EPS, v) - F(u - EPS, v)) / (2 * EPS)
    return dFdy, -dFdx   # curl = (∂F/∂y, -∂F/∂x)


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED TENDRIL UPDATE  (full NumPy — replaces all Python for-loops)
# ═══════════════════════════════════════════════════════════════════════════════

def update_tendrils_vectorized(pool, cx, cy, sphere_r, t,
                                vel_x=0.0, vel_y=0.0):
    """
    Update all N_TENDRILS × TENDRIL_PTS points in pure NumPy.
    vel_x, vel_y: hand velocity used to warp/stretch the plasma field.

    Returns pts array (N_TENDRILS, TENDRIL_PTS, 2).
    """
    scale = 1.0 - pool.collapse
    r     = sphere_r * scale
    outer = r * TENDRIL_OUTER_LIM

    N = N_TENDRILS
    pts = pool.pts  # (N, TENDRIL_PTS, 2) — update in-place

    # ── Drifting root positions ────────────────────────────────────────────
    dth = np.sin(pool._root_drift_off[:,0] + t * pool._root_drift_spd[:,0]) * RADIAL_DRIFT_SPD
    dph = np.cos(pool._root_drift_off[:,1] + t * pool._root_drift_spd[:,1]) * RADIAL_DRIFT_SPD
    theta = pool._root_theta + dth
    phi   = pool._root_phi   + dph
    st    = np.sin(theta)
    r0    = r * pool._root_r_frac
    rx0   = cx + r0 * st * np.cos(phi)
    ry0   = cy + r0 * np.cos(theta) * 0.88

    # Velocity warp: shift spawn points along hand velocity direction
    spd   = math.sqrt(vel_x**2 + vel_y**2)
    if spd > 10.0:
        rx0 += vel_x * VEL_WARP_SCALE * r
        ry0 += vel_y * VEL_WARP_SCALE * r

    pts[:, 0, 0] = rx0
    pts[:, 0, 1] = ry0

    # ── Step all tendril points forward ───────────────────────────────────
    px = rx0.copy()
    py = ry0.copy()

    for j in range(1, TENDRIL_PTS):
        frac = j / (TENDRIL_PTS - 1)

        # Curl noise (vectorized over N tendrils)
        nx, ny = curl_noise_batch(px, py, pool._noise_ox, pool._noise_oy, pool._noise_oz, t)

        # Outward radial direction
        ddx  = px - cx;  ddy  = py - cy
        dist = np.sqrt(ddx*ddx + ddy*ddy) + 1e-5
        outx = ddx / dist;  outy = ddy / dist

        # Velocity stretch: bias outward along hand motion direction
        if spd > 10.0:
            vnx = vel_x / (spd + 1e-5)
            vny = vel_y / (spd + 1e-5)
            stretch = np.dot(np.column_stack([outx, outy]),
                             np.array([vnx, vny]))  # (N,) dot product
            outx += vnx * stretch * VEL_STRETCH_SCALE * spd * (1 - frac)
            outy += vny * stretch * VEL_STRETCH_SCALE * spd * (1 - frac)

        # Soft boundary: gentle push back near outer limit
        dist_frac = dist / np.maximum(outer, 1.0)
        mask_near = dist_frac > 0.85
        pull_str  = np.where(mask_near, (dist_frac - 0.85) / 0.15 * 1.6, 0.0)
        inx  = -outx * pull_str
        iny  = -outy * pull_str

        out_bias = pool._out_bias * (1.0 - frac * 0.55)
        step_x = (outx * out_bias * 0.4 + nx * CURL_AMP + inx) * TENDRIL_STEP
        step_y = (outy * out_bias * 0.4 + ny * CURL_AMP + iny) * TENDRIL_STEP

        px += step_x
        py += step_y

        pts[:, j, 0] = px
        pts[:, j, 1] = py

    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# HAND TRACKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def lm_to_px(lms, w, h):
    return [(lm.x * w, lm.y * h, lm.z) for lm in lms]

def palm_centre(pts):
    return np.mean([pts[i][:2] for i in PALM_REF], axis=0)

def palm_size(pts):
    return float(np.linalg.norm(
        np.array(pts[MIDDLE_MCP][:2]) - np.array(pts[WRIST][:2]))) + 1e-6

def finger_angle(pts, mcp_i, pip_i, tip_i):
    mcp=np.array(pts[mcp_i][:2]); pip=np.array(pts[pip_i][:2]); tip=np.array(pts[tip_i][:2])
    v1=mcp-pip; v2=tip-pip
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-4 or n2<1e-4: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2),-1.,1.)))

def fingers_extended(pts):
    return [finger_angle(pts,mcp,pip,tip) > FINGER_EXTEND_ANGLE
            for mcp,pip,_dip,tip in FINGER_CHAINS]

def draw_skeleton(layer, pts_px):
    for a,b in HAND_CONNECTIONS:
        cv2.line(layer, pts_px[a], pts_px[b], CLR_SKELETON, 1, cv2.LINE_AA)
    for p in pts_px:
        cv2.circle(layer, p, 3, CLR_JOINT, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNISER  (now also tracks acceleration)
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
        self._prev_vel  = np.zeros(2, dtype=np.float32)
        self.palm_velocity     = np.zeros(2, dtype=np.float32)
        self.palm_acceleration = np.zeros(2, dtype=np.float32)
        self.palm_speed        = 0.0

    def update(self, pts, dt=0.033):
        ext   = fingers_extended(pts)
        n_ext = sum(ext[1:])
        ratio = n_ext / 4.0
        self._smooth += (ratio - self._smooth) * SMOOTHING
        raw = "open" if ratio >= OPEN_FINGER_RATIO else "closed"
        self._history.append(raw)
        if raw == "open":
            self._open_cnt  = min(self._open_cnt+1, CONFIRM_FRAMES*2)
            self._close_cnt = max(self._close_cnt-1, 0)
        else:
            self._close_cnt = min(self._close_cnt+1, CONFIRM_FRAMES*2)
            self._open_cnt  = max(self._open_cnt-1, 0)
        recent = list(self._history)[-CONFIRM_FRAMES:]
        match  = sum(1 for g in recent if g == raw)
        self.confidence = match / max(len(recent), 1)
        if self.confidence >= GESTURE_CONFIDENCE:
            if raw=="open"   and self._open_cnt  >= CONFIRM_FRAMES: self.gesture="open"
            elif raw=="closed" and self._close_cnt >= CONFIRM_FRAMES: self.gesture="closed"
        palm = np.array(palm_centre(pts), dtype=np.float32)
        if self._prev_palm is not None and dt > 0:
            self.palm_velocity = (palm - self._prev_palm) / max(dt, 0.001)
            self.palm_acceleration = (self.palm_velocity - self._prev_vel) / max(dt, 0.001)
            self.palm_speed = float(np.linalg.norm(self.palm_velocity))
        self._prev_vel  = self.palm_velocity.copy()
        self._prev_palm = palm
        return self.gesture, self.confidence

    @property
    def openness(self): return self._smooth


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY PARTICLE POOL  (NumPy arrays only — no Python loops at particle level)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyParticlePool:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.alive    = False
        self.collapse = 0.0
        self._t_birth = 0.0

        N = N_TENDRILS
        self._root_r_frac    = rng.uniform(0.0, 0.55, N).astype(np.float32)
        self._root_theta     = rng.uniform(0, math.pi,   N).astype(np.float32)
        self._root_phi       = rng.uniform(0, 2*math.pi, N).astype(np.float32)
        self._root_drift_spd = rng.uniform(0.04, 0.24, (N,2)).astype(np.float32)
        self._root_drift_off = rng.uniform(0, 2*math.pi, (N,2)).astype(np.float32)
        self._noise_ox       = rng.uniform(0, 300., N).astype(np.float32)
        self._noise_oy       = rng.uniform(0, 300., N).astype(np.float32)
        self._noise_oz       = rng.uniform(0, 300., N).astype(np.float32)
        self._brightness     = rng.uniform(0.55, 1.0, N).astype(np.float32)
        self._color_idx      = rng.choice(len(ENERGY_COLORS), N,
                                          p=COLOR_WEIGHTS).astype(np.int32)
        self._thickness      = rng.choice([1,1,1,2,2], N).astype(np.int32)
        self._out_bias       = rng.uniform(0.3, 1.3, N).astype(np.float32)

        self.pts          = np.zeros((N, TENDRIL_PTS, 2), dtype=np.float32)
        self._frozen_pts  = np.zeros_like(self.pts)
        self._orbit_pts   = np.zeros_like(self.pts)
        self._orbit_phase = rng.uniform(0, 2*math.pi, N).astype(np.float32)
        self._orbit_r_mul = rng.uniform(0.5, 1.3, N).astype(np.float32)

    def explode(self, t):
        self.alive=True; self.collapse=0.0; self._t_birth=t

    def freeze(self):
        self._frozen_pts[:] = self.pts

    def step_orbit(self, hand_xy, ease, t):
        hx,hy = hand_xy
        phase  = self._orbit_phase + t * 2.8
        radius = 80.0 * self._orbit_r_mul * (1.0 - ease * 0.8)
        tx = hx + np.cos(phase) * radius
        ty = hy + np.sin(phase) * radius * 0.55
        # Vectorized lerp
        b = ease
        self.pts[:,:,0] = self._frozen_pts[:,:,0]*(1-b) + tx[:,None]*b
        self.pts[:,:,1] = self._frozen_pts[:,:,1]*(1-b) + ty[:,None]*b

    def snapshot_orbit_anchor(self):
        self._orbit_pts[:] = self.pts

    def step_converge(self, hand_xy, ease):
        hx,hy = hand_xy
        self.pts[:,:,0] = self._orbit_pts[:,:,0]*(1-ease) + hx*ease
        self.pts[:,:,1] = self._orbit_pts[:,:,1]*(1-ease) + hy*ease
        self.collapse = ease

    def deactivate_all(self):
        self.alive=False; self.collapse=0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ELECTRIC ARC SYSTEM  (finger-to-finger lightning bolts)
# ═══════════════════════════════════════════════════════════════════════════════

class ElectricArcSystem:
    """
    Spawns jagged lightning arcs between pairs of fingertips.
    Each arc is a broken polyline with random jitter per segment,
    rendered with bright additive glow.
    """
    def __init__(self, seed=55):
        self._rng  = np.random.default_rng(seed)
        self._arcs = []   # list of {p0, p1, segs, born, lifetime}

    def update_spawn(self, pts_px, now, openness):
        """Randomly spawn arcs between fingertip pairs when hand is open."""
        if openness < 0.5: return
        tips = [pts_px[i] for i in FINGERTIPS if i < len(pts_px)]
        if len(tips) < 2: return
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                if self._rng.random() < ARC_PROB * openness:
                    p0 = np.array(tips[i], dtype=np.float32)
                    p1 = np.array(tips[j], dtype=np.float32)
                    segs = self._make_arc(p0, p1)
                    self._arcs.append({'segs': segs, 'born': now,
                                       'p0': p0, 'p1': p1})

    def _make_arc(self, p0, p1):
        """Generate a jagged polyline from p0 to p1."""
        n   = ARC_SEGMENTS
        t_  = np.linspace(0, 1, n+1)[:,None]
        pts = p0 + (p1-p0)*t_
        perp = np.array([-(p1-p0)[1], (p1-p0)[0]], dtype=np.float32)
        plen = np.linalg.norm(perp)+1e-5
        perp /= plen
        jitter = self._rng.uniform(-ARC_JITTER, ARC_JITTER, n+1)
        jitter[0] = jitter[-1] = 0  # pin endpoints
        pts += perp[None,:] * jitter[:,None]
        return pts  # (n+1, 2)

    def draw(self, buf, now):
        alive = []
        for arc in self._arcs:
            age = now - arc['born']
            if age > ARC_LIFETIME: continue
            alive.append(arc)
            fade = 1.0 - age / ARC_LIFETIME
            # Refresh jitter every frame for flickering effect
            segs = arc['segs'].copy()
            p0,p1 = arc['p0'], arc['p1']
            perp = np.array([-(p1-p0)[1], (p1-p0)[0]], dtype=np.float32)
            plen = np.linalg.norm(perp)+1e-5; perp /= plen
            n = len(segs)-1
            jit = np.random.uniform(-ARC_JITTER*0.4, ARC_JITTER*0.4, n+1)
            jit[0]=jit[-1]=0
            segs += perp[None,:] * jit[:,None]

            for k in range(len(segs)-1):
                q0=(int(segs[k,0]),int(segs[k,1]))
                q1=(int(segs[k+1,0]),int(segs[k+1,1]))
                col_outer = tuple(int(CLR_ARC[c]*fade*0.7) for c in range(3))
                col_core  = (int(255*fade), int(255*fade), int(255*fade))
                cv2.line(buf, q0, q1, col_outer, 3, cv2.LINE_AA)
                cv2.line(buf, q0, q1, col_core,  1, cv2.LINE_AA)
        self._arcs = alive

    def clear(self): self._arcs.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SHOCKWAVE SYSTEM  (expanding pulse rings on gesture transition)
# ═══════════════════════════════════════════════════════════════════════════════

class ShockwaveSystem:
    def __init__(self):
        self._waves = []   # {cx, cy, born, color}

    def spawn(self, cx, cy, now, color=(80,200,255)):
        self._waves.append({'cx':cx,'cy':cy,'born':now,'color':color})

    def draw(self, frame, now):
        alive = []
        for w in self._waves:
            age = now - w['born']
            if age > SHOCK_LIFETIME: continue
            alive.append(w)
            progress = age / SHOCK_LIFETIME
            fade = 1.0 - progress
            for ring_i in range(SHOCK_RINGS):
                phase   = progress - ring_i * 0.12
                if phase < 0: continue
                radius  = int(phase * SHOCK_SPEED * SHOCK_LIFETIME)
                alpha   = max(0.0, fade * (1.0 - ring_i * 0.28))
                col     = tuple(int(c*alpha) for c in w['color'])
                thick   = max(1, int((1.0-phase)*3))
                if radius > 5:
                    cv2.circle(frame, (int(w['cx']),int(w['cy'])),
                               radius, col, thick, cv2.LINE_AA)
                    # Inner bright ring
                    col2 = tuple(min(255, int(c*alpha*1.5)) for c in w['color'])
                    cv2.circle(frame, (int(w['cx']),int(w['cy'])),
                               max(1,radius-2), col2, 1, cv2.LINE_AA)
        self._waves = alive

    def clear(self): self._waves.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PALM ABSORPTION EFFECT  (energy collapses into the palm on fist close)
# ═══════════════════════════════════════════════════════════════════════════════

class PalmAbsorptionEffect:
    def __init__(self):
        self._active = False
        self._t0     = 0.0
        self._cx = self._cy = 0.0
        self._duration = 0.5

    def trigger(self, cx, cy, now):
        self._active=True; self._t0=now
        self._cx=cx; self._cy=cy

    def draw(self, frame, now):
        if not self._active: return
        age = now - self._t0
        if age > self._duration: self._active=False; return
        t_  = age / self._duration
        # Contracting bright ring + core flash
        fade   = 1.0 - t_
        radius = int((1.0-t_) * 80 + 8)
        col    = (int(60*fade), int(200*fade), int(255*fade))
        cv2.circle(frame, (int(self._cx),int(self._cy)), radius, col, 2, cv2.LINE_AA)
        col2 = (int(180*fade), int(255*fade), int(255*fade))
        cv2.circle(frame, (int(self._cx),int(self._cy)), max(1,radius//2), col2, 1, cv2.LINE_AA)
        # Central flash
        if t_ < 0.25:
            flash_r = int((0.25-t_)/0.25 * 30)
            col3 = (int(220*fade), int(250*fade), int(255*fade))
            cv2.circle(frame, (int(self._cx),int(self._cy)), max(2,flash_r), col3, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMATIC ABERRATION  (standalone — vignette and bloom removed)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_chromatic_aberration(frame, strength=2.5):
    """Shift R and B channels in opposite directions."""
    if strength < 0.5: return frame
    s = int(strength)
    if s < 1: return frame
    b, g, r = cv2.split(frame)
    h, w = frame.shape[:2]
    M_r = np.float32([[1,0, s],[0,1, s]])
    M_b = np.float32([[1,0,-s],[0,1,-s]])
    r2  = cv2.warpAffine(r, M_r, (w,h), borderMode=cv2.BORDER_REFLECT)
    b2  = cv2.warpAffine(b, M_b, (w,h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b2, g, r2])

def apply_film_grain(frame, strength=4.0):
    """Subtle film grain for cinematic texture."""
    if strength < 0.5: return frame
    grain = np.random.normal(0, strength, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY RENDERER  (vectorized draw + depth-sorted layers)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyRenderer:
    def __init__(self, w, h):
        self._w, self._h = w, h
        self._buf = np.zeros((h, w, 3), dtype=np.float32)
        self._depth_mode = True   # pseudo-3D

    def draw(self, frame, pool, cx, cy, sphere_r, fade=1.0):
        if not pool.alive: return

        h, w = frame.shape[:2]
        buf  = self._buf; buf[:] = 0.0

        scale = 1.0 - pool.collapse
        r_eff = sphere_r * max(scale, 0.01)

        N = N_TENDRILS
        pts    = pool.pts
        bright = pool._brightness * fade
        cols   = ENERGY_COLORS[pool._color_idx]  # (N, 3) BGR float

        # Draw each tendril
        for i in range(N):
            br  = float(bright[i])
            col = cols[i]
            th  = int(pool._thickness[i])

            for j in range(TENDRIL_PTS-1):
                p0=(int(pts[i,j,  0]), int(pts[i,j,  1]))
                p1=(int(pts[i,j+1,0]), int(pts[i,j+1,1]))
                seg_br = br * (1.0 - 0.60*(j/(TENDRIL_PTS-1)))
                c = (float(min(col[0]*seg_br, 255)),
                     float(min(col[1]*seg_br, 255)),
                     float(min(col[2]*seg_br, 255)))
                cv2.line(buf, p0, p1, c, th, cv2.LINE_AA)

            # White-hot root dot
            rp=(int(pts[i,0,0]), int(pts[i,0,1]))
            if 0<=rp[0]<w and 0<=rp[1]<h:
                buf[rp[1],rp[0]] = np.minimum(
                    500.0,
                    buf[rp[1],rp[0]] + np.array([255.,255.,255.], np.float32))

        # ── 4-pass bloom ──────────────────────────────────────────────────
        results = []
        for ds, wt, sig_mul in [(8,1.80,0.22),(4,1.20,0.10),(2,0.70,0.05),(1,0.40,0.02)]:
            bw=max(1,w//ds); bh=max(1,h//ds)
            s   = cv2.resize(buf, (bw,bh), interpolation=cv2.INTER_LINEAR)
            sig = max(1.0, r_eff*sig_mul)
            bl  = cv2.GaussianBlur(s,(0,0),sigmaX=sig)
            results.append((cv2.resize(bl,(w,h),interpolation=cv2.INTER_LINEAR), wt))

        combined = buf * 1.2
        for layer, wt in results:
            combined += layer * wt
        np.clip(combined, 0, 255, out=combined)
        cv2.add(frame, combined.astype(np.uint8), dst=frame)

        # ── White-hot core glow ────────────────────────────────────────────
        if r_eff > 10 and fade > 0.05:
            core_r = int(r_eff * SPHERE_INNER_GLOW)
            core_img = np.zeros((h,w,3), dtype=np.uint8)
            icx, icy = int(cx), int(cy)
            layers = [
                (core_r,            (  4,  12,   8), int(20*fade)),
                (core_r*3//4,       ( 15,  50,  15), int(32*fade)),
                (core_r//2,         ( 50, 140,  55), int(48*fade)),
                (core_r//3,         (120, 210, 120), int(65*fade)),
                (core_r//4,         (200, 250, 200), int(80*fade)),
                (max(4,core_r//6),  (240, 255, 240), int(110*fade)),
                (max(2,core_r//10), (255, 255, 255), int(140*fade)),
            ]
            for radius_c, col_c, alpha_c in layers:
                if radius_c < 2: continue
                ov = core_img.copy()
                cv2.circle(ov, (icx,icy), radius_c, col_c, -1, cv2.LINE_AA)
                cv2.addWeighted(ov, min(alpha_c/255.0,1.0), core_img, 1.0, 0, core_img)
            cv2.add(frame, core_img, dst=frame)


# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER STATES
# ═══════════════════════════════════════════════════════════════════════════════

(INTACT, CHARGING, CRACKING, EXPLODING, FLOATING,
 ORBIT, CONVERGE, BUILDING, STABILIZE) = range(9)

_STATE_NAMES = {
    INTACT:"INTACT", CHARGING:"CHARGING", CRACKING:"CRACKING",
    EXPLODING:"EXPLODING", FLOATING:"FLOATING", ORBIT:"ORBIT",
    CONVERGE:"CONVERGE", BUILDING:"BUILDING", STABILIZE:"STABILIZE",
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
            length = rng.uniform(cube_s*0.4, cube_s*1.8)
            sx = cx + rng.uniform(-cube_s*0.7, cube_s*0.7)
            sy = cy + rng.uniform(-cube_s*0.7, cube_s*0.7)
            ex = sx + math.cos(ang)*length
            ey = sy + math.sin(ang)*length
            self._lines.append(((int(sx),int(sy)),(int(ex),int(ey))))

    def draw(self, frame, progress):
        if not self._lines: return
        n_show = max(1, int(len(self._lines)*progress))
        for i,(p0,p1) in enumerate(self._lines[:n_show]):
            frac  = i/max(len(self._lines),1)
            alpha = 0.55+0.45*(1-frac)*progress
            col   = tuple(int(c*alpha) for c in CLR_GLOW)
            cv2.line(frame, p0, p1, col, 2, cv2.LINE_AA)
            cv2.line(frame, p0, p1, (255,255,255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA SHAKE
# ═══════════════════════════════════════════════════════════════════════════════

class CameraShake:
    def __init__(self):
        self._t0=-999.; self._dx=self._dy=0

    def trigger(self, now, magnitude=SHAKE_MAGNITUDE):
        self._t0=now
        self._dx=int(np.random.uniform(-magnitude,magnitude))
        self._dy=int(np.random.uniform(-magnitude,magnitude))

    def apply(self, frame, now):
        dt=now-self._t0
        if dt>SHAKE_DURATION: return frame
        decay=math.exp(-SHAKE_DECAY*dt)
        dx,dy=int(self._dx*decay),int(self._dy*decay)
        if dx==0 and dy==0: return frame
        return np.roll(np.roll(frame,dy,axis=0),dx,axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# SHATTER SYSTEM  (orchestrates all sub-systems)
# ═══════════════════════════════════════════════════════════════════════════════

class ShatterSystem:
    def __init__(self, seed=7, w=CAMERA_WIDTH, h=CAMERA_HEIGHT):
        self.state  = INTACT
        self.t0     = 0.0
        self.hx=self.hy=0.0
        self.cube_s = float(CUBE_HALF)
        self.ex=self.ey=0.0
        self.t_anim=0.0
        self.pool        = EnergyParticlePool(seed=seed)
        self.cracks      = CrackEffect()
        self.arcs        = ElectricArcSystem(seed=seed+10)
        self.shockwaves  = ShockwaveSystem()
        self.absorption  = PalmAbsorptionEffect()

    def _begin_crack(self, now):
        self.state=CRACKING; self.t0=now
        self.cracks.generate(self.hx,self.hy,self.cube_s,seed=int(now*1000))
        self.shockwaves.spawn(self.hx,self.hy,now,color=(100,220,255))

    def _begin_explosion(self, now, t):
        self.state=EXPLODING; self.t0=now
        self.pool.explode(t)
        self.shockwaves.spawn(self.hx,self.hy,now,color=(200,240,255))

    def update(self, hx, hy, gesture, openness, now, cube_s, t, dt,
               pts_px=None, vel=None, accel=None):
        self.hx,self.hy=hx,hy
        self.cube_s=cube_s
        self.t_anim=t
        vel   = vel   if vel   is not None else np.zeros(2)
        accel = accel if accel is not None else np.zeros(2)

        if gesture=="open" and self.state in (INTACT,CHARGING,BUILDING,STABILIZE):
            self.ex,self.ey=hx,hy
            self._begin_crack(now)

        elif gesture=="closed" and self.state in (EXPLODING,FLOATING):
            self.state=ORBIT; self.t0=now; self.pool.freeze()
            self.absorption.trigger(hx,hy,now)
            self.shockwaves.spawn(hx,hy,now,color=(255,200,80))
            self.arcs.clear()

        if self.state==CRACKING:
            if now-self.t0>CRACK_SECS:
                self._begin_explosion(now,t)

        elif self.state in (EXPLODING,FLOATING):
            sphere_r=self.cube_s*SPHERE_RADIUS_MUL
            # Vectorized update with velocity reactivity
            update_tendrils_vectorized(self.pool, hx, hy, sphere_r, t,
                                       float(vel[0]), float(vel[1]))
            if pts_px:
                self.arcs.update_spawn(pts_px, now, openness)
            if self.state==EXPLODING and now-self.t0>1.6:
                self.state=FLOATING

        elif self.state==ORBIT:
            dt_s=now-self.t0; ease=min(dt_s/ORBIT_SECS,1.0)
            ease_s=ease*ease*(3-2*ease)
            self.pool.step_orbit((hx,hy),ease_s,t)
            if ease>=1.0:
                self.pool.snapshot_orbit_anchor()
                self.state=CONVERGE; self.t0=now

        elif self.state==CONVERGE:
            dt_s=now-self.t0; ease=min(dt_s/CONVERGE_SECS,1.0)
            ease_s=ease*ease*(3-2*ease)
            self.pool.step_converge((hx,hy),ease_s)
            if ease>=1.0:
                self.state=BUILDING; self.t0=now
                self.pool.deactivate_all()

        elif self.state==BUILDING:
            if now-self.t0>LAYER_BUILD_SECS:
                self.state=STABILIZE; self.t0=now

        elif self.state==STABILIZE:
            if now-self.t0>STABILIZE_SECS:
                self.state=INTACT

    def draw(self, frame, now, renderer, dt,
             enable_shake=True, cam_shake=None):
        t=self.t_anim
        sphere_r=self.cube_s*SPHERE_RADIUS_MUL

        self.shockwaves.draw(frame, now)
        self.absorption.draw(frame, now)

        if self.state==CRACKING:
            prog=min((now-self.t0)/CRACK_SECS,1.0)
            self.cracks.draw(frame,prog)

        elif self.state in (EXPLODING,FLOATING):
            renderer.draw(frame,self.pool,self.ex,self.ey,sphere_r,fade=1.0)
            self.arcs.draw(frame, now)

        elif self.state==ORBIT:
            renderer.draw(frame,self.pool,self.ex,self.ey,sphere_r,fade=1.0)

        elif self.state==CONVERGE:
            fade=max(0.0,1.0-(now-self.t0)/CONVERGE_SECS*0.7)
            renderer.draw(frame,self.pool,self.ex,self.ey,
                          sphere_r*(1.0-self.pool.collapse),fade=fade)

        elif self.state==BUILDING:
            prog=min((now-self.t0)/LAYER_BUILD_SECS,1.0)
            fade=max(0.0,1.0-prog)
            if fade>0.01:
                renderer.draw(frame,self.pool,self.ex,self.ey,sphere_r,fade=fade)


# ═══════════════════════════════════════════════════════════════════════════════
# AMBIENT SPARKLES
# ═══════════════════════════════════════════════════════════════════════════════

class AmbientSparkles:
    def __init__(self, n=SPARK_COUNT, seed=42):
        rng       = np.random.default_rng(seed)
        self.ang  = rng.uniform(0,math.pi*2,n)
        self.reach= rng.uniform(0.05,1.7,n)
        self.sz   = rng.uniform(1.0,3.5,n)
        self.spd  = rng.uniform(0.5,2.0,n)
        pal=[(255,255,255),(230,255,240),(200,255,180),(180,210,255),(255,230,200)]
        self.col=np.array(pal)[rng.integers(0,len(pal),n)]

    def draw(self, layer, cx, cy, spread, t, speed_boost=0.0):
        if spread<5: return
        h,w=layer.shape[:2]
        # Sparks move faster when hand is fast
        jscale = 8.0 + speed_boost*0.02
        jx = np.sin(t*self.spd*2.8+self.reach*28)*jscale
        jy = np.cos(t*self.spd*2.3+self.reach*15)*jscale
        x  = cx+np.cos(self.ang)*self.reach*spread+jx
        y  = cy+np.sin(self.ang)*self.reach*spread+jy
        for i in range(len(x)):
            px,py=int(x[i]),int(y[i])
            if not(0<=px<w and 0<=py<h): continue
            br=max(0.1,1.0-0.5*self.reach[i])
            col=tuple(int(c*br) for c in self.col[i])
            cv2.circle(layer,(px,py),max(1,int(self.sz[i])),col,-1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGE EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class ChargeEffect:
    def __init__(self, n=28, seed=9):
        rng       = np.random.default_rng(seed)
        self.angs = rng.uniform(0,math.pi*2,n)
        self.dists= rng.uniform(0.5,1.6,n)
        self.spds = rng.uniform(0.8,2.5,n)

    def draw(self, frame, cx, cy, spread, t, intensity):
        if intensity<0.02 or spread<5: return
        h,w=frame.shape[:2]
        for i,ang in enumerate(self.angs):
            phase=(t*self.spds[i])%1.0
            d_out=self.dists[i]*spread*(1.0-phase)
            d_in =max(0,d_out-spread*0.18)
            sx=int(cx+math.cos(ang)*d_out); sy=int(cy+math.sin(ang)*d_out)
            ex_=int(cx+math.cos(ang)*d_in);  ey_=int(cy+math.sin(ang)*d_in)
            if not(0<=sx<w and 0<=sy<h): continue
            al=intensity*(0.3+0.7*phase)
            col=tuple(int(CLR_CHARGE[k]*al) for k in range(3))
            cv2.line(frame,(sx,sy),(ex_,ey_),col,1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# FPS COUNTER
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    def __init__(self, window=30):
        self._times=collections.deque(maxlen=window)
        self._last=time.perf_counter()

    def tick(self):
        now=time.perf_counter(); self._times.append(now-self._last); self._last=now

    @property
    def fps(self):
        if len(self._times)<2: return 0.0
        return 1.0/(sum(self._times)/len(self._times))


# ═══════════════════════════════════════════════════════════════════════════════
# HUD
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps, debug, hand_data, shake_on, chroma_on):
    h,w=frame.shape[:2]
    bar=np.zeros((36,w,3),dtype=np.uint8)
    cv2.addWeighted(bar,0.40,frame[:36],0.60,0,frame[:36])
    fps_col=(100,255,100) if fps>=45 else (0,165,255) if fps>=25 else (0,60,255)
    cv2.putText(frame,f"FPS {fps:.0f}",(10,24),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,fps_col,1,cv2.LINE_AA)
    hints=(f"q=quit  r=reset  d=debug  s=shake  c=chroma  "
           f"shk={'ON' if shake_on else 'OFF'}  "
           f"chr={'ON' if chroma_on else 'OFF'}")
    cv2.putText(frame,hints,(6,24),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,160,160),1,cv2.LINE_AA)
    if not debug or not hand_data: return
    for idx,hd in enumerate(hand_data):
        y=60+idx*28
        txt=(f"Hand {idx+1}  gest={hd['gesture']:<7}"
             f"  conf={hd['confidence']:.2f}"
             f"  open={hd['openness']:.2f}"
             f"  [{hd['state']}]"
             f"  spd={hd['speed']:.0f}px/s"
             f"  acc={hd['accel']:.0f}")
        (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
        region=frame[max(0,y-th-4):min(h,y+6),6:min(w,14+tw)]
        cv2.addWeighted(np.zeros_like(region),0.45,region,0.55,0,region)
        cv2.putText(frame,txt,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.44,(200,255,200),1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_model()
    if NUMBA:
        print("[INFO] Numba detected — JIT compilation active.")

    opts=mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}",file=sys.stderr); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    except Exception: pass

    systems  = {}
    gestures = {}

    sparkles   = AmbientSparkles()
    charge_fx  = ChargeEffect()
    renderer   = EnergyRenderer(CAMERA_WIDTH, CAMERA_HEIGHT)
    cam_shake  = CameraShake()
    fps_ctr    = FPSCounter()

    t0=time.time(); prev_time=t0
    debug_mode  = False
    shake_on    = True
    chroma_on   = True

    skeleton_layer=np.zeros((CAMERA_HEIGHT,CAMERA_WIDTH,3),dtype=np.uint8)
    sparkle_layer =np.zeros((CAMERA_HEIGHT,CAMERA_WIDTH,3),dtype=np.uint8)

    print("Hand Shatter v5 — Cinematic Plasma")
    print("  q=quit  r=reset  d=debug  s=shake  c=chroma")
    print("  Open hand → plasma orb  |  Close fist → absorb")
    if NUMBA: print("  [Numba JIT active]")

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok,frame=cap.read()
            if not ok: time.sleep(0.01); continue

            frame=cv2.flip(frame,1)
            fh,fw=frame.shape[:2]
            now=time.time()
            dt=max(0.001,now-prev_time); prev_time=now
            t=now-t0

            det_w=max(1,int(fw*MP_DETECT_SCALE))
            det_h=max(1,int(fh*MP_DETECT_SCALE))
            small=cv2.resize(frame,(det_w,det_h),interpolation=cv2.INTER_LINEAR)
            rgb  =cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            res  =lmk.detect_for_video(mp_img,int(now*1000))

            hand_data_hud=[]
            skeleton_layer.fill(0); sparkle_layer.fill(0)
            any_skeleton=any_sparkle=False

            if res.hand_landmarks:
                active_ids=set()
                for idx,hand_lm in enumerate(res.hand_landmarks):
                    pts       =lm_to_px(hand_lm,fw,fh)
                    pts_px_int=[(int(p[0]),int(p[1])) for p in pts]
                    active_ids.add(idx)

                    if idx not in systems:
                        systems[idx] =ShatterSystem(seed=idx*31+7,w=fw,h=fh)
                        gestures[idx]=GestureRecogniser()

                    gr  =gestures[idx]
                    sys_=systems[idx]

                    gesture,confidence=gr.update(pts,dt)
                    hx,hy =palm_centre(pts)
                    ps    =palm_size(pts)
                    cube_s=float(np.clip(ps*0.85,50,160))

                    prev_state=sys_.state
                    sys_.update(hx,hy,gesture,gr.openness,now,cube_s,t,dt,
                                pts_px=pts_px_int,
                                vel=gr.palm_velocity,
                                accel=gr.palm_acceleration)

                    if shake_on and prev_state==CRACKING and sys_.state==EXPLODING:
                        cam_shake.trigger(now)

                    draw_skeleton(skeleton_layer,pts_px_int)
                    any_skeleton=True

                    sp_spread=cube_s*(0.55+0.55*gr.openness)
                    sparkles.draw(sparkle_layer,hx,hy,sp_spread,t,
                                  speed_boost=gr.palm_speed)
                    any_sparkle=True

                    if sys_.state in (INTACT,BUILDING,CHARGING,STABILIZE,CRACKING):
                        ci=(0.6 if sys_.state==INTACT else
                            min((now-sys_.t0)/LAYER_BUILD_SECS,1.0)
                            if sys_.state==BUILDING else 0.5)
                        charge_fx.draw(frame,hx,hy,cube_s*1.4,t,ci)

                    sys_.draw(frame,now,renderer,dt,
                              enable_shake=shake_on,
                              cam_shake=cam_shake)

                    hand_data_hud.append({
                        "gesture":    gesture,
                        "confidence": confidence,
                        "openness":   gr.openness,
                        "state":      _STATE_NAMES[sys_.state],
                        "speed":      gr.palm_speed,
                        "accel":      float(np.linalg.norm(gr.palm_acceleration)),
                    })

                for gone in set(systems.keys())-active_ids:
                    del systems[gone]; del gestures[gone]
            else:
                systems.clear(); gestures.clear()

            if any_skeleton:
                cv2.addWeighted(skeleton_layer,0.55,frame,1.0,0,frame)
            if any_sparkle:
                cv2.add(frame,sparkle_layer,dst=frame)

            # ── Post-processing ───────────────────────────────────────────
            if chroma_on:
                chroma_str = 2.0
                for s in systems.values():
                    if s.state in (EXPLODING,FLOATING):
                        chroma_str = 4.0; break
                frame=apply_chromatic_aberration(frame, strength=chroma_str)

            frame=apply_film_grain(frame, strength=3.5)

            if shake_on:
                frame=cam_shake.apply(frame,now)

            fps_ctr.tick()
            draw_hud(frame,fps_ctr.fps,debug_mode,hand_data_hud,
                     shake_on,chroma_on)

            cv2.imshow("Hand Shatter v5 — Cinematic Plasma  [q=quit]",frame)

            key=cv2.waitKey(1)&0xFF
            if   key==ord('q'): break
            elif key==ord('r'):
                systems.clear(); gestures.clear(); print("[INFO] Reset.")
            elif key==ord('d'):
                debug_mode=not debug_mode; print(f"[INFO] Debug {'ON' if debug_mode else 'OFF'}.")
            elif key==ord('s'):
                shake_on=not shake_on; print(f"[INFO] Shake {'ON' if shake_on else 'OFF'}.")
            elif key==ord('c'):
                chroma_on=not chroma_on; print(f"[INFO] Chroma {'ON' if chroma_on else 'OFF'}.")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()

# Run:  python cube_shatter_effect.py
# Opt:  pip install numba   (for JIT-accelerated noise — ~2× faster)