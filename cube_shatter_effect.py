"""
hand_shatter_effect.py

  - Closed fist  -> კუბი ხელში "ჩნდება" და გამოჩნდება
  - Open hand    -> კუბი ადგილზე ფეთქდება, ნაჭრები სივრცეში ფრინავენ
  - Close again  -> ნაჭრები ხელში ბრუნდებიან და ისევ კუბს ქმნიან

pip install opencv-python mediapipe numpy
python hand_shatter_effect.py
q = quit
"""

import os, time, math, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── CONFIG ──────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE  = 0.6

CLOSED_RATIO = 0.55
OPEN_RATIO   = 2.0
SMOOTHING    = 0.15      # lower = smoother

CUBE_SCALE   = 100       # half-size of the intact cube (pixels)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

WRIST           = 0
PALM_REF_POINTS = (0, 5, 9, 13, 17)
FINGERTIPS      = (4, 8, 12, 16, 20)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ── MODEL ────────────────────────────────────────────────────────────────────
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-landmark model …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done:", MODEL_PATH)

# ── LANDMARK UTILS ───────────────────────────────────────────────────────────
def lm_to_px(lms, w, h):
    return [(lm.x*w, lm.y*h, lm.z) for lm in lms]

def draw_skeleton(frame, lms, w, h):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0,180,0), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (0,120,255), -1, cv2.LINE_AA)

def openness_raw(pts):
    wrist = np.array(pts[WRIST][:2])
    mcp   = np.array(pts[9][:2])
    sz    = np.linalg.norm(mcp - wrist)
    if sz < 1e-3: return 0.0
    pc    = np.mean([pts[i][:2] for i in PALM_REF_POINTS], axis=0)
    dists = [np.linalg.norm(np.array(pts[t][:2]) - pc) for t in FINGERTIPS]
    return float(np.mean(dists) / sz)

def palm_center(pts):
    return np.mean([pts[i][:2] for i in PALM_REF_POINTS], axis=0)

# ── CUBE GEOMETRY ─────────────────────────────────────────────────────────────
SKEW = 0.35   # isometric skew factor

def cube_verts(cx, cy, s):
    sk = s * SKEW
    f = np.array([[cx-s,cy-s],[cx+s,cy-s],[cx+s,cy+s],[cx-s,cy+s]], dtype=float)
    b = f + [sk, -sk]
    return np.vstack([f, b])   # 0-3 front, 4-7 back

# face definitions (vertex indices into cube_verts output)
FACES = [
    ([0,1,2,3], (210,215,255)),   # front
    ([4,5,6,7], (130,135,190)),   # back
    ([0,1,5,4], (180,200,255)),   # top
    ([3,2,6,7], (110,115,170)),   # bottom
    ([0,3,7,4], (155,165,215)),   # left
    ([1,2,6,5], (165,180,230)),   # right
]

def draw_cube(frame, cx, cy, s, alpha=1.0):
    v  = cube_verts(cx, cy, s)
    ov = frame.copy()
    for vidx, col in FACES:
        pts = np.array([v[i] for i in vidx], dtype=np.int32)
        c   = tuple(int(x*alpha) for x in col)
        cv2.fillPoly(ov, [pts], c)
        cv2.polylines(ov, [pts], True, (255,255,255), 1, cv2.LINE_AA)
    blend = min(0.85, 0.5 + 0.35*alpha)
    cv2.addWeighted(ov, blend, frame, 1-blend, 0, frame)
    for vx in v:
        r = max(3, int(s*0.06))
        cv2.circle(frame, (int(vx[0]),int(vx[1])), r, (255,255,255), -1, cv2.LINE_AA)

# ── FRAGMENT ──────────────────────────────────────────────────────────────────
class Frag:
    __slots__ = ('local','color','pos','vel','rot','rot_spd','scale','frozen_pos')
    def __init__(self, local, color):
        self.local      = np.array(local, dtype=float)  # (N,2) centred at 0
        self.color      = color
        self.pos        = np.zeros(2)
        self.vel        = np.zeros(2)
        self.rot        = 0.0
        self.rot_spd    = 0.0
        self.scale      = 1.0
        self.frozen_pos = None   # set when fragment stops moving

    def world_pts(self):
        c, s = math.cos(self.rot), math.sin(self.rot)
        R    = np.array([[c,-s],[s,c]])
        return (R @ (self.local * self.scale).T).T + self.pos

# ── SHATTER SYSTEM ────────────────────────────────────────────────────────────
# States
INTACT, EXPLODING, FLOATING, PULLING, BUILDING = 0,1,2,3,4

class ShatterSystem:
    def __init__(self, seed=7):
        self.rng   = np.random.default_rng(seed)
        self.state = INTACT
        self.t0    = 0.0
        # hand position (updated every frame)
        self.hx    = 0.0
        self.hy    = 0.0
        # where the explosion happened (stays fixed after shatter)
        self.ex    = 0.0
        self.ey    = 0.0
        self.frags = self._make_frags()

    # ── fragment factory ──────────────────────────────────────────────────
    def _make_frags(self):
        frags = []
        rng   = self.rng

        # For each FACE we produce a 3x3 grid of sub-quads in local [-1,1] space.
        # We'll map them to real pixels at draw-time using self.ex/ey + CUBE_SCALE.
        face_local_corners = [
            # front face:  x in [-1,1], y in [-1,1]  (no depth)
            [(-1,-1),(1,-1),(1,1),(-1,1)],
            # back face: offset by (SKEW*2, -SKEW*2) in local units
            [(-1+SKEW*2,-1-SKEW*2),(1+SKEW*2,-1-SKEW*2),(1+SKEW*2,1-SKEW*2),(-1+SKEW*2,1-SKEW*2)],
            # top:   front-top-edge + back-top-edge
            [(-1,-1),(1,-1),(1+SKEW*2,-1-SKEW*2),(-1+SKEW*2,-1-SKEW*2)],
            # bottom
            [(-1,1),(1,1),(1+SKEW*2,1-SKEW*2),(-1+SKEW*2,1-SKEW*2)],
            # left
            [(-1,-1),(-1,1),(-1+SKEW*2,1-SKEW*2),(-1+SKEW*2,-1-SKEW*2)],
            # right
            [(1,-1),(1,1),(1+SKEW*2,1-SKEW*2),(1+SKEW*2,-1-SKEW*2)],
        ]
        base_colors = [c for _,c in FACES]

        DIVS = 3   # subdivide each face NxN
        for fi, corners in enumerate(face_local_corners):
            c = np.array(corners, dtype=float)
            bc = base_colors[fi]
            for gi in range(DIVS):
                for gj in range(DIVS):
                    u0,u1 = gi/DIVS, (gi+1)/DIVS
                    v0,v1 = gj/DIVS, (gj+1)/DIVS
                    def blerp(u,v):
                        return (c[0]*(1-u)*(1-v) + c[1]*u*(1-v) +
                                c[3]*(1-u)*v      + c[2]*u*v)
                    pts = np.array([blerp(u0,v0),blerp(u1,v0),
                                    blerp(u1,v1),blerp(u0,v1)])
                    # centre the sub-quad
                    ctr  = pts.mean(axis=0)
                    local = pts - ctr   # shape (4,2) centred at 0
                    # add tiny jitter for organic feel
                    local += rng.uniform(-0.04, 0.04, local.shape)
                    noise  = rng.integers(-30, 30, 3)
                    color  = tuple(int(np.clip(bc[k]+noise[k],0,255)) for k in range(3))
                    frags.append(Frag(local, color))

        # Extra edge slivers
        for _ in range(12):
            ang = rng.uniform(0, math.pi*2)
            r   = rng.uniform(0.3, 0.9)
            w2  = 0.06
            sliver = np.array([
                [0, 0],
                [math.cos(ang)*r, math.sin(ang)*r],
                [math.cos(ang+0.3)*r+w2, math.sin(ang+0.3)*r+w2],
                [w2, w2],
            ])
            sliver -= sliver.mean(axis=0)
            frags.append(Frag(sliver, (255,255,255)))

        return frags

    # ── trigger helpers ───────────────────────────────────────────────────
    def _launch_frags(self):
        """Set physics for the explosion — fragments fly from (ex,ey)."""
        rng = self.rng
        for frag in self.frags:
            frag.pos      = np.array([self.ex, self.ey], dtype=float)
            frag.scale    = float(CUBE_SCALE)
            ang            = rng.uniform(0, math.pi*2)
            spd            = rng.uniform(5, 22)
            frag.vel      = np.array([math.cos(ang)*spd, math.sin(ang)*spd])
            frag.rot      = rng.uniform(0, math.pi*2)
            frag.rot_spd  = rng.uniform(-0.22, 0.22)
            frag.frozen_pos = None

    def _start_pull(self):
        """Freeze fragment positions, then they'll animate toward hand."""
        for frag in self.frags:
            frag.frozen_pos = frag.pos.copy()

    # ── update (called every frame) ───────────────────────────────────────
    def update(self, hx, hy, openness, now):
        """
        hx,hy     = current hand (palm) centre in pixels
        openness  = 0 (fist) … 1 (open)
        """
        self.hx, self.hy = hx, hy

        # ── state transitions ──
        if openness > 0.55 and self.state in (INTACT, PULLING, BUILDING):
            # EXPLODE at current hand position
            self.ex, self.ey = hx, hy
            self.state = EXPLODING
            self.t0    = now
            self._launch_frags()

        elif openness < 0.35 and self.state in (FLOATING, EXPLODING):
            # PULL back toward hand
            self.state = PULLING
            self.t0    = now
            self._start_pull()

        # ── physics ──
        if self.state == EXPLODING:
            for f in self.frags:
                f.vel     *= 0.91
                f.pos     += f.vel
                f.rot     += f.rot_spd
                f.rot_spd *= 0.97
            if now - self.t0 > 1.6:
                self.state = FLOATING
                for f in self.frags:
                    f.frozen_pos = f.pos.copy()

        elif self.state == FLOATING:
            # Slow drift
            for f in self.frags:
                f.vel     *= 0.97
                f.pos     += f.vel * 0.3
                f.rot     += f.rot_spd * 0.3

        elif self.state == PULLING:
            dt   = now - self.t0
            ease = min(dt / 0.65, 1.0)
            ease = ease*ease*(3-2*ease)   # smoothstep
            hp   = np.array([hx, hy])
            for f in self.frags:
                f.pos = f.frozen_pos + (hp - f.frozen_pos) * ease
                f.rot = f.rot * (1-ease*0.6)
            if ease >= 1.0:
                self.state = BUILDING
                self.t0    = now

        elif self.state == BUILDING:
            # brief pause then go INTACT
            if now - self.t0 > 0.18:
                self.state = INTACT

    # ── draw ─────────────────────────────────────────────────────────────
    def draw(self, frame):
        h, w = frame.shape[:2]
        if self.state == INTACT:
            draw_cube(frame, self.hx, self.hy, CUBE_SCALE, alpha=1.0)

        elif self.state == BUILDING:
            # ghost cube fades in
            dt    = time.time() - self.t0
            alpha = min(dt / 0.18, 1.0)
            draw_cube(frame, self.hx, self.hy, CUBE_SCALE, alpha=alpha)

        else:
            # Draw fragments at their world positions (NOT following hand)
            layer = np.zeros_like(frame)
            for f in self.frags:
                wp  = f.world_pts().astype(np.int32)
                # distance fade from explosion origin
                dist  = np.linalg.norm(f.pos - np.array([self.ex, self.ey]))
                fade  = float(np.clip(1.0 - dist/(CUBE_SCALE*4.0), 0.1, 1.0))
                col   = tuple(int(c*fade) for c in f.color)
                edge  = wp.reshape(-1,1,2)
                in_frame = ((wp[:,0]>=0)&(wp[:,0]<w)&(wp[:,1]>=0)&(wp[:,1]<h)).any()
                if not in_frame: continue
                cv2.fillPoly(layer, [edge], col)
                cv2.polylines(layer, [edge], True, (255,255,255), 1, cv2.LINE_AA)
                px,py = int(f.pos[0]), int(f.pos[1])
                if 0<=px<w and 0<=py<h:
                    cv2.circle(layer,(px,py),max(2,int(CUBE_SCALE*0.035*fade)),
                               tuple(min(255,int(c*1.4*fade)) for c in f.color),-1)
            cv2.add(frame, layer, dst=frame)

            # During PULLING draw ghost cube at hand growing in
            if self.state == PULLING:
                dt    = time.time() - self.t0
                ghost = min(dt/0.65, 1.0)
                ghost = ghost*ghost*(3-2*ghost)
                draw_cube(frame, self.hx, self.hy, CUBE_SCALE, alpha=ghost*0.5)

# ── SPARKLES ──────────────────────────────────────────────────────────────────
class Sparkles:
    def __init__(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        self.ang   = rng.uniform(0, math.pi*2, n)
        self.reach = rng.uniform(0.05, 1.0, n)
        self.sz    = rng.uniform(1.0, 3.0, n)
        pal = [[255,255,255],[220,255,230],[200,255,180],[180,200,255]]
        self.col = np.array(pal)[rng.integers(0,len(pal),n)]

    def draw(self, frame, cx, cy, spread, t):
        if spread < 5: return
        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        jx = np.sin(t*3.0+self.reach*30)*5
        jy = np.cos(t*2.7+self.reach*17)*5
        x  = cx + np.cos(self.ang)*self.reach*spread + jx
        y  = cy + np.sin(self.ang)*self.reach*spread + jy
        for i in range(len(x)):
            px,py = int(x[i]),int(y[i])
            if 0<=px<w and 0<=py<h:
                br  = 1.0-0.5*self.reach[i]
                col = tuple(int(c*br) for c in self.col[i])
                cv2.circle(layer,(px,py),max(1,int(self.sz[i])),col,-1)
        cv2.add(frame, layer, dst=frame)

# ── MAIN ──────────────────────────────────────────────────────────────────────
STATE_NAMES = {INTACT:"INTACT",EXPLODING:"EXPLODING",FLOATING:"FLOATING",
               PULLING:"PULLING",BUILDING:"BUILDING"}

def main():
    ensure_model()
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap      = cv2.VideoCapture(CAMERA_INDEX)
    systems  = {}
    prev_op  = {}
    sparkles = Sparkles()
    t0       = time.time()

    with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()

            rgb    = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = lmk.detect_for_video(mp_img, int(now*1000))

            if res.hand_landmarks:
                for idx, hand_lm in enumerate(res.hand_landmarks):
                    pts = lm_to_px(hand_lm, w, h)

                    raw     = openness_raw(pts)
                    target  = float(np.clip((raw-CLOSED_RATIO)/(OPEN_RATIO-CLOSED_RATIO),0,1))
                    prev    = prev_op.get(idx, target)
                    smooth  = prev + (target-prev)*SMOOTHING
                    prev_op[idx] = smooth

                    hx, hy = palm_center(pts)

                    if idx not in systems:
                        systems[idx] = ShatterSystem(seed=idx*17+3)
                    sys = systems[idx]
                    sys.update(hx, hy, smooth, now)

                    draw_skeleton(frame, hand_lm, w, h)
                    sys.draw(frame)

                    # Ambient sparkles around hand
                    sp_spread = 60 + 60*smooth
                    sparkles.draw(frame, hx, hy, sp_spread, now-t0)

                    cv2.putText(frame,
                        f"openness:{smooth:.2f} [{STATE_NAMES[sys.state]}] raw:{raw:.2f}",
                        (10, 25+idx*24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255,255,255), 1, cv2.LINE_AA)
            else:
                systems.clear()
                prev_op.clear()

            cv2.imshow("Hand Shatter  [q=quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
##C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe cube_shatter_effect.py
