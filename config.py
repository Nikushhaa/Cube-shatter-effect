"""
config.py

Central configuration hub for Cube Shatter Effect.
All magic numbers live here — never scattered across source files.
Edit this file to tune the entire application without touching logic code.

Sections:
  - CameraConfig      : capture device settings
  - HandConfig        : MediaPipe hand-tracking parameters
  - GestureConfig     : thresholds, smoothing, frame-confirmation
  - PhysicsConfig     : gravity, drag, explosion forces
  - CubeConfig        : geometry, glow, hologram palette
  - ParticleConfig    : sparkle / trail counts and lifetimes
  - SoundConfig       : volume, enabled flag
  - RenderConfig      : blend weights, motion-blur strength
  - DebugConfig       : overlay switches
  - PerformanceConfig : NumPy batch sizes, layer reuse
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, List


# ── Colour helpers ────────────────────────────────────────────────────────────
# BGR tuples used throughout the project.
CYAN    = (255, 220,  50)   # hologram primary    (BGR)
CYAN2   = (200, 255, 100)   # hologram secondary
WHITE   = (255, 255, 255)
GREEN   = (  0, 200,  60)   # skeleton
ORANGE  = (  0, 140, 255)   # joint dots
MAGENTA = (200,  50, 255)   # accent glow


# ── Camera ────────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """Settings for the capture device."""
    index:     int   = 0          # /dev/video0 on Linux; 0 on Windows
    width:     int   = 1280       # desired capture width  (may be clamped by hardware)
    height:    int   = 720        # desired capture height
    fps:       int   = 60         # requested frame-rate; actual FPS depends on hardware
    flip_h:    bool  = True       # mirror the frame so gestures feel natural
    flip_v:    bool  = False


# ── Hand Tracking ─────────────────────────────────────────────────────────────
@dataclass
class HandConfig:
    """MediaPipe HandLandmarker parameters."""
    max_hands:            int   = 2
    detection_confidence: float = 0.65
    tracking_confidence:  float = 0.65
    presence_confidence:  float = 0.65
    # Model will be auto-downloaded to assets/models/
    model_filename: str = "hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )


# ── Gesture Detection ─────────────────────────────────────────────────────────
@dataclass
class GestureConfig:
    """
    Thresholds that drive the gesture classifier.

    Frame-confirmation:
      A gesture is NOT acted upon the instant it is detected.
      It must be held for `confirm_frames` consecutive frames at or above
      `confirm_confidence` before a state transition fires.
      This prevents false triggers caused by detection noise.

    Openness ratio:
      Ratio of average fingertip distance from palm-centre to
      wrist-MCP distance. Low → fist, high → open hand.
    """
    # ── openness thresholds ──────────────────────────────────────────────
    closed_ratio:      float = 0.50   # below this → FIST
    open_ratio:        float = 1.90   # above this → OPEN HAND
    # ── smoothing (exponential moving average) ───────────────────────────
    smoothing_alpha:   float = 0.18   # 0 = frozen, 1 = no smoothing
    # ── frame confirmation ───────────────────────────────────────────────
    confirm_frames:    int   = 6      # frames a gesture must be held
    confirm_confidence:float = 0.82   # minimum confidence to count a frame
    # ── per-finger curl ──────────────────────────────────────────────────
    # A finger is "curled" if its tip is closer to the wrist than its base.
    curl_threshold:    float = 0.90   # ratio tip_wrist / base_wrist < this
    # ── pinch ────────────────────────────────────────────────────────────
    pinch_threshold:   float = 0.07   # normalised distance thumb↔index
    # ── point ────────────────────────────────────────────────────────────
    # Index extended, all others curled
    point_curl_min:    float = 0.55   # other fingers must be this curled


# ── Physics ───────────────────────────────────────────────────────────────────
@dataclass
class PhysicsConfig:
    """
    Newtonian fragment physics.

    Gravity pulls fragments downward each frame (pixels/frame²).
    Drag is a per-frame velocity multiplier simulating air resistance.
    Explosion force is sampled from [min_force, max_force] per fragment.
    Boundary damping: when a fragment hits the frame edge, its velocity
    component perpendicular to the wall is reversed and multiplied by
    `boundary_restitution`.
    """
    gravity:              float = 0.18    # downward acceleration (px/frame²)
    drag:                 float = 0.92    # velocity scale per frame (0–1)
    rotation_drag:        float = 0.97    # angular velocity scale per frame
    min_force:            float = 6.0     # minimum explosion impulse
    max_force:            float = 28.0    # maximum explosion impulse
    min_rotation_speed:   float = 0.08    # rad/frame
    max_rotation_speed:   float = 0.35    # rad/frame
    boundary_restitution: float = 0.35    # energy kept on wall bounce
    pull_duration:        float = 0.55    # seconds to animate "pull back"
    settle_velocity:      float = 0.5     # px/frame below which frag stops
    # How many seconds until EXPLODING → FLOATING (fragments slow down)
    float_after:          float = 1.4
    # How many seconds BUILDING state lasts (ghost cube → solid)
    build_duration:       float = 0.22


# ── Cube Visual ───────────────────────────────────────────────────────────────
@dataclass
class CubeConfig:
    """
    Hologram cube geometry and rendering.

    CUBE_SCALE is the half-extent in pixels: the cube spans 2×CUBE_SCALE
    from left to right and top to bottom at neutral size.

    SKEW is the isometric depth offset (fraction of CUBE_SCALE) used to
    give the illusion of a third dimension.

    Face colours use a neon-cyan hologram palette (BGR).
    Glow is rendered as layered alpha-blended circles.
    """
    scale:      int   = 95          # half-size of the cube in pixels
    skew:       float = 0.38        # isometric depth factor
    # ── hologram face colours (BGR) ──────────────────────────────────────
    face_colors: List[Tuple[int,int,int]] = field(default_factory=lambda: [
        ( 50, 220, 255),   # front   — bright cyan
        ( 20, 130, 190),   # back    — muted cyan
        ( 80, 240, 255),   # top     — lightest
        ( 10, 100, 160),   # bottom  — darkest
        ( 35, 180, 220),   # left
        ( 60, 210, 245),   # right
    ])
    # ── glow ─────────────────────────────────────────────────────────────
    glow_color:  Tuple[int,int,int] = (50, 220, 255)  # BGR
    glow_layers: int   = 5          # number of blurred glow halos
    glow_alpha:  float = 0.18       # per-layer alpha contribution
    glow_scale:  float = 1.05       # each glow layer is this × bigger
    # ── edge highlight ────────────────────────────────────────────────────
    edge_color:  Tuple[int,int,int] = (255, 255, 255)
    edge_thickness: int = 1
    # ── vertex dots ───────────────────────────────────────────────────────
    vertex_radius_ratio: float = 0.065   # radius = scale × this
    vertex_color: Tuple[int,int,int] = (200, 255, 255)
    # ── face subdivisions for shatter ────────────────────────────────────
    subdivisions: int = 4           # NxN grid per face  (was 3)
    extra_slivers: int = 16         # decorative edge shards


# ── Particles ─────────────────────────────────────────────────────────────────
@dataclass
class ParticleConfig:
    """
    Ambient sparkle cloud and per-fragment trails.

    Sparkles orbit the hand and pulse with hand openness.
    Trails are short-lived particles emitted by moving fragments.
    """
    # ── ambient sparkles ──────────────────────────────────────────────────
    sparkle_count:        int   = 140
    sparkle_spread_min:   float = 50.0    # pixels when fist
    sparkle_spread_max:   float = 130.0   # pixels when open
    sparkle_pulse_speed:  float = 2.8     # oscillation freq (rad/s)
    sparkle_size_min:     float = 1.0
    sparkle_size_max:     float = 3.2
    # ── fragment trails ───────────────────────────────────────────────────
    trail_per_fragment:   int   = 3       # particles emitted per moving frag
    trail_lifetime:       float = 0.35    # seconds before a trail particle fades
    trail_speed_scale:    float = 0.25    # fraction of fragment speed
    trail_size:           float = 2.0
    # ── colour palette (BGR) ─────────────────────────────────────────────
    sparkle_palette: List[Tuple[int,int,int]] = field(default_factory=lambda: [
        (255, 255, 255),   # white
        (200, 255, 220),   # pale cyan-green
        (180, 255, 180),   # pale green
        (255, 200, 180),   # pale blue
        ( 80, 220, 255),   # cyan
    ])


# ── Sound ─────────────────────────────────────────────────────────────────────
@dataclass
class SoundConfig:
    """
    Sound effects driven by pygame.mixer.
    Set `enabled = False` to run without audio (no pygame dependency required).

    Files are expected in assets/sounds/.
    """
    enabled:         bool  = True
    volume:          float = 0.6     # 0.0 – 1.0
    explosion_file:  str   = "explosion.wav"
    rebuild_file:    str   = "rebuild.wav"
    appear_file:     str   = "appear.wav"
    hum_file:        str   = "hum.wav"    # ambient loop while cube is INTACT
    hum_volume:      float = 0.25


# ── Rendering ─────────────────────────────────────────────────────────────────
@dataclass
class RenderConfig:
    """
    Compositing and post-processing parameters.

    Motion blur accumulates the previous frame at weight `mb_alpha`.
    A chromatic aberration pass shifts the red and blue channels slightly
    during EXPLODING state for a glitchy AR aesthetic.
    """
    # ── motion blur ────────────────────────────────────────────────────────
    motion_blur_enabled: bool  = True
    mb_alpha:            float = 0.22    # weight of previous frame (0 = off)
    # ── chromatic aberration ──────────────────────────────────────────────
    chroma_enabled:      bool  = True
    chroma_shift:        int   = 3       # pixel shift for R/B channels
    # ── vignette ──────────────────────────────────────────────────────────
    vignette_enabled:    bool  = True
    vignette_strength:   float = 0.35
    # ── scanlines (subtle CRT feel) ───────────────────────────────────────
    scanlines_enabled:   bool  = False
    scanlines_alpha:     float = 0.07
    # ── face blend weight of intact cube ─────────────────────────────────
    cube_face_alpha:     float = 0.75


# ── Debug ─────────────────────────────────────────────────────────────────────
@dataclass
class DebugConfig:
    """
    Overlay toggles for development and tuning.
    All disabled by default for production use; press 'd' at runtime to toggle.
    """
    show_fps:           bool = True
    show_openness:      bool = True
    show_state:         bool = True
    show_skeleton:      bool = True
    show_gesture_conf:  bool = True
    show_fragment_count:bool = False
    show_physics_vecs:  bool = False   # draw velocity arrows on fragments
    debug_mode:         bool = False   # master switch (press 'd')


# ── Performance ───────────────────────────────────────────────────────────────
@dataclass
class PerformanceConfig:
    """
    Low-level optimisation knobs.

    Fragments are drawn in batches onto a single numpy layer to minimise
    Python-loop overhead. `layer_reuse` skips zeroing the layer when it is
    fully overwritten anyway (safe for additive blending).
    """
    fragment_batch_size: int  = 64     # fragments rendered per numpy batch
    layer_reuse:         bool = True   # reuse pre-allocated numpy arrays
    downscale_glow:      bool = True   # compute glow at half resolution
    glow_downscale:      int  = 2      # downscale factor for glow pass


# ── Master Config ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    """
    Single object that carries every sub-config.
    Pass one `Config` instance through the whole application.

    Save / load from JSON so users can persist their settings:
        cfg = Config.load("settings.json")
        cfg.save("settings.json")
    """
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    hand:        HandConfig        = field(default_factory=HandConfig)
    gesture:     GestureConfig     = field(default_factory=GestureConfig)
    physics:     PhysicsConfig     = field(default_factory=PhysicsConfig)
    cube:        CubeConfig        = field(default_factory=CubeConfig)
    particle:    ParticleConfig    = field(default_factory=ParticleConfig)
    sound:       SoundConfig       = field(default_factory=SoundConfig)
    render:      RenderConfig      = field(default_factory=RenderConfig)
    debug:       DebugConfig       = field(default_factory=DebugConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialise the whole config tree to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """
        Deserialise from JSON.  Missing keys fall back to dataclass defaults,
        so older settings files stay forward-compatible.
        """
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        # Shallow merge each sub-config
        for key, sub_cls in {
            "camera": CameraConfig, "hand": HandConfig,
            "gesture": GestureConfig, "physics": PhysicsConfig,
            "cube": CubeConfig, "particle": ParticleConfig,
            "sound": SoundConfig, "render": RenderConfig,
            "debug": DebugConfig, "performance": PerformanceConfig,
        }.items():
            if key in data:
                try:
                    setattr(cfg, key, sub_cls(**{
                        k: v for k, v in data[key].items()
                        if k in sub_cls.__dataclass_fields__
                    }))
                except Exception:
                    pass   # silently ignore unknown/malformed fields
        return cfg


# Singleton default — import this for quick use:   from src.config import CFG
CFG = Config()