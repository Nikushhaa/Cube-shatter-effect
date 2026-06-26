"""
physics.py

Self-contained fragment physics engine.

Original code had physics spread across the ShatterSystem class with no
separation of concerns. Here each concern lives in its own function / class:

  PhysicsState  : the mutable state vector for one fragment
                  (position, velocity, rotation, angular velocity, scale)
  PhysicsEngine : applies forces, drag, gravity and boundary conditions
                  to a batch of PhysicsState objects every frame.

Benefits of the separation:
  - Physics can be unit-tested independently of rendering.
  - SwappingCollision rules or gravity direction is a one-line change.
  - NumPy vectorised operations work cleanly on batched arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.config import PhysicsConfig


# ── PhysicsState ─────────────────────────────────────────────────────────────

@dataclass
class PhysicsState:
    """
    Mutable physics state for a single fragment.

    All values are in pixel space (or radians for rotation).

    pos      : 2-D world position [x, y]  (top-left origin)
    vel      : 2-D velocity       [vx, vy] (pixels/frame)
    rot      : current rotation angle in radians
    rot_vel  : angular velocity (radians/frame)
    scale    : render scale multiplier (1.0 = normal size)
    alive    : False once the fragment leaves the frame permanently
               or is absorbed back into the cube
    frozen   : when True, physics integration is skipped
               (used during FLOATING phase)
    anchor   : optional fixed position used as pull target during PULLING
    """
    pos:     np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel:     np.ndarray = field(default_factory=lambda: np.zeros(2))
    rot:     float = 0.0
    rot_vel: float = 0.0
    scale:   float = 1.0
    alive:   bool  = True
    frozen:  bool  = False
    anchor:  Optional[np.ndarray] = None

    def __post_init__(self):
        # Ensure mutable numpy arrays even if caller passed tuples
        self.pos = np.array(self.pos, dtype=float)
        self.vel = np.array(self.vel, dtype=float)


# ── PhysicsEngine ─────────────────────────────────────────────────────────────

class PhysicsEngine:
    """
    Applies physics rules to a list of PhysicsState objects.

    Original version had physics hard-coded inside ShatterSystem.update().
    Extracting it here lets us:
      - Tune every constant from config.PhysicsConfig
      - Apply vectorised NumPy operations on position/velocity batches
      - Add / remove forces (e.g. gravity) without touching drawing code

    Usage:
        engine = PhysicsEngine(cfg.physics, frame_w=1280, frame_h=720)
        engine.integrate(states, dt=1/60)   # call every frame
    """

    def __init__(
        self,
        cfg:     PhysicsConfig,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> None:
        self._cfg  = cfg
        self._w    = frame_w
        self._h    = frame_h

    def resize(self, w: int, h: int) -> None:
        """Call when the capture resolution changes at runtime."""
        self._w = w
        self._h = h

    # ── core integration ──────────────────────────────────────────────────
    def integrate(
        self,
        states: Sequence[PhysicsState],
        dt:     float = 1.0 / 60.0,
    ) -> None:
        """
        Advance all states by one time step `dt` (seconds).

        Steps per fragment:
          1. Skip if frozen or dead.
          2. Apply gravity (downward constant acceleration).
          3. Apply drag (exponential decay per axis).
          4. Integrate velocity → position.
          5. Apply angular drag and integrate rotation.
          6. Enforce boundary conditions (bounce off walls).
          7. Mark as frozen if nearly stopped (settles to rest).

        We process positions and velocities as NumPy arrays over the whole
        batch where possible, falling back to per-fragment loops only for
        the boundary check (since each fragment can be at a different wall).
        """
        cfg = self._cfg
        # Collect live, non-frozen states for vectorised treatment
        active = [s for s in states if s.alive and not s.frozen]
        if not active:
            return

        # ── stack into arrays for batch operations ─────────────────────
        pos = np.array([s.pos for s in active], dtype=float)  # (N,2)
        vel = np.array([s.vel for s in active], dtype=float)  # (N,2)

        # ── 2. Gravity (applied only in y direction) ────────────────────
        # Gravity is in px/frame²; we scale by dt (normalised to 60 fps)
        gravity_step = cfg.gravity * dt * 60.0
        vel[:, 1] += gravity_step

        # ── 3. Drag ────────────────────────────────────────────────────
        vel *= cfg.drag

        # ── 4. Integrate position ──────────────────────────────────────
        pos += vel * dt * 60.0

        # ── 5. Angular dynamics (per fragment, cheap loop) ─────────────
        for s in active:
            s.rot_vel *= cfg.rotation_drag
            s.rot     += s.rot_vel

        # ── 6. Boundary conditions ─────────────────────────────────────
        # Right wall
        mask_r = pos[:, 0] > self._w
        vel[mask_r, 0] *= -cfg.boundary_restitution
        pos[mask_r, 0]  = self._w

        # Left wall
        mask_l = pos[:, 0] < 0
        vel[mask_l, 0] *= -cfg.boundary_restitution
        pos[mask_l, 0]  = 0.0

        # Bottom wall
        mask_b = pos[:, 1] > self._h
        vel[mask_b, 1] *= -cfg.boundary_restitution
        pos[mask_b, 1]  = self._h

        # Top wall
        mask_t = pos[:, 1] < 0
        vel[mask_t, 1] *= -cfg.boundary_restitution
        pos[mask_t, 1]  = 0.0

        # ── write back ─────────────────────────────────────────────────
        for i, s in enumerate(active):
            s.pos = pos[i]
            s.vel = vel[i]

        # ── 7. Freeze nearly-stopped fragments ─────────────────────────
        speeds = np.linalg.norm(vel, axis=1)
        for i, s in enumerate(active):
            if speeds[i] < cfg.settle_velocity:
                s.frozen = True

    def apply_explosion(
        self,
        states:   Sequence[PhysicsState],
        origin_x: float,
        origin_y: float,
        rng:      np.random.Generator,
    ) -> None:
        """
        Give each fragment an initial radial impulse from (origin_x, origin_y).

        The explosion force is sampled from [min_force, max_force].
        A small random tangential component adds natural irregularity.
        All fragments also get a random initial rotation.
        """
        cfg    = self._cfg
        origin = np.array([origin_x, origin_y])

        for s in states:
            s.pos     = origin.copy()
            s.frozen  = False
            s.anchor  = None
            s.alive   = True
            s.scale   = 1.0

            angle    = rng.uniform(0, math.pi * 2)
            force    = rng.uniform(cfg.min_force, cfg.max_force)
            # Tangential wobble ± 15 % of force
            tangent  = rng.uniform(-force * 0.15, force * 0.15)
            perp     = angle + math.pi / 2
            s.vel    = np.array([
                math.cos(angle) * force + math.cos(perp) * tangent,
                math.sin(angle) * force + math.sin(perp) * tangent,
            ])
            s.rot     = rng.uniform(0, math.pi * 2)
            s.rot_vel = rng.uniform(cfg.min_rotation_speed, cfg.max_rotation_speed)
            if rng.integers(0, 2):
                s.rot_vel = -s.rot_vel

    def apply_pull(
        self,
        states:  Sequence[PhysicsState],
        target:  np.ndarray,
        ease:    float,
    ) -> None:
        """
        Smoothly interpolate all fragment positions toward `target` (hand centre).

        `ease` is a value in [0, 1] produced by a smoothstep curve in the
        main state machine. When ease=1 the fragment is exactly at target.

        We also shrink rotation toward 0 as the fragments converge.
        """
        for s in states:
            if s.anchor is None:
                s.anchor = s.pos.copy()
            s.pos = s.anchor + (target - s.anchor) * ease
            s.rot = s.rot * (1.0 - ease * 0.6)

    def float_drift(self, states: Sequence[PhysicsState]) -> None:
        """
        Very gentle drift for FLOATING state — fragments that have settled still
        sway subtly so the effect doesn't look frozen.
        """
        for s in states:
            if s.frozen:
                s.vel *= 0.97
                s.pos += s.vel * 0.3
                s.rot += s.rot_vel * 0.2