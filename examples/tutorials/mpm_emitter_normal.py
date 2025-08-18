#!/usr/bin/env python3
# mpm_emitter2_droplet_fixed.py
#
# Genesis PBD demo — downward droplet jet for 5 s.
# Fixes the TaichiIndexError by patching emitter.emit() so that
#   • positions  → shape (B, N, 3)  ← expected
#   • velocities → shape (N, 3)     ← expected

import types
import numpy as np
import genesis as gs
import genesis.utils.geom     as gu
import genesis.utils.particle as pu

# ─── 1) Boot Genesis ─────────────────────────────────────────────────────────
gs.init()

# ─── 2) Scene setup ──────────────────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# 3) Ground plane -------------------------------------------------------------
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# 4) Carrier sphere (gives colour & sampler) ----------------------------------
carrier = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.10, pos=(0.0, 0.0, 1.0)),
    material = gs.materials.PBD.Particle(),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(1.0, 0.2, 0.2)),
)

# 5) Emitter ------------------------------------------------------------------
emitter = scene.add_emitter(
    material      = gs.materials.PBD.Particle(),
    max_particles = 200000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(1.0, 0.2, 0.2)),
)
emitter.set_entity(carrier)

# ─── Patch emitter.emit() so velocities are (N,3) not (B,N,3) ────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.08,
               droplet_length=None, pos=(0.0,0.0,1.0),
               direction=(0.0,0.0,-1.0), theta=0.0,
               speed=0.5, p_size=None):

    assert self._entity is not None

    # ── normalise direction ────────────────────────────────────────────────
    direction = np.asarray(direction, dtype=gs.np_float)
    if np.linalg.norm(direction) < gs.EPS:
        gs.raise_exception("Zero-length direction.")
    direction = direction / (np.linalg.norm(direction) + gs.EPS)

    # ── particle size used for filling ─────────────────────────────────────
    p_size = self._solver.particle_size if p_size is None else p_size

    # ── generate local positions for a spherical droplet (others omitted) ─
    if droplet_shape != "sphere":
        gs.raise_exception("This demo only patches 'sphere' droplets.")
    positions_local = pu.sphere_to_particles(
        p_size  = p_size,
        radius  = droplet_size * 0.5,
        sampler = self._entity.sampler,
    ).astype(gs.np_float, copy=False)                              # (N,3)

    # ── transform into world space & add batch dim (B,N,3) ────────────────
    positions_world = positions_local + np.asarray(pos, dtype=gs.np_float)
    positions_world = np.tile(positions_world[None], (self._sim._B, 1, 1))

    if not self._solver.boundary.is_inside(positions_world):
        gs.raise_exception("Emitted particles are outside the boundary.")

    # ── constant velocity per particle → shape (N,3) f32 ──────────────────
    v_single = (speed * direction).astype(gs.np_float, copy=False)
    vels     = np.tile(v_single[None], (positions_local.shape[0], 1))  # (N,3)

    # ── write into solver fields ───────────────────────────────────────────
    n  = positions_local.shape[0]
    idx = self._entity.particle_start + self._next_particle

    self._solver._kernel_set_particles_pos(
        self._sim.cur_substep_local, idx, n, positions_world
    )
    self._solver._kernel_set_particles_vel(
        self._sim.cur_substep_local, idx, n, vels
    )
    self._solver._kernel_set_particles_active(
        self._sim.cur_substep_local, idx, n, gs.ACTIVE
    )

    self._next_particle += n
    if self._next_particle + n > self._entity.n_particles:
        self._next_particle = 0

# bind the fix to *this* emitter instance
emitter.emit = types.MethodType(emit_fixed, emitter)

# 6) Build scene --------------------------------------------------------------
scene.build()

# 7) Emit & simulate for 5 s --------------------------------------------------
DURATION    = 5.0   # s
SIM_DT      = 1e-3
steps_total = int(DURATION / SIM_DT)

for _ in range(steps_total):
    emitter.emit(
        droplet_shape = "sphere",
        droplet_size  = 0.2,
        pos           = (0.0, 0.0, 1.0),
        direction     = (0.0, 1.0, 0.0),
        speed         = 1.0,
        p_size       = 0.1
    )
    scene.step()

# 8) Run viewer ---------------------------------------------------------------
scene.viewer.run()