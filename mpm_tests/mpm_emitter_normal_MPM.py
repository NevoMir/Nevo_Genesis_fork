#!/usr/bin/env python3
# mpm_emitter2_normal_MPM_fixed.py
#
# Genesis MPM demo — downward fluid-droplet jet for 5 s.
# Patches emitter.emit() so that both pos and vel are (B, N, 3).

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ─── 1) Boot Genesis ────────────────────────────────────────────────────────
gs.init()

# ─── 2) Scene setup ─────────────────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound = (-0.5, -0.5,  0.0),
        upper_bound = ( 0.5,  0.5,  1.5),
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# ─── 3) Ground plane ─────────────────────────────────────────────────────────
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─── 4) Carrier sphere (sampler only) ───────────────────────────────────────
carrier = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.10, pos=(0.0, 0.0, 1.0)),
    material = gs.materials.MPM.Liquid(),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.2, 0.2, 0.2)),
)

# ─── 5) Fluid emitter ────────────────────────────────────────────────────────
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(),
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.1, 0.4, 1.0)),
)
#emitter.set_entity(carrier)

# ─── 6) Patch emit() for correct vel shape ──────────────────────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.15,
               droplet_length=None, pos=(0.0,0.0,1.0),
               direction=(0.0,0.0,-1.0), theta=0.0,
               speed=0.4, p_size=None):

    assert self._entity is not None

    # normalize
    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    # fill spacing
    p_size = self._solver.particle_size if p_size is None else p_size

    # only sphere here
    positions_local = pu.sphere_to_particles(
        p_size  = p_size,
        radius  = droplet_size * 0.5,
        sampler = self._entity.sampler,
    ).astype(gs.np_float, copy=False)              # (N,3) float32

    # world positions + batch → (B,N,3)
    base_pos = np.asarray(pos, dtype=gs.np_float)
    positions_world = positions_local + base_pos
    positions_world = np.tile(positions_world[None], (self._sim._B, 1, 1))

    if not self._solver.boundary.is_inside(positions_world):
        gs.raise_exception("Emitted particles outside boundary.")

    # constant velocity per droplet → shape (N,3)
    v_single = (speed * direction).astype(gs.np_float, copy=False)
    vels_1d  = np.tile(v_single[None], (positions_local.shape[0], 1))

    # **tile to (B,N,3)** for MPM solver
    vels = np.tile(vels_1d[None], (self._sim._B, 1, 1)).astype(gs.np_float, copy=False)

    # write into fields
    n   = positions_local.shape[0]
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

    # advance & wrap
    self._next_particle += n
    if self._next_particle + n > self._entity.n_particles:
        self._next_particle = 0

# bind only this emitter
emitter.emit = types.MethodType(emit_fixed, emitter)

# ─── 7) Build scene ──────────────────────────────────────────────────────────
scene.build()

# ─── 8) Emit & simulate for 5 seconds ────────────────────────────────────────
DURATION    = 5.0
dt          = 1e-3
steps_total = int(DURATION / dt)

for _ in range(steps_total):
    emitter.emit(
        droplet_shape = "sphere",
        droplet_size  = 0.15,
        pos           = (0.0, 0.0, 1.0),
        direction     = (0.0, 0.0, -1.0),
        speed         = 0.4,
        p_size        = 0.04,
    )
    scene.step()

# ─── 9) Viewer ───────────────────────────────────────────────────────────────
scene.viewer.run()