#!/usr/bin/env python3
# mpm_emitter2_elastoplastic.py
#
# Genesis MPM demo — downward elasto-plastic droplet jet for 5 s.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ─── 1) Boot Genesis ────────────────────────────────────────────────────────
gs.init()

# ─── 2) Scene & solver options ──────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    # finer grid & bigger vertical span
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4,  0.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,        # ↓ cell size ⇒ more v-verts
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# ─── 3) Rigid ground ────────────────────────────────────────────────────────
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─── 4) “Carrier” sphere that the emitter samples ───────────────────────────
carrier = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.25, pos=(3.0, 0.0, 1.0)),  # ↑ bigger
    material = gs.materials.MPM.ElastoPlastic(),                    # ← elastoplastic
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.3, 0.15, 0.05)),
)

# 5) Elasto-plastic emitter --------------------------------------------
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(),   # dummy -> avoids v-vert check
    max_particles = 50_000,
    surface       = gs.surfaces.Default(
        vis_mode="particle",
        color=(0.8, 0.3, 0.1),
    ),
)
emitter.set_entity(carrier)      

# ─── 6) Patch emit() → pos & vel as (B,N,3) ─────────────────────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.20,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):

    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = self._solver.particle_size if p_size is None else p_size

    pts_local = pu.sphere_to_particles(
        p_size=p_size,
        radius=droplet_size * 0.5,
        sampler=self._entity.sampler,
    ).astype(gs.np_float, copy=False)                            # (N,3)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    pts_world = np.tile(pts_world[None], (self._sim._B, 1, 1))   # (B,N,3)

    if not self._solver.boundary.is_inside(pts_world):
        gs.raise_exception("Emitted particles are outside the boundary.")

    v_single = (speed * direction).astype(gs.np_float, copy=False)
    vels = np.tile(v_single, (self._sim._B, pts_local.shape[0], 1))  # (B,N,3)

    n   = pts_local.shape[0]
    idx = self._entity.particle_start + self._next_particle

    self._solver._kernel_set_particles_pos(
        self._sim.cur_substep_local, idx, n, pts_world
    )
    self._solver._kernel_set_particles_vel(
        self._sim.cur_substep_local, idx, n, vels
    )
    self._solver._kernel_set_particles_active(
        self._sim.cur_substep_local, idx, n, gs.ACTIVE
    )

    self._next_particle = (self._next_particle + n) % self._entity.n_particles

# bind the patch to this emitter
emitter.emit = types.MethodType(emit_fixed, emitter)

# --------------------------------------------------------------------------
# 7) Build & run  (emitter follows a 1-m circle centred at the origin) -----
# --------------------------------------------------------------------------
scene.build()

duration      = 5.0                      # seconds
dt            = scene.sim.dt             # 1e-3
steps_total   = int(duration / dt)

radius        = 0.3                      # 1-m diameter
omega         = 2 * np.pi / duration     # one full revolution in 'duration' s

for step in range(steps_total):
    t        = step * dt * 4                # current time
    x_offset =  radius * np.cos(omega * t)
    y_offset =  radius * np.sin(omega * t)

    emitter.emit(
        droplet_shape = "sphere",
        droplet_size  = 0.02,
        pos           = (x_offset, y_offset, 0.15),   # ← moving centre
        direction     = (0.0, 0.0, -1.0),            # always downward
        speed         = 0.4,
        p_size        = 0.04,
    )
    scene.step()

scene.viewer.run()