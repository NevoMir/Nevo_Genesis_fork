#!/usr/bin/env python3
# mpm_emitter_elastoplastic_multiphase_full.py
#
# Genesis MPM demo — liquid -> foam-like expansion -> hardening (elastoplastic)
# Uses multiple "carrier" entities to bucket particles by age.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ─── 1) Boot Genesis ────────────────────────────────────────────────────────
gs.init()

# ─── 2) Scene & solver options ─────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4,  0.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# ─── 3) Rigid ground ───────────────────────────────────────────────────────
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─── 4) Define the patched emit() BEFORE binding ───────────────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.20,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    """
    Emit a compact droplet with positions/vels in shape (B,N,3).
    """
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

    self._solver._kernel_set_particles_pos(self._sim.cur_substep_local, idx, n, pts_world)
    self._solver._kernel_set_particles_vel(self._sim.cur_substep_local, idx, n, vels)
    self._solver._kernel_set_particles_active(self._sim.cur_substep_local, idx, n, gs.ACTIVE)

    self._next_particle = (self._next_particle + n) % self._entity.n_particles

# ─── 5) Create K carriers (start as Liquid) ────────────────────────────────
K = 6
carriers = []
for k in range(K):
    carriers.append(scene.add_entity(
        morph    = gs.morphs.Sphere(radius=0.25, pos=(3.0, 0.0, 1.0)),
        material = gs.materials.MPM.Liquid(),   # keep minimal to avoid API mismatches
        surface  = gs.surfaces.Default(
            vis_mode="particle",
            color=(0.25, 0.12 + 0.08*k, 0.05),
        ),
    ))

# Track the last emit position for each carrier (for "foam rise" ring)
last_emit_pos = [None] * K

# ─── 6) One emitter; we’ll switch which carrier it samples from ────────────
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(),   # dummy; entity’s material is used
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.3, 0.1)),
)

# Bind our patched emit()
emitter.emit = types.MethodType(emit_fixed, emitter)

# ─── 7) Build after entities+emitter are set up ────────────────────────────
scene.build()

# ─── 8) Schedules & params ─────────────────────────────────────────────────
duration      = 5.0
dt            = scene.sim.dt
steps_total   = int(duration / dt)

# Circle path (like yours; 4× angular speed for a full loop in 5 s)
radius_path   = 0.3
omega         = 2 * np.pi / duration

# Age model (seconds)
t_rise0 = 0.20    # start "foam rise" window
t_rise1 = 0.50    # end "foam rise" window
t_gel   = 0.60    # switch to ElastoPlastic after this

# Age tracking per bucket
bucket_age   = [0.0] * K
stride_steps = 6    # how many steps we keep filling the same bucket

# Foam rise ring params
ring_R       = 0.015
ring_N       = 6
ring_speed   = 0.2
ring_p_size  = 0.03
ring_drop_sz = 0.01

# ─── 9) Sim loop ───────────────────────────────────────────────────────────
for step in range(steps_total):
    t        = step * dt
    # pick bucket (carrier) to receive the fresh droplet
    b        = (step // stride_steps) % K
    cur      = carriers[b]
    emitter.set_entity(cur)

    # Move emitter on a circle
    x_offset = radius_path * np.cos(omega * t * 4)
    y_offset = radius_path * np.sin(omega * t * 4)
    z_emit   = 0.15
    pos_emit = (x_offset, y_offset, z_emit)

    # Main droplet
    emitter.emit(
        droplet_shape = "sphere",
        droplet_size  = 0.02,
        pos           = pos_emit,
        direction     = (0.0, 0.0, -1.0),
        speed         = 0.4,
        p_size        = 0.04,
    )
    last_emit_pos[b] = pos_emit

    # Approximate "foam rise": for carriers in rise window, emit a thin ring
    for k, c in enumerate(carriers):
        age = bucket_age[k]
        if (t_rise0 <= age <= t_rise1) and (last_emit_pos[k] is not None):
            cx, cy, cz = last_emit_pos[k]
            emitter.set_entity(c)
            for j in range(ring_N):
                ang = 2*np.pi*j/ring_N
                dx, dy = ring_R*np.cos(ang), ring_R*np.sin(ang)
                emitter.emit(
                    droplet_shape="sphere",
                    droplet_size=ring_drop_sz,
                    pos=(cx+dx, cy+dy, cz),
                    direction=(0,0,-1),
                    speed=ring_speed,
                    p_size=ring_p_size,
                )

        # Hardening: flip to elastoplastic after gel time (per-bucket)
        # (Only flip once; checking instance of Liquid as a guard.)
        if age > t_gel and isinstance(c.material, gs.materials.MPM.Liquid):
            c.material = gs.materials.MPM.ElastoPlastic(
                E=6e5, nu=0.3, rho=1000.0,
                yield_lower=0.03, yield_higher=0.25,
                use_von_mises=True,
                von_mises_yield_stress=6e5 * 0.03
            )

    # Advance the sim
    scene.step()

    # Update ages; reset the fresh bucket’s age
    bucket_age = [a + dt for a in bucket_age]
    bucket_age[b] = 0.0

# ─── 10) Viewer ────────────────────────────────────────────────────────────
scene.viewer.run()