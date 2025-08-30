#!/usr/bin/env python3
# mpm_emitter2_elastoplastic.py
#
# Genesis MPM demo — two emitters printing concentric circular layers.

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
        substeps = 25,
        gravity  = (0.0, 0.0, -9.81),
    ),
    # finer grid & bigger vertical span
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4,  -1.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,        # ↓ cell size ⇒ more v-verts
        # grid_density = ,
        # enable_CPIC = True,
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# ─── 3) Rigid ground ────────────────────────────────────────────────────────
ground = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        needs_coup=True,           # ensure this geom participates in coupling
        coup_friction=0.9,         # ↑ tangential friction against MPM
        coup_softness=0.001,       # a bit stiffer contact
        coup_restitution=0.0
    ),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─── 4) “Carrier” sphere that the emitters sample ───────────────────────────
carrier1 = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.3, pos=(3.0, 0.0, 1.0)),
    material = gs.materials.MPM.ElastoPlastic(
                E=2.0e6,            # 2 MPa — typical for ~30–50 kg/m³ rigid PU; adjust with tests
                nu=0.30,            # closed-cell rigid foams often 0.25–0.35
                rho=40.0,           # kg/m³ — mid of in-situ 35–50 kg/m³
                sampler='pbs',
                yield_lower=0.05,   # elastic-to-plateau onset ~5% strain
                yield_higher=0.35,  # plateau ends ~35% strain (densification starts)
                use_von_mises=False,
                # von_mises_yield_stress=1.5e5  # 0.15 MPa — matches EN 826 ≥150 kPa plateau/σ10%
                ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.05, 0.05)),
)

carrier2 = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.3, pos=(2.0, 0.0, 1.0)),
    material = gs.materials.MPM.ElastoPlastic(
                E=2.0e6,            # 2 MPa — typical for ~30–50 kg/m³ rigid PU; adjust with tests
                nu=0.30,            # closed-cell rigid foams often 0.25–0.35
                rho=40.0,           # kg/m³ — mid of in-situ 35–50 kg/m³
                sampler='pbs',
                yield_lower=0.05,   # elastic-to-plateau onset ~5% strain
                yield_higher=0.35,  # plateau ends ~35% strain (densification starts)
                use_von_mises=False,
                # von_mises_yield_stress=1.5e5  # 0.15 MPa — matches EN 826 ≥150 kPa plateau/σ10%
                ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.05, 0.05)),
)

# ─── 5) Two Elasto-plastic emitters ─────────────────────────────────────────
emitter1 = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=True),
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.3, 0.1)),
)
emitter1.set_entity(carrier1)

emitter2 = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=True),
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.1, 0.4, 0.9)),
)
emitter2.set_entity(carrier2)

# ─── 6) Patch emit() → pos & vel as (B,N,3) ─────────────────────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
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

# bind the patch to both emitters
emitter1.emit = types.MethodType(emit_fixed, emitter1)
emitter2.emit = types.MethodType(emit_fixed, emitter2)

drag = scene.add_force_field(gs.force_fields.Drag(linear=2.0, quadratic=0.0))

# --------------------------------------------------------------------------
# 7) Build & run — two concentric circles per layer, same linear speed
# --------------------------------------------------------------------------
scene.build()

dt = scene.sim.dt

# Adjustable parameters (as requested)
n_layers     = 5           # number of layers
droplet_d    = 0.02        # droplet size; also sets radial gap & layer heights
base_radius  = 0.20        # inner circle radius
r_inner      = base_radius
r_outer      = base_radius + droplet_d*2  # radii differ by droplet size

path_speed   = 1.0        # SAME linear (tangential) speed for both circles [m/s]
omega_inner  = path_speed / max(r_inner, 1e-6)
omega_outer  = path_speed / max(r_outer, 1e-6)

# One full layer lasts as long as the OUTER circle needs for one revolution.
T_layer      = (2.0 * np.pi * r_outer) / path_speed
steps_layer  = int(np.ceil(T_layer / dt))

for layer in range(n_layers):
    z_emit = (layer + 1) * (droplet_d + 0.01) # heights: d, 2d, 3d, ...

    theta1 = 0.0
    theta2 = 0.0

    for _ in range(steps_layer):
        # Inner circle emission (stop after 1 turn)
        if theta1 < 2.0 * np.pi - 1e-9:
            x1 = r_inner * np.cos(theta1)
            y1 = r_inner * np.sin(theta1)
            emitter1.emit(
                droplet_shape = "sphere",
                droplet_size  = droplet_d,
                pos           = (x1, y1, z_emit),
                direction     = (0.0, 0.0, -1.0),
                speed         = 1.0,
                p_size        = 0.03,
            )
            theta1 = min(theta1 + omega_inner * dt, 2.0 * np.pi)

        # Outer circle emission (defines layer time)
        if theta2 < 2.0 * np.pi - 1e-9:
            x2 = r_outer * np.cos(theta2)
            y2 = r_outer * np.sin(theta2)
            emitter2.emit(
                droplet_shape = "square",
                droplet_size  = droplet_d,
                pos           = (x2, y2, z_emit),
                direction     = (0.0, 0.0, -1.0),
                speed         = 1.0,
                p_size        = 0.03,
            )
            theta2 = min(theta2 + omega_outer * dt, 2.0 * np.pi)

        scene.step()

scene.viewer.run()